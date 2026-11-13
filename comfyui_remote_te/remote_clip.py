import json
import time

import requests
import safetensors.torch
import torch

from . import log
from . import endpoint, version
from .endpoint import find_api


class RemoteTokens:
    def __init__(self, text, kwargs):
        self.text = text
        self.kwargs = kwargs
        self.prompt_ids = None

    def __getitem__(self, i):
        if self.prompt_ids is None:
            raise TypeError("Remote CLIP: tokens can be read only after generate()")
        return [(t, 1.0) for t in self.prompt_ids[i]]


_version_warned = set()
NO_MODEL = "(no model list yet)"
LEGACY_DEFAULT = "(server default)"


class RemoteCLIP:
    def __init__(self, server_url, model, clip_type=None, timeout=300):
        self.url = server_url.strip().rstrip("/")
        self.timeout = timeout
        self.local = endpoint.is_local(self.url)
        try:
            if self.local:
                self.base = self._local_base()
                r = requests.get(self.base + "/models", timeout=self.timeout)
            else:
                self.base, r = find_api(self.url, requests.get, self.timeout)
        except requests.RequestException as e:
            msg = (f"Remote CLIP Loader: cannot reach {self.url}: {type(e).__name__}: {e}. Check the address, that "
                   f"the encoder PC's ComfyUI runs with --listen, and its firewall")
            log.error(msg)
            raise RuntimeError(msg) from None
        log.debug(f"{self.url}: API at {self.base}, GET /models -> {r.status_code}")
        if r.status_code == 404:
            raise RuntimeError(f"remote text encoder {self.url} does not support model selection (update server.py)")
        if r.status_code != 200:
            raise RuntimeError(f"remote text encoder {self.url}/models -> {r.status_code}: {r.text[:500]}")
        try:
            listing = r.json()
            names = [m["name"] for m in listing["models"]]
        except (ValueError, TypeError, KeyError):
            raise RuntimeError(f"{self.url} is not a remote text encoder server (unexpected /models reply). If it "
                               f"is a ComfyUI, install this pack there and turn on serving in its Settings; "
                               f"otherwise use the encoder server's port") from None
        if not self.local:
            warning = version.mismatch(listing.get("version"), self.url)
            if warning and (self.url, listing.get("version")) not in _version_warned:
                _version_warned.add((self.url, listing.get("version")))
                log.warning(warning)
        on_demand = listing.get("mode") == "on_demand"
        kind = f"on-demand server, {listing.get('gpu', 'GPU')}" if on_demand else "preloaded server"
        if model in (None, "", NO_MODEL, LEGACY_DEFAULT):
            raise RuntimeError(f"Remote CLIP Loader: no model selected; {self.url} ({kind}) has: {names}")
        if model not in names:
            if on_demand:
                hint = "" if str(model).endswith(".safetensors") else \
                    "; the workflow may have been set up for a preloaded server (router)"
                raise RuntimeError(f"'{model}' is not a file on {self.url} ({kind}){hint}. Pick one of its files: "
                                   f"{names}")
            hint = "; the workflow may have been set up for an on-demand server" \
                if str(model).endswith(".safetensors") else ""
            raise RuntimeError(f"'{model}' is not a model on {self.url} ({kind}){hint}. Its models: {names}")
        info = next(m for m in listing["models"] if m["name"] == model)
        if on_demand and not clip_type:
            raise RuntimeError(f"Remote CLIP Loader: no type selected for '{model}' on {self.url}; suggested: "
                               f"{info.get('types', [])}")
        self.model_name = model
        self.model = info["file"]
        self.clip_type = clip_type if on_demand else info["type"]
        self._generated = None

    def tokenize(self, text, return_word_ids=False, **kwargs):
        if return_word_ids:
            raise NotImplementedError("Remote CLIP: return_word_ids is not supported")
        for k, v in kwargs.items():
            if _is_tensor_arg(v):
                continue
            try:
                json.dumps(v)
            except TypeError:
                raise NotImplementedError(f"Remote CLIP: tokenize argument '{k}' ({type(v).__name__}) can't be "
                                          "sent to the server; only JSON values, tensors and lists of tensors can")
        return RemoteTokens(text, kwargs)

    def encode_from_tokens_scheduled(self, tokens, unprojected=False, add_dict={}, show_pbar=True):
        conditioning = self._encode(tokens)
        for _, extra in conditioning:
            extra.update(add_dict)
        return conditioning

    def encode_from_tokens(self, tokens, return_pooled=False, return_dict=False):
        cond, extra = self._encode(tokens)[0]
        if return_dict:
            return {"cond": cond, **extra}
        if return_pooled:
            return cond, extra.get("pooled_output")
        return cond

    def generate(self, tokens, do_sample=True, max_length=256, temperature=1.0, top_k=50, top_p=0.95, min_p=0.0,
                 repetition_penalty=1.0, seed=None, presence_penalty=0.0, mtp=True):
        gen = {"do_sample": do_sample, "max_length": max_length, "temperature": temperature, "top_k": top_k,
               "top_p": top_p, "min_p": min_p, "repetition_penalty": repetition_penalty, "seed": seed,
               "presence_penalty": presence_penalty, "mtp": mtp}
        try:
            reply = self._post_tokens("/textgen", tokens, {"generate_kwargs": gen}).json()
        except RuntimeError as e:
            if '/textgen -> 404: {"error": "not found"}' in str(e):
                raise RuntimeError(f"{self.url} can't run Generate Text: its server.py is older than these nodes; "
                                   f"update server.py there") from None
            raise
        tokens.prompt_ids = reply["prompt_ids"]
        self._generated = (tuple(reply["ids"]), reply["text"])
        return reply["ids"]

    def decode(self, token_ids, skip_special_tokens=True):
        ids = [int(i) for i in token_ids]
        if skip_special_tokens and self._generated and self._generated[0] == tuple(ids):
            return self._generated[1]
        req = {"ids": ids, "skip_special_tokens": bool(skip_special_tokens), "model": self.model_name,
               "type": self.clip_type}
        return self._request("post", "/decode", json=req).json()["text"]

    def locate(self, image, what):
        reply = self._generate_with_images({"task": "locate", "prompt": what}, [image])
        return reply["boxes"], reply.get("text", "")

    def _generate_with_images(self, req, images):
        req = {**req, "model": self.model_name, "type": self.clip_type}
        tensors, _ = pack_tensor_args({"images": list(images)})
        meta = {"request": json.dumps(req), "model": self.model_name, "type": self.clip_type}
        try:
            r = self._request("post", "/generate", data=safetensors.torch.save(tensors, metadata=meta),
                              headers={"Content-Type": "application/octet-stream"})
        except RuntimeError as e:
            if "tokenize_kwargs" in str(e):
                raise RuntimeError(f"{self.url} does not read images yet: its server.py is older than these nodes; "
                                   f"update server.py there") from None
            raise
        return r.json()

    def _encode(self, tokens):
        return deserialize(self._post_tokens("/encode", tokens).content)

    def _post_tokens(self, path, tokens, extra={}):
        if not isinstance(tokens, RemoteTokens):
            raise TypeError("Remote CLIP: tokens must come from this clip's tokenize()")
        plain = {k: v for k, v in tokens.kwargs.items() if not _is_tensor_arg(v)}
        tensor_args = {k: v for k, v in tokens.kwargs.items() if _is_tensor_arg(v)}
        if not tensor_args:
            req = {"text": tokens.text, "tokenize_kwargs": plain, **extra, "model": self.model_name,
                   "type": self.clip_type}
            return self._request("post", path, json=req)
        tensors, layout = pack_tensor_args(tensor_args)
        meta = {"text": tokens.text, "tokenize_kwargs": json.dumps(plain), "tensor_kwargs": json.dumps(layout)}
        meta.update({k: json.dumps(v) for k, v in extra.items()})
        meta["model"], meta["type"] = self.model_name, self.clip_type
        return self._request("post", path, data=safetensors.torch.save(tensors, metadata=meta),
                             headers={"Content-Type": "application/octet-stream"})

    def _local_base(self):
        try:
            return endpoint.local_base()
        except endpoint.LocalUnavailable as e:
            msg = f"Remote CLIP Loader: server_url is 'local', but {e}"
            log.error(msg)
            raise RuntimeError(msg) from None

    def _request(self, method, path, _made_room=False, **kw):
        if self.local:
            self.base = self._local_base()
        t0 = time.perf_counter()
        sent = len(kw["data"]) if isinstance(kw.get("data"), (bytes, bytearray)) else None
        try:
            r = requests.request(method, self.base + path, timeout=self.timeout, **kw)
        except requests.Timeout:
            msg = (f"remote text encoder {self.url}{path}: no answer within {self.timeout} s (model {self.model_name}, "
                   f"{self.clip_type}); a first load or a long generate can take that long")
            log.error(msg)
            raise RuntimeError(msg) from None
        except requests.RequestException as e:
            msg = f"remote text encoder {self.url}{path}: {type(e).__name__}: {e}"
            log.error(msg)
            raise RuntimeError(msg) from None
        ms = round((time.perf_counter() - t0) * 1000)
        got = len(r.content) if isinstance(getattr(r, "content", None), (bytes, bytearray)) else None
        log.debug(f"{method.upper()} {self.base}{path} model={self.model_name} type={self.clip_type} -> {r.status_code} "
                  f"in {ms} ms" + (f", sent {sent} B" if sent is not None else "")
                  + (f", got {got} B" if got is not None else ""))
        if r.status_code == 507 and not _made_room and make_room_on_own_card(r, self.url):
            return self._request(method, path, _made_room=True, **kw)
        if r.status_code != 200:
            msg = f"remote text encoder {self.url}{path} -> {r.status_code}: {r.text[:500]}"
            log.error(msg)
            raise RuntimeError(msg)
        return r

    def __getattr__(self, name):
        raise AttributeError(f"Remote CLIP does not support '{name}' (only text encoding, text generation and "
                             f"Remote Locate)")


def own_card_uuid():
    try:
        import comfy.model_management as mm
        dev = mm.get_torch_device()
        return str(torch.cuda.get_device_properties(dev).uuid).lower() if dev.type == "cuda" else None
    except Exception:
        return None


def _bare_uuid(u):
    u = (u or "").strip().lower()
    return u[4:] if u.startswith("gpu-") else u


def make_room_on_own_card(r, url, own=own_card_uuid):
    try:
        info = r.json()
    except ValueError:
        return False
    shared = _bare_uuid(info.get("shared_gpu") if isinstance(info, dict) else None)
    if not shared or shared != own():
        return False
    import comfy.model_management as mm
    log.info(f"{url}: the encoder needs {info.get('need_mb')} MB on the card this ComfyUI renders on; unloading this "
             f"ComfyUI's models from it, then asking again")
    mm.free_memory(1e30, mm.get_torch_device())
    mm.soft_empty_cache()
    return True


def pack_tensor_args(tensor_args):
    tensors, layout, seen = {}, {}, set()

    def add(name, t):
        t = t.contiguous().cpu()
        if t.data_ptr() in seen:
            t = t.clone()
        seen.add(t.data_ptr())
        tensors[name] = t

    for k, v in tensor_args.items():
        if isinstance(v, torch.Tensor):
            add(k, v)
            layout[k] = k
        else:
            layout[k] = [f"{k}.{i}" for i in range(len(v))]
            for name, t in zip(layout[k], v):
                add(name, t)
    return tensors, layout


def _is_tensor_arg(v):
    if isinstance(v, torch.Tensor):
        return True
    return isinstance(v, (list, tuple)) and len(v) > 0 and all(isinstance(t, torch.Tensor) for t in v)


def deserialize(body):
    tensors = safetensors.torch.load(body)
    extras = json.loads(_metadata(body)["extras"])
    conditioning = []
    for i, extra in enumerate(extras):
        extra = dict(extra)
        prefix = f"{i}."
        for k, v in tensors.items():
            if k.startswith(prefix) and k != f"{i}.cond":
                extra[k[len(prefix):]] = v
        conditioning.append([tensors[f"{i}.cond"], extra])
    return conditioning


def _metadata(body):
    header_len = int.from_bytes(body[:8], "little")
    return json.loads(body[8:8 + header_len]).get("__metadata__", {})
