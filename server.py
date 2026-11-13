import argparse
import json
import os
import re
import sys
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import capabilities
import gguf_support
import keep_file
import type_file
import locate
import prompt_files
from gpus import GpuError, list_gpus, pin, resolve, select
from model_cache import DoesNotFit, ModelCache
from model_hints import Hints
from server_config import (ConfigError, ModelSpec, check_types, comfy_dir, list_models, load_config,
                           models_dir_path, parse_reserve, reserve_mb)
from server_control import SecretError, authorized, load_secret, preflight
from server_status import Stats, log_line, pack_version, repo_commit

MIN_FREE_MB = 2048
SHARED_IDLE_UNLOAD_S = 60
SCAN_IDLE_S = 20
SCAN_RETRY_S = 60
LISTING_TTL_S = 10
MAX_GENERATE_TOKENS = 32768
MAX_IMAGE_PIXELS = 4 * 1024 * 1024
VISION_TEST = (("blue", (0.0, 0.0, 1.0)), ("yellow", (1.0, 1.0, 0.0)))
VISION_HINT = ("wire a vision encoder (qwen3vl_8b_int8_convrot.safetensors or qwen3vl_4b_fp8_scaled.safetensors) "
               "into a second Remote CLIP Loader")


class StartupError(Exception):
    pass


class UnknownModel(Exception):
    pass


class BadRequest(Exception):
    pass


@dataclass
class Loaded:
    spec: object
    clip: object
    empty_think: bool
    generate_error: str | None
    vram_mb: int
    generate_checked: bool = True
    vision_error: str | None = None
    vision_checked: bool = False


models = {}
cache = None
models_dir = None
models_dirs = []
keep_path = None
kept = []
keep_errors = []
total_mb = 0
usable_mb = 0
valid_types = set()
hints = None
gguf = None
gguf_note = None
caps = capabilities.Store(None, "unknown")
gpu_name = None
gpu_uuid = None
render_uuid = None
types_path = None
picked = {}
scan_enabled = False
scan_thread = None
scan_again = threading.Event()
scan_lock = threading.Lock()
comfyui_url = None
COMFYUI_FREE_WAIT_S = 20
HERE = os.path.dirname(os.path.abspath(__file__))
stats = Stats()
commit = "unknown"
VERSION = pack_version(HERE)
port = None
secret = None
launch_argv = []
gpu_lock = threading.Lock()
busy = None
posts = set()
last_post = time.monotonic()
scan = None


def busy_info():
    b = busy
    if b is None:
        return None
    op, since, fields = b
    return {"op": op, "model": fields.get("model"), "for_s": round(time.monotonic() - since, 1)}


def is_gguf(path):
    return path.lower().endswith(gguf_support.EXT)


measured = {}


def _file_key(path):
    st = os.stat(path)
    return (path, st.st_mtime_ns, st.st_size)


def need_mb(path):
    key = _file_key(path)
    if key in measured:
        return measured[key]
    return (key[2] >> 20) + (gguf_support.extra_mb(path) if is_gguf(path) else 0)


def load_model(spec, probe=True, check_fit=True):
    import torch
    import comfy.sd

    dev = torch.device("cuda", 0)
    if is_gguf(spec.path) and gguf is None:
        raise StartupError(f"'{spec.name}' ({os.path.basename(spec.path)}) is a GGUF file: install ComfyUI-GGUF "
                           f"in ComfyUI's custom_nodes ({gguf_note})")
    need = need_mb(spec.path)
    free_mb = torch.cuda.mem_get_info()[0] >> 20
    if check_fit and need > free_mb:
        raise StartupError(f"'{spec.name}' ({os.path.basename(spec.path)}, {need} MB) does not fit: "
                           f"{free_mb} MB free on {gpu_name}")
    before = torch.cuda.memory_allocated()
    t0 = time.perf_counter()
    clip_type = comfy.sd.CLIPType[spec.type.upper()]
    if is_gguf(spec.path):
        if gguf_support.arch(spec.path) == "qwen3vl":
            print(f"WARNING: '{spec.name}' {gguf_support.QWEN3VL_WARNING}", flush=True)
        clip = gguf.load(spec.path, clip_type, dev)
    else:
        clip = comfy.sd.load_clip([spec.path], clip_type=clip_type,
                                  model_options={"load_device": dev, "offload_device": dev})
    cls = type(getattr(clip.cond_stage_model, clip.cond_stage_model.clip).transformer)
    empty_think = "BaseQwen3" in [b.__name__ for b in cls.__mro__] and not cls.__name__.startswith("Qwen3VL")
    m = Loaded(spec, clip, empty_think, None, 0)
    encode(m, "warmup", {})
    m.vram_mb = (torch.cuda.memory_allocated() - before) >> 20
    try:
        measured[_file_key(spec.path)] = m.vram_mb
    except OSError:
        pass
    if probe:
        check_generate(m)
    else:
        m.generate_checked = False
        remembered(m)
    can_generate = (m.generate_error or "ok") if probe else "checked at first use"
    print(f"loaded '{spec.name}' ({os.path.basename(spec.path)} as {spec.type}, {cls.__name__}) "
          f"in {time.perf_counter() - t0:.1f}s, {m.vram_mb} MB, text generation: {can_generate}", flush=True)
    return m


def load_all(specs):
    import torch

    for spec in specs:
        try:
            models[spec.name] = load_model(spec)
        except torch.OutOfMemoryError as e:
            raise StartupError(f"'{spec.name}' ran out of VRAM while loading on {gpu_name}: {e}") from None
    free_mb = torch.cuda.mem_get_info()[0] >> 20
    for m in models.values():
        print(f"  {m.spec.name:32} {m.vram_mb:6} MB", flush=True)
    print(f"  {'free':32} {free_mb:6} MB on {gpu_name}", flush=True)
    if free_mb < MIN_FREE_MB:
        print(f"WARNING: only {free_mb} MB VRAM left for working memory; long prompts or reference images "
              f"may run out of memory", flush=True)


def unload_model(m):
    import gc
    import torch
    import comfy.model_management as mm

    patcher = m.clip.patcher
    mm.current_loaded_models[:] = [lm for lm in mm.current_loaded_models if lm.model is not patcher]
    m.clip = None
    del patcher
    gc.collect()
    mm.soft_empty_cache()
    print(f"unloaded '{m.spec.name}' as {m.spec.type}; {torch.cuda.memory_allocated() >> 20} MB still allocated "
          f"on {gpu_name}", flush=True)


def free_vram_mb():
    import torch

    return (torch.cuda.mem_get_info()[0] + torch.cuda.memory_reserved() - torch.cuda.memory_allocated()) >> 20


_listing = (None, 0.0, {})


def model_paths(fresh=False):
    global _listing
    folders = tuple(models_dirs or [models_dir])
    key, when, paths = _listing
    if fresh or key != folders or time.monotonic() - when > LISTING_TTL_S:
        paths = list_models(list(folders), gguf=gguf is not None)
        _listing = (folders, time.monotonic(), paths)
    return paths


def folders_text():
    return ", ".join(models_dirs or [models_dir])


def load_on_demand(file, typ):
    import gc
    import comfy.model_management as mm

    try:
        return load_model(ModelSpec(file, model_paths()[file], typ, 0), probe=False, check_fit=False)
    except BaseException:
        gc.collect()
        mm.cleanup_models()
        mm.soft_empty_cache()
        raise


def first_model():
    return next(iter(models.values()))


def get_model(name, typ=None):
    if not name:
        raise BadRequest(f"no model given; this server has: {list(models)}")
    if name not in models:
        raise UnknownModel(f"unknown model '{name}'; this server has: {list(models)}")
    m = models[name]
    if typ and typ != m.spec.type:
        raise BadRequest(f"model '{name}' is type {m.spec.type} here, the request expects {typ}")
    return m


def get_on_demand(name, typ):
    files = list(model_paths())
    if not name:
        raise BadRequest(f"no model given; {folders_text()} has: {files}")
    if not isinstance(name, str):
        raise BadRequest(f"model must be a file name, got {name!r}")
    if name not in files:
        files = list(model_paths(fresh=True))
    if name not in files:
        raise UnknownModel(f"unknown model '{name}'; {folders_text()} has: {files}")
    if not typ:
        raise BadRequest(f"no type given; one of: {sorted(valid_types)}")
    if not isinstance(typ, str) or typ not in valid_types:
        raise BadRequest(f"unknown type {typ!r}; one of: {sorted(valid_types)}")
    try:
        return get_making_room(name, typ)
    except FileNotFoundError:
        model_paths(fresh=True)
        raise UnknownModel(f"'{name}' is not in {folders_text()} any more (deleted or moved); pick another file")


class NeedsComfyUIRoom(DoesNotFit):
    def __init__(self, message, need_mb):
        super().__init__(message)
        self.need_mb = need_mb


def room_from_comfyui(file):
    if not shares_card():
        return None
    need = need_mb(model_paths()[file]) + cache.reserve_mb
    kept_mb = sum(getattr(m, "vram_mb", 0) for f, t, m, _ in cache.loaded() if (f, t) in cache.kept)
    return need if need + kept_mb <= usable_mb else None


def evictable_mb():
    return sum(getattr(m, "vram_mb", 0) for f, t, m, _ in cache.loaded() if (f, t) not in cache.kept)


def comfyui_call(method, path, body=None, timeout=3):
    import urllib.request
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(comfyui_url + path, data=data, method=method,
                                 headers={"Content-Type": "application/json"} if data else {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read() or b"null")


def comfyui_idle():
    if not comfyui_url:
        return False
    try:
        q = comfyui_call("GET", "/queue")
        return not q.get("queue_running") and not q.get("queue_pending")
    except Exception as e:
        print(f"WARNING: can't ask ComfyUI at {comfyui_url} for its queue: {type(e).__name__}: {e}", flush=True)
        return False


def ask_comfyui_to_unload(need):
    print(f"{gpu_name} is shared with ComfyUI: asking it to unload its models ({need} MB needed, "
          f"{free_vram_mb() + evictable_mb()} MB free)", flush=True)
    try:
        comfyui_call("POST", "/free", {"unload_models": True})
    except Exception as e:
        print(f"WARNING: ComfyUI at {comfyui_url} did not take /free: {type(e).__name__}: {e}", flush=True)
        return False
    for _ in range(int(COMFYUI_FREE_WAIT_S * 2)):
        if free_vram_mb() + evictable_mb() >= need:
            return True
        time.sleep(0.5)
    print(f"WARNING: ComfyUI did not free enough within {COMFYUI_FREE_WAIT_S} s "
          f"({free_vram_mb() + evictable_mb()} MB free)", flush=True)
    return False


def get_making_room(file, typ):
    try:
        return cache.get(file, typ)
    except DoesNotFit as e:
        need = room_from_comfyui(file)
        if need is None:
            raise
        if comfyui_idle() and ask_comfyui_to_unload(need):
            return cache.get(file, typ)
        raise NeedsComfyUIRoom(f"{e}; the rest is held by the ComfyUI rendering on this card, which is busy: its "
                               f"models are unloaded for a request from that ComfyUI itself, otherwise try again when "
                               f"it is idle", need) from None


def lookup(req):
    if cache is not None:
        return get_on_demand(req.get("model"), req.get("type"))
    return get_model(req.get("model"), req.get("type"))


def encode(m, text, tokenize_kwargs):
    tokens = m.clip.tokenize(text, **tokenize_kwargs)
    return m.clip.encode_from_tokens_scheduled(tokens, show_pbar=False)


def generate(m, prompt, system_prompt, max_length, temperature, top_k, top_p, seed, repetition_penalty=1.0,
             think=False, images=()):
    text = f"<|im_start|>system\n{system_prompt}<|im_end|>\n" if system_prompt else ""
    text += f"<|im_start|>user\n{locate.user_turn(prompt, len(images))}<|im_end|>\n<|im_start|>assistant\n"
    if m.empty_think and not think:
        text += "<think>\n\n</think>\n\n"
    kwargs = {"images": [i[..., :3] for i in images]} if images else {}
    tokens = m.clip.tokenize(text, llama_template="{}", min_length=1, **kwargs)
    ids = m.clip.generate(tokens, do_sample=temperature > 0, max_length=max_length, temperature=temperature,
                          top_k=top_k, top_p=top_p, seed=seed, repetition_penalty=repetition_penalty)
    return m.clip.decode(ids)


def textgen(m, text, tokenize_kwargs, generate_kwargs):
    import inspect

    params = inspect.signature(m.clip.generate).parameters
    takes_any = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
    kept = {k: v for k, v in generate_kwargs.items() if takes_any or k in params}
    tokens = m.clip.tokenize(text, **tokenize_kwargs)
    ids = [int(i) for i in m.clip.generate(tokens, **kept)]
    return ids, m.clip.decode(ids), prompt_ids(tokens), sorted(set(generate_kwargs) - set(kept))


def prompt_ids(tokens):
    if isinstance(tokens, dict):
        tokens = next(iter(tokens.values()))
    return [[t[0] for t in batch if isinstance(t, tuple) and isinstance(t[0], int) and not isinstance(t[0], bool)]
            for batch in tokens]


def check_generate_kwargs(kw):
    if not isinstance(kw, dict):
        raise BadRequest("generate_kwargs must be a JSON object")
    if "tokens" in kw:
        raise BadRequest("generate_kwargs can't set 'tokens'; send the prompt as 'text'")
    if "max_length" in kw:
        n = kw["max_length"]
        if not isinstance(n, int) or isinstance(n, bool) or not 1 <= n <= MAX_GENERATE_TOKENS:
            raise BadRequest(f"max_length must be a whole number from 1 to {MAX_GENERATE_TOKENS}, got {n!r}")


def pe_refusal(file):
    name = os.path.basename(file) if isinstance(file, str) else ""
    if "pe_t2i" in name.lower() or "pe_i2i" in name.lower():
        return f"task 'locate' is not supported with Qwen's prompt-enhancer models ({name}); {VISION_HINT}"
    return None


def check_ids(ids):
    if not isinstance(ids, list) or not all(isinstance(i, int) and not isinstance(i, bool) for i in ids):
        raise BadRequest("'ids' must be a list of token ids (integers)")


def remembered(m):
    got = caps.get(m.spec.path, m.spec.type)
    if capabilities.TEXT in got:
        m.generate_checked, m.generate_error = True, got[capabilities.TEXT]["why"] or None
    if capabilities.VISION in got:
        m.vision_checked, m.vision_error = True, got[capabilities.VISION]["why"] or None


def check_generate(m):
    if m.generate_checked:
        return
    probe_generate(m)
    m.generate_checked = True
    caps.record(m.spec.path, m.spec.type, capabilities.TEXT, m.generate_error)


def check_vision(m):
    if m.vision_checked:
        return
    probe_vision(m)
    m.vision_checked = True
    caps.record(m.spec.path, m.spec.type, capabilities.VISION, m.vision_error)


def probe_generate(m):
    import torch

    name = os.path.basename(m.spec.path)
    if "pe_t2i" in name.lower():
        return
    try:
        text = generate(m, "What is the capital of France?", "Answer with one word.", 8, 0, 1, 1.0, 0)
    except torch.OutOfMemoryError:
        raise
    except Exception as e:
        m.generate_error = f"text generation is not supported by this encoder ({name}): {type(e).__name__}: {e}"
        return
    if "paris" not in text.lower() and not text.lstrip().startswith("<think>"):
        m.generate_error = f"text generation is not supported by this encoder ({name}): test answer was {text[:40]!r}"


def probe_vision(m):
    import torch

    name = os.path.basename(m.spec.path)
    for colour, rgb in VISION_TEST:
        picture = torch.tensor(rgb).repeat(1, 256, 256, 1)
        try:
            text = generate(m, "What colour is this image? Answer with one word.", "", 8, 0, 1, 1.0, 0,
                            images=[picture])
        except torch.OutOfMemoryError:
            raise
        except Exception as e:
            m.vision_error = f"this encoder can't read images ({name}): {type(e).__name__}: {e}; {VISION_HINT}"
            return
        if not re.search(rf"\b{colour}\b", text.lower()):
            m.vision_error = (f"this encoder can't read images ({name}): it called a plain {colour} test image "
                              f"{text[:40]!r}; {VISION_HINT}")
            return


def restart():
    print(log_line("RESTART", commit=repo_commit(HERE)), flush=True)
    sys.stderr.flush()
    os.execv(sys.executable, [sys.executable, os.path.abspath(__file__), *launch_argv])


def stop():
    print(log_line("STOP"), flush=True)
    sys.stderr.flush()
    os._exit(0)


def model_info(m):
    return {"name": m.spec.name, "file": os.path.basename(m.spec.path), "type": m.spec.type, "gpu": gpu_name,
            "generate": m.generate_error is None, "vram_mb": m.vram_mb,
            "vision": (m.vision_error is None) if m.vision_checked else None}


def ondemand_models():
    loaded = {}
    for file, typ, _, _ in cache.loaded():
        loaded.setdefault(file, []).append(typ)
    listing = []
    for f, path in model_paths().items():
        try:
            size_mb = os.path.getsize(path) >> 20
        except OSError:
            continue
        listing.append({"name": f, "file": f, "size_mb": size_mb, "loaded_as": loaded.get(f, []),
                        "kept_as": kept_types(f),
                        "arch": hints.arch(path), "types": file_types(f, path), "picked": picked.get(f),
                        "can": caps.for_file(path), "expected": expected_capabilities(path)})
    return {"mode": "on_demand", "gpu": gpu_name, "version": VERSION, "gpus": [card_info(full=False)],
            "models": listing}


def card_info(full=True):
    info = {"uuid": gpu_uuid, "name": gpu_name, "render": shares_card(), "up": True}
    if shares_card() and cache is not None:
        info["idle_unload_s"] = SHARED_IDLE_UNLOAD_S
    if full:
        info.update({"total_mb": total_mb, "usable_mb": usable_mb, "reserve_mb": cache.reserve_mb, "vram": vram(),
                     "busy": busy_info()})
    return info


def expected_capabilities(path):
    arch = gguf_support.arch(path) if is_gguf(path) else None
    mmproj = bool(arch == "qwen2vl" and gguf is not None and gguf.has_mmproj(path))
    out = capabilities.expected(hints.arch(path), arch, mmproj)
    if pe_refusal(path):
        out[capabilities.VISION] = {"ok": False, "why": "Qwen's prompt-enhancer models don't read images here "
                                                        "(Remote Locate refuses them)", "checked": False}
    return out


def file_types(file, path):
    hinted = hints.types(path)
    t = picked.get(file)
    return [t] + [h for h in hinted if h != t] if t else hinted


def pick_types(updates):
    changed = {f: t for f, t in updates.items() if picked.get(f) != t}
    if not changed:
        return 0
    picked.update(changed)
    if types_path:
        type_file.save(types_path, picked)
    if scan_enabled:
        request_scan()
    return len(changed)


def kept_types(file):
    return [t for f, t in kept if f == file]


def size_mb(file):
    try:
        return need_mb(model_paths()[file])
    except (OSError, KeyError):
        return 0


def ondemand_status():
    d = ondemand_models()
    loaded = {}
    for f, t, m, idle in cache.loaded():
        loaded.setdefault(f, []).append({
            "type": t, "gpu": gpu_uuid, "vram_mb": m.vram_mb, "idle_s": round(idle),
            "generate": (m.generate_error is None) if m.generate_checked else None,
            "vision": (m.vision_error is None) if m.vision_checked else None})
    for m in d["models"]:
        m["loaded"] = loaded.get(m["file"], [])
    d.update({"total_mb": total_mb, "usable_mb": usable_mb, "reserve_mb": cache.reserve_mb, "vram": vram(),
              "gpus": [card_info()],
              "kept": [{"file": f, "type": t, "gpu": gpu_uuid} for f, t in kept], "kept_mb": sum(size_mb(f) for f, _ in kept),
              "keep_errors": keep_errors, "keep_file": keep_path, "all_types": sorted(valid_types),
              "busy": busy_info(), "scan": scan_info()})
    return d


def keep_models(entries):
    global kept
    for key in kept:
        if key not in entries:
            cache.unload(*key)
    kept = []
    cache.kept = set(entries)
    try:
        for file, typ in entries:
            get_making_room(file, typ)
            kept.append((file, typ))
    finally:
        cache.kept = set(kept)


def load_kept_at_startup():
    entries, problem = keep_file.read(keep_path)
    if problem:
        keep_errors.append(problem)
    names = model_paths()
    for file, typ in entries:
        if file not in names:
            keep_errors.append(f"kept model '{file}' is not in the folders any more")
        elif typ not in valid_types:
            keep_errors.append(f"kept model '{file}' has an unknown type '{typ}'")
        else:
            cache.kept.add((file, typ))
            try:
                cache.get(file, typ)
                kept.append((file, typ))
            except Exception as e:
                cache.kept.discard((file, typ))
                keep_errors.append(f"kept model '{file}' as {typ} did not load: {e}")
    for err in keep_errors:
        print(f"WARNING: {err}", flush=True)
    if kept:
        print(f"kept: {', '.join(f'{f} as {t}' for f, t in kept)}", flush=True)


def ondemand_health():
    return {"status": "ok", "mode": "on_demand", "models_dir": models_dir, "models_dirs": models_dirs or [models_dir],
            "gpu": gpu_name, "gpu_uuid": gpu_uuid, "port": port, "version": VERSION,
            "pid": os.getpid(), "commit": commit, "uptime_s": stats.uptime_s(), "requests": stats.requests,
            "last_error": stats.last_error,
            "loaded": [{"file": f, "type": t, "vram_mb": m.vram_mb,
                        "generate": (m.generate_error is None) if m.generate_checked else None,
                        "vision": (m.vision_error is None) if m.vision_checked else None,
                        "idle_s": round(idle)} for f, t, m, idle in cache.loaded()],
            "gguf": gguf.folder if gguf else None, "gguf_note": gguf_note, "swaps": cache.swaps, "last_swap": cache.last_swap, "reserve_mb": cache.reserve_mb, "vram": vram(),
            "busy": busy_info(), "scan": scan_info()}


def serialize(conditioning):
    import torch
    from safetensors.torch import save

    tensors, extras = {}, []
    for i, (cond, extra) in enumerate(conditioning):
        tensors[f"{i}.cond"] = cond.contiguous().cpu()
        plain = {}
        for k, v in extra.items():
            if isinstance(v, torch.Tensor):
                tensors[f"{i}.{k}"] = v.contiguous().cpu()
            else:
                json.dumps(v)
                plain[k] = v
        extras.append(plain)
    return save(tensors, metadata={"extras": json.dumps(extras)})


def parse_tensor_request(body):
    from safetensors.torch import load

    tensors = load(body)
    header_len = int.from_bytes(body[:8], "little")
    meta = json.loads(body[8:8 + header_len])["__metadata__"]
    kwargs = json.loads(meta["tokenize_kwargs"])
    for k, names in json.loads(meta["tensor_kwargs"]).items():
        kwargs[k] = [tensors[n] for n in names] if isinstance(names, list) else tensors[names]
    req = {"model": meta.get("model"), "type": meta.get("type"), "text": meta.get("text"), "tokenize_kwargs": kwargs,
           "tensors": len(tensors)}
    if "generate_kwargs" in meta:
        req["generate_kwargs"] = json.loads(meta["generate_kwargs"])
    return req


def parse_generate_images(body):
    from safetensors.torch import load

    try:
        tensors = load(body)
        header_len = int.from_bytes(body[:8], "little")
        meta = json.loads(body[8:8 + header_len]).get("__metadata__") or {}
    except Exception as e:
        raise ValueError(f"not a safetensors body: {type(e).__name__}: {e}") from None
    req = json.loads(meta["request"])
    if not isinstance(req, dict):
        raise ValueError("metadata 'request' must be a JSON object")
    names = [f"images.{i}" for i in range(len(tensors))]
    if sorted(tensors) != sorted(names):
        raise ValueError(f"image tensors must be named images.0, images.1, ...; got {sorted(tensors)}")
    return req, [tensors[n] for n in names]


def check_images(images):
    for i, t in enumerate(images, 1):
        if t.ndim != 4 or t.shape[0] != 1 or t.shape[-1] < 3 or t.shape[1] == 0 or t.shape[2] == 0:
            raise BadRequest(f"image {i} must be one image shaped [1, height, width, 3], got {list(t.shape)}")
        if not t.is_floating_point():
            raise BadRequest(f"image {i} must hold floats from 0 to 1 (a ComfyUI IMAGE), got {t.dtype}")
        if t.shape[1] * t.shape[2] > MAX_IMAGE_PIXELS:
            raise BadRequest(f"image {i} is {t.shape[2]}x{t.shape[1]}; send at most {MAX_IMAGE_PIXELS} pixels per "
                             f"image (the nodes scale to 1048576)")


def vram():
    import torch

    free, total = torch.cuda.mem_get_info()
    return {"allocated_mb": torch.cuda.memory_allocated() // 2**20,
            "reserved_mb": torch.cuda.memory_reserved() // 2**20,
            "device_used_mb": (total - free) // 2**20}


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json", headers={}):
        self.sent = True
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for k, v in headers.items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            if cache is not None:
                return self._send(200, ondemand_health())
            d = first_model()
            return self._send(200, {
                "status": "ok", "model": os.path.basename(d.spec.path), "type": d.spec.type,
                "generate": d.generate_error is None, "port": port, "pid": os.getpid(), "commit": commit,
                "uptime_s": stats.uptime_s(), "requests": stats.requests, "last_error": stats.last_error,
                "startup": {m.spec.name: {"loaded": True, "generate": m.generate_error is None,
                                          "generate_error": m.generate_error} for m in models.values()},
                "models": [model_info(m) for m in models.values()], "vram": vram(), "busy": busy_info()})
        if self.path == "/status":
            if cache is not None:
                return self._send(200, ondemand_status())
            return self._send(200, {"mode": "preloaded", "gpu": gpu_name, "vram": vram(), "keep_file": None,
                                    "version": VERSION, "busy": busy_info(), "models": [model_info(m) for m in models.values()]})
        if self.path == "/models":
            if cache is not None:
                return self._send(200, ondemand_models())
            return self._send(200, {"mode": "preloaded", "version": VERSION,
                                    "models": [model_info(m) for m in models.values()]})
        self._send(404, {"error": "not found"})

    def do_POST(self):
        global busy, last_post
        posts.add(self)
        try:
            with gpu_lock:
                fields = {}
                busy = (self.path.lstrip("/"), time.monotonic(), fields)
                try:
                    self._do_post(fields)
                finally:
                    busy = None
        finally:
            last_post = time.monotonic()
            posts.discard(self)

    def _do_post(self, fields):
        op = self.path.lstrip("/")
        t0 = time.perf_counter()
        self.after = None
        self.sent = False
        try:
            try:
                code, error = self._post(fields)
            except UnknownModel as e:
                code, error = 404, str(e)
                self._reply_error(code, error)
            except DoesNotFit as e:
                code, error = 507, str(e)
                if not self.sent:
                    shared = {"shared_gpu": gpu_uuid, "need_mb": e.need_mb} if isinstance(e, NeedsComfyUIRoom) else {}
                    self._send(code, {"error": error, **shared})
            except BadRequest as e:
                code, error = 400, f"bad request: {e}"
                self._reply_error(code, error)
            except ConnectionError:
                raise
            except Exception as e:
                code, error = 500, f"{type(e).__name__}: {e}"
                self._reply_error(code, error)
        except ConnectionError as e:
            code, error = 499, f"client disconnected: {type(e).__name__}"
            self.after = None
        model_name = fields.pop("model", None)
        stats.record(op, model_name, error)
        print(log_line("REQ", op=op, model=model_name, ms=round((time.perf_counter() - t0) * 1000), **fields,
                       status="ok" if error is None else "error", code=code, **({"error": error} if error else {})),
              flush=True)
        if self.after:
            self.after()

    def _keep(self, req, fields):
        global keep_errors
        if cache is None:
            error = "this server keeps the models of its config; edit the config to change them"
        elif keep_path is None:
            error = "this server has no keep list (start it with --keep-file; a ComfyUI that serves does)"
        else:
            error = None
        if error:
            self._send(409, {"error": error})
            return 409, error
        try:
            entries = keep_file.parse(req, model_paths(), valid_types)
            new_types = type_file.parse(req.get("types") or {}, model_paths(), valid_types)
        except ValueError as e:
            raise BadRequest(str(e)) from None
        fields["keep"] = len(entries)
        plan_mb = sum(size_mb(f) for f, _ in entries)
        if plan_mb + cache.reserve_mb > usable_mb:
            error = (f"the kept models need {plan_mb} MB + {cache.reserve_mb} MB working memory, more than the "
                     f"{usable_mb} MB this server can use on {gpu_name}; keep fewer")
            self._send(409, {"error": error})
            return 409, error
        fields["types"] = pick_types({**new_types, **dict(entries)})
        try:
            keep_models(entries)
        except Exception as e:
            keep_errors = [f"{type(e).__name__}: {e}" if not isinstance(e, DoesNotFit) else str(e)]
            keep_file.save(keep_path, kept)
            raise
        keep_errors = []
        keep_file.save(keep_path, kept)
        self._send(200, ondemand_status())
        return 200, None

    def _reply_error(self, code, error):
        if not self.sent:
            self._send(code, {"error": error})

    def _post(self, fields):
        raw = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        images = []
        try:
            if self.headers.get("Content-Type") == "application/octet-stream":
                if self.path == "/generate":
                    req, images = parse_generate_images(raw)
                else:
                    req = parse_tensor_request(raw)
                    fields["images"] = req["tensors"]
            else:
                req = json.loads(raw or b"{}")
        except (ValueError, KeyError) as e:
            raise BadRequest(f"{type(e).__name__}: {e}") from None
        if not isinstance(req, dict):
            raise BadRequest("body must be a JSON object")
        if self.path in ("/restart", "/stop"):
            if not authorized(self.headers.get("X-Enc-Auth"), secret):
                self._send(401, {"error": "unauthorized"})
                return 401, "unauthorized"
            if self.path == "/restart":
                err = preflight(launch_argv)
                if err:
                    error = f"restart refused: {err}"
                    self._send(409, {"error": error})
                    return 409, error
                self._send(200, {"restarting": True})
                self.after = restart
            else:
                self._send(200, {"stopping": True})
                self.after = stop
            return 200, None
        if self.path == "/keep":
            return self._keep(req, fields)
        if self.path == "/encode":
            fields["model"] = req.get("model")
            m = lookup(req)
            fields["model"] = m.spec.name
            t0 = time.perf_counter()
            if not isinstance(req.get("text"), str):
                raise BadRequest("missing 'text'")
            conditioning = encode(m, req["text"], req.get("tokenize_kwargs", {}))
            t1 = time.perf_counter()
            body = serialize(conditioning)
            t2 = time.perf_counter()
            fields["tokens"] = conditioning[0][0].shape[1]
            self._send(200, body, "application/octet-stream", {
                "X-Encode-Ms": f"{(t1 - t0) * 1000:.1f}",
                "X-Serialize-Ms": f"{(t2 - t1) * 1000:.1f}",
            })
            return 200, None
        if self.path == "/textgen":
            fields["model"] = req.get("model")
            m = lookup(req)
            fields["model"] = m.spec.name
            if not isinstance(req.get("text"), str):
                raise BadRequest("missing 'text'")
            gen = req.get("generate_kwargs", {})
            check_generate_kwargs(gen)
            check_generate(m)
            if m.generate_error:
                return self._refuse(501, m.generate_error)
            t0 = time.perf_counter()
            ids, text, prompt, ignored = textgen(m, req["text"], req.get("tokenize_kwargs", {}), gen)
            fields["tokens"] = len(ids)
            if ignored:
                fields["ignored"] = ",".join(ignored)
            self._send(200, {"ids": ids, "text": text, "prompt_ids": prompt, "ignored": ignored,
                             "ms": round((time.perf_counter() - t0) * 1000, 1)})
            return 200, None
        if self.path == "/decode":
            fields["model"] = req.get("model")
            m = lookup(req)
            fields["model"] = m.spec.name
            check_ids(req.get("ids"))
            skip = req.get("skip_special_tokens", True)
            self._send(200, {"text": m.clip.decode(req["ids"], skip_special_tokens=bool(skip))})
            return 200, None
        if self.path == "/generate":
            fields["model"] = req.get("model")
            try:
                locate.parse_task(req, len(images))
            except ValueError as e:
                raise BadRequest(str(e)) from None
            check_images(images)
            refusal = pe_refusal(req.get("model")) if cache is not None else None
            if refusal:
                return self._refuse(501, refusal)
            m = lookup(req)
            fields["model"] = m.spec.name
            return self._locate(m, req["prompt"], images[0], fields)
        self._send(404, {"error": "not found"})
        return 404, "not found"

    def _refuse(self, code, error):
        self._send(code, {"error": error})
        return code, error

    def _locate(self, m, what, image, fields):
        fields.update(task="locate", images=1)
        refusal = pe_refusal(m.spec.path)
        if refusal:
            return self._refuse(501, refusal)
        check_generate(m)
        if m.generate_error:
            return self._refuse(501, f"{m.generate_error}; {VISION_HINT}")
        check_vision(m)
        if m.vision_error:
            return self._refuse(501, m.vision_error)
        t0 = time.perf_counter()
        height, width = image.shape[1:3]
        raw = generate(m, locate.locate_prompt(what), "", locate.LOCATE_TOKENS, 0, 1, 1.0, 0, images=[image])
        try:
            answer = locate.answer_text(raw)
            boxes = locate.parse_boxes(answer, width, height)
        except (locate.EmptyAnswer, locate.NothingFound) as e:
            said = str(e) if isinstance(e, locate.NothingFound) else "(nothing)"
            return self._refuse(422, f"nothing found for {what.strip()!r}; the model answered: {said}")
        fields["boxes"] = len(boxes)
        self._send(200, {"boxes": boxes, "text": answer, "ms": round((time.perf_counter() - t0) * 1000, 1),
                         "preset": "locate", "image_size": [width, height]})
        return 200, None

    def log_request(self, code="-", size="-"):
        if self.command == "GET" and int(code) < 400 and not debug_log():
            return
        super().log_request(code, size)

    def log_message(self, fmt, *args):
        print(f"{self.address_string()} {fmt % args}", flush=True)


def debug_log():
    return os.environ.get("REMOTE_TE_LOG", "").strip().lower() == "debug"


def shares_card():
    return bool(render_uuid and gpu_uuid and render_uuid.lower() == gpu_uuid.lower())


def unload_idle_once():
    if not gpu_lock.acquire(blocking=False):
        return []
    try:
        return cache.unload_idle(SHARED_IDLE_UNLOAD_S, ", card shared with ComfyUI")
    finally:
        gpu_lock.release()


def unload_idle_forever(every_s=10):
    while True:
        time.sleep(every_s)
        try:
            unload_idle_once()
        except Exception as e:
            print(f"WARNING: idle unload failed: {type(e).__name__}: {e}", flush=True)


BUSY, LATER, CHECKED, SKIPPED = "busy", "later", "checked", "skipped"


def idle_s():
    return 0.0 if posts else time.monotonic() - last_post


def scan_info():
    s = scan
    return dict(s) if s else None


def unchecked(file, path, typ):
    got = caps.get(path, typ)
    want = [] if capabilities.TEXT in got else [capabilities.TEXT]
    generates = got[capabilities.TEXT]["ok"] if capabilities.TEXT in got else True
    if capabilities.VISION not in got and generates and pe_refusal(file) is None:
        want.append(capabilities.VISION)
    return want


def scan_targets():
    return [(file, typ) for file, path in model_paths(fresh=True).items() for typ in file_types(file, path)
            if unchecked(file, path, typ)]


def scan_room(path):
    need = need_mb(path) + cache.reserve_mb
    if need > usable_mb:
        return "never"
    return "now" if free_vram_mb() >= need else "later"


def scan_checks(m):
    check_generate(m)
    if m.generate_error is None and pe_refusal(m.spec.path) is None:
        check_vision(m)


def scan_marks(path, typ):
    got = caps.get(path, typ)
    return ", ".join(f"{what} {'?' if what not in got else 'yes' if got[what]['ok'] else 'no'}"
                     for what in (capabilities.TEXT, capabilities.VISION))


def scan_one(file, typ):
    global busy
    if idle_s() < SCAN_IDLE_S or not gpu_lock.acquire(blocking=False):
        return BUSY
    label = f"{file} as {typ}"
    try:
        if idle_s() < SCAN_IDLE_S:
            return BUSY
        path = model_paths()[file]
        m = next((e for f, t, e, _ in cache.loaded() if (f, t) == (file, typ)), None)
        room = "now" if m is not None else scan_room(path)
        if room == "later":
            return LATER
        if room == "never":
            print(f"scan: skipped {label}: it needs {need_mb(path)} MB + {cache.reserve_mb} MB working memory, "
                  f"{gpu_name} has {usable_mb} MB", flush=True)
            return SKIPPED
        busy = ("scan", time.monotonic(), {"model": file})
        scan["now"] = label
        t0 = time.perf_counter()
        mine = load_on_demand(file, typ) if m is None else None
        try:
            scan_checks(mine or m)
        finally:
            if mine is not None:
                unload_model(mine)
        print(f"scan: {label}: {scan_marks(path, typ)}, {time.perf_counter() - t0:.1f} s", flush=True)
        return CHECKED
    except Exception as e:
        why = (str(e).splitlines() or [""])[0]
        print(f"scan: skipped {label}: {type(e).__name__}: {why} (not tried again until the server restarts)",
              flush=True)
        return SKIPPED
    finally:
        if scan["now"] == label:
            busy, scan["now"] = None, None
        gpu_lock.release()


def scan_pass(todo, sleep=time.sleep):
    later = []
    for file, typ in todo:
        while (did := scan_one(file, typ)) == BUSY:
            sleep(1)
        if did == LATER:
            later.append((file, typ))
        else:
            scan["done"] += 1
            scan[did] += 1
    return later


def scan_abilities(sleep=time.sleep):
    global scan
    todo = scan_targets()
    scan = {"done": 0, "total": len(todo), "now": None, CHECKED: 0, SKIPPED: 0}
    if not todo:
        print("scan: nothing to check (every file with a known type has its results)", flush=True)
        return
    print(f"scan: checking what {len(todo)} files can do (text generation, vision), one at a time while no request "
          f"has run for {SCAN_IDLE_S} s", flush=True)
    t0 = time.monotonic()
    waiting = False
    while todo := scan_pass(todo, sleep):
        if not waiting:
            waiting = True
            print(f"scan: {len(todo)} wait for free VRAM (no room beside the loaded models; nothing is unloaded for "
                  f"the scan): {', '.join(f'{f} as {t}' for f, t in todo)}", flush=True)
        sleep(SCAN_RETRY_S)
    print(f"scan: done in {time.monotonic() - t0:.0f} s: {scan[CHECKED]} checked, {scan[SKIPPED]} skipped",
          flush=True)


def scan_in_background():
    global scan_thread
    while True:
        try:
            scan_abilities()
        except Exception as e:
            print(f"WARNING: scan stopped: {type(e).__name__}: {e}", flush=True)
        with scan_lock:
            if not scan_again.is_set():
                scan_thread = None
                return
            scan_again.clear()


def request_scan():
    global scan_thread
    with scan_lock:
        if scan_thread is not None:
            scan_again.set()
            return
        scan_thread = threading.Thread(target=scan_in_background, name="ability scan", daemon=True)
        scan_thread.start()


def set_up_capabilities(path, comfy_path):
    global caps
    version = "unknown"
    try:
        with open(os.path.join(comfy_path, "comfyui_version.py"), encoding="utf-8") as f:
            found = re.search(r"__version__\s*=\s*['\"]([^'\"]+)", f.read())
            version = found.group(1) if found else version
    except OSError:
        pass
    caps = capabilities.Store(os.path.abspath(path) if path else None, version)


def set_up_gguf(custom_nodes_dirs):
    global gguf, gguf_note
    folder = gguf_support.find_pack(custom_nodes_dirs)
    if folder is None:
        gguf_note = f"ComfyUI-GGUF not found in {', '.join(custom_nodes_dirs)}"
        print(f"GGUF encoders: off ({gguf_note})", flush=True)
        return
    try:
        gguf = gguf_support.GgufLoader(folder)
    except Exception as e:
        gguf_note = f"ComfyUI-GGUF at {folder} did not import: {type(e).__name__}: {e}"
        print(f"WARNING: GGUF encoders: off ({gguf_note})", flush=True)
        return
    print(f"GGUF encoders: on (ComfyUI-GGUF at {folder})", flush=True)


def main():
    global gpu_name, commit, port, secret, launch_argv, cache, models_dir, models_dirs, valid_types, hints
    global keep_path, total_mb, usable_mb, gpu_uuid, render_uuid, last_post, comfyui_url, types_path, picked
    global scan_enabled
    launch_argv = [a for a in sys.argv[1:] if a != "--check"]
    ap = argparse.ArgumentParser()
    ap.add_argument("--comfy", help="ComfyUI folder (only its code is used); default: $COMFYUI_DIR")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--config", help="TOML config with the models to serve (see enc-servers/models.example.toml)")
    src.add_argument("--models-dir", action="append",
                     help="on-demand mode: serve any encoder file in this folder, loaded at first use (use with the "
                          "Remote CLIP Loader node); repeat for more folders, the first folder wins a name in two")
    ap.add_argument("--keep-file", help="on-demand mode: JSON list of the models to keep loaded (Remote Encoder "
                                        "Status writes it); loaded at startup, changed by POST /keep")
    ap.add_argument("--reserve", help="on-demand mode: VRAM kept free for working memory, a share of the card "
                                      "(default 10%%) or an amount (3GB, 2048MB)")
    ap.add_argument("--capabilities-file",
                    help="JSON file where the results of the text-generation / vision checks are kept (Remote "
                         "Encoder Status shows them); default: in memory only")
    ap.add_argument("--scan-abilities", action="store_true",
                    help="on-demand mode: after startup, check in the background what each file can do (text "
                         "generation, vision) that has no result in --capabilities-file; only while idle and into "
                         "free VRAM")
    ap.add_argument("--custom-nodes", action="append",
                    help="a custom_nodes folder to look for ComfyUI-GGUF in (GGUF encoders); repeat for more; "
                         "default: <ComfyUI>/custom_nodes")
    ap.add_argument("--gpu", help="serve the models on this GPU: part of its name, a UUID, or a number")
    ap.add_argument("--types-file",
                    help="on-demand mode: JSON file with the type picked per file in Remote Encoder Status (offered "
                         "first, scanned); default: in memory only")
    ap.add_argument("--comfyui-url",
                    help="the address of the ComfyUI that started this server (with --render-gpu): on a card shared "
                         "with it, a model that doesn't fit beside its models makes it unload them")
    ap.add_argument("--render-gpu", help="UUID of the card the ComfyUI that started this server renders on (marks a "
                                         "shared card in /status)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8199)
    ap.add_argument("--secret-file", default=os.path.join(HERE, "enc-servers", "secret"),
                    help="file with the secret for /restart and /stop (header X-Enc-Auth)")
    ap.add_argument("--check", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()
    port = args.port
    commit = repo_commit(HERE)
    try:
        problems = prompt_files.check()
        if problems:
            raise ConfigError("; ".join(problems))
        if args.keep_file is not None and not args.models_dir:
            raise ConfigError("--keep-file only applies to an on-demand server (--models-dir)")
        if args.reserve is not None and not args.models_dir:
            raise ConfigError("--reserve only applies to an on-demand server (--models-dir)")
        if args.scan_abilities and not args.models_dir:
            raise ConfigError("--scan-abilities only applies to an on-demand server (--models-dir)")
        if args.types_file and not args.models_dir:
            raise ConfigError("--types-file only applies to an on-demand server (--models-dir)")
        reserve = parse_reserve(args.reserve or "10%")
        if args.models_dir:
            models_dirs = [models_dir_path(d) for d in args.models_dir]
            models_dir = models_dirs[0]
            keep_path = os.path.abspath(args.keep_file) if args.keep_file else None
            types_path = os.path.abspath(args.types_file) if args.types_file else None
            if types_path:
                picked, problem = type_file.read(types_path)
                if problem:
                    print(f"WARNING: {problem}", flush=True)
            specs = []
            if args.gpu is not None:
                gpu = resolve(args.gpu, list_gpus())
                pin(gpu)
                gpu_name, gpu_uuid = gpu.name, gpu.uuid
        else:
            specs = load_config(args.config)
        if not args.models_dir:
            gpu, specs = select(specs, args.gpu, list_gpus())
            pin(gpu)
            gpu_name, gpu_uuid = gpu.name, gpu.uuid
        comfy_path = comfy_dir(args.comfy, os.environ)
        sys.path.insert(0, comfy_path)
        try:
            import comfy.sd
        except ImportError as e:
            raise ConfigError(f"cannot import ComfyUI from {comfy_path}: {e}. Run server.py with ComfyUI's "
                              f"Python environment (the python that runs ComfyUI)") from None
        valid_types = {t.name.lower() for t in comfy.sd.CLIPType}
        check_types(specs, valid_types)
        secret = load_secret(args.secret_file)
        if args.check:
            print("check ok", flush=True)
            return
        set_up_gguf(args.custom_nodes or [os.path.join(comfy_path, "custom_nodes")])
        set_up_capabilities(args.capabilities_file, comfy_path)
        if gpu_name is None:
            import torch
            gpu_name = torch.cuda.get_device_name(0)
        if gpu_uuid is None:
            import torch
            gpu_uuid = "GPU-" + str(torch.cuda.get_device_properties(0).uuid)
        render_uuid = args.render_gpu
        comfyui_url = (args.comfyui_url or "").rstrip("/") or None
        if models_dir:
            import torch
            free, total = torch.cuda.mem_get_info()
            total_mb, usable_mb = total >> 20, free >> 20
            hints = Hints(comfy.sd.detect_te_model)
            cache = ModelCache(load_on_demand, unload_model, free_vram_mb, lambda f: need_mb(model_paths()[f]),
                               reserve_mb=reserve_mb(reserve, total_mb), oom=(torch.OutOfMemoryError,), gpu=gpu_name,
                               log=lambda line: print(line, flush=True))
            print(f"WARNING: on-demand mode: models load at first use and switching models reloads from disk "
                  f"(several seconds). Folders: {folders_text()} ({len(model_paths())} files), "
                  f"GPU: {gpu_name}, kept free for working memory: {cache.reserve_mb} MB of {total_mb} MB",
                  flush=True)
            if keep_path:
                load_kept_at_startup()
        else:
            load_all(specs)
    except (ConfigError, GpuError, StartupError, SecretError) as e:
        print(f"cannot start: {e}", file=sys.stderr, flush=True)
        sys.stdout.flush()
        os._exit(1)
    print(log_line("READY", port=port, commit=commit, version=VERSION, mode="on_demand" if cache else "preloaded",
                   models=len(models), **({"kept": len(kept)} if cache else {}), vram=f"{vram()['allocated_mb']}MB"),
          flush=True)
    if cache is not None and shares_card():
        print(f"{gpu_name} is also the card ComfyUI renders on: models not kept unload after "
              f"{SHARED_IDLE_UNLOAD_S} s unused", flush=True)
        threading.Thread(target=unload_idle_forever, name="idle unload", daemon=True).start()
    if cache is not None and args.scan_abilities:
        last_post = time.monotonic()
        scan_enabled = True
        request_scan()
    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    httpd.daemon_threads = True
    httpd.serve_forever()


if __name__ == "__main__":
    main()
