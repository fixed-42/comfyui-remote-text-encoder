import asyncio
from urllib.parse import urlparse

import requests

from . import log
from .endpoint import LocalUnavailable, find_api, is_local, local_base

KEEP_TIMEOUT_S = 600
KEEP_ONLY_LOCAL = ("the kept models of another PC are changed in that PC's ComfyUI (Remote Encoder Status with "
                   "server_url local), not from here")


_once = log.Once()


def _failed(kind, url, message):
    _once.error((kind, url), f"{kind} from {url or '(no address)'}: {message}")
    return {"error": message}


def _worked(kind, url, base):
    log.debug(f"{kind} from {url}: OK ({base})")
    _once.ok((kind, url), f"{kind} from {url}: working again ({base})")


class RelayError(Exception):
    def __init__(self, status, message, starting=False):
        super().__init__(message)
        self.status = status
        self.starting = starting


def _check_url(url):
    parsed = urlparse(url or "")
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise RelayError(400, f"not an http(s) server URL: {url!r}")


def _local_base(get_child):
    try:
        return local_base(get_child)
    except LocalUnavailable as e:
        raise RelayError(e.status, str(e), e.starting) from None


def _starting(kind, url, message):
    log.debug(f"{kind} from {url}: {message}")
    return {"error": message, "starting": True}


def _is_local(url):
    return is_local(url)


def fetch_models(url, get_child=None):
    try:
        if _is_local(url):
            base = _local_base(get_child)
            r = requests.get(base + "/models", timeout=5)
        else:
            _check_url(url)
            base, r = find_api(url, requests.get, 5)
    except RelayError as e:
        return _starting("model list", url, str(e)) if e.starting else _failed("model list", url, str(e))
    except (OSError, requests.RequestException) as e:
        return _failed("model list", url, f"{url} unreachable: {type(e).__name__}: {e}")
    if r.status_code != 200:
        return _failed("model list", url, f"{base}/models -> {r.status_code}: {r.text[:200]}")
    try:
        data = r.json()
    except ValueError:
        return _failed("model list", url, f"{base}/models did not answer JSON (is it an encoder server?)")
    _worked("model list", url, base)
    return data


def fetch_status(url, get_child):
    try:
        if _is_local(url):
            base = _local_base(get_child)
            r = requests.get(base + "/status", timeout=5)
        else:
            _check_url(url)
            base, r = find_api(url, requests.get, 5, "/status")
    except RelayError as e:
        return _starting("status", url, str(e)) if e.starting else _failed("status", url, str(e))
    except (OSError, requests.RequestException) as e:
        return _failed("status", url, f"{url} unreachable: {type(e).__name__}: {e}")
    if r.status_code != 200:
        return _failed("status", url, f"{base}/status -> {r.status_code}: {r.text[:300]}")
    try:
        data = r.json()
    except ValueError:
        return _failed("status", url, f"{base}/status did not answer JSON")
    _worked("status", url, base)
    return data


def post_keep(url, body, get_child):
    if not _is_local(url):
        return 403, {"error": KEEP_ONLY_LOCAL}
    try:
        base = _local_base(get_child)
        r = requests.post(base + "/keep", json=body, timeout=KEEP_TIMEOUT_S)
    except RelayError as e:
        log.error(f"keep on {url}: {e}")
        return e.status, {"error": str(e)}
    except (OSError, requests.RequestException) as e:
        log.error(f"keep on {url}: unreachable: {type(e).__name__}: {e}")
        return 502, {"error": f"{url} unreachable: {e}"}
    try:
        out = r.json()
    except ValueError:
        out = {"error": r.text[:300]}
    if r.status_code != 200:
        log.error(f"keep on {url} -> {r.status_code}: {out.get('error', out)}")
    else:
        log.info(f"keep on {url}: {[k['file'] for k in body.get('keep', [])]}")
    return r.status_code, out


def register(prompt_server, get_child=lambda: None, serving=None):
    from aiohttp import web
    gpu_menu = []

    async def menu():
        if not gpu_menu:
            from . import serve
            render = await asyncio.to_thread(serve.render_gpu_uuid)
            gpu_menu.extend(await asyncio.to_thread(serve.gpu_options, dict(serving.env), render_uuid=render))
        return gpu_menu

    async def serve_get(request):
        if serving is None:
            return web.json_response({"error": "serving is not available in this ComfyUI"}, status=503)
        return web.json_response({**serving.describe(), "gpus": await menu()})

    async def serve_post(request):
        if serving is None:
            return web.json_response({"error": "serving is not available in this ComfyUI"}, status=503)
        try:
            body = await request.json()
        except ValueError:
            return web.json_response({"error": "body must be JSON"}, status=400)
        code, out = await asyncio.to_thread(serving.update, body)
        if code == 200:
            log.info(f"serving settings changed: {body}; now {out['state']}"
                     + (f" ({out['error']})" if out.get("error") else ""))
        return web.json_response(out, status=code)

    async def relay_models(request):
        return web.json_response(await asyncio.to_thread(fetch_models, request.query.get("url", ""), get_child))

    async def relay_status(request):
        return web.json_response(await asyncio.to_thread(fetch_status, request.query.get("url", ""), get_child))

    async def relay_keep(request):
        try:
            body = await request.json()
        except ValueError:
            return web.json_response({"error": "body must be JSON"}, status=400)
        code, out = await asyncio.to_thread(post_keep, request.query.get("url", ""), body, get_child)
        return web.json_response(out, status=code)

    prompt_server.routes.get("/remote_te/relay/models")(relay_models)
    prompt_server.routes.get("/remote_te/relay/status")(relay_status)
    prompt_server.routes.post("/remote_te/relay/keep")(relay_keep)

    async def pack_version(request):
        from .version import VERSION
        return web.json_response({"version": VERSION})

    prompt_server.routes.get("/remote_te/version")(pack_version)
    prompt_server.routes.get("/remote_te/serve")(serve_get)
    prompt_server.routes.post("/remote_te/serve")(serve_post)
