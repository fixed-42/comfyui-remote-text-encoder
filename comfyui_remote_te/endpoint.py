SERVED_PREFIX = "/remote_te/server"


def find_api(url, get, timeout, path="/models"):
    url = url.rstrip("/")
    r = get(url + SERVED_PREFIX + path, timeout=timeout)
    if r.status_code != 404:
        return url + SERVED_PREFIX, r
    return url, get(url + path, timeout=timeout)


LOCAL = "local"
NOT_SERVING = ("this ComfyUI does not serve text encoders: turn on Settings > Remote Text Encoder > Serve text "
               "encoders to other PCs, or enter PC-B's address (http://<pc-b>:8188)")
_get_child = None
_get_error = None


class LocalUnavailable(Exception):
    def __init__(self, status, message, starting=False):
        super().__init__(message)
        self.status = status
        self.starting = starting


def is_local(url):
    return (url or "").strip().lower() == LOCAL


def set_local(get_child, get_error=None):
    global _get_child, _get_error
    _get_child, _get_error = get_child, get_error


def not_serving(default):
    error = _get_error() if _get_error else None
    return f"this ComfyUI is set to serve text encoders, but its encoder server {error}" if error else default


def local_base(get_child=None):
    get_child = get_child or _get_child
    child = get_child() if get_child else None
    if child is None:
        raise LocalUnavailable(503, not_serving(NOT_SERVING))
    if child.state == "failed":
        raise LocalUnavailable(503, "the encoder server on this ComfyUI is not running (it stopped and was not "
                                    "restarted; see the console or <user dir>/remote_te/server.log)")
    if child.port is None or not getattr(child, "ready", True):
        raise LocalUnavailable(503, "the encoder server on this ComfyUI is starting; try again in a moment",
                               starting=True)
    return f"http://127.0.0.1:{child.port}"
