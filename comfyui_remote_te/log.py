import logging
import os
import sys

LEVELS = {"debug": logging.DEBUG, "info": logging.INFO, "warning": logging.WARNING, "error": logging.ERROR}

log = logging.getLogger("remote_te")
PREFIX = "[remote_te]"


def debug(msg):
    log.debug(f"{PREFIX} {msg}")


def info(msg):
    log.info(f"{PREFIX} {msg}")


def warning(msg):
    log.warning(f"{PREFIX} {msg}")


def error(msg):
    log.error(f"{PREFIX} {msg}")


class _Handler(logging.StreamHandler):
    pass


def configure(env=None, stream=None):
    env = os.environ if env is None else env
    name = (env.get("REMOTE_TE_LOG") or "").strip().lower()
    for h in [h for h in log.handlers if isinstance(h, _Handler)]:
        log.removeHandler(h)
    if name not in LEVELS:
        log.setLevel(logging.NOTSET)
        log.propagate = True
        return None
    handler = _Handler(stream or sys.stderr)
    handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    log.addHandler(handler)
    log.setLevel(LEVELS[name])
    log.propagate = False
    return name


class Once:

    def __init__(self):
        self._last = {}

    def error(self, key, message):
        if self._last.get(key) != message:
            self._last[key] = message
            error(message)

    def ok(self, key, message=None):
        if self._last.pop(key, None) is not None and message:
            info(message)
