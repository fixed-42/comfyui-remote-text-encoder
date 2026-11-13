import os
import re
import subprocess
import time


def repo_commit(repo_dir):
    def git(*args):
        return subprocess.run(["git", "-C", repo_dir, *args], capture_output=True, text=True, timeout=10,
                              check=True).stdout.strip()
    try:
        commit = git("rev-parse", "--short", "HEAD")
        return commit + ("-dirty" if git("status", "--porcelain") else "")
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def pack_version(repo_dir):
    try:
        with open(os.path.join(repo_dir, "comfyui_remote_te", "version.py"), encoding="utf-8") as f:
            m = re.search(r'^VERSION\s*=\s*"([^"]+)"', f.read(), re.M)
    except OSError:
        return "unknown"
    return m.group(1) if m else "unknown"


def _value(v):
    if v is None:
        return "none"
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return str(v)
    text = str(v).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n").replace("\r", "\\r")
    return '"' + text + '"'


def log_line(kind, **fields):
    return " ".join([kind] + [f"{k}={_value(v)}" for k, v in fields.items()])


class Stats:
    def __init__(self):
        self.started = time.time()
        self.requests = 0
        self.last_error = None

    def uptime_s(self):
        return int(time.time() - self.started)

    def record(self, op, model, error):
        self.requests += 1
        if error:
            self.last_error = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "op": op, "model": model,
                               "error": error}
