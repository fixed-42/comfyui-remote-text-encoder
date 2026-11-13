import hmac
import os
import subprocess
import sys

SERVER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "server.py")


class SecretError(Exception):
    pass


def load_secret(path):
    try:
        with open(path) as f:
            secret = f.read().strip()
    except OSError:
        secret = ""
    if not secret:
        raise SecretError(f"no secret in {path}; /restart and /stop need one. Create it with:\n"
                          f"  openssl rand -hex 16 > {path} && chmod 600 {path}\n"
                          f"(the same file on the machine that calls /restart)")
    return secret


def authorized(header_value, secret):
    return bool(header_value) and hmac.compare_digest(header_value.encode(), secret.encode())


def preflight(argv):
    try:
        r = subprocess.run([sys.executable, SERVER, "--check", *argv], capture_output=True, text=True,
                           timeout=120)
    except subprocess.TimeoutExpired:
        return "check timed out after 120 s"
    return None if r.returncode == 0 else (r.stdout + r.stderr).strip()[-2000:]
