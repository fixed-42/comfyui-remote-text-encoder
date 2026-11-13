import atexit
import collections
import json
import os
import secrets
import socket
import subprocess
import sys
import threading
import time

PACK = os.path.dirname(os.path.realpath(__file__))
REPO = os.path.dirname(PACK)
SERVER = os.path.join(REPO, "server.py")
ENV_NAMES = ("REMOTE_TE_SERVE", "REMOTE_TE_GPU", "REMOTE_TE_LOG")


class ServerMissing(RuntimeError):
    pass


def check_server_files(repo=REPO):
    missing = [n for n in ("server.py", "gpus.py") if not os.path.isfile(os.path.join(repo, n))]
    if missing:
        raise ServerMissing(f"{', '.join(missing)} not found in {repo}: put the "
                            f"whole repository folder in custom_nodes (git clone), not only its comfyui_remote_te "
                            f"folder")


def unknown_env(env):
    return sorted(k for k in env if k.startswith("REMOTE_TE_") and k not in ENV_NAMES)
PREFIX = "[remote_te server]"


def _gpus():
    mod = sys.modules.get("remote_te_gpus")
    if mod is None:
        check_server_files()
        import importlib.util
        spec = importlib.util.spec_from_file_location("remote_te_gpus", os.path.join(REPO, "gpus.py"))
        mod = importlib.util.module_from_spec(spec)
        sys.modules["remote_te_gpus"] = mod
        spec.loader.exec_module(mod)
    return mod


def enabled(env):
    return env_serve(env) is True


def env_serve(env):
    v = env.get("REMOTE_TE_SERVE", "").strip().lower()
    if not v:
        return None
    return v in ("1", "true", "yes", "on")


def env_gpu(env):
    return env.get("REMOTE_TE_GPU", "").strip() or None


AUTO = "auto"
DEFAULT_SETTINGS = {"serve": False, "gpu": AUTO}


def read_settings(path):
    try:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
    except (OSError, ValueError):
        return dict(DEFAULT_SETTINGS)
    out = dict(DEFAULT_SETTINGS)
    if isinstance(d, dict):
        if isinstance(d.get("serve"), bool):
            out["serve"] = d["serve"]
        if isinstance(d.get("gpu"), str) and d["gpu"].strip():
            out["gpu"] = d["gpu"].strip()
    return out


def save_settings(path, settings):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=1)
    os.replace(tmp, path)


CUDA_PROBE = ("import torch\n"
              "for i in range(torch.cuda.device_count()):\n"
              "    print('GPU-' + str(torch.cuda.get_device_properties(i).uuid))")


def cuda_order(python=sys.executable, env=None):
    env = dict(os.environ if env is None else env)
    for k in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES"):
        env.pop(k, None)
    out = subprocess.run([python, "-c", CUDA_PROBE], capture_output=True, text=True, env=env, timeout=120)
    if out.returncode != 0:
        raise RuntimeError(f"cannot list CUDA devices: {out.stderr.strip()[-300:]}")
    return [line.strip() for line in out.stdout.splitlines() if line.strip().startswith("GPU-")]


def choose_gpu(gpus, render_uuid=None, wanted=None, cuda_order=cuda_order):
    g = _gpus()
    if wanted:
        w = str(wanted).strip()
        if w.isdigit() and int(w) < max(len(gpus), 1) + 8:
            order = cuda_order()
            names = {x.uuid.lower(): x.name for x in gpus}
            if int(w) >= len(order):
                listing = ", ".join(f"{i} = {names.get(u.lower(), u)}" for i, u in enumerate(order))
                raise g.GpuError(f"REMOTE_TE_GPU={w}: no CUDA device {w}; this PC has {listing or 'none'}")
            uuid = order[int(w)].lower()
            for x in gpus:
                if x.uuid.lower() == uuid or uuid.startswith(x.uuid.lower()):
                    return x
            raise g.GpuError(f"REMOTE_TE_GPU={w}: CUDA device {w} ({order[int(w)]}) is not in nvidia-smi's list")
        return g.resolve(w, gpus)
    if not gpus:
        raise g.GpuError("no NVIDIA GPU found (nvidia-smi lists none)")
    others = [g for g in gpus if g.uuid.lower() != (render_uuid or "").lower()]
    return others[0] if others else gpus[0]


def render_gpu_uuid():
    try:
        import torch
        import comfy.model_management as mm
        dev = mm.get_torch_device()
        if dev.type != "cuda":
            return None
        return "GPU-" + str(torch.cuda.get_device_properties(dev).uuid)
    except Exception:
        return None


def ensure_secret(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        with open(path) as f:
            if f.read().strip():
                return path
    except OSError:
        pass
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(secrets.token_hex(16) + "\n")
    return path


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def model_folders(paths):
    out = []
    for p in paths:
        if os.path.isdir(p) and p not in out:
            out.append(p)
    return out


def comfyui_url(listen, port, tls=False):
    if tls:
        return None
    host = ((listen or "127.0.0.1").split(",")[0]).strip()
    if host in ("", "0.0.0.0", "::"):
        host = "127.0.0.1"
    return f"http://[{host}]:{port}" if ":" in host else f"http://{host}:{port}"


def child_argv(python, comfy_dir, models_dirs, gpu_uuid, port, secret_file, keep_file, server=SERVER,
               custom_nodes=(), capabilities_file=None, render_gpu=None, comfyui=None, types_file=None):
    argv = [python, server, "--comfy", comfy_dir]
    for d in models_dirs:
        argv += ["--models-dir", d]
    for d in custom_nodes:
        argv += ["--custom-nodes", d]
    if capabilities_file:
        argv += ["--capabilities-file", capabilities_file]
    if render_gpu:
        argv += ["--render-gpu", render_gpu]
    if types_file:
        argv += ["--types-file", types_file]
    if comfyui:
        argv += ["--comfyui-url", comfyui]
    return argv + ["--keep-file", keep_file, "--gpu", gpu_uuid, "--host", "127.0.0.1", "--port", str(port),
                   "--secret-file", secret_file,
                   "--scan-abilities"]


def _die_with_parent():
    import ctypes
    import signal
    ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGTERM)


def _job_structs():
    import ctypes
    dword, size_t = ctypes.c_uint32, ctypes.c_size_t

    class Basic(ctypes.Structure):
        _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                    ("LimitFlags", dword), ("MinimumWorkingSetSize", size_t), ("MaximumWorkingSetSize", size_t),
                    ("ActiveProcessLimit", dword), ("Affinity", size_t), ("PriorityClass", dword),
                    ("SchedulingClass", dword)]

    class Io(ctypes.Structure):
        _fields_ = [(n, ctypes.c_uint64) for n in ("ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
                                                    "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

    class Extended(ctypes.Structure):
        _fields_ = [("BasicLimitInformation", Basic), ("IoInfo", Io), ("ProcessMemoryLimit", size_t),
                    ("JobMemoryLimit", size_t), ("PeakProcessMemoryUsed", size_t), ("PeakJobMemoryUsed", size_t)]

    return Extended


JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9
_job = None


def _windows_job():
    global _job
    if _job is None:
        import ctypes
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.CreateJobObjectW.restype = ctypes.c_void_p
        k32.CreateJobObjectW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p]
        k32.SetInformationJobObject.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32]
        job = k32.CreateJobObjectW(None, None)
        if not job:
            raise ctypes.WinError(ctypes.get_last_error())
        info = _job_structs()()
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not k32.SetInformationJobObject(job, JOB_OBJECT_EXTENDED_LIMIT_INFORMATION, ctypes.byref(info),
                                           ctypes.sizeof(info)):
            raise ctypes.WinError(ctypes.get_last_error())
        _job = (k32, job)
    return _job


def contain(proc, platform=sys.platform):
    if not platform.startswith("win"):
        return None
    try:
        import ctypes
        k32, job = _windows_job()
        k32.AssignProcessToJobObject.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        if not k32.AssignProcessToJobObject(job, int(proc._handle)):
            raise ctypes.WinError(ctypes.get_last_error())
    except Exception as e:
        return f"{type(e).__name__}: {e}"
    return None


class ChildServer:

    def __init__(self, make_argv, log_path, out=print, max_deaths=3, window_s=60.0, restart_delay_s=1.0,
                 env=None, cwd=REPO):
        self.make_argv = make_argv
        self.log_path, self.out = log_path, out
        self.max_deaths, self.window_s, self.restart_delay_s = max_deaths, window_s, restart_delay_s
        self.env = dict(os.environ if env is None else env, PYTHONUNBUFFERED="1")
        self.cwd = cwd
        self.port = None
        self.proc = None
        self.ready = False
        self.state = "stopped"
        self.tail = collections.deque(maxlen=30)
        self.deaths = []
        self._stopping = threading.Event()
        self._thread = None

    def start(self):
        self._stopping.clear()
        self.state = "starting"
        self._thread = threading.Thread(target=self._supervise, name="remote_te server", daemon=True)
        self._thread.start()
        if not getattr(self, "_atexit", False):
            atexit.register(self.stop)
            self._atexit = True

    def _spawn(self):
        self.ready = False
        self.port = free_port()
        argv = self.make_argv(self.port)
        kw = {"preexec_fn": _die_with_parent} if sys.platform.startswith("linux") else {}
        self.proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                     cwd=self.cwd, env=self.env, **kw)
        self._say(f"started (pid {self.proc.pid}, port {self.port}): {' '.join(argv)}")
        problem = contain(self.proc)
        if problem:
            self._say(f"WARNING: could not tie the encoder server to ComfyUI ({problem}); if ComfyUI is killed, end "
                      f"pid {self.proc.pid} by hand")

    def _supervise(self):
        with open(self.log_path, "w", encoding="utf-8", errors="replace") as log:
            self._log = log
            while not self._stopping.is_set():
                try:
                    self._spawn()
                except OSError as e:
                    self._say(f"cannot start: {e}")
                    self.state = "failed"
                    return
                for raw in self.proc.stdout:
                    line = raw.decode("utf-8", "replace").rstrip("\n")
                    if line.startswith("READY "):
                        self.ready = True
                    self._say(line)
                self.ready = False
                code = self.proc.wait()
                if self._stopping.is_set():
                    break
                now = time.monotonic()
                self.deaths = [t for t in self.deaths if now - t < self.window_s] + [now]
                if len(self.deaths) >= self.max_deaths:
                    self.state = "failed"
                    self._say(f"exited with code {code}; {len(self.deaths)} exits within {self.window_s:.0f} s, "
                              f"not restarting. After fixing the cause, turn serving off and on in ComfyUI's Settings "
                              f"(Remote Text Encoder) or restart ComfyUI (log: {self.log_path})")
                    return
                self._say(f"exited with code {code}; restarting")
                if self._stopping.wait(self.restart_delay_s):
                    break
            self.state = "stopped"

    def _say(self, line):
        self.tail.append(line)
        self.out(f"{PREFIX} {line}")
        log = getattr(self, "_log", None)
        if log and not log.closed:
            log.write(line + "\n")
            log.flush()

    def stop(self, grace_s=10):
        self._stopping.set()
        p = self.proc
        if p and p.poll() is None:
            p.terminate()
            try:
                p.wait(grace_s)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait()
        if self._thread:
            self._thread.join(5)
        self.state = "stopped"


class Serving:

    def __init__(self, settings_path, env, start_child, out=print):
        self.path, self.env, self.start_child, self.out = settings_path, env, start_child, out
        self.child = None
        self.child_gpu = None
        self.error = None
        self._lock = threading.Lock()

    def wanted(self):
        s = read_settings(self.path)
        serve = env_serve(self.env)
        gpu = env_gpu(self.env) or (None if s["gpu"].lower() == AUTO else s["gpu"])
        return (s["serve"] if serve is None else serve), gpu

    def apply(self):
        with self._lock:
            serve, gpu = self.wanted()
            c = self.child
            if c is not None and (not serve or gpu != self.child_gpu or c.state in ("failed", "stopped")):
                c.stop()
                self.out(f"{PREFIX} stopped" + ("" if not serve else " (restarting with the new settings)"))
                self.child = None
            if serve and self.child is None:
                try:
                    self.child, self.child_gpu, self.error = self.start_child(gpu), gpu, None
                except Exception as e:
                    self.error = f"cannot start: {e}"
                    self.out(f"{PREFIX} {self.error}")
            elif not serve:
                self.error = None

    def update(self, body):
        if not isinstance(body, dict):
            return 400, {"error": "body must be a JSON object"}
        s = read_settings(self.path)
        if "serve" in body:
            if not isinstance(body["serve"], bool):
                return 400, {"error": "serve must be true or false"}
            if env_serve(self.env) is not None:
                return 409, {**self.describe(), "error": f"REMOTE_TE_SERVE={self.env.get('REMOTE_TE_SERVE')} is set "
                                                         f"where ComfyUI was started; it decides instead of this setting"}
            s["serve"] = body["serve"]
        if "gpu" in body:
            if not isinstance(body["gpu"], str) or not body["gpu"].strip():
                return 400, {"error": "gpu must be 'auto' or a CUDA device number"}
            if env_gpu(self.env):
                return 409, {**self.describe(), "error": f"REMOTE_TE_GPU={env_gpu(self.env)} is set where ComfyUI was "
                                                         f"started; it decides instead of this setting"}
            s["gpu"] = body["gpu"].strip()
        save_settings(self.path, s)
        self.apply()
        return 200, self.describe()

    def describe(self):
        serve, gpu = self.wanted()
        c = self.child
        overrides = {k: self.env[k] for k in ("REMOTE_TE_SERVE", "REMOTE_TE_GPU") if self.env.get(k, "").strip()}
        return {"serve": serve, "gpu": gpu or AUTO, "saved": read_settings(self.path), "overrides": overrides,
                "state": c.state if c is not None else "off", "port": c.port if c is not None else None,
                "gpu_name": getattr(c, "gpu_name", None), "error": self.error}


RENDERS_HERE = "ComfyUI renders here"


def gpu_options(env=None, python=sys.executable, render_uuid=None, list_gpus=None, order=None):
    out = [{"value": AUTO, "text": "auto"}]
    try:
        cards = {g.uuid.lower(): g for g in (list_gpus or _gpus().list_gpus)()}
        for i, uuid in enumerate(order() if order else cuda_order(python, env)):
            g = next((g for u, g in cards.items() if uuid.lower() == u or uuid.lower().startswith(u)), None)
            name, full = (g.name, g.uuid) if g else (uuid, uuid)
            render = bool(render_uuid) and full.lower() == render_uuid.lower()
            out.append({"value": str(i), "text": f"{i}: {name}" + (f" ({RENDERS_HERE})" if render else ""),
                        "uuid": full, "render": render})
    except Exception:
        pass
    return out


def starter_in_comfyui(env, out=print):
    import comfy.sd
    import folder_paths
    comfy_dir = os.path.dirname(os.path.dirname(os.path.abspath(comfy.sd.__file__)))
    own = os.path.join(folder_paths.get_user_directory(), "remote_te")
    from comfy.cli_args import args
    own_url = comfyui_url(args.listen, args.port, tls=bool(args.tls_keyfile and args.tls_certfile))

    def start_child(wanted):
        check_server_files()
        folders = model_folders(folder_paths.get_folder_paths("text_encoders"))
        if not folders:
            raise RuntimeError("no text_encoders folder found")
        render = render_gpu_uuid()
        gpu = choose_gpu(_gpus().list_gpus(), render, wanted,
                         cuda_order=lambda: cuda_order(sys.executable, env))
        secret_file = ensure_secret(os.path.join(own, "secret"))
        keep = os.path.join(own, "keep.json")
        custom_nodes = model_folders(folder_paths.get_folder_paths("custom_nodes"))
        child = ChildServer(lambda port: child_argv(sys.executable, comfy_dir, folders, gpu.uuid, port, secret_file,
                                                    keep, custom_nodes=custom_nodes,
                                                    capabilities_file=os.path.join(own, "capabilities.json"),
                                                    render_gpu=render, comfyui=own_url,
                                                    types_file=os.path.join(own, "types.json")),
                            os.path.join(own, "server.log"), out=out, env=env)
        child.gpu_name = gpu.name
        out(f"{PREFIX} serving encoders from {', '.join(folders)} on {gpu.name} ({gpu.uuid}); kept models: {keep}")
        child.start()
        return child

    return start_child


def serving_in_comfyui(env=None, out=print):
    import folder_paths
    env = os.environ if env is None else env
    for k in unknown_env(env):
        out(f"{PREFIX} WARNING: {k} is set but not used; this pack reads {', '.join(ENV_NAMES)}")
    path = os.path.join(folder_paths.get_user_directory(), "remote_te", "serve.json")
    serving = Serving(path, env, starter_in_comfyui(env, out), out=out)
    serving.apply()
    return serving
