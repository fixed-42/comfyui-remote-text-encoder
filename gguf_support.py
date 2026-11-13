import importlib.util
import os
import struct
import sys

EXT = ".gguf"
PACK_MODULE = "remote_te_comfyui_gguf"

TE_MODELS = {
    ("qwen2vl", 3584): "QWEN25_7B",
    ("qwen3", 2560): "QWEN3_4B",
    ("qwen3", 4096): "QWEN3_8B",
}
QWEN3VL_WARNING = ("is a Qwen3-VL GGUF: ComfyUI-GGUF loads it without its vision part, so ComfyUI builds a plain Qwen3 "
                   "text model and the conditioning is not what Krea2 / Qwen-Image 2.1 / MiniMax H3 expect; use the "
                   "safetensors file")
DEQUANTIZED_EMBEDDING = {"llama", "qwen2vl", "qwen3", "qwen3vl", "gemma3"}
GGML_BLOCKS = {0: (4, 1), 1: (2, 1), 2: (18, 32), 3: (20, 32), 6: (22, 32), 7: (24, 32), 8: (34, 32), 10: (84, 256),
               11: (110, 256), 12: (144, 256), 13: (176, 256), 14: (210, 256), 30: (2, 1)}
_SCALARS = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
_STRING, _ARRAY = 8, 9
MAX_ITEMS = 1 << 24


def offered(name):
    return name.lower().endswith(EXT) and "mmproj" not in os.path.basename(name).lower()


class _Reader:
    def __init__(self, f):
        self.f = f

    def take(self, fmt):
        size = struct.calcsize(fmt)
        data = self.f.read(size)
        if len(data) != size:
            raise ValueError("truncated GGUF header")
        return struct.unpack(fmt, data)[0]

    def string(self):
        n = self.take("<Q")
        if n > MAX_ITEMS:
            raise ValueError("bad GGUF string length")
        return self.f.read(n).decode("utf-8", "replace")

    def value(self, typ, keep=True):
        if typ in _SCALARS:
            return self.take(_SCALARS[typ])
        if typ == _STRING:
            return self.string()
        if typ == _ARRAY:
            item, n = self.take("<I"), self.take("<Q")
            if n > MAX_ITEMS:
                raise ValueError("bad GGUF array length")
            if item in _SCALARS and not keep:
                self.f.seek(n * struct.calcsize(_SCALARS[item]), os.SEEK_CUR)
                return None
            values = [self.value(item, keep) for _ in range(n)]
            return values if keep else None
        raise ValueError(f"unknown GGUF value type {typ}")


def read_header(path):
    with open(path, "rb") as f:
        r = _Reader(f)
        if f.read(4) != b"GGUF":
            raise ValueError("not a GGUF file")
        version = r.take("<I")
        if version < 2:
            raise ValueError(f"GGUF version {version} is not supported")
        n_tensors, n_kv = r.take("<Q"), r.take("<Q")
        if n_tensors > MAX_ITEMS or n_kv > MAX_ITEMS:
            raise ValueError("bad GGUF header counts")
        meta = {}
        for _ in range(n_kv):
            key, typ = r.string(), r.take("<I")
            wanted = key == "general.architecture" or key.endswith(".embedding_length")
            value = r.value(typ, keep=wanted)
            if wanted:
                meta[key] = value
        embedding = None
        for _ in range(n_tensors):
            name, dims = r.string(), r.take("<I")
            if dims > 8:
                raise ValueError("bad GGUF tensor dimensions")
            shape = [r.take("<Q") for _ in range(dims)]
            typ, _offset = r.take("<I"), r.take("<Q")
            if name == "token_embd.weight" and dims == 2:
                embedding = (shape[1], shape[0], typ)
                break
    arch = meta.get("general.architecture")
    return {"arch": arch, "width": meta.get(f"{arch}.embedding_length"), "embedding": embedding}


_headers = {}


def header(path):
    try:
        st = os.stat(path)
    except OSError:
        return None
    key = (st.st_mtime_ns, st.st_size)
    if _headers.get(path, (None,))[0] != key:
        try:
            found = read_header(path)
        except (OSError, ValueError, UnicodeDecodeError, struct.error):
            found = None
        _headers[path] = (key, found)
    return _headers[path][1]


def arch(path):
    h = header(path)
    return h["arch"] if h else None


def te_model(path):
    h = header(path)
    return TE_MODELS.get((h["arch"], h["width"])) if h else None


def extra_mb(path):
    h = header(path)
    if h is None or h["arch"] not in DEQUANTIZED_EMBEDDING or h["embedding"] is None:
        return 0
    rows, width, typ = h["embedding"]
    if rows < 64 * 1024:
        return 0
    block_bytes, block_values = GGML_BLOCKS.get(typ, (0, 1))
    stored = rows * width * block_bytes // block_values
    return max(0, rows * width * 2 - stored) >> 20


def find_pack(custom_nodes_dirs):
    for base in custom_nodes_dirs:
        try:
            entries = sorted(os.listdir(base))
        except OSError:
            continue
        for entry in entries:
            if entry.startswith(".") or entry.endswith(".disabled"):
                continue
            folder = os.path.join(base, entry)
            nodes_py = os.path.join(folder, "nodes.py")
            if not (os.path.isfile(nodes_py) and os.path.isfile(os.path.join(folder, "loader.py"))
                    and os.path.isfile(os.path.join(folder, "__init__.py"))):
                continue
            try:
                with open(nodes_py, encoding="utf-8", errors="replace") as f:
                    if "class CLIPLoaderGGUF" in f.read():
                        return folder
            except OSError:
                continue
    return None


class GgufLoader:

    def __init__(self, folder):
        self.folder = folder
        spec = importlib.util.spec_from_file_location(PACK_MODULE, os.path.join(folder, "__init__.py"),
                                                      submodule_search_locations=[folder])
        module = importlib.util.module_from_spec(spec)
        sys.modules[PACK_MODULE] = module
        try:
            spec.loader.exec_module(module)
            pack_nodes = sys.modules[f"{PACK_MODULE}.nodes"]
            pack_loader = sys.modules[f"{PACK_MODULE}.loader"]
            self._ops = pack_nodes.GGMLOps
            self._patcher = pack_nodes.GGUFModelPatcher
            self._read = pack_loader.gguf_clip_loader
            self._strip = pack_loader.strip_quant_suffix
        except BaseException:
            for name in [n for n in sys.modules if n == PACK_MODULE or n.startswith(PACK_MODULE + ".")]:
                del sys.modules[name]
            raise

    def has_mmproj(self, path):
        stem = self._strip(os.path.splitext(os.path.basename(path))[0].lower())
        try:
            names = os.listdir(os.path.dirname(path))
        except OSError:
            return False
        return any(n.lower().endswith(EXT) and "mmproj" in n.lower() and stem in n.lower() for n in names)

    def load(self, path, clip_type, device):
        import comfy.sd

        sd = self._read(path)
        clip = comfy.sd.load_text_encoder_state_dicts(
            [sd], clip_type=clip_type,
            model_options={"custom_operations": self._ops, "load_device": device, "offload_device": device,
                           "initial_device": device})
        clip.patcher = self._patcher.clone(clip.patcher)
        return clip
