import json
import os

TEXT, VISION = "text", "vision"
NO_TEXT_QWEN25 = "ComfyUI's Qwen2.5-VL encoder has no text generation (expected; not checked yet)"
NO_VISION_TEXT_ONLY = "a text-only encoder (Qwen3) (expected; not checked yet)"
EXPECTED = {
    "QWEN25_7B": {TEXT: NO_TEXT_QWEN25},
    "QWEN3_4B": {VISION: NO_VISION_TEXT_ONLY},
    "QWEN3_8B": {VISION: NO_VISION_TEXT_ONLY},
}
NO_VISION_QWEN3VL_GGUF = "a Qwen3-VL GGUF loads without its vision part (ComfyUI-GGUF)"
NO_VISION_NO_MMPROJ = "no mmproj file next to this GGUF: ComfyUI-GGUF loads it without its vision part"


def expected(te_model, gguf_arch=None, has_mmproj=False):
    out = {what: why for what, why in EXPECTED.get(te_model, {}).items()}
    if gguf_arch == "qwen3vl":
        out[VISION] = NO_VISION_QWEN3VL_GGUF
    elif gguf_arch == "qwen2vl" and not has_mmproj:
        out[VISION] = NO_VISION_NO_MMPROJ
    return {what: {"ok": False, "why": why, "checked": False} for what, why in out.items()}


def _stamp(path):
    st = os.stat(path)
    return [st.st_mtime_ns, st.st_size]


class Store:

    def __init__(self, path, comfyui_version):
        self.path, self.comfyui = path, comfyui_version
        self.entries = {}
        if path:
            try:
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict) and data.get("comfyui") == comfyui_version \
                        and isinstance(data.get("entries"), dict):
                    self.entries = data["entries"]
            except (OSError, ValueError):
                pass

    def get(self, file_path, typ):
        e = self.entries.get(f"{file_path}\n{typ}")
        try:
            if not isinstance(e, dict) or e.get("stamp") != _stamp(file_path):
                return {}
        except OSError:
            return {}
        return {w: e[w] for w in (TEXT, VISION) if isinstance(e.get(w), dict)}

    def for_file(self, file_path):
        out = {}
        for key in list(self.entries):
            path, _, typ = key.rpartition("\n")
            if path == file_path:
                got = self.get(path, typ)
                if got:
                    out[typ] = {w: dict(r, checked=True) for w, r in got.items()}
        return out

    def record(self, file_path, typ, what, error):
        key = f"{file_path}\n{typ}"
        try:
            stamp = _stamp(file_path)
        except OSError:
            return
        e = self.entries.get(key)
        if not isinstance(e, dict) or e.get("stamp") != stamp:
            e = self.entries[key] = {"stamp": stamp}
        e[what] = {"ok": error is None, "why": error or ""}
        self._save()

    def _save(self):
        if not self.path:
            return
        tmp = self.path + ".tmp"
        try:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump({"comfyui": self.comfyui, "entries": self.entries}, f, indent=1)
            os.replace(tmp, self.path)
        except OSError as e:
            print(f"WARNING: could not save {self.path}: {e}", flush=True)
