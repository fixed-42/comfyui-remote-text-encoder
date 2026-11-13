import json
import os
import struct
from types import SimpleNamespace

import gguf_support

TYPE_HINTS = {
    "QWEN3VL_4B": ["krea2"],
    "QWEN3VL_8B": ["qwen_image"],
    "QWEN3VL_32B": ["minimax"],
    "QWEN25_7B": ["qwen_image"],
    "QWEN3_4B": ["lumina2", "flux2"],
    "QWEN3_8B": ["flux2"],
    "QWEN35_9B": ["qwen_image"],
}
MAX_HEADER = 64 << 20


class Hints:
    def __init__(self, detect):
        self.detect = detect
        self._seen = {}

    def arch(self, path):
        try:
            st = os.stat(path)
        except OSError:
            return None
        key = (st.st_mtime_ns, st.st_size)
        if self._seen.get(path, (None,))[0] != key:
            self._seen[path] = (key, self._detect(path))
        return self._seen[path][1]

    def types(self, path):
        return list(TYPE_HINTS.get(self.arch(path), []))

    def _detect(self, path):
        if path.lower().endswith(gguf_support.EXT):
            return gguf_support.te_model(path)
        try:
            with open(path, "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                if n > MAX_HEADER:
                    return None
                header = json.loads(f.read(n))
            sd = {k: SimpleNamespace(shape=tuple(v["shape"])) for k, v in header.items() if k != "__metadata__"}
            found = self.detect(sd)
        except Exception:
            return None
        return found.name if found is not None else None
