import json
import os
import struct
from types import SimpleNamespace

import gguf_support

TEXT_GEN = "text_gen"
LOAD_AS = {TEXT_GEN: "qwen_image"}
PACK_TYPES = set(LOAD_AS)


def comfy_type(typ):
    return LOAD_AS.get(typ, typ)


def legacy_type(arch, typ):
    return TEXT_GEN if typ == "qwen_image" and (arch or "").startswith("QWEN35_") else typ


TYPE_HINTS = {
    "QWEN3VL_4B": ["krea2", "flux3", "fibo2", "mage", "flux2"],
    "QWEN3VL_8B": ["qwen_image", "ideogram4", "boogu", "joyimage", "flux2"],
    "QWEN3VL_32B": ["minimax"],
    "QWEN25_7B": ["qwen_image", "hunyuan_image", "longcat_image"],
    "QWEN25_3B": ["omnigen2"],
    "QWEN3_4B": ["lumina2", "flux2"],
    "QWEN3_8B": ["flux2", "ideogram4"],
    "QWEN3_2B": ["ovis"],
    "QWEN35_08B": [TEXT_GEN],
    "QWEN35_2B": [TEXT_GEN],
    "QWEN35_4B": [TEXT_GEN],
    "QWEN35_9B": [TEXT_GEN, "moss_image"],
    "QWEN35_27B": [TEXT_GEN],
    "MISTRAL3_24B": ["flux2"],
    "MISTRAL3_24B_PRUNED_FLUX2": ["flux2"],
    "GPT_OSS_20B": ["lens"],
    "GEMMA_2_2B": ["lumina2", "pixeldit"],
    "LLAMA3_8": ["hidream"],
    "T5_XXL": ["wan", "ltxv", "chroma", "pixart", "sd3", "mochi", "cogvideox", "hidream"],
    "T5_XXL_OLD": ["cosmos"],
    "T5_BASE": ["ace", "stable_audio"],
    "T5_GEMMA": ["stable_audio"],
    "CLIP_L": ["stable_diffusion", "sd3", "hidream"],
    "CLIP_G": ["stable_diffusion", "stable_cascade", "sd3", "hidream"],
    "CLIP_H": ["stable_diffusion"],
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
