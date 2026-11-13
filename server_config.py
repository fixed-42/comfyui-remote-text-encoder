import os
import tomllib

import gguf_support
from dataclasses import dataclass

MODEL_KEYS = {"name", "file", "type", "gpu"}


class ConfigError(Exception):
    pass


@dataclass(frozen=True)
class ModelSpec:
    name: str
    path: str
    type: str
    gpu: int | str


def load_config(path):
    try:
        with open(path, "rb") as f:
            data = tomllib.load(f)
    except (OSError, tomllib.TOMLDecodeError) as e:
        raise ConfigError(f"{path}: {e}") from None
    models_dir = data.get("models_dir", "")
    entries = data.get("model", [])
    if not entries:
        raise ConfigError(f"{path}: no [[model]] entries")

    specs, names, pairs = [], set(), set()
    for i, e in enumerate(entries, 1):
        where = f"{path}: model #{i}"
        for key in e:
            if key not in MODEL_KEYS:
                raise ConfigError(f"{where}: unknown key '{key}' (allowed: {', '.join(sorted(MODEL_KEYS))})")
        for key in ("name", "file", "type"):
            if not isinstance(e.get(key), str) or not e[key]:
                raise ConfigError(f"{where}: missing '{key}'")
        gpu = e.get("gpu", 0)
        if isinstance(gpu, bool) or not isinstance(gpu, (int, str)):
            raise ConfigError(f"{where}: gpu must be a number or text, got {gpu!r}")
        file = e["file"] if os.path.isabs(e["file"]) else os.path.join(models_dir, e["file"])
        if not os.path.isfile(file):
            raise ConfigError(f"{where} ({e['name']}): file not found: {file}")
        if e["name"] in names:
            raise ConfigError(f"{where}: duplicate name '{e['name']}'")
        pair = (os.path.abspath(file), e["type"].lower())
        if pair in pairs:
            raise ConfigError(f"{where} ({e['name']}): same file and type as an earlier model")
        names.add(e["name"])
        pairs.add(pair)
        specs.append(ModelSpec(e["name"], os.path.abspath(file), e["type"].lower(), gpu))
    return specs


def check_types(specs, valid_types):
    for s in specs:
        if s.type not in valid_types:
            raise ConfigError(f"{s.name}: unknown type '{s.type}' (ComfyUI types: {', '.join(sorted(valid_types))})")


def comfy_dir(arg, environ):
    path = arg or environ.get("COMFYUI_DIR")
    if not path:
        raise ConfigError("no ComfyUI folder: pass --comfy <ComfyUI folder> or set COMFYUI_DIR")
    if not os.path.isfile(os.path.join(path, "comfy", "sd.py")):
        raise ConfigError(f"ComfyUI not found at {path} (no comfy/sd.py); pass --comfy <ComfyUI folder>")
    return os.path.abspath(path)


def models_dir_path(arg):
    if not os.path.isdir(arg):
        raise ConfigError(f"--models-dir {arg}: not a folder")
    return os.path.abspath(arg)


def list_model_files(folder, gguf=False):
    found = []
    seen = {os.path.realpath(folder)}
    for root, dirs, files in os.walk(folder, followlinks=True):
        keep = []
        for d in sorted(dirs):
            real = os.path.realpath(os.path.join(root, d))
            if real not in seen:
                seen.add(real)
                keep.append(d)
        dirs[:] = keep
        for f in files:
            if f.endswith(".safetensors") or (gguf and gguf_support.offered(f)):
                found.append(os.path.relpath(os.path.join(root, f), folder).replace(os.sep, "/"))
    return sorted(found)


def list_models(folders, gguf=False):
    found = {}
    for folder in folders:
        for name in list_model_files(folder, gguf):
            found.setdefault(name, os.path.join(folder, *name.split("/")))
    return dict(sorted(found.items()))


def parse_reserve(text):
    t = text.strip().upper()
    try:
        if t.endswith("%"):
            kind, value = "pct", float(t[:-1])
            ok = 0 <= value <= 90
        else:
            scale, t = (1024, t[:-2]) if t.endswith("GB") else (1, t[:-2]) if t.endswith("MB") else (1, t)
            kind, value = "mb", round(float(t) * scale)
            ok = value >= 0
    except ValueError:
        ok = False
    if not ok:
        raise ConfigError(f"--reserve {text!r}: use a share of the card (10%, at most 90%) or an amount "
                          f"(3GB, 2048MB)")
    return kind, value


def reserve_mb(reserve, total_mb):
    kind, value = reserve
    return int(total_mb * value / 100) if kind == "pct" else value
