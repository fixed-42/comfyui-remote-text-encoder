import json
import os


def _entries(obj):
    if not isinstance(obj, dict):
        raise ValueError('the keep list must be a JSON object {"keep": [...]}')
    keep = obj.get("keep")
    if not isinstance(keep, list):
        raise ValueError('"keep" must be a list of {"file": ..., "type": ...}')
    out = []
    for item in keep:
        if not (isinstance(item, dict) and isinstance(item.get("file"), str) and isinstance(item.get("type"), str)):
            raise ValueError(f'each kept model needs a file and type, got {item!r}')
        out.append((item["file"], item["type"]))
    return out


def parse(obj, names, valid_types):
    entries = _entries(obj)
    names = set(names)
    seen = set()
    for file, typ in entries:
        if file not in names:
            raise ValueError(f"unknown model '{file}'; not a file in this server's folders")
        if typ not in valid_types:
            raise ValueError(f"unknown type '{typ}' for '{file}'; one of: {sorted(valid_types)}")
        if (file, typ) in seen:
            raise ValueError(f"'{file}' as {typ} is in the list twice")
        seen.add((file, typ))
    return entries


def read(path):
    try:
        with open(path, encoding="utf-8") as f:
            return _entries(json.load(f)), None
    except FileNotFoundError:
        return [], None
    except (OSError, ValueError) as e:
        return [], f"cannot read the keep list {path}: {e}"


def save(path, entries):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"keep": [{"file": file, "type": typ} for file, typ in entries]}, f, indent=1)
        f.write("\n")
    os.replace(tmp, path)
