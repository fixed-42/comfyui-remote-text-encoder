import json
import os


def parse(obj, names, valid_types):
    if not isinstance(obj, dict):
        raise ValueError('"types" must be a JSON object {"<file>": "<type>"}')
    names = set(names)
    for file, typ in obj.items():
        if file not in names:
            raise ValueError(f"unknown model '{file}'; not a file in this server's folders")
        if not isinstance(typ, str) or typ not in valid_types:
            raise ValueError(f"unknown type {typ!r} for '{file}'; one of: {sorted(valid_types)}")
    return dict(obj)


def read(path):
    try:
        with open(path, encoding="utf-8") as f:
            obj = json.load(f)
    except FileNotFoundError:
        return {}, None
    except (OSError, ValueError) as e:
        return {}, f"cannot read the picked types {path}: {e}"
    types = obj.get("types") if isinstance(obj, dict) else None
    if not isinstance(types, dict):
        return {}, f'cannot read the picked types {path}: expected {{"types": {{...}}}}'
    return {f: t for f, t in types.items() if isinstance(f, str) and isinstance(t, str)}, None


def save(path, types):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"types": dict(sorted(types.items()))}, f, indent=1)
        f.write("\n")
    os.replace(tmp, path)
