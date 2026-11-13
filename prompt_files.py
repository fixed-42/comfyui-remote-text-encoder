import os
import re

ROOT = os.environ.get("ENC_PROMPTS_DIR") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts")
VALUE_SLOTS = {
    "locate.txt": ("what",),
}
REQUIRED = ("locate.txt",)
_SLOT = re.compile(r"\{([a-z_]+)\}")


class PromptFileError(Exception):
    pass


def read(name, root=None):
    path = os.path.join(root or ROOT, name)
    try:
        with open(path, encoding="utf-8-sig") as f:
            text = f.read()
    except UnicodeDecodeError:
        raise PromptFileError(f"instruction file {path} is not UTF-8 text; save it as UTF-8") from None
    except OSError as e:
        raise PromptFileError(f"instruction file {path}: {e.strerror}") from None
    text = text.replace("\r\n", "\n").strip()
    if not text:
        raise PromptFileError(f"instruction file {path} is empty")
    return text


def assemble(name, values=None, root=None):
    values = values or {}
    folder = os.path.dirname(name)

    def fill(match):
        slot = match.group(1)
        if slot in values:
            return values[slot]
        try:
            return read(os.path.join(folder, "shared", slot + ".txt"), root)
        except PromptFileError as e:
            raise PromptFileError(f"instruction file {os.path.join(root or ROOT, name)} uses {{{slot}}}, "
                                  f"but {e}") from None

    return _SLOT.sub(fill, read(name, root))


def check(root=None):
    problems = []
    for name in REQUIRED:
        slots = VALUE_SLOTS.get(name, ())
        try:
            assemble(name, {slot: "x" for slot in slots}, root)
            text = read(name, root)
        except PromptFileError as e:
            problems.append(str(e))
            continue
        for slot in slots:
            if "{" + slot + "}" not in text:
                problems.append(f"instruction file {os.path.join(root or ROOT, name)} must contain {{{slot}}}: "
                                f"the server puts the request's {'words' if slot == 'what' else slot} there")
    return problems
