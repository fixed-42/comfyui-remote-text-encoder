VERSION = "0.1.0"


def mismatch(remote, where):
    if remote == VERSION:
        return None
    theirs = f"version {remote}" if remote else "an older version (it reports none)"
    return (f"{where} runs {theirs} of the remote text encoder pack, this ComfyUI {VERSION}: install the same "
            f"version on both PCs")
