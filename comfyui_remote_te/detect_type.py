CONFIG_TYPES = {
    "QwenImage": "qwen_image",
    "QwenImage21": "qwen_image",
    "Flux": "flux",
    "Flux2": "flux2",
    "Flux3": "flux3",
    "Lumina2": "lumina2",
    "ZImage": "lumina2",
    "Krea2": "krea2",
    "MiniMaxH3": "minimax",
    "HunyuanVideo": "hunyuan_video",
    "HunyuanVideo15": "hunyuan_video_15",
    "HunyuanVideo15_SR_Distilled": "hunyuan_video_15",
    "HunyuanImage21": "hunyuan_image",
    "LongCatImage": "longcat_image",
    "Ideogram4": "ideogram4",
    "Boogu": "boogu",
    "JoyImage": "joyimage",
    "MageFlow": "mage",
    "Fibo2": "fibo2",
    "MOSSImage": "moss_image",
    "Omnigen2": "omnigen2",
    "Kandinsky5": "kandinsky5",
    "Kandinsky5Image": "kandinsky5_image",
    "HiDream": "hidream",
    "Chroma": "chroma",
    "SD3": "sd3",
    "StableAudio": "stable_audio",
    "LTXV": "ltxv",
    "GenmoMochi": "mochi",
    "PixArtAlpha": "pixart",
    "ACEStep": "ace",
    "WAN21_T2V": "wan",
    "CosmosT2V": "cosmos",
    "Lens": "lens",
    "PixelDiTT2I": "pixeldit",
    "YuE2": "yue2",
    "CogVideoX_T2V": "cogvideox",
}


def config_of(model):
    return getattr(getattr(model, "model", None), "model_config", None)


def clip_type_for(model, known):
    config = config_of(model)
    if config is None:
        return None, type(model).__name__
    cls = type(config)
    for c in cls.__mro__:
        t = CONFIG_TYPES.get(c.__name__)
        if t is not None:
            return (t if t in known else None), cls.__name__
    return None, cls.__name__


CONFIG_ARCHS = {
    "QwenImage": ["QWEN25_7B"],
    "QwenImage21": ["QWEN3VL_8B"],
    "Krea2": ["QWEN3VL_4B"],
    "Flux3": ["QWEN3VL_4B"],
    "Fibo2": ["QWEN3VL_4B"],
    "MageFlow": ["QWEN3VL_4B"],
    "MiniMaxH3": ["QWEN3VL_32B"],
    "ZImage": ["QWEN3_4B"],
    "Lumina2": ["GEMMA_2_2B"],
    "Ideogram4": ["QWEN3VL_8B", "QWEN3_8B"],
    "Boogu": ["QWEN3VL_8B"],
    "JoyImage": ["QWEN3VL_8B"],
    "HunyuanImage21": ["QWEN25_7B"],
    "LongCatImage": ["QWEN25_7B"],
    "Omnigen2": ["QWEN25_3B"],
    "MOSSImage": ["QWEN35_9B"],
    "Lens": ["GPT_OSS_20B"],
}
FLUX2_ARCHS = {
    7680: ["QWEN3_4B", "QWEN3VL_4B"],
    12288: ["QWEN3_8B", "QWEN3VL_8B"],
    15360: ["MISTRAL3_24B", "MISTRAL3_24B_PRUNED_FLUX2"],
}


def expected_archs(model):
    config = config_of(model)
    if config is None:
        return None
    for c in type(config).__mro__:
        if c.__name__ in CONFIG_ARCHS:
            return list(CONFIG_ARCHS[c.__name__])
        if c.__name__ == "Flux2":
            dim = (getattr(config, "unet_config", None) or {}).get("context_in_dim")
            return list(FLUX2_ARCHS[dim]) if dim in FLUX2_ARCHS else None
    return None


def suits(entry, clip_type, archs):
    if archs:
        return entry.get("arch") in archs
    return clip_type in (entry.get("types") or [])


def pick_file(entries, current, clip_type, archs, room_mb=None):
    fits = [m for m in entries if suits(m, clip_type, archs)]
    kept_mb = sum(m.get("size_mb") or 0 for m in entries if m.get("kept_as"))

    def in_room(m):
        if room_mb is None or m.get("kept_as") or m.get("loaded_as"):
            return True
        return (m.get("size_mb") or 0) <= room_mb - kept_mb

    for m in fits:
        if m["name"] == current and in_room(m):
            return current
    for m in fits:
        if m.get("kept_as"):
            return m["name"]
    for m in fits:
        if m.get("loaded_as"):
            return m["name"]
    if not fits:
        return None
    by_size = sorted(fits, key=lambda m: m.get("size_mb") or 0)
    room = [m for m in by_size if in_room(m)] if room_mb is not None else []
    return (room[-1] if room else by_size[0])["name"]


def room_mb(listing):
    card = (listing.get("gpus") or [{}])[0]
    if card.get("usable_mb") is None:
        return None
    return card["usable_mb"] - (card.get("reserve_mb") or 0)
