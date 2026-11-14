from . import detect_type, log
from .image_nodes import RemoteLocate
from .remote_clip import NO_MODEL, RemoteCLIP

WEB_DIRECTORY = "./web"


def clip_types():
    try:
        import nodes
        return list(nodes.CLIPLoader.INPUT_TYPES()["required"]["type"][0])
    except (ImportError, AttributeError, KeyError, IndexError, TypeError):
        import comfy.sd
        return [t.name.lower() for t in comfy.sd.CLIPType]


class RemoteCLIPLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"server_url": ("STRING", {
                "default": "http://127.0.0.1:8188",
                "tooltip": "The encoder PC's ComfyUI, e.g. http://pc-b:8188 (it serves text encoders: Settings > "
                           "Remote Text Encoder). local: this ComfyUI's own encoder server, on its other GPU. A "
                           "standalone server.py or router: its own address."})},
            "optional": {
                "type": (clip_types(), {"tooltip": "How to load the encoder, as in CLIPLoader. The menu offers the "
                                        "types the server's files can be loaded as; pick it first, then the file. "
                                        "With auto_detect connected, set from the diffusion model at Run."}),
                "model": ([NO_MODEL], {"default": NO_MODEL, "tooltip": "Encoder on the server for the chosen type: "
                                       "files (on-demand server) or config names (preloaded server)."}),
                "auto_detect": ("MODEL", {"tooltip": "Optional: connect the diffusion model loader's MODEL. At Run "
                                          "the type is taken from that model, and an encoder file that suits it "
                                          "when the chosen one doesn't; the menus change to them."}),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, model=NO_MODEL, type=None, auto_detect=None):
        return True

    RETURN_TYPES = ("CLIP",)
    FUNCTION = "load"
    CATEGORY = "loaders/remote"

    def load(self, server_url, model=NO_MODEL, type=None, auto_detect=None):
        if auto_detect is None:
            return (RemoteCLIP(server_url, model, clip_type=type),)
        detected, name = detect_type.clip_type_for(auto_detect, clip_types())
        if detected is None:
            log.warning(f"Remote CLIP Loader: auto_detect doesn't know the type for {name}; using '{type}' and "
                        f"'{model}' as set")
            return (RemoteCLIP(server_url, model, clip_type=type),)
        if detected != type:
            log.info(f"Remote CLIP Loader: auto_detect: {name} -> type '{detected}' (menu had '{type}')")
        clip = RemoteCLIP(server_url, model, clip_type=detected, auto=(name, detect_type.expected_archs(auto_detect)))
        ui = {}
        if clip.clip_type == detected != type:
            ui["remote_te_type"] = [detected]
        if clip.model_name != model and clip.clip_type == detected:
            ui["remote_te_model"] = [clip.model_name]
        return {"ui": ui, "result": (clip,)} if ui else (clip,)


class RemoteEncoderStatus:

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"server_url": ("STRING", {
            "default": "local",
            "tooltip": "local: this ComfyUI's own encoder server (Settings > Remote Text Encoder). On the rendering "
                       "PC: PC-B's address, e.g. http://pc-b:8188"})}}

    RETURN_TYPES = ()
    FUNCTION = "show"
    CATEGORY = "loaders/remote"
    DESCRIPTION = ("PC-B's encoders: which are kept loaded (never unloaded), which are loaded on demand, and the "
                   "VRAM they use. Tick Keep and press Apply to change the kept models.")

    def show(self, server_url):
        return ()


NODE_CLASS_MAPPINGS = {
    "RemoteCLIPLoader": RemoteCLIPLoader,
    "RemoteEncoderStatus": RemoteEncoderStatus,
    "RemoteLocate": RemoteLocate,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "RemoteCLIPLoader": "Remote CLIP Loader",
    "RemoteEncoderStatus": "Remote Encoder Status",
    "RemoteLocate": "Remote Locate",
}

_serving = None


def _child():
    return _serving.child if _serving is not None else None


def _setup_in_comfyui(prompt_server):
    global _serving
    import logging
    from . import endpoint, forward, log, routes, serve
    level = log.configure()
    try:
        _serving = serve.serving_in_comfyui(out=logging.info)
    except Exception as e:
        logging.error(f"{serve.PREFIX} cannot start: {e}")
    serve_now = _serving is not None and _serving.wanted()[0]
    log.info(f"loaded; log level: {level or 'as ComfyUI (--verbose)'} (REMOTE_TE_LOG=debug for every request); "
             f"serving text encoders: {'on' if serve_now else 'off (Settings > Remote Text Encoder)'}")
    endpoint.set_local(_child, lambda: _serving.error if _serving is not None else None)
    routes.register(prompt_server, _child, _serving)
    forward.register(prompt_server, _child)


try:
    import server as _comfy_server
    if hasattr(_comfy_server, "PromptServer"):
        _setup_in_comfyui(_comfy_server.PromptServer.instance)
except ImportError:
    pass
