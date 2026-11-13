import torch

from .remote_clip import RemoteCLIP

MAX_PIXELS = 1024 * 1024


def fit(image, max_pixels=MAX_PIXELS):
    image = image[:1, :, :, :3]
    height, width = image.shape[1:3]
    if height * width <= max_pixels:
        return image
    scale = (max_pixels / (height * width)) ** 0.5
    size = (max(1, int(height * scale)), max(1, int(width * scale)))
    return torch.nn.functional.interpolate(image.movedim(-1, 1).float(), size=size, mode="area").movedim(1, -1)


def scale_boxes(boxes, sent, original):
    (sent_h, sent_w), (height, width) = sent, original
    out = []
    for b in boxes:
        x1, y1 = round(b["x"] * width / sent_w), round(b["y"] * height / sent_h)
        x2 = min(width, round((b["x"] + b["width"]) * width / sent_w))
        y2 = min(height, round((b["y"] + b["height"]) * height / sent_h))
        if x2 > x1 and y2 > y1:
            out.append({"x": x1, "y": y1, "width": x2 - x1, "height": y2 - y1, "label": b.get("label", "")})
    return out


def boxes_mask(boxes, height, width):
    mask = torch.zeros(1, height, width)
    for b in boxes:
        mask[0, b["y"]:b["y"] + b["height"], b["x"]:b["x"] + b["width"]] = 1.0
    return mask


def _remote(clip, node):
    if not isinstance(clip, RemoteCLIP):
        raise TypeError(f"{node} needs the CLIP from a Remote CLIP Loader")


class RemoteLocate:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "clip": ("CLIP", {"tooltip": "From a Remote CLIP Loader on an encoder that can read images."}),
            "image": ("IMAGE",),
            "what": ("STRING", {"default": "", "tooltip": "What to find, e.g. the hat, the face, the word on the sign. "
                                                          "Large things are found reliably, small ones less so."}),
        }}

    RETURN_TYPES = ("MASK", "BOUNDING_BOX")
    RETURN_NAMES = ("mask", "bboxes")
    FUNCTION = "locate"
    CATEGORY = "mask/remote"

    def locate(self, clip, image, what):
        _remote(clip, "Remote Locate")
        if not what.strip():
            raise ValueError("Remote Locate: say what to find")
        sent = fit(image)
        boxes, answer = clip.locate(sent, what)
        height, width = image.shape[1:3]
        boxes = scale_boxes(boxes, sent.shape[1:3], (height, width))
        if not boxes:
            raise RuntimeError(f"Remote Locate: nothing found for {what.strip()!r}; the model answered: {answer[:300]}")
        return boxes_mask(boxes, height, width), boxes
