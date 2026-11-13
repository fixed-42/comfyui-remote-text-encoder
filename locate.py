import json
import math
import re

import prompt_files

TASK = "locate"
LOCATE_TOKENS = 512
VISION = "<|vision_start|><|image_pad|><|vision_end|>"
_BOX = re.compile(r'"bbox_2d"\s*:\s*\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,'
                  r'\s*(-?\d+(?:\.\d+)?)\s*\]')


class NothingFound(Exception):
    pass


class EmptyAnswer(Exception):
    pass


def parse_task(req, images):
    task = req.get("task")
    if task != TASK:
        raise ValueError(f"unknown task {task!r}; /generate does only task 'locate'")
    if images != 1:
        raise ValueError(f"task 'locate' takes exactly 1 image, got {images}")
    if not isinstance(req.get("prompt"), str) or not req["prompt"].strip():
        raise ValueError("'prompt' is empty: say what to find")


def locate_prompt(what):
    return prompt_files.assemble("locate.txt", {"what": what.strip()})


def answer_text(raw):
    m = re.search(r"<think>(.*?)</think>", raw, re.S)
    if m:
        answer = raw[m.end():]
    elif "<think>" in raw:
        answer = ""
    elif "</think>" in raw:
        answer = raw.split("</think>", 1)[1]
    else:
        answer = raw
    answer = answer.strip()
    if not answer:
        raise EmptyAnswer("the model returned no answer")
    return answer


def user_turn(prompt, images):
    marks = "".join((f"Picture {i + 1}: " if images > 1 else "") + VISION for i in range(images))
    return marks + prompt


def parse_boxes(answer, width, height):
    text = answer.strip()
    found = []
    decoder, at = json.JSONDecoder(), 0
    while True:
        start = min((i for i in (text.find("[", at), text.find("{", at)) if i >= 0), default=-1)
        if start < 0:
            break
        try:
            data, at = decoder.raw_decode(text, start)
        except ValueError:
            at = start + 1
            continue
        for entry in data if isinstance(data, list) else [data]:
            box = entry.get("bbox_2d") if isinstance(entry, dict) else None
            if isinstance(box, list) and len(box) == 4 and all(_number(v) for v in box):
                label = entry.get("label") or entry.get("text_content") or ""
                found.append((box, label if isinstance(label, str) else ""))
    if not found:
        found = [([float(v) for v in m.groups()], "") for m in _BOX.finditer(text)]
        found = [(box, label) for box, label in found if all(map(math.isfinite, box))]
    boxes = []
    for (a, b, c, d), label in found:
        x1, x2 = sorted((_pixels(a, width), _pixels(c, width)))
        y1, y2 = sorted((_pixels(b, height), _pixels(d, height)))
        if x2 > x1 and y2 > y1:
            boxes.append({"x": x1, "y": y1, "width": x2 - x1, "height": y2 - y1, "label": label})
    if not boxes:
        raise NothingFound(text[:300])
    return boxes


def _number(v):
    if not isinstance(v, (int, float)) or isinstance(v, bool):
        return False
    try:
        return math.isfinite(v)
    except OverflowError:
        return False


def _pixels(v, size):
    return min(size, max(0, round(v / 1000 * size)))
