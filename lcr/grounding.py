"""Human-drawn boxes for Look evaluation, from Flickr30k Entities (Plummer et al., ICCV 2015).

Every Flickr8k image is also a Flickr30k image (same Flickr photo id, same size), and the F30kE
sentences of these images are our Flickr8k reference captions. F30kE links noun phrases in those
captions to boxes. We map each phrase to a vocabulary class with our own claim lexicon and collect,
per (image, class), all boxes of the matching entities.

USED ONLY TO EVALUATE LOOK. Never as training signal; test-split boxes are not read during development.
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from functools import lru_cache

import numpy as np

from . import claims, config, data

ROOT = config.DATA / "flickr30k_entities"
_PHRASE = re.compile(r"\[/EN#(\d+)/(\S+) ([^\]]+)\]")


@lru_cache(maxsize=None)
def _photo_ids(split: str) -> tuple[str, ...]:
    """Flickr photo id of every image of a Flickr8k split, in dataset order."""
    import pyarrow.parquet as pq
    files = sorted((config.DATA / "flickr8k" / "data").glob(f"{split}-*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files for {split}")
    # Experiments only need photo identifiers, not gigabytes of encoded images.
    paths = [v for f in files for v in pq.read_table(f, columns=["image.path"])["path"].to_pylist()]
    return tuple(str(path).split("_")[0].split(".")[0] for path in paths)


def photo_id(iid: str) -> str:
    split, idx = data.parse_id(iid)
    return _photo_ids(split)[idx]


def phrase_class(text: str) -> str | None:
    """Vocabulary class of a noun phrase: the last word that maps to a class (the head is usually last)."""
    cls = None
    for w in claims.words(text):
        c = claims.word_class(w)
        if c is not None:
            cls = c
    return cls


def parse_sentences(text: str) -> list[tuple[str, list[str], str]]:
    """[(chain_id, types, phrase)] for every annotated phrase of a Sentences/<id>.txt file."""
    return [(m.group(1), m.group(2).split("/"), m.group(3)) for m in _PHRASE.finditer(text)]


def parse_boxes(xml_text: str) -> tuple[tuple[int, int], dict[str, list[list[float]]]]:
    """((width, height), {chain_id: [[x0, y0, x1, y1] normalised to the image]})."""
    root = ET.fromstring(xml_text)
    w = int(root.find("size/width").text); h = int(root.find("size/height").text)
    out: dict[str, list[list[float]]] = {}
    for obj in root.findall("object"):
        bb = obj.find("bndbox")
        if bb is None:
            continue                                    # scene / no-box entities
        x0, y0, x1, y1 = (float(bb.find(k).text) for k in ("xmin", "ymin", "xmax", "ymax"))
        box = [max(x0 - 1, 0) / w, max(y0 - 1, 0) / h, min(x1, w) / w, min(y1, h) / h]   # VOC is 1-based
        for name in obj.findall("name"):
            out.setdefault(name.text, []).append(box)
    return (w, h), out


@lru_cache(maxsize=20_000)
def class_boxes(iid: str) -> dict[str, np.ndarray]:
    """{class: (N, 4) boxes} for one Flickr8k image id such as 'validation_0012'."""
    split, _ = data.parse_id(iid)
    pid = photo_id(iid)
    sent = (ROOT / "Sentences" / f"{pid}.txt").read_text()
    _, boxes = parse_boxes((ROOT / "Annotations" / f"{pid}.xml").read_text())
    per_cls: dict[str, list[list[float]]] = {}
    seen: set[tuple[str, str]] = set()
    for chain, types, phrase in parse_sentences(sent):
        if chain == "0" or "notvisual" in types or chain not in boxes:
            continue
        c = phrase_class(phrase)
        if c is None or (c, chain) in seen:
            continue
        seen.add((c, chain))
        per_cls.setdefault(c, []).extend(boxes[chain])
    return {c: np.unique(np.asarray(b, np.float32), axis=0) for c, b in per_cls.items()}


def coverage_mask(boxes: np.ndarray, g: int = config.BLIP_GRID) -> np.ndarray:
    """Exact patch coverage by the union of boxes, using disjoint x slabs.

    Merge active y intervals before distributing each slab's area to patches.
    This counts overlaps once and adds disjoint pieces that share a patch.
    """
    from .look import _overlap_1d
    if g <= 0:
        raise ValueError("grid size must be positive")
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    if not np.isfinite(boxes).all():
        raise ValueError("box coordinates must be finite")
    boxes = np.clip(boxes, 0, 1)
    boxes = boxes[(boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])]
    out = np.zeros((g, g), dtype=float)
    edges = np.unique(boxes[:, [0, 2]])
    for left, right in zip(edges[:-1], edges[1:]):
        mid = (left + right) / 2
        active = boxes[(boxes[:, 0] < mid) & (mid < boxes[:, 2])]
        if not len(active):
            continue
        merged = []
        for lo, hi in sorted(active[:, [1, 3]].tolist()):
            if merged and lo <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], hi)
            else:
                merged.append([lo, hi])
        intervals = np.asarray(merged)
        wy = _overlap_1d(intervals[:, 0], intervals[:, 1], g).sum(axis=0)
        wx = _overlap_1d(np.array([left]), np.array([right]), g)[0]
        out += wy[:, None] * wx[None, :]
    return np.clip(out, 0, 1)


SMALL_SIDE = 2 / config.BLIP_GRID        # a GT box is "small" if its shorter side is below two BLIP patches


def coverage_hits(pred_boxes, gt: np.ndarray, small_side: float = SMALL_SIDE, iou_min: float = .5) -> dict:
    """Do Look's ranked candidates cover the class's human boxes?

    `pred_boxes` are the ranked candidates (top-1 first). A small-object hit counts only when a candidate
    reaches IoU >= iou_min with a SMALL GT box: when a class has both a small and a large box in the image, a
    hit on the large one is not evidence that the small object was found (the error in scripts/37's first
    count, which credited it).
    """
    from .look import iou
    gt = np.asarray(gt, np.float32).reshape(-1, 4)
    small = np.min(gt[:, 2:] - gt[:, :2], axis=1) < small_side
    ious = (np.stack([iou(np.asarray(b, np.float32), gt) for b in pred_boxes]) if len(pred_boxes)
            else np.zeros((0, len(gt))))
    hit = ious >= iou_min
    return dict(top1=bool(len(hit) and hit[0].any()), top3=bool(hit.any()), has_small=bool(small.any()),
                small_top3=bool(hit[:, small].any()) if small.any() else False,
                small_top3_any_box=bool(small.any() and hit.any()))


def point_in_boxes(y: float, x: float, boxes: np.ndarray) -> bool:
    return bool(np.any((boxes[:, 0] <= x) & (x <= boxes[:, 2]) & (boxes[:, 1] <= y) & (y <= boxes[:, 3])))
