"""Look: turn decoder cross-attention into a region choice among detector proposals.

Coordinate conventions (all boxes are xyxy, normalised to the ORIGINAL image,
so x and y both run from 0 to 1):

* BLIP resizes the image to 384x384 without keeping the aspect ratio, so the
  24x24 attention grid is a uniform grid over the original image in normalised
  coordinates.
* OWLv2 pads the image to a square (bottom/right) before resizing, so its boxes
  are normalised to the padded square of side max(W, H). `owl_to_image`
  undoes that.
"""
from __future__ import annotations

import numpy as np
import torch

from . import config


def attention_map(attn: torch.Tensor, position: int, last_k: int = 4,
                  sink: bool = False, valid_t: int | None = None) -> np.ndarray:
    """24x24 map for one caption.

    attn: (L, H, T, 577) cross-attention of one caption; `position` is the
    index of the claim's first word piece in the id sequence (BOS = 0). The
    token at `position` is predicted from decoder position `position - 1`.
    Heads of the last `last_k` layers are averaged, the CLS patch is dropped
    and the map is renormalised to sum to one.

    sink=True subtracts the caption's mean map over its first `valid_t`
    positions and clips at zero. BLIP's decoder puts most of its attention on a
    few background patches regardless of the word being generated ("attention
    sinks"); those patches cancel out and the token-specific part remains.
    On validation this raised the pointing-game hit rate from below chance to
    well above it (scripts/09_attention_variants.py).
    """
    a = attn[-last_k:, :, position - 1, 1:].float().mean(dim=(0, 1))
    if sink:
        t = valid_t or attn.shape[2]
        mean = attn[-last_k:, :, :t, 1:].float().mean(dim=(0, 1, 2))
        a = (a - mean).clamp_min(0)
        if a.sum() <= 0:                       # degenerate: fall back to uniform
            a = torch.ones_like(a)
    a = a / a.sum().clamp_min(1e-12)
    g = config.BLIP_GRID
    return a.reshape(g, g).cpu().numpy()


def owl_to_image(boxes_cxcywh: np.ndarray, width: int, height: int) -> np.ndarray:
    """OWLv2 (cx, cy, w, h) on the padded square -> xyxy normalised to the image."""
    side = float(max(width, height))
    cx, cy, w, h = (boxes_cxcywh[:, i] for i in range(4))
    x0, x1 = (cx - w / 2) * side / width, (cx + w / 2) * side / width
    y0, y1 = (cy - h / 2) * side / height, (cy + h / 2) * side / height
    return np.clip(np.stack([x0, y0, x1, y1], axis=1), 0.0, 1.0)


def _overlap_1d(lo: np.ndarray, hi: np.ndarray, g: int) -> np.ndarray:
    """Fraction of each of the g cells covered by [lo, hi]; shape (N, g)."""
    edges = np.arange(g + 1) / g
    left = np.maximum(lo[:, None], edges[None, :-1])
    right = np.minimum(hi[:, None], edges[None, 1:])
    return np.clip(right - left, 0.0, None) * g


def attention_mass(att: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Attention mass inside each box, with area-weighted partial patches."""
    g = att.shape[0]
    wx = _overlap_1d(boxes[:, 0], boxes[:, 2], g)        # (N, g) columns
    wy = _overlap_1d(boxes[:, 1], boxes[:, 3], g)        # (N, g) rows
    return np.einsum("ni,ij,nj->n", wy, att, wx)


def box_area(boxes: np.ndarray) -> np.ndarray:
    return np.clip(boxes[:, 2] - boxes[:, 0], 0, None) * np.clip(boxes[:, 3] - boxes[:, 1], 0, None)


def excess_mass(att: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Attention mass inside the box minus the fraction of the image it covers.

    A uniform attention map gives exactly 0 for every box; a box covering the
    whole image also gives 0. Large boxes are not favoured (their area is
    subtracted) and neither are tiny boxes around one peak patch (the score is
    a mass, not a density).
    """
    return attention_mass(att, boxes) - box_area(boxes)


def size_ok(boxes: np.ndarray, min_patches: int = config.MIN_BOX_PATCHES) -> np.ndarray:
    m = min_patches / config.BLIP_GRID
    return ((boxes[:, 2] - boxes[:, 0]) >= m) & ((boxes[:, 3] - boxes[:, 1]) >= m)


def select_by_attention(att: np.ndarray, boxes: np.ndarray, threshold: float = 0.0):
    """Index of the proposal with the highest excess mass, or None.

    Returns (index or None, excess-mass array). None means 'no suitable region'.
    """
    if len(boxes) == 0:
        return None, np.zeros(0)
    ex = excess_mass(att, boxes)
    ex_valid = np.where(size_ok(boxes), ex, -np.inf)
    best = int(np.argmax(ex_valid))
    if not np.isfinite(ex_valid[best]) or ex_valid[best] < threshold:
        return None, ex
    return best, ex


def region_score(att: np.ndarray, boxes: np.ndarray, objectness: np.ndarray,
                 alpha: float = config.REGION_ALPHA, gamma: float = config.REGION_GAMMA) -> np.ndarray:
    """Look v2 map -> region score: attention mass / area**alpha, times objectness**gamma.

    alpha = 0 is the raw mass (favours large boxes), alpha = 1 the density (favours one peak patch);
    objectness is OWLv2's class-agnostic head, so no claim-specific detector score enters Look.
    Scale-free in `att`, so the map need not be normalised.
    """
    base = attention_mass(att, boxes) / np.clip(box_area(boxes), 1e-6, None) ** alpha
    return np.where(base > 0, base * np.clip(objectness, 1e-6, None) ** gamma, base) if gamma else base


def select_region(att: np.ndarray, boxes: np.ndarray, objectness: np.ndarray, **kw):
    """Index of the best proposal under `region_score` (size-filtered), or None; plus the score array."""
    if len(boxes) == 0:
        return None, np.zeros(0)
    s = np.where(size_ok(boxes), region_score(att, boxes, objectness, **kw), -np.inf)
    best = int(np.argmax(s))
    return (best if np.isfinite(s[best]) else None), s


def smooth_map(att: np.ndarray, sigma: float = 0) -> np.ndarray:
    """Frozen E1 preprocessing, including the precision used during selection."""
    from scipy.ndimage import gaussian_filter
    a = np.asarray(att, np.float32)
    total = float(a.sum())
    a = a / total if total > 0 else np.full_like(a, 1 / a.size)
    if sigma:
        a = gaussian_filter(a, sigma=float(sigma), mode="reflect")
        a = a / float(a.sum())
        a = a.astype(np.float16).astype(np.float32)
        a = a / float(a.sum())
    return a


def select_frozen(att: np.ndarray, boxes: np.ndarray, objectness: np.ndarray, kind: str):
    """Look v3: only E1 components passing the preregistered B threshold."""
    rule = config.LOOK_RULES_BY_KIND[kind]
    mapped = smooth_map(att, rule["sigma"])
    index, scores = select_region(mapped, boxes, objectness, alpha=rule["alpha"], gamma=rule["gamma"])
    return index, scores, mapped


def iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    x0 = np.maximum(box[0], boxes[:, 0]); y0 = np.maximum(box[1], boxes[:, 1])
    x1 = np.minimum(box[2], boxes[:, 2]); y1 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    union = box_area(box[None])[0] + box_area(boxes) - inter
    return inter / np.clip(union, 1e-12, None)
