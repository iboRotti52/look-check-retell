"""Human-pilot adjudication rules used by scripts/10_score_labels.py.

Every rule gives the same answer whatever order the annotator CSVs are passed in, and an adjudicator
only decides what the annotators did not agree on -- the same convention as the main label
(agreement, else adjudicator, else 'uncertain'). An `entry` is one annotator's CSV row for one claim.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np

BOX_AGREE_IOU = 0.5          # two annotators' referent boxes count as the same box above this IoU


def parse_box(text) -> list[float] | None:
    """'x0:y0:x1:y1' (annotate.html's export) -> [x0, y0, x1, y1] in [0, 1], or None."""
    if not text:
        return None
    box = [float(v) for v in str(text).split(":")]
    if len(box) != 4 or not all(np.isfinite(box)) or box[2] <= box[0] or box[3] <= box[1]:
        return None
    return box


def box_iou(a, b) -> float:
    ix = max(0., min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0., min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.


def final_error_type(entries, adjudicator=None) -> tuple[str, str]:
    """(error type, source) for a claim whose final label is 'incorrect'.

    Agreement: every annotator who labelled the claim said 'incorrect' and gave the same error type.
    Otherwise the adjudicator's error type, else 'uncertain'.
    """
    labelled = [e for e in entries if e.get("label")]
    types = {e.get("error_type") or "" for e in labelled}
    if labelled and all(e["label"] == "incorrect" for e in labelled) and len(types) == 1 and "" not in types:
        return types.pop(), "agreement"
    if adjudicator and adjudicator.get("error_type"):
        return adjudicator["error_type"], "adjudicator"
    return "uncertain", "unresolved"


def final_box(entries, adjudicator=None, min_iou=BOX_AGREE_IOU):
    """(gold referent box or None, source, mean pairwise annotator IoU or None) for a misidentified claim.

    Agreement: two or more annotators drew boxes and every pair overlaps with IoU >= min_iou -> their
    coordinate-wise mean. Otherwise the adjudicator's box; otherwise a single annotator's box if only one
    was drawn; otherwise None, i.e. the claim needs adjudication before it can enter the gold set.
    """
    boxes = [b for e in entries if e.get("error_type") == "misidentified" and (b := parse_box(e.get("box")))]
    pair = [box_iou(a, b) for a, b in combinations(boxes, 2)]
    agreement = float(np.mean(pair)) if pair else None
    if len(boxes) >= 2 and min(pair) >= min_iou:
        return np.mean(boxes, axis=0).tolist(), "agreement", agreement
    adj = parse_box(adjudicator.get("box")) if adjudicator else None
    if adj is not None:
        return adj, "adjudicator", agreement
    if len(boxes) == 1:
        return boxes[0], "single_annotator", None
    return None, ("disagreement" if boxes else "no_box"), agreement
