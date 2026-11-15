"""Synthetic claim corruption (training only).

A fraction rho of training claims has its label replaced by a wrong noun that
frequently co-occurs with the same context in the training captions
(frisbee -> ball). The region is kept; detector scores, alternative and verdict
are recomputed for the corrupted label, so the record now *disagrees* with the
image. The training target (the five references) is unchanged.
"""
from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from .claims import reference_classes
from .vocab import KIND


def confusion_table(reference_sets: list[set[str]], top: int = 8) -> dict[str, list[str]]:
    """For each class c, the classes of the same kind that most often share c's context.

    Context similarity is second-order co-occurrence: d is a plausible wrong
    substitute for c if d tends to appear with the same *other* classes as c
    (ball and frisbee both appear with dog and grass) -- which is exactly the
    language-prior confusion that hallucination follows.
    """
    co: dict[str, Counter] = defaultdict(Counter)
    for s in reference_sets:
        for a in s:
            for b in s:
                if a != b:
                    co[a][b] += 1
    classes = sorted(co)
    idx = {c: i for i, c in enumerate(classes)}
    m = np.zeros((len(classes), len(classes)), dtype=np.float64)
    for a, cnt in co.items():
        for b, n in cnt.items():
            m[idx[a], idx[b]] = n
    m = m / np.clip(np.linalg.norm(m, axis=1, keepdims=True), 1e-9, None)
    sim = m @ m.T
    out: dict[str, list[str]] = {}
    for c in classes:
        if KIND[c] == "person":
            continue                                # person claims are never corrupted
        order = np.argsort(-sim[idx[c]])
        cands = [classes[j] for j in order if classes[j] != c and KIND[classes[j]] == KIND[c]]
        out[c] = cands[:top]
    return out


def build_confusions(all_refs: list[list[str]], top: int = 8) -> dict[str, list[str]]:
    return confusion_table([reference_classes(r) for r in all_refs], top=top)


def corrupt_label(claim: str, present: set[str], table: dict[str, list[str]],
                  rng: np.random.Generator) -> str | None:
    """A wrong label for `claim` that no reference mentions for this image, or None."""
    cands = [c for c in table.get(claim, []) if c not in present]
    if not cands:
        return None
    w = np.arange(len(cands), 0, -1, dtype=np.float64)      # prefer the closest confusions
    return str(rng.choice(cands, p=w / w.sum()))
