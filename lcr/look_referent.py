"""Look L1: referent-evidence signals for one claim (docs/LOOK_L1_PROTOKOL.md).

Three signals, computed independently of which box Look finally picks:
* strength       -- lcr.look_v4.residual_strength (passed in, not recomputed here).
* concentration   -- entropy of the map, and the mass share inside the top-1 candidate box.
* null comparison -- how much more this claim's own map favours its top-1 box than the
                     OTHER claims' maps of the same caption do (margin and rank).

Pure numpy, independent of lcr.look / torch: this module is meant to run as a cheap CPU-only
pass over already-cached Look E3 attention maps (scripts/46_look_referent.py), so it
re-implements the small amount of box-mass geometry it needs instead of importing the
torch-coupled lcr.look module. The box-mass definition is identical to
lcr.look.attention_mass/excess_mass (single-box case); tests/test_look_referent.py checks
that the two agree.
"""
from __future__ import annotations

import numpy as np


def _box_mass(att: np.ndarray, box: np.ndarray) -> float:
    """Attention mass of a (g, g) map inside one xyxy box, with area-weighted edge patches."""
    g = att.shape[0]
    edges = np.arange(g + 1) / g
    wx = np.clip(np.minimum(box[2], edges[1:]) - np.maximum(box[0], edges[:-1]), 0, None) * g
    wy = np.clip(np.minimum(box[3], edges[1:]) - np.maximum(box[1], edges[:-1]), 0, None) * g
    return float(wy @ att @ wx)


def _box_area(box: np.ndarray) -> float:
    return float(max(box[2] - box[0], 0) * max(box[3] - box[1], 0))


def excess_mass(att: np.ndarray, box: np.ndarray) -> float:
    """Mass inside `box` minus the fraction of the image it covers (lcr.look.excess_mass, one box)."""
    return _box_mass(att, box) - _box_area(box)


def entropy(att: np.ndarray) -> float:
    """Shannon entropy of a (not necessarily normalised) map, in bits.

    0 = all mass on one patch; log2(n_patches) = uniform. A map with zero total mass (every
    weight clipped to 0, e.g. a claim whose sink-corrected map is entirely non-positive) is
    treated as uniform, matching look_v4.selected_map's fallback.
    """
    p = np.clip(np.asarray(att, np.float64), 0, None).ravel()
    total = p.sum()
    if total <= 0:
        return float(np.log2(p.size))
    p = p / total
    nz = p[p > 0]
    return float(-(nz * np.log2(nz)).sum())


def top1_concentration(att: np.ndarray, box: np.ndarray) -> float:
    """Share of a normalised map's mass inside the top-1 candidate box."""
    return _box_mass(att, box)


def null_margin(att: np.ndarray, box: np.ndarray, other_maps) -> dict:
    """How much more this claim's map favours `box` than the other claims' maps of the same caption do.

    `att` is this claim's own map; `box` is its top-1 candidate (from region_evidence.candidates);
    `other_maps` is an iterable of the (g, g) maps of every other claim extracted from the same
    caption (whatever box each of them chose is irrelevant -- only this claim's box is scored
    under every map). `margin` is this claim's excess mass on `box` minus the best excess mass
    any other claim's map reaches on the same box; `rank` is 1 if this claim's map explains the
    box better than all others, 2 if exactly one other map explains it better, and so on.
    """
    own = excess_mass(att, box)
    others = [excess_mass(m, box) for m in other_maps]
    if not others:
        return dict(margin=float("nan"), rank=1, n_other_claims=0)
    rank = 1 + sum(s > own for s in others)
    return dict(margin=own - max(others), rank=rank, n_other_claims=len(others))


def signals(strength: float, att: np.ndarray, box: np.ndarray, other_maps) -> dict:
    """All three Look-L1 referent signals for one claim (docs/LOOK_L1_PROTOKOL.md)."""
    out = dict(strength=float(strength), entropy=entropy(att),
              top1_concentration=top1_concentration(att, box))
    out.update({f"null_{k}": v for k, v in null_margin(att, box, other_maps).items()})
    return out
