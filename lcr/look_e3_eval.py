"""Fast region / pointing evaluation for Look E3 (docs/LOOK_E3_PROTOKOL.md).

Same definitions as lcr/look_eval.py (v3 box rules, IoU >= 0.5 against any human box of the class,
first-arg-max pointing), vectorised so that many map families can be cross-validated on half B.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, label

from . import calibrate, config, look, look_eval as ev

KINDS = ev.KINDS
G = 24


class Bank:
    """Claims of one split with boxes, objectness and (optional) human boxes, in pipeline order."""

    def __init__(self, stem: str):
        rows, entries = ev.load_bank(stem)
        self.rows, self.e = rows, entries
        self.n = len(entries)
        self.iid = np.array([e["iid"] for e in entries])
        self.kind = np.array([e["kind"] for e in entries])
        self.cls = np.array([e["cls"] for e in entries])
        self.has_gt = np.array([e["gtiou"] is not None for e in entries])
        self.fold = np.array([calibrate.fold_of(i) for i in self.iid])
        self.owner = np.array([[e["ri"], e["ci"]] for e in entries])
        self.archived = np.stack([e["att"] for e in entries]).astype(np.float32)
        # box weights: mass = einsum(wy, att, wx)
        self.wy, self.wx = [], []
        centres = (np.arange(G) + .5) / G
        self.ptmask = np.zeros((self.n, G * G), bool)
        for j, e in enumerate(entries):
            b = e["boxes"]
            self.wy.append(look._overlap_1d(b[:, 1], b[:, 3], G).astype(np.float32))
            self.wx.append(look._overlap_1d(b[:, 0], b[:, 2], G).astype(np.float32))
            if e["gt"] is not None:
                gt = e["gt"]
                yy, xx = np.meshgrid(centres, centres, indexing="ij")
                inside = np.zeros((G, G), bool)
                for box in gt:
                    inside |= (box[0] <= xx) & (xx <= box[2]) & (box[1] <= yy) & (yy <= box[3])
                self.ptmask[j] = inside.ravel()

    def prepare(self, maps: np.ndarray, rules=None) -> np.ndarray:
        """Normalise and apply the per-kind smoothing of the box rule (v3 by default)."""
        rules = rules or config.LOOK_RULES_BY_KIND
        out = np.empty((self.n, G, G), np.float32)
        for j in range(self.n):
            out[j] = look.smooth_map(maps[j].reshape(G, G), rules[self.kind[j]]["sigma"])
        return out

    def regions(self, maps: np.ndarray, rules=None, prepared=False) -> np.ndarray:
        """Chosen proposal index per claim (-1 if none)."""
        rules = rules or config.LOOK_RULES_BY_KIND
        att = maps if prepared else self.prepare(maps, rules)
        out = np.full(self.n, -1)
        for j, e in enumerate(self.e):
            r = rules[self.kind[j]]
            mass = np.einsum("ni,ij,nj->n", self.wy[j], att[j], self.wx[j])
            score = mass / np.clip(e["area"], 1e-6, None) ** r["alpha"] * np.clip(e["obj"], 1e-6, None) ** r["gamma"]
            if e["valid"].any():
                out[j] = int(np.argmax(np.where(e["valid"], score, -np.inf)))
        return out

    def hits(self, maps: np.ndarray, rules=None, prepared=False) -> np.ndarray:
        """Per-claim region hit (bool); claims without human boxes are False and masked by has_gt."""
        k = self.regions(maps, rules, prepared)
        return np.array([bool(k[j] >= 0 and self.e[j]["gtiou"] is not None and self.e[j]["gtiou"][k[j]] >= .5)
                         for j in range(self.n)])

    def pointing(self, maps: np.ndarray) -> np.ndarray:
        flat = maps.reshape(self.n, -1)
        return self.ptmask[np.arange(self.n), flat.argmax(1)]

    def summary(self, maps: np.ndarray, rules=None) -> dict:
        h, p = self.hits(maps, rules), self.pointing(maps)
        out = {}
        for kind in ("all", *KINDS):
            m = self.has_gt & ((self.kind == kind) if kind != "all" else True)
            out[kind] = dict(n=int(m.sum()), region=100 * float(h[m].mean()), pointing=100 * float(p[m].mean()))
        return out


def record_list(bank: Bank, hits: np.ndarray) -> list[dict]:
    return [dict(iid=bank.iid[j], kind=bank.kind[j], hit=bool(hits[j])) for j in range(bank.n) if bank.has_gt[j]]


def unit_maps(x: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(x, np.float32).reshape(len(x), -1), 0, None)
    s = x.sum(1, keepdims=True)
    return np.where(s > 0, x / np.where(s > 0, s, 1), 1 / x.shape[1])


def rank_maps(x: np.ndarray) -> np.ndarray:
    flat = np.asarray(x, np.float32).reshape(len(x), -1)
    from scipy.stats import rankdata
    # Equal attention values (especially zero background) must retain equal ranks.
    return ((rankdata(flat, axis=1, method="average") - 1) / (flat.shape[1] - 1)).astype(np.float32)


def component_entropy(m: np.ndarray) -> float:
    """Kang et al. (CVPR'25) spatial entropy: 8-connected components of the map thresholded at its mean."""
    lab, k = label(m > m.mean(), structure=np.ones((3, 3)))
    if k == 0:
        return 0.0
    sizes = np.bincount(lab.ravel())[1:].astype(np.float64)
    q = sizes / sizes.sum()
    return float(-(q * np.log(q)).sum())


def sanity(bank: Bank, maps: np.ndarray) -> dict:
    """Orientation (transpose) and word-specificity (swap with another class in the same image) gates."""
    m = maps.reshape(bank.n, G, G)
    own = bank.pointing(m)
    transposed = bank.pointing(np.transpose(m, (0, 2, 1)).copy())
    swap_own, swap_other = [], []
    by_img = {}
    for j in np.where(bank.has_gt)[0]:
        by_img.setdefault(bank.iid[j], []).append(j)
    for js in by_img.values():
        for j in js:
            alt = [i for i in np.where(bank.iid == bank.iid[j])[0] if bank.cls[i] != bank.cls[j]]
            if alt:
                swap_own.append(own[j])
                swap_other.append(bank.ptmask[j][m[alt[0]].argmax()])
    sel = bank.has_gt
    res = dict(pointing=100 * float(own[sel].mean()), transposed=100 * float(transposed[sel].mean()),
               swap_own=100 * float(np.mean(swap_own)), swap_other=100 * float(np.mean(swap_other)))
    res["passed"] = bool(res["pointing"] > res["transposed"] and res["swap_own"] > res["swap_other"])
    return res
