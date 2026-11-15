"""Check v2: per-class calibration of OWLv2 region scores (docs/LITERATUR_TARAMASI.md 6-8, 8a).

OWLv2 scores are not comparable across text queries (OWLv2 model card), yet the round-1 Check compared a
claim's score with other classes' scores under one shared threshold. Here every class's score is mapped to
a common scale before any threshold or comparison:

  percentile  rank of the score within that class's scores on the fitting data (global ranks for rare classes)
  offset      logistic regression on logit(score) with a shared slope and a per-class intercept, the intercepts
              shrunk to zero by an L2 penalty (Platt scaling, Guo+ ICML'17, with partial pooling across classes)
  offset+slope  as offset, plus a shrunk per-class slope

Target: 'some reference mentions the class' (automatic reference proxy, not human labels).
Fitting data are claims on half B; records on B itself are scored out-of-fold (cross-fitting), so no record
is scored by a calibrator that saw its own label.
"""
from __future__ import annotations

import hashlib

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from .vocab import CLASSES

N_CLASSES = len(CLASSES)
EPS = 1e-4


def fold_of(image_id: str, k: int = 5, salt: str = "lcr-check-v2") -> int:
    """Deterministic fold of an image (all claims of one image share a fold)."""
    return int(hashlib.sha1(f"{salt}:{image_id}".encode()).hexdigest(), 16) % k


def logit(s):
    s = np.clip(np.asarray(s, np.float64), EPS, 1 - EPS)
    return np.log(s / (1 - s))


class Percentile:
    """Per-class empirical CDF of the score; classes with fewer than `min_n` fitting samples use the global CDF."""

    def __init__(self, min_n: int = 20):
        self.min_n = min_n

    def fit(self, s, c, y=None):
        s, c = np.asarray(s, np.float64), np.asarray(c)
        self.glob = np.sort(s)
        self.per = {int(k): np.sort(s[c == k]) for k in np.unique(c) if (c == k).sum() >= self.min_n}
        return self

    def __call__(self, s, c):
        s, c = np.atleast_1d(np.asarray(s, np.float64)), np.atleast_1d(np.asarray(c))
        out = np.empty(len(s))
        for i, (si, ci) in enumerate(zip(s, c)):
            ref = self.per.get(int(ci), self.glob)
            out[i] = np.searchsorted(ref, si, side="right") / len(ref)
        return out


class Raw:
    def fit(self, s, c, y=None):
        return self

    def __call__(self, s, c):
        return np.atleast_1d(np.asarray(s, np.float64))


class ClassLogistic:
    """P(y=1 | s, class) = sigmoid((a + e_c) * logit(s) + b + d_c), with lam * (|d|^2 + |e|^2) shrinkage."""

    def __init__(self, lam: float = 1.0, per_class_slope: bool = False):
        self.lam, self.slope = lam, per_class_slope

    def _unpack(self, w):
        a, b = w[0], w[1]
        d = w[2:2 + N_CLASSES]
        e = w[2 + N_CLASSES:] if self.slope else np.zeros(N_CLASSES)
        return a, b, d, e

    def fit(self, s, c, y):
        x, c, y = logit(s), np.asarray(c), np.asarray(y, np.float64)
        n = len(x)
        onehot_idx = c

        def loss(w):
            a, b, d, e = self._unpack(w)
            z = (a + e[onehot_idx]) * x + b + d[onehot_idx]
            p = expit(z)
            nll = np.sum(np.logaddexp(0, z) - y * z)
            r = y - p                                     # d nll / d z = p - y
            g = np.empty_like(w)
            g[0] = -np.sum(r * x)
            g[1] = -np.sum(r)
            gd = -np.bincount(onehot_idx, weights=r, minlength=N_CLASSES)
            g[2:2 + N_CLASSES] = gd + 2 * self.lam * d
            pen = self.lam * np.sum(d ** 2)
            if self.slope:
                ge = -np.bincount(onehot_idx, weights=r * x, minlength=N_CLASSES)
                g[2 + N_CLASSES:] = ge + 2 * self.lam * e
                pen += self.lam * np.sum(e ** 2)
            return (nll + pen) / n, g / n

        w0 = np.zeros(2 + N_CLASSES * (2 if self.slope else 1))
        w0[0] = 1.0
        self.w = minimize(loss, w0, jac=True, method="L-BFGS-B").x
        return self

    def __call__(self, s, c):
        a, b, d, e = self._unpack(self.w)
        x, c = logit(np.atleast_1d(s)), np.atleast_1d(np.asarray(c))
        return expit((a + e[c]) * x + b + d[c])


def make(name: str, lam: float = 1.0):
    return {"raw": lambda: Raw(), "percentile": lambda: Percentile(),
            "offset": lambda: ClassLogistic(lam, False),
            "offset_slope": lambda: ClassLogistic(lam, True)}[name]()


def cross_fit(name: str, s, c, y, folds, lam: float = 1.0):
    """Out-of-fold calibrated values for the fitting data itself."""
    s, c, y, folds = map(np.asarray, (s, c, y, folds))
    out = np.empty(len(s))
    for f in np.unique(folds):
        m = folds == f
        cal = make(name, lam).fit(s[~m], c[~m], y[~m])
        out[m] = cal(s[m], c[m])
    return out


# ------------------------------------------------------------------ metrics

def auroc(score, y) -> float:
    """Mann-Whitney AUROC with average ranks for ties."""
    from scipy.stats import rankdata
    score, y = np.asarray(score, np.float64), np.asarray(y, bool)
    npos, nneg = y.sum(), (~y).sum()
    if npos == 0 or nneg == 0:
        return float("nan")
    r = rankdata(score)
    return float((r[y].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def ece(p, y, bins: int = 10) -> float:
    """Expected calibration error (Guo+ ICML'17), equal-width bins."""
    p, y = np.asarray(p, np.float64), np.asarray(y, np.float64)
    idx = np.minimum((p * bins).astype(int), bins - 1)
    return float(sum(abs(p[idx == b].mean() - y[idx == b].mean()) * (idx == b).mean()
                     for b in range(bins) if (idx == b).any()))


def log_loss(p, y) -> float:
    p = np.clip(np.asarray(p, np.float64), 1e-6, 1 - 1e-6)
    y = np.asarray(y, np.float64)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def ap_wrong(p, ok) -> float:
    """Average precision for wrong claims; equal scores enter at one threshold.

    Resolving ties claim by claim makes AP depend on input order, particularly
    for a class-only prior. No wrong claims yields 0, following the usual AP convention.
    """
    p, ok = np.asarray(p, np.float64), np.asarray(ok, bool)
    if not len(p) or ok.all():
        return 0.0
    order = np.argsort(p, kind="stable")
    hits = (~ok[order]).astype(float)
    ends = np.r_[np.flatnonzero(np.diff(p[order])), len(p) - 1]
    tp = np.cumsum(hits)[ends]
    return float(np.sum(np.diff(np.r_[0.0, tp]) * tp / (ends + 1)) / hits.sum())


def verdict_quality(verdicts, ok) -> dict:
    """Informedness J of a three-way verdict against a binary 'claim is correct' label, plus precisions."""
    v, ok = np.asarray(verdicts), np.asarray(ok, bool)
    sup, con = v == "supported", v == "contradicted"
    rate = lambda m, g: float(m[g].mean()) if g.any() else 0.0
    j = (rate(sup, ok) - rate(sup, ~ok)) + (rate(con, ~ok) - rate(con, ok))
    return dict(J=j, n=int(len(v)), base_rate_wrong=float((~ok).mean()),
                supported=float(sup.mean()), contradicted=float(con.mean()), uncertain=float(1 - sup.mean() - con.mean()),
                supported_precision=float(ok[sup].mean()) if sup.any() else float("nan"),
                contradicted_precision=float((~ok)[con].mean()) if con.any() else float("nan"),
                recall_wrong=float(con[~ok].mean()) if (~ok).any() else float("nan"))


class FeatureLogistic:
    """P(y=1) = sigmoid(X w + b + d_class): dense features (standardised on the fitting data) plus shrunk
    per-class intercepts. lam_c penalises the class intercepts, lam_w the feature weights. With no features it is
    the per-class prior alone (a language prior, not visual evidence); with use_class=False the class offsets are
    dropped."""

    def __init__(self, lam_c: float = 1.0, lam_w: float = 1.0, use_class: bool = True):
        self.lam_c, self.lam_w, self.use_class = lam_c, lam_w, use_class

    def fit(self, X, c, y):
        X = np.asarray(X, np.float64).reshape(len(y), -1)
        # Fit imputation inside each training fold. Held-out images must use
        # training means, including when a whole column is missing in that fold.
        count = np.sum(np.isfinite(X), axis=0)
        self.impute = np.divide(np.nansum(X, axis=0), count,
                                out=np.zeros(X.shape[1]), where=count > 0)
        X = np.where(np.isnan(X), self.impute, X)
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-9
        Z = (X - self.mu) / self.sd
        c, y = np.asarray(c), np.asarray(y, np.float64)
        n, d = Z.shape
        nc = N_CLASSES if self.use_class else 0

        def loss(p):
            w, b, dc = p[:d], p[d], p[d + 1:]
            z = Z @ w + b + (dc[c] if nc else 0.0)
            r = y - expit(z)
            nll = np.sum(np.logaddexp(0, z) - y * z)
            g = np.empty_like(p)
            g[:d] = -(Z.T @ r) + 2 * self.lam_w * w
            g[d] = -np.sum(r)
            pen = self.lam_w * np.sum(w ** 2)
            if nc:
                g[d + 1:] = -np.bincount(c, weights=r, minlength=nc) + 2 * self.lam_c * dc
                pen += self.lam_c * np.sum(dc ** 2)
            return (nll + pen) / n, g / n

        self.d = d
        self.p = minimize(loss, np.zeros(d + 1 + nc), jac=True, method="L-BFGS-B").x
        return self

    def __call__(self, X, c):
        X = np.asarray(X, np.float64).reshape(len(c), -1)
        X = np.where(np.isnan(X), self.impute, X)
        Z = (X - self.mu) / self.sd
        w, b, dc = self.p[:self.d], self.p[self.d], self.p[self.d + 1:]
        z = Z @ w + b + (dc[np.asarray(c)] if self.use_class else 0.0)
        return expit(z)

    def weights(self, names):
        return {n: float(v) for n, v in zip(names, self.p[:self.d])}
