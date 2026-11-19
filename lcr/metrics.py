"""Caption-quality scores, reference-agreement and human-evaluation statistics."""
from __future__ import annotations

import math
from collections import Counter

import numpy as np

from .claims import extract, reference_classes, words


# ----------------------------------------------------------------- caption quality
def caption_scores(gens: dict[str, str], refs: dict[str, list[str]]) -> dict[str, float]:
    """BLEU-4 and CIDEr with the COCO caption toolkit scorers (pycocoevalcap).

    NOTE: tokenisation is `claims.words` (lower-case alphabetic words), not the
    Java PTBTokenizer, and METEOR / SPICE (both need Java) are not computed
    here. Numbers are therefore comparable across our systems but not exactly
    with published tables. scripts/colab_full_metrics.py runs the full toolkit.
    """
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.cider.cider import Cider
    g = {k: [" ".join(words(v))] for k, v in gens.items()}
    r = {k: [" ".join(words(x)) for x in refs[k]] for k in gens}
    bleu, _ = Bleu(4).compute_score(r, g, verbose=0)
    cider, _ = Cider().compute_score(r, g)
    return {"BLEU-4": 100 * bleu[3], "CIDEr": float(cider)}


# ------------------------------------------------------------- reference agreement
def ruor(gens: dict[str, str], refs: dict[str, list[str]]) -> dict[str, float]:
    """Reference-unsupported object rate (auxiliary; NOT a hallucination rate).

    Share of generated vocabulary claims whose class appears in none of the
    five references. Also returns claims per image.
    """
    total = unsupported = 0
    for k, cap in gens.items():
        present = reference_classes(refs[k])
        for c in extract(cap):
            total += 1
            unsupported += c.cls not in present
    return {"RUOR": 100 * unsupported / max(total, 1), "claims": total,
            "claims_per_image": total / max(len(gens), 1)}


# ----------------------------------------------------------------- human evaluation
def hicr(labels: list[str]) -> dict[str, float]:
    """labels: 'correct' / 'incorrect' / 'uncertain' for every claim in the sample."""
    c = Counter(labels)
    decided = c["correct"] + c["incorrect"]
    return {
        "HICR": 100 * c["incorrect"] / decided if decided else float("nan"),
        "HICR_uncertain_as_incorrect": 100 * (c["incorrect"] + c["uncertain"]) / max(len(labels), 1),
        "n_correct": c["correct"], "n_incorrect": c["incorrect"], "n_uncertain": c["uncertain"],
    }


def cohen_kappa(a: list[str], b: list[str]) -> float:
    assert len(a) == len(b) and len(a) > 0
    cats = sorted(set(a) | set(b))
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = Counter(a), Counter(b)
    pe = sum(pa[c] * pb[c] for c in cats) / (n * n)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def repair_rates(traces: list[tuple[str, str]]) -> dict[str, float]:
    """Fix / break / net repair rate.

    traces: one (draft_status, output_status) pair per DRAFT claim, with
    draft_status in {correct, incorrect} and output_status in
    {correct, incorrect, removed}.
    """
    n = len(traces)
    wrong = [o for d, o in traces if d == "incorrect"]
    right = [o for d, o in traces if d == "correct"]
    fixed = sum(o == "correct" for o in wrong)
    broken = sum(o != "correct" for o in right)
    return {
        "fix_rate": 100 * fixed / len(wrong) if wrong else float("nan"),
        "break_rate": 100 * broken / len(right) if right else float("nan"),
        "NRR": 100 * (fixed - broken) / n if n else float("nan"),
    }


# ----------------------------------------------------------------------- statistics
def bootstrap_ci(per_image: list, stat, n_boot: int = 2000, seed: int = 415, alpha: float = 0.05):
    """Image-level bootstrap: `per_image` is resampled with replacement as whole units,
    so all claims of one image stay together; `stat` maps a list of units to a float.
    For a paired comparison, put both systems' values in the same unit."""
    rng = np.random.default_rng(seed)
    n = len(per_image)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        v = stat([per_image[i] for i in idx])
        if not (isinstance(v, float) and math.isnan(v)):
            vals.append(v)
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return float(stat(per_image)), float(lo), float(hi)


def _z(p: float) -> float:
    from statistics import NormalDist
    return NormalDist().inv_cdf(p)


def claims_needed(p0: float, relative_reduction: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """Claims per system to detect p0 -> p0*(1-r) with a two-sided two-proportion z-test.

    This ignores pairing (conservative when outputs are positively correlated)
    and within-image clustering (optimistic); it is used to bracket N, which is
    then fixed before any test-set run.
    """
    p1 = p0 * (1 - relative_reduction)
    pbar = (p0 + p1) / 2
    num = (_z(1 - alpha / 2) * math.sqrt(2 * pbar * (1 - pbar))
           + _z(power) * math.sqrt(p0 * (1 - p0) + p1 * (1 - p1))) ** 2
    return math.ceil(num / (p0 - p1) ** 2)


def wilson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    z = _z(1 - alpha / 2)
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return mid - half, mid + half


def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Probability that a random positive scores above a random negative (ties count 1/2)."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = allv.argsort(kind="mergesort")
    ranks = np.empty(len(allv))
    sv = allv[order]
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def fix_rate_for(relative_reduction: float, p0: float, break_rate: float) -> float:
    """Fix rate at which traced draft claims improve HICR by exactly `relative_reduction`.

    Expected HICR change is p0 * fix - (1 - p0) * break; setting it to relative_reduction * p0 gives the rate.
    A power table must hold the effect at the hypothesis' size: fixing a third of the errors while breaking 1%
    of correct claims is a 21.8% reduction at p0 = 8%, not H1's 15%.
    """
    return (relative_reduction * p0 + (1 - p0) * break_rate) / p0


def paired_power(n_claims: int, p0: float, fix_rate: float, break_rate: float,
                 alpha: float = 0.05, n_sim: int = 4000, seed: int = 415) -> float:
    """Power of the exact two-sided sign (McNemar) test for 'fixes != breaks'.

    Each of `n_claims` draft claims is wrong with probability p0. A wrong claim
    is fixed with probability `fix_rate`; a correct claim is broken with
    probability `break_rate`. Only discordant claims carry information, which
    is why the paired design needs far fewer claims than two independent samples.
    """
    from scipy.stats import binom
    rng = np.random.default_rng(seed)
    wrong = rng.binomial(n_claims, p0, n_sim)
    fixes = rng.binomial(wrong, fix_rate)
    breaks = rng.binomial(n_claims - wrong, break_rate)
    d = fixes + breaks
    k = np.minimum(fixes, breaks)
    p = np.where(d > 0, np.minimum(1.0, 2 * binom.cdf(k, np.maximum(d, 1), 0.5)), 1.0)
    return float(np.mean((p < alpha) & (fixes > breaks)))
