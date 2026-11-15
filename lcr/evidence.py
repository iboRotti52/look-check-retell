"""Turn pipeline rows into evidence tokens for each compared system."""
from __future__ import annotations

import numpy as np

from . import check, corrupt, pipeline, retell
from .vocab import CLASSES, CLASS_INDEX, KIND

MODES = ("ours", "b3", "b2", "b0plus")
MAX_TAGS = 6


def image_tags(scores: np.ndarray, t_support: float, n: int = MAX_TAGS) -> list[tuple[str, float]]:
    """B2 evidence: the detector's most confident vocabulary classes for the whole image."""
    if len(scores) == 0:
        return []
    best = scores.max(axis=0)
    order = np.argsort(-best)[:n]
    return [(CLASSES[j], float(best[j])) for j in order if best[j] >= t_support]


def tag_feature(cls: str, s: float):
    rec = dict(claim=cls, alt="", s_claim=0.0, s_alt=0.0, s_image=s, box=[], region=-1, excess=0.0, verdict="tag")
    return retell.record_features(rec, None)


def records_for(row: dict, mode: str, th: dict) -> list[dict]:
    """Finalised claim records of one image for 'ours' (attention regions) or 'b3' (detector regions)."""
    key = "rec_att" if mode == "ours" else "rec_det"
    return [pipeline.finalise(c[key], th) for c in row["claims"]]


def features(row: dict, mode: str, th: dict, prop: dict, scores: np.ndarray, *,
             rng: np.random.Generator | None = None, rho: float = 0.0, p_drop: float = 0.0,
             table: dict | None = None, present: set[str] | None = None,
             override: list[dict] | None = None) -> tuple[list[tuple], dict]:
    """Evidence tokens for one image.

    Training-time options: `rho` corrupts a claim's label (same region, scores
    recomputed), `p_drop` drops individual tokens. `override` replaces the
    records (used by the evidence interventions). Returns (features, info).
    """
    info = dict(n=0, corrupted=0, verdicts=[])
    if mode == "b0plus":
        return [], info
    if mode == "b2":
        feats = [tag_feature(c, s) for c, s in image_tags(scores, th["t_support"])]
        if rng is not None and p_drop > 0:
            feats = [f for f in feats if rng.random() >= p_drop]
        info["n"] = len(feats)
        return feats, info
    key = "rec_att" if mode == "ours" else "rec_det"
    recs = []
    for i, c in enumerate(row["claims"]):
        raw = c[key]
        if rho > 0 and raw.get("check") in ("v2", "v3", "v4"):
            # relabel() would re-decide a corrupted claim with the round-1 threshold rule while real claims carry
            # verifier verdicts -> Retell could learn 'round-1 style verdict = corrupted'. Corruption must re-score
            # the verifier for the new word first (docs/CHECK_V2.md, open item for Retell).
            raise NotImplementedError("Use unified LCR records to rescore calibrated synthetic corruption")
        if override is None and rng is not None and rho > 0 and KIND[c["cls"]] != "person" and rng.random() < rho:
            cand = [a for a in c["blip_alts"] if a not in (present or set()) and KIND[a] != "person"]
            new = cand[0] if cand else corrupt.corrupt_label(c["cls"], present or set(), table or {}, rng)
            if new is not None:
                raw = pipeline.relabel(raw, new, prop, scores, [c["cls"], *c["blip_alts"]])
                info["corrupted"] += 1
        recs.append(pipeline.finalise(raw, th))
    if override is not None:
        recs = override
    feats = []
    for r in recs:
        # One evidence token per actually inspected region. No averaging away its identity.
        for token in (r.get("regions") or [r]):
            if r.get("check") == "lcr":
                # A different region lacking the claim must not contradict a claim
                # that another attended region supports. Preserve local scores and
                # use the combined claim verdict as the caption-level instruction.
                token = dict(token, local_verdict=token["verdict"], verdict=r["verdict"])
                ts = token["thresholds"]["support"]
                # raw-logit records gate the alternative on its own class threshold inside RegionChecker
                if token.get("score_scale") != "raw_logit" and (ts is None or token["s_alt"] < ts):
                    token["alt"] = ""   # a candidate below presence threshold is no replacement evidence
            if rng is not None and p_drop > 0 and rng.random() < p_drop:
                continue
            emb = prop["embeds"][token["region"]].astype(np.float32) if token["region"] >= 0 else None
            feats.append(retell.record_features(token, emb))
            info["verdicts"].append(token["verdict"])
    info["n"] = len(feats)
    return feats, info


def false_evidence(recs: list[dict], absent: list[str], rng: np.random.Generator) -> list[dict]:
    """H4 false-evidence condition: the alternative LABEL of each contradicted record is
    swapped for a class that is absent from the image; scores, region and verdict are kept."""
    out = []
    for r in recs:
        r = dict(r)
        if r["verdict"] == "contradicted" and absent:
            r["alt"] = str(rng.choice(absent))
        if "regions" in r and r["verdict"] == "contradicted":
            r["regions"] = false_evidence(r["regions"], absent, rng)
        out.append(r)
    return out


__all__ = ["features", "records_for", "false_evidence", "image_tags", "MODES", "CLASS_INDEX", "check"]
