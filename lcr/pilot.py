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


def has_referent(entry) -> bool:
    """The annotator says the word points at a real object: correct, or misidentified (there, but named wrong)."""
    return entry.get("label") == "correct" or entry.get("error_type") == "misidentified"


def final_box(entries, adjudicator=None, min_iou=BOX_AGREE_IOU):
    """(gold referent box or None, source, mean pairwise annotator IoU or None) for a claim with a referent.

    Only boxes from annotators who say the word has a referent count (has_referent); a box left behind under
    "invented" is ignored.

    Agreement: two or more annotators drew boxes and every pair overlaps with IoU >= min_iou -> their
    coordinate-wise mean. Otherwise the adjudicator's box; otherwise a single annotator's box if only one
    was drawn; otherwise None, i.e. the claim needs adjudication before it can enter the gold set.
    """
    boxes = [b for e in entries if has_referent(e) and (b := parse_box(e.get("box")))]
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


def final_answer(entries, adjudicator, field) -> tuple[str, str]:
    """(answer, source) for a follow-up question such as `substitute_ok`: every annotator who labelled the claim
    gave the same non-empty answer -> that; else the adjudicator's; else 'uncertain'."""
    labelled = [e for e in entries if e.get("label")]
    values = {e.get(field) or "" for e in labelled}
    if labelled and len(values) == 1 and "" not in values:
        return values.pop(), "agreement"
    if adjudicator and adjudicator.get(field):
        return adjudicator[field], "adjudicator"
    return "uncertain", "unresolved"


# ------------------------------------------------------------------------------------------- B pilot sampling
# docs/CALISMA_GUNLUGU.md "Sıradaki adımlar" 0: a random sample for natural error rates, then every S4 contradiction
# and targeted S4/S2 disagreements. A claim belongs to the first group that takes it, in this order.
STRATA = ("random", "s4_replace", "s4_remove", "s2_contradicted_s4_keep", "s4_keep_elsewhere", "s4_keep")
CONTRADICTED = ("replace", "remove")


def pilot_pool(s4_rows, s2_rows):
    """Every non-person B claim with its S4 and S2 decision. Both files come from the same drafts and claim order."""
    s2 = {r["id"]: r for r in s2_rows}
    pool = []
    for r in s4_rows:
        o = s2[r["id"]]
        if o["draft"] != r["draft"] or [c["cls"] for c in o["claims"]] != [c["cls"] for c in r["claims"]]:
            raise ValueError(f"S4 and S2 records disagree on {r['id']}")
        for ci, (c, d) in enumerate(zip(r["claims"], o["claims"])):
            if c["kind"] == "person":
                continue
            a4 = c["rec_att"]
            pool.append(dict(id=r["id"], claim_index=ci, cls=c["cls"], kind=c["kind"], word=c["word"],
                             s4_action=a4.get("action"), s2_action=d["rec_att"].get("action"),
                             substitute=a4.get("substitute") or "", look_region=a4["region"], look_box=a4["box"]))
    return pool


def pilot_sample(pool, rng, n_random=60, n_s2_keep=80, n_elsewhere=60, n_keep=80):
    """Stratified pilot sample; returns (claims, population) where population[g] = (claims in group, sampled).

    `random` is a uniform draw from the whole pool (the only group that estimates natural rates). The targeted
    groups are drawn from what is left; s2_contradicted_s4_keep takes clothing claims first (Fable: S2's clothing
    contradictions are mostly wrong, so these are the disputed cases).
    """
    from .region_evidence import GROUP
    order = list(rng.permutation(len(pool)))
    take = set(order[:min(n_random, len(pool))])
    groups = {"random": sorted(take)}
    rest = [i for i in order if i not in take]
    members = dict(
        s4_replace=[i for i in rest if pool[i]["s4_action"] == "replace"],
        s4_remove=[i for i in rest if pool[i]["s4_action"] == "remove"],
        s2_contradicted_s4_keep=[i for i in rest if pool[i]["s4_action"] == "keep" and pool[i]["s2_action"] in CONTRADICTED],
        s4_keep_elsewhere=[i for i in rest if pool[i]["s4_action"] == "keep_elsewhere"],
    )
    used = set().union(*members.values())
    members["s4_keep"] = [i for i in rest if pool[i]["s4_action"] == "keep" and i not in used]
    clothing_first = sorted(members["s2_contradicted_s4_keep"], key=lambda i: GROUP[pool[i]["cls"]] != "clothing")
    caps = dict(s4_replace=None, s4_remove=None, s2_contradicted_s4_keep=n_s2_keep, s4_keep_elsewhere=n_elsewhere,
                s4_keep=n_keep)
    for g in STRATA[1:]:
        m = clothing_first if g == "s2_contradicted_s4_keep" else members[g]
        groups[g] = m if caps[g] is None else m[:caps[g]]
    population = dict(random=(len(pool), len(groups["random"])),
                      **{g: (len(members[g]), len(groups[g])) for g in STRATA[1:]})
    claims = [dict(pool[i], stratum=g) for g in STRATA for i in groups[g]]
    return claims, population


def summarise_strata(claims, answers):
    """Per-group results. `answers[(id, claim_index)]` = dict(label, error_type, substitute_ok) after adjudication.

    error_rate: incorrect / (correct + incorrect) -- for the S4 contradiction groups this is S4's precision, for
    the keep groups the share of errors S4 let through. removal_harm: a remove deleted a word that has a referent
    (correct, or misidentified = there but named wrong). substitute_right: the proposed substitute names the object.
    """
    from .metrics import wilson as _wilson

    def wilson(k, n):                                  # clamp float noise (-2.8e-17) to [0, 1]
        lo, hi = _wilson(k, n)
        return max(0., lo), min(1., hi)
    out = {}
    for g in STRATA:
        rows = [answers.get((c["id"], c["claim_index"])) or {} for c in claims if c["stratum"] == g]
        lab = [r.get("label") for r in rows]
        k_inc, n_dec = lab.count("incorrect"), lab.count("incorrect") + lab.count("correct")
        res = dict(n=len(rows), n_decided=n_dec, n_uncertain=lab.count("uncertain"), n_not_object=lab.count("na"),
                   error_rate=k_inc / n_dec if n_dec else None, error_rate_ci95=list(wilson(k_inc, n_dec)) if n_dec else None,
                   error_types={t: sum(r.get("error_type") == t for r in rows if r.get("label") == "incorrect")
                                for t in ("misidentified", "invented", "uncertain")})
        if g == "s4_remove":
            harm = [r.get("label") == "correct" or r.get("error_type") == "misidentified" for r in rows
                    if r.get("label") in ("correct", "incorrect") and r.get("error_type") != "uncertain"]
            res.update(removal_harm=sum(harm) / len(harm) if harm else None, n_removal_judged=len(harm))
        if g == "s4_replace":
            sub = [r.get("substitute_ok") for r in rows if r.get("substitute_ok") in ("yes", "no")]
            res.update(substitute_right=sub.count("yes") / len(sub) if sub else None, n_substitute_judged=len(sub),
                       substitute_right_ci95=list(wilson(sub.count("yes"), len(sub))) if sub else None)
        out[g] = res
    return out


# ------------------------------------------------------------- P6: detector box vs Look box vs human box
def proposal_pool(prop, rule):
    """Proposals any box source may use: the geometric pool of region_evidence.candidates (no attention filter)."""
    boxes = prop["boxes"].astype(np.float32)
    wh = boxes[:, 2:] - boxes[:, :2]
    ok = np.isfinite(boxes).all(1) & (wh > 0).all(1)
    if rule["min_patches"]:
        from . import config
        ok &= (wh >= rule["min_patches"] / config.BLIP_GRID).all(1)
    return ok


def matching_proposal(boxes, pool, human_box):
    """(index, IoU) of the pool proposal that best matches the human box, or (None, 0.)."""
    if not pool.any():
        return None, 0.
    ious = np.array([box_iou(b, human_box) if ok else -1. for b, ok in zip(boxes.tolist(), pool)])
    j = int(ious.argmax())
    return j, float(ious[j])


def box_evidence(z, v, tau, rival, p, claim_k, true_k=None):
    """What the frozen detector says on proposal p, by S4's own rules.

    z: (n, C) raw logits; v: (n, C) rival-aware evidence; tau: (C,) per-class thresholds (nan = none);
    rival: (C, C) bool, rival[c, j] = j is in c's exclusive set. supports_claim: v_p(c) >= tau_c (S4's keep test on
    this box). For a misidentified claim with a known true class t: claim_rejected (v_p(c) < tau_c),
    true_name_found (l_p(t) >= tau_t and l_p(t) > l_p(c), S4's replace test) and true_is_rival (S4 could only
    propose t if it is in c's set).
    """
    th = lambda j: tau[j] if np.isfinite(tau[j]) else np.inf
    out = dict(supports_claim=bool(v[p, claim_k] >= th(claim_k)))
    if true_k is not None and true_k != claim_k:
        out.update(claim_rejected=bool(v[p, claim_k] < th(claim_k)), true_is_rival=bool(rival[claim_k, true_k]),
                   true_name_found=bool(z[p, true_k] >= th(true_k) and z[p, true_k] > z[p, claim_k]))
    return out
