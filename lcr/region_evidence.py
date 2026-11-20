"""One attention -> frozen OWLv2 -> evidence implementation for live/cache/corruption.

No caption probabilities, language priors, reference labels, or second verifier at
inference. Labels are used only by the explicitly invoked B calibration routine.
"""
from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.special import expit

from . import config, look
from .vocab import CLASS_INDEX, CLASSES, KIND, THINGS

METHOD = "lcr"
# Default: Check S4 (rival-aware evidence, cost-sensitive thresholds; docs/CHECK_S4_PROTOKOL.md, docs/CHECK_S4.md).
BUNDLE_PATH = config.RESULTS / "lcr_check4_bundle.json"
HISTORICAL_BUNDLE_PATH = config.RESULTS / "lcr_bundle.json"     # Codex's first unified bundle (B_cal thresholds)
CHECK3_BUNDLE_PATH = config.RESULTS / "lcr_check3_bundle.json"   # positive-anchored Check (docs/CHECK_DUZELTME_PROTOKOL.md)
FIVE_WAY_BUNDLE_PATH = config.RESULTS / "lcr_check_bundle.json"  # five-way z-scale Check (docs/CHECK_TAMAMLAMA_PROTOKOL.md)
CHECK4_BUNDLE_PATH = config.RESULTS / "lcr_check4_bundle.json"   # S4 rival-aware, cost-sensitive (docs/CHECK_S4_PROTOKOL.md)
ACTIONS = ("keep", "keep_elsewhere", "replace", "remove", "skip")
ACTION_VERDICT = dict(keep="supported", keep_elsewhere="uncertain", replace="contradicted", remove="contradicted",
                      skip="uncertain")
_STARTS = [("animal", "dog"), ("vehicle", "bicycle"), ("play", "ball"), ("clothing", "shirt"),
           ("animal_gear", "collar"), ("handheld", "camera"), ("structure", "fence"), ("plant", "tree")]
GROUP = {}
_names = list(THINGS)
for i, (group, start) in enumerate(_STARTS):
    end = _names.index(_STARTS[i + 1][1]) if i + 1 < len(_STARTS) else len(_names)
    GROUP.update({n: group for n in _names[_names.index(start):end]})
for _c in CLASSES:
    GROUP.setdefault(_c, KIND[_c])

# Mutually exclusive confusion sets for `replace` (docs/CHECK_DUZELTME_PROTOKOL.md, C3). Hand-made from the vocabulary,
# not tuned on data: one object cannot correctly carry two names of the same set. Outside a set there is no replace.
CONFUSION = dict(
    upper="shirt jacket sweater vest", full_body="dress suit uniform costume swimsuit bikini", lower="pants shorts skirt",
    head="hat helmet headband bandanna", eyes="glasses goggles", feet="shoe boot", carried="backpack bag",
    animal="dog horse bird duck cat cow sheep goat fish camel elephant", two_wheel="bicycle motorcycle scooter unicycle",
    road="car truck bus train", boat="boat canoe kayak raft", board="skateboard snowboard surfboard ski sled",
    thrown="ball frisbee stick bone balloon", striker="racket bat", play_structure="swing slide trampoline ramp",
    seat="bench chair couch bed", container="cup bottle bucket", reading="book newspaper", barrier="fence wall railing",
    plant="tree bush flower", water_body="ocean lake river pond stream puddle", ground="grass sand snow dirt mud ice",
    road_surface="street sidewalk")
CONFUSION = {k: v.split() for k, v in CONFUSION.items()}
CONFUSION_OF = {c: k for k, names in CONFUSION.items() for c in names}
assert all(c in CLASS_INDEX for c in CONFUSION_OF) and len(CONFUSION_OF) == sum(map(len, CONFUSION.values()))


@lru_cache(maxsize=None)
def _confusion_mask(claim):
    s = CONFUSION_OF.get(claim)
    m = np.array([c != claim and s is not None and CONFUSION_OF.get(c) == s for c in CLASSES])
    m.setflags(write=False)
    return m


def confusion_mask(claim):
    return _confusion_mask(claim)


# S4 (docs/CHECK_S4_PROTOKOL.md, I5): hypernyms and overlapping names removed from the exclusive sets.
CONFUSION_CLEAN = {k: [c for c in v if c not in {"vest", "swimsuit", "bikini", "boat"}] for k, v in CONFUSION.items()}
CONFUSION_CLEAN = {k: v for k, v in CONFUSION_CLEAN.items() if len(v) > 1}


def set_matrix(confusion):
    """(C, C) bool: M[k, j] = j is a mutually exclusive rival of k."""
    of = {c: k for k, names in confusion.items() for c in names}
    return np.array([[a != b and of.get(a) is not None and of.get(a) == of.get(b) for b in CLASSES] for a in CLASSES])


def rival_matrix(lg, M, weight):
    """Rival-aware evidence per box: v_c(b) = l_c(b) - weight * max(0, max_{j in S(c)} l_j(b) - l_c(b)).
    Classes without rivals keep their raw logit. lg: (n_boxes, C) raw OWLv2 logits."""
    lg = np.asarray(lg, np.float64)
    if not len(lg):
        return lg
    riv = np.where(M[None], lg[:, None, :], -np.inf).max(2)          # (n, C) best rival of each class at each box
    return lg - weight * np.maximum(np.where(np.isfinite(riv), riv - lg, 0.), 0.)


def fit_rival_np(X, mentioned, count, lam, m0=10, strict_min=5):
    """Cost-sensitive per-class threshold (docs/CHECK_S4_PROTOKOL.md, I1): tau_c = argmax F_u(tau) - lam * F_p(tau).
    F_p: image evidence of images mentioned by >= 2 references (any mention if fewer than strict_min such images),
    smoothed with the kind's pooled centred shape (weight m0). F_u: images whose references do not mention c (an
    automatic proxy for absence). One lam for all classes; classes with < 2 positives get NaN (never contradicted)."""
    X, men, cnt = np.asarray(X, np.float64), np.asarray(mentioned, bool), np.asarray(count)
    tau = np.full(len(CLASSES), np.nan); P = {}
    for j in range(len(CLASSES)):
        strict = cnt[:, j] >= 2
        P[j] = np.sort(X[strict if strict.sum() >= strict_min else men[:, j], j])
    for kind in ("thing", "stuff"):
        cols = [j for j, c in enumerate(CLASSES) if KIND[c] == kind]
        G = np.sort(np.concatenate([P[j] - np.median(P[j]) for j in cols if len(P[j]) >= 20]))
        for j in cols:
            p, u = P[j], np.sort(X[~men[:, j], j])
            if len(p) < 2 or not len(u):
                continue
            cand = np.unique(np.r_[p, np.quantile(u, np.linspace(.5, .999, 200))])
            fp = (np.searchsorted(p, cand, side="left") + m0 * np.searchsorted(G, cand - np.median(p), side="left") / len(G)) / (len(p) + m0)
            fn = np.searchsorted(u, cand, side="left") / len(u)
            tau[j] = cand[np.argmax(fn - lam * fp)]
    return tau


def substitute_centred(top1_logits, k, tau, centre, M):
    """S4 substitute (I3): exclusive rival on the Look top-1 box, above its own tau and above the claim there,
    ranked by logit minus the class centre (median of its positives); else None."""
    t = np.where(np.isfinite(tau), tau, np.inf)
    ok = M[k] & (top1_logits >= t) & (top1_logits > top1_logits[k])
    return int(np.argmax(np.where(ok, top1_logits - centre, -np.inf))) if ok.any() else None


def partition(iid: str) -> str:
    """Reserved calibration images never enter Retell training."""
    h = hashlib.sha256(f"lcr-unified-cal:{iid}".encode()).hexdigest()
    return "cal" if int(h, 16) % 5 == 0 else "fit"


def proposal_logits(prop, queries):
    return ((prop["embeds"].astype(np.float32) @ queries.T +
             prop["shift"].astype(np.float32)[:, None]) * prop["scale"].astype(np.float32)[:, None])


def candidates(att, prop, kind, rule, n=3):
    """Rank spatial evidence without the claimed class or its detector score."""
    boxes = prop["boxes"].astype(np.float32)
    wh = boxes[:, 2:] - boxes[:, :2]
    valid = np.isfinite(boxes).all(1) & (wh > 0).all(1)
    if rule["min_patches"]:
        valid &= (wh >= rule["min_patches"] / config.BLIP_GRID).all(1)
    gamma = config.LOOK_RULES_BY_KIND[kind]["gamma"]
    rs = look.region_score(att, boxes, prop["objectness"].astype(np.float32),
                           alpha=rule["alpha"], gamma=gamma)
    valid &= np.isfinite(rs) & (rs > 0)
    order = np.argsort(-np.where(valid, rs, -np.inf), kind="stable")
    top = order[valid[order]][:n]
    w = rs[top].astype(np.float64)
    return top, w / w.sum() if len(w) else w


def alternative_mask(claim, mode):
    m = np.array([c != claim for c in CLASSES])
    if mode == "group":
        m &= np.array([GROUP[c] == GROUP[claim] for c in CLASSES])
    elif mode != "all":
        raise ValueError(f"Unknown alternatives: {mode}")
    if KIND[claim] == "person":
        m[:] = False
    return m


def pool_logits(logits, top, weights, pool):
    if not len(top):
        return np.full(len(CLASSES), -np.inf)
    if pool == "max":
        return logits[top].max(0)
    if pool == "weighted":
        return weights @ logits[top]
    raise ValueError(f"Unknown pooling: {pool}")


def decide(sc, sa, has_region, ts, tc):
    sc, sa, g = np.asarray(sc), np.asarray(sa), np.asarray(has_region)
    return np.where(g & (sc >= ts), "supported",
                    np.where(g & (sc < tc) & (sa >= ts), "contradicted", "uncertain"))


def decide5(sc, s_img, sa_any, sa_sub, has_region, ts, tc):
    """Completed Check (docs/CHECK_TAMAMLAMA_PROTOKOL.md). Returns (verdict, action, reason) arrays.
    supported only from the attended region; the image-level score only vetoes contradiction; contradiction is split
    into replace (same-group object present in the region) and remove (something else there, claim nowhere)."""
    sc, s_img, sa_any, sa_sub, g = map(np.asarray, (sc, s_img, sa_any, sa_sub, has_region))
    keep = g & (sc >= ts)
    elsewhere = g & ~keep & (s_img >= ts)
    con = g & ~keep & ~elsewhere & (sc < tc) & (sa_any >= ts)
    replace = con & (sa_sub >= ts)
    remove = con & ~replace
    action = np.select([keep, elsewhere, replace, remove], ["keep", "keep_elsewhere", "replace", "remove"], "skip")
    verdict = np.vectorize(ACTION_VERDICT.get, otypes=[object])(action) if action.ndim else ACTION_VERDICT[str(action)]
    reason = np.select([~g, keep, elsewhere, replace, remove],
                       ["no_region", "claim_present", "claim_elsewhere_in_image", "substitute_in_region",
                        "other_object_in_region_claim_absent"], "insufficient_region_evidence")
    return verdict, action, reason


def decide_pa(s_reg, s_img, has_sub, has_region, tau, abstain):
    """Positive-anchored Check (docs/CHECK_DUZELTME_PROTOKOL.md). Raw OWLv2 logits against the class threshold tau.
    keep: claim present where Look looked; keep_elsewhere: present elsewhere; abstaining classes are never
    contradicted; otherwise contradicted (absent everywhere), split by a confusion-set substitute on the top-1 box."""
    s_reg, s_img, has_sub, g, tau, ab = map(np.asarray, (s_reg, s_img, has_sub, has_region, tau, abstain))
    keep = g & (s_reg >= tau)
    elsewhere = g & ~keep & (s_img >= tau)
    con = g & ~keep & ~elsewhere & ~ab
    replace = con & has_sub
    remove = con & ~replace
    action = np.select([keep, elsewhere, replace, remove], ["keep", "keep_elsewhere", "replace", "remove"], "skip")
    verdict = np.vectorize(ACTION_VERDICT.get, otypes=[object])(action) if action.ndim else ACTION_VERDICT[str(action)]
    reason = np.select([~g, keep, elsewhere, replace, remove],
                       ["no_region", "claim_present", "claim_elsewhere_in_image", "substitute_in_region",
                        "claim_absent_in_image"], "abstain_class")
    return verdict, action, reason


def substitute(top1_logits, k, tau, abstain):
    """Confusion-set class on the Look top-1 box above its own threshold and above the claim there; else None."""
    ok = confusion_mask(CLASSES[k]) & ~np.asarray(abstain, bool) & (top1_logits >= tau) & (top1_logits > top1_logits[k])
    return int(np.argmax(np.where(ok, top1_logits - tau, -np.inf))) if ok.any() else None


def fit_positive_anchored(image_max, mentioned, alpha, m=20, min_n=5, abstain_auroc=.75, abstain_min=10):
    """Per-class tau from clean positives (images whose references mention the class): alpha-quantile of the class's
    image-max logit, shrunk to the kind-level pooled quantile. No 'wrong' labels are used. Abstain where image-level
    detection vs mention is weak (AUROC, an automatic proxy) or positives are few."""
    from .calibrate import auroc
    image_max, mentioned = np.asarray(image_max), np.asarray(mentioned, bool)
    tau = np.zeros(len(CLASSES)); abstain = np.zeros(len(CLASSES), bool); n_pos = mentioned.sum(0)
    for kind in ("person", "thing", "stuff"):
        cols = [j for j, c in enumerate(CLASSES) if KIND[c] == kind]
        pooled = np.concatenate([image_max[mentioned[:, j], j] for j in cols])
        q_kind = float(np.quantile(pooled, alpha)) if len(pooled) else float(np.quantile(image_max[:, cols], 1 - alpha))
        for j in cols:
            n = int(n_pos[j])
            tau[j] = q_kind if n < min_n else (n * np.quantile(image_max[mentioned[:, j], j], alpha) + m * q_kind) / (n + m)
            a = auroc(image_max[:, j], mentioned[:, j]) if 0 < n < len(mentioned) else np.nan
            abstain[j] = n < abstain_min or (abstain_auroc is not None and not a >= abstain_auroc)
    return tau, abstain


def fit_thresholds(sc, sa, ok, target=.5, min_n=20, s_img=None):
    """The original three-way rule; shared claim/alternative presence threshold. With s_img the image-level veto
    (contradiction only if s_img < tau_s) is part of the rule being fitted."""
    sc, sa, ok = np.asarray(sc), np.asarray(sa), np.asarray(ok, bool)
    si = np.full(len(sc), -np.inf) if s_img is None else np.asarray(s_img)
    g = np.isfinite(sc)
    if g.sum() < min_n:
        return np.inf, -np.inf
    grid = np.unique(np.quantile(sc[g], np.linspace(.01, .99, 99)))
    n_ok, n_wrong = max(ok.sum(), 1), max((~ok).sum(), 1)
    best = (-np.inf, np.inf, -np.inf)
    for ts in grid:
        sup = g & (sc >= ts)
        j_s = (sup & ok).sum() / n_ok - (sup & ~ok).sum() / n_wrong
        if j_s > best[0]:
            best = (j_s, ts, -np.inf)
        cand = g & (sc < ts) & (sa >= ts) & (si < ts)
        if cand.sum() < min_n:
            continue
        order = np.argsort(sc[cand])
        ss = sc[cand][order]
        cw = np.cumsum(~ok[cand][order])
        tcs = grid[grid <= ts]
        counts = np.searchsorted(ss, tcs, side="left")
        enough = counts >= min_n
        if not enough.any():
            continue
        ns = counts[enough]
        wrong = cw[ns - 1]
        j = j_s + wrong / n_wrong - (ns - wrong) / n_ok
        j = np.where(wrong / ns >= target, j, -np.inf)
        i = int(j.argmax())
        if j[i] > best[0]:
            best = (j[i], ts, tcs[enough][i])
    return float(best[1]), float(best[2])


def encode_thresholds(ts, tc):
    return dict(support=float(ts) if np.isfinite(ts) else None,
                contradiction=float(tc) if np.isfinite(tc) else None)


def decode_thresholds(th):
    return (float(th["support"]) if th["support"] is not None else np.inf,
            float(th["contradiction"]) if th["contradiction"] is not None else -np.inf)


def fingerprint(bundle):
    clean = {k: v for k, v in bundle.items() if k != "id"}
    return hashlib.sha256(json.dumps(clean, sort_keys=True, allow_nan=False).encode()).hexdigest()[:16]


@lru_cache(maxsize=4)
def _file_hash(path, mtime, size):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def file_hash(path):
    path = Path(path)
    s = path.stat()
    return _file_hash(str(path.resolve()), s.st_mtime_ns, s.st_size)


def attention_provenance():
    from .look_v4 import selection
    selected, person_rules = selection()
    return dict(heads=selected["heads"], person_rules=person_rules, rules_by_kind=config.LOOK_RULES_BY_KIND,
                correction="caption_mean_subtract_relu", caption_tokens="retokenised; BOS replaced; SEP excluded",
                decoder_sha256=file_hash(config.CACHE / "ckpt/decoder_A.pt"))


class RegionChecker:
    def __init__(self, bundle, queries):
        self.bundle = bundle
        self.queries = np.asarray(queries, np.float32)
        self.mu, self.sd = np.asarray(bundle["mu"]), np.asarray(bundle["sd"])
        if bundle["classes"] != CLASSES or self.mu.shape != (len(CLASSES),) or not np.all(self.sd > 0):
            raise ValueError("Invalid class scale in LCR bundle")
        if bundle["id"] != fingerprint(bundle):
            raise ValueError("LCR bundle fingerprint does not match its contents")
        if hashlib.sha256(self.queries.tobytes()).hexdigest() != bundle["query_sha256"]:
            raise ValueError("LCR queries differ from the calibrated queries")
        if bundle.get("decision") == "rival_np":
            self.M = set_matrix(bundle["confusion"])

    @classmethod
    def load(cls, path=None):
        path = Path(path or BUNDLE_PATH)
        bundle = json.loads(path.read_text())
        if bundle.get("attention") != attention_provenance():
            raise ValueError("Attention/decoder provenance differs from the calibrated LCR bundle")
        return cls(bundle, np.load(config.CACHE / "owl" / bundle["query_file"]))

    def record(self, claim, att, prop, *, person_record=None, logits=None, fixed_selection=None, fixed_excess=None,
               non_visual=False):
        kind = KIND[claim]
        top, weights = (candidates(att, prop, kind, self.bundle["look"])
                        if fixed_selection is None else fixed_selection)
        if kind == "person" and person_record is not None:
            reg = person_record["region"]
            top = np.array([reg], int) if reg >= 0 else np.array([], int)
            weights = np.ones(len(top))
        lg = proposal_logits(prop, self.queries) if logits is None else logits
        raw = pool_logits(lg, top, weights, self.bundle["pool"])
        row = (raw - self.mu) / self.sd
        scores = (lg - self.mu) / self.sd
        k = CLASS_INDEX[claim]
        mask = alternative_mask(claim, self.bundle["alternatives"])
        rv = self.bundle.get("decision") == "rival_np"
        pa = self.bundle.get("decision") == "positive_anchored" or rv
        if rv:
            tau_raw = np.array([np.nan if t is None else t for t in self.bundle["tau"]], np.float64)
            abstain = ~np.isfinite(tau_raw)
            tau = np.where(abstain, np.inf, tau_raw)
            ts, tc, off = (float(tau[k]) if np.isfinite(tau[k]) else np.inf), -np.inf, tau
        elif pa:
            tau, abstain = np.asarray(self.bundle["tau"], np.float64), np.array([c in self.bundle["abstain"] for c in CLASSES])
            ts, tc, off = float(tau[k]), -np.inf, tau
        else:
            ts, tc = decode_thresholds(self.bundle["thresholds"].get(kind, self.bundle["thresholds"]["thing"]))
            off = 0.
        j = int(np.argmax(np.where(mask, row - off, -np.inf))) if mask.any() and len(top) else None

        def one(vector, region, box, source, alt_index=None):
            a = alt_index if alt_index is not None else (int(np.argmax(np.where(mask, vector - off, -np.inf))) if mask.any() else None)
            if region < 0 or (pa and a is not None and vector[a] < tau[a]):
                a = None                                      # raw scale: an alternative must clear its own tau
            sa = float(vector[a]) if a is not None else -np.inf
            sc = float(vector[k])
            v = str(decide(sc, sa, region >= 0, ts, tc))
            if kind == "person":
                v = "supported" if region >= 0 and expit(lg[region, k]) >= .2 else "uncertain"
            alt = CLASSES[a] if a is not None else ""
            return dict(check=METHOD, look=METHOD, bundle_id=self.bundle["id"], score_scale="raw_logit" if pa else "class_z",
                        claim=claim, alt=alt, s_claim=sc if np.isfinite(sc) else 0.,
                        s_alt=sa if np.isfinite(sa) else 0., s_image=0.,
                        box=list(map(float, box)), region=int(region),
                        excess=(float(look.excess_mass(att, np.asarray(box)[None])[0]) if len(box) and att is not None
                                else float((fixed_excess or {}).get(region, 0.))),
                        verdict=v, reason=("claim_present" if v == "supported" else "alternative_evidence" if v == "contradicted"
                                           else "no_region" if region < 0 else "insufficient_region_evidence"),
                        by="attention", score_source=source,
                        thresholds=encode_thresholds(ts, tc))

        regions = []
        for t, w in zip(top, weights):
            box = person_record["box"] if kind == "person" and person_record is not None else prop["boxes"][t]
            r = one(scores[t], int(t), box, "owl_proposal")
            r.update(attention_weight=float(w), raw_claim_score=float(expit(lg[t, k])))
            if kind == "person" and person_record is not None:
                for field in ("box_source", "score_box", "score_source", "sam_proposal_iou"):
                    if field in person_record:
                        r[field] = person_record[field]
            regions.append(r)
        five = self.bundle.get("decision") in ("five_way", "positive_anchored", "rival_np")
        sub_mask = alternative_mask(claim, "group") & np.array([KIND[c] != "person" for c in CLASSES])
        if len(top):
            s_img = float(scores[:, k].max())
            sj = int(np.argmax(np.where(sub_mask, row, -np.inf))) if sub_mask.any() else None
            if rv and kind != "person":
                vk = rival_matrix(scores, self.M, self.bundle["rival_weight"])[:, k]
                s_img = float(vk.max()); row = row.copy(); row[k] = float(vk[top].max())   # rival-aware R_c, X_c
                sj = substitute_centred(scores[top[0]], k, tau, np.asarray(self.bundle["centre"]), self.M)
                v, action, reason = (str(x) for x in decide_pa(row[k], s_img, sj is not None, True, tau[k], abstain[k]))
                if non_visual:
                    v, action, reason = "uncertain", "skip", "non_visual_idiom"
                if j is not None and row[j] < tau[j]:
                    j = None
                alt_index = sj if action == "replace" else j
            elif pa and kind != "person":
                sj = substitute(scores[top[0]], k, tau, abstain)
                v, action, reason = (str(x) for x in decide_pa(row[k], s_img, sj is not None, True, tau[k], abstain[k]))
                if non_visual:
                    v, action, reason = "uncertain", "skip", "non_visual_idiom"
                if j is not None and row[j] < tau[j]:
                    j = None                                  # nothing in the region clears its own threshold
                alt_index = sj if action == "replace" else j
            elif five and kind != "person":
                v5, action, reason = (str(x) for x in decide5(row[k], s_img, row[j] if j is not None else -np.inf,
                                                              row[sj] if sj is not None else -np.inf, True, ts, tc))
                if non_visual:
                    v5, action, reason = "uncertain", "skip", "non_visual_idiom"
                alt_index = sj if action == "replace" else j
                v = v5
            else:
                v = str(decide(row[k], row[j] if j is not None else -np.inf, True, ts, tc))
                alt_index = j
            witness_class = alt_index if v == "contradicted" and alt_index is not None else k
            witness_scores = scores[top, witness_class]
            if rv and kind != "person" and witness_class == k:
                witness_scores = vk[top]                      # S4 decides on rival-aware evidence: show that box
            witness = int(top[np.argmax(witness_scores)])
            if pa and kind != "person" and action == "replace":
                witness = int(top[0])                         # the substitute was read on the top-1 Look box
            box = person_record["box"] if kind == "person" and person_record is not None else prop["boxes"][witness]
            result = one(row, witness, box, "pooled_owl_proposals", alt_index=alt_index)
            if pa and kind != "person" and action == "replace":
                # s_alt must belong to the box shown; the pooled top-k score is kept under its own name
                result.update(s_alt=float(scores[witness, alt_index]), s_alt_pooled=result["s_alt"])
            if five and kind != "person":
                result.update(verdict=v, action=action, reason=reason,
                              substitute=CLASSES[sj] if action == "replace" else "",
                              alt_any=CLASSES[j] if j is not None else "",
                              s_alt_any=float(row[j]) if j is not None else 0.,
                              s_substitute_best=float((scores[top[0]] if pa else row)[sj]) if sj is not None else 0.,
                              s_image_veto=s_img)
                if pa:
                    result.update(tau_claim=float(tau[k]) if np.isfinite(tau[k]) else None, abstain_class=bool(abstain[k]))
        else:
            result = one(np.full(len(CLASSES), -np.inf), -1, [], "no_region")
            if five:
                result.update(action="skip", substitute="", alt_any="", s_alt_any=0., s_substitute_best=0.,
                              s_image_veto=0.)
        if five and kind == "person":
            result.update(action="keep" if result["verdict"] == "supported" else "skip", substitute="")
        result.update(regions=regions, region_indices=top.tolist(), region_weights=weights.tolist(),
                      pooled_claim_source="max" if self.bundle["pool"] == "max" else "r_weighted",
                      image_claim_raw=float(expit(lg[:, k].max())) if len(lg) else 0.)
        if kind == "person" and person_record is not None:
            for field in ("box_source", "score_box", "score_source", "sam_proposal_iou"):
                if field in person_record:
                    result[field] = person_record[field]
        return result

    def relabel(self, rec, claim, prop):
        """Same attended candidates, new label; do not silently use the old checker."""
        if rec["bundle_id"] != self.bundle["id"]:
            raise ValueError("Corruption requires the exact checker that produced the record")
        top = np.asarray(rec["region_indices"], int)
        return self.record(claim, None, prop, non_visual=False,
                           fixed_selection=(top, np.asarray(rec["region_weights"])),
                           fixed_excess={r["region"]: r["excess"] for r in rec["regions"]})
