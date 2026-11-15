"""Independent Check v2 review; no training artefacts or existing results are overwritten.

    OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/review_check_v2.py

Reproduces stored probabilities, checks token positions and gradients, audits lambda
selection, and computes paired image-cluster bootstrap intervals for frozen systems.
Reads only train halves A/B and validation, never the test split or pilot answer key.
"""
import argparse
from collections import Counter
from itertools import combinations, product
from unittest.mock import patch

import numpy as np
import pyarrow.parquet as pq
from scipy.optimize import check_grad
from transformers import AutoTokenizer

import _common  # noqa: F401
from lcr import cache, calibrate, claims, config, data, pipeline
from lcr.captioner import Captioner
from lcr.vocab import CLASS_INDEX

GROUPS = dict(prior=["cooc_mean", "cooc_max", "n_claims"],
              det=["s_claim", "objectness", "s_image"],
              look=["excess", "att_entropy", "iou_att_det"],
              cap=["lp", "entropy", "margin", "vis_gain", "rel_pos"])
P = config.CACHE / "pipeline"
GRID = list(product((0.3, 1.0, 3.0), (0.1, 1.0, 10.0)))


def references():
    # Caption columns only: neither image bytes nor any test parquet is opened.
    out = {}
    for split in ("train", "validation"):
        rows = [r for path in sorted((config.DATA / "flickr8k/data").glob(f"{split}-*.parquet"))
                for r in pq.read_table(path, columns=data.CAP_COLS).to_pylist()]
        out.update({data.image_id(split, i): claims.reference_classes([r[k] for k in data.CAP_COLS])
                    for i, r in enumerate(rows)})
    return out


def feature_data(rows, sig, owl, refs, a_ids):
    cnt, pair = Counter(), Counter()
    for iid in a_ids:
        cnt.update(refs[iid])
        pair.update(frozenset(v) for v in combinations(sorted(refs[iid]), 2))

    def pmi(x, y):
        return np.log((pair[frozenset((x, y))] + 0.5) * len(a_ids) / ((cnt[x] + 1) * (cnt[y] + 1)))

    out = []
    for r in rows:
        prop = owl[data.parse_id(r["id"])[1]]
        others = [c["cls"] for c in r["claims"]]
        for c in r["claims"]:
            rec = c["rec_att"]
            reg = rec["region"]
            cooc = [pmi(c["cls"], k) for k in others if k != c["cls"]]
            feat = dict(cooc_mean=np.mean(cooc) if cooc else 0., cooc_max=max(cooc) if cooc else 0.,
                        n_claims=len(others), s_claim=float(calibrate.logit(rec["s_claim"])),
                        objectness=float(prop["objectness"][reg]) if reg >= 0 else 0.,
                        s_image=float(calibrate.logit(rec["s_image"])), excess=rec["excess"],
                        att_entropy=c["att_entropy"], iou_att_det=c["iou_att_det"], **sig[r["id"]][c["cls"]])
            out.append(dict(id=r["id"], c=CLASS_INDEX[c["cls"]], ok=c["in_refs"],
                            thing_stuff=c["kind"] != "person", feat=feat))
    return out


def arrays(xs, names):
    return (np.array([[x["feat"][n] for n in names] for x in xs]),
            np.array([x["c"] for x in xs]), np.array([x["ok"] for x in xs], bool),
            np.array([calibrate.fold_of(x["id"]) for x in xs]))


def feature_oof(X, c, y, folds, lc, lw, use_class=True):
    out = np.empty(len(y))
    for f in np.unique(folds):
        m = folds == f
        out[m] = calibrate.FeatureLogistic(lc, lw, use_class).fit(X[~m], c[~m], y[~m])(X[m], c[m])
    return out


def gradient_audit():
    rng = np.random.default_rng(415)
    X, c, y = rng.normal(size=(60, 4)), rng.integers(0, 4, 60), rng.integers(0, 2, 60)
    errors = {}
    real = calibrate.minimize

    def inspect(fun, initial, **kwargs):
        w = rng.normal(0, .2, len(initial))
        errors[label] = float(check_grad(lambda v: fun(v)[0], lambda v: fun(v)[1], w))
        return real(fun, initial, **kwargs)

    with patch.object(calibrate, "minimize", inspect):
        for slope in (False, True):
            label = f"class_logistic_slope_{slope}"
            calibrate.ClassLogistic(.7, slope).fit(1 / (1 + np.exp(-X[:, 0])), c, y)
        for use_class in (False, True):
            label = f"feature_logistic_class_{use_class}"
            calibrate.FeatureLogistic(.7, .4, use_class).fit(X, c, y)
    return errors


def operating_point(p, ok, target=.5, sup_target=.95):
    """Same strict/inclusive threshold convention as script 22, grouping ties."""
    order = np.argsort(p, kind="stable")
    ps, ys = p[order], ok[order]
    starts = np.r_[0, np.flatnonzero(np.diff(ps)) + 1]
    correct = np.r_[0, np.cumsum(ys)]
    n_con, n_sup = starts, len(p) - starts
    valid_c = (n_con >= 20) & ((n_con - correct[starts]) / np.maximum(n_con, 1) >= target)
    valid_s = (n_sup >= 20) & ((correct[-1] - correct[starts]) / np.maximum(n_sup, 1) >= sup_target)
    tc = float(ps[starts[valid_c][-1]]) if valid_c.any() else -np.inf
    ts = float(ps[starts[valid_s][0]]) if valid_s.any() else np.inf
    return tc, max(tc, ts)


def nested_audit(xs):
    """Fixed feature set; both lambda selection and threshold choice exclude the outer image fold.

    The old family-selection history cannot be erased by this audit. These are
    diagnostic B scores, not replacement Retell training artefacts.
    """
    names = GROUPS["prior"] + GROUPS["det"] + GROUPS["cap"]
    tuning = GROUPS["prior"] + GROUPS["det"] + GROUPS["look"] + GROUPS["cap"]
    X, c, y, folds = arrays(xs, names)
    Xt, _, _, _ = arrays(xs, tuning)
    mask = np.array([x["thing_stuff"] for x in xs])
    p, v, records = np.empty(len(y)), np.full(len(y), "uncertain", dtype="<U12"), []
    for f in np.unique(folds):
        train, held = folds != f, folds == f
        losses = [(calibrate.log_loss(feature_oof(Xt[train], c[train], y[train], folds[train], lc, lw), y[train]), lc, lw)
                  for lc, lw in GRID]
        _, lc, lw = min(losses)
        inner = feature_oof(X[train], c[train], y[train], folds[train], lc, lw)
        tc, ts = operating_point(inner[mask[train]], y[train][mask[train]])
        pp = calibrate.FeatureLogistic(lc, lw).fit(X[train], c[train], y[train])(X[held], c[held])
        p[held] = pp
        v[held] = np.where(pp >= ts, "supported", np.where(pp < tc, "contradicted", "uncertain"))
        records.append(dict(held_fold=int(f), lam_c=lc, lam_w=lw, tc=tc, ts=ts))
        print(f"nested outer fold {f} complete", flush=True)
    return dict(folds=records, auroc=calibrate.auroc(p[mask], y[mask]),
                log_loss=calibrate.log_loss(p[mask], y[mask]), ece=calibrate.ece(p[mask], y[mask]),
                **calibrate.verdict_quality(v[mask], y[mask]),
                caveat="Fixed previously explored feature family; excludes outer labels in lambda and threshold selection; not a fresh generalisation estimate.")


def alignment_audit(rows, signals):
    cap = Captioner.__new__(Captioner)
    cap.tok = AutoTokenizer.from_pretrained(config.BLIP_NAME, local_files_only=True)
    cap.bos = 30522
    bad, n, multi = [], 0, 0
    for r in rows:
        ids, mask = cap.encode_captions([r["draft"]])
        ws, pos = cap.word_alignment(ids[0])
        found = {c.cls: (pos[c.index], c.word) for c in claims.extract_from_words(ws)}
        for c in r["claims"]:
            n += 1
            p, word = found.get(c["cls"], (-1, ""))
            expected_rel = p / (int(mask[0].sum()) - 1)
            actual_rel = signals[r["id"]].get(c["cls"], {}).get("rel_pos", -1)
            if (p, word) != (c["pos"], c["word"]) or not np.isclose(expected_rel, actual_rel):
                bad.append(dict(id=r["id"], cls=c["cls"], stored_pos=c["pos"], new_pos=p))
            toks = cap.tok.convert_ids_to_tokens(ids[0].tolist())
            multi += p >= 0 and p + 1 < len(toks) and toks[p + 1].startswith("##")
    return dict(n=n, mismatches=bad, multi_piece_claims=int(multi),
                limitation="Checks stored position, surface word and relative position; does not re-run BLIP logits.")


def bootstrap(ids, y, scores, verdicts, n_boot):
    unique, inv = np.unique(ids, return_inverse=True)
    groups = [np.flatnonzero(inv == i) for i in range(len(unique))]
    def metrics(idx):
        out = {f"auc:{name}": calibrate.auroc(p[idx], y[idx]) for name, p in scores.items()}
        for name, v in verdicts.items():
            q = calibrate.verdict_quality(v[idx], y[idx])
            for k in ("contradicted_precision", "recall_wrong", "supported_precision"):
                out[f"{k}:{name}"] = q[k]
        for left, right in (("prior+det", "prior"), ("prior+det+cap", "prior+det"),
                            ("prior+det+look", "prior+det"), ("prior+region", "prior+detector_region")):
            out[f"delta_auc:{left}-{right}"] = out[f"auc:{left}"] - out[f"auc:{right}"]
        for k in ("contradicted_precision", "recall_wrong", "supported_precision"):
            out[f"delta_{k}:v2-round1"] = out[f"{k}:v2"] - out[f"{k}:round1"]
        return out
    estimate = metrics(np.arange(len(y)))
    samples = {k: [] for k in estimate}
    rng = np.random.default_rng(415)
    for _ in range(n_boot):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        for k, v in metrics(idx).items():
            samples[k].append(v)
    return {k: dict(estimate=v, low=float(np.nanquantile(samples[k], .025)),
                    high=float(np.nanquantile(samples[k], .975))) for k, v in estimate.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bootstrap", type=int, default=2000)
    args = ap.parse_args()
    refs = references()
    raw = {s: cache.load_json(P / f"A_{s}_look2.json") for s in ("B", "validation")}
    sig = {s: cache.load_json(P / f"A_{s}_look2_capsig.json") for s in raw}
    # Reconstruct A/B from protocol without loading the image column.
    n_train = sum(i.startswith("train_") for i in refs)
    perm = np.random.default_rng(config.SPLIT_SEED).permutation(n_train)
    a_ids = {data.image_id("train", int(i)) for i in perm[:n_train // 2]}
    b_ids = {data.image_id("train", int(i)) for i in perm[n_train // 2:]}
    observed_b = {r["id"] for r in raw["B"]}
    assert observed_b == b_ids and not (observed_b & a_ids)
    out = dict(protocol=dict(n_A=len(a_ids), n_B=len(b_ids), n_validation=len(raw["validation"]),
                             A_B_disjoint=True, test_read=False, label="automatic reference proxy"),
               gradients=gradient_audit(),
               alignment={s: alignment_audit(raw[s], sig[s]) for s in raw})
    print("alignment/gradient checks complete", flush=True)
    features = {s: feature_data(raw[s], sig[s], cache.OwlCache("train" if s == "B" else "validation"), refs, a_ids)
                for s in raw}
    original = cache.load_json(config.RESULTS / "check_verifier.json")
    masks = {s: np.array([x["thing_stuff"] for x in xs]) for s, xs in features.items()}
    bv_scores, errors = {}, {}
    for old in original["results"]["rec_att"]["configs"]:
        name, names, use_class = old["config"], old["features"], old["use_class"]
        XB, cB, yB, fB = arrays(features["B"], names)
        XV, cV, yV, _ = arrays(features["validation"], names)
        lc, lw = original["results"]["rec_att"]["lam_c"], original["results"]["rec_att"]["lam_w"]
        pb = feature_oof(XB, cB, yB, fB, lc, lw, use_class)
        pv = calibrate.FeatureLogistic(lc, lw, use_class).fit(XB, cB, yB)(XV, cV)
        bv_scores[name] = (pb, pv)
        errors[name] = {s: calibrate.auroc(p[masks[s]], y[masks[s]]) - old[split]["auroc"]
                        for s, split, p, y in (("B", "B", pb, yB), ("validation", "val", pv, yV))}
    out["reproduction_auroc_delta"] = errors
    out["corrected_AP_wrong"] = {
        name: {s: calibrate.ap_wrong(p[masks[s]], np.array([x["ok"] for x in features[s]])[masks[s]])
               for s, p in zip(("B", "validation"), scores)} for name, scores in bv_scores.items()}
    print("frozen feature models reproduced", flush=True)
    out["nested_B_diagnostic"] = nested_audit(features["B"])
    # Audit every lambda rather than trusting a repeated best value.
    out["class_logistic_lambda_losses"] = {}
    for key in ("rec_att", "rec_det"):
        cc = [c for r in raw["B"] for c in r["claims"]]
        s = np.array([c[key]["s_claim"] for c in cc])
        c = np.array([CLASS_INDEX[c["cls"]] for c in cc])
        y = np.array([c["in_refs"] for c in cc])
        f = np.array([calibrate.fold_of(r["id"]) for r in raw["B"] for _ in r["claims"]])
        out["class_logistic_lambda_losses"][key] = {
            transform: {str(lam): calibrate.log_loss(calibrate.cross_fit(transform, s, c, y, f, lam), y)
                        for lam in (.1, .3, 1., 3., 10.)} for transform in ("offset", "offset_slope")}
    val = raw["validation"]
    learned = cache.load_json(P / "A_validation_look2_check2.json")
    assert [r["id"] for r in learned] == [r["id"] for r in val]
    flat = [c for r in val for c in r["claims"]]
    learned_flat = [c for r in learned for c in r["claims"]]
    assert all(a["cls"] == b["cls"] for a, b in zip(flat, learned_flat))
    m = masks["validation"]
    y = np.array([c["in_refs"] for c in flat], bool)[m]
    ids = np.array([r["id"] for r in val for _ in r["claims"]])[m]
    th = cache.load_json(config.RESULTS / "thresholds_proxy_att_look2.json")
    v1 = np.array([pipeline.finalise(c["rec_att"], th)["verdict"] for c in flat])[m]
    v2 = np.array([c["rec_att"]["verdict"] for c in learned_flat])[m]
    stored_p = np.array([c["rec_att"]["p_correct"] for c in learned_flat])
    out["stored_probability_max_abs_error"] = float(np.max(np.abs(stored_p - bv_scores["prior+det+cap"][1])))
    # H6 detector-region scores from the same frozen B-fitted calibration setup.
    names = GROUPS["prior"] + ["s_claim", "objectness"]
    det_feats = {}
    for split in raw:
        xs = [dict(x, feat=dict(x["feat"])) for x in features[split]]
        owl = cache.OwlCache("train" if split == "B" else "validation")
        j = 0
        for r in raw[split]:
            prop = owl[data.parse_id(r["id"])[1]]
            for c in r["claims"]:
                rec = c["rec_det"]
                xs[j]["feat"].update(s_claim=float(calibrate.logit(rec["s_claim"])),
                                     objectness=float(prop["objectness"][rec["region"]]))
                j += 1
        det_feats[split] = xs
    xb, cb, yb, _ = arrays(det_feats["B"], names)
    xv, cv, _, _ = arrays(det_feats["validation"], names)
    dp = calibrate.FeatureLogistic(lc, lw).fit(xb, cb, yb)(xv, cv)
    scores = {k: v[1][m] for k, v in bv_scores.items()}
    scores["prior+detector_region"] = dp[m]
    out["paired_bootstrap"] = dict(n_boot=args.bootstrap, seed=415, unit="image with >=1 thing/stuff claim",
                                   frozen_systems=True, intervals=bootstrap(ids, y, scores, dict(round1=v1, v2=v2), args.bootstrap))
    out["operating_points"] = next(c["val"] for c in original["results"]["rec_att"]["configs"]
                                  if c["config"] == "prior+det+cap")
    out["caption_signal_correlations_B"] = {
        k: float(np.corrcoef([x["feat"]["lp"] for x in features["B"]],
                             [x["feat"][k] for x in features["B"]])[0, 1])
        for k in ("vis_gain", "lp_null", "margin", "entropy")}
    # Alternative quality uses the actual stored suggestion, not a candidate oracle.
    for label, subset in (("all_reference_unsupported", ~y), ("contradicted_reference_unsupported", (~y) & (v2 == "contradicted"))):
        selected = [(r, c) for r in learned for c in r["claims"] if c["kind"] != "person"]
        items = [item for item, keep in zip(selected, subset) if keep]
        has = [(r, c) for r, c in items if c["rec_att"]["alt"]]
        out.setdefault("alternative_quality", {})[label] = dict(n=len(items), with_suggestion=len(has),
            in_refs=sum(c["rec_att"]["alt"] in refs[r["id"]] for r, c in has),
            limitation="Reference occurrence does not establish a locally correct replacement.")
    path = config.RESULTS / "check_v2_review.json"
    cache.save_json(out, path)
    write_metrics(out)
    print(path, flush=True)
    for k, v in out["paired_bootstrap"]["intervals"].items():
        if k.startswith("delta"):
            print(k, v, flush=True)


def write_metrics(out):
    """Keep the review's displayed numbers generated from the saved audit."""
    rows = ["# Check v2 — otomatik inceleme ölçümleri", "",
            "Üreten: `scripts/review_check_v2.py`; kaynak: `results/check_v2_review.json`.", "",
            "Validation; yalnızca thing/stuff; otomatik referans proxy'si. "
            "Eşleştirilmiş görüntü bootstrap'ı, %95 yüzdelik aralık. Modeller ve eşikler sabit; "
            "eğitim/seçim belirsizliği ve çoklu karşılaştırma düzeltmesi dahil değil.", "",
            "| Ölçüm | Tahmin | %95 GA |", "|---|---:|---:|"]
    for k, v in out["paired_bootstrap"]["intervals"].items():
        if k.startswith("delta") or k in ("auc:prior", "auc:prior+det", "auc:prior+det+cap"):
            rows.append(f"| {k} | {v['estimate']:.4f} | [{v['low']:.4f}, {v['high']:.4f}] |")
    rows += ["", "## B'de iç içe seçim tanısı", "",
             "Sabit özellik ailesi. Lambda ve eşikler dışarıda tutulan görüntü katını görmez; "
             "önceki özellik ailesi seçiminin geçmişini ortadan kaldırmaz.", "",
             "| Ölçüm | Değer |", "|---|---:|"]
    for k in ("auroc", "log_loss", "ece", "contradicted_precision", "recall_wrong", "supported_precision"):
        rows.append(f"| {k} | {out['nested_B_diagnostic'][k]:.6f} |")
    rows += ["", "## Doğrulamalar", "",
             f"- Olasılık yeniden üretimi, en büyük mutlak fark: {out['stored_probability_max_abs_error']:.3g}."]
    for s, a in out["alignment"].items():
        rows.append(f"- {s}: {a['n']} claim, {len(a['mismatches'])} konum/kelime uyumsuzluğu, {a['multi_piece_claims']} çok-parçalı claim.")
    rows += [f"- Gradyan sonlu fark kontrolünde en büyük hata: {max(out['gradients'].values()):.3g}.", "",
             "## Alternatif önerisinin referans proxy'si", "",
             "| Küme | Claim | Öneri var | Öneri referansta | Oran |", "|---|---:|---:|---:|---:|"]
    for name, r in out["alternative_quality"].items():
        rows.append(f"| {name} | {r['n']} | {r['with_suggestion']} | {r['in_refs']} | {r['in_refs']/r['n']:.3%} |")
    rows += ["", "## Eşit skorları gruplayan AP (yanlış), validation", "",
             "| Özellikler | AP |", "|---|---:|"]
    for name, r in out["corrected_AP_wrong"].items():
        rows.append(f"| {name} | {r['validation']:.6f} |")
    path = config.ROOT / "docs/generated/CHECK_V2_REVIEW_METRICS.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows) + "\n")


if __name__ == "__main__":
    main()
