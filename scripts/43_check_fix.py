"""Positive-anchored Check on half B (docs/CHECK_DUZELTME_PROTOKOL.md). No validation/test.

    OMP_NUM_THREADS=2 python scripts/43_check_fix.py
S3: raw OWLv2 logits, per-class thresholds from clean positives, abstaining classes, confusion-set substitutes on the
top-1 Look box, operating point by out-of-fold FCR <= 0.05. Compared with S2 (five-way z rule, script 42) on the same
image folds by FCR and the synthetic swap test (Fable's construction). Writes results/lcr_check3_bundle.json,
results/check_fix_B.json, docs/generated/CHECK_FIX_B.md and data/cache/pipeline/A_B_lcr3.json.
"""
import collections
import hashlib
import importlib
import json
import shutil

import numpy as np

import _common  # noqa: F401
from lcr import cache, calibrate, claims, config, data
from lcr import region_evidence as re
from lcr.vocab import CLASS_INDEX, CLASSES, KIND

P = config.CACHE / "pipeline"
QFILE = "class_queries_single.npy"
ALPHAS = (0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20, 0.25)
FCR_TARGET = 0.05
s42 = importlib.import_module("42_check_complete")


def build():
    hist = json.loads(re.HISTORICAL_BUNDLE_PATH.read_text())
    rows = cache.load_json(P / "A_B_look4.json")
    assert all(r["id"].startswith("train_") for r in rows)
    maps = np.load(P / "A_B_look4.maps.npz"); owner = {tuple(o): i for i, o in enumerate(maps["owner"])}
    owl = cache.OwlCache("train")
    q = np.load(config.CACHE / "owl" / QFILE).astype(np.float32)
    image_max, xs, pooled, top1 = [], [], [], []
    for ri, row in enumerate(rows):
        prop = owl[int(row["id"].rsplit("_", 1)[1])]
        lg = re.proposal_logits(prop, q)
        image_max.append(lg.max(0))
        for ci, c in enumerate(row["claims"]):
            if c["kind"] == "person":
                continue
            top, w = re.candidates(maps["maps"][owner[(ri, ci)]], prop, c["kind"], hist["look"])
            pooled.append(re.pool_logits(lg, top, w, hist["pool"]))
            top1.append(lg[top[0]] if len(top) else np.full(len(CLASSES), -np.inf, np.float32))
            xs.append(dict(ri=ri, ci=ci, id=row["id"], cls=c["cls"], kind=c["kind"], ok=bool(c["in_refs"]),
                           fold=calibrate.fold_of(row["id"]), part=re.partition(row["id"]), has=len(top) > 0,
                           non_visual=claims.non_visual(row["draft"], c["cls"]), blip_alts=c["blip_alts"]))
        if (ri + 1) % 1000 == 0:
            print(f"B features {ri + 1}/{len(rows)}", flush=True)
    return rows, maps, owl, q, hist, np.asarray(image_max), xs, np.asarray(pooled), np.asarray(top1)


def swap_labels(xs, refs):
    """Fable's swap construction (scripts/review_fable_B/d2.py), unchanged: reference-confirmed thing claims keep
    their Look region; the label becomes a captioner alternative absent from the references, else a random
    same-GROUP class absent from the references (seed 0)."""
    rng = np.random.default_rng(0)
    by_group = {}
    for c in CLASSES:
        by_group.setdefault(re.GROUP[c], []).append(c)
    sw = []
    for x in xs:
        if not (x["ok"] and x["kind"] == "thing"):
            sw.append(-1); continue
        r = refs[x["id"]]
        cand = [a for a in x["blip_alts"] if a not in r and KIND[a] == "thing"]
        if not cand:
            cand = [c for c in by_group[re.GROUP[x["cls"]]] if c not in r and c != x["cls"]]
            cand = [cand[rng.integers(len(cand))]] if cand else []
        sw.append(CLASS_INDEX[cand[0]] if cand else -1)
    return np.array(sw)


def s3(IM, men, pooled, top1, xs, kk, alpha, train_img, test, nv, **fit):
    """S3 actions for claims `test` with class indices kk; thresholds from images train_img only."""
    tau, ab = re.fit_positive_anchored(IM[train_img], men[train_img], alpha, **fit)
    idx = np.where(test)[0]
    ri = np.array([xs[i]["ri"] for i in idx]); k = kk[idx]
    has = np.array([xs[i]["has"] for i in idx], bool)
    subs = [re.substitute(top1[i], kk[i], tau, ab) if xs[i]["has"] else None for i in idx]
    v, a, _ = re.decide_pa(pooled[idx, k], IM[ri, k], np.array([s is not None for s in subs]), has, tau[k], ab[k])
    a = np.where(nv[idx], "skip", a); v = np.where(nv[idx], "uncertain", v)
    sub = np.array([CLASSES[s] if s is not None and act == "replace" else "" for s, act in zip(subs, a)], object)
    return idx, v, a, sub, tau, ab


def s2(IM, xs, xs_eval, pooled, rows, train_img, train, test):
    """S2 (script 42 rule): thresholds fit on unswapped training claims, applied to xs_eval (possibly swapped)."""
    S = s42.scores(IM, xs, pooled, train_img)
    _, _, th = s42.fit_decide(S, xs, train, train, complete=True)
    Se = s42.scores(IM, xs_eval, pooled, train_img)
    kinds = np.array([x["kind"] for x in xs_eval]); has = np.isfinite(Se["sc"])
    v = np.full(len(xs), "uncertain", object); a = np.full(len(xs), "skip", object); sub = np.full(len(xs), "", object)
    for kind in ("thing", "stuff"):
        ts, tc = re.decode_thresholds(th[kind]); m = test & (kinds == kind)
        v[m], a[m], _ = re.decide5(Se["sc"][m], Se["si"][m], Se["sa"][m], Se["ss"][m], has[m], ts, tc)
    sub[a == "replace"] = [CLASSES[j] for j in Se["sj"][a == "replace"]]
    return v, a, sub


def summarise(a, sub, xs, ok, nv, sw, a_sw, sub_sw, hibase):
    group = np.array(["clothing" if re.GROUP[x["cls"]] == "clothing" else x["kind"] if x["kind"] == "stuff"
                      else "thing_other" for x in xs])
    con = np.isin(a, ["replace", "remove"]); pos = ok & ~nv
    m = sw >= 0; det = np.isin(a_sw, ["replace", "remove"])
    true = np.array([CLASSES[CLASS_INDEX[x["cls"]]] for x in xs], object)
    rp = m & (a_sw == "replace")
    out = dict(FCR=float(con[pos].mean()),
               FCR_by_group={g: float(con[pos & (group == g)].mean()) for g in ("clothing", "thing_other", "stuff")},
               actions={k: int(v) for k, v in collections.Counter(a).items()},
               swap=dict(n=int(m.sum()), detected=float(det[m].mean()), kept=float(np.isin(a_sw[m], ["keep"]).mean()),
                         kept_elsewhere=float((a_sw[m] == "keep_elsewhere").mean()), skipped=float((a_sw[m] == "skip").mean()),
                         replaced_with_true=float((rp & (sub_sw == true)).sum() / m.sum()),
                         substitute_precision=float((sub_sw[rp] == true[rp]).mean()) if rp.any() else None,
                         n_hibase=int((m & hibase).sum()), detected_hibase=float(det[m & hibase].mean()),
                         detected_clothing=float(det[m & (group == "clothing")].mean()),
                         detected_thing_other=float(det[m & (group == "thing_other")].mean())))
    return out


def main():
    if re.CHECK3_BUNDLE_PATH.exists():
        raise FileExistsError(f"{re.CHECK3_BUNDLE_PATH} exists: positive-anchored Check is calibrated")
    rows, maps, owl, q, hist, IM, xs, pooled, top1 = build()
    ids = [r["id"] for r in rows]
    refs = {i: claims.reference_classes(data.references(i)) for i in ids}
    men = np.array([[c in refs[i] for c in CLASSES] for i in ids])
    ok = np.array([x["ok"] for x in xs]); nv = np.array([x["non_visual"] for x in xs])
    k = np.array([CLASS_INDEX[x["cls"]] for x in xs])
    sw = swap_labels(xs, refs); m = sw >= 0
    kk_sw = np.where(m, sw, k)
    xs_sw = [dict(x, cls=CLASSES[s]) if s >= 0 else x for x, s in zip(xs, sw)]
    npos = men.sum(0); hibase = m & (npos[np.maximum(sw, 0)] > npos[k])
    folds = np.array([x["fold"] for x in xs]); img_folds = np.array([calibrate.fold_of(i) for i in ids])
    nv0 = np.zeros(len(xs), bool)

    # ---- S2 out-of-fold (true labels and swapped labels) ----
    a2 = np.empty(len(xs), object); v2 = np.empty(len(xs), object); sub2 = np.empty(len(xs), object)
    a2s = np.empty(len(xs), object); sub2s = np.empty(len(xs), object)
    for f in range(5):
        tr, te = folds != f, folds == f
        v, a, s = s2(IM, xs, xs, pooled, rows, img_folds != f, tr, te)
        v[te & nv], a[te & nv] = "uncertain", "skip"
        v2[te], a2[te], sub2[te] = v[te], a[te], s[te]
        _, a, s = s2(IM, xs, xs_sw, pooled, rows, img_folds != f, tr, te & m)
        a2s[te], sub2s[te] = a[te], s[te]
    S2 = summarise(a2, sub2, xs, ok, nv, sw, a2s, sub2s, hibase)
    S2["proxy"] = calibrate.verdict_quality(v2, ok)

    # ---- S3 out-of-fold over the alpha grid ----
    curve, oof = {}, {}
    for alpha in ALPHAS:
        a3 = np.empty(len(xs), object); v3 = np.empty(len(xs), object); sub3 = np.empty(len(xs), object)
        a3s = np.full(len(xs), "", object); sub3s = np.full(len(xs), "", object); n_ab = []
        for f in range(5):
            te = folds == f
            idx, v, a, s, _, ab = s3(IM, men, pooled, top1, xs, k, alpha, img_folds != f, te, nv)
            v3[idx], a3[idx], sub3[idx] = v, a, s
            idx, _, a, s, _, _ = s3(IM, men, pooled, top1, xs, kk_sw, alpha, img_folds != f, te & m, nv0)
            a3s[idx], sub3s[idx] = a, s
            n_ab.append(int(ab.sum()))
        r = summarise(a3, sub3, xs, ok, nv, sw, a3s, sub3s, hibase)
        r["proxy"] = calibrate.verdict_quality(v3, ok); r["n_abstain_classes_per_fold"] = n_ab
        curve[str(alpha)] = r; oof[alpha] = (v3, a3, sub3)
        print(f"alpha {alpha}: FCR {r['FCR']:.3f} swap det {r['swap']['detected']:.3f} "
              f"sub-true {r['swap']['replaced_with_true']:.3f}", flush=True)
    feasible = [al for al in ALPHAS if curve[str(al)]["FCR"] <= FCR_TARGET]
    alpha_star = max(feasible) if feasible else min(ALPHAS)
    matched = [al for al in ALPHAS if curve[str(al)]["FCR"] <= S2["FCR"]]
    alpha_match = max(matched) if matched else None
    det_match = curve[str(alpha_match)]["swap"]["detected"] if alpha_match is not None else None
    crit_a = det_match is not None and det_match >= S2["swap"]["detected"]
    crit_b = curve[str(alpha_star)]["FCR"] < S2["FCR"]
    decision = dict(alpha_star=alpha_star, fcr_target_met=bool(feasible), alpha_matched_to_S2_FCR=alpha_match,
                    S3_detection_at_matched_FCR=det_match, S2_detection=S2["swap"]["detected"],
                    criterion_a_matched_detection=bool(crit_a), criterion_b_lower_FCR=bool(crit_b),
                    becomes_default=bool(crit_a and crit_b))

    # ---- descriptive: audited claims, substitutes in references ----
    v3, a3, sub3 = oof[alpha_star]
    rp = a3 == "replace"
    breakdown = dict(replace_substitute_in_refs=float(np.mean([s in refs[x["id"]] for s, x in zip(sub3[rp], np.array(xs)[rp])])) if rp.any() else None,
                     remove_refs_lack_same_confusion_set=float(np.mean([not any(re.CONFUSION_OF.get(c) == re.CONFUSION_OF.get(x["cls"]) and c != x["cls"]
                                                                              and re.CONFUSION_OF.get(c) for c in refs[x["id"]])
                                                                       for x in np.array(xs)[a3 == "remove"]])) if (a3 == "remove").any() else None)
    audit = json.loads((config.RESULTS / "claude_visual_audit_B.json").read_text())
    pos = {(x["id"], x["cls"]): i for i, x in enumerate(xs)}; audited = []
    for sample in audit["samples"].values():
        for key, judged in sample["items"].items():
            iid, cls = key.split(":"); i = pos.get((iid, cls))
            if i is not None:
                audited.append(dict(item=key, claude_visual=judged, proxy="correct" if ok[i] else "wrong",
                                    S2_action=str(a2[i]), S2_substitute=str(sub2[i]), S3_action=str(a3[i]), S3_substitute=str(sub3[i])))

    # ---- final bundle on all B at alpha_star; records through the shared checker ----
    all_img = np.ones(len(rows), bool); allc = np.ones(len(xs), bool)
    _, vf, af, subf, tau, ab = s3(IM, men, pooled, top1, xs, k, alpha_star, all_img, allc, nv)
    bundle = dict(schema_version=3, method="lcr", decision="positive_anchored", detector=config.OWL_NAME, classes=CLASSES,
                  attention=re.attention_provenance(), query_file=QFILE,
                  query_sha256=hashlib.sha256(q.tobytes()).hexdigest(), look=hist["look"], pool=hist["pool"],
                  alternatives="all", score_scale="raw_logit", mu=[0.] * len(CLASSES), sd=[1.] * len(CLASSES),
                  tau=[float(t) for t in tau], abstain=[c for c, b in zip(CLASSES, ab) if b], alpha=alpha_star,
                  confusion=re.CONFUSION, substitutes="confusion_set_top1_box_above_own_tau_and_claim",
                  calibration="all_B", calibration_ids=ids,
                  label="clean positives = images whose references mention the class (automatic proxy)",
                  protocol="docs/CHECK_DUZELTME_PROTOKOL.md")
    bundle["id"] = re.fingerprint(bundle)
    checker = re.RegionChecker(bundle, q)
    owner = {tuple(o): i for i, o in enumerate(maps["owner"])}
    generated = []
    for ri, row in enumerate(rows):
        prop = owl[int(row["id"].rsplit("_", 1)[1])]
        lg = re.proposal_logits(prop, q)
        out = []
        for ci, c in enumerate(row["claims"]):
            rec = checker.record(c["cls"], maps["maps"][owner[(ri, ci)]], prop, logits=lg,
                                 person_record=c["rec_att"] if c["kind"] == "person" else None,
                                 non_visual=claims.non_visual(row["draft"], c["cls"]))
            out.append(dict(c, rec_att=rec, regions_by_k={"4": rec["region"]}, excess_by_k={"4": rec["excess"]}))
        generated.append(dict(row, claims=out, bundle_id=bundle["id"], evidence_partition="all_B",
                              look_metadata=dict(row["look_metadata"], region_rule="lcr", proposal_rule=bundle["look"]),
                              check_metadata=dict(method="lcr", decision="positive_anchored", bundle_id=bundle["id"],
                                                  calibration="all_B (in-sample thresholds; OOF metrics in results/check_fix_B.json)",
                                                  label=bundle["label"])))
        if (ri + 1) % 1000 == 0:
            print(f"records {ri + 1}/{len(rows)}", flush=True)
    for i, x in enumerate(xs):           # two code paths (vectorised fit vs RegionChecker.record) must agree
        rec = generated[x["ri"]]["claims"][x["ci"]]["rec_att"]
        assert rec["verdict"] == vf[i] and rec["action"] == af[i] and rec["substitute"] == subf[i], (x, rec["action"], af[i])
    result = dict(bundle_id=bundle["id"], protocol=bundle["protocol"], alpha_grid=list(ALPHAS), fcr_target=FCR_TARGET,
                  n_claims=len(xs), n_swap=int(m.sum()), S2_five_way_z_oof=S2, S3_curve_oof=curve, decision=decision,
                  breakdown_S3_oof=breakdown, audited=audited,
                  final=dict(alpha=alpha_star, abstain=bundle["abstain"], n_abstain=len(bundle["abstain"]),
                             tau={c: round(float(t), 3) for c, t in zip(CLASSES, tau) if KIND[c] != "person"},
                             in_sample_actions=dict(collections.Counter(af))),
                  caveat="B only; FCR uses reference mention as clean positives; swap test is synthetic; proxy metrics descriptive; no validation/test")
    re.CHECK3_BUNDLE_PATH.write_text(json.dumps(bundle, indent=1, allow_nan=False) + "\n")
    cache.save_json(result, config.RESULTS / "check_fix_B.json")
    cache.save_json(generated, P / "A_B_lcr3.json")
    shutil.copyfile(P / "A_B_look4.maps.npz", P / "A_B_lcr3.maps.npz")
    table(result)


def table(r):
    f = lambda v: "–" if v is None or v != v else f"{v:.3f}"
    L = ["# Check düzeltmesi — B (otomatik tablo)", "", "Üreten: `scripts/43_check_fix.py`. Protokol: "
         "`docs/CHECK_DUZELTME_PROTOKOL.md`. Yalnız B, dış-kat (5 görüntü katı). FCR: referansta doğrulanmış claim'lerin "
         "contradicted olma oranı (temiz pozitif = referansta anılma, otomatik proxy). Swap: sentetik yanlış etiket.", "",
         f"Bundle `{r['bundle_id']}`; kişi dışı claim {r['n_claims']}; swap {r['n_swap']}.", "",
         "| sistem | FCR | FCR giysi | FCR diğer thing | FCR stuff | swap yakalama | yüksek-taban yakalama | giysi yakalama | "
         "gerçek sınıfla replace | ikame kesinliği | proxy J | proxy contr. kesinlik |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    def line(name, s):
        w = s["swap"]; g = s["FCR_by_group"]
        return (f"| {name} | {f(s['FCR'])} | {f(g['clothing'])} | {f(g['thing_other'])} | {f(g['stuff'])} | {f(w['detected'])} | "
                f"{f(w['detected_hibase'])} | {f(w['detected_clothing'])} | {f(w['replaced_with_true'])} | {f(w['substitute_precision'])} | "
                f"{f(s['proxy']['J'])} | {f(s['proxy']['contradicted_precision'])} |")
    L.append(line("S2 beş-yollu z", r["S2_five_way_z_oof"]))
    for a, s in r["S3_curve_oof"].items():
        L.append(line(f"S3 α={a}", s))
    d = r["decision"]
    L += ["", f"- Seçilen α* = {d['alpha_star']} (FCR ≤ {r['fcr_target']} hedefi tutuldu: {d['fcr_target_met']}).",
          f"- Eşleşmiş FCR (S2'ninkini aşmayan en büyük α = {d['alpha_matched_to_S2_FCR']}): S3 yakalama "
          f"{f(d['S3_detection_at_matched_FCR'])} vs S2 {f(d['S2_detection'])} → ölçüt (a) {d['criterion_a_matched_detection']}.",
          f"- Ölçüt (b) α*'da FCR S2'den düşük: {d['criterion_b_lower_FCR']}. **Varsayılan olur: {d['becomes_default']}.**",
          f"- Son bundle çekimser sınıflar ({r['final']['n_abstain']}): {', '.join(r['final']['abstain']) or '–'}.",
          f"- α*'da replace ikamesi referansta: {f(r['breakdown_S3_oof']['replace_substitute_in_refs'])}; remove'da referansta "
          f"aynı karışma kümesinden sınıf yok: {f(r['breakdown_S3_oof']['remove_refs_lack_same_confusion_set'])}.", "",
          "## Eylemler (dış-kat)", "", "| sistem | " + " | ".join(re.ACTIONS) + " |", "|---" * (len(re.ACTIONS) + 1) + "|"]
    for name, s in [("S2", r["S2_five_way_z_oof"]), (f"S3 α*={d['alpha_star']}", r["S3_curve_oof"][str(d["alpha_star"])])]:
        L.append(f"| {name} | " + " | ".join(str(s["actions"].get(a, 0)) for a in re.ACTIONS) + " |")
    L += ["", "## Görsel denetlenen claim'ler (Claude gözlemi, insan etiketi değil)", "",
          "| claim | Claude | proxy | S2 | S2 ikame | S3 | S3 ikame |", "|---|---|---|---|---|---|---|"]
    for a in r["audited"]:
        L.append(f"| {a['item']} | {a['claude_visual']} | {a['proxy']} | {a['S2_action']} | {a['S2_substitute']} | "
                 f"{a['S3_action']} | {a['S3_substitute']} |")
    (config.ROOT / "docs/generated/CHECK_FIX_B.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
