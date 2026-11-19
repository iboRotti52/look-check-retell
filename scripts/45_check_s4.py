"""Check S4 on half B (docs/CHECK_S4_PROTOKOL.md). No validation/test.

    OMP_NUM_THREADS=2 python scripts/45_check_s4.py
S4 = rival-aware OWLv2 evidence (I2) + cost-sensitive per-class thresholds (I1) + centred substitute on the Look
top-1 box (I3) + cleaned confusion sets and compound skips (I5). Components were explored by Fable on the same B folds
(docs/FABLE_INCELEME_2.md); this run implements them in the project code and applies the pre-registered criteria.
Writes results/lcr_check4_bundle.json, results/check_s4_B.json, docs/generated/CHECK_S4_B.md, data/cache/pipeline/A_B_lcr4.json.
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
LAMS = (1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 48, 64, 100)
TARGET, SECONDARY = 0.03, 0.05
WEIGHT = 2.0
M = re.set_matrix(re.CONFUSION_CLEAN)
s43 = importlib.import_module("43_check_fix")


def build():
    hist = json.loads(re.HISTORICAL_BUNDLE_PATH.read_text())
    rows = cache.load_json(P / "A_B_look4.json")
    assert all(r["id"].startswith("train_") for r in rows)
    maps = np.load(P / "A_B_look4.maps.npz"); owner = {tuple(o): i for i, o in enumerate(maps["owner"])}
    owl = cache.OwlCache("train")
    q = np.load(config.CACHE / "owl" / QFILE).astype(np.float32)
    IM, XR, xs, pooled, rreg, top1 = [], [], [], [], [], []
    for ri, row in enumerate(rows):
        prop = owl[int(row["id"].rsplit("_", 1)[1])]
        lg = re.proposal_logits(prop, q)
        rv = re.rival_matrix(lg, M, WEIGHT)
        IM.append(lg.max(0)); XR.append(rv.max(0))
        for ci, c in enumerate(row["claims"]):
            if c["kind"] == "person":
                continue
            top, w = re.candidates(maps["maps"][owner[(ri, ci)]], prop, c["kind"], hist["look"])
            ninf = np.full(len(CLASSES), -np.inf)
            pooled.append(re.pool_logits(lg, top, w, hist["pool"]))
            rreg.append(rv[top].max(0) if len(top) else ninf)
            top1.append(lg[top[0]].astype(np.float64) if len(top) else ninf)
            xs.append(dict(ri=ri, ci=ci, id=row["id"], cls=c["cls"], kind=c["kind"], ok=bool(c["in_refs"]),
                           fold=calibrate.fold_of(row["id"]), part=re.partition(row["id"]), has=len(top) > 0,
                           non_visual=claims.non_visual(row["draft"], c["cls"]),
                           skip_s4=claims.non_visual(row["draft"], c["cls"], compounds=True), blip_alts=c["blip_alts"]))
        if (ri + 1) % 1000 == 0:
            print(f"B features {ri + 1}/{len(rows)}", flush=True)
    return (rows, maps, owl, q, hist, np.asarray(IM, np.float64), np.asarray(XR), xs, np.asarray(pooled),
            np.asarray(rreg), np.asarray(top1))


def s4(F, kk, lam, train_img, test, skip, rival=True, centred=True):
    """S4 actions for claims `test` with class indices kk; everything fitted on images train_img only."""
    X, R = (F["XR"], F["rreg"]) if rival else (F["IM"], F["pooled"])
    tau = re.fit_rival_np(X[train_img], F["men"][train_img], F["cnt"][train_img], lam)
    centre = re.fit_positive_anchored(F["IM"][train_img], (F["cnt"] >= 2)[train_img], .5,
                                      abstain_auroc=None, abstain_min=0)[0]
    ab = ~np.isfinite(tau); ts = np.where(ab, np.inf, tau)
    idx = np.where(test)[0]; xs = F["xs"]
    ri = np.array([xs[i]["ri"] for i in idx]); k = kk[idx]; has = np.array([xs[i]["has"] for i in idx], bool)
    subs = []
    for i, kc, h in zip(idx, k, has):
        if not h:
            subs.append(None)
        elif centred:
            subs.append(re.substitute_centred(F["top1"][i], kc, ts, centre, M))
        else:                                             # S3 ranking (logit - tau) on the cleaned sets
            l = F["top1"][i]; ok = M[kc] & (l >= ts) & (l > l[kc])
            subs.append(int(np.argmax(np.where(ok, l - ts, -np.inf))) if ok.any() else None)
    v, a, _ = re.decide_pa(R[idx, k], X[ri, k], np.array([s is not None for s in subs]), has, ts[k], ab[k])
    a = np.where(skip[idx], "skip", a); v = np.where(skip[idx], "uncertain", v)
    sub = np.array([CLASSES[s] if s is not None and act == "replace" else "" for s, act in zip(subs, a)], object)
    return idx, v, a, sub, tau, centre


def oof(F, run, kk_sw):
    """run(kk, train_img, test, skip) -> idx, v, a, sub; returns OOF arrays for true and swapped labels."""
    n = len(F["xs"]); v = np.empty(n, object); a = np.empty(n, object); sub = np.empty(n, object)
    a_sw = np.full(n, "", object); sub_sw = np.full(n, "", object)
    for f in range(5):
        te = F["folds"] == f; tr = F["img_folds"] != f
        idx, vv, aa, ss = run(F["k"], tr, te, F["skip"])[:4]
        v[idx], a[idx], sub[idx] = vv, aa, ss
        idx, _, aa, ss = run(kk_sw, tr, te & F["m"], np.zeros(n, bool))[:4]
        a_sw[idx], sub_sw[idx] = aa, ss
    return v, a, sub, a_sw, sub_sw


def metrics(F, v, a, sub, a_sw, sub_sw):
    xs, ok, m, k = F["xs"], F["ok"], F["m"], F["k"]
    kind = np.array([x["kind"] for x in xs]); clo = F["clo"]
    pos = ok & ~F["nv_base"]; th = kind == "thing"
    con = np.isin(a, ["replace", "remove"]); det = np.isin(a_sw, ["replace", "remove"])
    true = np.array(CLASSES, object)[k]; rp = m & (a_sw == "replace"); hit = rp & (sub_sw == true)
    S = F["strata"]; strict = F["cnt"][F["ri"], k] >= 2
    rate = lambda x, g: float(x[g].mean()) if g.any() else None
    return dict(
        FCR_all=rate(con, pos), FCR_thing=rate(con, pos & th), FCR_clothing=rate(con, pos & clo),
        FCR_thing_other=rate(con, pos & th & ~clo), FCR_dog=rate(con, pos & (k == CLASS_INDEX["dog"])),
        FCR_stuff=rate(con, pos & ~th), FCR_thing_2refs=rate(con, pos & th & strict),
        detection={s: rate(det, g) for s, g in S.items()},
        swap_kept=rate(a_sw == "keep", m), swap_skipped=rate(a_sw == "skip", m),
        substitute_precision=rate(sub_sw == true, rp), n_replaced_swaps=int(rp.sum()),
        recovery_detected_same_set=float((hit & det & S["same_set"]).sum() / max((det & S["same_set"]).sum(), 1)),
        actions={x: int(y) for x, y in collections.Counter(a).items()},
        proxy=calibrate.verdict_quality(v, ok))


def generate_records(checker, rows, maps, owl, q):
    """A_B_lcr4.json rows through the shared checker. scripts/48 reuses this to rewrite records with the frozen
    bundle when RegionChecker.record changes without changing any decision."""
    bundle = checker.bundle
    owner = {tuple(o): i for i, o in enumerate(maps["owner"])}
    generated = []
    for ri, row in enumerate(rows):
        prop = owl[int(row["id"].rsplit("_", 1)[1])]
        lg = re.proposal_logits(prop, q)
        out = []
        for ci, c in enumerate(row["claims"]):
            rec = checker.record(c["cls"], maps["maps"][owner[(ri, ci)]], prop, logits=lg,
                                 person_record=c["rec_att"] if c["kind"] == "person" else None,
                                 non_visual=claims.non_visual(row["draft"], c["cls"], compounds=True))
            out.append(dict(c, rec_att=rec, regions_by_k={"4": rec["region"]}, excess_by_k={"4": rec["excess"]}))
        generated.append(dict(row, claims=out, bundle_id=bundle["id"], evidence_partition="all_B",
                              look_metadata=dict(row["look_metadata"], region_rule="lcr", proposal_rule=bundle["look"]),
                              check_metadata=dict(method="lcr", decision="rival_np", bundle_id=bundle["id"],
                                                  calibration="all_B (in-sample thresholds; OOF metrics in results/check_s4_B.json)",
                                                  label=bundle["label"])))
        if (ri + 1) % 1000 == 0:
            print(f"records {ri + 1}/{len(rows)}", flush=True)
    return generated


def pick(curve, target):
    ok = [lam for lam in curve if curve[lam]["FCR_thing"] <= target]
    return max(ok, key=lambda lam: curve[lam]["FCR_thing"]) if ok else None


def main():
    if re.CHECK4_BUNDLE_PATH.exists():
        raise FileExistsError(f"{re.CHECK4_BUNDLE_PATH} exists: S4 is calibrated")
    rows, maps, owl, q, hist, IM, XR, xs, pooled, rreg, top1 = build()
    ids = [r["id"] for r in rows]
    refs = {i: claims.reference_classes(data.references(i)) for i in ids}
    cnt = np.zeros((len(ids), len(CLASSES)), np.int16)
    for i, iid in enumerate(ids):
        for r in data.references(iid):
            for c in claims.reference_classes([r]):
                cnt[i, CLASS_INDEX[c]] += 1
    men = np.array([[c in refs[i] for c in CLASSES] for i in ids]); assert ((cnt > 0) == men).all()
    k = np.array([CLASS_INDEX[x["cls"]] for x in xs]); sw = s43.swap_labels(xs, refs); m = sw >= 0
    kk_sw = np.where(m, sw, k)
    clo = np.array([re.GROUP[x["cls"]] == "clothing" for x in xs]); same = m & M[k, np.maximum(sw, 0)]
    strata = dict(all=m, clean_exclusive=same & ~clo, same_set_clothing=same & clo, cross_set=m & ~same, same_set=same)
    F = dict(xs=xs, IM=IM, XR=XR, pooled=pooled, rreg=rreg, top1=top1, men=men, cnt=cnt, k=k, m=m, clo=clo,
             ok=np.array([x["ok"] for x in xs]), ri=np.array([x["ri"] for x in xs]), strata=strata,
             nv_base=np.array([x["non_visual"] for x in xs]), skip=np.array([x["skip_s4"] for x in xs]),
             folds=np.array([x["fold"] for x in xs]), img_folds=np.array([calibrate.fold_of(i) for i in ids]))
    print({s: int(g.sum()) for s, g in strata.items()}, flush=True)

    # ---- comparison rows: S2 (default) and S3 (pre-registered) through the project's own functions ----
    n = len(xs); xs_sw = [dict(x, cls=CLASSES[s]) if s >= 0 else x for x, s in zip(xs, sw)]
    v2 = np.empty(n, object); a2 = np.empty(n, object); sub2 = np.empty(n, object)
    a2s = np.full(n, "", object); sub2s = np.full(n, "", object)
    for f in range(5):
        tr, te = F["folds"] != f, F["folds"] == f
        v, a, s = s43.s2(IM, xs, xs, pooled, rows, F["img_folds"] != f, tr, te)
        nvb = te & F["nv_base"]; v[nvb], a[nvb] = "uncertain", "skip"
        v2[te], a2[te], sub2[te] = v[te], a[te], s[te]
        _, a, s = s43.s2(IM, xs, xs_sw, pooled, rows, F["img_folds"] != f, tr, te & m)
        a2s[te & m], sub2s[te & m] = a[te & m], s[te & m]
    rows_out = dict(S2_five_way_z=metrics(F, v2, a2, sub2, a2s, sub2s))
    s3run = lambda kk, tr, te, skip: s43.s3(IM, men, pooled, top1, xs, kk, 0.075, tr, te, F["nv_base"] if skip.any() else skip)
    rows_out["S3_positive_anchored_a0.075"] = metrics(F, *oof(F, s3run, kk_sw))

    # ---- S4 curve and ablations ----
    variants = dict(S4=dict(rival=True, centred=True), S4a_no_rival=dict(rival=False, centred=True),
                    S4_S3_substitute=dict(rival=True, centred=False))
    curves = {}
    for name, opt in variants.items():
        curves[name] = {}
        for lam in LAMS:
            run = lambda kk, tr, te, skip, lam=lam, opt=opt: s4(F, kk, lam, tr, te, skip, **opt)
            curves[name][lam] = metrics(F, *oof(F, run, kk_sw))
            c = curves[name][lam]
            print(f"{name} lam {lam}: thing-FCR {c['FCR_thing']:.3f} clean {c['detection']['clean_exclusive']:.3f} "
                  f"clothing-FCR {c['FCR_clothing']:.3f} sub-prec {c['substitute_precision'] or 0:.3f}", flush=True)
    lam_star = pick(curves["S4"], TARGET); lam_sec = pick(curves["S4"], SECONDARY)
    if lam_star is None:
        raise RuntimeError("no lambda reaches the FCR target")
    for name in variants:
        for tag, t in (("fcr03", TARGET), ("fcr05", SECONDARY)):
            lam = pick(curves[name], t)
            rows_out[f"{name}@{tag}"] = dict(curves[name][lam], lam=lam) if lam is not None else None
    s4m, s2m = curves["S4"][lam_star], rows_out["S2_five_way_z"]
    crit = dict(a_clean_detection=s4m["detection"]["clean_exclusive"] >= s2m["detection"]["clean_exclusive"],
                b_clothing_fcr=s4m["FCR_clothing"] <= 0.05,
                c_substitute_precision=(s4m["substitute_precision"] or 0) >= (s2m["substitute_precision"] or 0))
    decision = dict(lam_star=lam_star, lam_secondary=lam_sec, **crit, becomes_default=bool(all(crit.values())))

    # ---- final bundle on all B at lam_star; records through the shared checker ----
    all_img = np.ones(len(rows), bool); allc = np.ones(n, bool)
    _, vf, af, subf, tau, centre = s4(F, k, lam_star, all_img, allc, F["skip"])
    bundle = dict(schema_version=4, method="lcr", decision="rival_np", detector=config.OWL_NAME, classes=CLASSES,
                  attention=re.attention_provenance(), query_file=QFILE,
                  query_sha256=hashlib.sha256(q.tobytes()).hexdigest(), look=hist["look"], pool=hist["pool"],
                  alternatives="all", score_scale="raw_logit", mu=[0.] * len(CLASSES), sd=[1.] * len(CLASSES),
                  tau=[float(t) if np.isfinite(t) else None for t in tau], centre=[float(c) for c in centre],
                  lam=lam_star, rival_weight=WEIGHT, confusion=re.CONFUSION_CLEAN, compounds=True,
                  substitutes="centred_argmax_confusion_set_top1_box_above_own_tau_and_claim",
                  calibration="all_B", calibration_ids=ids,
                  label="positives: >=2 references mention the class; negatives: no reference mentions it (automatic proxy)",
                  protocol="docs/CHECK_S4_PROTOKOL.md")
    bundle["id"] = re.fingerprint(bundle)
    checker = re.RegionChecker(bundle, q)
    generated = generate_records(checker, rows, maps, owl, q)
    for i, x in enumerate(xs):            # vectorised fit and RegionChecker.record must agree on every claim
        rec = generated[x["ri"]]["claims"][x["ci"]]["rec_att"]
        assert rec["verdict"] == vf[i] and rec["action"] == af[i] and rec["substitute"] == subf[i], (x, rec["action"], af[i])
    result = dict(bundle_id=bundle["id"], protocol=bundle["protocol"], lambdas=list(LAMS), target=TARGET,
                  n_claims=n, strata_n={s: int(g.sum()) for s, g in strata.items()}, rows=rows_out,
                  S4_curve={str(l): c for l, c in curves["S4"].items()}, decision=decision,
                  final=dict(lam=lam_star, n_classes_without_tau=int(sum(t is None for t in bundle["tau"])),
                             in_sample_actions={str(x): int(y) for x, y in collections.Counter(af).items()}),
                  caveat="B only; components explored on the same B folds by Fable (docs/FABLE_INCELEME_2.md); FCR uses "
                         "reference mention as clean positives; swap test synthetic; proxy metrics descriptive; no validation/test")
    re.CHECK4_BUNDLE_PATH.write_text(json.dumps(bundle, indent=1, allow_nan=False) + "\n")
    cache.save_json(result, config.RESULTS / "check_s4_B.json")
    cache.save_json(generated, P / "A_B_lcr4.json")
    shutil.copyfile(P / "A_B_look4.maps.npz", P / "A_B_lcr4.maps.npz")
    table(result)


def table(r):
    f = lambda v: "–" if v is None or v != v else f"{v:.3f}"
    L = ["# Check S4 — B (otomatik tablo)", "", "Üreten: `scripts/45_check_s4.py`. Protokol: `docs/CHECK_S4_PROTOKOL.md`. "
         "Yalnız B, dış-kat. **Bileşenler Fable tarafından aynı B katlarında keşfedildi; bağımsız kanıt değil.** FCR: "
         "referansta doğrulanmış claim'lerin contradicted olma oranı (otomatik proxy). Swap: sentetik yanlış etiket.", "",
         f"Bundle `{r['bundle_id']}`; kişi dışı claim {r['n_claims']}; swap tabakaları {r['strata_n']}.", "",
         "| sistem | λ | thing-FCR | giysi FCR | köpek FCR | stuff FCR | FCR (≥2 ref) | yakalama tümü | temiz dışlayan | "
         "aynı-küme giysi | kümeler arası | ikame kesinliği | geri kazanım | proxy contr. kes. |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, s in r["rows"].items():
        if s is None:
            L.append(f"| {name} | – | hedefe ulaşılamadı |" + " |" * 11); continue
        d = s["detection"]
        L.append(f"| {name} | {s.get('lam', '–')} | {f(s['FCR_thing'])} | {f(s['FCR_clothing'])} | {f(s['FCR_dog'])} | "
                 f"{f(s['FCR_stuff'])} | {f(s['FCR_thing_2refs'])} | {f(d['all'])} | {f(d['clean_exclusive'])} | "
                 f"{f(d['same_set_clothing'])} | {f(d['cross_set'])} | {f(s['substitute_precision'])} | "
                 f"{f(s['recovery_detected_same_set'])} | {f(s['proxy']['contradicted_precision'])} |")
    d = r["decision"]
    L += ["", f"- λ* = {d['lam_star']} (thing-FCR ≤ {r['target']}); ikincil λ = {d['lam_secondary']} (≤ 0.05).",
          f"- Ölçütler: (a) temiz dışlayan yakalama ≥ S2: {d['a_clean_detection']}; (b) giysi FCR ≤ 0.05: "
          f"{d['b_clothing_fcr']}; (c) ikame kesinliği ≥ S2: {d['c_substitute_precision']}. "
          f"**Varsayılan olur: {d['becomes_default']}.**",
          f"- Son bundle: τ'su olmayan (çekimser) sınıf {r['final']['n_classes_without_tau']}; tüm B eylemleri (örneklem içi) "
          f"{r['final']['in_sample_actions']}.", "", "## S4 eğrisi (dış-kat)", "",
          "| λ | thing-FCR | giysi FCR | temiz dışlayan | aynı-küme giysi | tümü | ikame kesinliği |", "|---|---|---|---|---|---|---|"]
    for lam, s in r["S4_curve"].items():
        L.append(f"| {lam} | {f(s['FCR_thing'])} | {f(s['FCR_clothing'])} | {f(s['detection']['clean_exclusive'])} | "
                 f"{f(s['detection']['same_set_clothing'])} | {f(s['detection']['all'])} | {f(s['substitute_precision'])} |")
    (config.ROOT / "docs/generated/CHECK_S4_B.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
