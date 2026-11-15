"""Completed Check on half B (docs/CHECK_TAMAMLAMA_PROTOKOL.md). No configuration search, no validation/test.

    python scripts/42_check_complete.py
Look/pool are Codex's LCR choice; adds the image-level veto, replace/remove split, the camera idiom and 5-fold
image cross-fitting. Writes results/lcr_check_bundle.json (new default), results/check_complete_B.json,
docs/generated/CHECK_COMPLETE_B.md and data/cache/pipeline/A_B_lcr2.json.
"""
import collections
import hashlib
import json
import shutil

import numpy as np

import _common  # noqa: F401
from lcr import cache, calibrate, claims, config, data
from lcr import region_evidence as re
from lcr.vocab import CLASS_INDEX, CLASSES, KIND

P = config.CACHE / "pipeline"
QFILE = "class_queries_single.npy"


def build():
    hist = json.loads(re.HISTORICAL_BUNDLE_PATH.read_text())
    look_rule, pool = hist["look"], hist["pool"]
    rows = cache.load_json(P / "A_B_look4.json")
    assert all(r["id"].startswith("train_") for r in rows)
    maps = np.load(P / "A_B_look4.maps.npz"); owner = {tuple(o): i for i, o in enumerate(maps["owner"])}
    owl = cache.OwlCache("train")
    q = np.load(config.CACHE / "owl" / QFILE).astype(np.float32)
    image_max, xs, pooled = [], [], []
    for ri, row in enumerate(rows):
        prop = owl[int(row["id"].rsplit("_", 1)[1])]
        lg = re.proposal_logits(prop, q)
        image_max.append(lg.max(0))
        for ci, c in enumerate(row["claims"]):
            if c["kind"] == "person":
                continue
            att = maps["maps"][owner[(ri, ci)]]
            top, w = re.candidates(att, prop, c["kind"], look_rule)
            pooled.append(re.pool_logits(lg, top, w, pool))
            xs.append(dict(ri=ri, ci=ci, id=row["id"], cls=c["cls"], kind=c["kind"], ok=bool(c["in_refs"]),
                           fold=calibrate.fold_of(row["id"]), part=re.partition(row["id"]), has=len(top) > 0,
                           non_visual=claims.non_visual(row["draft"], c["cls"])))
        if (ri + 1) % 1000 == 0:
            print(f"B features {ri + 1}/{len(rows)}", flush=True)
    return rows, maps, owl, q, hist, np.asarray(image_max), xs, np.asarray(pooled)


def scores(image_max, xs, pooled, train_images):
    mu, sd = image_max[train_images].mean(0), image_max[train_images].std(0) + 1e-6
    z = (pooled - mu) / sd
    k = np.array([CLASS_INDEX[x["cls"]] for x in xs])
    allm = np.array([re.alternative_mask(x["cls"], "all") for x in xs])
    subm = np.array([re.alternative_mask(x["cls"], "group") for x in xs]) & np.array([KIND[c] != "person" for c in CLASSES])
    n = np.arange(len(xs))
    sc = np.where([x["has"] for x in xs], z[n, k], -np.inf)
    sa = np.where(allm, z, -np.inf).max(1); ss = np.where(subm, z, -np.inf).max(1)
    sj = np.where(subm, z, -np.inf).argmax(1)
    img = (image_max[[x["ri"] for x in xs], k] - mu[k]) / sd[k]
    return dict(sc=sc, sa=sa, ss=ss, sj=sj, si=img, mu=mu, sd=sd)


def fit_decide(S, xs, train, test, complete):
    kinds = np.array([x["kind"] for x in xs]); ok = np.array([x["ok"] for x in xs])
    nv = np.array([x["non_visual"] for x in xs]) if complete else np.zeros(len(xs), bool)
    has = np.isfinite(S["sc"])
    verdict = np.full(len(xs), "uncertain", object); action = np.full(len(xs), "skip", object); th = {}
    for kind in ("thing", "stuff"):
        km = kinds == kind; fm = km & train & ~nv
        ts, tc = re.fit_thresholds(S["sc"][fm], S["sa"][fm], ok[fm], s_img=S["si"][fm] if complete else None)
        tm = km & test
        if complete:
            v, a, _ = re.decide5(S["sc"][tm], S["si"][tm], S["sa"][tm], S["ss"][tm], has[tm], ts, tc)
            verdict[tm], action[tm] = v, a
        else:
            verdict[tm] = re.decide(S["sc"][tm], S["sa"][tm], has[tm], ts, tc)
        th[kind] = re.encode_thresholds(ts, tc)
    if complete:
        verdict[nv & test], action[nv & test] = "uncertain", "skip"
    return verdict, action, th


def crossfit(image_max, xs, pooled, rows, complete):
    folds = np.array([x["fold"] for x in xs]); img_folds = np.array([calibrate.fold_of(r["id"]) for r in rows])
    verdict = np.empty(len(xs), object); action = np.empty(len(xs), object)
    for f in range(5):
        S = scores(image_max, xs, pooled, img_folds != f)
        v, a, _ = fit_decide(S, xs, folds != f, folds == f, complete)
        verdict[folds == f], action[folds == f] = v[folds == f], a[folds == f]
    return verdict, action


def main():
    if re.BUNDLE_PATH.exists():
        raise FileExistsError(f"{re.BUNDLE_PATH} exists: completed Check is calibrated")
    rows, maps, owl, q, hist, image_max, xs, pooled = build()
    ok = np.array([x["ok"] for x in xs]); fitp = np.array([x["part"] == "fit" for x in xs])
    old = cache.load_json(P / "A_B_lcr.json")
    s0 = np.array([old[x["ri"]]["claims"][x["ci"]]["rec_att"]["verdict"] for x in xs], object)
    s1, _ = crossfit(image_max, xs, pooled, rows, complete=False)
    s2, a2 = crossfit(image_max, xs, pooled, rows, complete=True)
    Q = calibrate.verdict_quality
    systems = dict(S0_codex_Bcal=dict(B_fit=Q(s0[fitp], ok[fitp])),
                   S1_codex_rule_crossfit=dict(B_fit=Q(s1[fitp], ok[fitp]), B_all=Q(s1, ok)),
                   S2_complete_crossfit=dict(B_fit=Q(s2[fitp], ok[fitp]), B_all=Q(s2, ok)))
    # action breakdown (S2, all B, out-of-fold)
    refs = {i: claims.reference_classes(data.references(i)) for i in {x["id"] for x in xs}}
    # Substitute names come from the same fold scale as the out-of-fold verdict.
    folds = np.array([x["fold"] for x in xs]); img_folds = np.array([calibrate.fold_of(r["id"]) for r in rows])
    sub_name = np.empty(len(xs), object)
    for f in range(5):
        S = scores(image_max, xs, pooled, img_folds != f)
        sub_name[folds == f] = [CLASSES[j] for j in S["sj"][folds == f]]
    same_group_in_refs = np.array([any(re.GROUP.get(r) == re.GROUP[x["cls"]] and r != x["cls"] for r in refs[x["id"]])
                                   for x in xs])
    sub_in_refs = np.array([s in refs[x["id"]] for s, x in zip(sub_name, xs)])
    acts = {}
    for a in re.ACTIONS:
        m = a2 == a
        acts[a] = dict(n=int(m.sum()), rate=float(m.mean()), proxy_correct=float(ok[m].mean()) if m.any() else None)
    rp, rm = a2 == "replace", a2 == "remove"
    breakdown = dict(
        replace_substitute_in_refs=float(sub_in_refs[rp].mean()) if rp.any() else None,
        replace_proxy_wrong_substitute_in_refs=float(sub_in_refs[rp & ~ok].mean()) if (rp & ~ok).any() else None,
        replace_refs_have_same_group=float(same_group_in_refs[rp].mean()) if rp.any() else None,
        remove_refs_lack_same_group=float((~same_group_in_refs[rm]).mean()) if rm.any() else None,
        non_visual_claims=int(sum(x["non_visual"] for x in xs)))
    audit = json.loads((config.RESULTS / "claude_visual_audit_B.json").read_text())
    pos = {(x["id"], x["cls"]): i for i, x in enumerate(xs)}
    audited = []
    for sample in audit["samples"].values():
        for key, judged in sample["items"].items():
            iid, cls = key.split(":"); i = pos.get((iid, cls))
            if i is not None:
                audited.append(dict(item=key, claude_visual=judged, proxy="correct" if ok[i] else "wrong",
                                    S0=str(s0[i]), S2_action=str(a2[i]), S2_substitute=str(sub_name[i]) if a2[i] == "replace" else ""))
    # ---- final bundle on all B, records with the shared checker ----
    S = scores(image_max, xs, pooled, np.ones(len(rows), bool))
    allc = np.ones(len(xs), bool)
    vf, af, th = fit_decide(S, xs, allc, allc, complete=True)
    bundle = dict(schema_version=2, method="lcr", decision="five_way", detector=config.OWL_NAME, classes=CLASSES,
                  attention=re.attention_provenance(), query_file=QFILE,
                  query_sha256=hashlib.sha256(q.tobytes()).hexdigest(), look=hist["look"], pool=hist["pool"],
                  alternatives="all", substitutes="same_group_no_person", veto="image_max_ge_support",
                  mu=S["mu"].tolist(), sd=S["sd"].tolist(), thresholds=th, calibration="all_B",
                  calibration_ids=[r["id"] for r in rows], label="automatic reference proxy; not human correctness",
                  protocol="docs/CHECK_TAMAMLAMA_PROTOKOL.md")
    bundle["id"] = re.fingerprint(bundle)
    checker = re.RegionChecker(bundle, q)
    owner = {tuple(o): i for i, o in enumerate(maps["owner"])}
    generated = []
    for ri, row in enumerate(rows):
        prop = owl[int(row["id"].rsplit("_", 1)[1])]
        lg = re.proposal_logits(prop, q)
        out_claims = []
        for ci, c in enumerate(row["claims"]):
            rec = checker.record(c["cls"], maps["maps"][owner[(ri, ci)]], prop, logits=lg,
                                 person_record=c["rec_att"] if c["kind"] == "person" else None,
                                 non_visual=claims.non_visual(row["draft"], c["cls"]))
            out_claims.append(dict(c, rec_att=rec, regions_by_k={"4": rec["region"]}, excess_by_k={"4": rec["excess"]}))
        generated.append(dict(row, claims=out_claims, bundle_id=bundle["id"], evidence_partition="all_B",
                              look_metadata=dict(row["look_metadata"], region_rule="lcr", proposal_rule=bundle["look"]),
                              check_metadata=dict(method="lcr", decision="five_way", bundle_id=bundle["id"],
                                                  calibration="all_B (in-sample thresholds; OOF metrics in results/check_complete_B.json)",
                                                  label=bundle["label"])))
    for i, x in enumerate(xs):
        rec = generated[x["ri"]]["claims"][x["ci"]]["rec_att"]
        assert rec["verdict"] == vf[i] and rec["action"] == af[i], (x, rec["verdict"], vf[i], rec["action"], af[i])
    result = dict(bundle_id=bundle["id"], protocol=bundle["protocol"], thresholds_all_B=th,
                  n_claims=len(xs), n_B_fit_claims=int(fitp.sum()), systems=systems, actions_S2_oof=acts,
                  breakdown_S2_oof=breakdown, audited=audited,
                  in_sample_all_B=dict(quality=Q(vf, ok), actions=dict(collections.Counter(af))),
                  caveat="B only; proxy labels; S0 thresholds from B_cal so compared on B_fit; no selection")
    re.BUNDLE_PATH.write_text(json.dumps(bundle, indent=1, allow_nan=False) + "\n")
    cache.save_json(result, config.RESULTS / "check_complete_B.json")
    cache.save_json(generated, P / "A_B_lcr2.json")
    shutil.copyfile(P / "A_B_look4.maps.npz", P / "A_B_lcr2.maps.npz")
    table(result)


def table(r):
    f = lambda v: "–" if v is None or v != v else f"{v:.3f}"
    cols = ("J", "contradicted_precision", "recall_wrong", "supported_precision", "supported", "contradicted", "uncertain")
    L = ["# Check tamamlama — B (otomatik tablo)", "", "Üreten: `scripts/42_check_complete.py`. Protokol: "
         "`docs/CHECK_TAMAMLAMA_PROTOKOL.md`. Yalnız B; seçim yok; etiket otomatik referans proxy'si.", "",
         f"Bundle `{r['bundle_id']}`; kişi dışı claim {r['n_claims']} (B_fit {r['n_B_fit_claims']}).", "",
         "## Sistemler", "", "| sistem | küme | J | contr. kesinlik | yanlış yakalama | sup. kesinlik | supported | contradicted | uncertain |",
         "|---|---|---|---|---|---|---|---|---|"]
    for name, d in r["systems"].items():
        for part, q in d.items():
            L.append(f"| {name} | {part} | " + " | ".join(f(q[c]) for c in cols) + " |")
    L += ["", "## S2 eylemleri (tüm B, dış-kat)", "", "| eylem | n | oran | proxy-doğru oranı |", "|---|---|---|---|"]
    for a, d in r["actions_S2_oof"].items():
        L.append(f"| {a} | {d['n']} | {f(d['rate'])} | {f(d['proxy_correct'])} |")
    b = r["breakdown_S2_oof"]
    L += ["", f"- replace: ikame referansta {f(b['replace_substitute_in_refs'])} (proxy-yanlışlarda "
          f"{f(b['replace_proxy_wrong_substitute_in_refs'])}); referansta aynı gruptan sınıf var {f(b['replace_refs_have_same_group'])}.",
          f"- remove: referansta aynı gruptan sınıf yok (silme uygun) {f(b['remove_refs_lack_same_group'])}.",
          f"- 'camera' kalıbı (nesne iddiası değil): {b['non_visual_claims']} claim.", "",
          "## Görsel denetlenen claim'ler (Claude gözlemi, insan etiketi değil)", "",
          "| claim | Claude | proxy | S0 kararı | S2 eylemi | ikame |", "|---|---|---|---|---|---|"]
    for a in r["audited"]:
        L.append(f"| {a['item']} | {a['claude_visual']} | {a['proxy']} | {a['S0']} | {a['S2_action']} | {a['S2_substitute']} |")
    L += ["", "Son eşikler (tüm B, z ölçeği): " + "; ".join(
        f"{k}: τs {f(v['support'])}, τc {f(v['contradiction'])}" for k, v in r["thresholds_all_B"].items())]
    (config.ROOT / "docs/generated/CHECK_COMPLETE_B.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
