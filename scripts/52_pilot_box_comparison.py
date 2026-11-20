"""Pilot P6 (docs/PILOT_PROTOKOL.md): detector box vs Look box vs human box, under the same frozen OWLv2 (S4). B only.

For every sampled claim whose word has a referent (labelled correct, or misidentified with a box), three proposals of
the same cached OWLv2 pool are compared: the detector's best box for the claimed word (B3), the box S4's record shows
(Look), and the proposal that best matches the human-drawn box (the oracle location). On each, S4's own rules ask:
does the box support a correct word, and for a misidentified word with a known correct name, does it reject the
claim and name the true object? Reading (Astra): if the human-matched box finds the right name but Look's does not,
improving Look pays off; if even the human-matched box does not, the detector's proposals or class separation come
first. Proposal coverage (does any proposal match the human box) is reported, because a missing proposal caps all
three. Image-level bootstrap intervals; differences are paired on the same claims.

    python scripts/52_pilot_box_comparison.py --name pilot_b
"""
import argparse

import numpy as np

import _common  # noqa: F401
from lcr import cache, config, metrics, pilot
from lcr import region_evidence as re
from lcr.vocab import CLASS_INDEX

SOURCES = ("detector", "look", "human")


def evaluate(rows, s4, owl, queries):
    tau = np.array([np.nan if t is None else t for t in s4["tau"]], np.float64)
    M = re.set_matrix(s4["confusion"])
    out = []
    for r in rows:
        if not r["id"].startswith("train_"):
            raise ValueError("P6 is B-only")
        if r["human_box"] is None:
            continue
        prop = owl[int(r["id"].rsplit("_", 1)[1])]
        boxes = prop["boxes"].astype(np.float32)
        pool = pilot.proposal_pool(prop, s4["look"])
        z = (re.proposal_logits(prop, queries) - np.asarray(s4["mu"])) / np.asarray(s4["sd"])
        v = re.rival_matrix(z, M, float(s4["rival_weight"]))
        k = CLASS_INDEX[r["cls"]]
        true_k = CLASS_INDEX.get(r["true_cls"]) if r.get("true_cls") else None
        det = int(np.argmax(np.where(pool, z[:, k], -np.inf))) if pool.any() else None
        hum, coverage = pilot.matching_proposal(boxes, pool, r["human_box"])
        index = dict(detector=det, look=r["look_region"] if r["look_region"] is not None and r["look_region"] >= 0 else None,
                     human=hum)
        rec = dict(id=r["id"], cls=r["cls"], label=r["label"], error_type=r.get("error_type"), true_cls=r.get("true_cls"),
                   stratum=r["stratum"], proposal_coverage=coverage)
        for s, p in index.items():
            if p is None:
                rec[s] = None
                continue
            rec[s] = dict(proposal=p, iou_with_human=pilot.box_iou(boxes[p].tolist(), r["human_box"]),
                          **pilot.box_evidence(z, v, tau, M, p, k, true_k if r["label"] == "incorrect" else None))
        out.append(rec)
    return out


def by_image(rows):
    groups = {}
    for r in rows:
        groups.setdefault(r["id"], []).append(r)
    return list(groups.values())


def rate(rows, source, field):
    """Share of `rows` where `source` has the field true (a missing box counts as false), image bootstrap CI."""
    get = lambda r: bool(r[source] and r[source].get(field))
    if not rows:
        return dict(n=0, rate=None, ci95=None)
    v, lo, hi = metrics.bootstrap_ci(by_image(rows), lambda us: float(np.mean([get(r) for u in us for r in u])))
    return dict(n=len(rows), rate=v, ci95=[lo, hi])


def diff(rows, a, b, field):
    get = lambda r, s: bool(r[s] and r[s].get(field))
    if not rows:
        return None
    v, lo, hi = metrics.bootstrap_ci(by_image(rows), lambda us: float(np.mean([get(r, a) - get(r, b) for u in us for r in u])))
    return dict(diff=v, ci95=[lo, hi])


def summarise(recs):
    correct = [r for r in recs if r["label"] == "correct"]
    mis = [r for r in recs if r["label"] == "incorrect" and r["error_type"] == "misidentified"]
    mis_known = [r for r in mis if r["true_cls"] and r["true_cls"] != r["cls"]]
    for r in recs:                                  # localisation: the source's box overlaps the human box
        for s in SOURCES:
            if r[s]:
                r[s]["localised"] = r[s]["iou_with_human"] >= .5
    out = dict(n_claims=len(recs), n_correct=len(correct), n_misidentified=len(mis), n_misidentified_named=len(mis_known),
               proposal_coverage=dict(n=len(recs), share_iou_05=float(np.mean([r["proposal_coverage"] >= .5 for r in recs]))
                                      if recs else None),
               localised={s: rate(recs, s, "localised") for s in ("detector", "look")},
               correct_supported={s: rate(correct, s, "supports_claim") for s in SOURCES},
               correct_supported_diff=dict(human_minus_look=diff(correct, "human", "look", "supports_claim"),
                                           look_minus_detector=diff(correct, "look", "detector", "supports_claim")))
    for f in ("claim_rejected", "true_name_found", "true_is_rival"):
        out[f"misidentified_{f}"] = {s: rate(mis_known, s, f) for s in SOURCES}
    out["misidentified_true_name_found_diff"] = dict(
        human_minus_look=diff(mis_known, "human", "look", "true_name_found"),
        look_minus_detector=diff(mis_known, "look", "detector", "true_name_found"))
    return out


def write_markdown(out, name, path):
    f = lambda d: "–" if not d or d.get("rate") is None else f"{100 * d['rate']:.1f} [{100 * d['ci95'][0]:.1f}, {100 * d['ci95'][1]:.1f}] (n={d['n']})"
    g = lambda d: "–" if not d or d["diff"] != d["diff"] else f"{100 * d['diff']:+.1f} [{100 * d['ci95'][0]:+.1f}, {100 * d['ci95'][1]:+.1f}]"
    L = [f"# Pilot P6 — detektör / Look / insan kutusu ({name}, B)", "",
         "Üreten: `scripts/52_pilot_box_comparison.py`. Protokol: `docs/PILOT_PROTOKOL.md` P6. Aynı donmuş OWLv2, S4 kuralları. "
         "% ve görüntü düzeyi bootstrap %95 GA; farklar aynı claim'lerde eşleştirilmiş.", "",
         f"Referansı olan claim: {out['n_claims']} (doğru {out['n_correct']}, yanlış tanımlanmış {out['n_misidentified']}, "
         f"doğru adı bilinen {out['n_misidentified_named']}). İnsan kutusuna IoU ≥ 0,5 bir öneri olan: "
         + ("–" if out["proposal_coverage"]["share_iou_05"] is None else f"{100 * out['proposal_coverage']['share_iou_05']:.1f}%")
         + " (hiçbir kaynak bunun üstüne çıkamaz).", "",
         "| ölçüm | detektör (B3) | Look | insan kutusu |", "|---|---|---|---|",
         f"| kutu insan kutusuyla örtüşüyor | {f(out['localised']['detector'])} | {f(out['localised']['look'])} | – |",
         f"| doğru claim'i destekliyor | " + " | ".join(f(out['correct_supported'][s]) for s in SOURCES) + " |"]
    for fld, title in (("claim_rejected", "yanlış adı reddediyor"), ("true_name_found", "doğru adı buluyor"),
                       ("true_is_rival", "doğru ad S4'ün ikame kümesinde")):
        L.append(f"| yanlış tanımlanmış: {title} | " + " | ".join(f(out[f'misidentified_{fld}'][s]) for s in SOURCES) + " |")
    L += ["", "| fark | insan − Look | Look − detektör |", "|---|---|---|",
          f"| doğru claim desteği | {g(out['correct_supported_diff']['human_minus_look'])} | {g(out['correct_supported_diff']['look_minus_detector'])} |",
          f"| doğru adı bulma | {g(out['misidentified_true_name_found_diff']['human_minus_look'])} | "
          f"{g(out['misidentified_true_name_found_diff']['look_minus_detector'])} |", "",
          "Okuma: insan kutusu doğru adı bulup Look bulamıyorsa Look'u iyileştirmek anlamlı; insan kutusu da bulamıyorsa "
          "önce detektörün önerileri ve sınıf ayrımı. Etiketler insan pilotundan; doğru ad etiketleyicinin serbest metninden "
          "sözlük sınıfına eşlendi."]
    path.write_text("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="pilot name used with scripts/07 and scripts/10")
    a = ap.parse_args()
    key = cache.load_json(config.ROOT / "annotation" / a.name / "key.json")
    if key.get("mode") != "lcr_pilot":
        raise ValueError("P6 needs a B pilot made with scripts/07 --pilot lcr")
    s4 = cache.load_json(re.BUNDLE_PATH)
    if key["s4_bundle_id"] != s4["id"]:
        raise ValueError("the pilot was sampled from another S4 bundle")
    rows = cache.load_json(config.RESULTS / f"pilot_referent_boxes_{a.name}.json")
    recs = evaluate(rows, s4, cache.OwlCache("train"), np.load(config.CACHE / "owl" / s4["query_file"]))
    out = dict(name=a.name, s4_bundle_id=s4["id"], summary=summarise(recs), records=recs)
    cache.save_json(out, config.RESULTS / f"pilot_box_comparison_{a.name}.json")
    write_markdown(out["summary"], a.name, config.ROOT / "docs" / "generated" / f"PILOT_BOX_COMPARISON_{a.name}.md")
    print({k: v for k, v in out["summary"].items() if k.startswith("n_")})


if __name__ == "__main__":
    main()
