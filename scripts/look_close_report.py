"""Build the single Look closure ablation and E2 table from measured artifacts."""
import numpy as np

from _common import macros, write_tex
from lcr import cache, config, look_eval as ev
from lcr.look_sources import METHODS


def stored_records(rows, entries):
    records = []
    for e in entries:
        if e["gtiou"] is None:
            continue
        c = rows[e["ri"]]["claims"][e["ci"]]
        assert c["cls"] == e["cls"]
        j = c["rec_att"]["region"]
        records.append(dict(iid=e["iid"], kind=e["kind"],
            hit=bool(j >= 0 and e["gtiou"][j] >= .5)))
    return records


def rate(records):
    return 100*float(np.mean([r["hit"] for r in records]))


def generate_close_report():
    r, p = config.RESULTS, config.CACHE / "pipeline"
    e1, e2 = cache.load_json(r / "look_e1.json"), cache.load_json(r / "look_e2.json")
    assert e1["sanity_passed"] and e2["intervention_version"] == 2
    assert e2["threshold_winner"] is None and e2["accepted_source"] is None
    assert len(e2["protocol"]["image_ids"]) == 500
    rb3, xb = ev.load_bank("A_B_look3")
    rv3, xv = ev.load_bank("A_validation_look3")
    rb2 = cache.load_json(p / "A_B_look2.json")
    rv2 = cache.load_json(p / "A_validation_look2.json")
    before = stored_records(rv2, xv)
    ids = [row["id"] for row in rv3]
    ablation = []
    for label, stem in (("Raw + R1", ""), ("Sink + R1", "_sink"),
                         ("Region rule v2", "_look2"), ("E1: accepted components", "_look3")):
        br = rb3 if stem == "_look3" else cache.load_json(p / f"A_B{stem}.json")
        vr = rv3 if stem == "_look3" else cache.load_json(p / f"A_validation{stem}.json")
        bs, vs = stored_records(br, xb), stored_records(vr, xv)
        ablation.append(dict(stage=label, B_region=rate(bs), validation_region=rate(vs),
            delta_vs_v2=ev.paired_bootstrap(before, vs, ids)["all"]))
    ablation.append(dict(ablation[-1], stage="After E2 (no addition)"))
    final_stats, _ = ev.measure(xv, rules=config.LOOK_RULES_BY_KIND)
    assert abs(final_stats["all"]["region"]-e1["selected_validation"]["all"]["region"]) < 1e-8
    assert abs(ablation[-1]["B_region"]-e1["selected_B"]["all"]["region"]) < 1e-8
    out = dict(status="frozen; final checks pending", version=config.LOOK_VERSION,
        scope="Look only; Check not started by this task", final=final_stats,
        ablation=ablation, accepted_rules=config.LOOK_RULES_BY_KIND,
        E1_acceptance=e1["acceptance"], E2_accepted_source=e2["accepted_source"],
        E2_reference_limit=e2["reproduction"], feature_level_intervention=e2["causal"],
        provenance=["results/look_e1.json", "results/look_e2.json", "results/look_frozen.json"],
        records=["A_B_look3", "A_validation_look3"])
    target = r / "look_closed.json"
    if target.exists() and cache.load_json(target).get("status") == "closed":
        out["status"] = "closed"
    cache.save_json(out, target)
    f = lambda x: f"{x:.2f}"
    lines = []
    for a in ablation:
        d = a["delta_vs_v2"]; lo, hi = d["ci95_pp"]
        lines.append(f"{a['stage']} & {f(a['B_region'])} & {f(a['validation_region'])} & "
                     f"{d['delta_pp']:+.2f} [{lo:+.2f}, {hi:+.2f}]" + r" \\")
    write_tex("look_ablation_table", r"""\begin{table*}[t]
\caption{Look closure ablation. Region accuracy (\%) against human boxes. Decisions were made on B.
Intervals compare each stage with v2, using 2,000 paired image-level bootstrap samples on validation.
E1 changes a kind only after a gain of at least one point on B. E2 added no source.}
\label{tab:look-close}
\centering\footnotesize
\begin{tabular}{@{}lrrr@{}}
\toprule
Stage & B & Validation & Difference from v2 [95\% CI], points \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule\end{tabular}
\end{table*}
""")
    source_rows = [("Archived S3 control", e2["baseline"]["all"], 0.)]
    for name in METHODS:
        m = e2["sources"][name]["all"]
        source_rows.append((name.replace("_", r"\_"), m, m["region"]-e2["baseline"]["all"]["region"]))
    lines = [f"{name} & {f(m['pointing'])} & {f(m['energy'])} & {f(m['region'])} & {delta:+.2f}" + r" \\"
             for name, m, delta in source_rows]
    write_tex("look_e2_table", r"""\begin{table}[t]
\caption{E2 on the fixed half-B image subset, using the unchanged v2 box rule.
Localisation excludes claims without matched human boxes. No candidate exceeds the archived S3 control
by the required two points. The recomputed caption reference differs from the archived map; it is shown separately.}
\label{tab:look-e2}
\centering\footnotesize\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}lrrrr@{}}
\toprule
Source & Point & Energy & Region & $\Delta$ \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule\end{tabular}
\end{table}
""")
    c = e2["causal"]; interval = ablation[-1]["delta_vs_v2"]["ci95_pp"]
    macros("look_closed_macros", dict(LookClosedRegion=f(final_stats["all"]["region"]),
        LookPersonSigma=str(config.LOOK_RULES_BY_KIND["person"]["sigma"]),
        LookStuffGamma=str(config.LOOK_RULES_BY_KIND["stuff"]["gamma"]),
        LookClosedPerson=f(final_stats["person"]["region"]), LookClosedStuff=f(final_stats["stuff"]["region"]),
        LookClosedThing=f(final_stats["thing"]["region"]), LookClosedB=f(ablation[-1]["B_region"]),
        LookClosedGain=f(ablation[-1]["delta_vs_v2"]["delta_pp"]), LookClosedLow=f(interval[0]), LookClosedHigh=f(interval[1]),
        LookFeatureGain=f(c["look_minus_random"]), LookFeatureLow=f(c["ci95"][0]), LookFeatureHigh=f(c["ci95"][1]),
        LookFeatureDrop=f(c["mean_logp_drop"]["look"]), LookRandomDrop=f(c["mean_logp_drop"]["random"]),
        LookENClaims=e2["n_claims"], LookENBoxClaims=e2["baseline"]["all"]["n"],
        LookERawError=f"{e2['reproduction']['max_abs_S1']:.8f}",
        LookECaptionError=f"{e2['reproduction']['max_abs_S3']:.3f}"))
    return out
