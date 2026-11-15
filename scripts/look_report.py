"""Generate Look report tables/macros exclusively from measured JSON artifacts."""
from _common import macros, write_tex
from lcr import cache, config


def main():
    r = config.RESULTS
    raw, sink, v2 = [cache.load_json(r / f"look_benchmark_{stem}.json")
                     for stem in ("A_validation", "A_validation_sink", "A_validation_look2")]
    regions = cache.load_json(r / "look_regions_A_validation_sink.json")
    audit = cache.load_json(r / "look_v2_review_B.json")
    sanity = cache.load_json(r / "look_sanity_A_validation_look2.json")
    f = lambda x: f"{x:.1f}"
    n = lambda x: f"{x:,}".replace(",", "{,}")
    macros("look_macros", dict(
        LookAllClaims=n(v2["claims"]), LookGtClaims=n(v2["claims_with_gt"]),
        LookCoverage=f(100 * v2["claims_with_gt"] / v2["claims"]),
        LookRawPoint=f(raw["all"]["pointing"]), LookSinkPoint=f(sink["all"]["pointing"]),
        LookVOneRegion=f(sink["all"]["region_att"]), LookVTwoRegion=f(v2["all"]["region_att"]),
        LookPersonRegion=f(v2["person"]["region_att"]), LookDetectorRegion=f(v2["all"]["region_det"]),
        LookOracleRegion=f(v2["all"]["region_oracle"]), LookAlpha=str(config.REGION_ALPHA),
        LookGamma=str(config.REGION_GAMMA), LookGridCount=len(regions["grid_on_select"]),
        LookBRegion=f(audit["all"]["own"]["accuracy_pct"]),
        LookBObjectness=f(audit["all"]["objectness_only"]["accuracy_pct"]),
        LookBUniform=f(audit["all"]["uniform_v2"]["accuracy_pct"]),
        LookSwapOwn=f(audit["matched_map_swap"]["own_accuracy_pct"]),
        LookSwapOther=f(audit["matched_map_swap"]["swapped_accuracy_pct"]),
        LookPersonErrors=n(audit["person_failures"]["n"]),
        LookPersonPeakErrors=n(audit["person_failures"]["with_correct_point"]),
        LookTranspose=f(sanity["transposed"]), LookSwapPointOwn=f(sanity["swap_own"]),
        LookSwapPointOther=f(sanity["swap_other"])))
    lines = []
    for label, obj in (("Raw + v1", raw), ("Sink + v1", sink), ("Sink + v2", v2)):
        a = obj["all"]
        lines.append(f"{label} & {f(a['pointing'])} & {f(a['energy'])} & {f(a['region_att'])}" + r" \\")
    write_tex("look_human_table", r"""\begin{table}[t]
\caption{Human-box localisation on validation (\LookGtClaims{} claims). All entries are percentages.
Energy uses the exact union of annotated boxes. Pointing and energy share the map; region accuracy
also depends on the proposal rule.}
\label{tab:look-human}
\centering\footnotesize
\begin{tabular}{@{}lrrr@{}}
\toprule
Map + region rule & Pointing & Energy & Region \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}
""")
    lines = []
    for label in ("all", "person", "thing", "stuff"):
        a = v2[label]
        lines.append(f"{label.capitalize()} & {n(a['n'])} & {f(a['region_att'])} & {f(a['region_det'])} & {f(a['region_oracle'])}" + r" \\")
    write_tex("look_kind_table", r"""\begin{table}[t]
\caption{Look v2 region accuracy by kind, against any matching human box at IoU $\geq 0.5$.
B3 uses the claimed class score; oracle asks whether any eligible proposal succeeds. Percentages.}
\label{tab:look-kind}
\centering\footnotesize\setlength{\tabcolsep}{4pt}
\begin{tabular}{@{}lrrrr@{}}
\toprule
Kind & Claims & Look v2 & B3 & Oracle \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}
""")
    if (config.CACHE / "pipeline" / "A_validation_look3.json").exists():
        from look_close_report import generate_close_report
        generate_close_report()


if __name__ == "__main__":
    main()
