"""Sample size for H1 (report Table "tab:power" and its macros). CPU only, no data.

H1 asks for a 15% relative reduction of HICR. The paired sign-test power is simulated at exactly that effect:
for each draft error rate p0 and a 1% break rate, the fix rate is the one that makes the expected relative
reduction 15% (lcr.metrics.fix_rate_for). The first version of this table (scripts/08) fixed a third of the
errors instead, which is a 14-26% reduction depending on p0, so its power was not H1's power (2026-11-19).

Not modelled, so the paired column is optimistic: claims of one image are not independent (fewer effective
units), and the test traces draft claims only, while HICR also counts objects Retell adds.

    python scripts/50_h1_power.py
"""
import _common
from lcr import cache, config, metrics

TARGET, BREAK = 0.15, 0.01
P0S = (0.05, 0.08, 0.12)
NS = (500, 1000, 2000, 4000)


def main():
    rows = []
    for p0 in P0S:
        fix = metrics.fix_rate_for(TARGET, p0, BREAK)
        rows.append(dict(p0=p0, independent=metrics.claims_needed(p0, TARGET), fix_rate=fix,
                         paired={n: metrics.paired_power(n, p0, fix, BREAK) for n in NS}))
    cache.save_json(dict(target=TARGET, break_rate=BREAK, rows=rows), config.RESULTS / "h1_power.json")
    body = "\n".join(f"{100 * r['p0']:.0f}\\% & {r['independent']:,} & {r['fix_rate']:.2f} & "
                     + " & ".join(f"{100 * r['paired'][n]:.0f}\\%" for n in NS) + r" \\" for r in rows).replace(",", "{,}")
    _common.write_tex("power_table", r"""\begin{table}[t]
\caption{Sample size for H1 (15\% relative HICR reduction, $\alpha=0.05$). Left: claims per system for an
independent two-proportion test at 80\% power. Right: simulated power of the paired sign test with $N$ traced
draft claims at exactly that effect: 1\% of correct claims are broken and the fix rate is set so that HICR falls
by 15\%. Optimistic: claims of one image are treated as independent, and objects added by Retell are not traced.}
\label{tab:power}
\centering\footnotesize
\begin{tabular}{@{}crc|cccc@{}}
\toprule
Draft error & Independent & Fix & \multicolumn{4}{c}{Paired power at $N$ claims} \\
rate $p_0$ & claims needed & rate & 500 & 1{,}000 & 2{,}000 & 4{,}000 \\
\midrule
""" + body + r"""
\bottomrule
\end{tabular}
\end{table}
""")
    mid = next(r for r in rows if r["p0"] == 0.08)
    low = next(r for r in rows if r["p0"] == 0.05)
    _common.macros("power_macros", dict(
        HPowerFixEight=f"{mid['fix_rate']:.2f}", HPowerEightOneK=f"{100 * mid['paired'][1000]:.0f}",
        HPowerEightTwoK=f"{100 * mid['paired'][2000]:.0f}", HPowerFiveTwoK=f"{100 * low['paired'][2000]:.0f}",
        HPowerFiveFourK=f"{100 * low['paired'][4000]:.0f}", HPowerIndepEight=f"{mid['independent']:,}".replace(",", "{,}")))
    print({r["p0"]: {n: round(p, 2) for n, p in r["paired"].items()} for r in rows})


if __name__ == "__main__":
    main()
