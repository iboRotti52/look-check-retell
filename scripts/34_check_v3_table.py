"""Write docs/generated/CHECK_V3_METRICS.md from results/check_v3_selection.json and results/check_v3.json
(no hand-typed numbers). Runs with only the selection file too (validation part left out)."""
import json

import _common  # noqa: F401
from lcr import config

R = config.RESULTS
sel = json.loads((R / "check_v3_selection.json").read_text())
val = json.loads((R / "check_v3.json").read_text()) if (R / "check_v3.json").exists() else None


def f3(x):
    return "–" if x is None or x != x else f"{x:.3f}"


L = ["# Check v3 — otomatik tablo", "",
     "Üreten: `scripts/34_check_v3_table.py`. Kaynak: results/check_v3_selection.json, results/check_v3.json.",
     "Etiket otomatik referans proxy'si (insan etiketi değil); değerlendirme kişi dışı claim'ler.", "",
     f"Seçilen özellik ailesi: **{sel['feature_choice']}**, model: **{sel['model_choice']}**, "
     f"alternatif kuralı: **{sel['alternative_choice']}** (tolerans {sel['tolerance']}).", "",
     "## B, tam iç içe dış-kat (seçim burada)", "",
     "| Aile | log-loss | AUROC | AP(yanlış) | kesinlik@%20 | contr. kesinlik | yanlış yakalama |",
     "|---|---|---|---|---|---|---|"]
for name, v in sorted(sel["families"].items(), key=lambda kv: kv[1]["oof_log_loss"]):
    b = v["B"]
    L.append(f"| {name} | {v['oof_log_loss']:.4f} | {f3(b['auroc'])} | {f3(b['ap_wrong'])} | {f3(b['prec_at_cov20'])} | "
             f"{f3(b['contradicted_precision'])} | {f3(b['recall_wrong'])} |")
L += ["", "### Model sınıfı (seçilen aileyle)", "", "| Model | log-loss | AUROC | kesinlik@%20 |", "|---|---|---|---|"]
for k, v in sel["models"].items():
    L.append(f"| {k} | {v['oof_log_loss']:.4f} | {f3(v['B']['auroc'])} | {f3(v['B']['prec_at_cov20'])} |")
L += ["", "### Ablation (B, λ sabit 1/1)", "", "| | AUROC | AP(yanlış) | kesinlik@%20 |", "|---|---|---|---|"]
for k, v in sel["ablations_B"].items():
    L.append(f"| {k} | {f3(v['auroc'])} | {f3(v['ap_wrong'])} | {f3(v['prec_at_cov20'])} |")
L += ["", "### Alternatif kuralı (B; referans-dışı claim'de önerinin referansta olma oranı)", "",
      "| Kural | n | oran |", "|---|---|---|"]
for k, v in sel["alternative_B"].items():
    L.append(f"| {k} | {v['n']} | {f3(v['rate'])} |")
if val:
    L += ["", "## Validation (bir kez)", "",
          "| Grup | sürüm | n | AUROC | AP(yanlış) | kesinlik@%10/20/30 | contr. kesinlik | yanlış yakalama | contradicted oranı |",
          "|---|---|---|---|---|---|---|---|---|"]
    for g, d in val["groups"].items():
        for name, q in (("v2", d["v2"]), ("v3 (0.5)", d["v3"]["0.5"]), ("v3 (0.6)", d["v3"]["0.6"]),
                        ("v3 (0.7)", d["v3"]["0.7"])):
            L.append(f"| {g} | {name} | {q['n']} | {f3(q['auroc'])} | {f3(q['ap_wrong'])} | "
                     f"{f3(q['prec_at_cov10'])}/{f3(q['prec_at_cov20'])}/{f3(q['prec_at_cov30'])} | "
                     f"{f3(q['contradicted_precision'])} | {f3(q['recall_wrong'])} | {f3(q['contradicted'])} |")
    L += ["", "| v3 − v2 (hepsi) | fark | GA95 (görüntü bootstrap, 2.000) |", "|---|---|---|"]
    for k, v in val["bootstrap_v3_minus_v2"].items():
        L.append(f"| {k} | {v['diff']:+.4f} | [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}] |")
    L += ["", "Alternatif (validation): " + ", ".join(f"{k} {f3(v['rate'])} (n={v['n']})"
                                                     for k, v in val["alternative_val"].items())]
    if val.get("coefficients"):
        L += ["", "Katsayılar (standartlaştırılmış): " + ", ".join(f"{k} {v:+.2f}" for k, v in val["coefficients"].items())]
out = config.ROOT / "docs" / "generated" / "CHECK_V3_METRICS.md"
out.write_text("\n".join(L) + "\n"); print(out)
