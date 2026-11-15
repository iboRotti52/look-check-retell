"""Write docs/generated/CHECK_V4_METRICS.md from results/check_v4_selection.json and results/check_v4.json
(no hand-typed numbers). Runs with only the selection file too (validation part left out)."""
import json

import _common  # noqa: F401
from lcr import config

R = config.RESULTS
sel = json.loads((R / "check_v4_selection.json").read_text())
val = json.loads((R / "check_v4.json").read_text()) if (R / "check_v4.json").exists() else None
COLS = ("J", "contradicted_precision", "recall_wrong", "supported_precision", "supported", "contradicted")


def f3(x):
    return "–" if x is None or x != x else f"{x:.3f}"


def row(label, q, extra=""):
    return f"| {label} |{extra} " + " | ".join(f3(q[k]) for k in COLS) + " |"


HEAD = "| J | contr. kesinlik | yanlış yakalama | sup. kesinlik | supported oranı | contradicted oranı |"
L = ["# Check v4 — otomatik tablo", "",
     "Üreten: `scripts/36_check_v4_table.py`. Kaynak: results/check_v4_selection.json, results/check_v4.json.",
     "Etiket otomatik referans proxy'si (insan etiketi değil); değerlendirme kişi dışı claim'ler.",
     "Bileşen kodları: `docs/CHECK_V4_PROTOKOL.md` §1; eşik politikası §5 (değişiklik 1).", "",
     f"Seçilen konfigürasyon: **{sel['choice']}** (varsayılan {sel['default']}; en iyi ÇD J {f3(sel['best_J'])}).", "",
     "## B, görüntüye göre 5 katlı ÇD (seçim burada; J'ye göre ilk 20 + varsayılan)", "",
     "| Konfigürasyon | değişen |" + HEAD[1:], "|---|---|" + "---|" * len(COLS)]
ranked = sorted(sel["table"].items(), key=lambda kv: -kv[1]["B_cv"]["J"])
for k, v in ranked[:20] + [(sel["default"], sel["table"][sel["default"]])]:
    L.append(row(("**" + k + "**") if k == sel["choice"] else k, v["B_cv"], f" {v['changed']} |"))
L += ["", "### Tek bileşen etkisi (B ÇD; varsayılandan yalnız bir bileşen değişik)", "",
      "| Değişiklik |" + HEAD[1:], "|---|" + "---|" * len(COLS)]
for k, v in sel["table"].items():
    if v["changed"] == 1:
        L.append(row(k, v["B_cv"]))
if val:
    L += ["", "## Validation (bir kez, eşikler tüm B'den)", "",
          f"Kalibre bölge skorunun AUROC'u (sürekli ölçek, seçilen konfigürasyon): {f3(val['region_score_auroc'])}", "",
          "| Sistem | hedef | grup |" + HEAD[1:], "|---|---|---|" + "---|" * len(COLS)]
    for sn in ("original_live", "original_tuned", "v4", "ref_check_v2", "ref_check_v3"):
        for t, d in val["systems"][sn].items():
            for g in ("all", "thing", "clothing", "stuff"):
                L.append(row(sn, d[g], f" {t} | {g} |"))
    L += ["", "| Fark (hedef 0.5, hepsi) | metrik | fark | GA95 (görüntü bootstrap, 2.000) |", "|---|---|---|---|"]
    for k, v in val["bootstrap"].items():
        for m, d in v.items():
            ci = d["ci95"]
            L.append(f"| {k} | {m} | {f3(d['diff'])} | [{f3(ci[0])}, {f3(ci[1])}] |")
    L += ["", "Eşikler (τs, τc, τalt; C2'de z ölçeği, C1'de olasılık): " +
          "; ".join(f"{k}: " + ", ".join(f"{kind} ({f3(a)}, {f3(b)}, {f3(c)})" for kind, (a, b, c) in th.items())
                    for k, th in val["thresholds"].items())]
out = config.ROOT / "docs" / "generated" / "CHECK_V4_METRICS.md"
out.write_text("\n".join(L) + "\n"); print(out)
