"""Write docs/generated/LOOK_E3_METRICS.md from the E3 result files (no hand-typed numbers)."""
import json

import _common  # noqa: F401
from lcr import config

R = config.RESULTS
sel, a, boxes, fin = (json.loads((R / f).read_text()) for f in
                      ("look_e3_selection.json", "look_e3_stageA.json", "look_e3_boxes_selection.json", "look_e3.json"))
heads = sel["winner"]["heads"]
L = ["# Look E3 — otomatik tablo", "", "Üreten: `scripts/30_look_e3_table.py`. Kaynak: results/look_e3*.json.", "",
     "Başlıklar (iki eksen de 1-tabanlı): " + ", ".join(f"L{h//12+1}H{h%12+1}" for h in heads) + ".", "",
     "Aile/tür GA95 aralıkları tekildir; eşzamanlı güvence vermez. E3 uyarlama sınırları ve canlı v4: `docs/LOOK_E3_INCELEME.md`.", "",
     f"B v3 = {sel['B_v3']:.2f}, B yeniden hesaplanan S3 = {sel['B_S3_recomputed']:.2f}; validation v3 = "
     f"{a['v3']['all']['region']:.2f} (pointing {a['v3']['all']['pointing']:.2f}).", "",
     "| Aile | seçilen | B ÇD kazancı | val bölge | val pointing | person | thing | stuff | val GA95 vs v3 |",
     "|---|---|---|---|---|---|---|---|---|"]
for fam, v in a["families"].items():
    s = v["summary"]; ci = v["paired_bootstrap"]["all"]["ci95_pp"]
    L.append(f"| {fam} | {v['config']} | {sel['families'][fam]['cv_gain_pp']:+.2f} | {s['all']['region']:.2f} | "
             f"{s['all']['pointing']:.2f} | {s['person']['region']:.2f} | {s['thing']['region']:.2f} | "
             f"{s['stuff']['region']:.2f} | [{ci[0]:+.2f}, {ci[1]:+.2f}] |")
L += ["", "## B aşaması (SAM)", "", "| Tür | B taban | B ÇD | kazanç | kural |", "|---|---|---|---|---|"]
for k, v in boxes["kinds"].items():
    L.append(f"| {k} | {v['base']:.2f} | {v['cv']:.2f} | {v['cv_gain_pp']:+.2f} | {boxes['rule'][k]} |")
L += ["", "## Look v4, validation", "", "| | hepsi | person | thing | stuff |", "|---|---|---|---|---|"]
for name, key in (("v3", "v3"), ("A sonrası", "stage_a"), ("v4", "final"), ("SAM üst sınır", "sam_any_mask_upper_bound")):
    d = fin[key]; L.append(f"| {name} | " + " | ".join(f"{d[k]['region']:.2f}" for k in ("all", "person", "thing", "stuff")) + " |")
L += ["", "| v4 − v3 | fark (puan) | GA95 |", "|---|---|---|"]
for k, v in fin["bootstrap_final_vs_v3"].items():
    L.append(f"| {k} | {v['delta_pp']:+.2f} | [{v['ci95_pp'][0]:+.2f}, {v['ci95_pp'][1]:+.2f}] |")
out = config.ROOT / "docs" / "generated" / "LOOK_E3_METRICS.md"
out.write_text("\n".join(L) + "\n"); print(out)
