"""First-principles diagnostics on half B only (no validation/test): what Retell action wrong claims need,
how often the claimed object is detected outside Look's region, alternative quality, the 'camera' idiom, and a
summary of Claude's visual audit (results/claude_visual_audit_B.json; not human labels).
Uses Check v4 features (scripts/35_check_v4.py). Output: results/first_principles_B.json,
docs/generated/FIRST_PRINCIPLES_B.md."""
import collections
import importlib.util
import json
import re

import numpy as np

import _common  # noqa: F401
from lcr import cache, claims, config, data

spec = importlib.util.spec_from_file_location("c4", config.ROOT / "scripts" / "35_check_v4.py")
c4 = importlib.util.module_from_spec(spec); spec.loader.exec_module(c4)
G = c4.GROUP


def main():
    rows, B = c4.features(c4.STEMS["B"])
    refs = {i: claims.reference_classes(data.references(i)) for i in {x["id"] for x in B}}
    X = [x for x in B if x["has_region"]]
    ok = np.array([x["ok"] for x in X]); w = ~ok
    kind = np.array([x["kind"] for x in X]); clothing = np.array([x["group"] == "clothing" for x in X])
    reg = np.array([x["v"][("Q1", "C1", "R1", "A1")][0] for x in X])
    img = np.array([x["v"][("Q1", "C1", "R1", "A1")][3] for x in X])
    sub = np.array([any(G.get(r) == G[x["cls"]] and r != x["cls"] for r in refs[x["id"]]) for x in X])
    out = dict(n_claims_with_region=len(X), n_proxy_wrong=int(w.sum()))
    out["substitution_possible_in_refs"] = {
        "all_wrong": float(sub[w].mean()), "thing": float(sub[w & (kind == "thing")].mean()),
        "stuff": float(sub[w & (kind == "stuff")].mean()), "clothing": float(sub[w & clothing].mean())}
    out["alternative_quality"] = {}
    for a in ("A1", "A2", "A3"):
        alt = [x["v"][("Q1", "C2", "R2", a)][2] for x in X]
        inref = np.array([al in refs[x["id"]] for al, x in zip(alt, X)])
        same = np.array([G.get(al) == G[x["cls"]] for al, x in zip(alt, X)])
        out["alternative_quality"][a] = dict(alt_in_refs=float(inref[w].mean()), alt_same_group=float(same[w].mean()),
                                             alt_in_refs_when_substitution_possible=float(inref[w & sub].mean()))
    alt1 = [x["v"][("Q1", "C2", "R2", "A1")][2] for x in X]
    out["top_A1_alternatives_wrong"] = collections.Counter(a for a, o in zip(alt1, ok) if not o).most_common(10)
    out["presence"] = {}
    for t in (0.1, 0.3):
        lo, hi = reg < t, img >= t
        out["presence"][str(t)] = dict(wrong_detected_in_image=float(hi[w].mean()), wrong_detected_in_region=float((~lo)[w].mean()),
                                       correct_not_in_region_but_in_image=float((lo & hi)[ok].mean()),
                                       correct_nowhere=float((~hi)[ok].mean()))
    absent = w & (img < 0.1)
    out["wrong_absent_everywhere_0.1"] = dict(n=int(absent.sum()), substitution_possible=float(sub[absent].mean()))
    cam = [(c, r["draft"]) for r in rows for c in r["claims"] if c["cls"] == "camera"]
    idiom = [bool(re.search(r"\b(at|to|toward|towards|from|for|into) the camera", d)) for _, d in cam]
    out["camera"] = dict(n=len(cam), idiom=sum(idiom), idiom_and_proxy_wrong=sum(i and not c["in_refs"] for i, (c, _) in zip(idiom, cam)))
    audit = json.loads((config.RESULTS / "claude_visual_audit_B.json").read_text())
    out["claude_visual_audit"] = {k: dict(collections.Counter(v["items"].values()), n=len(v["items"]))
                                  for k, v in audit["samples"].items()}
    cache.save_json(out, config.RESULTS / "first_principles_B.json")
    f = lambda v: f"{v:.3f}"
    L = ["# İlk ilkeler tanısı (yalnız B)", "", "Üreten: `scripts/review_first_principles_B.py`. Validation/test okunmadı.",
         "Etiket otomatik referans proxy'si; görsel denetim Claude'un gözlemi, insan etiketi değil.", "",
         f"Bölgesi olan kişi dışı claim: {len(X)}; proxy-yanlış: {int(w.sum())}.", "",
         "## 1. Yanlış claim'de doğru eylem: yerine koymak mı, silmek mi?",
         "Referanslarda claim'le aynı ikame grubundan başka bir sınıf var mı (varsa yerine koymak mümkün):", "",
         "| grup | oran |", "|---|---|"]
    L += [f"| {k} | {f(v)} |" for k, v in out["substitution_possible_in_refs"].items()]
    L += ["", "## 2. Alternatif önerisinin kalitesi (C2 R2; proxy-yanlış claim'ler)", "",
          "| alternatif kümesi | referansta | aynı grup | ikame mümkünken referansta |", "|---|---|---|---|"]
    L += [f"| {a} | {f(q['alt_in_refs'])} | {f(q['alt_same_group'])} | {f(q['alt_in_refs_when_substitution_possible'])} |"
          for a, q in out["alternative_quality"].items()]
    L += ["", "En sık A1 alternatifleri (proxy-yanlış): " + ", ".join(f"{a} {n}" for a, n in out["top_A1_alternatives_wrong"]),
          "", "## 3. Nesne bölgede değil ama görüntüde mi? (ham OWLv2 olasılığı, Look v4 bölgesi)", "",
          "| eşik | yanlış: görüntüde bulunur | yanlış: bölgede bulunur | doğru: bölgede yok, görüntüde var | doğru: hiçbir yerde |",
          "|---|---|---|---|---|"]
    L += [f"| {t} | {f(q['wrong_detected_in_image'])} | {f(q['wrong_detected_in_region'])} | "
          f"{f(q['correct_not_in_region_but_in_image'])} | {f(q['correct_nowhere'])} |" for t, q in out["presence"].items()]
    a = out["wrong_absent_everywhere_0.1"]
    L += ["", f"Görüntünün hiçbir yerinde bulunmayan (<0.1) proxy-yanlış: {a['n']}; bunlarda ikame mümkün: {f(a['substitution_possible'])}.",
          "", f"'camera' claim'i: {out['camera']['n']}; 'at/to/from the camera' kalıbı: {out['camera']['idiom']}; "
          f"kalıp ve proxy-yanlış: {out['camera']['idiom_and_proxy_wrong']}.", "",
          "## 4. Claude görsel denetimi (proxy-yanlış claim'ler; küçük örnek, insan etiketi değil)", "",
          "| örnek | n | doğru | yanlış | belirsiz | nesne iddiası değil |", "|---|---|---|---|---|---|"]
    for k, c in out["claude_visual_audit"].items():
        L.append(f"| {k} | {c['n']} | {c.get('correct', 0)} | {c.get('wrong', 0)} | {c.get('unclear', 0)} | {c.get('non_visual', 0)} |")
    L += ["", "Görseller: `results/claude_audit_B_detector_confident.png`, `results/claude_audit_B_random.png`."]
    (config.ROOT / "docs/generated/FIRST_PRINCIPLES_B.md").write_text("\n".join(L) + "\n")
    print("wrote docs/generated/FIRST_PRINCIPLES_B.md")


if __name__ == "__main__":
    main()
