"""B-only audit of frozen Check v4 alternatives and the current Retell input path.

No selection, threshold fitting, validation access or test access. Reference mentions
measure image-level lexical support, not whether an alternative can replace a claim.
"""
from collections import Counter
import json

import _common  # noqa: F401
import pandas as pd

from lcr import claims, config, pipeline


def main():
    stem = "A_B_look4"
    source = config.CACHE / "pipeline"
    rows = json.loads((source / f"{stem}_check4.json").read_text())
    original = {r["id"]: r for r in json.loads((source / f"{stem}.json").read_text())}
    # Project only caption columns; never load images or another split.
    cols = [f"caption_{i}" for i in range(5)]
    frames = [pd.read_parquet(p, columns=cols)
              for p in sorted((config.DATA / "flickr8k" / "data").glob("train-*.parquet"))]
    refs = pd.concat(frames, ignore_index=True)
    stats = {k: Counter() for k in ("all", "reference_unsupported", "contradicted",
                                   "contradicted_reference_unsupported", "stuff_reference_unsupported")}
    transitions, pairs = Counter(), Counter()
    mismatched_alt = 0
    th = pipeline.thresholds()
    for row in rows:
        split, idx = row["id"].rsplit("_", 1)
        assert split == "train"
        present = claims.reference_classes(refs.iloc[int(idx)].tolist())
        for ci, c in enumerate(row["claims"]):
            if c["kind"] == "person":
                continue
            rec = c["rec_att"]
            assert rec["check"] == "v4"
            old = original[row["id"]]["claims"][ci]
            assert old["cls"] == c["cls"]
            assert rec["s_alt"] == old["rec_att"]["s_alt"]
            mismatched_alt += rec["alt"] != old["rec_att"]["alt"]
            final = pipeline.finalise(rec, th)
            transitions[(rec["verdict"], final["verdict"])] += 1
            groups = ["all"]
            if not c["in_refs"]:
                groups.append("reference_unsupported")
                if c["kind"] == "stuff":
                    groups.append("stuff_reference_unsupported")
            if rec["verdict"] == "contradicted":
                groups.append("contradicted")
                pairs[(c["cls"], rec["alt"])] += 1
                if not c["in_refs"]:
                    groups.append("contradicted_reference_unsupported")
            for group in groups:
                s = stats[group]
                s["n"] += 1
                s[rec["verdict"]] += 1
                s["alternative_in_references"] += rec["alt"] in present
                s["alternative_already_in_draft"] += any(
                    x["cls"] == rec["alt"] for x in row["claims"])
                s["alternative_is_person"] += rec["alt"] == "person"
    result = dict(split="B", label="automatic reference proxy, not human truth or replacement validity",
                  scope="frozen records only; exploratory B diagnostic; no new model selection",
                  source=f"{stem}_check4.json", groups=stats,
                  raw_alt_label_changed_but_raw_score_retained=mismatched_alt,
                  current_pipeline_thresholds=th,
                  verdict_transitions_after_current_finalise=[dict(before=a, after=b, n=n)
                      for (a, b), n in sorted(transitions.items())],
                  most_common_contradicted_pairs=[dict(claim=a, alternative=b, n=n)
                      for (a, b), n in pairs.most_common(15)])
    path = config.RESULTS / "check_v4_contract_B.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    lines = ["# Check v4 → Retell: B tanısı", "",
             "Üreten: `scripts/review_check_v4_contract.py`. Yalnız B; seçim yapılmadı.",
             "Etiket: otomatik referans proxy'si. Referansta geçmek, doğru ikame olduğunu kanıtlamaz.", "",
             "| Grup | n | supported | contradicted | uncertain | alternatif referansta |",
             "|---|---:|---:|---:|---:|---:|"]
    for group, s in stats.items():
        n = s["n"]
        rate = f"{s['alternative_in_references'] / n:.1%}" if n else "—"
        lines.append(f"| {group} | {n} | {s['supported']} | {s['contradicted']} | {s['uncertain']} | {rate} |")
    lines += ["", f"Alternatif adı değiştiği halde eski ham s_alt kalan kayıt: **{mismatched_alt}**.", "",
              "Mevcut `pipeline.finalise` ile karar geçişleri (yukarıdaki JSON'da kaydedilen eşiklerle):", "",
              "| v4 kararı | finalise sonrası | n |", "|---|---|---:|"]
    for (a, b), n in sorted(transitions.items()):
        lines.append(f"| {a} | {b} | {n} |")
    lines += ["", "Bu sayılar mevcut tüketici yolunun tanısıdır; yeni bir sistem performansı değildir.", ""]
    (config.ROOT / "docs/generated/CHECK_V4_CONTRACT_B.md").write_text("\n".join(lines))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
