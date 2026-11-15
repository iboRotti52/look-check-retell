"""Check v2 error analysis on validation: where do the "contradicted" flags land?

Splits thing/stuff claims into clothing (garments worn by people) / other things / stuff and reports, per group,
the proxy wrong-rate, flag count, precision and recall of the contradicted verdict, and the lift of precision
over the group's base rate. Labels are the automatic reference proxy (in_refs), not human labels.

Writes results/check_errors.json.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REC = ROOT / "data/cache/pipeline/A_validation_look2_check2.json"
OUT = ROOT / "results/check_errors.json"

# garments and worn accessories from lcr/vocab.py "clothing and wearables" (bags, leashes, collars excluded)
CLOTHING = {"shirt", "jacket", "sweater", "hat", "helmet", "dress", "shorts", "pants", "suit", "uniform",
            "costume", "vest", "skirt", "scarf", "bikini", "swimsuit", "glove", "boot", "shoe", "glasses",
            "goggles", "mask", "headband", "bandanna"}


def group(c):
    return "clothing" if c["cls"] in CLOTHING else c["kind"]


def stats(rows):
    n = len(rows)
    wrong = sum(w for w, _ in rows)
    flag = sum(f for _, f in rows)
    tp = sum(w and f for w, f in rows)
    base = wrong / n if n else 0.0
    prec = tp / flag if flag else 0.0
    return {"n": n, "wrong_rate": base, "flagged": flag, "flag_rate": flag / n if n else 0.0,
            "precision": prec, "recall": tp / wrong if wrong else 0.0,
            "lift": prec / base if base else 0.0}


def main():
    recs = json.loads(REC.read_text())
    by = {}
    for r in recs:
        for c in r["claims"]:
            if c["kind"] == "person":
                continue
            by.setdefault(group(c), []).append((not c["in_refs"], c["rec_att"]["verdict"] == "contradicted"))
    out = {k: stats(v) for k, v in sorted(by.items())}
    out["all"] = stats([x for v in by.values() for x in v])
    out["all_but_clothing"] = stats([x for k, v in by.items() if k != "clothing" for x in v])
    out["_note"] = "validation, Check v2 prior+det+cap, attention region; labels = automatic reference proxy"
    OUT.write_text(json.dumps(out, indent=1))
    for k, s in out.items():
        if k.startswith("_"):
            continue
        print(f"{k:17s} n={s['n']:4d} wrong={s['wrong_rate']:.2f} flagged={s['flagged']:3d} "
              f"prec={s['precision']:.2f} rec={s['recall']:.2f} lift={s['lift']:.2f}")


if __name__ == "__main__":
    main()
