"""Rewrite data/cache/pipeline/A_B_lcr4.json with the FROZEN S4 bundle after a display-only fix to
RegionChecker.record (2026-11-19: evidence box and s_alt now belong to the decision, docs/CHECK_S4.md).

No refit: results/lcr_check4_bundle.json is loaded as it is and every record is regenerated through
scripts/45's own generate_records. The script refuses to write unless every decision field (verdict,
action, substitute, reason, alt, s_claim, candidate list) is identical to the stored record; only the
shown box (region/box/excess) and the replace-score display (s_alt, new s_alt_pooled) may change. B only.

    python scripts/48_rewrite_lcr4_records.py              # dry run: report what would change
    python scripts/48_rewrite_lcr4_records.py --overwrite  # old file kept in data/cache/pipeline/archive/
Then: python scripts/39_verify_lcr.py --live
"""
import argparse
import importlib
import shutil
from collections import Counter

import numpy as np

import _common  # noqa: F401
from lcr import cache, config
from lcr import region_evidence as re

P = config.CACHE / "pipeline"
ARCHIVE = P / "archive" / "pre_witness_fix"
DECISION = ("verdict", "action", "substitute", "reason", "alt", "s_claim", "region_indices", "region_weights")
DISPLAY = ("region", "box", "excess", "s_alt")
s45 = importlib.import_module("45_check_s4")


def compare(stored, new):
    changed, examples = Counter(), []
    for old_row, new_row in zip(stored, new, strict=True):
        if (old_row["id"], old_row["draft"]) != (new_row["id"], new_row["draft"]):
            raise RuntimeError(f"row mismatch: {old_row['id']} vs {new_row['id']}")
        for a, b in zip(old_row["claims"], new_row["claims"], strict=True):
            x, y = a["rec_att"], b["rec_att"]
            for f in DECISION:
                if x.get(f) != y.get(f):
                    raise RuntimeError(f"decision field changed, refusing to write: {old_row['id']} {a['cls']} {f}: "
                                       f"{x.get(f)!r} -> {y.get(f)!r}")
            diff = [f for f in DISPLAY if x.get(f) != y.get(f)]
            for f in diff:
                changed[f"{x.get('action')}:{f}"] += 1
            if diff and len(examples) < 40:
                examples.append(dict(id=old_row["id"], cls=a["cls"], action=x.get("action"), fields=diff,
                                     region=[x["region"], y["region"]], s_alt=[x.get("s_alt"), y.get("s_alt")]))
    return changed, examples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--overwrite", action="store_true", help="write A_B_lcr4.json (old file archived first)")
    a = ap.parse_args()
    checker = re.RegionChecker.load(re.CHECK4_BUNDLE_PATH)
    stored = cache.load_json(P / "A_B_lcr4.json")
    assert all(r["id"].startswith("train_") and r["bundle_id"] == checker.bundle["id"] for r in stored)
    rows = cache.load_json(P / "A_B_look4.json")
    maps = np.load(P / "A_B_look4.maps.npz")
    new = s45.generate_records(checker, rows, maps, cache.OwlCache("train"), checker.queries)
    changed, examples = compare(stored, new)
    report = dict(bundle_id=checker.bundle["id"], claims=sum(len(r["claims"]) for r in stored),
                  decisions_identical=True, changed_display_fields=dict(changed), examples=examples,
                  written=bool(a.overwrite))
    if a.overwrite:
        ARCHIVE.mkdir(parents=True, exist_ok=True)
        if (ARCHIVE / "A_B_lcr4.json").exists():
            raise FileExistsError(f"{ARCHIVE / 'A_B_lcr4.json'} exists; move it before rewriting again")
        shutil.copyfile(P / "A_B_lcr4.json", ARCHIVE / "A_B_lcr4.json")
        cache.save_json(new, P / "A_B_lcr4.json")
    cache.save_json(report, config.RESULTS / "lcr4_witness_rewrite.json")
    print({k: v for k, v in report.items() if k != "examples"})


if __name__ == "__main__":
    main()
