"""Gate for operation 5b (docs/LITERATUR_TARAMASI.md): are the captioner's alternatives useful?

    python scripts/15_alt_quality.py --pipe A_validation_sink

For thing/stuff draft claims that no reference mentions (automatic proxy for 'wrong'), how often is the
captioner's top-1 / any top-3 alternative class mentioned by a reference? Low values cap what
'contradicted -> use the alternative' can fix. The alternative may also be mentioned elsewhere in the
draft itself; that share is reported separately.
"""
import argparse

import _common  # noqa: F401
from lcr import cache, claims, config, data

ap = argparse.ArgumentParser()
ap.add_argument("--pipe", default="A_validation_sink")
a = ap.parse_args()
assert "test" not in a.pipe, "the test split is not used during development"
rows = cache.load_json(config.CACHE / "pipeline" / f"{a.pipe}.json")
n = hit1 = hit3 = in_draft = 0
for r in rows:
    pres = claims.reference_classes(data.references(r["id"]))
    drafted = {c["cls"] for c in r["claims"]}
    for c in r["claims"]:
        if c["kind"] == "person" or c["in_refs"]:
            continue
        n += 1
        alts = c["blip_alts"]
        hit1 += bool(alts) and alts[0] in pres
        hit3 += any(x in pres for x in alts)
        in_draft += bool(alts) and alts[0] in drafted
out = dict(pipe=a.pipe, unsupported_claims=n, top1_in_refs=100 * hit1 / n, top3_in_refs=100 * hit3 / n,
           top1_already_in_draft=100 * in_draft / n, label="reference proxy (automatic)")
cache.save_json(out, config.RESULTS / f"alt_quality_{a.pipe}.json")
print(out)
