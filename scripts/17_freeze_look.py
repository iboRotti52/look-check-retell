"""Freeze saved sink maps under v2 and migrate region-dependent metadata.

    python scripts/17_freeze_look.py --stem A_B_sink --out A_B_look2 --overwrite

Only k=4 maps are available. Earlier k=1/12 results are retained under legacy_look;
current regions_by_k/excess_by_k explicitly contain only k=4. No decoder run.
"""
import argparse
import shutil

import numpy as np

import _common  # noqa: F401
from lcr import cache, check, config, data, look, pipeline


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stem", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--version", default="v2", choices=["v2", "v3"])
    a = ap.parse_args()
    p = config.CACHE / "pipeline"
    source_suffix, target_suffix = ("_sink", "_look2") if a.version == "v2" else ("_look2", "_look3")
    if a.stem not in (f"A_B{source_suffix}", f"A_validation{source_suffix}"):
        ap.error("Input must be the corresponding existing development artifact")
    if a.out != a.stem.removesuffix(source_suffix) + target_suffix:
        ap.error("Output must use the corresponding split and version suffix")
    if a.version == "v3":
        e1 = cache.load_json(config.RESULTS / "look_e1.json")
        e2 = cache.load_json(config.RESULTS / "look_e2.json")
        if not e1.get("sanity_passed") or e2.get("accepted_source") is not None:
            ap.error("This frozen v3 requires accepted E1 rules and no accepted E2 source")
        if e1["chosen"] != config.LOOK_RULES_BY_KIND:
            ap.error("Config differs from the rules actually accepted in E1")
    target = p / f"{a.out}.json"
    if not a.overwrite and (target.exists() or target.with_suffix(".maps.npz").exists()):
        ap.error(f"Output exists: {target}; pass --overwrite to replace it")
    rows = cache.load_json(p / f"{a.stem}.json")
    expected_ids = set(data.ids("B" if a.stem.startswith("A_B_") else "validation"))
    if not rows or any(r["id"] not in expected_ids for r in rows):
        ap.error("Artifact contains an image outside its declared development split")
    z = np.load(p / f"{a.stem}.maps.npz")
    map_values = z["maps"]
    midx = {(int(r), int(c)): i for i, (r, c) in enumerate(z["owner"])}
    expected_owners = {(ri, ci) for ri, row in enumerate(rows) for ci in range(len(row["claims"]))}
    if set(midx) != expected_owners or len(midx) != len(z["maps"]):
        raise ValueError("Map owners must cover all claims exactly once")
    owl = cache.OwlCache(data.parse_id(rows[0]["id"])[0])
    q = pipeline.class_queries()
    out, changed, n, maps_out = [], 0, 0, []
    for ri, r in enumerate(rows):
        prop = owl[data.parse_id(r["id"])[1]]
        scores = check.score(prop, q)
        r2 = dict(r, claims=[], look_metadata=dict(pipeline.look_metadata(True, a.version, available_last_k=[4]),
                                                  source_stem=a.stem))
        for ci, c in enumerate(r["claims"]):
            att = map_values[midx[ri, ci]].astype(np.float32)
            params = None
            if a.version == "v3":
                rule = config.LOOK_RULES_BY_KIND[c["kind"]]
                att = look.smooth_map(att, rule["sigma"])
                params = dict(alpha=rule["alpha"], gamma=rule["gamma"])
            c2 = pipeline.rewrite_look_claim(c, att, prop, scores, region_parameters=params, look_version=a.version)
            maps_out.append(att)
            n += 1
            changed += c2["rec_att"]["region"] != c["rec_att"]["region"]
            r2["claims"].append(c2)
        out.append(r2)
    cache.save_json(out, target)
    if a.version == "v3":
        np.savez_compressed(target.with_suffix(".maps.npz"), maps=np.asarray(maps_out, np.float32), owner=z["owner"])
    else:
        shutil.copyfile(p / f"{a.stem}.maps.npz", target.with_suffix(".maps.npz"))
    print(f"{a.out}: {n} claims; {changed} regions changed from {a.stem}; current metadata migrated")


if __name__ == "__main__":
    main()
