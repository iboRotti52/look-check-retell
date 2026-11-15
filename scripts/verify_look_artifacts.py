"""Verify repaired artifacts and record numerical before/after evidence.

Run after scripts 17 (B/validation), 13, 14, 16 and review_look_v2.
Reads development artifacts only, never test data or annotations.
"""
import numpy as np

import _common  # noqa: F401
from lcr import cache, config, look


def main():
    p, r = config.CACHE / "pipeline", config.RESULTS
    archived = r / "archive" / "pre_consistency_fix"
    result = dict(purpose="measurement/metadata correction; unchanged Look v2 configuration", splits={})
    for stem in ("A_B_look2", "A_validation_look2"):
        before = cache.load_json(p / "archive" / "pre_consistency_fix" / f"{stem}.json")
        after = cache.load_json(p / f"{stem}.json")
        assert [x["id"] for x in before] == [x["id"] for x in after]
        counts = dict(claims=0, region_changes=0, score_changes=0, stale_iou=0, stale_k4=0, stale_excess=0)
        for old, new in zip(before, after):
            meta = new["look_metadata"]
            assert meta["region_rule"] == "v2" and meta["available_last_k"] == [4]
            assert meta["map_correction"] == "caption_mean_subtract_relu"
            assert len(old["claims"]) == len(new["claims"])
            for a, b in zip(old["claims"], new["claims"]):
                counts["claims"] += 1
                x, y = a["rec_att"], b["rec_att"]
                counts["region_changes"] += x["region"] != y["region"] or x["box"] != y["box"]
                counts["score_changes"] += any(x[k] != y[k] for k in ("s_claim", "s_alt", "s_image", "alt", "verdict"))
                expected_iou = float(look.iou(np.asarray(y["box"]), np.asarray(b["rec_det"]["box"])[None])[0]) \
                    if y["box"] and b["rec_det"]["box"] else 0.0
                counts["stale_iou"] += not np.isclose(expected_iou, b["iou_att_det"], atol=1e-7)
                counts["stale_k4"] += b["regions_by_k"] != {"4": y["region"]}
                counts["stale_excess"] += b["excess_by_k"] != {"4": y["excess"]}
                assert b["legacy_look"]["region_rule"] == "v1"
        assert all(v == 0 for k, v in counts.items() if k != "claims"), counts
        result["splits"][stem] = counts
    changes = {}
    for stem in ("A_validation", "A_validation_sink", "A_validation_look2"):
        name = f"look_benchmark_{stem}.json"
        old, new = cache.load_json(archived / name), cache.load_json(r / name)
        for kind in ("all", "person", "thing", "stuff"):
            assert old[kind]["n"] == new[kind]["n"]
            for key in ("pointing", "region_att", "region_det", "region_oracle"):
                assert np.isclose(old[kind][key], new[kind][key]), (stem, kind, key)
        changes[stem] = {key: dict(before=old["all"][key], after=new["all"][key],
                                  delta=new["all"][key] - old["all"][key])
                         for key in ("pointing_chance", "energy", "region_att")}
    result["benchmark_changes"] = changes
    name = "look_regions_A_validation_sink.json"
    old, new = cache.load_json(archived / name), cache.load_json(r / name)
    assert old["chosen_cfg"] == new["chosen_cfg"]
    result["region_configuration_unchanged"] = new["chosen_cfg"]
    result["auroc_changes"] = []
    for a, b in zip(old["report_rows"], new["report_rows"]):
        assert a["cfg"] == b["cfg"]
        for split in ("select", "report"):
            assert a[split]["region"] == b[split]["region"]
        result["auroc_changes"].append(dict(cfg=b["cfg"],
            before=a["report"]["auroc_thing_stuff"], after=b["report"]["auroc_thing_stuff"]))
    audit = cache.load_json(r / "look_v2_review_B.json")
    assert not any(audit["cache_metadata"].values()), audit["cache_metadata"]
    sanity = cache.load_json(r / "look_sanity_A_validation_look2.json")
    assert sanity["pointing"] > sanity["transposed"]
    assert sanity["swap_own"] > sanity["swap_other"]
    result["sanity"] = sanity
    cache.save_json(result, r / "look_consistency_fix.json")
    print(result)


if __name__ == "__main__":
    main()
