"""Check actual B evidence contracts; optionally replay two fixed drafts through BLIP.

No validation/test access, fitting, selection, or changes to pipeline records.
"""
import argparse

import _common  # noqa: F401
import numpy as np

from lcr import cache, config, evidence, pipeline, retell
from lcr import region_evidence as re
from lcr.vocab import CLASS_INDEX


MODES = dict(check4=(re.CHECK4_BUNDLE_PATH, "A_B_lcr4.json", "lcr4_contract.json"),
             check3=(re.CHECK3_BUNDLE_PATH, "A_B_lcr3.json", "lcr3_contract.json"),
             five_way=(re.FIVE_WAY_BUNDLE_PATH, "A_B_lcr2.json", "lcr2_contract.json"),
             historical=(re.HISTORICAL_BUNDLE_PATH, "A_B_lcr.json", "lcr_contract.json"))


def default_mode():
    return next(m for m, (path, _, _) in MODES.items() if path.resolve() == re.BUNDLE_PATH.resolve())


def run(live=False, mode=None):
    """Default: the bundle at region_evidence.BUNDLE_PATH and its B records. Other modes: earlier bundles."""
    mode = mode or default_mode()
    bundle_path, records, contract = MODES[mode]
    checker = re.RegionChecker.load(bundle_path)
    rows = cache.load_json(config.CACHE / "pipeline" / records)
    owl = cache.OwlCache("train")
    split = "fit_ids" in checker.bundle
    if split:
        fit, cal = set(checker.bundle["fit_ids"]), set(checker.bundle["calibration_ids"])
        assert not fit & cal and fit | cal == {r["id"] for r in rows}
    n = tokens = synthetic = 0
    for row in rows:
        assert row["id"].startswith("train_")
        assert row["bundle_id"] == checker.bundle["id"]
        if split:
            assert row["evidence_partition"] == ("fit" if row["id"] in fit else "cal")
        prop = owl[int(row["id"].rsplit("_", 1)[1])]
        feats, _ = evidence.features(row, "ours", pipeline.DEFAULT_THRESHOLDS, prop, None)
        assert len(feats) == sum(max(1, len(c["rec_att"]["regions"])) for c in row["claims"])
        for feat in feats:
            assert np.isfinite(feat[2]).all() and np.isfinite(feat[3]).all()
            assert feat[3][2] == 0
        tokens += len(feats)
        for c in row["claims"]:
            rec = c["rec_att"]
            assert pipeline.finalise(rec, pipeline.DEFAULT_THRESHOLDS) == rec
            for region in rec["regions"]:
                t = region["region"]
                for label, field in ((rec["claim"], "s_claim"), (region["alt"], "s_alt")):
                    if not label:
                        continue
                    k = CLASS_INDEX[label]
                    raw = ((prop["embeds"][t].astype(np.float32) @ checker.queries[k]) + float(prop["shift"][t])) * float(prop["scale"][t])
                    expected = (raw - checker.mu[k]) / checker.sd[k]
                    assert np.isclose(expected, region[field], atol=3e-5), (row["id"], label, expected, region[field])
                assert region["box"] == (rec["box"] if c["kind"] == "person" else prop["boxes"][t].astype(float).tolist())
            if c["kind"] != "person" and n % 40 == 0:
                label = "dog" if c["cls"] != "dog" else "frisbee"
                altered = checker.relabel(rec, label, prop)
                assert altered["claim"] == label and altered["region_indices"] == rec["region_indices"]
                assert altered["bundle_id"] == checker.bundle["id"]
                synthetic += 1
            n += 1
    old = {r["id"]: r for r in cache.load_json(config.CACHE / "pipeline/A_B_look4_check4.json")}
    frisbee = []
    for row in rows:
        for c in row["claims"]:
            if c["cls"] != "frisbee" or not c["in_refs"]:
                continue
            prior = next(x for x in old[row["id"]]["claims"] if x["cls"] == "frisbee")["rec_att"]
            if prior["verdict"] != "supported":
                frisbee.append(dict(id=row["id"], before=prior["verdict"], after=c["rec_att"]["verdict"]))
    result = dict(bundle_id=checker.bundle["id"], images=len(rows), claims=n, evidence_tokens=tokens,
                  synthetic_rescores=synthetic, fit_cal_disjoint=split or None, all_region_scores_independently_checked=True,
                  all_stored_verdicts_preserved=True, image_maximum_excluded=True, frisbee_proxy_cases=frisbee)
    if live:
        if mode != default_mode():
            raise ValueError("live replay runs the pipeline, which uses the default bundle")
        result["live_fixed_draft_replay"] = live_replay(rows)
    result["mode"] = mode
    cache.save_json(result, config.RESULTS / contract)
    print(result)


def live_replay(rows):
    import torch
    from lcr.captioner import Captioner
    from lcr.look_sources import LookDecoder
    decoder = LookDecoder()
    class Replay:
        dev = torch.device("cpu")
        tok = decoder.tok
        bos = decoder.bos
        pad = 0
        encode_captions = Captioner.encode_captions
        word_alignment = Captioner.word_alignment
        decode = Captioner.decode
        def generate(self, emb):
            return decoder.tokens(self.row)
        def cross_attention(self, emb, ids):
            with torch.no_grad(), decoder.capture() as attention:
                out = decoder.forward(ids, emb)
            return torch.stack(attention, 1), out.logits
    cap = Replay()
    results = []
    for saved in [r for r in rows if all(c["kind"] != "person" for c in r["claims"])][:2]:
        cap.row = saved
        row, _ = next(pipeline.run(cap, [saved["id"]], bs=1, region_rule="lcr"))
        assert row["draft"] == saved["draft"]
        drift = 0.
        for a, b in zip(saved["claims"], row["claims"], strict=True):
            x, y = a["rec_att"], b["rec_att"]
            assert x["region_indices"] == y["region_indices"] and x["verdict"] == y["verdict"]
            assert x.get("action") == y.get("action")
            drift = max(drift, abs(x["s_claim"] - y["s_claim"]), abs(x["s_alt"] - y["s_alt"]))
            assert drift < 3e-5
            assert b["in_refs"] is None
        results.append(dict(id=saved["id"], claims=len(row["claims"]), max_score_drift=drift,
                            identical_regions_and_verdicts=True, references_not_used=True))
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--live", action="store_true", help="recompute attention from two fixed B drafts on CPU")
    p.add_argument("--mode", choices=list(MODES), help="default: the bundle at region_evidence.BUNDLE_PATH")
    p.add_argument("--historical", action="store_true", help="alias for --mode historical")
    a = p.parse_args()
    run(a.live, "historical" if a.historical else a.mode)
