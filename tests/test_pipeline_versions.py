import copy
import runpy
import sys

import numpy as np
import pytest

from lcr import config, pipeline
from lcr.vocab import CLASSES, CLASS_INDEX


def sample():
    prop = dict(boxes=np.array([[0, 0, .5, .5], [.5, .5, 1, 1]], np.float32),
                objectness=np.array([.9, .9], np.float32))
    scores = np.zeros((2, len(CLASSES)), np.float32)
    scores[:, CLASS_INDEX["dog"]] = [.8, .1]
    claim = dict(cls="dog", blip_alts=[], rec_att=dict(region=1, look="v1"),
                 rec_det=dict(region=0), iou_att_det=0.0,
                 regions_by_k={"1": 1, "4": 1, "12": 1},
                 excess_by_k={"1": -.2, "4": -.3, "12": -.4})
    att = np.zeros((24, 24), np.float32)
    att[:12, :12] = 1 / 144
    return claim, att, prop, scores


def test_migration_recomputes_current_fields_and_archives_unavailable_layers():
    c, att, prop, scores = sample()
    before = copy.deepcopy(c)
    out = pipeline.rewrite_look_claim(c, att, prop, scores)
    assert c == before
    assert out["rec_att"]["region"] == 0
    assert out["rec_att"]["look"] == "v2"
    assert out["iou_att_det"] == 1
    assert out["regions_by_k"] == {"4": 0}
    assert out["excess_by_k"]["4"] == out["rec_att"]["excess"]
    assert out["legacy_look"]["regions_by_k"] == before["regions_by_k"]
    assert out == pipeline.rewrite_look_claim(out, att, prop, scores)


def test_migration_clears_stale_selection_when_no_box_is_eligible():
    c, att, prop, scores = sample()
    prop["boxes"] = np.zeros((2, 4), np.float32)
    out = pipeline.rewrite_look_claim(c, att, prop, scores)
    assert out["rec_att"]["region"] == -1
    assert out["rec_att"]["box"] == []
    assert out["regions_by_k"] == {"4": -1}
    assert out["iou_att_det"] == 0


def test_versions_and_map_sources_have_distinct_output_names():
    stems = {pipeline.output_stem("A", "B", sink, rule)
             for sink in (False, True) for rule in ("v1", "v2")}
    assert stems == {"A_B", "A_B_sink", "A_B_raw_look2", "A_B_look2"}
    assert pipeline.look_metadata(True, available_last_k=[4])["available_last_k"] == [4]
    with pytest.raises(ValueError):
        pipeline.output_stem("A", "B", True, "unknown")


def test_corruption_preserves_v2_so_the_old_excess_gate_is_not_reapplied():
    c, att, prop, scores = sample()
    rec = pipeline.rewrite_look_claim(c, att, prop, scores)["rec_att"]
    rec["excess"] = -1
    corrupted = pipeline.relabel(rec, "ball", prop, scores, ["dog"])
    final = pipeline.finalise(corrupted, dict(t_excess=0, t_support=.2, t_contra=.05))
    assert final["look"] == "v2"
    assert final["region"] == rec["region"]


@pytest.mark.parametrize("extension", [".json", ".maps.npz"])
def test_draft_cli_refuses_existing_outputs_before_loading_a_model(tmp_path, monkeypatch, extension):
    output = tmp_path / "pipeline" / f"A_validation_{config.LOOK_VERSION.replace('v', 'look')}{extension}"
    output.parent.mkdir()
    output.write_bytes(b"existing experiment")
    monkeypatch.setattr(config, "CACHE", tmp_path)
    monkeypatch.syspath_prepend(str(config.ROOT / "scripts"))
    monkeypatch.setattr(sys, "argv", ["04_draft_look_check.py", "--split", "validation"])
    with pytest.raises(SystemExit) as exc:
        runpy.run_path(str(config.ROOT / "scripts" / "04_draft_look_check.py"), run_name="__main__")
    assert exc.value.code == 2
    assert output.read_bytes() == b"existing experiment"
