"""Failures at stage boundaries, without models, data downloads, or dataset access."""
import copy
import hashlib
import runpy
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lcr import evidence, pipeline, retell
from lcr import region_evidence as re
from lcr.vocab import CLASSES, CLASS_INDEX


@pytest.fixture
def setup():
    q = np.zeros((len(CLASSES), 2), np.float32)
    q[CLASS_INDEX["frisbee"]] = [2, 0]
    q[CLASS_INDEX["dog"]] = [0, 2]
    prop = dict(boxes=np.array([[.1, .1, .15, .13], [.4, .4, .9, .9]], np.float32),
                objectness=np.ones(2, np.float32), embeds=np.eye(2, dtype=np.float32),
                shift=np.zeros(2, np.float32), scale=np.ones(2, np.float32))
    bundle = dict(classes=CLASSES, query_sha256=hashlib.sha256(q.tobytes()).hexdigest(),
                  mu=[0.] * len(CLASSES), sd=[1.] * len(CLASSES), look=dict(min_patches=0, alpha=.5),
                  pool="max", alternatives="all", thresholds={k:dict(support=1., contradiction=.5) for k in ("thing", "stuff")})
    bundle["id"] = re.fingerprint(bundle)
    checker = re.RegionChecker(bundle, q)
    att = np.ones((24, 24), np.float32) / 576
    return checker, prop, att


def test_small_boxes_reach_checker_without_query_driven_selection(setup):
    checker, prop, att = setup
    top, w = re.candidates(att, prop, "thing", checker.bundle["look"])
    assert set(top) == {0, 1} and np.isclose(w.sum(), 1)
    old, _ = re.candidates(att, prop, "thing", dict(min_patches=2, alpha=.5))
    assert list(old) == [1]
    a = checker.record("dog", att, prop); b = checker.record("frisbee", att, prop)
    assert a["region_indices"] == b["region_indices"]
    assert b["verdict"] == "supported" and b["region"] == 0


def test_invalid_and_zero_attention_boxes_are_not_evidence(setup):
    checker, prop, att = setup
    prop["boxes"][:] = 0
    rec = checker.record("frisbee", att, prop)
    assert rec["verdict"] == "uncertain" and rec["region"] == -1
    assert rec["regions"] == [] and rec["alt"] == ""


def test_relabel_recomputes_same_region_evidence_with_same_checker(setup):
    checker, prop, att = setup
    before = checker.record("ball", att, prop)
    saved = copy.deepcopy(before)
    new = checker.relabel(before, "frisbee", prop)
    direct = checker.record("frisbee", att, prop)
    assert before == saved
    assert new["verdict"] == "supported"
    assert new["region_indices"] == before["region_indices"]
    for k in ("s_claim", "s_alt", "alt", "region", "verdict"):
        assert new[k] == direct[k]
    with pytest.raises(ValueError, match="exact checker"):
        checker.relabel(dict(before, bundle_id="different"), "dog", prop)


def test_retell_keeps_pooled_support_and_separate_local_scores(setup):
    checker, prop, att = setup
    rec = checker.record("frisbee", att, prop)
    assert {r["verdict"] for r in rec["regions"]} == {"supported", "contradicted"}
    row = dict(claims=[dict(cls="frisbee", rec_att=rec, blip_alts=[])])
    feats, info = evidence.features(row, "ours", pipeline.DEFAULT_THRESHOLDS, prop, np.zeros((2, len(CLASSES))))
    assert len(feats) == 2
    assert {f[-1] for f in feats} == {retell.TYPE_IDS["supported"]}
    assert sorted(f[3][0] for f in feats) == [0., 2.]
    assert all(f[3][2] == 0 for f in feats)  # no image-level shortcut
    assert pipeline.finalise(rec, dict(t_excess=100, t_support=100, t_contra=100)) == rec


def test_historical_v4_does_not_lose_decision_or_pair_new_alt_with_old_score():
    rec = dict(check="v4", claim="ball", alt="frisbee", s_claim=.9, s_alt=.8, s_image=.9,
               s_claim_cal=-2., s_alt_cal=3., s_image_cal=4., excess=.1, box=[0,0,1,1], region=0,
               verdict="contradicted")
    assert pipeline.finalise(rec, pipeline.DEFAULT_THRESHOLDS)["verdict"] == "contradicted"
    _, alt, _, scalars, verdict = retell.record_features(rec, np.zeros(512))
    assert alt == CLASS_INDEX["frisbee"] and scalars[:3].tolist() == [-2.,3.,4.]
    with pytest.raises(NotImplementedError):
        pipeline.relabel(rec, "dog", {}, None, [])


def test_false_evidence_reaches_actual_region_tokens_without_mutation(setup):
    checker, prop, att = setup
    rec = checker.record("ball", att, prop)
    saved = copy.deepcopy(rec)
    swapped = evidence.false_evidence([rec], ["cat"], np.random.default_rng(1))[0]
    assert rec == saved
    assert all(r["alt"] == "cat" for r in swapped["regions"] if r["verdict"] == "contradicted")


def test_live_path_uses_shared_checker_and_requests_no_references(setup, monkeypatch):
    checker, prop, att = setup
    row = dict(id="train_0001", draft="a frisbee", claims=[dict(cls="frisbee", kind="thing", rec_att={})], look_metadata={})
    def base(*args, **kwargs):
        assert kwargs["include_references"] is False
        yield copy.deepcopy(row), [att]
    monkeypatch.setattr(pipeline, "run_v4", base)
    monkeypatch.setattr(re.RegionChecker, "load", classmethod(lambda cls, path=None: checker))
    monkeypatch.setattr(pipeline.cache, "OwlCache", lambda _: {1: prop})
    out, _ = next(pipeline.run(SimpleNamespace(), ["train_0001"], region_rule="lcr"))
    assert out["claims"][0]["rec_att"] == checker.record("frisbee", att, prop)
    assert out["bundle_id"] == checker.bundle["id"]


def test_bundle_identity_and_query_identity_fail_loudly(setup):
    checker, prop, att = setup
    with pytest.raises(ValueError, match="fingerprint"):
        re.RegionChecker(dict(checker.bundle, pool="weighted"), checker.queries)
    with pytest.raises(ValueError, match="queries"):
        re.RegionChecker(checker.bundle, checker.queries + 1)


def test_gates_biases_and_norms_are_not_decayed():
    module = torch.nn.Module()
    module.gate_attn = torch.nn.Parameter(torch.zeros(()))
    module.norm = torch.nn.LayerNorm(4)
    module.proj = torch.nn.Linear(4, 4)
    groups = retell.optimiser_groups(module)
    decay = {id(p) for p in groups[0]["params"]}; no_decay = {id(p) for p in groups[1]["params"]}
    assert decay.isdisjoint(no_decay)
    assert id(module.proj.weight) in decay
    assert {id(module.gate_attn), id(module.norm.weight), id(module.proj.bias)} <= no_decay
    assert len(decay | no_decay) == len(list(module.parameters()))


def test_main_retell_training_is_blocked_before_loading_a_model(monkeypatch):
    from lcr import config
    monkeypatch.syspath_prepend(str(config.ROOT / "scripts"))
    monkeypatch.setattr(sys, "argv", ["06_train_retell.py", "--mode", "ours"])
    with pytest.raises(SystemExit) as exc:
        runpy.run_path(str(config.ROOT / "scripts/06_train_retell.py"), run_name="__main__")
    assert exc.value.code == 2
