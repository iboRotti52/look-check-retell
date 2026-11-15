"""Completed Check (five-way decision, veto, idiom): docs/CHECK_TAMAMLAMA_PROTOKOL.md. CPU, no data."""
import hashlib

import numpy as np
import pytest

from lcr import claims
from lcr import region_evidence as re
from lcr.vocab import CLASS_INDEX, CLASSES


def test_decide5_outcomes():
    sc = np.array([2., 0., 0., 0., 0., 0.]); si = np.array([2., 2., 0., 0., 0., 0.])
    sa = np.array([0., 3., 3., 3., 0., 3.]); ss = np.array([0., 3., 3., 0., 0., 0.])
    g = np.array([True, True, True, True, True, False])
    v, a, r = re.decide5(sc, si, sa, ss, g, ts=1., tc=.5)
    assert list(a) == ["keep", "keep_elsewhere", "replace", "remove", "skip", "skip"]
    assert list(v) == ["supported", "uncertain", "contradicted", "contradicted", "uncertain", "uncertain"]
    assert r[-1] == "no_region"
    v, a, _ = re.decide5(0., 0., 3., 3., True, 1., .5)            # scalars
    assert str(v) == "contradicted" and str(a) == "replace"


def test_veto_is_part_of_threshold_fitting():
    rng = np.random.default_rng(0); n = 2000
    ok = rng.random(n) < .7
    sc = np.where(ok, rng.normal(1.5, 1, n), rng.normal(-.5, 1, n)); sa = rng.normal(1, 1, n)
    si = np.full(n, 10.)                                         # claim present everywhere -> no contradiction
    ts, tc = re.fit_thresholds(sc, sa, ok, s_img=si)
    assert tc == -np.inf
    ts2, tc2 = re.fit_thresholds(sc, sa, ok)
    assert np.isfinite(tc2)


def test_camera_idiom():
    assert claims.non_visual("a boy smiles at the camera .", "camera")
    assert claims.non_visual("A girl looking away from the camera", "camera")
    assert not claims.non_visual("a man holding a camera", "camera")
    assert not claims.non_visual("a dog looks at the camera", "dog")


@pytest.fixture
def checker():
    q = np.zeros((len(CLASSES), 3), np.float32)
    q[CLASS_INDEX["frisbee"]] = [2, 0, 0]; q[CLASS_INDEX["ball"]] = [0, 2, 0]; q[CLASS_INDEX["dog"]] = [0, 0, 2]
    bundle = dict(classes=CLASSES, query_sha256=hashlib.sha256(q.tobytes()).hexdigest(), decision="five_way",
                  mu=[0.] * len(CLASSES), sd=[1.] * len(CLASSES), look=dict(min_patches=0, alpha=.5),
                  pool="max", alternatives="all", thresholds={k: dict(support=1., contradiction=.5) for k in ("thing", "stuff")})
    bundle["id"] = re.fingerprint(bundle)
    return re.RegionChecker(bundle, q)


def prop(embeds):
    n = len(embeds)
    boxes = np.array([[.05 + .3 * i, .1, .3 + .3 * i, .5] for i in range(n)], np.float32)
    return dict(boxes=boxes, objectness=np.ones(n, np.float32), embeds=np.asarray(embeds, np.float32),
                shift=np.zeros(n, np.float32), scale=np.ones(n, np.float32))


def att_on(first_box_only=True):
    a = np.zeros((24, 24), np.float32); a[3:12, 2:7] = 1
    return a / a.sum()


def test_record_replace_remove_elsewhere(checker):
    p = prop([[0, 1, 0], [0, 0, 0]])                              # a ball where attention looks
    rec = checker.record("frisbee", att_on(), p)
    assert rec["action"] == "replace" and rec["substitute"] == "ball" and rec["verdict"] == "contradicted"
    assert rec["alt"] == "ball"
    p = prop([[0, 0, 1], [0, 0, 0]])                              # a dog there, no frisbee anywhere
    rec = checker.record("frisbee", att_on(), p)
    assert rec["action"] == "remove" and rec["substitute"] == "" and rec["alt_any"] == "dog"
    p = prop([[0, 0, 1], [0, 0, 0], [1, 0, 0]])                   # frisbee elsewhere in the image
    rec = checker.record("frisbee", att_on(), p)
    assert rec["region_indices"] == [0]                           # attention only on the first box
    assert rec["action"] == "keep_elsewhere" and rec["verdict"] == "uncertain"
    rec = checker.record("camera", att_on(), prop([[0, 0, 1], [0, 0, 0]]), non_visual=True)
    assert rec["action"] == "skip" and rec["reason"] == "non_visual_idiom"


def test_relabel_keeps_five_way(checker):
    p = prop([[0, 1, 0], [0, 0, 0]])
    rec = checker.record("ball", att_on(), p)
    assert rec["action"] == "keep"
    new = checker.relabel(rec, "frisbee", p)
    assert new["action"] == "replace" and new["region_indices"] == rec["region_indices"]
