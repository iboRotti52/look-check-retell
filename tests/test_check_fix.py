"""Positive-anchored Check (docs/CHECK_DUZELTME_PROTOKOL.md). CPU, no data."""
import hashlib

import numpy as np
import pytest

from lcr import region_evidence as re
from lcr.vocab import CLASS_INDEX, CLASSES, KIND


def test_confusion_sets_are_exclusive_and_valid():
    assert re.CONFUSION_OF["shirt"] == re.CONFUSION_OF["jacket"] != re.CONFUSION_OF["uniform"]
    assert re.CONFUSION_OF["pants"] != re.CONFUSION_OF["shirt"]
    m = re.confusion_mask("frisbee")
    assert m[CLASS_INDEX["ball"]] and not m[CLASS_INDEX["frisbee"]] and not m[CLASS_INDEX["dog"]]
    assert not re.confusion_mask("camera").any()                  # outside every set: no replace
    assert all(KIND[c] != "person" for c in re.CONFUSION_OF)


def test_decide_pa_outcomes():
    s_reg = np.array([2., 0., 0., 0., 0., 0., 0.]); s_img = np.array([2., 2., 0., 0., 0., 0., 0.])
    sub = np.array([False, False, True, False, False, True, False])
    g = np.array([True, True, True, True, True, True, False]); ab = np.array([False, False, False, False, True, True, False])
    v, a, r = re.decide_pa(s_reg, s_img, sub, g, 1., ab)
    assert list(a) == ["keep", "keep_elsewhere", "replace", "remove", "skip", "skip", "skip"]
    assert list(v) == ["supported", "uncertain", "contradicted", "contradicted", "uncertain", "uncertain", "uncertain"]
    assert r[4] == "abstain_class" and r[6] == "no_region"
    v, a, _ = re.decide_pa(0., 0., False, True, 1., False)
    assert str(a) == "remove"


def test_fit_positive_anchored():
    rng = np.random.default_rng(0); n = 3000; C = len(CLASSES)
    men = rng.random((n, C)) < .05
    base = rng.normal(0, 1, C)                                   # class-specific offset, as raw OWLv2 logits have
    im = base + rng.normal(-3, 1, (n, C)) + 4 * men
    j = CLASS_INDEX["dog"]
    tau, ab = re.fit_positive_anchored(im, men, .05)
    pos = im[men[:, j], j]
    assert abs((pos < tau[j]).mean() - .05) < .03                # bounds the false-contradiction rate per class
    assert not ab[j]
    men[:, CLASS_INDEX["camera"]] = False; men[:3, CLASS_INDEX["camera"]] = True
    tau, ab = re.fit_positive_anchored(im, men, .05)
    assert ab[CLASS_INDEX["camera"]]                             # too few positives -> never contradicted
    im[:, CLASS_INDEX["suit"]] = rng.normal(0, 1, n)            # detector blind to the class -> abstain
    tau, ab = re.fit_positive_anchored(im, men, .05)
    assert ab[CLASS_INDEX["suit"]]


def test_substitute_rule():
    tau = np.zeros(len(CLASSES)); ab = np.zeros(len(CLASSES), bool)
    lg = np.full(len(CLASSES), -5.); k = CLASS_INDEX["frisbee"]
    lg[CLASS_INDEX["ball"]] = 1.; lg[CLASS_INDEX["dog"]] = 3.         # dog is not in frisbee's set
    assert re.substitute(lg, k, tau, ab) == CLASS_INDEX["ball"]
    lg[k] = 2.                                                        # claim scores higher on that box
    assert re.substitute(lg, k, tau, ab) is None
    lg[k] = -5.; ab[CLASS_INDEX["ball"]] = True
    assert re.substitute(lg, k, tau, ab) is None


@pytest.fixture
def checker():
    q = np.zeros((len(CLASSES), 3), np.float32)
    q[CLASS_INDEX["frisbee"]] = [2, 0, 0]; q[CLASS_INDEX["ball"]] = [0, 2, 0]; q[CLASS_INDEX["dog"]] = [0, 0, 2]
    bundle = dict(classes=CLASSES, query_sha256=hashlib.sha256(q.tobytes()).hexdigest(), decision="positive_anchored",
                  mu=[0.] * len(CLASSES), sd=[1.] * len(CLASSES), look=dict(min_patches=0, alpha=.5),
                  pool="max", alternatives="all", tau=[1.] * len(CLASSES), abstain=["camera"])
    bundle["id"] = re.fingerprint(bundle)
    return re.RegionChecker(bundle, q)


def prop(embeds):
    n = len(embeds)
    boxes = np.array([[.05 + .3 * i, .1, .3 + .3 * i, .5] for i in range(n)], np.float32)
    return dict(boxes=boxes, objectness=np.ones(n, np.float32), embeds=np.asarray(embeds, np.float32),
                shift=np.zeros(n, np.float32), scale=np.ones(n, np.float32))


def att():
    a = np.zeros((24, 24), np.float32); a[3:12, 2:7] = 1
    return a / a.sum()


def test_record_positive_anchored(checker):
    rec = checker.record("frisbee", att(), prop([[0, 1, 0], [0, 0, 0]]))       # a ball where Look looks
    assert rec["action"] == "replace" and rec["substitute"] == "ball" and rec["score_scale"] == "raw_logit"
    assert rec["region"] == rec["region_indices"][0] and rec["s_claim"] == 0.
    rec = checker.record("frisbee", att(), prop([[0, 0, 1], [0, 0, 0]]))       # a dog there: not a frisbee confusion
    assert rec["action"] == "remove" and rec["substitute"] == "" and rec["alt_any"] == "dog"
    rec = checker.record("frisbee", att(), prop([[0, 0, 1], [0, 0, 0], [1, 0, 0]]))
    assert rec["action"] == "keep_elsewhere"
    rec = checker.record("frisbee", att(), prop([[1, 0, 0], [0, 0, 0]]))
    assert rec["action"] == "keep" and rec["verdict"] == "supported"
    rec = checker.record("camera", att(), prop([[0, 0, 1], [0, 0, 0]]))         # abstaining class
    assert rec["action"] == "skip" and rec["reason"] == "abstain_class"
    new = checker.relabel(checker.record("ball", att(), prop([[0, 1, 0], [0, 0, 0]])), "frisbee", prop([[0, 1, 0], [0, 0, 0]]))
    assert new["action"] == "replace" and new["substitute"] == "ball"


# ---------------- S4 (docs/CHECK_S4_PROTOKOL.md) ----------------
from lcr import claims


def test_clean_sets_and_compounds():
    assert "boat" not in re.CONFUSION_CLEAN.get("boat", []) and "kayak" in re.CONFUSION_CLEAN["boat"]
    assert "vest" not in re.CONFUSION_CLEAN["upper"] and "swimsuit" not in re.CONFUSION_CLEAN["full_body"]
    M = re.set_matrix(re.CONFUSION_CLEAN)
    assert M[CLASS_INDEX["dog"], CLASS_INDEX["cat"]] and not M[CLASS_INDEX["dog"], CLASS_INDEX["dog"]]
    assert not M[CLASS_INDEX["shirt"], CLASS_INDEX["vest"]]
    assert claims.non_visual("a man eating a hot dog", "dog", compounds=True)
    assert not claims.non_visual("a man eating a hot dog", "dog")
    assert claims.non_visual("a boy rides a dirt bike", "bicycle", compounds=True)
    assert not claims.non_visual("a boy rides a bike", "bicycle", compounds=True)


def test_rival_matrix():
    M = re.set_matrix(re.CONFUSION_CLEAN); lg = np.full((2, len(CLASSES)), -5.)
    d, c = CLASS_INDEX["dog"], CLASS_INDEX["cat"]
    lg[0, d], lg[0, c] = 1., 3.                     # box 0: OWLv2 prefers cat
    lg[1, d] = 0.5
    v = re.rival_matrix(lg, M, 2.)
    assert v[0, d] == 1. - 2 * 2. and v[0, c] == 3. and v[1, d] == .5
    assert v[0, CLASS_INDEX["camera"]] == -5.       # no rivals: raw logit


def test_fit_rival_np_cost_sensitive():
    rng = np.random.default_rng(0); n = 4000; C = len(CLASSES)
    cnt = (rng.random((n, C)) < .05) * rng.integers(1, 4, (n, C)); men = cnt > 0
    X = rng.normal(-4, 1, (n, C)) + 4 * men
    tau_lo = re.fit_rival_np(X, men, cnt, lam=1.); tau_hi = re.fit_rival_np(X, men, cnt, lam=50.)
    j = CLASS_INDEX["dog"]
    assert tau_hi[j] < tau_lo[j]                    # a larger lambda protects more positives
    assert (X[men[:, j], j] < tau_hi[j]).mean() < .05
    cnt[:, CLASS_INDEX["camel"]] = 0; cnt[0, CLASS_INDEX["camel"]] = 1
    assert np.isnan(re.fit_rival_np(X, cnt > 0, cnt, 4.)[CLASS_INDEX["camel"]])


def test_record_rival_np():
    q = np.zeros((len(CLASSES), 3), np.float32)
    q[CLASS_INDEX["dog"]] = [2, 0, 0]; q[CLASS_INDEX["cat"]] = [0, 2, 0]; q[CLASS_INDEX["frisbee"]] = [0, 0, 2]
    tau = [1.] * len(CLASSES); tau[CLASS_INDEX["camel"]] = None
    bundle = dict(classes=CLASSES, query_sha256=hashlib.sha256(q.tobytes()).hexdigest(), decision="rival_np",
                  mu=[0.] * len(CLASSES), sd=[1.] * len(CLASSES), look=dict(min_patches=0, alpha=.5), pool="max",
                  alternatives="all", tau=tau, centre=[0.] * len(CLASSES), rival_weight=2., confusion=re.CONFUSION_CLEAN)
    bundle["id"] = re.fingerprint(bundle)
    ch = re.RegionChecker(bundle, q)
    rec = ch.record("dog", att(), prop([[1, 0, 0], [0, 0, 0]]))
    assert rec["action"] == "keep"
    rec = ch.record("dog", att(), prop([[.6, 1, 0], [0, 0, 0]]))          # dog 1.2 but cat 2.0 at the same box
    assert rec["action"] == "replace" and rec["substitute"] == "cat"      # rival-aware: 1.2 - 2*0.8 < tau
    rec = ch.record("dog", att(), prop([[0, 0, 1], [0, 0, 0]]))
    assert rec["action"] == "remove"
    rec = ch.record("camel", att(), prop([[0, 0, 1], [0, 0, 0]]))
    assert rec["action"] == "skip" and rec["reason"] == "abstain_class"
