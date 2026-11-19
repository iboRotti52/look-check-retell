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


def _rival_checker(tau_value=1.):
    q = np.zeros((len(CLASSES), 3), np.float32)
    q[CLASS_INDEX["dog"]] = [2, 0, 0]; q[CLASS_INDEX["cat"]] = [0, 2, 0]; q[CLASS_INDEX["frisbee"]] = [0, 0, 2]
    bundle = dict(classes=CLASSES, query_sha256=hashlib.sha256(q.tobytes()).hexdigest(), decision="rival_np",
                  mu=[0.] * len(CLASSES), sd=[1.] * len(CLASSES), look=dict(min_patches=0, alpha=.5), pool="max",
                  alternatives="all", tau=[tau_value] * len(CLASSES), centre=[0.] * len(CLASSES), rival_weight=2.,
                  confusion=re.CONFUSION_CLEAN)
    bundle["id"] = re.fingerprint(bundle)
    return re.RegionChecker(bundle, q)


def _two_attended_boxes(embeds):
    """Two proposals inside Look's attention, so both are top-3 candidates."""
    boxes = np.array([[.06, .13, .29, .49], [.09, .15, .27, .45]], np.float32)
    return dict(boxes=boxes, objectness=np.ones(2, np.float32), embeds=np.asarray(embeds, np.float32),
                shift=np.zeros(2, np.float32), scale=np.ones(2, np.float32))


def test_rival_np_keep_shows_the_box_that_supports_the_decision():
    """Reviewed bug (S4 keep): the box with the highest RAW dog logit has a stronger cat rival, so its rival-aware
    score is below tau; the keep is earned by the other box, and that box must be the one shown."""
    ch = _rival_checker(tau_value=1.4)
    p = _two_attended_boxes([[1., 1.2, 0], [.8, 0, 0]])     # box0: dog 2.0, cat 2.4 -> v 1.2; box1: dog 1.6 -> v 1.6
    rec = ch.record("dog", att(), p)
    assert rec["action"] == "keep" and sorted(rec["region_indices"]) == [0, 1]
    assert rec["region"] == 1 and rec["box"] == p["boxes"][1].tolist()
    v = re.rival_matrix(re.proposal_logits(p, ch.queries), ch.M, 2.)[:, CLASS_INDEX["dog"]]
    assert np.isclose(rec["s_claim"], v[rec["region"]]) and v[rec["region"]] >= rec["tau_claim"]


def test_rival_np_replace_s_alt_belongs_to_the_shown_box():
    """Reviewed display bug (S4 replace): the substitute is read on the top-1 box, but s_alt carried the pooled
    max over all candidates, which can sit on another box. Decision unchanged; pooled value kept separately."""
    ch = _rival_checker()
    p = _two_attended_boxes([[0, 0, 0], [0, 0, 0]])
    top, _ = re.candidates(att(), p, "thing", ch.bundle["look"])
    p["embeds"][top[0]] = [0, .75, 0]                        # cat 1.5 on the top-1 box
    p["embeds"][top[1]] = [0, 1.5, 0]                        # cat 3.0 on the other candidate
    rec = ch.record("dog", att(), p)
    assert rec["action"] == "replace" and rec["substitute"] == "cat"
    assert rec["region"] == int(top[0]) and rec["box"] == p["boxes"][top[0]].tolist()
    assert np.isclose(rec["s_alt"], 1.5) and np.isclose(rec["s_alt_pooled"], 3.0)


def _script(name):
    import importlib.util
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(root / "scripts"))
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), root / "scripts" / name)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def test_verifier_witness_contract_rejects_the_pre_fix_records():
    v39 = _script("39_verify_lcr.py")
    ch = _rival_checker(tau_value=1.4)
    p = _two_attended_boxes([[1., 1.2, 0], [.8, 0, 0]])
    rec = ch.record("dog", att(), p)
    z = re.proposal_logits(p, ch.queries); v = re.rival_matrix(z, ch.M, 2.)
    v39.witness_contract("x", rec, z, v)
    with pytest.raises(AssertionError):
        v39.witness_contract("x", dict(rec, region=0), z, v)          # the raw-score box the old code showed
    ch = _rival_checker()
    p = _two_attended_boxes([[0, 0, 0], [0, 0, 0]])
    top, _ = re.candidates(att(), p, "thing", ch.bundle["look"])
    p["embeds"][top[0]] = [0, .75, 0]; p["embeds"][top[1]] = [0, 1.5, 0]
    rec = ch.record("dog", att(), p)
    z = re.proposal_logits(p, ch.queries); v = re.rival_matrix(z, ch.M, 2.)
    v39.witness_contract("x", rec, z, v)
    with pytest.raises(AssertionError):
        v39.witness_contract("x", dict(rec, s_alt=rec["s_alt_pooled"]), z, v)   # the old pooled s_alt


def test_record_rewrite_allows_display_changes_and_refuses_decision_changes():
    import copy
    s48 = _script("48_rewrite_lcr4_records.py")
    rec = dict(verdict="supported", action="keep", substitute="", reason="claim_present", alt="", s_claim=1.6,
               region_indices=[0, 1], region_weights=[.5, .5], region=0, box=[0., 0., .5, .5], excess=.1, s_alt=0.)
    stored = [dict(id="train_0001", draft="a dog", claims=[dict(cls="dog", rec_att=rec)])]
    shown = copy.deepcopy(stored); shown[0]["claims"][0]["rec_att"].update(region=1, box=[.1, .1, .4, .4], excess=.2)
    changed, examples = s48.compare(stored, shown)
    assert changed == {"keep:region": 1, "keep:box": 1, "keep:excess": 1} and examples[0]["region"] == [0, 1]
    decided = copy.deepcopy(stored); decided[0]["claims"][0]["rec_att"]["action"] = "keep_elsewhere"
    with pytest.raises(RuntimeError, match="decision field changed"):
        s48.compare(stored, decided)
