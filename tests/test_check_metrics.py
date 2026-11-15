import math

import numpy as np

from lcr import check, corrupt, metrics
from lcr.vocab import CLASSES, CLASS_INDEX


def test_verdict_rules():
    v = lambda sc, sa, has=True, alt="ball": check.verdict(sc, sa, has, alt, 0.2, 0.05)
    assert v(0.5, 0.9) == "supported"            # a strong alternative never overrides support
    assert v(0.01, 0.6) == "contradicted"
    assert v(0.01, 0.1) == "uncertain"           # low claim score alone is not a contradiction
    assert v(0.1, 0.6) == "uncertain"            # claim not clearly absent
    assert v(0.9, 0.0, has=False) == "uncertain" # no region -> uncertain
    assert v(0.01, 0.6, alt="") == "uncertain"


def _prop(k=3):
    return dict(boxes=np.array([[0, 0, .5, .5], [.5, .5, 1, 1], [0, 0, 1, 1]], np.float16)[:k])


def test_record_picks_strongest_other_class():
    scores = np.zeros((3, len(CLASSES)), np.float32)
    scores[0, CLASS_INDEX["ball"]] = 0.02
    scores[0, CLASS_INDEX["frisbee"]] = 0.7
    scores[1, CLASS_INDEX["ball"]] = 0.4
    r = check.make_record("ball", 0, 0.3, _prop(), scores)
    assert (r.alt, r.verdict) == ("frisbee", "contradicted")
    assert math.isclose(r.s_image, 0.4, rel_tol=1e-6)      # image-level score looks at all regions
    r = check.make_record("ball", None, 0.0, _prop(), scores)
    assert (r.region, r.verdict, r.box) == (-1, "uncertain", [])


def test_detector_selection_uses_claim_score():
    scores = np.zeros((3, len(CLASSES)), np.float32)
    scores[:, CLASS_INDEX["dog"]] = [0.1, 0.8, 0.3]
    assert check.select_by_detector(scores, "dog", _prop()["boxes"]) == 1


def test_kappa_and_hicr():
    a = ["correct"] * 8 + ["incorrect"] * 2
    assert metrics.cohen_kappa(a, a) == 1.0
    b = ["correct"] * 7 + ["incorrect"] * 3
    assert 0.7 < metrics.cohen_kappa(a, b) < 0.8             # hand value: (0.9-0.62)/(1-0.62)=0.737
    h = metrics.hicr(["correct"] * 17 + ["incorrect"] * 2 + ["uncertain"])
    assert math.isclose(h["HICR"], 100 * 2 / 19) and math.isclose(h["HICR_uncertain_as_incorrect"], 15.0)


def test_repair_rates():
    t = [("incorrect", "correct"), ("incorrect", "incorrect"),
         ("correct", "correct"), ("correct", "removed"), ("correct", "correct"), ("correct", "correct")]
    r = metrics.repair_rates(t)
    assert (r["fix_rate"], r["break_rate"], r["NRR"]) == (50.0, 25.0, 0.0)


def test_auroc_and_power():
    assert metrics.auroc(np.array([.9, .8]), np.array([.1, .2])) == 1.0
    assert metrics.auroc(np.array([.5, .5]), np.array([.5, .5])) == 0.5
    # textbook check: 50% -> 40% at alpha .05 / power .8 needs about 388 per group
    assert 380 <= metrics.claims_needed(0.5, 0.2) <= 395


def test_auroc_partial_ties_match_pairwise_definition_and_ignore_order():
    pos, neg = np.array([.5, .8, .5]), np.array([.5, .2])
    expected = np.mean((pos[:, None] > neg) + .5 * (pos[:, None] == neg))
    assert metrics.auroc(pos, neg) == expected
    assert metrics.auroc(pos[::-1], neg[::-1]) == expected


def test_bootstrap_ci_contains_point_estimate():
    units = [(1, 4)] * 30 + [(0, 3)] * 70                    # (errors, claims) per image
    rate = lambda u: sum(e for e, _ in u) / sum(c for _, c in u)
    est, lo, hi = metrics.bootstrap_ci(units, rate, n_boot=300)
    assert lo < est < hi


def test_corruption_table_prefers_shared_context():
    refs = ([{"dog", "ball", "grass"}] * 20 + [{"dog", "frisbee", "grass"}] * 20
            + [{"person", "guitar", "stage"}] * 20 + [{"person", "microphone", "stage"}] * 20)
    table = corrupt.confusion_table(refs)
    assert table["ball"][0] == "frisbee" and table["guitar"][0] == "microphone"
    assert "person" not in table                             # person claims are never corrupted
    rng = np.random.default_rng(0)
    assert corrupt.corrupt_label("ball", {"dog", "ball", "grass", "frisbee"}, {"ball": ["frisbee"]}, rng) is None


def test_paired_power_behaves():
    none = metrics.paired_power(1000, 0.08, 0.0, 0.0, n_sim=500)
    weak = metrics.paired_power(300, 0.08, 0.3, 0.01, n_sim=2000)
    strong = metrics.paired_power(1500, 0.08, 0.3, 0.01, n_sim=2000)
    assert none == 0.0 and weak < strong and strong > 0.8
    # equal expected fixes and breaks -> the directional rejection rate stays near alpha / 2
    assert metrics.paired_power(1000, 0.5, 0.05, 0.05, n_sim=2000) < 0.06
