import pytest
from itertools import permutations

from lcr import pilot

TL = "0.0000:0.0000:0.4000:0.4000"          # top-left box
TL2 = "0.0500:0.0500:0.4000:0.4000"         # overlaps TL with IoU 0.766
BR = "0.6000:0.6000:1.0000:1.0000"          # bottom-right, disjoint from TL


def row(label="incorrect", error_type="misidentified", box=""):
    return dict(label=label, error_type=error_type, box=box)


def test_parse_box_rejects_empty_and_degenerate():
    assert pilot.parse_box("") is None and pilot.parse_box(None) is None
    assert pilot.parse_box("0.5:0.5:0.4:0.9") is None                # x1 < x0
    assert pilot.parse_box(TL) == [0., 0., .4, .4]


def test_error_type_agreement_then_adjudicator_then_uncertain():
    assert pilot.final_error_type([row(), row()]) == ("misidentified", "agreement")
    split = [row(), row(error_type="invented")]
    assert pilot.final_error_type(split) == ("uncertain", "unresolved")
    assert pilot.final_error_type(split, row(error_type="invented")) == ("invented", "adjudicator")
    # label was adjudicated to incorrect: one annotator said correct, so there is no type agreement
    assert pilot.final_error_type([row(label="correct", error_type=""), row()], row(error_type="invented")) == \
        ("invented", "adjudicator")
    assert pilot.final_error_type([row(error_type=""), row(error_type="")]) == ("uncertain", "unresolved")


def test_error_type_is_order_independent():
    entries = [row(), row(error_type="invented"), row()]
    assert len({pilot.final_error_type(list(p), row(error_type="invented")) for p in permutations(entries)}) == 1


def test_box_agreement_is_the_mean_and_order_independent():
    a, b = row(box=TL), row(box=TL2)
    box, source, iou = pilot.final_box([a, b])
    assert source == "agreement" and abs(iou - pilot.box_iou(pilot.parse_box(TL), pilot.parse_box(TL2))) < 1e-12
    assert box == [0.025, 0.025, 0.4, 0.4]
    assert pilot.final_box([b, a])[0] == box


def test_box_disagreement_never_depends_on_csv_order():
    """The reviewed bug: with two different boxes, the first CSV's box won."""
    a, b = row(box=TL), row(box=BR)
    assert pilot.final_box([a, b])[:2] == pilot.final_box([b, a])[:2] == (None, "disagreement")
    assert pilot.final_box([a, b], row(box=BR))[:2] == ([.6, .6, 1., 1.], "adjudicator")
    assert pilot.final_box([b, a], row(box=BR))[:2] == ([.6, .6, 1., 1.], "adjudicator")


def test_box_single_annotator_and_invented_rows_ignored():
    assert pilot.final_box([row(box=TL), row(box="")])[:2] == ([0., 0., .4, .4], "single_annotator")
    assert pilot.final_box([row(error_type="invented", box=BR), row(box=TL)])[:2] == ([0., 0., .4, .4], "single_annotator")
    assert pilot.final_box([row(box=""), row(box="")])[:2] == (None, "no_box")


# ---------------------------------------------------------------- B pilot sampling and scoring
import numpy as np  # noqa: E402


def _rows(actions4, actions2, classes):
    def make(actions):
        return [dict(id=f"train_{i:04d}", draft=f"draft {i}",
                     claims=[dict(cls=c, kind="thing", word=c, rec_att=dict(action=a, substitute="frisbee" if a == "replace" else "",
                                                                       region=0, box=[0., 0., .5, .5]))])
                for i, (a, c) in enumerate(zip(actions, classes))]
    return make(actions4), make(actions2)


def test_pilot_sample_groups_are_disjoint_and_keep_every_s4_contradiction():
    n = 400
    a4 = ["replace"] * 10 + ["remove"] * 15 + ["keep_elsewhere"] * 30 + ["keep"] * 345
    a2 = ["remove"] * 25 + ["keep"] * 30 + ["replace"] * 40 + ["keep"] * 305     # 40 S2-contradicted S4-keeps
    cls = ["ball"] * 55 + ["shirt"] * 20 + ["dog"] * 20 + ["ball"] * 305
    s4, s2 = _rows(a4, a2, cls)
    claims, pop = pilot.pilot_sample(pilot.pilot_pool(s4, s2), np.random.default_rng(0), n_random=50, n_s2_keep=25,
                                     n_elsewhere=10, n_keep=40)
    keys = [(c["id"], c["claim_index"]) for c in claims]
    assert len(keys) == len(set(keys))
    by = {g: [c for c in claims if c["stratum"] == g] for g in pilot.STRATA}
    assert len(by["random"]) == 50 and pop["random"] == (n, 50)
    # every S4 contradiction outside the random draw is in its group
    assert len(by["s4_replace"]) + sum(c["s4_action"] == "replace" for c in by["random"]) == 10
    assert len(by["s4_remove"]) + sum(c["s4_action"] == "remove" for c in by["random"]) == 15
    # disputed keeps: clothing first, then the rest, up to the cap
    disputed = by["s2_contradicted_s4_keep"]
    assert len(disputed) == 25 and all(c["s4_action"] == "keep" and c["s2_action"] in pilot.CONTRADICTED for c in disputed)
    shirts = [c["cls"] == "shirt" for c in disputed]
    assert shirts == sorted(shirts, reverse=True)
    assert len(by["s4_keep"]) == 40 and all(c["s2_action"] not in pilot.CONTRADICTED for c in by["s4_keep"])


def test_pilot_pool_refuses_mismatched_records():
    s4, s2 = _rows(["keep"], ["keep"], ["ball"])
    s2[0]["draft"] = "another draft"
    with pytest.raises(ValueError):
        pilot.pilot_pool(s4, s2)


def test_summarise_strata_precision_harm_and_substitutes():
    claims = ([dict(id=f"r{i}", claim_index=0, stratum="s4_replace") for i in range(4)] +
              [dict(id=f"m{i}", claim_index=0, stratum="s4_remove") for i in range(3)])
    answers = {("r0", 0): dict(label="incorrect", error_type="misidentified", substitute_ok="yes"),
               ("r1", 0): dict(label="incorrect", error_type="misidentified", substitute_ok="no"),
               ("r2", 0): dict(label="correct", substitute_ok="no"),
               ("r3", 0): dict(label="uncertain", substitute_ok="uncertain"),
               ("m0", 0): dict(label="incorrect", error_type="invented"),         # deletion is right
               ("m1", 0): dict(label="incorrect", error_type="misidentified"),    # an object is lost
               ("m2", 0): dict(label="correct")}                                  # a correct word is lost
    s = pilot.summarise_strata(claims, answers)
    assert s["s4_replace"]["error_rate"] == 2 / 3 and s["s4_replace"]["n_uncertain"] == 1
    assert s["s4_replace"]["substitute_right"] == 1 / 3 and s["s4_replace"]["n_substitute_judged"] == 3
    assert s["s4_remove"]["removal_harm"] == 2 / 3 and s["s4_remove"]["error_types"]["invented"] == 1


def test_referent_box_counts_correct_claims_too():
    """The pilot asks for the referent box of correct claims as well (three-box comparison)."""
    assert pilot.final_box([row(label="correct", error_type="", box=TL)])[:2] == ([0., 0., .4, .4], "single_annotator")
    assert pilot.final_answer([row(), row()], None, "label") == ("incorrect", "agreement")
    assert pilot.final_answer([dict(label="correct", substitute_ok="yes"), dict(label="correct", substitute_ok="no")],
                              dict(substitute_ok="no"), "substitute_ok") == ("no", "adjudicator")


def test_box_evidence_follows_s4_rules():
    from lcr import region_evidence as re
    from lcr.vocab import CLASSES, CLASS_INDEX
    M = re.set_matrix(re.CONFUSION_CLEAN)
    d, c, f = CLASS_INDEX["dog"], CLASS_INDEX["cat"], CLASS_INDEX["frisbee"]
    z = np.full((2, len(CLASSES)), -9.)
    z[0, d], z[0, c] = 1., 3.                   # box 0: OWLv2 prefers cat over the claimed dog
    z[1, d] = 2.                                # box 1: a clear dog
    v = re.rival_matrix(z, M, 2.)
    tau = np.full(len(CLASSES), 1.5)
    on0 = pilot.box_evidence(z, v, tau, M, 0, d, true_k=c)
    assert on0 == dict(supports_claim=False, claim_rejected=True, true_is_rival=True, true_name_found=True)
    assert pilot.box_evidence(z, v, tau, M, 1, d)["supports_claim"]
    assert not pilot.box_evidence(z, v, tau, M, 0, d, true_k=f)["true_is_rival"]
    tau[d] = np.nan                             # class without a threshold never supports or is found
    assert not pilot.box_evidence(z, v, tau, M, 1, d)["supports_claim"]


def test_matching_proposal_respects_the_pool():
    boxes = np.array([[0, 0, .5, .5], [0, 0, .45, .45], [.6, .6, 1, 1]], np.float32)
    pool = np.array([False, True, True])
    j, iou = pilot.matching_proposal(boxes, pool, [0, 0, .5, .5])
    assert j == 1 and abs(iou - .81) < 1e-6
    assert pilot.matching_proposal(boxes, np.zeros(3, bool), [0, 0, .5, .5]) == (None, 0.)
