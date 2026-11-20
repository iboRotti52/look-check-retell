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
