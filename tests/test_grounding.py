import numpy as np

from lcr import grounding

SENT = ("[/EN#28094/people the boy] laying [/EN#28096/bodyparts face] down on [/EN#28093/other a skateboard] "
        "is being pushed along [/EN#0/notvisual the ground] by [/EN#28095/people another boy] .")
XML = """<annotation><size><width>500</width><height>250</height><depth>3</depth></size>
<object><name>28093</name><name>28095</name><bndbox><xmin>1</xmin><ymin>1</ymin><xmax>250</xmax><ymax>125</ymax></bndbox></object>
<object><name>28092</name><scene>1</scene></object></annotation>"""


def test_parse_sentences_and_boxes():
    ph = grounding.parse_sentences(SENT)
    assert ph[0] == ("28094", ["people"], "the boy")
    assert ("0", ["notvisual"], "the ground") in ph
    (w, h), boxes = grounding.parse_boxes(XML)
    assert (w, h) == (500, 250)
    assert np.allclose(boxes["28093"][0], [0, 0, 0.5, 0.5])      # 1-based VOC corner -> 0
    assert "28092" not in boxes                                     # scene entity has no box


def test_phrase_class_uses_head_noun():
    assert grounding.phrase_class("a water bottle") == "bottle"
    assert grounding.phrase_class("Two small children") == "person"
    assert grounding.phrase_class("the ground") is None


def test_coverage_mask_and_point():
    m = grounding.coverage_mask(np.array([[0, 0, 0.5, 0.5]], np.float32), g=4)
    assert np.isclose(m.sum(), 4) and m[0, 0] == 1 and m[3, 3] == 0
    assert grounding.point_in_boxes(0.25, 0.25, np.array([[0, 0, 0.5, 0.5]]))
    assert not grounding.point_in_boxes(0.75, 0.25, np.array([[0, 0, 0.5, 0.5]]))


def test_union_adds_disjoint_fragments_in_the_same_patch():
    boxes = np.array([[0, 0, .5 / 24, 1 / 24], [.5 / 24, 0, 1 / 24, 1 / 24]])
    m = grounding.coverage_mask(boxes)
    assert np.isclose(m[0, 0], 1)
    assert np.isclose(m.sum(), 1)


def test_union_counts_overlaps_once_and_preserves_geometric_area():
    boxes = np.array([[0, 0, .6, .6], [.4, .4, 1, 1]])
    for grid in (1, 7, 24):
        m = grounding.coverage_mask(boxes, g=grid)
        assert np.isclose(m.mean(), .68)
        assert np.all((0 <= m) & (m <= 1))
        assert np.allclose(m, grounding.coverage_mask(np.vstack([boxes, boxes[0]]), g=grid))


def test_union_handles_empty_degenerate_and_clipped_boxes():
    assert not grounding.coverage_mask(np.empty((0, 4))).any()
    boxes = np.array([[-1, -1, .5, .5], [1, 1, 0, 0], [.2, .2, .2, .9]])
    assert np.isclose(grounding.coverage_mask(boxes).mean(), .25)
