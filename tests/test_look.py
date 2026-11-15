import numpy as np

from lcr import look


def test_uniform_attention_has_zero_excess_everywhere():
    att = np.full((24, 24), 1 / 576)
    boxes = np.array([[0, 0, 1, 1], [0.1, 0.2, 0.5, 0.9], [0.33, 0.33, 0.41, 0.47]])
    assert np.allclose(look.excess_mass(att, boxes), 0, atol=1e-12)


def test_whole_image_box_scores_zero_and_tight_box_wins():
    att = np.zeros((24, 24)); att[6:12, 6:12] = 1 / 36      # all mass in a 6x6 patch block
    tight = [0.25, 0.25, 0.5, 0.5]
    boxes = np.array([[0, 0, 1, 1], tight, [0.25, 0.25, 0.3, 0.3], [0.6, 0.6, 0.9, 0.9]])
    ex = look.excess_mass(att, boxes)
    assert abs(ex[0]) < 1e-12                               # big box: mass 1 - area 1
    assert np.isclose(ex[1], 1 - 1 / 16)                    # tight box: all mass, small area
    assert ex[1] > ex[2] > 0                                # tiny box inside peak scores lower
    assert ex[3] < 0                                        # box with no attention is negative
    best, _ = look.select_by_attention(att, boxes)
    assert best == 1


def test_partial_patch_overlap_is_area_weighted():
    att = np.zeros((24, 24)); att[0, 0] = 1.0
    half = np.array([[0, 0, 0.5 / 24, 1 / 24]])             # left half of the first patch
    assert np.isclose(look.attention_mass(att, half)[0], 0.5)


def test_min_size_and_threshold():
    att = np.zeros((24, 24)); att[0, 0] = 1.0
    small = np.array([[0, 0, 1 / 24, 1 / 24]])              # 1x1 patch: below the 2x2 minimum
    assert look.select_by_attention(att, small)[0] is None
    ok = np.array([[0, 0, 2 / 24, 2 / 24]])
    assert look.select_by_attention(att, ok)[0] == 0
    assert look.select_by_attention(att, ok, threshold=1.5)[0] is None


def test_owl_padding_is_undone():
    # landscape 500x375: OWLv2 pads the bottom, so y must be rescaled by 500/375
    b = look.owl_to_image(np.array([[0.5, 0.375, 1.0, 0.75]]), 500, 375)
    assert np.allclose(b, [[0, 0, 1, 1]])
    # portrait 300x600: padding on the right
    b = look.owl_to_image(np.array([[0.25, 0.5, 0.5, 1.0]]), 300, 600)
    assert np.allclose(b, [[0, 0, 1, 1]])


def test_iou():
    a = np.array([0, 0, 0.5, 0.5])
    assert np.allclose(look.iou(a, np.array([[0, 0, 0.5, 0.5], [0.25, 0, 0.75, 0.5], [0.5, 0.5, 1, 1]])),
                       [1, 1 / 3, 0])


def test_sink_correction_removes_token_independent_patch():
    """A patch that every position attends to (a sink) must not win after correction."""
    import torch
    from lcr import look
    L, H, T, g = 2, 2, 5, 24
    attn = torch.full((L, H, T, 1 + g * g), 1e-4)
    attn[..., 1 + 0] = 0.5                       # sink patch (0, 0) for every position
    attn[:, :, 2, 1 + 300] = 0.3                 # token at position 3 looks at patch 300
    raw = look.attention_map(attn, position=3, last_k=2)
    fixed = look.attention_map(attn, position=3, last_k=2, sink=True, valid_t=T)
    assert raw.argmax() == 0
    assert fixed.argmax() == 300
    assert abs(fixed.sum() - 1) < 1e-6


def test_select_region_prefers_tight_objectlike_box():
    att = np.zeros((24, 24), np.float32)
    att[4:8, 4:8] = 1.0                                   # the object
    boxes = np.array([[4 / 24, 4 / 24, 8 / 24, 8 / 24],   # tight box on it
                      [0, 0, 1, 1],                        # whole image: same mass, much larger area
                      [4 / 24, 4 / 24, 7 / 24, 7 / 24]],   # most of it
                     np.float32)
    obj = np.array([0.9, 0.9, 0.9], np.float32)
    assert look.select_region(att, boxes, obj)[0] == 0
    obj_low = np.array([0.01, 0.9, 0.9], np.float32)       # a non-object box loses to the object-like partial box
    assert look.select_region(att, boxes, obj_low)[0] == 2
    assert look.select_region(att, np.zeros((0, 4), np.float32), np.zeros(0))[0] is None
