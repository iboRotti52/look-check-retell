import numpy as np

from lcr import look_v4


def test_residual_strength_is_the_pre_normalisation_sum_of_selected_map():
    """selected_map clips+normalises to sum 1; residual_strength returns the sum before that
    division, and must not change selected_map's own output."""
    rng = np.random.default_rng(0)
    heads = np.asarray(rng.normal(size=(144, 576)), np.float16)
    sel = [3, 17, 40, 101]
    strength = look_v4.residual_strength(heads, sel)
    m = look_v4.selected_map(heads, sel)
    x = np.clip(np.asarray(heads, np.float32).reshape(144, 576)[sel].mean(0), 0, None)
    assert np.isclose(strength, x.sum())
    assert np.isclose(m.sum(), 1.0)                 # selected_map itself still normalises


def test_residual_strength_is_zero_for_an_all_negative_map():
    heads = -np.ones((144, 576), np.float16)
    assert look_v4.residual_strength(heads, [0, 1]) == 0.0
    m = look_v4.selected_map(heads, [0, 1])          # selected_map still falls back to uniform
    assert np.allclose(m, 1 / 576)
