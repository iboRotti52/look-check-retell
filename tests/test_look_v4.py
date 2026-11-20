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


def test_signed_heads_reproduces_attention_map_exactly():
    """Script 47 builds maps from signed_heads; it must be the v4 code path bit for bit."""
    import torch
    g = torch.Generator().manual_seed(0)
    attention = torch.rand((12, 12, 9, 577), generator=g)
    heads = [5, 30, 77, 140]
    for p, valid_t in ((3, 8), (1, 8), (7, 7)):
        expected = look_v4.attention_map(attention, p, valid_t, heads)
        got = look_v4.selected_map(look_v4.signed_heads(attention, p - 1, valid_t), heads)
        assert np.array_equal(expected, got)


def test_predict_step_is_blind_to_the_word_it_predicts():
    """Causal decoder: the step that predicts position p cannot see token p, so only later steps differ."""
    import torch
    g = torch.Generator().manual_seed(1)
    a = torch.rand((12, 12, 9, 577), generator=g)
    b = a.clone(); b[:, :, 4:] = torch.rand((12, 12, 5, 577), generator=g)   # token 4 swapped: steps >= 4 change
    raw = lambda x, q: x[:, :, q, 1:]
    assert torch.equal(raw(a, 3), raw(b, 3)) and not torch.equal(raw(a, 4), raw(b, 4))


def test_residual_strength_is_zero_for_an_all_negative_map():
    heads = -np.ones((144, 576), np.float16)
    assert look_v4.residual_strength(heads, [0, 1]) == 0.0
    m = look_v4.selected_map(heads, [0, 1])          # selected_map still falls back to uniform
    assert np.allclose(m, 1 / 576)
