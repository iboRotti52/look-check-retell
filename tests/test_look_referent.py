import numpy as np

from lcr import look_referent as lr

G = 24                              # config.BLIP_GRID; look_referent itself takes no dependency on it


def uniform():
    return np.full((G, G), 1 / (G * G))


def peak(i, j):
    a = np.zeros((G, G)); a[i, j] = 1.0
    return a


# ----------------------------------------------------------------------------- box-mass parity
def test_box_mass_matches_lcr_look_attention_mass():
    from lcr import look
    rng = np.random.default_rng(0)
    att = rng.random((G, G)); att /= att.sum()
    box = np.array([0.1, 0.2, 0.6, 0.9], np.float32)
    assert np.isclose(lr.top1_concentration(att, box), look.attention_mass(att, box[None])[0])


def test_excess_mass_matches_lcr_look_excess_mass():
    from lcr import look
    rng = np.random.default_rng(1)
    att = rng.random((G, G)); att /= att.sum()
    box = np.array([0.25, 0.25, 0.5, 0.5], np.float32)
    assert np.isclose(lr.excess_mass(att, box), look.excess_mass(att, box[None])[0])


# ------------------------------------------------------------------------------------- entropy
def test_entropy_uniform_is_log2_n():
    assert np.isclose(lr.entropy(uniform()), np.log2(G * G))


def test_entropy_single_patch_is_zero():
    assert np.isclose(lr.entropy(peak(0, 0)), 0.0)


def test_entropy_all_zero_map_falls_back_to_uniform():
    assert np.isclose(lr.entropy(np.zeros((G, G))), np.log2(G * G))


def test_entropy_is_scale_invariant():
    a = peak(5, 5) + peak(5, 6)
    assert np.isclose(lr.entropy(a), lr.entropy(3 * a))


# -------------------------------------------------------------------------- top1_concentration
def test_top1_concentration_matches_box_mass_fraction():
    att = np.zeros((G, G)); att[6:12, 6:12] = 1 / 36      # all mass in a 6x6 patch block
    box = np.array([0.25, 0.25, 0.5, 0.5], np.float32)
    assert np.isclose(lr.top1_concentration(att, box), 1.0)
    box_half = np.array([0.25, 0.25, 0.375, 0.5], np.float32)
    assert np.isclose(lr.top1_concentration(att, box_half), 0.5)


# -------------------------------------------------------------------------------- null_margin
def test_null_margin_prefers_claim_whose_map_actually_peaks_there():
    box = np.array([0.0, 0.0, 2 / G, 2 / G], np.float32)
    own = peak(0, 0)
    other = uniform()                                      # no preference for the box
    out = lr.null_margin(own, box, [other])
    assert out["margin"] > 0 and out["rank"] == 1 and out["n_other_claims"] == 1


def test_null_margin_detects_when_another_claim_explains_the_box_better():
    box = np.array([0.0, 0.0, 2 / G, 2 / G], np.float32)
    own = uniform()
    other = peak(0, 0)
    out = lr.null_margin(own, box, [other])
    assert out["margin"] < 0 and out["rank"] == 2


def test_null_margin_rank_counts_every_stronger_other_claim():
    box = np.array([0.0, 0.0, 2 / G, 2 / G], np.float32)
    own = uniform()
    others = [peak(0, 0), peak(0, 0), uniform()]            # two strictly stronger, one tied
    out = lr.null_margin(own, box, others)
    assert out["rank"] == 3 and out["n_other_claims"] == 3


def test_null_margin_no_other_claims_is_nan_margin_rank_one():
    box = np.array([0, 0, 2 / G, 2 / G], np.float32)
    out = lr.null_margin(uniform(), box, [])
    assert np.isnan(out["margin"]) and out["rank"] == 1 and out["n_other_claims"] == 0


# ---------------------------------------------------------------------------------- signals()
def test_signals_bundles_all_three_families():
    box = np.array([0.0, 0.0, 2 / G, 2 / G], np.float32)
    att = peak(0, 0)
    out = lr.signals(strength=3.5, att=att, box=box, other_maps=[uniform()])
    assert out["strength"] == 3.5
    assert np.isclose(out["entropy"], 0.0)
    assert np.isclose(out["top1_concentration"], 1.0)
    assert out["null_rank"] == 1 and out["null_n_other_claims"] == 1
