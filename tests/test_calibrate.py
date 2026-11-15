import numpy as np

from lcr import calibrate


def test_fold_is_deterministic_and_balanced():
    ids = [f"train_{i:04d}" for i in range(2000)]
    f = [calibrate.fold_of(i) for i in ids]
    assert f == [calibrate.fold_of(i) for i in ids]
    assert min(np.bincount(f)) > 300


def test_class_offsets_fix_cross_class_scale():
    rng = np.random.default_rng(0)
    # class 0: correct claims score ~0.6, wrong ~0.3; class 1 has the same separation but 10x smaller scores
    n = 400
    y = rng.random(2 * n) < 0.7
    c = np.repeat([0, 1], n)
    base = np.where(y, 0.6, 0.3) * np.where(c == 1, 0.1, 1.0)
    s = np.clip(base * np.exp(rng.normal(0, 0.15, 2 * n)), 1e-3, 0.99)
    raw_auc = calibrate.auroc(s, y)
    cal = calibrate.make("offset", 1.0).fit(s, c, y)
    assert calibrate.auroc(cal(s, c), y) > raw_auc + 0.05
    pct = calibrate.make("percentile").fit(s, c)
    assert calibrate.auroc(pct(s, c), y) > raw_auc + 0.05


def test_cross_fit_never_uses_own_fold():
    rng = np.random.default_rng(1)
    s = rng.uniform(0.2, 0.8, 100)                       # same score distribution in both folds
    c = np.zeros(100, int)
    y = np.r_[np.ones(50), np.zeros(50)]
    folds = np.r_[np.zeros(50, int), np.ones(50, int)]   # each fold contains only one label
    p = calibrate.cross_fit("offset", s, c, y, folds)
    # fold 0 is scored by a model fitted on fold 1 only (all negatives) -> low probabilities
    assert p[:50].max() < 0.5


def test_verdict_quality_informedness():
    ok = np.array([True, True, False, False])
    v = np.array(["supported", "supported", "contradicted", "contradicted"])
    q = calibrate.verdict_quality(v, ok)
    assert np.isclose(q["J"], 2.0) and q["contradicted_precision"] == 1.0
    assert np.isclose(calibrate.auroc([0.1, 0.2, 0.2, 0.9], [0, 1, 0, 1]), 0.875)


def test_finalise_keeps_check_v2_verdict_and_blocks_v1_corruption():
    import pytest
    from lcr import evidence, pipeline
    rec = dict(claim="dog", alt="", s_claim=0.9, s_alt=0.0, s_image=0.9, box=[0, 0, 1, 1], region=0, excess=0.5,
               verdict="contradicted", by="attention", look="v2", check="v2", p_correct=0.1)
    assert pipeline.finalise(rec, pipeline.thresholds())["verdict"] == "contradicted"
    row = dict(claims=[dict(cls="dog", blip_alts=["cat"], rec_att=rec, rec_det=rec)])
    with pytest.raises(NotImplementedError):
        evidence.features(row, "ours", pipeline.thresholds(), {}, np.zeros((0, 169)), rng=np.random.default_rng(0), rho=0.3)


def test_feature_imputation_uses_training_mean_for_every_prediction_batch():
    X = np.array([[0.0], [2.0], [np.nan], [4.0]])
    c, y = np.zeros(4, int), np.array([0, 0, 1, 1])
    model = calibrate.FeatureLogistic().fit(X, c, y)
    assert model.impute[0] == 2.0
    alone = model([[np.nan]], [0])[0]
    in_batch = model([[np.nan], [10000.0]], [0, 0])[0]
    assert alone == in_batch == model([[2.0]], [0])[0]
    # An entirely missing training feature cannot borrow a held-out value.
    missing = calibrate.FeatureLogistic().fit(np.full((4, 1), np.nan), c, y)
    assert missing.impute[0] == 0.0
    assert np.isfinite(missing([[np.nan]], [0])).all()


def test_average_precision_groups_ties_and_does_not_depend_on_input_order():
    assert calibrate.ap_wrong([.5, .5], [False, True]) == .5
    assert calibrate.ap_wrong([.5, .5], [True, False]) == .5
    # First threshold retrieves one wrong claim, second adds one wrong + one correct.
    assert np.isclose(calibrate.ap_wrong([.1, .5, .5], [False, True, False]), 5 / 6)
    assert calibrate.ap_wrong([.1, .2], [True, True]) == 0.0


def test_logistic_gradients_agree_with_finite_differences(monkeypatch):
    from scipy.optimize import check_grad
    real_minimize = calibrate.minimize
    rng = np.random.default_rng(415)
    errors = []

    def inspect(fun, initial, **kwargs):
        point = rng.normal(0, .2, len(initial))
        errors.append(check_grad(lambda w: fun(w)[0], lambda w: fun(w)[1], point))
        return real_minimize(fun, initial, **kwargs)

    monkeypatch.setattr(calibrate, "minimize", inspect)
    X, c, y = rng.normal(size=(40, 3)), rng.integers(0, 4, 40), rng.integers(0, 2, 40)
    for slope in (False, True):
        calibrate.ClassLogistic(.8, slope).fit(1 / (1 + np.exp(-X[:, 0])), c, y)
    for use_class in (False, True):
        calibrate.FeatureLogistic(.8, .3, use_class).fit(X, c, y)
    assert len(errors) == 4 and max(errors) < 1e-5
