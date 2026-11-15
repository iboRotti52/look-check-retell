"""Check v3 selection helpers (scripts/33_check_v3.py) on synthetic data; CPU only."""
import importlib.util
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


@pytest.fixture(scope="module")
def c3():
    spec = importlib.util.spec_from_file_location("check_v3", ROOT / "scripts" / "33_check_v3.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def synthetic(n=600, seed=0):
    rng = np.random.default_rng(seed)
    c = rng.integers(0, 20, n); y = rng.random(n) < 0.7; f = rng.integers(0, 5, n); m = rng.random(n) < 0.8
    X = rng.normal(size=(n, 6)); X[:, 4] += 1.5 * y
    X[rng.random((n, 6)) < 0.02] = np.nan
    return X, c, y, f, m


def test_families_and_groups(c3):
    fams = c3.families()
    assert len(fams) == 48 and fams[0] == ("v2", c3.BASE)
    assert all(g[:3] == c3.BASE for _, g in fams)          # prior first: N_PRIOR offset in Factorized
    assert len({n for n, _ in fams}) == 48


def test_operating_point_precision(c3):
    rng = np.random.default_rng(1)
    ok = rng.random(2000) < 0.7
    p = np.clip(0.5 * ok + 0.5 * rng.random(2000), 0, 1)
    tc, ts = c3.operating_point(p, ok, 0.6)
    assert tc <= ts
    m = p < tc
    assert m.sum() >= 20 and (~ok[m]).mean() >= 0.6


@pytest.mark.parametrize("kind", ["logistic", "factorized", "gbm"])
def test_nested_models_learn_signal(c3, kind):
    X, c, y, f, m = synthetic()
    oof, thr, chosen = c3.nested(X, c, y, f, m, kind)
    assert oof.shape == y.shape and np.all((oof >= 0) & (oof <= 1))
    assert set(thr) == set(c3.TARGETS) and len(chosen) == 5
    from lcr import calibrate
    assert calibrate.auroc(oof[m], y[m]) > 0.75
    v = c3.fold_verdicts(oof, f, thr[0.5])
    assert set(v) <= {"supported", "contradicted", "uncertain"}


def test_quality_coverage(c3):
    p = np.linspace(0, 1, 100); ok = p > 0.3
    q = c3.quality(p, ok)
    assert q["prec_at_cov10"] == 1.0 and q["prec_at_cov30"] == 1.0
