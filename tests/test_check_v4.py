"""Check v4 helpers (scripts/35_check_v4.py) on synthetic data; CPU only, no caches needed."""
import importlib.util
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


@pytest.fixture(scope="module")
def c4():
    spec = importlib.util.spec_from_file_location("check_v4", ROOT / "scripts" / "35_check_v4.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_alternative_sets(c4):
    from lcr.vocab import CLASS_INDEX, CLASSES, KIND
    for a in ("A1", "A2", "A3"):
        m = c4.eligible("shirt", a)
        assert not m[CLASS_INDEX["shirt"]]
    assert c4.eligible("shirt", "A1")[CLASS_INDEX["person"]]          # A1 = original (person allowed)
    for a in ("A2", "A3"):
        assert not c4.eligible("shirt", a)[CLASS_INDEX["person"]]
    m2, m3 = c4.eligible("ball", "A2"), c4.eligible("ball", "A3")
    assert all(KIND[CLASSES[i]] == "thing" for i in np.flatnonzero(m2))
    assert np.all(m3 <= m2) and m3[CLASS_INDEX["frisbee"]] and not m3[CLASS_INDEX["dog"]]
    assert c4.GROUP["collar"] == "animal_gear" and c4.GROUP["dog"] == "animal" and c4.GROUP["grass"] == "stuff"


def test_rule_form(c4):
    sc = np.array([0.9, 0.01, 0.01, 0.1, 0.01]); sa = np.array([0.0, 0.9, 0.1, 0.9, 0.9])
    si = np.array([0.9, 0.01, 0.01, 0.1, 0.9]); g = np.array([True, True, True, True, False])
    v = c4.rule(sc, sa, si, g, 0.5, 0.05, guard=False)
    assert list(v) == ["supported", "contradicted", "uncertain", "uncertain", "uncertain"]
    v = c4.rule(sc, sa, np.array([0.9, 0.9, 0.0, 0.0, 0.0]), g, 0.5, 0.05, guard=True)
    assert v[1] == "uncertain"                                          # image guard blocks it


def test_thresholds_meet_constraint(c4):
    rng = np.random.default_rng(0); n = 3000
    ok = rng.random(n) < 0.75
    sc = np.clip(0.4 * ok + 0.6 * rng.random(n), 0, 1)
    sa = np.clip(0.5 * ~ok + 0.5 * rng.random(n), 0, 1)
    si = sc + 0.05; g = np.ones(n, bool)
    for target in (0.5, 0.7):
        ts, tc, ta = c4.fit_thresholds(sc, sa, si, g, ok, False, target)
        assert ta == ts and tc <= ts
        v = c4.rule(sc, sa, si, g, ts, tc, False, ta)
        con = v == "contradicted"
        if np.isfinite(tc):
            assert con.sum() >= c4.MIN_N and (~ok[con]).mean() >= target
        q = c4.quality(v, ok)
        j_sup_only = c4.quality(np.where(sc >= ts, "supported", "uncertain"), ok)["J"]
        assert q["J"] >= j_sup_only - 1e-12


def test_infeasible_gives_no_contradictions(c4):
    rng = np.random.default_rng(1); n = 500
    ok = rng.random(n) < 0.9; sc = rng.random(n); sa = rng.random(n)
    ts, tc, _ = c4.fit_thresholds(sc, sa, sc, np.ones(n, bool), ok, False, 0.99)
    assert tc == -np.inf
