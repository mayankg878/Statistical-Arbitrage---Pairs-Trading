import numpy as np

from pairs_trading.position_sizing import kelly_fraction, position_size


def test_default_fraction_used_below_min_trades():
    f = kelly_fraction([0.01, 0.02], fraction_cap=0.5, default_fraction=0.1, min_trades=10)
    assert f == 0.1


def test_zero_variance_zero_edge_returns_zero():
    # var == 0 and mu == 0 exactly -> the var==0 short-circuit applies
    f = kelly_fraction([0.0] * 20, fraction_cap=1.0, default_fraction=0.1, min_trades=10)
    assert f == 0.0


def test_kelly_matches_hand_computation():
    # Textbook continuous form: f* = mu / sigma^2
    rng = np.random.default_rng(0)
    returns = list(rng.normal(0.01, 0.05, 200))
    mu = np.mean(returns)
    var = np.var(returns)
    expected_full = np.clip(mu / var, 0.0, 1.0)
    f = kelly_fraction(returns, fraction_cap=0.5, default_fraction=0.1, min_trades=10)
    assert np.isclose(f, expected_full * 0.5)


def test_negative_edge_gives_zero_kelly():
    returns = [-0.01] * 20
    f = kelly_fraction(returns, fraction_cap=0.5, default_fraction=0.1, min_trades=10)
    assert f == 0.0


def test_fraction_cap_applied():
    returns = [1.0] * 20  # huge edge, tiny variance would blow past 1.0 pre-clip
    f = kelly_fraction(returns, fraction_cap=0.5, default_fraction=0.1, min_trades=10)
    assert f <= 0.5


def test_position_size_respects_cap():
    size = position_size(capital=100_000, kelly_f=0.8, max_capital_per_pair=0.15)
    assert size == 15_000
