import numpy as np
import pandas as pd

from pairs_trading.pair_selection import find_cointegrated_pairs, half_life_of_mean_reversion


def _make_cointegrated_series(n=600, seed=0):
    rng = np.random.default_rng(seed)
    common = np.cumsum(rng.normal(0, 1, n))
    x = common + rng.normal(0, 0.2, n)
    y = 1.3 * common + rng.normal(0, 0.2, n)
    return x, y


def _make_independent_series(n=600, seed=0):
    rng = np.random.default_rng(seed)
    x = np.cumsum(rng.normal(0, 1, n))
    y = np.cumsum(rng.normal(0, 1, n + 1))[1:]  # independent random walk
    return x, y


def test_finds_cointegrated_pair():
    dates = pd.date_range("2020-01-01", periods=600, freq="D")
    x, y = _make_cointegrated_series()
    prices = pd.DataFrame({"AAA": y, "BBB": x}, index=dates)

    candidates = find_cointegrated_pairs(
        prices, {"sector1": ["AAA", "BBB"]}, pvalue_threshold=0.05,
        min_half_life=0, max_half_life=1000, max_pairs=5,
    )
    assert len(candidates) == 1
    assert candidates[0].pvalue < 0.05


def test_rejects_independent_pair():
    dates = pd.date_range("2020-01-01", periods=600, freq="D")
    x, y = _make_independent_series()
    prices = pd.DataFrame({"CCC": y, "DDD": x}, index=dates)

    candidates = find_cointegrated_pairs(
        prices, {"sector1": ["CCC", "DDD"]}, pvalue_threshold=0.05,
        min_half_life=0, max_half_life=1000, max_pairs=5,
    )
    assert len(candidates) == 0


def test_only_within_sector_pairs_considered():
    dates = pd.date_range("2020-01-01", periods=600, freq="D")
    x, y = _make_cointegrated_series()
    prices = pd.DataFrame({"AAA": y, "BBB": x, "ZZZ": x * 2}, index=dates)

    candidates = find_cointegrated_pairs(
        prices, {"s1": ["AAA", "BBB"], "s2": ["ZZZ"]}, pvalue_threshold=0.05,
        min_half_life=0, max_half_life=1000, max_pairs=5,
    )
    pairs_found = {(c.y, c.x) for c in candidates}
    assert all("ZZZ" not in p for p in pairs_found)


def test_half_life_positive_for_mean_reverting_series():
    rng = np.random.default_rng(3)
    n = 500
    spread = np.zeros(n)
    for t in range(1, n):
        spread[t] = 0.9 * spread[t - 1] + rng.normal(0, 1)
    hl = half_life_of_mean_reversion(pd.Series(spread))
    assert 0 < hl < 100


def test_half_life_large_or_infinite_for_random_walk():
    # A true random walk has kappa == 0, but finite-sample OLS bias on a
    # unit-root process can yield a small spurious negative kappa (the same
    # bias behind the Dickey-Fuller distribution), so we only require the
    # implied half-life to be long, not exactly infinite.
    rng = np.random.default_rng(4)
    spread = pd.Series(np.cumsum(rng.normal(0, 1, 500)))
    hl = half_life_of_mean_reversion(spread)
    assert hl == np.inf or hl > 100
