import numpy as np
import pandas as pd

from pairs_trading.pair_selection import (
    benjamini_hochberg_threshold,
    find_cointegrated_pairs,
    half_life_of_mean_reversion,
)


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


def test_bh_threshold_hand_computed():
    # m=5, alpha=0.05: eligible k are those with p_(k) <= (k/5)*0.05
    # p = [0.01, 0.02, 0.03, 0.20, 0.50] -> cutoffs [0.01, 0.02, 0.03, 0.04, 0.05]
    # first 3 satisfy p_(k) <= cutoff -> largest surviving p-value is 0.03
    pvalues = [0.5, 0.01, 0.20, 0.03, 0.02]
    threshold = benjamini_hochberg_threshold(pvalues, fdr_alpha=0.05)
    assert np.isclose(threshold, 0.03)


def test_bh_threshold_none_survive():
    threshold = benjamini_hochberg_threshold([0.5, 0.6, 0.7], fdr_alpha=0.05)
    assert threshold == 0.0


def test_bh_threshold_empty():
    assert benjamini_hochberg_threshold([], fdr_alpha=0.05) == 0.0


def test_fdr_correction_controls_false_positives_across_many_independent_pairs():
    # 12 independent tickers -> 66 independent-pair tests, none genuinely
    # cointegrated. A flat p<0.05 screen is expected to let a few through by
    # chance (roughly 0.05 * 66 ~= 3); BH-FDR at the same nominal level
    # should let through no more than that, and typically fewer.
    dates = pd.date_range("2020-01-01", periods=500, freq="D")
    rng = np.random.default_rng(42)
    tickers = [f"T{i}" for i in range(12)]
    prices = pd.DataFrame(
        {t: np.cumsum(rng.normal(0, 1, 500)) + 100 for t in tickers}, index=dates,
    )
    universe = {"sector1": tickers}

    common_kwargs = dict(
        min_half_life=0, max_half_life=1e6, max_pairs=1000,
        min_hedge_ratio=0.0, max_hedge_ratio=1e6,
    )
    raw_candidates = find_cointegrated_pairs(
        prices, universe, pvalue_threshold=0.05, use_fdr_correction=False, **common_kwargs,
    )
    fdr_candidates = find_cointegrated_pairs(
        prices, universe, use_fdr_correction=True, fdr_alpha=0.05, **common_kwargs,
    )
    assert len(fdr_candidates) <= len(raw_candidates)
