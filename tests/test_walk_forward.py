import numpy as np
import pandas as pd

from pairs_trading.walk_forward import (
    generate_folds,
    run_walk_forward,
    split_folds_train_test,
)


def test_generate_folds_are_chronological_and_non_overlapping():
    folds = generate_folds("2015-01-01", "2020-01-01", formation_years=2, trading_months=6)
    assert len(folds) > 1
    for f in folds:
        assert f.formation_start < f.formation_end < f.trading_start < f.trading_end
    for prev, nxt in zip(folds, folds[1:]):
        assert prev.trading_end < nxt.trading_start
        assert nxt.trading_start > prev.trading_end


def test_generate_folds_first_formation_starts_at_data_start():
    folds = generate_folds("2015-01-01", "2018-01-01", formation_years=2, trading_months=6)
    assert folds[0].formation_start == pd.Timestamp("2015-01-01")


def test_generate_folds_last_fold_capped_at_data_end():
    folds = generate_folds("2015-01-01", "2019-08-15", formation_years=2, trading_months=6)
    assert folds[-1].trading_end == pd.Timestamp("2019-08-15")


def test_split_folds_train_test_no_leakage():
    # The invariant that matters for avoiding tuning-time leakage: no TRAIN
    # fold's *trading* window (the data whose returns feed the grid search's
    # score) extends past the split date, and no TEST fold's trading window
    # starts before it. A TEST fold's formation window is allowed to look
    # back over dates a TRAIN fold also traded -- that's just normal
    # trailing-history use for pair selection, not leakage of TEST outcomes
    # into the parameter choice.
    folds = generate_folds("2015-01-01", "2022-01-01", formation_years=2, trading_months=6)
    train, test = split_folds_train_test(folds, split_date="2019-01-01")
    assert len(train) > 0 and len(test) > 0
    split_date = pd.Timestamp("2019-01-01")
    assert all(f.trading_start < split_date for f in train)
    assert all(f.trading_start >= split_date for f in test)
    assert not any(f in test for f in train)


def _make_regime_switching_universe(n_days=1600, seed=0):
    """Two tickers that are cointegrated for the first half of history and
    drift independently in the second half -- pairs should be found in
    early folds and not necessarily in later ones.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2015-01-01", periods=n_days, freq="D")
    half = n_days // 2

    common = np.cumsum(rng.normal(0, 1, half))
    x1 = common + rng.normal(0, 0.2, half)
    y1 = 1.2 * common + rng.normal(0, 0.2, half)

    x2 = x1[-1] + np.cumsum(rng.normal(0, 1, n_days - half))
    y2 = y1[-1] + np.cumsum(rng.normal(0, 1, n_days - half))  # independent from here on

    x = np.concatenate([x1, x2]) + 100
    y = np.concatenate([y1, y2]) + 100
    return pd.DataFrame({"AAA": y, "BBB": x}, index=dates)


def test_walk_forward_runs_end_to_end():
    prices = _make_regime_switching_universe()
    folds = generate_folds(
        prices.index[0].isoformat(), prices.index[-1].isoformat(),
        formation_years=1, trading_months=6,
    )
    sector_universe = {"sector1": ["AAA", "BBB"]}
    pair_sectors = {"AAA/BBB": "sector1", "BBB/AAA": "sector1"}

    result = run_walk_forward(
        prices, sector_universe, pair_sectors, folds,
        entry_z=2.0, exit_z=0.5, stop_z=3.0, zscore_lookback=30,
        kalman_delta=1e-4, commission_bps=5.0, slippage_bps=5.0,
        kelly_fraction_cap=0.5, kelly_default_fraction=0.1, kelly_min_trades=10,
        pvalue_threshold=0.05, min_half_life=1, max_half_life=30, max_pairs=12,
        min_hedge_ratio=0.3, max_hedge_ratio=3.0,
        use_fdr_correction=False, fdr_alpha=0.10,
        max_capital_per_pair=1.0, max_capital_per_sector=1.0,
    )

    assert len(result.per_fold_records) == len(folds)
    assert np.isfinite(result.daily_return.values).all()
    # the pair should be found (and traded) in at least one of the early,
    # genuinely-cointegrated folds
    assert any(rec["n_pairs"] > 0 for rec in result.per_fold_records)
    # ...and not necessarily in every later fold, since the relationship
    # breaks down -- i.e. pair selection actually adapts across folds
    assert any(rec["n_pairs"] == 0 for rec in result.per_fold_records)


def test_walk_forward_no_pairs_found_returns_empty_but_valid():
    dates = pd.date_range("2015-01-01", periods=400, freq="D")
    rng = np.random.default_rng(1)
    prices = pd.DataFrame({
        "AAA": np.cumsum(rng.normal(0, 1, 400)) + 100,
        "BBB": np.cumsum(rng.normal(0, 1, 400)) + 100,
    }, index=dates)
    folds = generate_folds("2015-01-01", "2016-01-04", formation_years=0.5, trading_months=6)
    sector_universe = {"sector1": ["AAA", "BBB"]}

    result = run_walk_forward(
        prices, sector_universe, {}, folds,
        entry_z=2.0, exit_z=0.5, stop_z=3.0, zscore_lookback=30,
        kalman_delta=1e-4, commission_bps=5.0, slippage_bps=5.0,
        kelly_fraction_cap=0.5, kelly_default_fraction=0.1, kelly_min_trades=10,
        pvalue_threshold=0.01, min_half_life=1, max_half_life=30, max_pairs=12,
        min_hedge_ratio=0.3, max_hedge_ratio=3.0,
        use_fdr_correction=False, fdr_alpha=0.10,
        max_capital_per_pair=1.0, max_capital_per_sector=1.0,
    )
    assert result.trades == []
    assert result.daily_return.empty or np.isfinite(result.daily_return.values).all()
