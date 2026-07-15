import numpy as np
import pandas as pd

from pairs_trading.backtest import PairBacktestResult, aggregate_portfolio, run_pair_backtest


def _make_oscillating_cointegrated_series(n=400, seed=0):
    """A synthetic pair whose spread oscillates with a clear amplitude, so the
    z-score reliably crosses entry/exit/stop thresholds many times -- this
    lets us test that the state machine + trade bookkeeping behave sanely
    without needing to hand-derive Kalman filter internals.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    trend = np.cumsum(rng.normal(0, 0.3, n)) + 100
    oscillation = 5 * np.sin(np.linspace(0, 12 * np.pi, n))
    x = pd.Series(trend, index=dates)
    y = pd.Series(trend + oscillation + rng.normal(0, 0.1, n), index=dates)
    return y, x


def _run(commission_bps=5.0, slippage_bps=5.0):
    y, x = _make_oscillating_cointegrated_series()
    return run_pair_backtest(
        y, x, pair_name="TEST/PAIR",
        backtest_start=y.index[100].isoformat(),
        zscore_lookback=20, entry_z=1.5, exit_z=0.5, stop_z=3.0,
        kalman_delta=1e-3, kalman_obs_cov=0.5,
        commission_bps=commission_bps, slippage_bps=slippage_bps,
        kelly_fraction_cap=0.5, kelly_default_fraction=0.1, kelly_min_trades=3,
    )


def test_produces_trades_with_valid_fields():
    result = _run()
    assert len(result.trades) > 0
    for trade in result.trades:
        assert trade.holding_days >= 1
        assert np.isfinite(trade.pnl_pct)
        assert trade.direction in (1, -1)
        assert trade.exit_date >= trade.entry_date


def test_daily_returns_are_finite():
    result = _run()
    assert np.isfinite(result.daily_return.values).all()


def test_transaction_costs_reduce_total_return():
    with_cost = _run(commission_bps=20.0, slippage_bps=20.0)
    without_cost = _run(commission_bps=0.0, slippage_bps=0.0)

    total_with = result_total_pnl(with_cost)
    total_without = result_total_pnl(without_cost)
    assert total_with < total_without


def result_total_pnl(result):
    return sum(t.pnl_pct for t in result.trades)


def test_aggregate_portfolio_combines_pairs():
    dates = pd.date_range("2020-01-01", periods=5, freq="D")
    r1 = PairBacktestResult(pair_name="A/B", trades=[], daily_return=pd.Series([0.01] * 5, index=dates))
    r2 = PairBacktestResult(pair_name="C/D", trades=[], daily_return=pd.Series([0.02] * 5, index=dates))

    portfolio = aggregate_portfolio([r1, r2], max_capital_per_pair=0.5)
    # equal-weighted at 0.5 each: 0.5*0.01 + 0.5*0.02 = 0.015
    assert np.allclose(portfolio.values, 0.015)


def test_aggregate_portfolio_empty():
    portfolio = aggregate_portfolio([], max_capital_per_pair=0.5)
    assert portfolio.empty
