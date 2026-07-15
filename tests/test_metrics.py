import numpy as np
import pandas as pd

from pairs_trading.backtest import Trade
from pairs_trading.metrics import (
    average_holding_period,
    cagr,
    max_drawdown,
    sharpe_ratio,
    summarize,
    win_rate,
)


def _trade(pnl_pct, holding_days):
    return Trade(
        pair="A/B", entry_date=pd.Timestamp("2020-01-01"), exit_date=pd.Timestamp("2020-01-05"),
        direction=1, entry_z=-2.0, exit_z=0.0, holding_days=holding_days,
        pnl_pct=pnl_pct, exit_reason="mean_reversion",
    )


def test_sharpe_zero_for_no_variance():
    returns = pd.Series([0.0] * 10)
    assert sharpe_ratio(returns) == 0.0


def test_sharpe_positive_for_positive_drift():
    rng = np.random.default_rng(0)
    returns = pd.Series(rng.normal(0.001, 0.01, 500))
    assert sharpe_ratio(returns) > 0


def test_max_drawdown_is_nonpositive():
    returns = pd.Series([0.05, -0.10, 0.02, -0.03, 0.04])
    dd = max_drawdown(returns)
    assert dd <= 0


def test_max_drawdown_known_case():
    # equity: 1 -> 1.1 -> 0.99 (a 10% drawdown from the peak of 1.1)
    returns = pd.Series([0.10, -0.10])
    dd = max_drawdown(returns)
    assert np.isclose(dd, -0.10, atol=1e-6)


def test_win_rate():
    trades = [_trade(0.02, 3), _trade(-0.01, 2), _trade(0.03, 5), _trade(-0.02, 1)]
    assert win_rate(trades) == 0.5


def test_win_rate_empty():
    assert win_rate([]) == 0.0


def test_average_holding_period():
    trades = [_trade(0.01, 2), _trade(0.01, 4), _trade(0.01, 6)]
    assert average_holding_period(trades) == 4.0


def test_cagr_positive_growth():
    returns = pd.Series([0.001] * 252)
    value = cagr(returns)
    assert value > 0


def test_summarize_keys_present():
    returns = pd.Series([0.001, -0.002, 0.003])
    trades = [_trade(0.01, 3)]
    summary = summarize(returns, trades)
    expected_keys = {
        "sharpe_ratio", "cagr", "max_drawdown", "win_rate",
        "avg_holding_period_days", "num_trades", "total_return",
    }
    assert expected_keys == set(summary.keys())
