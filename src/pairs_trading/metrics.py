"""Performance metrics computed from a portfolio return series and trade log."""

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def sharpe_ratio(daily_returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    excess = daily_returns - risk_free_rate / TRADING_DAYS_PER_YEAR
    if excess.std() == 0 or excess.empty:
        return 0.0
    return float(np.sqrt(TRADING_DAYS_PER_YEAR) * excess.mean() / excess.std())


def max_drawdown(daily_returns: pd.Series) -> float:
    equity = (1 + daily_returns).cumprod()
    running_max = equity.cummax()
    drawdown = equity / running_max - 1
    return float(drawdown.min())


def cagr(daily_returns: pd.Series) -> float:
    equity = (1 + daily_returns).cumprod()
    if len(equity) == 0:
        return 0.0
    total_return = equity.iloc[-1]
    years = len(equity) / TRADING_DAYS_PER_YEAR
    if years <= 0 or total_return <= 0:
        return 0.0
    return float(total_return ** (1 / years) - 1)


def win_rate(trades: list) -> float:
    if not trades:
        return 0.0
    wins = sum(1 for t in trades if t.pnl_pct > 0)
    return wins / len(trades)


def average_holding_period(trades: list) -> float:
    if not trades:
        return 0.0
    return float(np.mean([t.holding_days for t in trades]))


def summarize(daily_returns: pd.Series, trades: list) -> dict:
    return {
        "sharpe_ratio": sharpe_ratio(daily_returns),
        "cagr": cagr(daily_returns),
        "max_drawdown": max_drawdown(daily_returns),
        "win_rate": win_rate(trades),
        "avg_holding_period_days": average_holding_period(trades),
        "num_trades": len(trades),
        "total_return": float((1 + daily_returns).prod() - 1) if len(daily_returns) else 0.0,
    }
