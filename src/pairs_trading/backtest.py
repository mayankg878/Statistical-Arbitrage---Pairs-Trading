"""Event-driven backtest for a single pair, plus portfolio aggregation.

Each pair is traded dollar-neutral: long 1 unit of y / short beta units of x
(or the reverse). Daily P&L per unit of capital deployed is computed as the
day's raw spread change divided by the prior day's gross notional -- the
standard normalization (Ernie Chan, "Algorithmic Trading") that turns a
price-level spread into a comparable percentage return series.

Kelly sizing is re-estimated after every closed trade using that pair's own
trailing trade-return history (see position_sizing.py), so early trades use
`KELLY_DEFAULT_FRACTION` and sizing adapts as a track record accumulates.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .kalman import kalman_hedge_ratio
from .position_sizing import kelly_fraction
from .signals import generate_positions, rolling_zscore


@dataclass
class Trade:
    pair: str
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    direction: int  # +1 or -1, in spread units
    entry_z: float
    exit_z: float
    holding_days: int
    pnl_pct: float  # net-of-cost return on capital deployed for this trade
    exit_reason: str


@dataclass
class PairBacktestResult:
    pair_name: str
    trades: list[Trade] = field(default_factory=list)
    daily_return: pd.Series = None  # pct return of capital *allocated to this pair*, indexed by date


def _gross_notional(y_t: float, x_t: float, beta_t: float) -> float:
    return abs(y_t) + abs(beta_t * x_t)


def run_pair_backtest(
    y: pd.Series,
    x: pd.Series,
    pair_name: str,
    backtest_start: str,
    zscore_lookback: int,
    entry_z: float,
    exit_z: float,
    stop_z: float,
    kalman_delta: float,
    kalman_obs_cov: float,
    commission_bps: float,
    slippage_bps: float,
    kelly_fraction_cap: float,
    kelly_default_fraction: float,
    kelly_min_trades: int,
) -> PairBacktestResult:
    common_idx = y.index.intersection(x.index)
    y, x = y.loc[common_idx], x.loc[common_idx]

    kf = kalman_hedge_ratio(y, x, delta=kalman_delta, obs_cov=kalman_obs_cov)
    zscore = rolling_zscore(kf["spread"], zscore_lookback)

    bt_zscore = zscore.loc[backtest_start:]
    positions = generate_positions(bt_zscore, entry_z=entry_z, exit_z=exit_z, stop_z=stop_z)

    dates = positions.index
    cost_rate = (commission_bps + slippage_bps) / 10_000.0

    trades: list[Trade] = []
    daily_returns = pd.Series(0.0, index=dates)
    deployed_fraction = pd.Series(kelly_default_fraction, index=dates)

    trailing_trade_returns: list[float] = []
    current_kelly = kelly_default_fraction
    trade_kelly = kelly_default_fraction  # fraction actually deployed for the in-flight trade, frozen at entry

    state = 0
    entry_date = None
    entry_z_val = None
    trade_return_accum = 0.0

    for i in range(1, len(dates)):
        t_prev, t = dates[i - 1], dates[i]
        pos_prev = positions.loc[t_prev]

        beta_prev = kf["beta"].loc[t_prev]
        y_prev, y_now = y.loc[t_prev], y.loc[t]
        x_prev, x_now = x.loc[t_prev], x.loc[t]

        raw_pnl = (y_now - y_prev) - beta_prev * (x_now - x_prev)
        notional = _gross_notional(y_prev, x_prev, beta_prev)
        day_return = pos_prev * raw_pnl / notional if notional > 0 else 0.0

        pos_now = positions.loc[t]
        entering = state == 0 and pos_now != 0
        exiting = state != 0 and pos_now == 0
        # Was capital deployed to this pair *during* day t? Note this must be
        # decided from pos_prev/entering, not from `state` after the
        # entering/exiting blocks below mutate it -- otherwise the exit day's
        # return (computed from pos_prev, which was still non-zero) gets
        # multiplied by a deployed_fraction of 0 and silently vanishes from
        # the return series, even though the trade log still recorded it.
        had_exposure = pos_prev != 0 or entering

        if entering:
            state = pos_now
            entry_date = t
            entry_z_val = bt_zscore.loc[t]
            trade_return_accum = 0.0
            trade_kelly = current_kelly
            day_return -= cost_rate  # entry cost charged on entry day

        if state != 0:
            trade_return_accum += day_return

        if exiting:
            day_return -= cost_rate  # exit cost
            trade_return_accum -= cost_rate
            holding_days = max((t - entry_date).days, 1)
            exit_reason = "stop_loss" if abs(bt_zscore.loc[t_prev]) >= stop_z else "mean_reversion"
            trades.append(Trade(
                pair=pair_name,
                entry_date=entry_date,
                exit_date=t,
                direction=state,
                entry_z=float(entry_z_val),
                exit_z=float(bt_zscore.loc[t]) if not pd.isna(bt_zscore.loc[t]) else float("nan"),
                holding_days=holding_days,
                pnl_pct=float(trade_return_accum),
                exit_reason=exit_reason,
            ))
            trailing_trade_returns.append(trade_return_accum)
            current_kelly = kelly_fraction(
                trailing_trade_returns,
                fraction_cap=kelly_fraction_cap,
                default_fraction=kelly_default_fraction,
                min_trades=kelly_min_trades,
            )
            state = 0

        deployed_fraction.loc[t] = trade_kelly if had_exposure else 0.0
        daily_returns.loc[t] = day_return * deployed_fraction.loc[t]

    # Force-close an open position at the end of the data window.
    if state != 0 and entry_date is not None:
        holding_days = max((dates[-1] - entry_date).days, 1)
        trades.append(Trade(
            pair=pair_name,
            entry_date=entry_date,
            exit_date=dates[-1],
            direction=state,
            entry_z=float(entry_z_val),
            exit_z=float(bt_zscore.iloc[-1]) if not pd.isna(bt_zscore.iloc[-1]) else float("nan"),
            holding_days=holding_days,
            pnl_pct=float(trade_return_accum),
            exit_reason="end_of_data",
        ))

    return PairBacktestResult(pair_name=pair_name, trades=trades, daily_return=daily_returns)


def aggregate_portfolio(results: list[PairBacktestResult], max_capital_per_pair: float) -> pd.Series:
    """Equal-weight the per-pair daily returns (each pair capped at
    `max_capital_per_pair` of total capital) into one portfolio return series.
    """
    if not results:
        return pd.Series(dtype=float)

    all_dates = sorted(set().union(*(r.daily_return.index for r in results)))
    portfolio = pd.Series(0.0, index=all_dates)

    weight = min(max_capital_per_pair, 1.0 / len(results))
    for r in results:
        portfolio = portfolio.add(r.daily_return.reindex(all_dates, fill_value=0.0) * weight, fill_value=0.0)

    return portfolio
