#!/usr/bin/env python3
"""End-to-end pipeline: fetch data -> screen pairs -> backtest -> report.

Usage:
    python scripts/run_backtest.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import pandas as pd

import config
from pairs_trading.backtest import aggregate_portfolio, run_pair_backtest
from pairs_trading.data import get_price_data
from pairs_trading.metrics import summarize
from pairs_trading.pair_selection import find_cointegrated_pairs
from pairs_trading.plotting import plot_equity_curve, plot_pair_spread
from pairs_trading.signals import rolling_zscore
from pairs_trading.kalman import kalman_hedge_ratio


def main():
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    all_tickers = sorted({t for tickers in config.SECTOR_UNIVERSE.values() for t in tickers})
    print(f"Fetching {len(all_tickers)} tickers from {config.FORMATION_START} to {config.BACKTEST_END}...")
    prices = get_price_data(all_tickers, config.FORMATION_START, config.BACKTEST_END, data_dir=config.DATA_DIR)
    print(f"Fetched price data: {prices.shape[0]} days x {prices.shape[1]} tickers")

    formation_prices = prices.loc[config.FORMATION_START:config.FORMATION_END]
    print("Screening for cointegrated pairs on the formation window...")
    candidates = find_cointegrated_pairs(
        formation_prices,
        config.SECTOR_UNIVERSE,
        pvalue_threshold=config.COINT_PVALUE_THRESHOLD,
        min_half_life=config.MIN_HALF_LIFE_DAYS,
        max_half_life=config.MAX_HALF_LIFE_DAYS,
        max_pairs=config.MAX_PAIRS,
        min_hedge_ratio=config.MIN_HEDGE_RATIO,
        max_hedge_ratio=config.MAX_HEDGE_RATIO,
    )
    print(f"Selected {len(candidates)} pairs:")
    for c in candidates:
        print(f"  {c.y}/{c.x}  p={c.pvalue:.4g}  static_beta={c.hedge_ratio:.3f}  half_life={c.half_life:.1f}d")

    if not candidates:
        print("No cointegrated pairs found -- aborting backtest.")
        return

    results = []
    for c in candidates:
        pair_name = f"{c.y}/{c.x}"
        y = prices[c.y].dropna()
        x = prices[c.x].dropna()
        result = run_pair_backtest(
            y, x, pair_name=pair_name,
            backtest_start=config.BACKTEST_START,
            zscore_lookback=config.ZSCORE_LOOKBACK,
            entry_z=config.ENTRY_ZSCORE, exit_z=config.EXIT_ZSCORE, stop_z=config.STOP_LOSS_ZSCORE,
            kalman_delta=config.KALMAN_DELTA, kalman_obs_cov=c.residual_variance,
            commission_bps=config.COMMISSION_BPS, slippage_bps=config.SLIPPAGE_BPS,
            kelly_fraction_cap=config.KELLY_FRACTION_CAP,
            kelly_default_fraction=config.KELLY_DEFAULT_FRACTION,
            kelly_min_trades=config.KELLY_LOOKBACK_TRADES,
        )
        results.append(result)
        print(f"  {pair_name}: {len(result.trades)} trades")

    portfolio_returns = aggregate_portfolio(results, config.MAX_CAPITAL_PER_PAIR)
    all_trades = [t for r in results for t in r.trades]
    summary = summarize(portfolio_returns, all_trades)

    print("\n=== Portfolio Summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # Per-pair summary for the README table.
    per_pair_summary = []
    for r in results:
        per_pair_summary.append({
            "pair": r.pair_name,
            **summarize(r.daily_return, r.trades),
        })

    with open(os.path.join(config.RESULTS_DIR, "metrics.json"), "w") as f:
        json.dump({
            "portfolio": summary,
            "per_pair": per_pair_summary,
            "selected_pairs": [
                {"pair": f"{c.y}/{c.x}", "pvalue": c.pvalue, "static_beta": c.hedge_ratio, "half_life": c.half_life}
                for c in candidates
            ],
            "backtest_start": config.BACKTEST_START,
            "backtest_end": config.BACKTEST_END,
        }, f, indent=2, default=float)

    trade_rows = [{
        "pair": t.pair, "entry_date": t.entry_date, "exit_date": t.exit_date,
        "direction": t.direction, "entry_z": t.entry_z, "exit_z": t.exit_z,
        "holding_days": t.holding_days, "pnl_pct": t.pnl_pct, "exit_reason": t.exit_reason,
    } for t in all_trades]
    pd.DataFrame(trade_rows).to_csv(os.path.join(config.RESULTS_DIR, "trade_log.csv"), index=False)

    plot_equity_curve(portfolio_returns, os.path.join(config.RESULTS_DIR, "equity_curve.png"))

    if candidates:
        top = candidates[0]
        y, x = prices[top.y].dropna(), prices[top.x].dropna()
        common = y.index.intersection(x.index)
        kf = kalman_hedge_ratio(y.loc[common], x.loc[common], delta=config.KALMAN_DELTA, obs_cov=top.residual_variance)
        z = rolling_zscore(kf["spread"], config.ZSCORE_LOOKBACK).loc[config.BACKTEST_START:]
        plot_pair_spread(
            z, config.ENTRY_ZSCORE, config.EXIT_ZSCORE, config.STOP_LOSS_ZSCORE,
            f"{top.y}/{top.x}", os.path.join(config.RESULTS_DIR, "example_pair_spread.png"),
        )

    print(f"\nWrote results to {config.RESULTS_DIR}/")


if __name__ == "__main__":
    main()
