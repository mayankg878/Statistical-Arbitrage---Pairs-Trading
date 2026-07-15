#!/usr/bin/env python3
"""Walk-forward validation: grid-search entry_z/exit_z/lookback on TRAIN
folds only, then run the single winning combo once on held-out TEST folds.

TEST folds are never touched during the grid search -- see
walk_forward.split_folds_train_test and for_me_readme.md for why this
matters. Usage:

    python scripts/run_walk_forward.py
"""

import itertools
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import pandas as pd

import config
from pairs_trading.data import get_price_data
from pairs_trading.metrics import summarize
from pairs_trading.plotting import plot_equity_curve
from pairs_trading.walk_forward import generate_folds, run_walk_forward, split_folds_train_test


def score_combo(prices, sector_universe, pair_sectors, folds, entry_z, exit_z, lookback):
    result = run_walk_forward(
        prices, sector_universe, pair_sectors, folds,
        entry_z=entry_z, exit_z=exit_z, stop_z=config.STOP_LOSS_ZSCORE, zscore_lookback=lookback,
        kalman_delta=config.KALMAN_DELTA,
        commission_bps=config.COMMISSION_BPS, slippage_bps=config.SLIPPAGE_BPS,
        kelly_fraction_cap=config.KELLY_FRACTION_CAP,
        kelly_default_fraction=config.KELLY_DEFAULT_FRACTION,
        kelly_min_trades=config.KELLY_LOOKBACK_TRADES,
        pvalue_threshold=config.COINT_PVALUE_THRESHOLD,
        min_half_life=config.MIN_HALF_LIFE_DAYS, max_half_life=config.MAX_HALF_LIFE_DAYS,
        max_pairs=config.MAX_PAIRS,
        min_hedge_ratio=config.MIN_HEDGE_RATIO, max_hedge_ratio=config.MAX_HEDGE_RATIO,
        use_fdr_correction=config.USE_FDR_CORRECTION, fdr_alpha=config.FDR_ALPHA,
        max_capital_per_pair=config.MAX_CAPITAL_PER_PAIR, max_capital_per_sector=config.MAX_CAPITAL_PER_SECTOR,
    )
    fold_sharpes = [r["sharpe"] for r in result.per_fold_records if r["sharpe"] == r["sharpe"]]  # drop NaN
    median_sharpe = statistics.median(fold_sharpes) if fold_sharpes else float("-inf")
    return result, median_sharpe


def main():
    os.makedirs(config.WALK_FORWARD_RESULTS_DIR, exist_ok=True)

    all_tickers = sorted({t for tickers in config.SECTOR_UNIVERSE.values() for t in tickers})
    print(f"Fetching {len(all_tickers)} tickers from {config.WF_DATA_START} to {config.BACKTEST_END}...")
    prices = get_price_data(all_tickers, config.WF_DATA_START, config.BACKTEST_END, data_dir=config.DATA_DIR)
    print(f"Fetched: {prices.shape[0]} days x {prices.shape[1]} tickers")

    pair_sectors = config.pair_sector_map(config.SECTOR_UNIVERSE)
    folds = generate_folds(config.WF_DATA_START, config.BACKTEST_END, config.FORMATION_YEARS, config.TRADING_MONTHS)
    train_folds, test_folds = split_folds_train_test(folds, config.TRAIN_TEST_SPLIT_DATE)
    print(f"{len(folds)} total folds: {len(train_folds)} TRAIN (< {config.TRAIN_TEST_SPLIT_DATE}), "
          f"{len(test_folds)} TEST (>= {config.TRAIN_TEST_SPLIT_DATE})")

    print("\n=== TRAIN grid search (entry_z, exit_z, lookback) ===")
    grid = list(itertools.product(config.ENTRY_ZSCORE_GRID, config.EXIT_ZSCORE_GRID, config.ZSCORE_LOOKBACK_GRID))
    grid_results = []
    best = None
    for entry_z, exit_z, lookback in grid:
        _, median_sharpe = score_combo(prices, config.SECTOR_UNIVERSE, pair_sectors, train_folds, entry_z, exit_z, lookback)
        grid_results.append({"entry_z": entry_z, "exit_z": exit_z, "lookback": lookback, "median_fold_sharpe": median_sharpe})
        print(f"  entry_z={entry_z} exit_z={exit_z} lookback={lookback}: median fold Sharpe = {median_sharpe:.3f}")
        if best is None or median_sharpe > best["median_fold_sharpe"]:
            best = grid_results[-1]

    print(f"\nBest TRAIN combo: {best}")

    with open(os.path.join(config.WALK_FORWARD_RESULTS_DIR, "train_grid_search.json"), "w") as f:
        json.dump({"grid_results": grid_results, "best": best}, f, indent=2, default=float)

    print("\n=== Running best combo on held-out TEST folds ===")
    test_result, test_median_fold_sharpe = score_combo(
        prices, config.SECTOR_UNIVERSE, pair_sectors, test_folds,
        best["entry_z"], best["exit_z"], best["lookback"],
    )
    summary = summarize(test_result.daily_return, test_result.trades)
    summary["median_fold_sharpe"] = test_median_fold_sharpe

    print("\n=== TEST Summary (out-of-sample, never seen by the grid search) ===")
    for k, v in summary.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    fold_df = pd.DataFrame(test_result.per_fold_records)
    fold_df.to_csv(os.path.join(config.WALK_FORWARD_RESULTS_DIR, "test_fold_diagnostics.csv"), index=False)

    trade_rows = [{
        "pair": t.pair, "entry_date": t.entry_date, "exit_date": t.exit_date,
        "direction": t.direction, "entry_z": t.entry_z, "exit_z": t.exit_z,
        "holding_days": t.holding_days, "pnl_pct": t.pnl_pct, "exit_reason": t.exit_reason,
    } for t in test_result.trades]
    pd.DataFrame(trade_rows).to_csv(os.path.join(config.WALK_FORWARD_RESULTS_DIR, "test_trade_log.csv"), index=False)

    with open(os.path.join(config.WALK_FORWARD_RESULTS_DIR, "test_metrics.json"), "w") as f:
        json.dump({
            "best_train_combo": best,
            "test_summary": summary,
            "n_train_folds": len(train_folds),
            "n_test_folds": len(test_folds),
            "train_test_split_date": config.TRAIN_TEST_SPLIT_DATE,
        }, f, indent=2, default=float)

    plot_equity_curve(test_result.daily_return, os.path.join(config.WALK_FORWARD_RESULTS_DIR, "test_equity_curve.png"))

    print(f"\nWrote results to {config.WALK_FORWARD_RESULTS_DIR}/")


if __name__ == "__main__":
    main()
