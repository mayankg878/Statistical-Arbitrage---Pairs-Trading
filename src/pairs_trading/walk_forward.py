"""Rolling walk-forward pair re-selection + backtest.

Instead of picking pairs once on a single formation window and trading them
for years afterward (which goes stale -- see for_me_readme.md), this rolls
forward: re-run the cointegration screen every `trading_months`, trade the
freshly-selected pairs for that window only, then roll to the next window.
Concatenating many such windows across a long price history also gives a
much more varied out-of-sample sample (different market regimes) than one
single split.
"""

from dataclasses import dataclass, field

import pandas as pd

from .backtest import PairBacktestResult, Trade, aggregate_portfolio_capped, run_pair_backtest
from .metrics import sharpe_ratio
from .pair_selection import find_cointegrated_pairs


@dataclass
class Fold:
    formation_start: pd.Timestamp
    formation_end: pd.Timestamp
    trading_start: pd.Timestamp
    trading_end: pd.Timestamp


def generate_folds(data_start: str, data_end: str, formation_years: float, trading_months: int) -> list[Fold]:
    """Non-overlapping, chronological folds covering [data_start, data_end].

    Each fold's formation window is the `formation_years` immediately before
    its trading window, so pair selection for a fold never sees that fold's
    own trading-window data (or any later fold's data).
    """
    data_start = pd.Timestamp(data_start)
    data_end = pd.Timestamp(data_end)
    formation_days = int(formation_years * 365.25)

    folds: list[Fold] = []
    trading_start = data_start + pd.Timedelta(days=formation_days)

    while trading_start < data_end:
        trading_end = min(trading_start + pd.DateOffset(months=trading_months), data_end)
        formation_start = trading_start - pd.Timedelta(days=formation_days)
        formation_end = trading_start - pd.Timedelta(days=1)
        folds.append(Fold(formation_start, formation_end, trading_start, trading_end))
        if trading_end >= data_end:
            break
        trading_start = trading_end + pd.Timedelta(days=1)

    return folds


def split_folds_train_test(folds: list[Fold], split_date: str) -> tuple[list[Fold], list[Fold]]:
    """Partition folds by `trading_start` relative to `split_date`.

    Since each fold's formation+trading data is entirely within
    [formation_start, trading_end], and folds are chronological and
    non-overlapping, every TRAIN fold's data lies strictly before every TEST
    fold's data -- the grid search over TRAIN folds never touches a date
    used by a TEST fold.
    """
    split_date = pd.Timestamp(split_date)
    train = [f for f in folds if f.trading_start < split_date]
    test = [f for f in folds if f.trading_start >= split_date]
    return train, test


@dataclass
class WalkForwardResult:
    folds: list[Fold] = field(default_factory=list)
    daily_return: pd.Series = None
    trades: list[Trade] = field(default_factory=list)
    per_fold_records: list[dict] = field(default_factory=list)


def run_walk_forward(
    prices: pd.DataFrame,
    sector_universe: dict[str, list[str]],
    pair_sectors: dict[str, str],
    folds: list[Fold],
    entry_z: float,
    exit_z: float,
    stop_z: float,
    zscore_lookback: int,
    kalman_delta: float,
    commission_bps: float,
    slippage_bps: float,
    kelly_fraction_cap: float,
    kelly_default_fraction: float,
    kelly_min_trades: int,
    pvalue_threshold: float,
    min_half_life: float,
    max_half_life: float,
    max_pairs: int,
    min_hedge_ratio: float,
    max_hedge_ratio: float,
    use_fdr_correction: bool,
    fdr_alpha: float,
    max_capital_per_pair: float,
    max_capital_per_sector: float,
) -> WalkForwardResult:
    all_daily_returns: list[pd.Series] = []
    all_trades: list[Trade] = []
    per_fold_records: list[dict] = []

    for fold in folds:
        formation_prices = prices.loc[fold.formation_start:fold.formation_end]
        candidates = find_cointegrated_pairs(
            formation_prices, sector_universe,
            pvalue_threshold=pvalue_threshold,
            min_half_life=min_half_life, max_half_life=max_half_life, max_pairs=max_pairs,
            min_hedge_ratio=min_hedge_ratio, max_hedge_ratio=max_hedge_ratio,
            use_fdr_correction=use_fdr_correction, fdr_alpha=fdr_alpha,
        )

        fold_results: list[PairBacktestResult] = []
        for c in candidates:
            pair_name = f"{c.y}/{c.x}"
            y = prices[c.y].dropna().loc[:fold.trading_end]
            x = prices[c.x].dropna().loc[:fold.trading_end]
            if len(y.index.intersection(x.index)) < 50:
                continue
            result = run_pair_backtest(
                y, x, pair_name=pair_name,
                backtest_start=fold.trading_start.isoformat(),
                zscore_lookback=zscore_lookback,
                entry_z=entry_z, exit_z=exit_z, stop_z=stop_z,
                kalman_delta=kalman_delta, kalman_obs_cov=c.residual_variance,
                commission_bps=commission_bps, slippage_bps=slippage_bps,
                kelly_fraction_cap=kelly_fraction_cap,
                kelly_default_fraction=kelly_default_fraction,
                kelly_min_trades=kelly_min_trades,
            )
            fold_results.append(result)

        if fold_results:
            portfolio = aggregate_portfolio_capped(
                fold_results, pair_sectors, max_capital_per_pair, max_capital_per_sector,
            )
        else:
            portfolio = pd.Series(dtype=float)

        all_daily_returns.append(portfolio)
        fold_trades = [t for r in fold_results for t in r.trades]
        all_trades.extend(fold_trades)

        per_fold_records.append({
            "formation_start": fold.formation_start, "formation_end": fold.formation_end,
            "trading_start": fold.trading_start, "trading_end": fold.trading_end,
            "n_pairs": len(candidates),
            "pairs": ",".join(f"{c.y}/{c.x}" for c in candidates),
            "n_trades": len(fold_trades),
            "sharpe": sharpe_ratio(portfolio) if not portfolio.empty else float("nan"),
        })

    non_empty = [r for r in all_daily_returns if not r.empty]
    combined = pd.concat(non_empty).sort_index() if non_empty else pd.Series(dtype=float)

    return WalkForwardResult(folds=folds, daily_return=combined, trades=all_trades, per_fold_records=per_fold_records)
