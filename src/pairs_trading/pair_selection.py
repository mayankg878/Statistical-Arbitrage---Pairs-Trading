"""Engle-Granger cointegration screening for candidate pairs."""

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from statsmodels.tsa.stattools import coint


@dataclass
class PairCandidate:
    y: str
    x: str
    pvalue: float
    hedge_ratio: float  # static OLS beta over the formation window, for reference
    half_life: float
    residual_variance: float  # formation-window spread variance; used to scale the Kalman filter's observation noise


def half_life_of_mean_reversion(spread: pd.Series) -> float:
    """Half-life (in days) of an AR(1) mean-reverting spread, via
    spread_t - spread_{t-1} = kappa * spread_{t-1} + noise.

    Returns +inf if the spread does not appear mean-reverting (kappa >= 0).
    """
    spread = spread.dropna()
    lagged = spread.shift(1).dropna()
    delta = spread.diff().dropna()
    lagged = lagged.loc[delta.index]

    model = OLS(delta.values, add_constant(lagged.values)).fit()
    kappa = model.params[1]
    if kappa >= 0:
        return np.inf
    return -np.log(2) / kappa


def _static_hedge_ratio(y: pd.Series, x: pd.Series) -> float:
    model = OLS(y.values, add_constant(x.values)).fit()
    return float(model.params[1])


def benjamini_hochberg_threshold(pvalues: list[float], fdr_alpha: float = 0.10) -> float:
    """Largest p-value cutoff such that the false discovery rate is
    controlled at `fdr_alpha`, given `m` independent-ish hypothesis tests.

    Standard BH step-up procedure: sort p-values ascending, find the
    largest k with p_(k) <= (k/m)*fdr_alpha; every p-value at or below
    p_(k) is declared significant. Returns 0.0 (nothing survives) if no
    such k exists.

    This matters here because screening ~200 within-sector pairs at a flat
    p<0.05 implies ~10 expected false positives even if no pair in the
    universe were genuinely cointegrated -- BH control keeps the expected
    fraction of false discoveries among *declared* pairs bounded instead.
    """
    m = len(pvalues)
    if m == 0:
        return 0.0
    sorted_p = np.sort(np.asarray(pvalues, dtype=float))
    ranks = np.arange(1, m + 1)
    eligible = sorted_p <= (ranks / m) * fdr_alpha
    if not eligible.any():
        return 0.0
    k = np.max(np.where(eligible)[0]) + 1
    return float(sorted_p[k - 1])


def find_cointegrated_pairs(
    prices: pd.DataFrame,
    sector_universe: dict[str, list[str]],
    pvalue_threshold: float = 0.05,
    min_half_life: float = 1,
    max_half_life: float = 30,
    max_pairs: int = 12,
    min_hedge_ratio: float = 0.3,
    max_hedge_ratio: float = 3.0,
    use_fdr_correction: bool = False,
    fdr_alpha: float = 0.10,
) -> list[PairCandidate]:
    """Screen within-sector pairs for cointegration on the formation window.

    `min_hedge_ratio`/`max_hedge_ratio` reject degenerate fits where the OLS
    beta is near zero (or huge): a beta near zero means the Engle-Granger
    residual is essentially just `y` itself, so the ADF test is picking up
    that `y` alone happens to be range-bound over the formation window, not a
    genuine linear relationship between the two names. That's a spurious
    "cointegrated pair" in practice, not a tradeable one.

    `use_fdr_correction`: if True, `pvalue_threshold` is ignored in favor of
    a Benjamini-Hochberg cutoff computed from the p-values of *every*
    within-sector pair tested this call (see `benjamini_hochberg_threshold`).
    Off by default to keep the original flat-threshold behavior for
    existing callers; the walk-forward pipeline turns it on.

    Returns candidates sorted by p-value (ascending), truncated to max_pairs.
    """
    raw: list[tuple[str, str, float, pd.Series, pd.Series]] = []

    for tickers in sector_universe.values():
        available = [t for t in tickers if t in prices.columns]
        for y_ticker, x_ticker in combinations(available, 2):
            y = prices[y_ticker].dropna()
            x = prices[x_ticker].dropna()
            common_idx = y.index.intersection(x.index)
            if len(common_idx) < 100:
                continue
            y, x = y.loc[common_idx], x.loc[common_idx]

            _, pvalue, _ = coint(y, x)
            raw.append((y_ticker, x_ticker, pvalue, y, x))

    if use_fdr_correction:
        effective_threshold = benjamini_hochberg_threshold([r[2] for r in raw], fdr_alpha)
    else:
        effective_threshold = pvalue_threshold

    candidates: list[PairCandidate] = []
    for y_ticker, x_ticker, pvalue, y, x in raw:
        if pvalue > effective_threshold:
            continue

        beta = _static_hedge_ratio(y, x)
        if not (min_hedge_ratio <= abs(beta) <= max_hedge_ratio):
            continue

        spread = y - beta * x
        hl = half_life_of_mean_reversion(spread)
        if not (min_half_life <= hl <= max_half_life):
            continue

        candidates.append(PairCandidate(
            y=y_ticker, x=x_ticker, pvalue=pvalue, hedge_ratio=beta, half_life=hl,
            residual_variance=float(spread.var()),
        ))

    candidates.sort(key=lambda c: c.pvalue)
    return candidates[:max_pairs]
