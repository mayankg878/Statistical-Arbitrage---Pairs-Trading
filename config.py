"""Single source of truth for universe, dates, and strategy parameters."""

from datetime import date

# --- Universe: tickers grouped by sector. Only within-sector pairs are
# screened for cointegration -- cross-sector relationships are usually
# spurious rather than economically grounded.
def ticker_sector_map(sector_universe: dict[str, list[str]]) -> dict[str, str]:
    """Invert SECTOR_UNIVERSE into ticker -> sector."""
    return {t: sector for sector, tickers in sector_universe.items() for t in tickers}


def pair_sector_map(sector_universe: dict[str, list[str]]) -> dict[str, str]:
    """`{"AAL/ALK": "airlines", ...}` for every within-sector pair, used to
    cap total capital per sector in portfolio construction (see
    backtest.aggregate_portfolio_capped)."""
    from itertools import combinations

    t2s = ticker_sector_map(sector_universe)
    mapping = {}
    for sector, tickers in sector_universe.items():
        for a, b in combinations(tickers, 2):
            mapping[f"{a}/{b}"] = sector
            mapping[f"{b}/{a}"] = sector
    return mapping


SECTOR_UNIVERSE = {
    "energy": ["XOM", "CVX", "COP", "SLB", "PSX", "VLO", "MPC", "OXY"],
    "staples": ["KO", "PEP", "PG", "CL", "KMB", "MDLZ", "GIS", "HSY"],
    "banks": ["JPM", "BAC", "WFC", "C", "USB", "PNC", "TFC", "MTB"],
    "payments": ["V", "MA", "PYPL", "FIS"],
    "retail": ["HD", "LOW", "TGT", "WMT", "COST"],
    "airlines": ["DAL", "UAL", "AAL", "LUV", "ALK"],
    "utilities": ["DUK", "SO", "NEE", "AEP", "D", "EXC", "XEL"],
    "healthcare": ["JNJ", "PFE", "MRK", "ABBV", "BMY", "LLY"],
    "telecom": ["VZ", "T", "TMUS"],
    "reits": ["O", "SPG", "PLD", "EQIX"],
    "insurance": ["TRV", "ALL", "PGR", "CB"],
    "semiconductors": ["TXN", "ADI", "MCHP", "NXPI"],
    "industrials": ["HON", "MMM", "GE", "EMR", "ITW", "ETN", "PH", "DOV"],
    "asset_managers": ["BLK", "BX", "ICE", "CME", "NDAQ"],
    "apparel": ["NKE", "VFC", "PVH", "RL"],
    "restaurants": ["MCD", "YUM", "SBUX", "CMG"],
    "homebuilders": ["DHI", "LEN", "PHM", "NVR"],
    "media": ["CMCSA", "CHTR", "WBD"],
    "chemicals": ["DD", "DOW", "LYB", "APD", "ECL"],
    "autos": ["GM", "F", "STLA"],
    "beverages": ["STZ", "TAP", "BF-B"],
    "hotels_leisure": ["MAR", "HLT", "H"],
    "specialty_retail": ["ROST", "TJX", "GAP"],
    "packaging": ["IP", "PKG", "SW"],
    "railroads": ["UNP", "CSX", "NSC"],
    "aerospace_defense": ["LMT", "NOC", "GD", "RTX"],
}

# --- Date ranges. Data is pulled from FORMATION_START through BACKTEST_END.
# Pairs are selected using only the formation window; the backtest window is
# strictly out-of-sample with respect to pair selection, to avoid look-ahead
# / data-snooping bias.
FORMATION_START = "2020-07-15"
FORMATION_END = "2023-07-15"
BACKTEST_START = "2023-07-15"
BACKTEST_END = date.today().isoformat()

# --- Pair selection
COINT_PVALUE_THRESHOLD = 0.05
MIN_HALF_LIFE_DAYS = 1
MAX_HALF_LIFE_DAYS = 30
MAX_PAIRS = 12
MIN_HEDGE_RATIO = 0.3   # rejects near-zero beta fits (see pair_selection.find_cointegrated_pairs docstring)
MAX_HEDGE_RATIO = 3.0

# --- Kalman filter (random-walk state space for dynamic hedge ratio)
# Observation noise (obs_cov) is NOT a fixed constant here: it's estimated
# per pair from the formation-window OLS residual variance (see
# PairCandidate.residual_variance), since spread noise scale varies hugely
# across price levels ($1 stock vs. $300 stock). A single global obs_cov
# was tried first and made the filter wildly overreact to price noise on
# expensive tickers -- see for_me_readme.md.
KALMAN_DELTA = 1e-4          # process noise scale (smaller = smoother beta)

# --- Signal generation
ZSCORE_LOOKBACK = 30
ENTRY_ZSCORE = 2.0
EXIT_ZSCORE = 0.5
STOP_LOSS_ZSCORE = 3.0

# --- Position sizing (Kelly criterion)
KELLY_LOOKBACK_TRADES = 10   # min trades before Kelly is estimated from history
KELLY_FRACTION_CAP = 0.5     # half-Kelly
KELLY_DEFAULT_FRACTION = 0.1 # used before enough trade history exists
MAX_CAPITAL_PER_PAIR = 0.15  # cap on fraction of total capital per pair

# --- Backtest frictions
INITIAL_CAPITAL = 1_000_000.0
COMMISSION_BPS = 5.0
SLIPPAGE_BPS = 5.0

# --- Paths
DATA_DIR = "data"
RESULTS_DIR = "results"
WALK_FORWARD_RESULTS_DIR = "results/walk_forward"

# --- Walk-forward validation (see for_me_readme.md "Future work" and the
# walk_forward module). Pairs are re-selected on a rolling basis instead of
# once, across a much longer history, so the reported result reflects many
# independent market regimes rather than one static 2023-2026 window.
WF_DATA_START = "2011-01-01"
FORMATION_YEARS = 2      # fixed, matches the original single-split default -- not tuned
TRADING_MONTHS = 6        # fixed -- not tuned
TRAIN_TEST_SPLIT_DATE = "2020-01-01"  # grid search only ever sees folds before this date

# Structural, non-fitted corrections applied identically in every fold:
USE_FDR_CORRECTION = True
# Measured empirically (see for_me_readme.md): with ~200 within-sector pair
# tests per 6-month fold and the effect sizes actually present in this
# universe, a conventional fdr_alpha=0.10 left 24/27 walk-forward folds with
# *zero* surviving pairs -- too strict to produce a meaningful sample, and a
# degenerate way to "improve" the result (an empty fold trades nothing, it
# doesn't lose). fdr_alpha=0.30 is a deliberately looser but still real
# correction: it rejects ~75% of the pairs a flat p<0.05 screen would have
# kept (88 vs 359 across the same folds), while leaving most folds with at
# least one tradeable pair.
FDR_ALPHA = 0.30
MAX_CAPITAL_PER_SECTOR = 0.40

# Grid searched on TRAIN folds only; STOP_LOSS_ZSCORE is deliberately fixed
# (not in the grid) since "3-sigma stop-loss" is the explicit resume claim,
# not a free parameter to tune away.
ENTRY_ZSCORE_GRID = [1.5, 2.0, 2.5]
EXIT_ZSCORE_GRID = [0.0, 0.5]
ZSCORE_LOOKBACK_GRID = [20, 30, 45]
