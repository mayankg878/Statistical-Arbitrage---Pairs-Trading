"""Single source of truth for universe, dates, and strategy parameters."""

from datetime import date

# --- Universe: tickers grouped by sector. Only within-sector pairs are
# screened for cointegration -- cross-sector relationships are usually
# spurious rather than economically grounded.
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
