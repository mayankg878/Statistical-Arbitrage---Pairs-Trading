import numpy as np
import pandas as pd

from pairs_trading.signals import generate_positions, rolling_zscore


def test_rolling_zscore_basic():
    spread = pd.Series([1, 2, 3, 4, 5, 4, 3, 2, 1, 0], dtype=float)
    z = rolling_zscore(spread, lookback=3)
    assert z.iloc[:2].isna().all()
    assert np.isfinite(z.iloc[3:]).all()


def test_entry_and_mean_reversion_exit():
    dates = pd.date_range("2020-01-01", periods=6, freq="D")
    z = pd.Series([0.0, -2.5, -2.5, -0.4, -0.4, -0.4], index=dates)
    positions = generate_positions(z, entry_z=2.0, exit_z=0.5, stop_z=3.0)
    assert list(positions) == [0, 1, 1, 0, 0, 0]


def test_short_entry_and_exit():
    dates = pd.date_range("2020-01-01", periods=5, freq="D")
    z = pd.Series([0.0, 2.5, 2.5, 0.3, 0.3], index=dates)
    positions = generate_positions(z, entry_z=2.0, exit_z=0.5, stop_z=3.0)
    assert list(positions) == [0, -1, -1, 0, 0]


def test_stop_loss_triggers():
    dates = pd.date_range("2020-01-01", periods=5, freq="D")
    z = pd.Series([0.0, -2.2, -2.8, -3.5, -1.0], index=dates)
    positions = generate_positions(z, entry_z=2.0, exit_z=0.5, stop_z=3.0)
    # enters at -2.2, stays in through -2.8, stop triggers once |z| >= 3.0,
    # then stays flat since -1.0 doesn't re-cross the entry threshold
    assert list(positions) == [0, 1, 1, 0, 0]


def test_nan_zscore_holds_state():
    dates = pd.date_range("2020-01-01", periods=4, freq="D")
    z = pd.Series([np.nan, np.nan, -2.5, np.nan], index=dates)
    positions = generate_positions(z, entry_z=2.0, exit_z=0.5, stop_z=3.0)
    assert list(positions) == [0, 0, 1, 1]
