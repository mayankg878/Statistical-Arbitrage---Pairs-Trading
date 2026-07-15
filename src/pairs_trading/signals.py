"""Rolling z-score of the spread and stateful entry/exit/stop-loss logic."""

import numpy as np
import pandas as pd


def rolling_zscore(spread: pd.Series, lookback: int) -> pd.Series:
    mean = spread.rolling(lookback).mean()
    std = spread.rolling(lookback).std()
    return (spread - mean) / std.replace(0, np.nan)


def generate_positions(
    zscore: pd.Series,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    stop_z: float = 3.0,
) -> pd.Series:
    """Return a position series in {-1, 0, +1} (in "spread units": +1 means
    long y / short beta*x, -1 means the reverse).

    State machine, evaluated causally day by day:
      - flat -> enter +1 when z <= -entry_z, enter -1 when z >= +entry_z
      - in position -> exit to flat when |z| <= exit_z (mean reversion achieved)
      - in position -> exit to flat when the position has moved further adverse
        than stop_z (the "3-sigma stop-loss": the spread kept diverging past
        the entry rather than reverting, so the trade is cut)
    """
    positions = pd.Series(0, index=zscore.index, dtype=int)
    state = 0

    for t, z in zscore.items():
        if pd.isna(z):
            positions.loc[t] = state
            continue

        if state == 0:
            if z <= -entry_z:
                state = 1
            elif z >= entry_z:
                state = -1
        elif state == 1:
            if z >= -exit_z or z <= -stop_z:
                state = 0
        elif state == -1:
            if z <= exit_z or z >= stop_z:
                state = 0

        positions.loc[t] = state

    return positions
