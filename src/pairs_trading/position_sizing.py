"""Kelly-criterion position sizing from trailing trade returns."""

import numpy as np


def kelly_fraction(
    trade_returns: list[float],
    fraction_cap: float = 0.5,
    default_fraction: float = 0.1,
    min_trades: int = 10,
) -> float:
    """Kelly fraction for a strategy with per-trade returns `trade_returns`.

    Uses the continuous-return form f* = mu / sigma^2 (edge over variance),
    clipped to [0, 1] and then scaled by `fraction_cap` (e.g. 0.5 = half-Kelly,
    the standard practitioner haircut since full Kelly is far too aggressive
    given estimation error in mu/sigma).

    Falls back to `default_fraction` until at least `min_trades` trades of
    history are available, since Kelly estimated from a handful of trades is
    noise.
    """
    if len(trade_returns) < min_trades:
        return default_fraction

    returns = np.asarray(trade_returns, dtype=float)
    mu = returns.mean()
    var = returns.var()

    if var == 0 or mu <= 0:
        return 0.0

    f_full = mu / var
    f_full = float(np.clip(f_full, 0.0, 1.0))
    return f_full * fraction_cap


def position_size(
    capital: float,
    kelly_f: float,
    max_capital_per_pair: float,
) -> float:
    """Dollar amount to allocate to a single pair given total `capital`."""
    fraction = min(kelly_f, max_capital_per_pair)
    return capital * fraction
