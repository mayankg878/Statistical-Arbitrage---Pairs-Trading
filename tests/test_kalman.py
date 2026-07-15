import numpy as np
import pandas as pd

from pairs_trading.kalman import kalman_hedge_ratio


def test_recovers_constant_beta():
    rng = np.random.default_rng(0)
    n = 500
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    x = pd.Series(np.cumsum(rng.normal(0, 1, n)) + 100, index=dates)
    true_beta = 1.5
    y = 2.0 + true_beta * x + pd.Series(rng.normal(0, 0.1, n), index=dates)

    kf = kalman_hedge_ratio(y, x, delta=1e-4, obs_cov=0.1)

    # Filter needs time to converge; check the back half is close to truth.
    tail_beta = kf["beta"].iloc[250:]
    assert np.allclose(tail_beta.mean(), true_beta, atol=0.1)


def test_tracks_slowly_varying_beta():
    rng = np.random.default_rng(1)
    n = 800
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    x = pd.Series(np.cumsum(rng.normal(0, 1, n)) + 100, index=dates)
    beta_path = 1.0 + 0.5 * np.sin(np.linspace(0, 2 * np.pi, n))
    y = pd.Series(beta_path * x.values + rng.normal(0, 0.1, n), index=dates)

    kf = kalman_hedge_ratio(y, x, delta=1e-3, obs_cov=0.1)

    # Filtered beta should broadly track the true (slow) path after convergence.
    corr = np.corrcoef(kf["beta"].iloc[100:], beta_path[100:])[0, 1]
    assert corr > 0.8


def test_spread_is_finite():
    rng = np.random.default_rng(2)
    n = 200
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    x = pd.Series(np.cumsum(rng.normal(0, 1, n)) + 50, index=dates)
    y = 1.2 * x + pd.Series(rng.normal(0, 1, n), index=dates)

    kf = kalman_hedge_ratio(y, x)
    assert np.isfinite(kf["spread"]).all()
    assert np.isfinite(kf["beta"]).all()
