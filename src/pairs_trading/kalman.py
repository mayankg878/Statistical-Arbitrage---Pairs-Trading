"""Recursive Kalman filter for a dynamic (time-varying) hedge ratio.

Standard random-walk state-space formulation (as in Ernie Chan's
"Algorithmic Trading"):

    state:       theta_t = [alpha_t, beta_t]'   (random walk: theta_t = theta_{t-1} + w_t)
    observation: y_t = [1, x_t] . theta_t + e_t

This is run strictly causally: theta_t is estimated using only observations
up to and including t, so it can be plugged into a backtest without
look-ahead bias.
"""

import numpy as np
import pandas as pd


def kalman_hedge_ratio(
    y: pd.Series,
    x: pd.Series,
    delta: float = 1e-4,
    obs_cov: float = 1e-3,
) -> pd.DataFrame:
    """Return a DataFrame indexed like y/x with columns alpha, beta, spread.

    `delta` controls process noise (how fast beta is allowed to drift);
    `obs_cov` is the observation noise variance.
    """
    n = len(y)
    y_vals = y.values
    x_vals = x.values

    # Process noise covariance for the 2-state [alpha, beta] random walk.
    Q = delta / (1 - delta) * np.eye(2)

    theta = np.zeros(2)          # [alpha, beta]
    P = np.zeros((2, 2))         # state covariance

    alphas = np.zeros(n)
    betas = np.zeros(n)
    spreads = np.zeros(n)

    for t in range(n):
        # Predict
        P = P + Q

        H = np.array([1.0, x_vals[t]])  # observation matrix row
        y_hat = H @ theta
        residual = y_vals[t] - y_hat

        S = H @ P @ H.T + obs_cov  # innovation variance
        K = (P @ H) / S            # Kalman gain

        theta = theta + K * residual
        P = P - np.outer(K, H) @ P

        alphas[t] = theta[0]
        betas[t] = theta[1]
        spreads[t] = residual  # residual == y_t - (alpha_{t-1} + beta_{t-1} x_t), the tradeable spread

    return pd.DataFrame(
        {"alpha": alphas, "beta": betas, "spread": spreads},
        index=y.index,
    )
