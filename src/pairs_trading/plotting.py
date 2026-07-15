"""Matplotlib charts: portfolio equity curve, and an example pair's spread/z-score."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def plot_equity_curve(daily_returns: pd.Series, out_path: str) -> None:
    equity = (1 + daily_returns).cumprod()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(equity.index, equity.values, color="#1f77b4", linewidth=1.5)
    ax.set_title("Portfolio Equity Curve")
    ax.set_xlabel("Date")
    ax.set_ylabel("Growth of $1")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_pair_spread(zscore: pd.Series, entry_z: float, exit_z: float, stop_z: float,
                      pair_name: str, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(zscore.index, zscore.values, color="#333333", linewidth=1.0, label="z-score")
    for level, color, label in [
        (entry_z, "red", "entry"), (-entry_z, "red", None),
        (exit_z, "green", "exit"), (-exit_z, "green", None),
        (stop_z, "black", "stop-loss"), (-stop_z, "black", None),
    ]:
        ax.axhline(level, color=color, linestyle="--", linewidth=0.8, label=label)
    ax.set_title(f"Spread z-score: {pair_name}")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
