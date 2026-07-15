"""Download and cache daily adjusted-close price data via yfinance."""

import os
import pandas as pd
import yfinance as yf


def _cache_path(data_dir: str) -> str:
    return os.path.join(data_dir, "prices.csv")


def get_price_data(
    tickers: list[str],
    start: str,
    end: str,
    data_dir: str = "data",
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Return a DataFrame of daily close prices, columns=tickers, indexed by date.

    Caches to `{data_dir}/prices.csv` so repeated runs and tests don't hit the
    network. If cached tickers/date range don't cover the request, refetches.
    """
    os.makedirs(data_dir, exist_ok=True)
    cache_file = _cache_path(data_dir)

    if not force_refresh and os.path.exists(cache_file):
        cached = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        if set(tickers).issubset(cached.columns) and cached.index.min() <= pd.Timestamp(start) \
                and cached.index.max() >= pd.Timestamp(end) - pd.Timedelta(days=5):
            return cached[tickers].loc[start:end]

    raw = yf.download(tickers, start=start, end=end, progress=False, auto_adjust=True)
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]]
        prices.columns = tickers

    # Large batch requests occasionally drop a ticker's data transiently
    # (rate limiting, not necessarily delisting) -- retry those individually
    # before giving up on them.
    missing = [t for t in tickers if t not in prices.columns or prices[t].dropna().empty]
    for ticker in missing:
        retry = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        if not retry.empty:
            prices[ticker] = retry["Close"]

    prices = prices.dropna(axis=1, how="all").dropna(how="all").ffill()

    prices.to_csv(cache_file)
    return prices.loc[start:end]
