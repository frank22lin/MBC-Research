from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd


def load_aligned_data(
    market_cap_path: Path | str,
    price_path: Path | str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load market cap and price data keeping assets available in both sources."""

    market_cap_df = (
        pd.read_csv(market_cap_path, parse_dates=["DateTime"], index_col="DateTime")
        .sort_index()
    )
    price_df = (
        pd.read_csv(price_path, parse_dates=["DateTime"], index_col="DateTime")
        .sort_index()
    )

    cap_cols = [col for col in market_cap_df.columns if "Market Cap" in col]
    common_assets: list[str] = []
    price_col_map: dict[str, str] = {}

    for cap_col in cap_cols:
        price_col = cap_col.replace(" - Market Cap", " - Price")
        if price_col in price_df.columns:
            asset_name = cap_col.replace(" - Market Cap", "")
            common_assets.append(cap_col)
            price_col_map[price_col] = asset_name

    market_caps = market_cap_df[common_assets].rename(
        columns=lambda c: c.replace(" - Market Cap", "")
    )
    prices = price_df[list(price_col_map.keys())].rename(columns=price_col_map)

    shared_index = market_caps.index.intersection(prices.index)
    market_caps = market_caps.loc[shared_index]
    prices = prices.loc[shared_index]
    returns = prices.pct_change()

    return market_caps, prices, returns


def construct_market_portfolio(
    market_caps: pd.DataFrame,
    asset_returns: pd.DataFrame,
    rebalancing_freq: str = "D",
    top_n: int = 10,
) -> pd.Series:
    """Construct a market-cap weighted portfolio with arbitrary rebalancing."""

    market_caps = market_caps.sort_index()
    asset_returns = asset_returns.sort_index()
    available_dates = market_caps.index.intersection(asset_returns.index)
    market_caps = market_caps.loc[available_dates]
    asset_returns = asset_returns.loc[available_dates]

    if rebalancing_freq.upper() == "D":
        rebalancing_dates = market_caps.index
    else:
        rebalancing_dates = (
            market_caps.resample(rebalancing_freq).first().index
            .intersection(market_caps.index)
        )

    portfolio_returns: list[tuple[pd.Timestamp, float]] = []
    current_weights: pd.Series | None = None

    for date in asset_returns.index:
        if date in rebalancing_dates:
            day_caps = market_caps.loc[date].dropna()
            if day_caps.empty:
                current_weights = None
                continue
            selected = day_caps.nlargest(min(top_n, len(day_caps)))
            weights = selected / selected.sum()
            current_weights = weights

        if current_weights is None:
            continue

        day_ret = asset_returns.loc[date, current_weights.index].dropna()
        if day_ret.empty:
            continue

        valid_weights = current_weights.loc[day_ret.index]
        valid_weights = valid_weights / valid_weights.sum()
        portfolio_returns.append((date, float(np.dot(day_ret.values, valid_weights.values))))

    return pd.Series(dict(portfolio_returns), name="Market_Portfolio_Return").sort_index()


def compute_horizon_returns(daily_returns: pd.Series, horizon_days: int) -> pd.Series:
    """Aggregate daily simple returns into overlapping k-day simple returns."""

    daily_returns = daily_returns.dropna()
    if horizon_days <= 1:
        return daily_returns.copy()
    compounded = (1 + daily_returns).rolling(horizon_days).apply(np.prod, raw=True)
    return (compounded - 1).dropna()


def compute_resampled_returns(daily_returns: pd.Series, freq: str = "W") -> pd.Series:
    """Aggregate daily returns into non-overlapping period returns via resampling."""

    daily_returns = daily_returns.dropna()
    wealth_index = (1 + daily_returns).cumprod()
    period_wealth = wealth_index.resample(freq).last().dropna()
    period_returns = period_wealth.pct_change().dropna()
    return period_returns


def summarize_returns(returns: pd.Series, label: str, annual_factor: int = 365) -> None:
    """Print simple statistics for a return series."""

    returns = returns.dropna()
    mean = returns.mean()
    std = returns.std()
    print(label)
    print(f"  Observations: {len(returns)}")
    print(f"  Mean: {mean:.6f}")
    print(f"  Std: {std:.6f}")
    print(f"  Annualized Return: {mean * annual_factor:.2%}")
    print(f"  Annualized Volatility: {std * np.sqrt(annual_factor):.2%}\n")


