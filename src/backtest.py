import numpy as np
import pandas as pd

from src.greeks import bs_gamma

TRADING_DAYS = 252


def forward_realized_vol(prices, window=21):
    """Annualized realized vol measured over the next `window` trading days."""
    prices = pd.Series(prices, dtype=float).dropna()
    if window < 2:
        raise ValueError("window must be at least 2 trading days")
    if len(prices) <= window:
        return pd.Series(np.nan, index=prices.index, dtype=float)

    log_returns = np.log(prices / prices.shift(1))
    realized = pd.Series(np.nan, index=prices.index, dtype=float)

    for i in range(len(prices) - window):
        sample = log_returns.iloc[i + 1 : i + 1 + window]
        realized.iloc[i] = sample.std(ddof=1) * np.sqrt(TRADING_DAYS)

    return realized


def variance_premium_backtest(
    prices,
    implied_vol,
    window=21,
    rate=0.04,
    stride=None,
):
    """
    Gamma-weighted variance-spread proxy.

    `implied_vol` should be a decimal series (0.20 = 20%). This is not an
    historical option-price backtest; it isolates the difference between
    entry implied variance and subsequent realized variance.
    """
    data = pd.concat(
        [
            pd.Series(prices, name="spot", dtype=float),
            pd.Series(implied_vol, name="implied_vol", dtype=float),
        ],
        axis=1,
        join="inner",
    ).dropna()

    if window < 2:
        raise ValueError("window must be at least 2 trading days")
    if stride is None:
        stride = window
    if stride < 1:
        raise ValueError("stride must be positive")

    realized = forward_realized_vol(data["spot"], window)
    time_to_expiry = window / TRADING_DAYS
    records = []

    for i in range(0, len(data) - window, stride):
        row = data.iloc[i]
        sigma_i = row["implied_vol"]
        sigma_r = realized.iloc[i]

        if sigma_i <= 0 or pd.isna(sigma_r):
            continue

        spot = row["spot"]
        gamma = bs_gamma(spot, spot, time_to_expiry, rate, sigma_i)
        long_pnl = (
            0.5
            * gamma
            * spot**2
            * (sigma_r**2 - sigma_i**2)
            * time_to_expiry
        )

        records.append(
            {
                "date": data.index[i],
                "spot": spot,
                "implied_vol": sigma_i,
                "realized_vol": sigma_r,
                "variance_spread": sigma_i**2 - sigma_r**2,
                "gamma": gamma,
                "long_gamma_pnl": long_pnl,
                "short_gamma_pnl": -long_pnl,
            }
        )

    return pd.DataFrame(records)


def daily_gamma_backtest(
    prices,
    implied_vol,
    window=21,
    rate=0.04,
    stride=None,
):
    """Daily gamma/theta approximation using a fixed entry volatility."""
    data = pd.concat(
        [
            pd.Series(prices, name="spot", dtype=float),
            pd.Series(implied_vol, name="implied_vol", dtype=float),
        ],
        axis=1,
        join="inner",
    ).dropna()

    if window < 2:
        raise ValueError("window must be at least 2 trading days")
    if stride is None:
        stride = window
    if stride < 1:
        raise ValueError("stride must be positive")

    dt = 1 / TRADING_DAYS
    records = []

    for i in range(0, len(data) - window, stride):
        strike = data["spot"].iloc[i]
        sigma = data["implied_vol"].iloc[i]
        if sigma <= 0:
            continue

        daily_pnl = []
        for j in range(1, window + 1):
            spot_prev = data["spot"].iloc[i + j - 1]
            spot_now = data["spot"].iloc[i + j]
            time_left = (window - j + 1) / TRADING_DAYS
            gamma = bs_gamma(spot_prev, strike, time_left, rate, sigma)

            d_spot = spot_now - spot_prev
            pnl = 0.5 * gamma * (
                d_spot**2 - sigma**2 * spot_prev**2 * dt
            )
            daily_pnl.append(pnl)

        long_pnl = float(np.sum(daily_pnl))
        records.append(
            {
                "date": data.index[i],
                "spot": strike,
                "implied_vol": sigma,
                "long_gamma_pnl": long_pnl,
                "short_gamma_pnl": -long_pnl,
            }
        )

    return pd.DataFrame(records)


def summarize_results(results, pnl_column="long_gamma_pnl"):
    if results.empty:
        return {
            "periods": 0,
            "avg_pnl": np.nan,
            "total_pnl": np.nan,
            "win_rate": np.nan,
        }

    pnl = results[pnl_column]
    summary = {
        "periods": len(results),
        "avg_pnl": pnl.mean(),
        "total_pnl": pnl.sum(),
        "win_rate": (pnl > 0).mean(),
        "pnl_std": pnl.std(ddof=1),
        "pnl_skew": pnl.skew(),
    }

    if "implied_vol" in results:
        summary["avg_implied_vol"] = results["implied_vol"].mean()
    if "realized_vol" in results:
        summary["avg_realized_vol"] = results["realized_vol"].mean()

    return summary
