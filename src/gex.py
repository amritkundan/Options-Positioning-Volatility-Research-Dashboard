from datetime import datetime, time
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from src.greeks import implied_volatility

CONTRACT_MULTIPLIER = 100
NEW_YORK = ZoneInfo("America/New_York")
MIN_VALID_IV = 0.005
MAX_VALID_IV = 5.0


def time_to_expiry(expiration, now=None):
    """Year fraction until 4:00 p.m. New York time on expiration day."""
    now = now or datetime.now(NEW_YORK)
    if now.tzinfo is None:
        now = now.replace(tzinfo=NEW_YORK)
    else:
        now = now.astimezone(NEW_YORK)

    expiry_date = pd.Timestamp(expiration).date()
    expiry = datetime.combine(expiry_date, time(16, 0), tzinfo=NEW_YORK)
    seconds = max((expiry - now).total_seconds(), 60.0)
    return seconds / (365.0 * 24 * 60 * 60)


def _numeric_column(frame, column):
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _clean_side(frame, option_type):
    required = {"strike", "openInterest"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"option chain missing columns: {sorted(missing)}")

    clean = pd.DataFrame(index=frame.index)
    clean["option_type"] = option_type
    clean["strike"] = _numeric_column(frame, "strike")
    clean["iv"] = _numeric_column(frame, "impliedVolatility")
    clean["open_interest"] = _numeric_column(frame, "openInterest")
    clean["bid"] = _numeric_column(frame, "bid")
    clean["ask"] = _numeric_column(frame, "ask")
    clean["last_price"] = _numeric_column(frame, "lastPrice")

    if "expiration" in frame.columns:
        clean["expiration"] = frame["expiration"].astype(str).to_numpy()

    clean = clean.dropna(subset=["strike", "open_interest"])
    clean = clean[(clean["strike"] > 0) & (clean["open_interest"] > 0)].copy()
    return clean.reset_index(drop=True)


def prepare_chain(calls, puts, now=None):
    """Normalize calls and puts while preserving every contract with positive OI."""
    calls_clean = _clean_side(calls, "call")
    puts_clean = _clean_side(puts, "put")
    chain = pd.concat([calls_clean, puts_clean], ignore_index=True)

    if "expiration" in chain.columns and not chain.empty:
        t_by_expiry = {
            expiry: time_to_expiry(expiry, now=now)
            for expiry in chain["expiration"].dropna().unique()
        }
        chain["time_to_expiry"] = chain["expiration"].map(t_by_expiry).astype(float)

    return chain


def _mark_price(row):
    bid = row["bid"]
    ask = row["ask"]
    last_price = row["last_price"]

    if np.isfinite(bid) and np.isfinite(ask) and bid > 0 and ask > 0 and ask >= bid:
        return (bid + ask) / 2.0, "mid"
    if np.isfinite(last_price) and last_price > 0:
        return float(last_price), "last"
    return np.nan, "missing"


def _ensure_time_column(chain, time_to_expiry_years=None):
    result = chain.copy()
    if "time_to_expiry" in result.columns:
        result["time_to_expiry"] = pd.to_numeric(
            result["time_to_expiry"], errors="coerce"
        )
    elif time_to_expiry_years is not None:
        result["time_to_expiry"] = float(time_to_expiry_years)
    else:
        raise ValueError("time to expiry is required")

    result = result[result["time_to_expiry"].gt(0)].copy()
    return result


def _resolve_implied_volatility(
    chain,
    spot,
    rate,
    time_to_expiry_years=None,
    fallback_volatility=None,
    fallback_label="proxy",
):
    result = _ensure_time_column(chain, time_to_expiry_years)
    valid_yahoo = result["iv"].between(MIN_VALID_IV, MAX_VALID_IV, inclusive="neither")
    result["iv_source"] = np.where(valid_yahoo, "yahoo", "missing")

    for idx in result.index[~valid_yahoo]:
        row = result.loc[idx]
        price, price_source = _mark_price(row)
        if not np.isfinite(price):
            continue

        solved = implied_volatility(
            price,
            spot,
            row["strike"],
            row["time_to_expiry"],
            rate,
            row["option_type"],
        )
        if np.isfinite(solved) and MIN_VALID_IV < solved < MAX_VALID_IV:
            result.at[idx, "iv"] = solved
            result.at[idx, "iv_source"] = f"inferred_{price_source}"

    group_cols = ["option_type"]
    if "expiration" in result.columns:
        group_cols.append("expiration")

    for _, side in result.groupby(group_cols, dropna=False):
        side = side.sort_values("strike")
        usable = side[side["iv"].between(MIN_VALID_IV, MAX_VALID_IV, inclusive="neither")]
        missing = side[~side["iv"].between(MIN_VALID_IV, MAX_VALID_IV, inclusive="neither")]
        if usable.empty or missing.empty:
            continue

        values = np.interp(
            missing["strike"].to_numpy(),
            usable["strike"].to_numpy(),
            usable["iv"].to_numpy(),
        )
        result.loc[missing.index, "iv"] = values
        result.loc[missing.index, "iv_source"] = "interpolated"

    usable = result["iv"].between(MIN_VALID_IV, MAX_VALID_IV, inclusive="neither")
    if fallback_volatility is not None:
        fallback_volatility = float(fallback_volatility)
        if not MIN_VALID_IV < fallback_volatility < MAX_VALID_IV:
            raise ValueError("fallback volatility is outside the supported range")
        missing = ~usable
        if missing.any():
            result.loc[missing, "iv"] = fallback_volatility
            result.loc[missing, "iv_source"] = fallback_label
            usable = result["iv"].between(MIN_VALID_IV, MAX_VALID_IV, inclusive="neither")

    return result.loc[usable].reset_index(drop=True)


def _gamma_array(spot, strikes, times, volatility, rate):
    strikes = np.asarray(strikes, dtype=float)
    times = np.asarray(times, dtype=float)
    volatility = np.asarray(volatility, dtype=float)

    sqrt_t = np.sqrt(times)
    d1 = (
        np.log(float(spot) / strikes)
        + (rate + 0.5 * volatility**2) * times
    ) / (volatility * sqrt_t)
    pdf = np.exp(-0.5 * d1**2) / np.sqrt(2.0 * np.pi)
    return pdf / (float(spot) * volatility * sqrt_t)


def add_gamma_exposure(
    chain,
    spot,
    time_to_expiry_years=None,
    rate=0.04,
    fallback_volatility=None,
    fallback_label="proxy",
):
    """Add signed dollar gamma exposure per 1% move to an option book."""
    if chain.empty:
        raise ValueError("no contracts with positive open interest were returned")

    result = _resolve_implied_volatility(
        chain,
        spot,
        rate,
        time_to_expiry_years=time_to_expiry_years,
        fallback_volatility=fallback_volatility,
        fallback_label=fallback_label,
    )
    if result.empty:
        raise ValueError("no usable implied volatility values were available")

    result["gamma"] = _gamma_array(
        spot,
        result["strike"].to_numpy(),
        result["time_to_expiry"].to_numpy(),
        result["iv"].to_numpy(),
        rate,
    )

    sign = np.where(result["option_type"].eq("call"), 1.0, -1.0)
    result["gex"] = (
        sign
        * result["gamma"].to_numpy()
        * result["open_interest"].to_numpy()
        * CONTRACT_MULTIPLIER
        * float(spot) ** 2
        * 0.01
    )
    return result


def gex_by_strike(chain_with_gex):
    """Return call, put, and net GEX at each strike."""
    grouped = (
        chain_with_gex.groupby(["strike", "option_type"], as_index=False)["gex"].sum()
        .pivot(index="strike", columns="option_type", values="gex")
        .fillna(0.0)
        .reset_index()
    )
    grouped.columns.name = None
    if "call" not in grouped:
        grouped["call"] = 0.0
    if "put" not in grouped:
        grouped["put"] = 0.0
    grouped = grouped.rename(columns={"call": "call_gex", "put": "put_gex"})
    grouped["net_gex"] = grouped["call_gex"] + grouped["put_gex"]
    return grouped.sort_values("strike").reset_index(drop=True)


def find_walls(chain_with_gex):
    """Return call and put wall strikes from the full option book."""
    if chain_with_gex.empty:
        return np.nan, np.nan

    grouped = chain_with_gex.groupby(["option_type", "strike"], as_index=False)["gex"].sum()
    calls = grouped[grouped["option_type"].eq("call")]
    puts = grouped[grouped["option_type"].eq("put")]

    call_wall = calls.loc[calls["gex"].idxmax(), "strike"] if not calls.empty else np.nan
    put_wall = puts.loc[puts["gex"].idxmin(), "strike"] if not puts.empty else np.nan
    return float(call_wall), float(put_wall)


def zero_gamma_level(
    chain,
    current_spot,
    time_to_expiry_years=None,
    rate=0.04,
    fallback_volatility=None,
    fallback_label="proxy",
    lower=0.82,
    upper=1.18,
    points=181,
):
    """
    Re-price every contract across a spot grid and find the nearest GEX zero cross.

    Each contract keeps its own expiration and IV. This is the structural gamma
    flip; it is not a cumulative-by-strike shortcut.
    """
    if chain.empty:
        return np.nan

    resolved = _resolve_implied_volatility(
        chain,
        current_spot,
        rate,
        time_to_expiry_years=time_to_expiry_years,
        fallback_volatility=fallback_volatility,
        fallback_label=fallback_label,
    )
    if resolved.empty:
        return np.nan

    strikes = resolved["strike"].to_numpy(dtype=float)
    times = resolved["time_to_expiry"].to_numpy(dtype=float)
    volatility = resolved["iv"].to_numpy(dtype=float)
    open_interest = resolved["open_interest"].to_numpy(dtype=float)
    sign = np.where(resolved["option_type"].eq("call"), 1.0, -1.0)

    spots = np.linspace(float(current_spot) * lower, float(current_spot) * upper, points)
    totals = np.empty(points, dtype=float)

    for i, candidate in enumerate(spots):
        gamma = _gamma_array(candidate, strikes, times, volatility, rate)
        gex = (
            sign
            * gamma
            * open_interest
            * CONTRACT_MULTIPLIER
            * candidate**2
            * 0.01
        )
        totals[i] = gex.sum()

    crossings = np.where((totals[:-1] * totals[1:]) < 0)[0]
    if not len(crossings):
        return np.nan

    idx = crossings[np.argmin(np.abs(spots[crossings] - current_spot))]
    x1, x2 = spots[idx], spots[idx + 1]
    y1, y2 = totals[idx], totals[idx + 1]
    return float(x1 - y1 * (x2 - x1) / (y2 - y1))
