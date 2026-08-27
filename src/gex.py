from datetime import datetime, time
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from src.greeks import bs_gamma, implied_volatility

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

    clean = clean.dropna(subset=["strike", "open_interest"])
    clean = clean[(clean["strike"] > 0) & (clean["open_interest"] > 0)].copy()
    return clean.reset_index(drop=True)


def prepare_chain(calls, puts):
    """
    Normalize Yahoo's calls and puts while preserving contracts with valid OI.

    Yahoo occasionally returns missing or placeholder implied-volatility values,
    especially around stale/delayed snapshots. Those rows are kept here so IV can
    be recovered later from option prices instead of silently dropping the chain.
    """
    calls_clean = _clean_side(calls, "call")
    puts_clean = _clean_side(puts, "put")
    return pd.concat([calls_clean, puts_clean], ignore_index=True)


def _mark_price(row):
    bid = row["bid"]
    ask = row["ask"]
    last_price = row["last_price"]

    if np.isfinite(bid) and np.isfinite(ask) and bid > 0 and ask > 0 and ask >= bid:
        return (bid + ask) / 2.0, "mid"
    if np.isfinite(last_price) and last_price > 0:
        return float(last_price), "last"
    return np.nan, "missing"


def _resolve_implied_volatility(chain, spot, time_to_expiry_years, rate):
    result = chain.copy()
    valid_yahoo = result["iv"].between(MIN_VALID_IV, MAX_VALID_IV, inclusive="neither")
    result["iv_source"] = np.where(valid_yahoo, "yahoo", "missing")

    missing_indices = result.index[~valid_yahoo]
    for idx in missing_indices:
        row = result.loc[idx]
        price, price_source = _mark_price(row)
        if not np.isfinite(price):
            continue

        solved = implied_volatility(
            price,
            spot,
            row["strike"],
            time_to_expiry_years,
            rate,
            row["option_type"],
        )
        if np.isfinite(solved) and MIN_VALID_IV < solved < MAX_VALID_IV:
            result.at[idx, "iv"] = solved
            result.at[idx, "iv_source"] = f"inferred_{price_source}"

    # If an individual stale price cannot be inverted, use the local smile from
    # contracts whose IV is usable. This avoids deleting an otherwise valid OI
    # concentration because one Yahoo field is missing.
    for option_type in ("call", "put"):
        side_mask = result["option_type"].eq(option_type)
        side = result.loc[side_mask].sort_values("strike")
        resolved = side[side["iv"].between(MIN_VALID_IV, MAX_VALID_IV, inclusive="neither")]
        unresolved = side[~side["iv"].between(MIN_VALID_IV, MAX_VALID_IV, inclusive="neither")]
        if unresolved.empty or resolved.empty:
            continue

        interpolated = np.interp(
            unresolved["strike"].to_numpy(),
            resolved["strike"].to_numpy(),
            resolved["iv"].to_numpy(),
        )
        result.loc[unresolved.index, "iv"] = interpolated
        result.loc[unresolved.index, "iv_source"] = "interpolated"

    usable = result["iv"].between(MIN_VALID_IV, MAX_VALID_IV, inclusive="neither")
    return result.loc[usable].reset_index(drop=True)


def add_gamma_exposure(chain, spot, time_to_expiry_years, rate=0.04):
    """
    Add signed gamma exposure per 1% underlying move.

    Sign is a positioning convention: calls positive, puts negative. Open
    interest does not reveal the dealer side of each trade, so this should be
    interpreted as a market-positioning proxy rather than literal dealer risk.
    """
    if chain.empty:
        raise ValueError("no contracts with positive open interest were returned")

    result = _resolve_implied_volatility(chain, spot, time_to_expiry_years, rate)
    if result.empty:
        raise ValueError(
            "Yahoo returned open interest, but no usable or recoverable implied volatility values"
        )

    result["gamma"] = result.apply(
        lambda row: bs_gamma(
            spot,
            row["strike"],
            time_to_expiry_years,
            rate,
            row["iv"],
        ),
        axis=1,
    )

    sign = np.where(result["option_type"].eq("call"), 1.0, -1.0)
    result["gex"] = (
        sign
        * result["gamma"]
        * result["open_interest"]
        * CONTRACT_MULTIPLIER
        * spot**2
        * 0.01
    )
    return result


def gex_by_strike(chain_with_gex):
    return (
        chain_with_gex.groupby("strike", as_index=False)["gex"]
        .sum()
        .sort_values("strike")
    )


def find_walls(chain_with_gex):
    """Return the strikes with the largest call and put gamma concentrations."""
    if chain_with_gex.empty:
        return np.nan, np.nan

    by_type = chain_with_gex.groupby(["option_type", "strike"], as_index=False)["gex"].sum()
    calls = by_type[by_type["option_type"] == "call"]
    puts = by_type[by_type["option_type"] == "put"]

    call_wall = calls.loc[calls["gex"].idxmax(), "strike"] if not calls.empty else np.nan
    put_wall = puts.loc[puts["gex"].idxmin(), "strike"] if not puts.empty else np.nan
    return float(call_wall), float(put_wall)


def total_gex_at_spot(chain, spot, time_to_expiry_years, rate=0.04):
    exposed = add_gamma_exposure(chain, spot, time_to_expiry_years, rate)
    return exposed["gex"].sum()


def zero_gamma_level(
    chain,
    current_spot,
    time_to_expiry_years,
    rate=0.04,
    lower=0.8,
    upper=1.2,
    points=161,
):
    """Estimate the spot where aggregate signed GEX crosses zero."""
    if chain.empty:
        return np.nan

    # Resolve IV once at the current spot. Re-solving IV at every hypothetical
    # spot would change the volatility surface and is not what this scan means.
    resolved = _resolve_implied_volatility(chain, current_spot, time_to_expiry_years, rate)
    if resolved.empty:
        return np.nan

    spots = np.linspace(current_spot * lower, current_spot * upper, points)
    totals = []
    for candidate_spot in spots:
        exposed = resolved.copy()
        exposed["gamma"] = exposed.apply(
            lambda row: bs_gamma(
                candidate_spot,
                row["strike"],
                time_to_expiry_years,
                rate,
                row["iv"],
            ),
            axis=1,
        )
        sign = np.where(exposed["option_type"].eq("call"), 1.0, -1.0)
        exposed["gex"] = (
            sign
            * exposed["gamma"]
            * exposed["open_interest"]
            * CONTRACT_MULTIPLIER
            * candidate_spot**2
            * 0.01
        )
        totals.append(exposed["gex"].sum())

    totals = np.asarray(totals)

    # Only count a genuine sign change. Far from the active strikes, gamma can
    # numerically underflow to exactly zero; treating that as a zero-gamma level
    # creates a false crossing at the edge of the scan.
    crossings = np.where((totals[:-1] * totals[1:]) < 0)[0]
    if not len(crossings):
        return np.nan

    idx = crossings[np.argmin(np.abs(spots[crossings] - current_spot))]
    x1, x2 = spots[idx], spots[idx + 1]
    y1, y2 = totals[idx], totals[idx + 1]
    return float(x1 - y1 * (x2 - x1) / (y2 - y1))
