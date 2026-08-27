from datetime import datetime, time
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from src.greeks import bs_gamma

CONTRACT_MULTIPLIER = 100
NEW_YORK = ZoneInfo("America/New_York")


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


def _clean_side(frame, option_type):
    required = {"strike", "impliedVolatility", "openInterest"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"option chain missing columns: {sorted(missing)}")

    clean = frame.copy()
    clean["option_type"] = option_type
    clean["strike"] = pd.to_numeric(clean["strike"], errors="coerce")
    clean["iv"] = pd.to_numeric(clean["impliedVolatility"], errors="coerce")
    clean["open_interest"] = pd.to_numeric(clean["openInterest"], errors="coerce")

    clean = clean.dropna(subset=["strike", "iv", "open_interest"])
    clean = clean[
        (clean["strike"] > 0)
        & (clean["iv"] > 0.005)
        & (clean["iv"] < 5.0)
        & (clean["open_interest"] > 0)
    ]
    return clean[["strike", "iv", "open_interest", "option_type"]]


def prepare_chain(calls, puts):
    calls_clean = _clean_side(calls, "call")
    puts_clean = _clean_side(puts, "put")
    return pd.concat([calls_clean, puts_clean], ignore_index=True)


def add_gamma_exposure(chain, spot, time_to_expiry_years, rate=0.04):
    """
    Add signed gamma exposure per 1% underlying move.

    Sign is a positioning convention: calls positive, puts negative. Open
    interest does not reveal the dealer side of each trade, so this should be
    interpreted as a market-positioning proxy rather than literal dealer risk.
    """
    result = chain.copy()
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
    grouped = (
        chain_with_gex.groupby("strike", as_index=False)["gex"]
        .sum()
        .sort_values("strike")
    )
    return grouped


def find_walls(chain_with_gex):
    """Return the strikes with the largest call and put gamma concentrations."""
    if chain_with_gex.empty:
        return np.nan, np.nan

    by_type = (
        chain_with_gex.groupby(["option_type", "strike"], as_index=False)["gex"]
        .sum()
    )
    calls = by_type[by_type["option_type"] == "call"]
    puts = by_type[by_type["option_type"] == "put"]

    call_wall = (
        calls.loc[calls["gex"].idxmax(), "strike"] if not calls.empty else np.nan
    )
    put_wall = (
        puts.loc[puts["gex"].idxmin(), "strike"] if not puts.empty else np.nan
    )
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

    spots = np.linspace(current_spot * lower, current_spot * upper, points)
    totals = np.array(
        [
            total_gex_at_spot(chain, s, time_to_expiry_years, rate)
            for s in spots
        ]
    )

    exact = np.where(np.isclose(totals, 0.0, atol=1e-9))[0]
    if len(exact):
        return float(spots[exact[0]])

    crossings = np.where(np.sign(totals[:-1]) != np.sign(totals[1:]))[0]
    if not len(crossings):
        return np.nan

    idx = crossings[np.argmin(np.abs(spots[crossings] - current_spot))]
    x1, x2 = spots[idx], spots[idx + 1]
    y1, y2 = totals[idx], totals[idx + 1]
    return float(x1 - y1 * (x2 - x1) / (y2 - y1))
