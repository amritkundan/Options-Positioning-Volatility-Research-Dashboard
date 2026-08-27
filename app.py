from datetime import datetime, time
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import streamlit as st

from src.backtest import summarize_results, variance_premium_backtest
from src.charts import (
    cumulative_pnl_chart,
    gex_profile_chart,
    pnl_histogram,
    vol_comparison_chart,
)
from src.gex import (
    add_gamma_exposure,
    find_walls,
    gex_by_strike,
    prepare_chain,
    time_to_expiry,
    zero_gamma_level,
)
from src.market_data import load_vrp_data, option_chain, option_expirations

st.set_page_config(page_title="Options Research Dashboard", layout="wide")

NEW_YORK = ZoneInfo("America/New_York")


def compact_dollars(value):
    magnitude = abs(value)
    if magnitude >= 1e9:
        return f"${value / 1e9:,.2f}B"
    if magnitude >= 1e6:
        return f"${value / 1e6:,.2f}M"
    if magnitude >= 1e3:
        return f"${value / 1e3:,.1f}K"
    return f"${value:,.0f}"


def regular_session_is_open(now=None):
    now = (now or datetime.now(NEW_YORK)).astimezone(NEW_YORK)
    if now.weekday() >= 5:
        return False
    return time(9, 30) <= now.time() < time(16, 0)


@st.cache_data(ttl=900, show_spinner=False)
def cached_vrp_data(ticker, vol_ticker, period):
    return load_vrp_data(ticker, vol_ticker, period)


@st.cache_data(ttl=300, show_spinner=False)
def cached_expirations(ticker):
    return option_expirations(ticker)


@st.cache_data(ttl=300, show_spinner=False)
def cached_chain(ticker, expiration):
    return option_chain(ticker, expiration)


st.title("Options Positioning & Volatility Research")
st.caption(
    "Live gamma-exposure positioning plus a historical volatility-risk-premium proxy. "
    "Built for research and education, not live execution."
)

gex_tab, vrp_tab, notes_tab = st.tabs(
    ["Live GEX", "Volatility Risk Premium", "Methodology"]
)

with gex_tab:
    left, right = st.columns([1, 3])

    with left:
        ticker = st.text_input("Ticker", "SPY").strip().upper()
        rate = st.number_input(
            "Risk-free rate",
            min_value=0.0,
            max_value=0.20,
            value=0.04,
            step=0.005,
            format="%.3f",
        )

        try:
            expirations = cached_expirations(ticker)
            expiration = st.selectbox("Expiration", expirations)
            strike_width = st.slider("Strike range around spot", 5, 40, 15)
        except Exception as exc:
            st.error(f"Could not load expirations: {exc}")
            expirations = []
            expiration = None

    with right:
        if expiration:
            try:
                calls, puts, spot = cached_chain(ticker, expiration)
                t_exp = time_to_expiry(expiration)
                raw_contracts = len(calls) + len(puts)
                chain_with_oi = prepare_chain(calls, puts)
                chain = add_gamma_exposure(chain_with_oi, spot, t_exp, rate)
                profile = gex_by_strike(chain)

                lower = spot * (1 - strike_width / 100)
                upper = spot * (1 + strike_width / 100)
                visible = profile[profile["strike"].between(lower, upper)]
                visible_chain = chain[chain["strike"].between(lower, upper)]

                call_wall, put_wall = find_walls(visible_chain)
                zero_gamma = zero_gamma_level(chain_with_oi, spot, t_exp, rate)
                total_gex = chain["gex"].sum()

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Spot", f"${spot:,.2f}")
                m2.metric("Net GEX", compact_dollars(total_gex))
                m3.metric(
                    "Zero gamma",
                    "No crossing" if np.isnan(zero_gamma) else f"${zero_gamma:,.2f}",
                )
                wall_text = (
                    f"${put_wall:,.0f} / ${call_wall:,.0f}"
                    if not np.isnan(put_wall) and not np.isnan(call_wall)
                    else "Unavailable"
                )
                m4.metric("Put / call wall", wall_text)

                inferred = int(chain["iv_source"].ne("yahoo").sum())
                session_text = (
                    "Regular session open"
                    if regular_session_is_open()
                    else "Outside regular session — using Yahoo's latest available snapshot"
                )
                st.caption(
                    f"{session_text} • {raw_contracts:,} contracts returned • "
                    f"{len(chain_with_oi):,} with positive OI • {len(chain):,} usable for GEX"
                    + (f" • IV recovered for {inferred:,}" if inferred else "")
                )

                if inferred:
                    st.info(
                        "Some Yahoo implied-volatility fields were missing or placeholders. "
                        "The dashboard recovered those IVs from option prices or the nearby volatility smile instead of dropping the contracts."
                    )

                st.plotly_chart(
                    gex_profile_chart(
                        visible,
                        spot,
                        call_wall,
                        put_wall,
                        zero_gamma,
                    ),
                    width="stretch",
                )

                top = visible.assign(abs_gex=visible["gex"].abs()).nlargest(
                    12, "abs_gex"
                )[["strike", "gex"]]
                st.dataframe(top, width="stretch", hide_index=True)
            except Exception as exc:
                st.error(f"Could not build the GEX profile: {exc}")

with vrp_tab:
    controls, output = st.columns([1, 3])

    with controls:
        underlying = st.text_input("Underlying", "SPY", key="vrp_underlying").strip().upper()
        vol_ticker = st.text_input("Volatility proxy", "^VIX").strip().upper()
        period = st.selectbox("History", ["2y", "5y", "10y"], index=2)
        window = st.select_slider(
            "Forward window",
            options=[5, 10, 21, 42, 63],
            value=21,
            format_func=lambda x: f"{x} trading days",
        )
        overlap = st.checkbox("Allow overlapping windows", value=False)
        side = st.radio("View P&L from", ["Short gamma", "Long gamma"])

    with output:
        try:
            history = cached_vrp_data(underlying, vol_ticker, period)
            stride = 1 if overlap else window
            results = variance_premium_backtest(
                history["spot"],
                history["implied_vol_proxy"],
                window=window,
                stride=stride,
            )

            pnl_col = "short_gamma_pnl" if side == "Short gamma" else "long_gamma_pnl"
            summary = summarize_results(results, pnl_col)

            a, b, c, d = st.columns(4)
            a.metric("Periods", f"{summary['periods']:,}")
            b.metric("Win rate", f"{summary['win_rate']:.1%}")
            c.metric("Average P&L", f"${summary['avg_pnl']:,.2f}")
            d.metric("Total P&L", f"${summary['total_pnl']:,.2f}")

            st.plotly_chart(vol_comparison_chart(results), width="stretch")

            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(
                    cumulative_pnl_chart(
                        results,
                        side="short" if side == "Short gamma" else "long",
                    ),
                    width="stretch",
                )
            with c2:
                st.plotly_chart(
                    pnl_histogram(
                        results,
                        side="short" if side == "Short gamma" else "long",
                    ),
                    width="stretch",
                )

            yearly = results.assign(year=pd.to_datetime(results["date"]).dt.year)
            yearly = yearly.groupby("year")[pnl_col].agg(
                periods="count",
                avg_pnl="mean",
                total_pnl="sum",
                win_rate=lambda x: (x > 0).mean(),
            )
            st.subheader("Year-by-year")
            st.dataframe(yearly, width="stretch")
        except Exception as exc:
            st.error(f"Could not run the historical analysis: {exc}")

with notes_tab:
    st.markdown(
        """
### What this project measures

**Live GEX** aggregates Black-Scholes gamma across the current option chain and
weights each contract by open interest. Calls are assigned positive exposure and
puts negative exposure as a common positioning convention.

**Volatility Risk Premium** compares an implied-volatility proxy at each start
date with the volatility realized over the following window. P&L is a
gamma-weighted variance-spread approximation, not a historical option-price
backtest.

### Important limitations

- VIX is a 30-day SPX variance measure, not the exact ATM implied volatility of SPY.
- Open interest does not identify which side dealers actually hold. Signed GEX is
  therefore a positioning proxy.
- The historical module does not include bid/ask spreads, commissions, discrete
  hedge slippage, dividends, or actual historical option prices.
- Yahoo Finance data is convenient for a portfolio project, but production
  research should use a licensed market-data source with point-in-time chains.
        """
    )
