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
from src.gex import add_gamma_exposure, find_walls, gex_by_strike, prepare_chain, zero_gamma_level
from src.market_data import gex_volatility_proxy, load_vrp_data, option_book

st.set_page_config(page_title="Options Gamma Dashboard", page_icon="Γ", layout="wide")

NEW_YORK = ZoneInfo("America/New_York")


def compact_dollars(value):
    magnitude = abs(value)
    if magnitude >= 1e9:
        return f"${value / 1e9:,.2f}B"
    if magnitude >= 1e6:
        return f"${value / 1e6:,.1f}M"
    if magnitude >= 1e3:
        return f"${value / 1e3:,.1f}K"
    return f"${value:,.0f}"


def regular_session_is_open(now=None):
    now = (now or datetime.now(NEW_YORK)).astimezone(NEW_YORK)
    return now.weekday() < 5 and time(9, 30) <= now.time() < time(16, 0)


@st.cache_data(ttl=300, show_spinner=False)
def cached_option_book(ticker, horizon_days, max_expirations):
    return option_book(ticker, horizon_days=horizon_days, max_expirations=max_expirations)


@st.cache_data(ttl=300, show_spinner=False)
def cached_gex_volatility_proxy(ticker):
    return gex_volatility_proxy(ticker)


@st.cache_data(ttl=900, show_spinner=False)
def cached_vrp_data(ticker, vol_ticker, period):
    return load_vrp_data(ticker, vol_ticker, period)


st.title("Options Gamma Dashboard")
st.caption("Near-term dealer gamma map using public option-chain data. Research only.")

gamma_tab, vrp_tab = st.tabs(["Gamma Map", "Volatility Risk Premium"])

with gamma_tab:
    controls, output = st.columns([1, 3.2], gap="large")

    with controls:
        ticker = st.text_input("Ticker", "SPY").strip().upper()
        horizon_days = st.select_slider(
            "Book horizon",
            options=[21, 30, 45, 60],
            value=45,
            format_func=lambda x: f"{x} days",
        )
        max_expirations = st.select_slider(
            "Expirations to aggregate",
            options=[8, 12, 16, 20],
            value=16,
        )
        strike_width = st.slider("Chart range around spot", 8, 25, 15, format="±%d%%")

        with st.expander("Model settings"):
            rate = st.number_input(
                "Risk-free rate",
                min_value=0.0,
                max_value=0.20,
                value=0.04,
                step=0.005,
                format="%.3f",
            )

    with output:
        if ticker:
            try:
                with st.spinner("Building gamma book..."):
                    calls, puts, spot, expirations, diagnostics = cached_option_book(
                        ticker,
                        horizon_days,
                        max_expirations,
                    )
                    book = prepare_chain(calls, puts)

                    proxy_vol = None
                    proxy_label = None
                    try:
                        proxy_vol, proxy_label = cached_gex_volatility_proxy(ticker)
                    except Exception:
                        pass

                    exposed = add_gamma_exposure(
                        book,
                        spot,
                        rate=rate,
                        fallback_volatility=proxy_vol,
                        fallback_label=proxy_label or "volatility proxy",
                    )
                    profile = gex_by_strike(exposed)
                    call_wall, put_wall = find_walls(exposed)
                    gamma_flip = zero_gamma_level(
                        book,
                        spot,
                        rate=rate,
                        fallback_volatility=proxy_vol,
                        fallback_label=proxy_label or "volatility proxy",
                    )
                    net_gex = float(exposed["gex"].sum())

                regime = "Positive Γ" if net_gex >= 0 else "Negative Γ"

                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Spot", f"${spot:,.2f}")
                m2.metric("Regime", regime, compact_dollars(net_gex))
                m3.metric(
                    "Gamma flip",
                    f"${gamma_flip:,.2f}" if np.isfinite(gamma_flip) else "Outside ±18%",
                )
                m4.metric("Put wall", f"${put_wall:,.0f}" if np.isfinite(put_wall) else "—")
                m5.metric("Call wall", f"${call_wall:,.0f}" if np.isfinite(call_wall) else "—")

                low = spot * (1 - strike_width / 100)
                high = spot * (1 + strike_width / 100)

                if np.isfinite(put_wall) and put_wall >= spot * 0.75:
                    low = min(low, put_wall - 1)
                if np.isfinite(call_wall) and call_wall <= spot * 1.25:
                    high = max(high, call_wall + 1)

                visible = profile[profile["strike"].between(low, high)].copy()
                st.plotly_chart(
                    gex_profile_chart(visible, spot, call_wall, put_wall, gamma_flip),
                    width="stretch",
                    config={"displayModeBar": False},
                )

                session = "regular session" if regular_session_is_open() else "latest available snapshot"
                valid_expiries = sum(
                    int(row.get("positive_oi_contracts", 0) > 0) for row in diagnostics
                )
                yahoo_iv = int(exposed["iv_source"].eq("yahoo").sum())
                recovered_iv = len(exposed) - yahoo_iv
                last_expiry = expirations[-1] if expirations else "n/a"

                st.caption(
                    f"Yahoo Finance {session} • {valid_expiries}/{len(expirations)} expirations with OI "
                    f"• {len(exposed):,} contracts in model • horizon through {last_expiry}"
                    + (f" • {recovered_iv:,} IV values recovered/approximated" if recovered_iv else "")
                )

                with st.expander("Data quality"):
                    st.dataframe(pd.DataFrame(diagnostics), width="stretch", hide_index=True)
                    if proxy_vol is not None:
                        proxy_rows = int(exposed["iv_source"].eq(proxy_label).sum())
                        if proxy_rows:
                            st.write(
                                f"{proxy_rows:,} contracts used {proxy_label} ({proxy_vol:.1%}) because "
                                "Yahoo did not provide a usable contract-level IV or quote."
                            )

                with st.expander("How the levels are calculated"):
                    st.markdown(
                        """
- **GEX:** Black-Scholes gamma × open interest × 100 × spot² × 1%. Calls are positive and puts negative under the standard public dealer-positioning convention.
- **Call / put wall:** the strikes with the largest call-side and put-side gamma concentration across the aggregated near-term book.
- **Gamma flip:** every contract is re-priced across a ±18% spot grid using its own expiration and IV; the closest point where total signed GEX changes sign is interpolated.
- Open interest is not dealer-position data, so these levels are a positioning model rather than observed dealer inventory.
                        """
                    )

            except Exception as exc:
                st.error(f"Could not build the gamma map: {exc}")
                st.info(
                    "Yahoo's free options feed can occasionally return incomplete OI/IV data. "
                    "Try refreshing once; the model does not replace missing open interest with volume."
                )

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
                    cumulative_pnl_chart(results, side="short" if side == "Short gamma" else "long"),
                    width="stretch",
                )
            with c2:
                st.plotly_chart(
                    pnl_histogram(results, side="short" if side == "Short gamma" else "long"),
                    width="stretch",
                )
        except Exception as exc:
            st.error(f"Could not run the historical analysis: {exc}")
