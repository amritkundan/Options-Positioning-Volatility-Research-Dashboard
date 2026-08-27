import numpy as np
import pandas as pd
import yfinance as yf

MIN_VALID_IV = 0.005
MAX_VALID_IV = 5.0


def _close_series(frame, ticker):
    if frame.empty:
        raise RuntimeError(f"no price data returned for {ticker}")

    close = frame["Close"]
    if isinstance(close, pd.DataFrame):
        if ticker in close.columns:
            close = close[ticker]
        else:
            close = close.iloc[:, 0]

    return pd.to_numeric(close, errors="coerce").dropna().rename(ticker)


def download_close(ticker, period="10y"):
    frame = yf.download(
        ticker,
        period=period,
        auto_adjust=True,
        progress=False,
        multi_level_index=False,
    )
    return _close_series(frame, ticker)


def load_vrp_data(ticker="SPY", vol_ticker="^VIX", period="10y"):
    spot = download_close(ticker, period)
    vol_index = download_close(vol_ticker, period)

    data = pd.concat([spot, vol_index], axis=1, join="inner").dropna()
    data.columns = ["spot", "vol_index"]
    data["implied_vol_proxy"] = data["vol_index"] / 100.0
    return data


def option_expirations(ticker):
    expirations = list(yf.Ticker(ticker).options)
    if not expirations:
        raise RuntimeError(f"no listed option expirations returned for {ticker}")
    return expirations


def option_chain(ticker, expiration):
    instrument = yf.Ticker(ticker)
    chain = instrument.option_chain(expiration)

    history = instrument.history(period="5d", auto_adjust=True)
    if history.empty:
        raise RuntimeError(f"no spot data returned for {ticker}")

    spot = float(history["Close"].dropna().iloc[-1])
    return chain.calls.copy(), chain.puts.copy(), spot


def _positive_oi_mask(frame):
    if "openInterest" not in frame.columns:
        return pd.Series(False, index=frame.index)
    return pd.to_numeric(frame["openInterest"], errors="coerce").gt(0)


def positive_open_interest_count(calls, puts):
    """Count contracts whose reported open interest is strictly positive."""
    return int(_positive_oi_mask(calls).sum() + _positive_oi_mask(puts).sum())


def usable_iv_count(calls, puts):
    """Count positive-OI contracts with a plausible Yahoo IV value."""
    total = 0
    for frame in (calls, puts):
        if "impliedVolatility" not in frame.columns:
            continue
        iv = pd.to_numeric(frame["impliedVolatility"], errors="coerce")
        valid_iv = iv.between(MIN_VALID_IV, MAX_VALID_IV, inclusive="neither")
        total += int((_positive_oi_mask(frame) & valid_iv).sum())
    return total


def recoverable_price_count(calls, puts):
    """Count positive-OI contracts with a quote that can potentially imply IV."""
    total = 0
    for frame in (calls, puts):
        oi = _positive_oi_mask(frame)
        bid = pd.to_numeric(frame.get("bid"), errors="coerce") if "bid" in frame else pd.Series(np.nan, index=frame.index)
        ask = pd.to_numeric(frame.get("ask"), errors="coerce") if "ask" in frame else pd.Series(np.nan, index=frame.index)
        last = pd.to_numeric(frame.get("lastPrice"), errors="coerce") if "lastPrice" in frame else pd.Series(np.nan, index=frame.index)
        has_mid = bid.gt(0) & ask.gt(0) & ask.ge(bid)
        has_last = last.gt(0)
        total += int((oi & (has_mid | has_last)).sum())
    return total


def option_data_quality(calls, puts):
    """Summarize whether a chain has enough information for a GEX estimate."""
    oi_count = positive_open_interest_count(calls, puts)
    iv_count = usable_iv_count(calls, puts)
    quote_count = recoverable_price_count(calls, puts)
    return {
        "contracts": len(calls) + len(puts),
        "positive_oi_contracts": oi_count,
        "usable_iv_contracts": iv_count,
        "priced_oi_contracts": quote_count,
        "pricing_signal_contracts": iv_count + quote_count,
    }


def _expiration_candidates(preferred_expiration, expirations):
    """Try the selected expiry first, then the nearest later expiries."""
    available = [str(expiration) for expiration in expirations]
    preferred = str(preferred_expiration)

    if preferred not in available:
        return [preferred] + available

    idx = available.index(preferred)
    later = available[idx + 1 :]
    earlier = list(reversed(available[:idx]))
    return [preferred] + later + earlier


def option_chain_with_oi_fallback(
    ticker,
    preferred_expiration,
    expirations,
    max_expirations=8,
):
    """
    Load the nearest option chain that can support a GEX estimate.

    The selected expiration is preferred. If Yahoo returns OI but no usable IV
    or option prices for that expiration, nearby expirations are checked before
    accepting the degraded chain. If every nearby chain has the same Yahoo data
    problem, the first chain with positive OI is returned so the caller can use
    an explicitly labeled volatility proxy instead of fabricating IV data.
    """
    attempts = []
    last_error = None
    oi_backup = None

    candidates = _expiration_candidates(preferred_expiration, expirations)
    for expiration in candidates[:max_expirations]:
        try:
            calls, puts, spot = option_chain(ticker, expiration)
            quality = option_data_quality(calls, puts)
            attempts.append({"expiration": expiration, **quality})

            if quality["positive_oi_contracts"] <= 0:
                continue

            if oi_backup is None:
                oi_backup = (calls, puts, spot, expiration)

            if quality["pricing_signal_contracts"] > 0:
                return calls, puts, spot, expiration, attempts
        except Exception as exc:
            last_error = exc
            attempts.append(
                {
                    "expiration": expiration,
                    "contracts": 0,
                    "positive_oi_contracts": 0,
                    "usable_iv_contracts": 0,
                    "priced_oi_contracts": 0,
                    "pricing_signal_contracts": 0,
                    "error": str(exc),
                }
            )

    if oi_backup is not None:
        calls, puts, spot, expiration = oi_backup
        return calls, puts, spot, expiration, attempts

    attempted = ", ".join(item["expiration"] for item in attempts)
    detail = (
        "Yahoo returned option chains, but none of the checked expirations had "
        "positive open interest. This is a data-source problem, not a market-hours rule."
    )
    if last_error and all("error" in item for item in attempts):
        detail = f"Yahoo option-chain requests failed: {last_error}"

    raise RuntimeError(f"{detail} Checked: {attempted}")


def gex_volatility_proxy(ticker):
    """
    Return a fallback annualized volatility and a human-readable source label.

    For SPY, VIX is used first because it is the closest freely available
    forward-looking volatility proxy. For other symbols (or if VIX is
    unavailable), use 20-day realized volatility from the underlying.
    """
    symbol = ticker.strip().upper()

    if symbol == "SPY":
        try:
            vix = download_close("^VIX", period="5d")
            value = float(vix.iloc[-1]) / 100.0
            if np.isfinite(value) and 0.05 <= value <= 2.0:
                return value, "VIX close proxy"
        except Exception:
            pass

    prices = download_close(symbol, period="3mo")
    returns = np.log(prices / prices.shift(1)).dropna()
    if len(returns) < 10:
        raise RuntimeError("not enough price history to estimate fallback volatility")

    window = min(20, len(returns))
    realized = float(returns.iloc[-window:].std() * np.sqrt(252))
    if not np.isfinite(realized) or realized <= 0:
        raise RuntimeError("could not estimate fallback volatility")

    realized = float(np.clip(realized, 0.05, 3.0))
    return realized, f"{window}-day realized-vol proxy"


def select_gex_expirations(expirations, horizon_days=45, max_expirations=16, now=None):
    """Choose the nearest listed expirations for the aggregate GEX book."""
    if not expirations:
        return []

    today = (pd.Timestamp(now) if now is not None else pd.Timestamp.now()).date()
    parsed = []
    for value in expirations:
        try:
            expiry = pd.Timestamp(value).date()
        except Exception:
            continue
        if expiry >= today:
            parsed.append((expiry, str(value)))

    parsed.sort(key=lambda item: item[0])
    if not parsed:
        return []

    within_horizon = [
        value for expiry, value in parsed if (expiry - today).days <= int(horizon_days)
    ]
    candidates = within_horizon or [value for _, value in parsed]
    return candidates[: int(max_expirations)]


def option_book(ticker, horizon_days=45, max_expirations=16):
    """
    Load a near-term option book across several expirations.

    GEX walls and the structural gamma flip are book-level quantities. Using a
    single weekly expiry makes the result fragile when Yahoo has sparse OI for
    that date, so the dashboard aggregates several nearby expirations instead.
    """
    symbol = ticker.strip().upper()
    instrument = yf.Ticker(symbol)
    expirations = list(instrument.options)
    selected = select_gex_expirations(
        expirations,
        horizon_days=horizon_days,
        max_expirations=max_expirations,
    )
    if not selected:
        raise RuntimeError(f"no usable option expirations returned for {symbol}")

    history = instrument.history(period="5d", auto_adjust=True)
    if history.empty:
        raise RuntimeError(f"no spot data returned for {symbol}")
    spot = float(pd.to_numeric(history["Close"], errors="coerce").dropna().iloc[-1])

    call_frames = []
    put_frames = []
    diagnostics = []

    for expiration in selected:
        try:
            chain = instrument.option_chain(expiration)
            calls = chain.calls.copy()
            puts = chain.puts.copy()
            quality = option_data_quality(calls, puts)
            diagnostics.append({"expiration": expiration, **quality})

            calls["expiration"] = expiration
            puts["expiration"] = expiration
            call_frames.append(calls)
            put_frames.append(puts)
        except Exception as exc:
            diagnostics.append(
                {
                    "expiration": expiration,
                    "contracts": 0,
                    "positive_oi_contracts": 0,
                    "usable_iv_contracts": 0,
                    "priced_oi_contracts": 0,
                    "pricing_signal_contracts": 0,
                    "error": str(exc),
                }
            )

    if not call_frames and not put_frames:
        raise RuntimeError("Yahoo did not return any option chains for the selected horizon")

    calls = pd.concat(call_frames, ignore_index=True) if call_frames else pd.DataFrame()
    puts = pd.concat(put_frames, ignore_index=True) if put_frames else pd.DataFrame()

    if positive_open_interest_count(calls, puts) <= 0:
        raise RuntimeError(
            "Yahoo returned option chains but no positive open interest across the near-term book"
        )

    return calls, puts, spot, selected, diagnostics
