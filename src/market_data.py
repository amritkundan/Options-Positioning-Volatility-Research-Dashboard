import pandas as pd
import yfinance as yf


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
