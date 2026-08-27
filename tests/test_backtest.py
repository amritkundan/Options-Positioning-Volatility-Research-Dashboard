import numpy as np
import pandas as pd

from src.backtest import forward_realized_vol, variance_premium_backtest


def test_forward_realized_vol_uses_future_returns():
    prices = pd.Series(
        [100.0, 101.0, 99.0, 102.0, 103.0],
        index=pd.bdate_range("2026-01-01", periods=5),
    )
    result = forward_realized_vol(prices, window=3)

    returns = np.log(prices / prices.shift(1)).iloc[1:4]
    expected = returns.std(ddof=1) * np.sqrt(252)
    assert np.isclose(result.iloc[0], expected)


def test_overpriced_implied_vol_hurts_long_gamma_on_smooth_path():
    dates = pd.bdate_range("2026-01-01", periods=80)
    prices = pd.Series(100 * np.exp(np.linspace(0, 0.02, len(dates))), index=dates)
    implied = pd.Series(0.30, index=dates)

    result = variance_premium_backtest(prices, implied, window=10, stride=10)
    assert not result.empty
    assert result["long_gamma_pnl"].mean() < 0
    assert result["short_gamma_pnl"].mean() > 0


def test_long_and_short_pnl_are_mirrors():
    dates = pd.bdate_range("2026-01-01", periods=80)
    rng = np.random.default_rng(7)
    prices = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(dates)))), index=dates)
    implied = pd.Series(0.20, index=dates)

    result = variance_premium_backtest(prices, implied, window=10, stride=10)
    assert np.allclose(result["long_gamma_pnl"], -result["short_gamma_pnl"])
