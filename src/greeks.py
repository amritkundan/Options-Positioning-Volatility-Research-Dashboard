import numpy as np
from scipy.stats import norm


def _validate_inputs(spot, strike, time_to_expiry, volatility):
    if spot <= 0 or strike <= 0:
        raise ValueError("spot and strike must be positive")
    if time_to_expiry < 0:
        raise ValueError("time_to_expiry cannot be negative")
    if volatility < 0:
        raise ValueError("volatility cannot be negative")


def bs_d1(spot, strike, time_to_expiry, rate, volatility, dividend_yield=0.0):
    _validate_inputs(spot, strike, time_to_expiry, volatility)
    if time_to_expiry == 0 or volatility == 0:
        return np.nan

    numerator = (
        np.log(spot / strike)
        + (rate - dividend_yield + 0.5 * volatility**2) * time_to_expiry
    )
    denominator = volatility * np.sqrt(time_to_expiry)
    return numerator / denominator


def bs_gamma(spot, strike, time_to_expiry, rate, volatility, dividend_yield=0.0):
    """Black-Scholes gamma per $1 move in the underlying."""
    _validate_inputs(spot, strike, time_to_expiry, volatility)
    if time_to_expiry == 0 or volatility == 0:
        return 0.0

    d1 = bs_d1(
        spot,
        strike,
        time_to_expiry,
        rate,
        volatility,
        dividend_yield,
    )
    discount = np.exp(-dividend_yield * time_to_expiry)
    return discount * norm.pdf(d1) / (
        spot * volatility * np.sqrt(time_to_expiry)
    )
