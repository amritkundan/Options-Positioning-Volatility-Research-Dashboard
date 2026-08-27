import numpy as np
from scipy.optimize import brentq
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


def bs_price(
    spot,
    strike,
    time_to_expiry,
    rate,
    volatility,
    option_type,
    dividend_yield=0.0,
):
    """Black-Scholes European option value."""
    _validate_inputs(spot, strike, time_to_expiry, volatility)
    if option_type not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'")

    if time_to_expiry == 0:
        intrinsic = spot - strike if option_type == "call" else strike - spot
        return max(float(intrinsic), 0.0)

    if volatility == 0:
        discounted_spot = spot * np.exp(-dividend_yield * time_to_expiry)
        discounted_strike = strike * np.exp(-rate * time_to_expiry)
        intrinsic = (
            discounted_spot - discounted_strike
            if option_type == "call"
            else discounted_strike - discounted_spot
        )
        return max(float(intrinsic), 0.0)

    d1 = bs_d1(
        spot,
        strike,
        time_to_expiry,
        rate,
        volatility,
        dividend_yield,
    )
    d2 = d1 - volatility * np.sqrt(time_to_expiry)
    discounted_spot = spot * np.exp(-dividend_yield * time_to_expiry)
    discounted_strike = strike * np.exp(-rate * time_to_expiry)

    if option_type == "call":
        return float(discounted_spot * norm.cdf(d1) - discounted_strike * norm.cdf(d2))
    return float(discounted_strike * norm.cdf(-d2) - discounted_spot * norm.cdf(-d1))


def implied_volatility(
    option_price,
    spot,
    strike,
    time_to_expiry,
    rate,
    option_type,
    dividend_yield=0.0,
    min_vol=1e-4,
    max_vol=5.0,
):
    """Solve Black-Scholes implied volatility, returning NaN when no valid root exists."""
    if not np.isfinite(option_price) or option_price <= 0 or time_to_expiry <= 0:
        return np.nan

    if option_type not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'")

    discounted_spot = spot * np.exp(-dividend_yield * time_to_expiry)
    discounted_strike = strike * np.exp(-rate * time_to_expiry)
    if option_type == "call":
        lower_bound = max(discounted_spot - discounted_strike, 0.0)
        upper_bound = discounted_spot
    else:
        lower_bound = max(discounted_strike - discounted_spot, 0.0)
        upper_bound = discounted_strike

    tolerance = 1e-8
    if option_price < lower_bound - tolerance or option_price > upper_bound + tolerance:
        return np.nan

    def objective(volatility):
        return bs_price(
            spot,
            strike,
            time_to_expiry,
            rate,
            volatility,
            option_type,
            dividend_yield,
        ) - option_price

    low_value = objective(min_vol)
    high_value = objective(max_vol)
    if low_value == 0:
        return float(min_vol)
    if high_value == 0:
        return float(max_vol)
    if low_value * high_value > 0:
        return np.nan

    try:
        return float(brentq(objective, min_vol, max_vol, xtol=1e-8, maxiter=100))
    except (ValueError, RuntimeError):
        return np.nan
