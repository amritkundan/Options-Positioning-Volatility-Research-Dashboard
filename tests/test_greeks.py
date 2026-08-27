import math

import pytest

from src.greeks import bs_gamma


def test_atm_gamma_matches_reference_value():
    gamma = bs_gamma(100, 100, 1.0, 0.05, 0.20)
    assert gamma == pytest.approx(0.018762, rel=1e-4)


def test_gamma_is_same_for_call_and_put_by_definition():
    gamma = bs_gamma(100, 110, 0.5, 0.04, 0.25)
    assert math.isfinite(gamma)
    assert gamma > 0


def test_expired_option_has_zero_gamma():
    assert bs_gamma(100, 100, 0.0, 0.04, 0.20) == 0.0


def test_invalid_spot_raises():
    with pytest.raises(ValueError):
        bs_gamma(0, 100, 1.0, 0.04, 0.20)
