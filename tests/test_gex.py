from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from src.gex import add_gamma_exposure, gex_by_strike, prepare_chain, time_to_expiry


def sample_chain():
    calls = pd.DataFrame(
        {
            "strike": [95, 100, 105],
            "impliedVolatility": [0.22, 0.20, 0.21],
            "openInterest": [1000, 1500, 900],
        }
    )
    puts = pd.DataFrame(
        {
            "strike": [95, 100, 105],
            "impliedVolatility": [0.23, 0.21, 0.22],
            "openInterest": [800, 1200, 1100],
        }
    )
    return calls, puts


def test_call_and_put_exposures_use_opposite_signs():
    calls, puts = sample_chain()
    chain = prepare_chain(calls, puts)
    exposed = add_gamma_exposure(chain, spot=100, time_to_expiry_years=30 / 365)

    assert (exposed.loc[exposed["option_type"] == "call", "gex"] > 0).all()
    assert (exposed.loc[exposed["option_type"] == "put", "gex"] < 0).all()


def test_gex_groups_to_one_row_per_strike():
    calls, puts = sample_chain()
    chain = prepare_chain(calls, puts)
    exposed = add_gamma_exposure(chain, spot=100, time_to_expiry_years=30 / 365)
    profile = gex_by_strike(exposed)

    assert list(profile["strike"]) == [95, 100, 105]


def test_same_day_expiry_has_positive_time_before_close():
    ny = ZoneInfo("America/New_York")
    now = datetime(2026, 8, 27, 10, 0, tzinfo=ny)
    t = time_to_expiry("2026-08-27", now=now)
    assert t > 0
    assert t < 1 / 365
