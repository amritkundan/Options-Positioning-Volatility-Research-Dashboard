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


def test_placeholder_iv_is_recovered_instead_of_dropping_open_interest():
    calls = pd.DataFrame({"strike": [100.0], "impliedVolatility": [0.00001], "openInterest": [1500], "bid": [2.20], "ask": [2.40], "lastPrice": [2.30]})
    puts = pd.DataFrame({"strike": [100.0], "impliedVolatility": [0.00001], "openInterest": [1200], "bid": [2.10], "ask": [2.30], "lastPrice": [2.20]})
    chain = prepare_chain(calls, puts)
    exposed = add_gamma_exposure(chain, spot=100, time_to_expiry_years=30 / 365)
    assert len(exposed) == 2
    assert exposed["iv"].gt(0.005).all()
    assert exposed["gex"].abs().gt(0).all()
    assert exposed["iv_source"].str.startswith("inferred_").all()


def test_after_hours_snapshot_still_produces_gex_and_walls():
    calls = pd.DataFrame({"strike": [760.0, 765.0, 770.0], "impliedVolatility": [0.00001, 0.00001, 0.00001], "openInterest": [8000, 12000, 6500], "bid": [8.0, 4.4, 2.0], "ask": [8.4, 4.8, 2.3], "lastPrice": [8.2, 4.6, 2.1]})
    puts = pd.DataFrame({"strike": [760.0, 765.0, 770.0], "impliedVolatility": [0.00001, 0.00001, 0.00001], "openInterest": [9000, 15000, 10000], "bid": [1.8, 3.2, 6.0], "ask": [2.1, 3.6, 6.4], "lastPrice": [2.0, 3.4, 6.2]})
    ny = ZoneInfo("America/New_York")
    t = time_to_expiry("2026-08-28", now=datetime(2026, 8, 27, 3, 0, tzinfo=ny))
    chain = prepare_chain(calls, puts)
    exposed = add_gamma_exposure(chain, spot=766.08, time_to_expiry_years=t)
    from src.gex import find_walls
    call_wall, put_wall = find_walls(exposed)
    assert np.isfinite(exposed["gex"].sum())
    assert exposed["gex"].abs().sum() > 0
    assert np.isfinite(call_wall)
    assert np.isfinite(put_wall)


def test_zero_gamma_ignores_numerical_underflow_at_scan_edge():
    from src.gex import zero_gamma_level
    calls = pd.DataFrame({"strike": [760.0, 765.0, 770.0], "impliedVolatility": [0.16, 0.17, 0.14], "openInterest": [8000, 12000, 6500]})
    puts = pd.DataFrame({"strike": [760.0, 765.0, 770.0], "impliedVolatility": [0.18, 0.16, 0.15], "openInterest": [9000, 15000, 10000]})
    chain = prepare_chain(calls, puts)
    level = zero_gamma_level(chain, 766.08, 0.0015)
    assert np.isnan(level) or 766.08 * 0.8 < level < 766.08 * 1.2


def test_flat_vol_proxy_keeps_oi_chain_renderable_when_yahoo_has_no_iv_or_prices():
    calls = pd.DataFrame({"strike": [760.0, 765.0, 770.0], "impliedVolatility": [0.00001, np.nan, 0.00001], "openInterest": [8000, 12000, 6500], "bid": [0.0, 0.0, 0.0], "ask": [0.0, 0.0, 0.0], "lastPrice": [0.0, 0.0, 0.0]})
    puts = pd.DataFrame({"strike": [760.0, 765.0, 770.0], "impliedVolatility": [np.nan, 0.00001, np.nan], "openInterest": [9000, 15000, 10000], "bid": [0.0, 0.0, 0.0], "ask": [0.0, 0.0, 0.0], "lastPrice": [0.0, 0.0, 0.0]})
    chain = prepare_chain(calls, puts)
    exposed = add_gamma_exposure(chain, spot=766.08, time_to_expiry_years=7 / 365, fallback_volatility=0.18, fallback_label="VIX close proxy")
    assert len(exposed) == 6
    assert exposed["gex"].abs().sum() > 0
    assert exposed["iv_source"].eq("VIX close proxy").all()


def test_multi_expiry_book_uses_contract_specific_time_and_finds_flip():
    from src.gex import find_walls, zero_gamma_level
    calls = pd.DataFrame({"strike": [108.0, 112.0], "impliedVolatility": [0.20, 0.22], "openInterest": [12000, 7000], "expiration": ["2026-09-18", "2026-10-16"]})
    puts = pd.DataFrame({"strike": [88.0, 92.0], "impliedVolatility": [0.22, 0.20], "openInterest": [7000, 12000], "expiration": ["2026-10-16", "2026-09-18"]})
    ny = ZoneInfo("America/New_York")
    now = datetime(2026, 8, 27, 10, 0, tzinfo=ny)
    book = prepare_chain(calls, puts, now=now)
    assert book.groupby("expiration")["time_to_expiry"].first().nunique() == 2
    exposed = add_gamma_exposure(book, spot=100.0)
    call_wall, put_wall = find_walls(exposed)
    flip = zero_gamma_level(book, current_spot=100.0)
    assert call_wall in {108.0, 112.0}
    assert put_wall in {88.0, 92.0}
    assert np.isfinite(flip)
    assert 82.0 < flip < 118.0
