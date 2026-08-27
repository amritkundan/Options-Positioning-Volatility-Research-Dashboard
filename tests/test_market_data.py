import sys
import types

import pandas as pd

sys.modules.setdefault("yfinance", types.SimpleNamespace())
from src import market_data


def _chain(open_interest):
    calls = pd.DataFrame(
        {
            "strike": [100.0, 101.0],
            "openInterest": open_interest,
            "impliedVolatility": [0.20, 0.21],
        }
    )
    puts = pd.DataFrame(
        {
            "strike": [100.0, 99.0],
            "openInterest": open_interest,
            "impliedVolatility": [0.22, 0.23],
        }
    )
    return calls, puts, 100.0


def test_positive_open_interest_count_ignores_missing_and_zero_values():
    calls = pd.DataFrame({"openInterest": [100, 0, None, "25"]})
    puts = pd.DataFrame({"openInterest": [0, 50, None, "bad"]})
    assert market_data.positive_open_interest_count(calls, puts) == 3


def test_option_chain_falls_forward_when_selected_expiry_has_no_oi(monkeypatch):
    chains = {
        "2026-08-27": _chain([0, None]),
        "2026-08-28": _chain([1200, 900]),
    }
    monkeypatch.setattr(market_data, "option_chain", lambda ticker, expiration: chains[expiration])
    calls, puts, spot, used, attempts = market_data.option_chain_with_oi_fallback(
        "SPY", "2026-08-27", ["2026-08-27", "2026-08-28", "2026-08-31"]
    )
    assert used == "2026-08-28"
    assert spot == 100.0
    assert market_data.positive_open_interest_count(calls, puts) == 4
    assert attempts[0]["positive_oi_contracts"] == 0
    assert attempts[1]["positive_oi_contracts"] == 4


def test_option_chain_keeps_selected_expiry_when_oi_is_available(monkeypatch):
    monkeypatch.setattr(market_data, "option_chain", lambda ticker, expiration: _chain([1500, 700]))
    _, _, _, used, attempts = market_data.option_chain_with_oi_fallback(
        "SPY", "2026-08-27", ["2026-08-27", "2026-08-28"]
    )
    assert used == "2026-08-27"
    assert len(attempts) == 1


def test_select_gex_expirations_uses_near_term_horizon():
    expirations = ["2026-08-28", "2026-09-04", "2026-09-18", "2026-10-16", "2026-12-18"]
    selected = market_data.select_gex_expirations(
        expirations, horizon_days=45, max_expirations=10, now="2026-08-27"
    )
    assert selected == ["2026-08-28", "2026-09-04", "2026-09-18"]
