# Options Positioning & Volatility Research Dashboard

A small options-research project built around two questions:

1. Where is current option gamma concentrated by strike?
2. How has an implied-volatility proxy compared with subsequent realized volatility?

The app combines a live gamma-exposure (GEX) view with a historical volatility-risk-premium study. The goal is not to recreate an execution system from free data; it is to make the assumptions explicit, keep the analytics testable, and provide a usable research interface.

## What is in the dashboard

### Live GEX
- Pulls the current option chain for a selected expiration.
- Computes Black-Scholes gamma contract by contract.
- Weights gamma by open interest and contract multiplier.
- Aggregates signed exposure by strike.
- Estimates call wall, put wall, and a zero-gamma level.

### Volatility risk premium
- Downloads the underlying and a volatility-index proxy.
- Measures forward realized volatility over a selectable horizon.
- Compares entry implied volatility with subsequent realized volatility.
- Converts the variance spread into a gamma-weighted P&L proxy.
- Shows cumulative results, distribution, and year-by-year behavior.

## Run locally

Python 3.10+ is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

On Windows PowerShell, activate with:

```powershell
.venv\Scripts\Activate.ps1
```

## Run the tests

```bash
python -m pytest -q
```

The core tests do not require a live Yahoo request. They check the Black-Scholes gamma implementation, forward-realized-vol calculation, P&L sign behavior, and GEX aggregation.

## Methodology notes

The project intentionally separates **current-chain positioning** from **historical volatility research**.

For GEX, calls are assigned positive exposure and puts negative exposure. That is a useful market-positioning convention, but open interest alone cannot identify the dealer side of each contract. The output should not be interpreted as a literal dealer inventory report.

For the historical module, VIX is used as an implied-volatility proxy. VIX is a 30-day SPX variance measure, not the exact ATM implied volatility of SPY. Because Yahoo Finance does not provide a clean point-in-time history of full option chains, the module is labeled as a **variance-spread / gamma-P&L proxy**, not an actual historical option-price backtest.


