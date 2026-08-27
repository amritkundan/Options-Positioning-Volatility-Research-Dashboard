# Options Positioning & Volatility Research Dashboard

A small options-research project built around two questions:

1. Where is current option gamma concentrated by strike?
2. How has an implied-volatility proxy compared with subsequent realized volatility?

The app combines a live gamma-exposure (GEX) view with a historical volatility-risk-premium study. The goal is not to recreate an execution system from free data; it is to make the assumptions explicit, keep the analytics testable, and provide a usable research interface.

## What is in the dashboard

### Live GEX
- Aggregates several near-term expirations into one option book instead of relying on a single weekly chain.
- Computes Black-Scholes gamma contract by contract using each option's own expiration and IV.
- Weights gamma by open interest and the standard 100-share contract multiplier.
- Shows call-side and put-side GEX by strike, plus net gamma regime.
- Estimates call wall, put wall, and the structural gamma flip by re-pricing the full book across a ±18% spot grid.

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

The core tests do not require a live Yahoo request. They check Black-Scholes calculations, forward-realized volatility, P&L sign behavior, multi-expiry GEX aggregation, walls, and structural gamma-flip logic.

## Methodology notes

The project intentionally separates **current-chain positioning** from **historical volatility research**.

For GEX, calls are assigned positive exposure and puts negative exposure. That is a useful market-positioning convention, but open interest alone cannot identify the dealer side of each contract. The output should not be interpreted as a literal dealer inventory report.

The live GEX view also works outside regular market hours. Open interest is a standing-position measure, so the app uses Yahoo's latest available snapshot rather than requiring current-session volume. To reduce the fragility of free option-chain data, the dashboard builds one near-term book from several expirations. Missing IV is recovered from option prices when possible, interpolated only within the same expiry/option side, and finally approximated with a labeled volatility proxy if necessary. Missing open interest is never replaced with volume.

For the historical module, VIX is used as an implied-volatility proxy. VIX is a 30-day SPX variance measure, not the exact ATM implied volatility of SPY. Because Yahoo Finance does not provide a clean point-in-time history of full option chains, the module is labeled as a **variance-spread / gamma-P&L proxy**, not an actual historical option-price backtest.
