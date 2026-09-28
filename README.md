# PULSE-Inspired Toxic Flow Detection — Coinbase Crypto

Adapts the predict-then-decide toxicity framework from **Cartea,
Duran-Martin & Sánchez-Betancourt, "Detecting Toxic Flow" (2023,
arXiv:2312.05827)** to public Coinbase order book / trade data,
using standard ML methods plus a simplified online-updating model,
rather than the paper's full Bayesian PULSE method.

**Scope, stated honestly:** the original paper predicts whether a
specific broker client's trade will be toxic, using private FX
transaction data. This project predicts toxicity of anonymous,
market-wide aggressive flow from public exchange data — a related
but different problem, adapted for what's actually accessible outside
an institutional desk. See `src/models/online_pulse.py` for exactly
what is and isn't implemented from the original method.

This is a sibling project to an earlier LOBSTER/AAPL-INTC equity
version — same pipeline shape (data → features → label → model →
decide → backtest), same design discipline (event-based windows,
no-look-ahead labeling, chronological splits), new venue and data
source.

## Status: data collection and labeling working

The collector, book/trade parsing, feature engineering, toxicity
labeling, and threshold sweep have been run against real Coinbase data.
The project is still not a finished research result: model training and
backtesting wait for the feature/label merge TODO described below.

## Structure

```
config/config.yaml       — all tunable parameters in one place, with TODOs
src/data/                — WS collector, raw-message storage, book/trade parsing
src/features/            — rolling microstructure features
src/labeling/            — toxicity labeling + threshold sweep + leakage check
src/models/               — static baselines (LogReg/LightGBM) + online PULSE-inspired model
src/evaluation/           — TWAP vs. adaptive-pause execution backtest
scripts/                 — collect_data.py, run_pipeline.py entry points
tests/                   — pytest suite, 32 tests, all passing against synthetic data
```

## Running the project

1. **Create and activate the environment.** In PowerShell:

   ```powershell
   py -3 -m venv .venv
   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

2. **Collect data.** Run `python scripts/collect_data.py` and let it
   run for at least several hours. The collector writes rotating JSONL
   files under `data/raw/` and does not require a Coinbase API key for
   the public `level2` and `market_trades` channels. Stop it with
   `Ctrl+C`.
3. **Run the pipeline.** Run `python scripts/run_pipeline.py` after
   collecting data. It loads the raw files, builds features, prints a
   toxicity threshold sweep, and applies labels.
4. **Review the toxicity threshold.** `config.yaml`'s
   `horizon_events=300, threshold_bps=2.0` are carried over from the
   AAPL/INTC project. The current real-data run produced a 19.5% toxic
   rate at those settings, which is inside the target 10–40% band, but
   more data is needed before treating that as validated.
5. **Finish the feature/label merge TODO** in `run_pipeline.py` — left
   explicit because the right join strategy depends on real data
   volume/shape, which doesn't exist yet in this skeleton.
6. **Extend `depth_imbalance`** — currently a placeholder (NaN) in
   `microstructure_features.py`, because `loader.py`'s book
   reconstruction only tracks best bid/ask, not full depth. TODO
   comment marks exactly where to extend it.

## What's real vs. what's a stand-in

- **Real, working, tested:** WS collector with reconnect/backoff, raw
  message storage, book reconstruction, feature engineering, labeling
  with leakage checks, static ML baselines, TWAP + adaptive backtest
  with both known bugs (IS sign, degenerate pause threshold) fixed and
  regression-tested.
- **Explicit stand-in, not the real thing:** `OnlineToxicityModel` in
  `online_pulse.py` is a simple online logistic regression, not
  PULSE's Bayesian last-layer update. Docstring in that file says
  exactly what a real implementation would need to replace.

## Running the tests

```
pip install -r requirements.txt
python -m pytest tests/ -v
```

32 tests, all passing against synthetic data (no live connection
needed to verify the logic itself).

## Current real-data result

One collected session loaded 4,147 book events and 1,566 trades. The
pipeline labeled 1,472 events and reported a 19.5% toxic rate. It then
stopped intentionally because the feature/label merge is not implemented
yet; baseline model training and the TWAP versus adaptive backtest have
not run on real data.
