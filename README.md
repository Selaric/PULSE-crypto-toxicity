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

## Status: data collection and baseline pipeline working

The collector, book/trade parsing, feature engineering, toxicity
labeling, threshold sweep, feature/label merging, and baseline model
training have been run against real Coinbase data. The current collection
is still too small for a reliable research conclusion.

## Structure

```
config/config.yaml       — all tunable parameters in one place, with TODOs
src/data/                — WS collector, raw-message storage, book/trade parsing
src/features/            — rolling microstructure features
src/labeling/            — toxicity labeling + threshold sweep + leakage check
src/models/               — static baselines (LogReg/LightGBM) + online PULSE-inspired model
src/evaluation/           — TWAP vs. adaptive-pause execution backtest
scripts/                 — collect_data.py, check_collection_status.py, run_pipeline.py entry points
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
   For a lightweight collection sanity check, run
   `python scripts/check_collection_status.py`; it reports raw file and
   message counts plus the earliest/latest collection timestamps without
   running the pipeline. The local Windows task runs this check every two
   hours.
4. **Review the toxicity threshold.** `config.yaml`'s
   `horizon_events=300, threshold_bps=2.0` are carried over from the
   AAPL/INTC project. The current real-data run produced a 19.5% toxic
   rate at those settings, which is inside the target 10–40% band, but
   more data is needed before treating that as validated.
5. **Extend `depth_imbalance`** — currently a placeholder (NaN) in
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

## Historical real-data baseline

An earlier collected session loaded 4,147 book events and 1,566 trades.
The pipeline labeled 1,472 events and reported a 19.5% toxic rate. This
is a preliminary baseline, not a validated result.

## Collection and pipeline report — 2026-09-29

The unattended collector was restarted with a clean raw-data directory.
The latest status snapshot reported:

- collector stopped after the overnight collection run
- full pipeline completed at 02:41 local time
- 676,307 book events and 181,577 trades loaded

The configured label settings (`horizon_events=300`, `threshold_bps=2.0`)
labeled 181,424 events and produced a 13.5% toxic rate, inside the target
10–40% band. The threshold sweep showed material sensitivity: 26.4% at
1,000 events and 2 bps, versus 5.0% at 100 events and 2 bps.

The validation split contained 27,214 events. LogReg reached ROC AUC
0.518, PR AUC 0.106, and Brier score 0.237; LightGBM reached ROC AUC
0.514, PR AUC 0.107, and Brier score 0.096. The larger sample improves
confidence in the label-rate estimate, but both AUC values remain close to
random and do not yet demonstrate useful predictive power.

These counts are a collection-health check only; they do not measure data
quality or model performance. The lightweight status check is scheduled
locally every two hours and writes to `data/collection_status.log`. A
separate local task runs the full pipeline after 24 hours, then once per
day, writing output to `data/pipeline_runs.log`. The collector is currently
stopped after this completed run.

### Measures to improve confidence

1. Compare calm and active market periods separately rather than pooling
   all regimes together.
2. Re-sweep the equity-derived horizon and threshold values for BTC-USD.
3. Implement full-depth `depth_imbalance` before treating model gains as
   robust.
4. Investigate why AUC remains near random despite the larger sample,
   including feature quality, label horizon, and chronological drift.
