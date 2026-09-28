"""
Toxicity labeling, ported from the equity (AAPL/INTC) pipeline with
the same event-based, leakage-safe design: horizon in NUMBER OF BOOK
EVENTS ahead (not wall-clock seconds), relative bps threshold (not a
fixed dollar amount, since crypto prices span orders of magnitude
across assets), and an explicit leakage assertion.

IMPORTANT: the equity project's calibrated values (horizon_events=300,
threshold_bps=2.0) were tuned for AAPL/INTC volatility and event
density. Crypto trades 24/7 with a very different arrival-rate and
volatility profile -- re-run sweep_toxicity_thresholds() on real
collected data before trusting any specific number here. Config
defaults are placeholders, not answers.
"""

import numpy as np
import pandas as pd


def label_toxicity(
    book_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    horizon_events: int = 300,
    threshold_bps: float = 2.0,
) -> pd.DataFrame:
    """For each trade (aggressive event), look horizon_events ahead in
    the book_df timeline and label 1 (toxic) if mid-price moved in the
    aggressor's favor by more than threshold_bps. Trades within
    horizon_events of the end of collected data are dropped, not
    faked -- see tests/test_labeling.py for the boundary behavior."""

    if book_df.empty or trades_df.empty:
        return pd.DataFrame(columns=list(trades_df.columns) + ["label"])

    book_df = book_df.sort_values("time").reset_index(drop=True)
    labels, valid_rows = [], []

    for idx, trade in trades_df.iterrows():
        pos = book_df["time"].searchsorted(trade["time"])
        future_pos = pos + horizon_events
        if future_pos >= len(book_df):
            continue  # can't look forward past the end of collected data

        mid_now = book_df.loc[min(pos, len(book_df) - 1), "mid_price"]
        mid_future = book_df.loc[future_pos, "mid_price"]
        if pd.isna(mid_now) or pd.isna(mid_future):
            continue  # a NaN mid-price (thin/one-sided book moment) can't be labeled honestly

        side = trade["aggressor_side"]
        threshold = mid_now * threshold_bps / 10000.0
        signed_move = side * (mid_future - mid_now)

        labels.append(1 if signed_move > threshold else 0)
        valid_rows.append(idx)

    out = trades_df.loc[valid_rows].copy()
    out["label"] = labels
    return out.reset_index(drop=True)


def check_no_leakage(book_df: pd.DataFrame, labeled_df: pd.DataFrame, horizon_events: int, n_samples: int = 20):
    """Confirms every labeled event's forward window genuinely stayed
    within collected data bounds. Raises AssertionError on failure --
    this should never fail if label_toxicity is used correctly, so a
    failure here means someone modified the indexing logic upstream."""
    if labeled_df.empty:
        return  # nothing to check -- not a leakage failure, just no data yet
    sample = labeled_df.sample(min(n_samples, len(labeled_df)))
    book_sorted = book_df.sort_values("time").reset_index(drop=True)
    for _, row in sample.iterrows():
        pos = book_sorted["time"].searchsorted(row["time"])
        assert pos + horizon_events < len(book_sorted), (
            f"Leakage check failed at time={row['time']}: forward window exceeds collected data."
        )


def sweep_toxicity_thresholds(
    book_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    horizons: list[int] = (100, 300, 500, 1000),
    bps_values: list[float] = (1.0, 2.0, 5.0, 10.0),
) -> pd.DataFrame:
    """Grid-search helper -- mirrors the equity project's sweep. Run
    this FIRST on real crypto data before trusting config.yaml's
    horizon_events/threshold_bps defaults. Returns toxic rate + sample
    count for every combination so you can pick a cell landing in a
    sane 10-40% band (see config.yaml min_toxic_rate/max_toxic_rate)."""
    rows = []
    for h in horizons:
        for b in bps_values:
            lbl = label_toxicity(book_df, trades_df, horizon_events=h, threshold_bps=b)
            rows.append({
                "horizon_events": h,
                "threshold_bps": b,
                "toxic_rate": lbl["label"].mean() if not lbl.empty else np.nan,
                "n_labeled": len(lbl),
            })
    return pd.DataFrame(rows)
