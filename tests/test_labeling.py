"""Edge cases for toxicity labeling: empty input, events too close to
the end of data, NaN mid-prices, the leakage assertion itself, and
threshold boundary conditions."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.labeling.toxicity_label import label_toxicity, check_no_leakage, sweep_toxicity_thresholds


def _make_book(n=1000, start_price=100.0, drift=0.0):
    prices = start_price + np.cumsum(np.full(n, drift))
    return pd.DataFrame({"time": np.arange(n, dtype=float), "mid_price": prices, "spread": 0.1})


def test_empty_inputs_return_empty_with_label_column():
    empty_book = pd.DataFrame(columns=["time", "mid_price"])
    empty_trades = pd.DataFrame(columns=["time", "price", "size", "aggressor_side"])
    out = label_toxicity(empty_book, empty_trades)
    assert "label" in out.columns
    assert out.empty


def test_events_too_close_to_end_are_dropped_not_faked():
    book = _make_book(n=100)
    # trade at time=99 with horizon_events=300 can never have a valid future window
    trades = pd.DataFrame({"time": [99.0], "price": [100.0], "size": [1.0], "aggressor_side": [1]})
    out = label_toxicity(book, trades, horizon_events=300, threshold_bps=2.0)
    assert out.empty  # dropped, not labeled with a fabricated value


def test_buy_aggressor_with_upward_move_is_toxic():
    book = _make_book(n=1000, start_price=100.0, drift=0.01)  # price steadily rises
    trades = pd.DataFrame({"time": [10.0], "price": [100.1], "size": [1.0], "aggressor_side": [1]})
    out = label_toxicity(book, trades, horizon_events=50, threshold_bps=1.0)
    assert len(out) == 1
    assert out["label"].iloc[0] == 1


def test_buy_aggressor_with_downward_move_is_not_toxic():
    book = _make_book(n=1000, start_price=100.0, drift=-0.01)  # price steadily falls
    trades = pd.DataFrame({"time": [10.0], "price": [99.9], "size": [1.0], "aggressor_side": [1]})
    out = label_toxicity(book, trades, horizon_events=50, threshold_bps=1.0)
    assert out["label"].iloc[0] == 0


def test_sell_aggressor_sign_is_flipped_correctly():
    # price falls -- toxic for a SELL aggressor (they were right to sell), not a buy
    book = _make_book(n=1000, start_price=100.0, drift=-0.01)
    trades = pd.DataFrame({"time": [10.0], "price": [99.9], "size": [1.0], "aggressor_side": [-1]})
    out = label_toxicity(book, trades, horizon_events=50, threshold_bps=1.0)
    assert out["label"].iloc[0] == 1


def test_nan_mid_price_event_is_dropped():
    book = _make_book(n=1000)
    book.loc[10, "mid_price"] = np.nan
    trades = pd.DataFrame({"time": [10.0], "price": [100.0], "size": [1.0], "aggressor_side": [1]})
    out = label_toxicity(book, trades, horizon_events=50, threshold_bps=1.0)
    assert out.empty  # NaN mid-price at the event itself -- can't honestly label it


def test_leakage_check_passes_on_valid_labels():
    book = _make_book(n=1000)
    trades = pd.DataFrame({
        "time": np.arange(10, 500, 10, dtype=float),
        "price": 100.0, "size": 1.0, "aggressor_side": 1,
    })
    labeled = label_toxicity(book, trades, horizon_events=100, threshold_bps=2.0)
    check_no_leakage(book, labeled, horizon_events=100)  # should not raise


def test_leakage_check_empty_labeled_df_does_not_raise():
    book = _make_book(n=1000)
    empty_labeled = pd.DataFrame(columns=["time", "label"])
    check_no_leakage(book, empty_labeled, horizon_events=100)  # no data -- not a leakage failure


def test_sweep_returns_all_combinations():
    book = _make_book(n=2000)
    trades = pd.DataFrame({
        "time": np.arange(10, 1500, 5, dtype=float),
        "price": 100.0, "size": 1.0, "aggressor_side": np.random.choice([1, -1], 298),
    })
    sweep = sweep_toxicity_thresholds(book, trades, horizons=[50, 100], bps_values=[1.0, 5.0])
    assert len(sweep) == 4  # 2 horizons x 2 bps values
    assert set(sweep.columns) == {"horizon_events", "threshold_bps", "toxic_rate", "n_labeled"}
