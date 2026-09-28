"""Edge cases for microstructure feature engineering: empty input,
zero depth, cold-start windows, zero/duplicate timestamps."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.features.microstructure_features import add_microstructure_features, _safe_div


def test_safe_div_zero_over_zero_is_zero_not_nan():
    result = _safe_div(np.array([0.0]), np.array([0.0]))
    assert result[0] == 0.0


def test_safe_div_normal_case():
    result = _safe_div(np.array([10.0]), np.array([4.0]))
    assert result[0] == 2.5


def test_empty_book_df_returns_empty_with_expected_columns():
    empty_book = pd.DataFrame(columns=["time", "best_bid", "best_ask", "mid_price", "spread"])
    empty_trades = pd.DataFrame(columns=["time", "price", "size", "aggressor_side"])
    out = add_microstructure_features(empty_book, empty_trades, window_size=50)
    assert out.empty
    assert "micro_price" in out.columns


def test_no_trades_yields_zero_flow_imbalance_not_crash():
    book = pd.DataFrame({
        "time": [1.0, 2.0, 3.0],
        "mid_price": [100.0, 100.5, 101.0],
        "spread": [0.1, 0.1, 0.1],
    })
    empty_trades = pd.DataFrame(columns=["time", "price", "size", "aggressor_side"])
    out = add_microstructure_features(book, empty_trades, window_size=2)
    assert (out["order_flow_imbalance"] == 0.0).all()


def test_single_row_cold_start_does_not_crash():
    book = pd.DataFrame({"time": [1.0], "mid_price": [100.0], "spread": [0.1]})
    trades = pd.DataFrame({"time": [1.0], "price": [100.0], "size": [1.0], "aggressor_side": [1]})
    out = add_microstructure_features(book, trades, window_size=50)
    assert len(out) == 1
    assert pd.isna(out["short_horizon_volatility"].iloc[0])  # min_periods=2 -- correctly undefined, not zero


def test_duplicate_timestamps_do_not_divide_by_zero():
    # simulates a burst of WS messages arriving with identical collection timestamps
    book = pd.DataFrame({
        "time": [1.0, 1.0, 1.0, 1.0],
        "mid_price": [100.0, 100.1, 100.0, 99.9],
        "spread": [0.1, 0.1, 0.1, 0.1],
    })
    trades = pd.DataFrame(columns=["time", "price", "size", "aggressor_side"])
    out = add_microstructure_features(book, trades, window_size=3)
    assert not out["arrival_rate"].isin([np.inf, -np.inf]).any()


def test_all_buys_gives_flow_imbalance_of_one():
    book = pd.DataFrame({"time": [1.0, 2.0], "mid_price": [100.0, 100.0], "spread": [0.1, 0.1]})
    trades = pd.DataFrame({
        "time": [0.5, 1.5], "price": [100.0, 100.0], "size": [1.0, 1.0], "aggressor_side": [1, 1],
    })
    out = add_microstructure_features(book, trades, window_size=10)
    assert out["order_flow_imbalance"].iloc[-1] == pytest.approx(1.0)
