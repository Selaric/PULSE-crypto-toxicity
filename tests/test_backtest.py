"""Edge cases for execution backtesting -- specifically the two bugs
from the prior equity project's post-mortem: IS sign convention for
sells, and the pause threshold degenerating to 0%/100%."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.evaluation.backtest import run_twap, run_adaptive


def test_sell_filling_above_arrival_is_a_gain_not_a_cost():
    # price rises steadily -- a SELL executed through this should show
    # a NEGATIVE (good) shortfall, not positive, since selling into a
    # rising market beats the arrival-price benchmark
    prices = np.linspace(100.0, 105.0, 1000)
    result = run_twap(prices, side="sell", total_shares=1000, n_slices=30)
    assert result.implementation_shortfall_dollars < 0


def test_buy_filling_above_arrival_is_a_cost():
    prices = np.linspace(100.0, 105.0, 1000)
    result = run_twap(prices, side="buy", total_shares=1000, n_slices=30)
    assert result.implementation_shortfall_dollars > 0


def test_invalid_side_raises_explicitly():
    prices = np.linspace(100.0, 105.0, 100)
    with pytest.raises(ValueError):
        run_twap(prices, side="hold", total_shares=1000, n_slices=10)


def test_twap_insufficient_price_points_raises():
    prices = np.array([100.0, 101.0])
    with pytest.raises(ValueError):
        run_twap(prices, side="sell", total_shares=1000, n_slices=30)


def test_adaptive_pause_rate_is_not_degenerate_when_probs_vary():
    # this is the direct regression test for the 100%-pause bug:
    # predictions genuinely vary, so the rolling quantile threshold
    # must produce SOME fills and SOME pauses, not all-or-nothing
    rng = np.random.default_rng(0)
    n = 2000
    prices = 100.0 + np.cumsum(rng.normal(0, 0.01, n))
    preds = rng.uniform(0, 1, n)  # genuinely varying predictions

    result = run_adaptive(
        prices, preds, side="sell", total_shares=1000, n_slices=30,
        pause_quantile=0.90, pause_lookback_predictions=200,
    )
    assert 0 < result.n_paused_slices < 30  # neither "never pauses" nor "always pauses"


def test_adaptive_constant_low_predictions_never_pause():
    # if the model is confident nothing is toxic, the agent should
    # basically never pause -- sanity check on the quantile logic itself
    n = 2000
    prices = np.linspace(100.0, 100.0, n)
    preds = np.full(n, 0.01)
    result = run_adaptive(
        prices, preds, side="sell", total_shares=1000, n_slices=30,
        pause_quantile=0.90, pause_lookback_predictions=200,
    )
    assert result.n_paused_slices == 0


def test_adaptive_total_shares_conserved_even_with_pauses():
    rng = np.random.default_rng(1)
    n = 2000
    prices = 100.0 + np.cumsum(rng.normal(0, 0.01, n))
    preds = rng.uniform(0, 1, n)
    result = run_adaptive(
        prices, preds, side="sell", total_shares=1000, n_slices=30,
        pause_quantile=0.90, pause_lookback_predictions=200,
    )
    # pausing rolls shares into later slices -- none should be silently dropped
    assert result.total_shares == 1000
