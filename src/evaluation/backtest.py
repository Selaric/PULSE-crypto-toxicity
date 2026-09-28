"""
Execution simulation: naive TWAP baseline vs. a toxicity-adaptive
agent that pauses child slices when predicted toxicity is high.

Two deliberate fixes baked in from the AAPL/INTC project's post-mortem:

1. IMPLEMENTATION SHORTFALL SIGN CONVENTION is side-aware. A sell that
   fills ABOVE arrival price is a GAIN, not a cost -- the earlier
   project's formula silently assumed a buy order's sign convention
   even when simulating a sell. `side` is now a required, explicit
   parameter, not an assumption baked into the arithmetic.

2. THE PAUSE THRESHOLD IS A ROLLING QUANTILE, not a hardcoded
   probability. The earlier project hardcoded threshold=0.24 while the
   model's median output hovered right at 0.237, causing a 100% pause
   rate -- meaning the "adaptive" agent was actually just "always
   pause," and the backtest was silently not testing the model at
   all. A rolling quantile threshold cannot degenerate to 0%/100%
   pause rates the same way a fixed constant can.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ExecutionResult:
    side: str
    arrival_price: float
    avg_execution_price: float
    total_shares: float
    total_value: float
    implementation_shortfall_dollars: float
    implementation_shortfall_bps: float
    n_paused_slices: int


def _signed_shortfall(side: str, arrival_price: float, avg_execution_price: float, shares: float) -> float:
    """Positive return = cost (bad). Negative = gain (good). This is the
    piece that was backwards in the prior project for sell orders."""
    if side == "buy":
        return (avg_execution_price - arrival_price) * shares
    elif side == "sell":
        return (arrival_price - avg_execution_price) * shares
    else:
        raise ValueError(f"side must be 'buy' or 'sell', got {side!r}")


def run_twap(mid_prices: np.ndarray, side: str, total_shares: float, n_slices: int) -> ExecutionResult:
    """Naive baseline: split shares evenly across n_slices, evenly
    spaced through the available mid_prices series."""
    if len(mid_prices) < n_slices:
        raise ValueError(f"Need at least {n_slices} price points, got {len(mid_prices)}")

    slice_indices = np.linspace(0, len(mid_prices) - 1, n_slices, dtype=int)
    fill_prices = mid_prices[slice_indices]
    shares_per_slice = total_shares / n_slices

    arrival_price = mid_prices[0]
    avg_execution_price = float(np.mean(fill_prices))
    total_value = float(np.sum(fill_prices) * shares_per_slice)

    shortfall = _signed_shortfall(side, arrival_price, avg_execution_price, total_shares)
    shortfall_bps = (shortfall / (arrival_price * total_shares)) * 10000

    return ExecutionResult(
        side=side, arrival_price=arrival_price, avg_execution_price=avg_execution_price,
        total_shares=total_shares, total_value=total_value,
        implementation_shortfall_dollars=shortfall, implementation_shortfall_bps=shortfall_bps,
        n_paused_slices=0,
    )


def run_adaptive(
    mid_prices: np.ndarray,
    predicted_probs: np.ndarray,
    side: str,
    total_shares: float,
    n_slices: int,
    pause_quantile: float = 0.90,
    pause_lookback_predictions: int = 200,
) -> ExecutionResult:
    """Same TWAP schedule, but each slice is skipped (rolled into the
    next one) if its predicted P(toxic) exceeds the pause_quantile-th
    percentile of recent predictions. Using a ROLLING quantile rather
    than a fixed cutoff means the pause rate is bounded away from 0%
    and 100% by construction -- see tests/test_backtest.py for the
    degenerate-case assertions."""
    if len(mid_prices) < n_slices or len(predicted_probs) < n_slices:
        raise ValueError("Need at least n_slices price points and predictions")

    slice_indices = np.linspace(0, len(mid_prices) - 1, n_slices, dtype=int)
    shares_per_slice = total_shares / n_slices
    arrival_price = mid_prices[0]

    fills = []
    n_paused = 0
    pending_shares = 0.0

    for i, idx in enumerate(slice_indices):
        lookback_start = max(0, idx - pause_lookback_predictions)
        recent_preds = predicted_probs[lookback_start:idx + 1]
        # need enough history for a quantile to mean anything -- with
        # under 10 points, don't pause based on noise
        threshold = np.quantile(recent_preds, pause_quantile) if len(recent_preds) >= 10 else 1.1

        current_shares = shares_per_slice + pending_shares
        if predicted_probs[idx] > threshold and i < n_slices - 1:
            n_paused += 1
            pending_shares = current_shares  # roll into next slice rather than skip the shares entirely
            continue

        fills.append((mid_prices[idx], current_shares))
        pending_shares = 0.0

    if not fills:  # everything got rolled into a final forced fill -- shouldn't happen but don't silently lose shares
        fills = [(mid_prices[slice_indices[-1]], total_shares)]

    fill_prices = np.array([p for p, _ in fills])
    fill_shares = np.array([s for _, s in fills])
    avg_execution_price = float(np.average(fill_prices, weights=fill_shares))
    total_value = float(np.sum(fill_prices * fill_shares))

    shortfall = _signed_shortfall(side, arrival_price, avg_execution_price, total_shares)
    shortfall_bps = (shortfall / (arrival_price * total_shares)) * 10000

    return ExecutionResult(
        side=side, arrival_price=arrival_price, avg_execution_price=avg_execution_price,
        total_shares=total_shares, total_value=total_value,
        implementation_shortfall_dollars=shortfall, implementation_shortfall_bps=shortfall_bps,
        n_paused_slices=n_paused,
    )
