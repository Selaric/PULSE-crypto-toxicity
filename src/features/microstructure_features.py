"""
Rolling microstructure features computed on the book_df timeline.
Same feature family as the equity project (order-flow imbalance,
depth, micro-price, spread, aggressiveness, volatility, arrival
rate) -- ported here rather than reinvented, since the *reasoning*
for each feature (see project design notes) doesn't change just
because the venue changed from NASDAQ to Coinbase.

Every function is written to degrade gracefully on edge cases
(empty windows, zero depth, a single row) rather than raising --
see tests/test_features.py for the exact behaviors asserted.
"""

import numpy as np
import pandas as pd


def _safe_div(numerator, denominator):
    """Elementwise safe division: 0/0 -> 0.0, x/0 -> 0.0 (not inf/NaN).
    A zero-imbalance reading during a zero-depth moment is a more
    honest default than propagating NaN through every downstream
    rolling feature."""
    denominator = np.where(denominator == 0, np.nan, denominator)
    result = numerator / denominator
    return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)


def add_microstructure_features(
    book_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    window_size: int = 50,
) -> pd.DataFrame:
    """Returns book_df with feature columns appended. window_size is in
    number of book-update events, matching the equity project's
    event-based (not wall-clock) window convention -- see design notes
    on why event-based windows are preferred over fixed-time windows
    for irregularly-spaced order book data."""

    df = book_df.copy()
    if df.empty:
        # return the empty frame with the expected columns so downstream
        # code doesn't need a separate empty-input branch
        for col in ["micro_price", "spread_feature", "short_horizon_volatility"]:
            df[col] = pd.Series(dtype=float)
        return df

    df["micro_price"] = df["mid_price"]  # TODO: refine with size-weighted micro-price if you add L2 depth columns
    df["spread_feature"] = df["spread"]

    # short-horizon volatility: rolling std of mid-price returns.
    # min_periods=2 so early "cold start" rows get NaN instead of a
    # meaningless single-point std of zero -- see tests/test_features.py
    returns = df["mid_price"].pct_change()
    df["short_horizon_volatility"] = returns.rolling(window_size, min_periods=2).std()

    # arrival rate: events per second over the rolling window. Guard
    # against a zero or negative time span (duplicate timestamps are
    # possible with WS message bursts) rather than dividing by zero.
    time_span = df["time"].rolling(window_size, min_periods=2).apply(
        lambda w: w.iloc[-1] - w.iloc[0], raw=False
    )
    event_count = df["time"].rolling(window_size, min_periods=2).count()
    df["arrival_rate"] = _safe_div(event_count.values, time_span.values)

    # order-flow imbalance from trades, mapped onto the book timeline
    # by nearest-preceding trade window. Rebuilt as its own frame
    # because trades and book updates arrive on different message
    # streams with different timestamps -- merge_asof aligns them
    # without look-ahead (direction='backward' -- see labeling notes
    # on why this must never be 'nearest' or 'forward').
    if not trades_df.empty:
        trades_sorted = trades_df.sort_values("time")
        trades_sorted["signed_size"] = trades_sorted["aggressor_side"] * trades_sorted["size"]
        trades_sorted["buy_size"] = trades_sorted["size"].where(trades_sorted["aggressor_side"] == 1, 0.0)
        trades_sorted["sell_size"] = trades_sorted["size"].where(trades_sorted["aggressor_side"] == -1, 0.0)

        rolled = trades_sorted[["time", "signed_size", "buy_size", "sell_size"]].copy()
        rolled["roll_signed"] = rolled["signed_size"].rolling(window_size, min_periods=1).sum()
        rolled["roll_buy"] = rolled["buy_size"].rolling(window_size, min_periods=1).sum()
        rolled["roll_sell"] = rolled["sell_size"].rolling(window_size, min_periods=1).sum()
        rolled["roll_total"] = rolled["roll_buy"] + rolled["roll_sell"]
        rolled["order_flow_imbalance"] = _safe_div(rolled["roll_signed"].values, rolled["roll_total"].values)
        rolled["aggressiveness_volume_imbalance"] = _safe_div(
            (rolled["roll_buy"] - rolled["roll_sell"]).values, rolled["roll_total"].values
        )

        df = pd.merge_asof(
            df.sort_values("time"),
            rolled[["time", "order_flow_imbalance", "aggressiveness_volume_imbalance"]].sort_values("time"),
            on="time",
            direction="backward",  # never look at a trade that hasn't happened yet
        )
    else:
        df["order_flow_imbalance"] = 0.0
        df["aggressiveness_volume_imbalance"] = 0.0

    # depth_imbalance placeholder -- needs full L2 depth columns
    # (total_ask_depth / total_bid_depth) which build_book_snapshots
    # doesn't currently expose beyond best bid/ask. TODO: extend
    # loader.py's _BookState to also emit summed depth across the top
    # N levels (book_depth_levels in config.yaml), then compute:
    #   depth_imbalance = (bid_depth - ask_depth) / (bid_depth + ask_depth)
    df["total_bid_depth"] = np.nan
    df["total_ask_depth"] = np.nan
    df["depth_imbalance"] = np.nan

    return df
