"""
Turns raw Coinbase WS messages into the same shape the equity-LOBSTER
version of this project used: a per-event DataFrame with best bid/ask,
mid_price, spread, and a separate "aggressive events" frame (trades)
with a signed aggressor_side. Keeping this shape consistent is what
lets features.py and labeling.py be reused almost unchanged from the
AAPL/INTC pipeline.

Coinbase level2 messages: snapshot (full book) + update (incremental
price-level changes). market_trades messages: executed trades, each
already tagged with the taker's side.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class _BookState:
    bids: dict = field(default_factory=dict)  # price -> size
    asks: dict = field(default_factory=dict)

    def best_bid(self):
        return max(self.bids) if self.bids else np.nan

    def best_ask(self):
        return min(self.asks) if self.asks else np.nan

    def apply_level2(self, side: str, price: float, new_quantity: float):
        book = self.bids if side == "bid" else self.asks
        if new_quantity == 0:
            book.pop(price, None)
        else:
            book[price] = new_quantity


def build_book_snapshots(raw_messages: list[dict]) -> pd.DataFrame:
    """Reconstruct a best-bid/best-ask/mid/spread time series from a
    stream of level2 snapshot + update messages. One output row per
    incoming level2 message (i.e. per book change event)."""
    state = _BookState()
    rows = []

    for msg in raw_messages:
        if msg.get("channel") != "l2_data":
            continue
        for event in msg.get("events", []):
            event_type = event.get("type")
            updates = event.get("updates", [])
            if event_type == "snapshot":
                state = _BookState()  # snapshot replaces prior state entirely
            for u in updates:
                try:
                    side = u["side"]  # "bid" or "offer"
                    side = "bid" if side == "bid" else "ask"
                    price = float(u["price_level"])
                    qty = float(u["new_quantity"])
                except (KeyError, TypeError, ValueError):
                    continue  # malformed update -- skip, don't crash the whole run
                state.apply_level2(side, price, qty)

        bid, ask = state.best_bid(), state.best_ask()
        rows.append({
            "time": msg.get("_collected_at", np.nan),
            "best_bid": bid,
            "best_ask": ask,
            "mid_price": (bid + ask) / 2.0 if bid and ask and not np.isnan(bid) and not np.isnan(ask) else np.nan,
            "spread": (ask - bid) if bid and ask and not np.isnan(bid) and not np.isnan(ask) else np.nan,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # a book with only one side populated (early in collection, or a
    # thin/illiquid moment) produces NaN mid/spread -- drop those rather
    # than let them silently poison downstream rolling features
    return df.dropna(subset=["mid_price"]).sort_values("time").reset_index(drop=True)


def build_trade_events(raw_messages: list[dict]) -> pd.DataFrame:
    """Extract executed trades (the crypto analogue of LOBSTER's Type
    4/5 execution rows) with a signed aggressor_side: +1 if the taker
    bought (aggressor was a buyer), -1 if the taker sold."""
    rows = []
    for msg in raw_messages:
        if msg.get("channel") != "market_trades":
            continue
        for event in msg.get("events", []):
            for t in event.get("trades", []):
                try:
                    side = t["side"]  # coinbase reports the taker's side directly
                    aggressor_side = 1 if side.upper() == "BUY" else -1
                    rows.append({
                        "time": msg.get("_collected_at", np.nan),
                        "price": float(t["price"]),
                        "size": float(t["size"]),
                        "aggressor_side": aggressor_side,
                    })
                except (KeyError, TypeError, ValueError):
                    continue
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values("time").reset_index(drop=True)


def load_session(raw_dir: str, product_id: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convenience entry point: raw JSONL files -> (book_df, trades_df)."""
    from .storage import iter_raw_messages
    raw_messages = list(iter_raw_messages(raw_dir, product_id))
    book_df = build_book_snapshots(raw_messages)
    trades_df = build_trade_events(raw_messages)
    return book_df, trades_df
