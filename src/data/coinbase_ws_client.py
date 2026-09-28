"""
Coinbase Advanced Trade WebSocket client.

Connects to the public level2 + market_trades feeds, writes every raw
message to a rotating JSONL file, and reconnects with backoff on
disconnect. This is intentionally "dumb" -- it does not parse or
reconstruct the order book. That happens later, offline, in loader.py,
so that collection can never silently drop data due to a parsing bug.

Run standalone via scripts/collect_data.py.
"""

import asyncio
import json
import time
from pathlib import Path
from typing import Optional

import websockets


class CoinbaseCollector:
    def __init__(
        self,
        product_id: str,
        channels: list[str],
        ws_url: str,
        raw_dir: str,
        rotate_every_n_messages: int = 50_000,
        reconnect_backoff_seconds: float = 5,
        max_reconnect_backoff_seconds: float = 60,
    ):
        self.product_id = product_id
        self.channels = channels
        self.ws_url = ws_url
        self.raw_dir = Path(raw_dir)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.rotate_every_n_messages = rotate_every_n_messages
        self.reconnect_backoff_seconds = reconnect_backoff_seconds
        self.max_reconnect_backoff_seconds = max_reconnect_backoff_seconds

        self._current_file = None
        self._current_file_msg_count = 0
        self._total_messages = 0

    def _new_file(self):
        if self._current_file:
            self._current_file.close()
        ts = time.strftime("%Y%m%dT%H%M%S")
        path = self.raw_dir / f"{self.product_id}_{ts}.jsonl"
        self._current_file = open(path, "a", buffering=1)  # line-buffered
        self._current_file_msg_count = 0

    def _write(self, msg: dict):
        if self._current_file is None or self._current_file_msg_count >= self.rotate_every_n_messages:
            self._new_file()
        self._current_file.write(json.dumps(msg) + "\n")
        self._current_file_msg_count += 1
        self._total_messages += 1

    def _subscribe_payload(self) -> dict:
        # NOTE: public market-data channels (level2, market_trades) do not
        # require auth on Coinbase's Advanced Trade WS feed. If you add
        # authenticated channels later, signing goes here -- TODO for you.
        return {
            "type": "subscribe",
            "product_ids": [self.product_id],
            "channel": ",".join(self.channels) if len(self.channels) == 1 else self.channels[0],
        }

    async def run_forever(self):
        backoff = self.reconnect_backoff_seconds
        while True:
            try:
                async with websockets.connect(
                    self.ws_url, ping_interval=20, max_size=None
                ) as ws:
                    for channel in self.channels:
                        await ws.send(json.dumps({
                            "type": "subscribe",
                            "product_ids": [self.product_id],
                            "channel": channel,
                        }))
                    backoff = self.reconnect_backoff_seconds  # reset after a clean connect
                    async for raw in ws:
                        msg = json.loads(raw)
                        msg["_collected_at"] = time.time()
                        self._write(msg)
            except (websockets.ConnectionClosed, OSError) as e:
                print(f"[collector] disconnected ({e}), reconnecting in {backoff}s")
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, self.max_reconnect_backoff_seconds)
            except Exception as e:
                print(f"[collector] unexpected error ({e}), reconnecting in {backoff}s")
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, self.max_reconnect_backoff_seconds)

    def close(self):
        if self._current_file:
            self._current_file.close()


def run(config: dict):
    collector = CoinbaseCollector(
        product_id=config["coinbase"]["product_id"],
        channels=config["coinbase"]["channels"],
        ws_url=config["coinbase"]["ws_url"],
        raw_dir=config["storage"]["raw_dir"],
        rotate_every_n_messages=config["coinbase"]["rotate_every_n_messages"],
        reconnect_backoff_seconds=config["coinbase"]["reconnect_backoff_seconds"],
        max_reconnect_backoff_seconds=config["coinbase"]["max_reconnect_backoff_seconds"],
    )
    try:
        asyncio.run(collector.run_forever())
    except KeyboardInterrupt:
        print("[collector] stopped by user")
    finally:
        collector.close()
