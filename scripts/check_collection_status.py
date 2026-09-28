#!/usr/bin/env python3
"""Print a lightweight status summary for collected raw messages."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--product-id", default="BTC-USD")
    args = parser.parse_args()

    files = sorted(Path(args.raw_dir).glob(f"{args.product_id}_*.jsonl"))
    message_count = 0
    timestamps = []

    for path in files:
        with path.open() as raw_file:
            for line in raw_file:
                try:
                    message = json.loads(line)
                except json.JSONDecodeError:
                    continue
                message_count += 1
                collected_at = message.get("_collected_at")
                if isinstance(collected_at, (int, float)):
                    timestamps.append(collected_at)

    print(f"files: {len(files)}")
    print(f"messages: {message_count}")
    if timestamps:
        first = datetime.fromtimestamp(min(timestamps), timezone.utc).isoformat()
        last = datetime.fromtimestamp(max(timestamps), timezone.utc).isoformat()
        print(f"earliest_collected_at: {first}")
        print(f"latest_collected_at: {last}")
    else:
        print("earliest_collected_at: n/a")
        print("latest_collected_at: n/a")


if __name__ == "__main__":
    main()