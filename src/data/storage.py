"""
Thin I/O helpers around the raw JSONL files the collector writes.
Kept separate from parsing logic (loader.py) on purpose -- storage
should never need to know what's *inside* a message.
"""

import json
from pathlib import Path
from typing import Iterator

import pandas as pd


def iter_raw_messages(raw_dir: str, product_id: str | None = None) -> Iterator[dict]:
    """Yield every raw message across all JSONL files in raw_dir, in
    file-sorted (i.e. roughly chronological) order. Returns an empty
    iterator -- not an error -- if the directory has no matching files,
    since "no data collected yet" is a normal state, not a bug."""
    raw_path = Path(raw_dir)
    if not raw_path.exists():
        return

    pattern = f"{product_id}_*.jsonl" if product_id else "*.jsonl"
    for file in sorted(raw_path.glob(pattern)):
        with open(file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue  # tolerate blank lines from a killed collector process
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    # a truncated last line (process killed mid-write) should not
                    # crash an hours-long collection run -- skip and move on
                    continue


def raw_messages_to_dataframe(raw_dir: str, product_id: str | None = None) -> pd.DataFrame:
    records = list(iter_raw_messages(raw_dir, product_id))
    if not records:
        return pd.DataFrame()
    return pd.json_normalize(records)


def save_processed(df: pd.DataFrame, processed_dir: str, name: str) -> Path:
    out_dir = Path(processed_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.parquet"
    df.to_parquet(path, index=False)
    return path


def load_processed(processed_dir: str, name: str) -> pd.DataFrame:
    path = Path(processed_dir) / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"No processed file at {path}. Run the loader/feature pipeline first."
        )
    return pd.read_parquet(path)
