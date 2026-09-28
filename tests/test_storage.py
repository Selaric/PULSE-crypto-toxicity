"""Edge cases for raw-message I/O: no data yet, blank lines, truncated
last line from a killed process."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data.storage import iter_raw_messages, raw_messages_to_dataframe


def test_missing_directory_returns_empty(tmp_path):
    messages = list(iter_raw_messages(str(tmp_path / "does_not_exist")))
    assert messages == []


def test_empty_directory_returns_empty_dataframe(tmp_path):
    df = raw_messages_to_dataframe(str(tmp_path))
    assert df.empty


def test_blank_lines_are_skipped(tmp_path):
    f = tmp_path / "BTC-USD_20260101T000000.jsonl"
    f.write_text('{"a": 1}\n\n{"a": 2}\n')
    messages = list(iter_raw_messages(str(tmp_path)))
    assert len(messages) == 2


def test_truncated_last_line_does_not_crash(tmp_path):
    f = tmp_path / "BTC-USD_20260101T000000.jsonl"
    f.write_text('{"a": 1}\n{"a": 2, "b": tru')  # simulates a process killed mid-write
    messages = list(iter_raw_messages(str(tmp_path)))
    assert len(messages) == 1
    assert messages[0]["a"] == 1


def test_product_id_filter(tmp_path):
    (tmp_path / "BTC-USD_20260101T000000.jsonl").write_text('{"a": 1}\n')
    (tmp_path / "ETH-USD_20260101T000000.jsonl").write_text('{"a": 2}\n')
    messages = list(iter_raw_messages(str(tmp_path), product_id="BTC-USD"))
    assert len(messages) == 1
    assert messages[0]["a"] == 1
