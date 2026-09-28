#!/usr/bin/env python3
"""Entry point: `python scripts/collect_data.py`
Runs the Coinbase collector until you Ctrl+C. Let it run for at least
several hours before attempting labeling -- see config.yaml for
rotate_every_n_messages and the README for how much data is "enough."
"""

import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data.coinbase_ws_client import run

if __name__ == "__main__":
    config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    run(config)
