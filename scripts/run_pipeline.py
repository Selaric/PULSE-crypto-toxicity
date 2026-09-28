#!/usr/bin/env python3
"""Entry point: `python scripts/run_pipeline.py`
Load collected data -> build features -> sweep + apply toxicity labels
-> train baseline models -> run TWAP vs adaptive backtest. Prints a
summary; nothing here is fully automated end-to-end without you
looking at the sweep output and picking horizon/threshold values --
see the TODOs printed along the way.
"""

import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data.loader import load_session
from src.features.microstructure_features import add_microstructure_features
from src.labeling.toxicity_label import label_toxicity, check_no_leakage, sweep_toxicity_thresholds
from src.models.baseline_models import chronological_split, train_logreg, train_lightgbm, evaluate
from src.evaluation.backtest import run_twap, run_adaptive
from src.utils.logging_utils import get_logger

log = get_logger("run_pipeline")


def main():
    config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    log.info("Loading collected session data...")
    book_df, trades_df = load_session(config["storage"]["raw_dir"], config["coinbase"]["product_id"])
    log.info(f"Loaded {len(book_df)} book events, {len(trades_df)} trades")

    if book_df.empty or trades_df.empty:
        log.warning("No data found. Run scripts/collect_data.py first and let it run a while.")
        return

    log.info("Building features...")
    book_df = add_microstructure_features(book_df, trades_df, window_size=config["features"]["rolling_window_updates"])

    log.info("TODO: run sweep_toxicity_thresholds() and inspect the output below before")
    log.info("trusting config.yaml's horizon_events/threshold_bps -- crypto's volatility")
    log.info("profile has not been validated against the equity-derived defaults.")
    sweep = sweep_toxicity_thresholds(book_df, trades_df)
    print(sweep.pivot_table(index="horizon_events", columns="threshold_bps", values="toxic_rate"))

    horizon = config["labeling"]["horizon_events"]
    bps = config["labeling"]["threshold_bps"]
    labeled = label_toxicity(book_df, trades_df, horizon_events=horizon, threshold_bps=bps)
    check_no_leakage(book_df, labeled, horizon_events=horizon)
    log.info(f"Labeled {len(labeled)} events, toxic rate={labeled['label'].mean():.3f}")

    merged = pd.merge_asof(
        labeled.sort_values("time"),
        book_df.sort_values("time"),
        on="time", direction="backward",
    ) if False else None  # TODO: you'll need to merge labeled trades with their contemporaneous
                            # feature row (as-of, backward direction) before this split works --
                            # left as an explicit step since the exact join key depends on how
                            # you end up storing feature snapshots once you have real data volume

    log.info("Feature/label merge is a TODO -- see comment above. Stopping here for the skeleton.")


if __name__ == "__main__":
    import pandas as pd
    main()
