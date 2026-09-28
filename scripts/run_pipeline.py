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

import pandas as pd
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

    # Merge each labeled trade with the most recent feature snapshot from at
    # or before its own timestamp to preserve the no-look-ahead constraint.
    merged = pd.merge_asof(
        labeled.sort_values("time"),
        book_df.sort_values("time"),
        on="time", direction="backward",
    )
    log.info(f"Merged {len(merged)} labeled events with contemporaneous features")

    split = chronological_split(
        merged, label_col="label",
        train_frac=config["models"]["train_frac"],
        val_frac=config["models"]["val_frac"],
    )
    log.info(
        f"Split: {len(split.y_train)} train / {len(split.y_val)} val / "
        f"{len(split.y_test)} test"
    )
    print(split.y_val.mean())

    logreg_model, scaler = train_logreg(split, **config["models"]["logreg"])
    logreg_metrics = evaluate(logreg_model, split.X_val, split.y_val, scaler=scaler)
    log.info(f"LogReg validation: {logreg_metrics}")

    lgbm_model = train_lightgbm(split, **config["models"]["lightgbm"])
    lgbm_metrics = evaluate(lgbm_model, split.X_val, split.y_val)
    log.info(f"LightGBM validation: {lgbm_metrics}")


if __name__ == "__main__":
    main()
