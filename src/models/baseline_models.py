"""
Static baseline models: logistic regression and LightGBM, trained once
on a chronological split. These are the "not updated during deploy"
benchmarks that PULSE (Cartea et al. 2023) compares itself against --
keeping them static and separate from online_pulse.py is intentional,
so the comparison in evaluation/ means what it claims to mean.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss


FEATURE_COLUMNS = [
    "order_flow_imbalance",
    "aggressiveness_volume_imbalance",
    "micro_price",
    "spread_feature",
    "short_horizon_volatility",
    "arrival_rate",
]


@dataclass
class ChronoSplit:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray


def chronological_split(df: pd.DataFrame, label_col: str = "label",
                         train_frac: float = 0.70, val_frac: float = 0.15) -> ChronoSplit:
    """Splits by TIME ORDER, not randomly -- a random split would leak
    future regime information into training, exactly the mistake the
    equity project's design notes warned about for labeling."""
    if df.empty:
        raise ValueError("Cannot split an empty DataFrame -- run labeling first.")

    df = df.sort_values("time").reset_index(drop=True)
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    cols = [c for c in FEATURE_COLUMNS if c in df.columns]
    X = df[cols].fillna(0.0).values
    y = df[label_col].values

    return ChronoSplit(
        X_train=X[:train_end], y_train=y[:train_end],
        X_val=X[train_end:val_end], y_val=y[train_end:val_end],
        X_test=X[val_end:], y_test=y[val_end:],
    )


def train_logreg(split: ChronoSplit, C: float = 1.0, class_weight: str | None = "balanced"):
    scaler = StandardScaler().fit(split.X_train)
    model = LogisticRegression(C=C, class_weight=class_weight, max_iter=1000)
    model.fit(scaler.transform(split.X_train), split.y_train)
    return model, scaler


def train_lightgbm(split: ChronoSplit, learning_rate: float = 0.05, num_leaves: int = 31):
    import lightgbm as lgb
    model = lgb.LGBMClassifier(learning_rate=learning_rate, num_leaves=num_leaves, verbosity=-1)
    model.fit(split.X_train, split.y_train)
    return model


def evaluate(model, X, y, scaler=None) -> dict:
    """Returns ROC-AUC, PR-AUC, and Brier score. Handles the edge case
    of a single-class y (all-toxic or all-safe window) explicitly --
    sklearn's AUC metrics raise on that, and a silent crash mid-sweep
    is worse than a clearly-labeled NaN result."""
    if len(np.unique(y)) < 2:
        return {"roc_auc": np.nan, "pr_auc": np.nan, "brier": np.nan,
                "note": "single-class evaluation window -- metrics undefined"}

    X_in = scaler.transform(X) if scaler is not None else X
    proba = model.predict_proba(X_in)[:, 1]
    return {
        "roc_auc": roc_auc_score(y, proba),
        "pr_auc": average_precision_score(y, proba),
        "brier": brier_score_loss(y, proba),
        "mean_predicted_prob": float(np.mean(proba)),
    }
