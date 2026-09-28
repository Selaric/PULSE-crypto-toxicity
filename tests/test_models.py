"""Edge cases for the online PULSE-inspired model: prediction before
any fit, single-sample updates, and a sanity check that the model
actually learns something on an easy separable case."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.models.online_pulse import OnlineToxicityModel


def test_predict_before_fit_returns_uninformative_default():
    model = OnlineToxicityModel(n_features=3)
    pred = model.predict_proba_one(np.array([1.0, 2.0, 3.0]))
    assert pred == 0.5


def test_single_update_does_not_crash():
    model = OnlineToxicityModel(n_features=2)
    model.update_one(np.array([1.0, 0.0]), 1)
    pred = model.predict_proba_one(np.array([1.0, 0.0]))
    assert 0.0 <= pred <= 1.0


def test_run_stream_returns_pre_update_predictions_only():
    model = OnlineToxicityModel(n_features=1)
    X = np.array([[0.0], [0.0], [0.0]])
    y = np.array([1, 1, 1])
    preds = model.run_stream(X, y)
    assert len(preds) == 3
    assert preds[0] == 0.5  # first prediction happens before any learning -- must be the default


def test_model_learns_a_trivially_separable_pattern():
    rng = np.random.default_rng(0)
    n = 200
    X = rng.normal(size=(n, 1))
    y = (X[:, 0] > 0).astype(int)  # perfectly separable by one feature
    model = OnlineToxicityModel(n_features=1, learning_rate=0.1)
    preds = model.run_stream(X, y)
    # predictions in the second half (after the model has seen enough data)
    # should correlate with the true labels better than a coin flip
    late_preds = preds[100:]
    late_labels = y[100:]
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(late_labels, late_preds)
    assert auc > 0.6  # not claiming great performance, just "better than random"
