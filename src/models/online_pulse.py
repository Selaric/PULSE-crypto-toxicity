"""
PULSE-INSPIRED online model -- explicitly NOT a replication of the
full method in Cartea, Duran-Martin & Sánchez-Betancourt (2023,
arXiv:2312.05827).

The real PULSE is a Bayesian procedure for online training of a
neural network's LAST LAYER via a projected Kalman-filter-style
update, giving calibrated uncertainty with sub-millisecond updates.
That is a substantial piece of Bayesian ML engineering in its own
right and is deliberately NOT implemented here.

What IS implemented: a simple recursive/online logistic regression
via scikit-learn's SGDClassifier with partial_fit, updated one
labeled event at a time. This gives you the same PIPELINE SHAPE
(a model that updates continuously as new labeled toxicity events
arrive, rather than being trained once and frozen) without the
Bayesian machinery. Treat this class as the seam where a real PULSE
implementation would plug in later, not as PULSE itself.

TODO (future project, not this skeleton): replace the SGDClassifier
internals with an actual Bayesian last-layer update if you want the
real method.
"""

import numpy as np
from sklearn.linear_model import SGDClassifier


class OnlineToxicityModel:
    def __init__(self, n_features: int, learning_rate: float = 0.01, random_state: int = 0):
        self.model = SGDClassifier(
            loss="log_loss",
            learning_rate="constant",
            eta0=learning_rate,
            random_state=random_state,
        )
        self._classes = np.array([0, 1])
        self._is_initialized = False
        self.n_features = n_features

    def predict_proba_one(self, x: np.ndarray) -> float:
        """Predicted P(toxic) for a single feature vector. Returns 0.5
        (maximally uninformative) before the model has seen any data --
        an explicit, documented default rather than an arbitrary crash
        or a silently-wrong extrapolation from an unfit model."""
        if not self._is_initialized:
            return 0.5
        x = np.asarray(x).reshape(1, -1)
        return float(self.model.predict_proba(x)[0, 1])

    def update_one(self, x: np.ndarray, y: int):
        """Online update on a single new labeled (feature, label) pair."""
        x = np.asarray(x).reshape(1, -1)
        if not self._is_initialized:
            # partial_fit's first call requires the full class list up front
            self.model.partial_fit(x, [y], classes=self._classes)
            self._is_initialized = True
        else:
            self.model.partial_fit(x, [y])

    def run_stream(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Convenience: predict-then-update over a whole array in order,
        simulating what deployment looks like -- each prediction uses
        only what the model has learned from events BEFORE it, matching
        the same no-look-ahead discipline as labeling.py. Returns the
        array of pre-update predictions (what you'd actually have had
        at decision time)."""
        predictions = np.zeros(len(X))
        for i in range(len(X)):
            predictions[i] = self.predict_proba_one(X[i])
            self.update_one(X[i], int(y[i]))
        return predictions
