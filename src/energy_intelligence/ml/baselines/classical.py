"""Classical ML baselines: what a competent tabular model achieves.

Two models, deliberately, because they fail differently:

``ridge``
    A linear model on the same features. Its value is not accuracy, it is the
    **coefficient table**: it shows how much each lag and calendar term contributes
    in a form a human can read, which is the interpretability diagnostic the phase
    brief asks for without adding an explainability dependency.

``hist_gradient_boosting``
    scikit-learn's histogram gradient boosting. This is the strongest classical
    baseline that trains in seconds on CPU, handles non-linear interactions and
    missing values natively, and needs no tuning to be respectable.

Both are fitted on training rows only, with a fixed ``random_state``. Neither is
tuned in this phase: a small, controlled experiment beats a large search on a
dataset this size, and a tuned number would not transfer to a different year.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

__all__ = ["ClassicalModel", "CLASSICAL_MODELS", "fit_classical", "model_defaults"]

#: Reported in this order.
CLASSICAL_MODELS: tuple[str, ...] = ("classical_ridge", "classical_hist_gbm")


def model_defaults(name: str) -> dict[str, Any]:
    """Hyperparameters for a named classical model.

    Defaults are stated here rather than hidden in the estimator so the experiment
    registry records the same values a reader would reproduce.
    """
    if name == "classical_ridge":
        return {
            "alpha": 1.0,
            "fit_intercept": True,
            "note": "alpha=1.0 is scikit-learn's own default; no search was run",
        }
    if name == "classical_hist_gbm":
        return {
            "max_iter": 200,
            "learning_rate": 0.1,
            "max_depth": None,
            "max_leaf_nodes": 31,
            "min_samples_leaf": 20,
            "l2_regularization": 0.0,
            "early_stopping": False,
            "note": (
                "defaults except max_iter=200 and early_stopping=False, so the "
                "training duration is deterministic and not split-dependent"
            ),
        }
    raise KeyError(f"unknown classical model {name!r}; available: {list(CLASSICAL_MODELS)}")


@dataclass(slots=True)
class ClassicalModel:
    """A fitted classical baseline.

    Attributes:
        name: Model name.
        estimator: The fitted scikit-learn estimator.
        feature_names: Column order the estimator was fitted on.
        params: Hyperparameters actually used.
        seed: Random seed, recorded even where the estimator is deterministic.
        fit_seconds: Wall-clock training duration.
        coefficients: Ridge coefficients aligned to ``feature_names``, if applicable.
    """

    name: str
    estimator: Any
    feature_names: tuple[str, ...]
    params: dict[str, Any]
    seed: int
    fit_seconds: float = 0.0
    coefficients: np.ndarray | None = None
    feature_importance: dict[str, float] = field(default_factory=dict)

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predict. Raises if the column count does not match training."""
        features = np.asarray(features, dtype=np.float64)
        if features.shape[1] != len(self.feature_names):
            raise ValueError(
                f"model {self.name} was fitted on {len(self.feature_names)} features, "
                f"got {features.shape[1]}"
            )
        return np.asarray(self.estimator.predict(features), dtype=np.float64)

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "name": self.name,
            "params": self.params,
            "seed": self.seed,
            "fit_seconds": round(self.fit_seconds, 3),
            "feature_count": len(self.feature_names),
        }
        if self.coefficients is not None:
            payload["coefficients"] = {
                name: float(value)
                for name, value in zip(self.feature_names, self.coefficients, strict=True)
            }
        if self.feature_importance:
            payload["feature_importance"] = dict(self.feature_importance)
        return payload


def fit_classical(
    name: str,
    features: np.ndarray,
    targets: np.ndarray,
    *,
    feature_names: tuple[str, ...],
    seed: int,
) -> ClassicalModel:
    """Fit one classical baseline.

    Args:
        name: One of :data:`CLASSICAL_MODELS`.
        features: ``(n_samples, n_features)`` training features.
        targets: ``(n_samples,)`` or ``(n_samples, n_targets)`` training targets.
        feature_names: Column names, retained for interpretability.
        seed: Recorded for reproducibility.

    Returns:
        The fitted model.

    Raises:
        ValueError: If the matrices are empty or misaligned.
    """
    import time

    features = np.asarray(features, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    if features.shape[0] != targets.shape[0]:
        raise ValueError(
            f"features and targets disagree: {features.shape[0]} rows vs {targets.shape[0]}"
        )
    if features.shape[0] == 0:
        raise ValueError("cannot fit a model on zero rows")

    params = model_defaults(name)
    started = time.perf_counter()

    if name == "classical_ridge":
        estimator = Ridge(alpha=params["alpha"], fit_intercept=params["fit_intercept"])
        estimator.fit(features, targets)
        coefficients = np.asarray(estimator.coef_, dtype=np.float64).reshape(-1)
        model = ClassicalModel(
            name=name,
            estimator=estimator,
            feature_names=feature_names,
            params=params,
            seed=seed,
            coefficients=coefficients,
        )
    elif name == "classical_hist_gbm":
        estimator = HistGradientBoostingRegressor(
            max_iter=params["max_iter"],
            learning_rate=params["learning_rate"],
            max_leaf_nodes=params["max_leaf_nodes"],
            min_samples_leaf=params["min_samples_leaf"],
            l2_regularization=params["l2_regularization"],
            early_stopping=params["early_stopping"],
            random_state=seed,
        )
        estimator.fit(features, targets)
        # No built-in impurity importances without extra passes; permutation-free
        # proxy reported instead: the correlation of each feature with the residual
        # is NOT used. Feature importance is left empty rather than faked, and the
        # ridge table carries the interpretability requirement.
        model = ClassicalModel(
            name=name,
            estimator=estimator,
            feature_names=feature_names,
            params=params,
            seed=seed,
        )
    else:
        raise KeyError(f"unknown classical model {name!r}; available: {list(CLASSICAL_MODELS)}")

    model.fit_seconds = time.perf_counter() - started
    return model