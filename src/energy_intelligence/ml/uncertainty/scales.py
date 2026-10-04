"""Scale functions: the one idea that separates this phase's interval methods.

Every symmetric interval in this phase is ``point +/- multiplier * scale(x)``. The family
- empirical residual quantile, split conformal, pinball quantile regression - decides how
``multiplier`` is obtained. The scale decides whether the width is allowed to *vary* with
the row.

Three scale functions are implemented, and the whole phase is the question of whether any
of them beats a constant:

``GlobalScale``
    One number per horizon. The hypothesis under test is "uncertainty does not need to
    vary", and this is the thing it has to beat. It is not a strawman: for load forecasting
    at 15-minute resolution, error may genuinely be close to stationary given a decent
    point predictor, and a method that establishes that has done real work.

``LearnedScale``
    A gradient-boosted model of ``|residual|`` from origin-observable state, fitted on one
    split and applied to another. The hypothesis is "error magnitude depends on conditions
    the router can see".

``DisagreementScale``
    The Phase 7 expert pool's own spread. The hypothesis is the prompt's §23 one: "greater
    expert disagreement may indicate greater forecast uncertainty". This is the only scale
    that needs no fitting at all, which makes it the most deployable of the three and the
    most interesting to test.

A scale is deliberately **not** a probability or a confidence level. It is a positive
number with the units of the residual (or is dimensionless, when normalised), and what it
means is set by how it is used downstream. Keeping that separation is what stops a
"confidence: 0.8" in a report from being an unbacked claim.

Fitting discipline, inherited from Phase 7 and not relaxed here: a scale is fitted on one
chronological portion of the validation split and its multiplier is taken from a *different*
portion, so the scale never depends on the residuals it will be asked to normalise.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = [
    "GlobalScale",
    "LearnedScale",
    "DisagreementScale",
    "fit_learned_scale",
    "fitted_scale",
]

# There is deliberately no shared ``ScaleFunction`` protocol. The three scales have
# genuinely different call signatures - one is stateless, one needs a feature matrix, one
# needs a precomputed spread - and forcing them behind one signature would mean either an
# ``Optional`` argument nobody reads or an adapter that hides what each one actually needs.
# What they share is the contract that matters: ``predict``/``__call__`` returns
# ``[rows, horizons]``, strictly positive, in the units of the residual.
@dataclass(frozen=True, slots=True)
class GlobalScale:
    """One scale per horizon, fitted from the mean absolute calibration residual.

    Using the **mean absolute** residual rather than the median makes this the scale a
    plain ``+/- 1 sigma`` rule would use, which is the fairest possible strawman: it is
    what a practitioner reaches for by default.

    Attributes:
        values: ``[horizons]``, in kW.
        horizons: Horizon steps.
        fitted_on: How many calibration rows produced each value.
    """

    values: np.ndarray
    horizons: tuple[int, ...]
    fitted_on: int

    name = "global"

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=np.float64)
        if values.shape != (len(self.horizons),):
            raise ValueError(
                f"values must be [horizons] = {(len(self.horizons),)}, got {values.shape}"
            )
        if np.any(values <= 0):
            raise ValueError(
                f"a scale must be strictly positive; got a minimum of "
                f"{float(values.min())}"
            )

    def predict(self, n_rows: int) -> np.ndarray:
        """The scale for ``n_rows`` rows, every row identical."""
        return np.tile(np.asarray(self.values, dtype=np.float64), (int(n_rows), 1))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "horizons": [int(h) for h in self.horizons],
            "values_kw": [float(v) for v in self.values],
            "fitted_on_rows": int(self.fitted_on),
            "units": "kW, the mean absolute calibration residual",
        }


@dataclass(frozen=True, slots=True)
class LearnedScale:
    """A gradient-boosted model of the absolute residual, from origin-observable state.

    Fits ``E[|y - yhat| | x]`` with an L1 objective, which is the conditional median of
    ``|residual|`` rather than its conditional mean. The median is used deliberately: the
    absolute residual has a long right tail (peak errors are 5-8x typical), and a model of
    the conditional mean would chase the tail and inflate every row's width.

    Attributes:
        model: The fitted ``HistGradientBoostingRegressor``.
        feature_names: The columns it was fitted on, in order.
        horizon_index: Which horizon column the model predicts.
        floor_kw: Lower bound on the prediction, so a near-zero row cannot produce a
            near-zero width and claim false precision.
        name: Recorded on the result.
    """

    model: Any
    feature_names: tuple[str, ...]
    horizon_index: int
    floor_kw: float = 1e-3
    name: str = "learned"

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predicted absolute residual for one horizon, floored at :attr:`floor_kw`."""
        matrix = np.asarray(features, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != len(self.feature_names):
            raise ValueError(
                f"features must be [rows, {len(self.feature_names)}], got {matrix.shape}"
            )
        prediction = np.asarray(self.model.predict(matrix), dtype=np.float64)
        return np.maximum(prediction, float(self.floor_kw))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "horizon_index": int(self.horizon_index),
            "features": list(self.feature_names),
            "floor_kw": float(self.floor_kw),
            "objective": "L1 on |residual|; a model of the conditional median, not mean",
        }


@dataclass(frozen=True, slots=True)
class DisagreementScale:
    """The expert pool's own spread, used directly as a width scale.

    No fitting, no calibration data, no parameters. If this works it is the cheapest
    deployable uncertainty available, which is why it is tested rather than assumed.

    Attributes:
        measure: Which spread statistic to use, from
            :data:`uncertainty.measures.PAIRWISE_MEASURES`.
        horizons: Horizon steps.
        source: Label of where the spread came from, for the record.
        floor: Lower bound applied, so that three exactly-agreeing experts cannot produce
            a zero width. A zero width is a claim of infinite precision, which is never
            defensible and would break the conformal division.
    """

    measure: str = "max_minus_min"
    horizons: tuple[int, ...] = (1, 4, 96)
    source: str = "expert_pool"
    floor: float = 1e-4
    name: str = "disagreement"

    def __call__(self, spread: np.ndarray, row_scale_kw: np.ndarray) -> np.ndarray:
        """Turn a per-unit spread into a kW scale.

        Args:
            spread: ``[N, horizons]`` per-unit spread.
            row_scale_kw: ``[N]`` rated kW.

        Returns:
            ``[N, horizons]`` kW, floored.

        Raises:
            ValueError: On shape disagreement.
        """
        values = np.asarray(spread, dtype=np.float64)
        scale = np.asarray(row_scale_kw, dtype=np.float64)
        if values.shape[1] != len(self.horizons):
            raise ValueError(
                f"spread has {values.shape[1]} horizon columns for {len(self.horizons)} "
                f"horizons"
            )
        if scale.shape != (values.shape[0],):
            raise ValueError(
                f"row_scale_kw must be [rows] = {(values.shape[0],)}, got {scale.shape}"
            )
        return np.maximum(values * scale[:, None], float(self.floor))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "measure": self.measure,
            "source": self.source,
            "horizons": [int(h) for h in self.horizons],
            "floor": float(self.floor),
            "units": "kW; the per-unit spread multiplied by the row's rated kW",
            "fitted": False,
        }


def fitted_scale(
    *,
    residual_kw: np.ndarray,
    horizons: tuple[int, ...],
) -> GlobalScale:
    """Fit the global scale: the mean absolute residual per horizon.

    Args:
        residual_kw: ``[N, horizons]`` residuals, kW.
        horizons: Horizon steps.

    Returns:
        The scale.

    Raises:
        ValueError: If a horizon's mean absolute residual is not strictly positive, which
            would mean the point forecast was exact and no width is meaningful.
    """
    residual = np.abs(np.asarray(residual_kw, dtype=np.float64))
    horizons = tuple(int(h) for h in horizons)
    if residual.shape[1] != len(horizons):
        raise ValueError(
            f"residual_kw has {residual.shape[1]} horizon columns for {len(horizons)} "
            f"horizons"
        )
    values = residual.mean(axis=0)
    return GlobalScale(values=values, horizons=horizons, fitted_on=int(residual.shape[0]))


def fit_learned_scale(
    *,
    features: np.ndarray,
    residual_kw: np.ndarray,
    feature_names: tuple[str, ...],
    horizon_index: int,
    seed: int,
    max_iter: int = 200,
    learning_rate: float = 0.06,
    max_leaf_nodes: int = 31,
    min_samples_leaf: int = 40,
) -> LearnedScale:
    """Fit one horizon's conditional-median absolute residual.

    Args:
        features: ``[N, n_features]`` origin-observable features.
        residual_kw: ``[N]`` residual for this horizon, kW.
        feature_names: Column names, stored with the model.
        horizon_index: Which horizon column this predicts, for the record.
        seed: Estimator seed.
        max_iter: Boosting iterations.
        learning_rate: Shrinkage.
        max_leaf_nodes: Tree size.
        min_samples_leaf: Regularisation.

    Returns:
        The fitted scale.
    """
    from sklearn.ensemble import HistGradientBoostingRegressor

    matrix = np.asarray(features, dtype=np.float64)
    target = np.abs(np.asarray(residual_kw, dtype=np.float64))
    model = HistGradientBoostingRegressor(
        loss="absolute_error",
        max_iter=int(max_iter),
        learning_rate=float(learning_rate),
        max_leaf_nodes=int(max_leaf_nodes),
        min_samples_leaf=int(min_samples_leaf),
        early_stopping=False,
        random_state=int(seed),
    )
    model.fit(matrix, target)
    return LearnedScale(
        model=model,
        feature_names=tuple(feature_names),
        horizon_index=int(horizon_index),
    )