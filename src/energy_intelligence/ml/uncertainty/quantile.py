"""Direct quantile regression on the residual, by pinball loss.

The third interval family. Where the other two take quantiles of *realised* residuals and
attach them to a point forecast, this fits a model of the residual quantiles themselves, so
the width varies with the features the model sees rather than being scaled afterwards.

**Target representation, stated because it is the part that is easy to get wrong.** The
model predicts quantiles of the **residual** ``e = y - yhat_fixed``, not of ``y``. The
interval is then ``yhat_fixed + Q_q(e)``. Two consequences:

* The point forecast is byte-identical to the Phase 7 fixed ensemble, so the interval
  method is the only thing varying. Comparing a quantile model's *intervals* against a
  constant-width interval built on a different median would confound interval quality with
  point-forecast quality.
* It is still direct quantile estimation of the predictive distribution:
  ``Q_q(y | x) = yhat(x) + Q_q(e | x)`` is the same object as ``Q_q(y | x)``. Nothing is
  approximated.

The alternative - fitting quantiles of ``y`` directly - would change the median away from
the fixed ensemble's, so its intervals would be judged partly on a better or worse central
forecast. That is a real question but a different one, and it is recorded as not run rather
than half-run.

**Asymmetry is available and used.** The residual distribution is not symmetric about zero:
under-forecasting a demand peak costs more than over-forecasting it, and Phase 7 measured
a peak-to-median error ratio above 8. So the lower and upper quantiles are fitted
separately rather than being reflected.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from .intervals import PredictionInterval, lower_from_point, quantile_levels

__all__ = ["QuantileResidualModel", "fit_quantile_model"]


@dataclass(frozen=True, slots=True)
class QuantileResidualModel:
    """Fitted residual quantiles for one horizon, at a set of levels.

    Attributes:
        models: ``{quantile: estimator}``, each predicting that quantile of the residual.
        quantiles: The fitted quantile levels, ascending.
        feature_names: Columns the models were fitted on.
        horizon: Horizon step this model covers.
        fit_seconds: Wall time for the fit.
        train_rows: Rows used.
        target_units: Recorded on the result, because a per-unit target here would silently
            produce intervals in the wrong unit.
    """

    models: dict[float, Any]
    quantiles: tuple[float, ...]
    feature_names: tuple[str, ...]
    horizon: int
    fit_seconds: float
    train_rows: int
    target_units: str = "kW"

    def predict_quantiles(self, features: np.ndarray) -> dict[float, np.ndarray]:
        """Predicted residual quantiles, in kW.

        Args:
            features: ``[N, n_features]`` origin-observable features.

        Returns:
            ``{quantile: [N] array}``.

        Raises:
            ValueError: On shape disagreement.
        """
        matrix = np.asarray(features, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != len(self.feature_names):
            raise ValueError(
                f"features must be [rows, {len(self.feature_names)}], got {matrix.shape}"
            )
        return {
            float(q): np.asarray(model.predict(matrix), dtype=np.float64)
            for q, model in self.models.items()
        }

    def monotone_quantiles(
        self,
        features: np.ndarray,
        *,
        nominal_levels: tuple[float, ...] = (),
    ) -> tuple[dict[float, np.ndarray], dict[float, int]]:
        """Predicted residual quantiles, re-sorted into ascending order per row.

        Independently fitted pinball-loss models **cross**: each quantile is fitted without
        reference to the others, so on some rows a lower quantile is predicted above a
        higher one. An interval built from a crossed pair has negative width, so the
        crossing has to be dealt with rather than discovered later.

        The repair is the standard monotone rearrangement - sort each row's fitted
        quantiles - which is the smallest change that restores a valid interval. It is
        **counted and reported** rather than applied silently, because a method whose
        output has to be re-sorted is a weaker method than one whose does not.

        The count is per *reported level*, not per grid pair. The grid carries seven levels
        and the phase reports four intervals from them; adjacent pairs such as 0.90 against
        0.95 cross far more often than the 0.05/0.95 pair that the headline interval
        actually uses, and quoting the grid-wide rate would overstate the problem by an
        order of magnitude.

        Args:
            features: ``[N, n_features]`` origin-observable features.
            nominal_levels: Levels whose lower/upper pairs should be counted.

        Returns:
            ``(quantiles, crossings)``. ``quantiles`` is ``{quantile: [N] array}`` with each
            row ascending; ``crossings`` maps each requested nominal level to how many rows
            had its own lower/upper pair reversed before the sort.

        Raises:
            ValueError: If a requested level's snapped pair is the same quantile twice.
        """
        raw = self.predict_quantiles(features)
        levels = sorted(raw)
        stacked = np.column_stack([raw[level] for level in levels])
        ordered = np.sort(stacked, axis=1)
        index = {level: position for position, level in enumerate(levels)}
        crossings: dict[float, int] = {}
        for nominal in nominal_levels:
            lower_tail, upper_tail = quantile_levels((float(nominal),))[0]
            lower_q = self.quantile_for_tail(lower_tail)
            upper_q = self.quantile_for_tail(upper_tail)
            if lower_q == upper_q:
                raise ValueError(
                    f"nominal level {nominal} snapped both tails to quantile "
                    f"{lower_q}; an interval built from one quantile has no width"
                )
            lo, hi = index[lower_q], index[upper_q]
            crossings[float(nominal)] = int(
                np.count_nonzero(stacked[:, lo] > stacked[:, hi])
            )
        return {level: ordered[:, position] for position, level in enumerate(levels)}, crossings

    def intervals(
        self,
        *,
        point_kw: np.ndarray,
        features: np.ndarray,
        nominal_levels: tuple[float, ...],
        horizons: tuple[int, ...],
        method: str = "quantile_regression",
        clip_at_zero: bool = True,
    ) -> dict[float, PredictionInterval]:
        """Intervals at every nominal level, for this model's horizon column.

        Args:
            point_kw: ``[N, horizons]`` point forecast, kW.
            features: ``[N, n_features]`` the features the models were fitted on.
            nominal_levels: Nominal coverage levels.
            horizons: Horizon steps, so the column index is derived rather than assumed.
            method: Procedure name recorded on each interval.
            clip_at_zero: Raise negative lower bounds to zero.

        Returns:
            ``{nominal_level: PredictionInterval}`` with ``[N, 1]`` bounds.

        Raises:
            ValueError: If the model covers a horizon not in ``horizons``.
        """
        horizons = tuple(int(h) for h in horizons)
        point = np.asarray(point_kw, dtype=np.float64)
        if self.horizon not in horizons:
            raise ValueError(
                f"this model covers horizon {self.horizon}, not in {horizons}"
            )
        matrix_rows = int(np.asarray(features).shape[0])
        column = horizons.index(self.horizon)
        centre = point[:, column][:, None]
        predicted, crossings = self.monotone_quantiles(
            features, nominal_levels=tuple(nominal_levels)
        )
        out: dict[float, PredictionInterval] = {}
        for nominal in nominal_levels:
            lower_tail, upper_tail = quantile_levels((float(nominal),))[0]
            lower_q = self.quantile_for_tail(lower_tail)
            upper_q = self.quantile_for_tail(upper_tail)
            out[float(nominal)] = lower_from_point(
                point_kw=centre,
                lower_delta_kw=predicted[lower_q][:, None],
                upper_delta_kw=predicted[upper_q][:, None],
                method=method,
                nominal_level=float(nominal),
                clip_at_zero=clip_at_zero,
                notes=(
                    f"pinball loss on the residual, horizon {self.horizon}; the nominal "
                    f"{nominal:.0%} level uses fitted quantiles "
                    f"{lower_q:.3f}/{upper_q:.3f}; {crossings[float(nominal)]} of "
                    f"{matrix_rows} rows had that pair reversed and were re-sorted into "
                    f"ascending order"
                ),
            )
        return out

    def quantile_for_tail(self, tail: float, *, strict: bool = True) -> float:
        """The fitted quantile for a tail probability.

        The model is fitted at a fixed grid of quantiles. A requested tail that is not on
        the grid has to be approximated by the nearest fitted level, and **an interval
        labelled 90% that is really 89% is a miscalibrated interval with a misleading
        name** - the coverage tables would report a number the interval does not have.

        So this is strict by default: it raises rather than quietly moving the tail, and
        the runner's grid is built from exactly the tails the reported levels need, so a
        strict lookup always succeeds. ``strict=False`` is available for exploration, where
        a near-miss is acceptable because nothing is being published.

        Args:
            tail: Tail probability in ``(0, 1)``.
            strict: Whether an off-grid tail is an error.

        Returns:
            The fitted quantile to use.

        Raises:
            ValueError: If no quantile has been fitted, the tail is outside ``(0, 1)``, or
                the tail is off the fitted grid and ``strict`` is set.
        """
        if not self.quantiles:
            raise ValueError("no quantiles have been fitted")
        if not 0.0 < float(tail) < 1.0:
            raise ValueError(f"tail must lie in (0, 1), got {tail}")
        nearest = float(min(self.quantiles, key=lambda q: (abs(q - float(tail)), q)))
        # The tail and the fitted grid are both rounded to the same precision before the
        # comparison. ``(1 - 0.95) / 2`` is 0.025000000000000022 in binary floating point,
        # and an exact ``!=`` would reject a tail that is on the grid.
        if strict and round(nearest, 9) != round(float(tail), 9):
            achieved = 1.0 - 2.0 * min(float(tail), 1.0 - float(tail))
            raise ValueError(
                f"tail {float(tail)} is not on the fitted quantile grid "
                f"{list(self.quantiles)}. Snapping to {nearest} would build an interval "
                f"covering {achieved:.1%} of the residual distribution while the phase "
                f"reports it at a higher nominal level. Build the grid from "
                f"quantile_levels(nominal_levels) so every reported tail is exact, or pass "
                f"strict=False when exploring."
            )
        return nearest

    def to_dict(self) -> dict[str, Any]:
        return {
            "horizon": int(self.horizon),
            "quantiles": [float(q) for q in self.quantiles],
            "features": list(self.feature_names),
            "fit_seconds": round(float(self.fit_seconds), 3),
            "train_rows": int(self.train_rows),
            "target_units": self.target_units,
            "loss": "pinball (quantile) loss on the residual y - yhat_fixed",
            "target_representation": (
                "residual, not level: the point forecast is the Phase 7 fixed ensemble "
                "and is not modified, so only the interval method varies"
            ),
        }


def fit_quantile_model(
    *,
    features: np.ndarray,
    residual_kw: np.ndarray,
    feature_names: tuple[str, ...],
    horizon: int,
    quantiles: tuple[float, ...],
    seed: int,
    max_iter: int = 200,
    learning_rate: float = 0.06,
    max_leaf_nodes: int = 31,
    min_samples_leaf: int = 40,
    log: Any = None,
) -> QuantileResidualModel:
    """Fit residual quantiles by pinball loss, one estimator per quantile.

    Args:
        features: ``[N, n_features]`` origin-observable features.
        residual_kw: ``[N]`` **signed** residual ``y - yhat``, kW.
        feature_names: Column names, stored with the model.
        horizon: Horizon step, recorded.
        quantiles: Levels to fit, all in ``(0, 1)``.
        seed: Estimator seed.
        max_iter: Boosting iterations per quantile.
        learning_rate: Shrinkage.
        max_leaf_nodes: Tree size.
        min_samples_leaf: Regularisation.
        log: Optional progress callable.

    Returns:
        The fitted model.

    Raises:
        ValueError: On shape disagreement or an out-of-range quantile.
    """
    from sklearn.ensemble import HistGradientBoostingRegressor

    matrix = np.asarray(features, dtype=np.float64)
    target = np.asarray(residual_kw, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != len(feature_names):
        raise ValueError(
            f"features must be [rows, {len(feature_names)}], got {matrix.shape}"
        )
    if target.shape != (matrix.shape[0],):
        raise ValueError(
            f"residual_kw must be [rows] = {(matrix.shape[0],)}, got {target.shape}"
        )
    levels = tuple(sorted({float(q) for q in quantiles}))
    for level in levels:
        if not 0.0 < level < 1.0:
            raise ValueError(f"quantiles must lie in (0, 1), got {level}")

    started = time.perf_counter()
    models: dict[float, Any] = {}
    for level in levels:
        model = HistGradientBoostingRegressor(
            loss="quantile",
            quantile=level,
            max_iter=int(max_iter),
            learning_rate=float(learning_rate),
            max_leaf_nodes=int(max_leaf_nodes),
            min_samples_leaf=int(min_samples_leaf),
            early_stopping=False,
            random_state=int(seed),
        )
        model.fit(matrix, target)
        models[level] = model
        if log is not None:
            log(f"    quantile {level:.3f} for h={horizon} fitted")
    return QuantileResidualModel(
        models=models,
        quantiles=levels,
        feature_names=tuple(feature_names),
        horizon=int(horizon),
        fit_seconds=time.perf_counter() - started,
        train_rows=int(matrix.shape[0]),
    )