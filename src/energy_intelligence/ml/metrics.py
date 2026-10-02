"""Metrics: generic error measures plus the energy-specific ones that matter here.

Generic metrics
---------------
``MAE``, ``RMSE``, ``sMAPE`` and ``R^2``. Two deliberate choices:

* **No MAPE.** Half of the PV target is exactly zero (18,304 of 35,040 steps). MAPE
  divides by the actual value, so it is undefined at every night-time step and
  explodes near dawn and dusk. Any percentage error reported here is symmetric and
  zero-safe: see :func:`smape`.
* **MAE is the headline.** Energy operators care about absolute error in kW,
  because that is what a capacity decision is made against. RMSE is reported
  alongside because it is the sensitive one: a handful of bad peak steps dominate
  it, which is information rather than noise.

Energy-specific metrics
-----------------------
Average error hides the conditions that matter, so these are computed too:

* **peak_error_mae** - error restricted to the top decile of actual values.
* **ramp_error_mae** - error in the *change* between consecutive horizon steps,
  ``|(pred[t+h+1]-pred[t+h]) - (actual[t+h+1]-actual[t+h])|``. Ramping is the
  operationally dangerous part of both load and PV.
* **zero_period_error_mae** - error where actual generation is exactly zero
  (PV night). A model that predicts 40 kW at midnight is wrong in a way average
  metrics hide completely.
* **daytime_error_mae** - error where actual generation is above zero.

Every metric reports ``n`` alongside its value, so a metric computed over 37
samples is never mistaken for one computed over 10,000.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "MetricSet",
    "mae",
    "rmse",
    "smape",
    "r2_score",
    "peak_error_mae",
    "zero_period_error_mae",
    "daytime_error_mae",
    "ramp_error_mae",
    "evaluate",
]


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean absolute error in the target's own unit."""
    actual = np.asarray(actual, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    if actual.shape != predicted.shape:
        raise ValueError(f"shape mismatch: {actual.shape} vs {predicted.shape}")
    if actual.size == 0:
        return float("nan")
    return float(np.mean(np.abs(actual - predicted)))


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Root mean squared error, in the target's own unit."""
    actual = np.asarray(actual, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    if actual.shape != predicted.shape:
        raise ValueError(f"shape mismatch: {actual.shape} vs {predicted.shape}")
    if actual.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(np.square(actual - predicted))))


def smape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Symmetric mean absolute percentage error, in percent, zero-safe.

    ``0 / 0`` is defined as ``0`` because a zero actual and a zero forecast is a
    perfect prediction, and dividing would report ``NaN`` for the majority of a
    solar night. Values are reported in ``[0, 200]``.
    """
    actual = np.asarray(actual, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    if actual.shape != predicted.shape:
        raise ValueError(f"shape mismatch: {actual.shape} vs {predicted.shape}")
    if actual.size == 0:
        return float("nan")
    denominator = (np.abs(actual) + np.abs(predicted)) / 2.0
    ratio = np.zeros_like(denominator)
    nonzero = denominator > 0
    ratio[nonzero] = np.abs(actual[nonzero] - predicted[nonzero]) / denominator[nonzero]
    return float(100.0 * np.mean(ratio))


def r2_score(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Coefficient of determination against the mean of ``actual``.

    Returns ``nan`` when ``actual`` is constant: there is no variance to explain,
    so the score is undefined rather than perfect or terrible.
    """
    actual = np.asarray(actual, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    if actual.shape != predicted.shape:
        raise ValueError(f"shape mismatch: {actual.shape} vs {predicted.shape}")
    if actual.size == 0:
        return float("nan")
    variance = float(np.var(actual))
    if variance <= 1e-15:
        return float("nan")
    return float(1.0 - np.sum(np.square(actual - predicted)) / np.sum(np.square(actual - actual.mean())))


def peak_error_mae(
    actual: np.ndarray, predicted: np.ndarray, *, quantile: float = 0.90
) -> float | None:
    """MAE over the top ``quantile`` of actual values, or ``None`` if undefined."""
    actual = np.asarray(actual, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    if actual.size == 0:
        return None
    threshold = float(np.quantile(actual, quantile))
    mask = actual >= threshold
    if not np.any(mask):
        return None
    return mae(actual[mask], predicted[mask])


def zero_period_error_mae(actual: np.ndarray, predicted: np.ndarray) -> float | None:
    """MAE where the actual value is exactly zero, or ``None`` if there are none."""
    actual = np.asarray(actual, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    mask = actual == 0.0
    if not np.any(mask):
        return None
    return mae(actual[mask], predicted[mask])


def daytime_error_mae(actual: np.ndarray, predicted: np.ndarray) -> float | None:
    """MAE where the actual value is above zero, or ``None`` if there are none."""
    actual = np.asarray(actual, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    mask = actual > 0.0
    if not np.any(mask):
        return None
    return mae(actual[mask], predicted[mask])


def ramp_error_mae(
    actual: np.ndarray, predicted: np.ndarray, *, axis: int = -1
) -> float | None:
    """MAE of the step-to-step change, or ``None`` when no consecutive pair exists.

    Args:
        actual: Shape ``(n_samples, n_steps)`` for a multi-step horizon.
        predicted: Same shape.
        axis: Axis holding consecutive horizon steps.
    """
    actual = np.asarray(actual, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    if actual.ndim != 2 or actual.shape[1] < 2:
        return None
    actual_ramp = np.diff(actual, axis=axis)
    predicted_ramp = np.diff(predicted, axis=axis)
    return mae(actual_ramp.ravel(), predicted_ramp.ravel())


@dataclass(frozen=True, slots=True)
class MetricSet:
    """Every metric for one evaluation, with the sample count that produced it."""

    n: int
    mae: float
    rmse: float
    smape: float
    r2: float
    peak_error_mae: float | None = None
    zero_period_error_mae: float | None = None
    daytime_error_mae: float | None = None
    ramp_error_mae: float | None = None
    mean_actual: float | None = None
    mean_predicted: float | None = None
    extra: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "n": self.n,
            "mae": self.mae,
            "rmse": self.rmse,
            "smape": self.smape,
            "r2": self.r2,
            "peak_error_mae": self.peak_error_mae,
            "zero_period_error_mae": self.zero_period_error_mae,
            "daytime_error_mae": self.daytime_error_mae,
            "ramp_error_mae": self.ramp_error_mae,
            "mean_actual": self.mean_actual,
            "mean_predicted": self.mean_predicted,
            **self.extra,
        }

    def render_row(self) -> str:
        """One line for the comparison table."""

        def fmt(value: float | None, digits: int = 3) -> str:
            if value is None or value != value:
                return "n/a"
            return f"{value:.{digits}f}"

        return (
            f"n={self.n:<8d} MAE={fmt(self.mae):>10s} RMSE={fmt(self.rmse):>10s} "
            f"sMAPE={fmt(self.smape, 2):>7s}% R2={fmt(self.r2):>7s} "
            f"peak={fmt(self.peak_error_mae):>9s} ramp={fmt(self.ramp_error_mae):>9s}"
        )


def evaluate(
    actual: np.ndarray,
    predicted: np.ndarray,
    *,
    ramp_actual: np.ndarray | None = None,
    ramp_predicted: np.ndarray | None = None,
) -> MetricSet:
    """Compute the full metric set.

    Args:
        actual: ``(n_samples,)`` or ``(n_samples, n_steps)`` truth.
        predicted: Same shape.
        ramp_actual: Optional ``(n_samples, n_steps)`` multi-step arrays, used for
            the ramp metric. When omitted and ``actual`` is 2-D, ``actual`` itself
            is used.
        ramp_predicted: Companion to ``ramp_actual``.
    """
    actual = np.asarray(actual, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    flat_actual = actual.reshape(-1)
    flat_predicted = predicted.reshape(-1)

    if ramp_actual is None and actual.ndim == 2:
        ramp_actual = actual
    if ramp_predicted is None and predicted.ndim == 2:
        ramp_predicted = predicted

    return MetricSet(
        n=int(flat_actual.size),
        mae=mae(flat_actual, flat_predicted),
        rmse=rmse(flat_actual, flat_predicted),
        smape=smape(flat_actual, flat_predicted),
        r2=r2_score(flat_actual, flat_predicted),
        peak_error_mae=peak_error_mae(flat_actual, flat_predicted),
        zero_period_error_mae=zero_period_error_mae(flat_actual, flat_predicted),
        daytime_error_mae=daytime_error_mae(flat_actual, flat_predicted),
        ramp_error_mae=(
            ramp_error_mae(np.asarray(ramp_actual), np.asarray(ramp_predicted))
            if ramp_actual is not None and ramp_predicted is not None
            else None
        ),
        mean_actual=float(flat_actual.mean()) if flat_actual.size else None,
        mean_predicted=float(flat_predicted.mean()) if flat_actual.size else None,
    )