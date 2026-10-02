"""Where the baselines fail, and whether the data contains real regimes.

Two analyses, both driven by the measured residuals rather than by assumption.

Regime analysis
---------------
The future Energy-MoE will assume that some conditions are hard enough to deserve a
specialist. This module tests that assumption **on the data** instead of designing
the specialists first. Regimes here are cheap, objective, and derived from the
target itself:

* ``load_level`` - terciles of the actual per-unit demand.
* ``load_ramp`` - terciles of the absolute step-to-step change, because ramping is
  the operationally dangerous part of load.
* ``solar_phase`` - night / morning-ramp / midday / evening-ramp, derived from the
  sign and size of the generation derivative.
* ``solar_level`` - terciles of positive generation, separating a clear midday from
  a cloudy one.

If per-regime error differs by a large factor, regime structure exists and is worth
studying in Phase 5. If it does not, that is equally worth knowing, and it is
recorded just as plainly. Nothing here creates experts.

Error analysis
--------------
Average metrics hide the cases that matter, so this reports the worst individual
rows, the worst series, and the error profile by hour of day and by month - the
shape an operator would recognise.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "RegimeDefinition",
    "assign_regimes",
    "regime_report",
    "error_analysis",
    "REGIME_LABELS",
]

#: Regime labels used in reports. Kept as constants so a report and a test agree.
REGIME_LABELS: tuple[str, ...] = (
    "low",
    "typical",
    "high",
    "night",
    "morning_ramp",
    "midday",
    "evening_ramp",
)


@dataclass(frozen=True, slots=True)
class RegimeDefinition:
    """One way of slicing the data into regimes.

    Attributes:
        name: Regime-axis name, e.g. ``"load_level"``.
        description: What the axis means and how it is derived.
        labels: Regime label per row.
        detail: Optional per-row scalar the label was derived from.
    """

    name: str
    description: str
    labels: np.ndarray
    detail: np.ndarray | None = None

    @property
    def labels_present(self) -> tuple[str, ...]:
        seen = {str(label) for label in self.labels}
        return tuple(label for label in REGIME_LABELS if label in seen) + tuple(
            sorted(seen - set(REGIME_LABELS))
        )


def _terciles(values: np.ndarray) -> tuple[np.ndarray, tuple[float, float]]:
    """Label rows low/typical/high by tercile of ``values``."""
    low, high = np.quantile(values, [1.0 / 3.0, 2.0 / 3.0])
    labels = np.where(values <= low, "low", np.where(values <= high, "typical", "high"))
    return labels.astype(object), (float(low), float(high))


def assign_regimes(
    actual: np.ndarray,
    *,
    target_id: str,
    origin_index: np.ndarray,
    previous_actual: np.ndarray | None = None,
) -> tuple[RegimeDefinition, ...]:
    """Derive the regime axes that apply to a target.

    Args:
        actual: Target values in the dataset's stored units.
        target_id: Which declared target, which selects the applicable axes.
        origin_index: Forecast origin per row, for the hour/month profile.
        previous_actual: The value one step earlier, when available, so a ramp can
            be measured rather than inferred from consecutive rows that are four
            steps apart.

    Returns:
        The regime axes for this target.
    """
    actual = np.asarray(actual, dtype=np.float64)
    axes: list[RegimeDefinition] = []

    if target_id in {"customer_load", "feeder_load"}:
        labels, edges = _terciles(actual)
        axes.append(
            RegimeDefinition(
                name="load_level",
                description=(
                    "terciles of actual per-unit demand: "
                    f"low <= {edges[0]:.4f}, typical <= {edges[1]:.4f}, high above"
                ),
                labels=labels,
                detail=actual,
            )
        )
        ramp = (
            np.abs(actual - np.asarray(previous_actual, dtype=np.float64))
            if previous_actual is not None
            else np.zeros_like(actual)
        )
        if previous_actual is not None and np.any(ramp > 0):
            ramp_labels, ramp_edges = _terciles(ramp)
            axes.append(
                RegimeDefinition(
                    name="load_ramp",
                    description=(
                        "terciles of |actual(t) - actual(t-1)| in stored units: "
                        f"low <= {ramp_edges[0]:.5f}, typical <= {ramp_edges[1]:.5f}"
                    ),
                    labels=ramp_labels,
                    detail=ramp,
                )
            )
    else:
        ramp = (
            actual - np.asarray(previous_actual, dtype=np.float64)
            if previous_actual is not None
            else np.zeros_like(actual)
        )
        phase = np.full(actual.shape, "night", dtype=object)
        if previous_actual is not None:
            rising = ramp > 1e-9
            falling = ramp < -1e-9
            generating = actual > 0
            phase[generating & rising] = "morning_ramp"
            phase[generating & falling] = "evening_ramp"
            phase[generating & ~rising & ~falling] = "midday"
        axes.append(
            RegimeDefinition(
                name="solar_phase",
                description=(
                    "night = zero generation; morning/evening ramp = generation "
                    "rising/falling; midday = generating and steady"
                ),
                labels=phase,
                detail=ramp,
            )
        )
        positive = actual[actual > 0]
        if positive.size:
            labels, edges = _terciles(positive)
            # ``labels`` was computed on the positive subset only, so it must be
            # indexed by the positions of those same rows - not by a boolean mask
            # over the full array, which would be a different length.
            mapped = np.full(actual.shape, "night", dtype=object)
            positions = np.flatnonzero(actual > 0)
            mapped[positions] = labels
            axes.append(
                RegimeDefinition(
                    name="solar_level",
                    description=(
                        "terciles of positive generation (clear vs cloudy midday): "
                        f"low <= {edges[0]:.3f}, typical <= {edges[1]:.3f} kW"
                    ),
                    labels=mapped,
                    detail=actual,
                )
            )
    return tuple(axes)


def regime_report(
    axes: tuple[RegimeDefinition, ...],
    errors: dict[str, np.ndarray],
) -> dict[str, object]:
    """Per-regime error for every model, plus how much the regimes differ.

    Args:
        axes: Regime definitions from :func:`assign_regimes`.
        errors: ``{model_name: absolute_error_per_row}``.

    Returns:
        A report with per-axis, per-regime counts and MAEs, and a
        ``max_regime_mae_ratio`` per model: the largest ratio between the worst and
        best regime. A ratio far from 1.0 means the regime split explains real error
        structure; a ratio near 1.0 means it does not, and that is a finding too.
    """
    report: dict[str, object] = {}
    for axis in axes:
        per_axis: dict[str, object] = {}
        for label in axis.labels_present:
            mask = axis.labels == label
            entry: dict[str, object] = {
                "n": int(np.count_nonzero(mask)),
                "share_of_samples": float(np.mean(mask)) if mask.size else 0.0,
            }
            for model, error in errors.items():
                entry[model] = (
                    float(np.mean(error[mask])) if np.any(mask) else None
                )
            per_axis[label] = entry
        report[axis.name] = {
            "description": axis.description,
            "regimes": per_axis,
        }

    ratios: dict[str, float | None] = {}
    for model, error in errors.items():
        worst, best = None, None
        for axis in axes:
            for label in axis.labels_present:
                mask = axis.labels == label
                if not np.any(mask):
                    continue
                value = float(np.mean(error[mask]))
                worst = value if worst is None else max(worst, value)
                best = value if best is None else min(best, value)
        if worst is None or best is None or best <= 0:
            ratios[model] = None
        else:
            ratios[model] = worst / best
    report["max_regime_mae_ratio"] = ratios
    report["interpretation"] = (
        "a large max_regime_mae_ratio means the error is concentrated in specific "
        "operating conditions, which is the premise a regime-specialised model "
        "would rest on; a ratio near 1.0 means the split explains little and such a "
        "model would be architecture for its own sake"
    )
    return report


def error_analysis(
    *,
    actual: np.ndarray,
    predicted: np.ndarray,
    model: str,
    origin_index: np.ndarray,
    series_ids: tuple[str, ...],
    series_index: np.ndarray,
    timestep_minutes: int,
    origin_iso: str,
    top_rows: int = 20,
    quality_flags: tuple[str, ...] = (),
) -> dict[str, object]:
    """Describe where one model's error is concentrated.

    Returns:
        Worst individual rows with their timestamps, worst series, and the error
        profile by hour of day and by month.
    """
    from datetime import datetime, timedelta

    actual = np.asarray(actual, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    error = np.abs(actual - predicted)
    start = datetime.fromisoformat(origin_iso)
    steps_per_day = 24 * 60 // max(1, timestep_minutes)

    origins = np.asarray(origin_index, dtype=np.int64)
    hours = (origins % steps_per_day) * timestep_minutes // 60
    days = origins // steps_per_day
    months = (
        (start + timedelta(days=int(day))).month for day in days
    )
    month_array = np.asarray([m for m in months], dtype=np.int64)

    order = np.argsort(-error)[: min(top_rows, error.size)]
    worst_rows = [
        {
            "timestamp": (start + timedelta(minutes=int(origins[i]) * timestep_minutes)).isoformat(),
            "series_id": series_ids[int(series_index[i])],
            "actual": float(actual[i]),
            "predicted": float(predicted[i]),
            "absolute_error": float(error[i]),
        }
        for i in order
    ]

    per_series: list[tuple[str, float, int]] = []
    for index, series_id in enumerate(series_ids):
        mask = series_index == index
        if not np.any(mask):
            continue
        per_series.append((series_id, float(np.mean(error[mask])), int(np.count_nonzero(mask))))
    per_series.sort(key=lambda item: item[1], reverse=True)

    return {
        "model": model,
        "n": int(error.size),
        "mean_absolute_error": float(error.mean()),
        "worst_rows": worst_rows,
        "worst_series": [
            {"series_id": sid, "mae": value, "n": count} for sid, value, count in per_series[:10]
        ],
        "error_by_hour_of_day": _profile(error, hours, 24),
        "error_by_month": _profile(error, month_array, 12),
        "data_quality_flags": list(quality_flags),
    }


def _profile(error: np.ndarray, groups: np.ndarray, size: int) -> list[dict[str, object]]:
    """Mean error per group value, 1-based, with counts."""
    out: list[dict[str, object]] = []
    for value in range(1, size + 1):
        mask = groups == value
        if not np.any(mask):
            continue
        out.append({"group": value, "n": int(np.count_nonzero(mask)), "mae": float(np.mean(error[mask]))})
    return out