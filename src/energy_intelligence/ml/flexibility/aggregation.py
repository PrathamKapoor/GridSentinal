"""Aggregating customer-scale flexibility, and the trap in doing so.

The trap
    -------
    "Five customers each moved 4 kW last Tuesday, so together they offer 20 kW of
    demand response" is a non-sequitur, and it is the single most common way a flexibility
    number gets inflated before it reaches an optimiser. Summing *individual historical
    excursions* gives the width of the aggregate's own historical range only if the
    excursions happened together - and if they did, it says the aggregate *varied* by that
    much, not that anyone can *make* it vary by that much.

What this module does instead
    ---------------------------
    It measures both quantities and reports them side by side:

    ``sum_of_individual``
        The naive sum. Recorded because it is what a reader will otherwise assume, and
        because the gap between it and the truth is the point.

    ``aggregate_envelope``
        Directional quantiles of the **pooled aggregate deviation**, computed on the same
        calibration rows. This is what the aggregate has actually been observed to do.

    ``diversification_ratio``
        ``1 - aggregate / sum_of_individual``. Positive when customers move independently,
        which is what makes an aggregate figure smaller than the sum.

    ``mean_pairwise_correlation``
        Measured from the calibration rows, not assumed. It is the mechanism behind the
    ratio, and quoting it lets a reader check the arithmetic.

The honest label
    ---------------
    Every aggregate this module produces is ``basis=STATISTICAL_PROXY`` with
    ``authority=NOT_CONTROLLABLE`` and an explicit limitation. Diversification is a fact
    about historical correlation, not a resource. Correlation is also not stable - customers
    whose deviations co-move in winter need not co-move in summer, and a single measured
    coefficient over one calibration window is not a law.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .demand import DemandField, calendar_slots
from .envelope import EnvelopeFit

__all__ = ["AggregationResult", "aggregate_deviations", "pairwise_correlation"]


def aggregate_deviations(
    deviation_kw: np.ndarray, *, nominal_level: float = 0.90
) -> tuple[float, float]:
    """Directional magnitudes of a pooled aggregate deviation.

    Args:
        deviation_kw: ``[rows, members]`` per-member deviations from expected demand, kW.
            Rows are calibration rows in chronological order; members are the grouped
            assets, already aligned.
        nominal_level: Stated two-sided coverage.

    Returns:
        ``(upward_kw, downward_kw)``, non-negative.

    Raises:
        ValueError: If the input is not 2-D, has no members, or ``nominal_level`` is
            outside ``(0, 1)``.
    """
    deviation = np.asarray(deviation_kw, dtype=np.float64)
    if deviation.ndim != 2:
        raise ValueError(f"deviation_kw must be [rows, members], got {deviation.shape}")
    if deviation.shape[1] == 0:
        raise ValueError("an aggregate needs at least one member")
    if not 0.0 < float(nominal_level) < 1.0:
        raise ValueError(f"nominal_level must lie in (0, 1), got {nominal_level}")
    total = deviation.sum(axis=1)
    alpha = 1.0 - float(nominal_level)
    return (
        max(0.0, float(np.quantile(total, 1.0 - alpha))),
        max(0.0, float(-np.quantile(total, alpha))),
    )


def pairwise_correlation(deviation_kw: np.ndarray) -> dict[str, Any]:
    """Mean and median pairwise correlation between members, measured.

    Args:
        deviation_kw: ``[rows, members]`` per-member deviations.

    Returns:
        ``mean``, ``median``, ``p10``, ``p90``, ``n_pairs``, and
        ``share_positive``. Returns ``None`` statistics when fewer than two members have
        usable variation.

    Raises:
        ValueError: If the input is not 2-D.
    """
    deviation = np.asarray(deviation_kw, dtype=np.float64)
    if deviation.ndim != 2:
        raise ValueError(f"deviation_kw must be [rows, members], got {deviation.shape}")
    n_members = deviation.shape[1]
    if n_members < 2:
        return {
            "mean": None,
            "median": None,
            "p10": None,
            "p90": None,
            "n_pairs": 0,
            "share_positive": None,
            "note": "fewer than two members, so no pair exists",
        }
    usable = deviation.std(axis=0) > 0
    if int(usable.sum()) < 2:
        return {
            "mean": None,
            "median": None,
            "p10": None,
            "p90": None,
            "n_pairs": 0,
            "share_positive": None,
            "note": "fewer than two members vary, so no correlation is defined",
        }
    columns = deviation[:, usable]
    with np.errstate(invalid="ignore", divide="ignore"):
        matrix = np.corrcoef(columns, rowvar=False)
    values = matrix[np.triu_indices_from(matrix, k=1)]
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            "mean": None,
            "median": None,
            "p10": None,
            "p90": None,
            "n_pairs": 0,
            "share_positive": None,
            "note": "no finite pairwise correlation",
        }
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p10": float(np.quantile(values, 0.10)),
        "p90": float(np.quantile(values, 0.90)),
        "n_pairs": int(values.size),
        "share_positive": float(np.mean(values > 0)),
    }


@dataclass(frozen=True, slots=True)
class AggregationResult:
    """Aggregate flexibility, with the naive sum reported beside it.

    Attributes:
        target: The aggregate's name.
        n_members: How many assets were aggregated.
        upward_kw: Aggregate upward magnitude, kW.
        downward_kw: Aggregate downward magnitude, kW.
        sum_individual_upward_kw: Naive sum of member upward magnitudes, kW.
        sum_individual_downward_kw: Naive sum of member downward magnitudes, kW.
        diversification_upward: ``1 - upward / sum_individual_upward``.
        diversification_downward: ``1 - downward / sum_individual_downward``.
        correlation: Measured pairwise correlation statistics.
        coverage_measured: Two-sided coverage the aggregate envelope actually achieved on the
            evaluation rows, or ``None`` when not evaluated.
    """

    target: str
    n_members: int
    upward_kw: float
    downward_kw: float
    sum_individual_upward_kw: float
    sum_individual_downward_kw: float
    diversification_upward: float | None
    diversification_downward: float | None
    correlation: dict[str, Any]
    coverage_measured: float | None = None

    def describe(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "n_members": int(self.n_members),
            "upward_kw": float(self.upward_kw),
            "downward_kw": float(self.downward_kw),
            "sum_individual_upward_kw": float(self.sum_individual_upward_kw),
            "sum_individual_downward_kw": float(self.sum_individual_downward_kw),
            "diversification_upward": self.diversification_upward,
            "diversification_downward": self.diversification_downward,
            "correlation": self.correlation,
            "coverage_measured": self.coverage_measured,
            "basis": "statistical_proxy",
            "authority": "not_controllable",
            "warning": (
                "an aggregate of historical variation is NOT a dispatchable resource. "
                "Diversification is measured historical correlation, not a control "
                "capability, and correlation over one calibration window is not a law."
            ),
        }


def aggregate_envelope(
    *,
    target: str,
    field: DemandField,
    envelope: EnvelopeFit,
    calibration_rows: np.ndarray,
    horizon_steps: int,
    nominal_level: float,
) -> AggregationResult:
    """Aggregate every series in the panel into one group-level envelope.

    Args:
        target: Name for the aggregate.
        field: The reconstructed demand field.
        envelope: The fitted per-series envelope, used for the individual sum.
        calibration_rows: Panel row positions on ``CAL_FIT``.
        horizon_steps: Forecast horizon.
        nominal_level: Stated two-sided coverage.

    Returns:
        The aggregate, with the naive sum beside it.

    Raises:
        ValueError: If no calibration rows are supplied.
    """
    from .baselines import evaluate_baseline

    calibration_rows = np.asarray(calibration_rows, dtype=np.int64)
    if calibration_rows.size == 0:
        raise ValueError("an aggregate envelope needs calibration rows")
    expected = evaluate_baseline(
        envelope.baseline,
        field,
        rows=calibration_rows,
        horizon_steps=int(horizon_steps),
        granularity=envelope.granularity,
    )
    realised = np.asarray(
        [field.target_kw(int(row), int(horizon_steps)) for row in calibration_rows],
        dtype=np.float64,
    )
    delta = realised - expected

    series = field.series[calibration_rows]
    # [rows, members] by transposing the per-member series: each column is one asset.
    by_member = np.zeros((calibration_rows.size, field.n_series), dtype=np.float64)
    for index in range(field.n_series):
        by_member[:, index] = np.where(series == index, delta, 0.0)

    up, down = aggregate_deviations(by_member, nominal_level=nominal_level)
    correlation = pairwise_correlation(by_member)

    target_steps = field.origins[calibration_rows] + int(horizon_steps)
    slots = calendar_slots(target_steps, envelope.granularity)
    try:
        column = int(envelope.horizons.index(int(horizon_steps)))
    except ValueError as exc:
        raise ValueError(
            f"horizon {horizon_steps} was not fitted; this envelope covers "
            f"{list(envelope.horizons)}"
        ) from exc
    fallback = envelope.fallback_magnitude_kw(column)
    member_up = np.empty(field.n_series, dtype=np.float64)
    member_down = np.empty(field.n_series, dtype=np.float64)
    for index in range(field.n_series):
        rows_of = series == index
        if not rows_of.any():
            member_up[index] = fallback
            member_down[index] = fallback
            continue
        # The dominant slot for this member, so the naive sum is built from magnitudes that
        # actually apply to the member rather than from an arbitrary cell.
        position = int(np.bincount(slots[rows_of]).argmax())
        member_up[index], member_down[index] = envelope.magnitudes(column, index, position)

    sum_up = float(member_up.sum())
    sum_down = float(member_down.sum())
    return AggregationResult(
        target=target,
        n_members=field.n_series,
        upward_kw=up,
        downward_kw=down,
        sum_individual_upward_kw=sum_up,
        sum_individual_downward_kw=sum_down,
        diversification_upward=(1.0 - up / sum_up) if sum_up > 0 else None,
        diversification_downward=(1.0 - down / sum_down) if sum_down > 0 else None,
        correlation=correlation,
    )