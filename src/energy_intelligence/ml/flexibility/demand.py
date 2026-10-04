"""Demand series and calendar slots: the primitives every Phase 9 estimator shares.

Three things live here, all of which exist because getting them wrong would leak the
future:

**The full per-step demand series.** The evaluation panel carries targets only at
``t+1``, ``t+4`` and ``t+96``, which is not enough to characterise how a load behaves
across a day. :func:`demand_series_kw` reconstructs ``y(s, t)`` for every step from the
per-unit ``values`` array and the row's rated kW. That reconstruction is not assumed
trustworthy: :func:`verify_demand_reconstruction` checks it against the targets the panel
*does* carry and refuses to proceed unless it reproduces them exactly, because a silently
mis-scaled baseline would make every envelope in the phase wrong in a way no coverage
check would catch.

**Calendar slots.** A same-time-of-day reference needs a slot key for each instant. The
convention is Phase 4's, deliberately reused rather than reinvented: absolute day index is
``t // 96`` and day-of-week is ``absolute_day % 7`` **with no offset**, so days 5 and 6 are
Saturday and Sunday. An offset here would silently shift the weekend by however many days
and no test would notice.

**Row selection by split.** Every estimator takes explicit row positions rather than
timestamps, and the split is decided once in :mod:`uncertainty.split`. Nothing in this
package decides what it may look at.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = [
    "STEPS_PER_DAY",
    "SLOT_GRANULARITIES",
    "DemandField",
    "calendar_slots",
    "demand_series_kw",
    "slot_index",
    "slot_labels",
    "verify_demand_reconstruction",
]

#: Steps per day on the project's native 15-minute grid. Fixed by
#: :class:`~energy_intelligence.domain.timebase.TimeBase`, restated here because every
#: slot function divides by it.
STEPS_PER_DAY = 96

#: Slot granularities the envelope estimator can be built at, coarsest first.
#:
#: The phase fits each on ``CAL_FIT`` and selects between them on ``CAL_CONF``, so this is
#: a declared search space rather than a hyper-parameter tuned against test.
SLOT_GRANULARITIES: tuple[str, ...] = ("horizon", "time_of_day", "day_type_time_of_day")


@dataclass(frozen=True, slots=True)
class DemandField:
    """The reconstructed demand series and the panel geometry that indexes it.

    Attributes:
        demand_kw: ``[n_series, n_steps]`` absolute demand in kW.
        rated_kw: ``[n_series]`` each series' rated power.
        series: ``[N]`` series index per panel row.
        origins: ``[N]`` forecast-origin step index per panel row.
        row_scale_kw: ``[N]`` rated kW per panel row, equal to ``rated_kw[series]``.
        horizons: Forecast horizons in native steps.
        timestep_minutes: Minutes per step.
    """

    demand_kw: np.ndarray
    rated_kw: np.ndarray
    series: np.ndarray
    origins: np.ndarray
    row_scale_kw: np.ndarray
    horizons: tuple[int, ...]
    timestep_minutes: int = 15

    @property
    def n_series(self) -> int:
        return int(self.demand_kw.shape[0])

    @property
    def n_steps(self) -> int:
        return int(self.demand_kw.shape[1])

    @property
    def n_rows(self) -> int:
        return int(self.origins.size)

    def target_step(self, row: int, horizon_steps: int) -> int:
        """The step whose realised value a row/horizon predicts."""
        return int(self.origins[int(row)]) + int(horizon_steps)

    def target_kw(self, row: int, horizon_steps: int) -> float:
        """Realised demand at the predicted step, in kW."""
        step = self.target_step(row, horizon_steps)
        if step < 0 or step >= self.n_steps:
            raise IndexError(
                f"row {row} at horizon {horizon_steps} needs step {step}, outside the "
                f"reconstructed series of {self.n_steps} steps"
            )
        return float(self.demand_kw[int(self.series[int(row)]), step])

    def origin_kw(self, row: int) -> float:
        """Demand at the forecast origin itself, in kW."""
        step = int(self.origins[int(row)])
        return float(self.demand_kw[int(self.series[int(row)]), step])

    def describe(self) -> dict[str, Any]:
        return {
            "n_series": self.n_series,
            "n_steps": self.n_steps,
            "n_rows": self.n_rows,
            "horizons": [int(h) for h in self.horizons],
            "timestep_minutes": int(self.timestep_minutes),
            "origin_range": [int(self.origins.min()), int(self.origins.max())],
            "rated_kw": {
                "min": float(self.rated_kw.min()),
                "median": float(np.median(self.rated_kw)),
                "max": float(self.rated_kw.max()),
            },
            "day_of_week_convention": (
                "absolute_day = step // 96; day_of_week = absolute_day % 7 with no offset, "
                "so 5 and 6 are Saturday and Sunday (Phase 4's convention)"
            ),
        }


def demand_series_kw(
    *,
    values: np.ndarray,
    series: np.ndarray,
    row_scale_kw: np.ndarray,
    origins: np.ndarray,
    horizons: tuple[int, ...],
    timestep_minutes: int = 15,
) -> DemandField:
    """Reconstruct absolute demand in kW for every step and series.

    Args:
        values: ``[n_series, n_steps]`` per-unit values, float32, as Phase 7 and 8 use.
        series: ``[N]`` series index per panel row.
        row_scale_kw: ``[N]`` rated kW per panel row.
        origins: ``[N]`` forecast-origin step index per panel row.
        horizons: Forecast horizons in native steps.
        timestep_minutes: Minutes per step, for the record.

    Returns:
        The :class:`DemandField`.

    Raises:
        ValueError: If shapes disagree, or a series' rated power is not constant across its
            rows - which would mean the per-unit-to-kW mapping is not a single scale
            factor and the reconstruction would be meaningless.
    """
    values = np.asarray(values, dtype=np.float64)
    series = np.asarray(series, dtype=np.int64)
    scale = np.asarray(row_scale_kw, dtype=np.float64)
    origins = np.asarray(origins, dtype=np.int64)
    if values.ndim != 2:
        raise ValueError(f"values must be [n_series, n_steps], got {values.shape}")
    n_series = values.shape[0]
    if series.size != origins.size:
        raise ValueError(f"series {series.shape} and origins {origins.shape} must agree")
    if scale.size != origins.size:
        raise ValueError(f"row_scale_kw {scale.shape} must match origins {origins.shape}")
    if series.size and (series.min() < 0 or series.max() >= n_series):
        raise ValueError(
            f"series indices must lie in [0, {n_series}), got "
            f"[{series.min()}, {series.max()}]"
        )
    if (scale <= 0).any():
        raise ValueError("rated kW must be positive")

    rated = np.zeros(n_series, dtype=np.float64)
    for index in range(n_series):
        rows = series == index
        if not rows.any():
            rated[index] = np.nan
            continue
        per_series = scale[rows]
        if np.ptp(per_series) > 0.0:
            raise ValueError(
                f"series {index} has {len(np.unique(per_series))} different rated kW values "
                f"({np.unique(per_series)[:5].tolist()}); a per-unit series maps to kW by a "
                f"single scale factor, so the reconstruction would be ill-defined"
            )
        rated[index] = float(per_series[0])
    if not np.isfinite(rated).all():
        missing = np.flatnonzero(~np.isfinite(rated)).tolist()
        raise ValueError(f"series {missing} have no panel row, so their rated kW is unknown")

    if not np.isfinite(values).all():
        raise ValueError(
            f"values contain {int((~np.isfinite(values)).sum())} non-finite entries; the "
            f"demand series cannot be reconstructed from them"
        )
    return DemandField(
        demand_kw=values * rated[:, None],
        rated_kw=rated,
        series=series,
        origins=origins,
        row_scale_kw=scale,
        horizons=tuple(int(h) for h in horizons),
        timestep_minutes=int(timestep_minutes),
    )


def verify_demand_reconstruction(
    field: DemandField, *, actual_kw: np.ndarray, tolerance: float = 0.0
) -> dict[str, Any]:
    """Check the reconstruction against the targets the panel carries.

    The panel stores ``y(t+h)`` for each horizon. If ``values * rated_kw`` did not
    reproduce those exactly, the full-step series would be on a different scale from the
    targets and every baseline and envelope built on it would be wrong by that factor -
    invisibly, because a uniformly scaled demand series still produces plausible coverage
    numbers.

    Args:
        field: The reconstructed field.
        actual_kw: ``[N, horizons]`` panel targets in kW.
        tolerance: Maximum absolute discrepancy tolerated, kW. Zero by default: the
            reconstruction is an identity, not an estimate, so anything above floating
            point noise means something is wrong.

    Returns:
        The measured discrepancy per horizon.

    Raises:
        ValueError: If shapes disagree, or any discrepancy exceeds ``tolerance``.
    """
    actual = np.asarray(actual_kw, dtype=np.float64)
    if actual.shape != (field.n_rows, len(field.horizons)):
        raise ValueError(
            f"actual_kw {actual.shape} does not match the field "
            f"({field.n_rows} rows, {len(field.horizons)} horizons)"
        )
    per_horizon: dict[str, float] = {}
    worst = 0.0
    for column, horizon in enumerate(field.horizons):
        rebuilt = np.asarray(
            [
                field.target_kw(row, horizon)
                for row in range(field.n_rows)
            ],
            dtype=np.float64,
        )
        gap = float(np.abs(rebuilt - actual[:, column]).max())
        per_horizon[str(horizon)] = gap
        worst = max(worst, gap)
    if worst > float(tolerance):
        raise ValueError(
            f"the reconstructed demand series disagrees with the panel targets by up to "
            f"{worst:.6g} kW (tolerance {float(tolerance):.6g}). Per horizon: "
            f"{per_horizon}. The per-unit series and the cached targets are not on the "
            f"same scale, so no envelope computed from them would be trustworthy"
        )
    return {
        "max_absolute_gap_kw": worst,
        "per_horizon_kw": per_horizon,
        "tolerance_kw": float(tolerance),
        "rows_checked": field.n_rows,
    }


def calendar_slots(steps: np.ndarray, granularity: str = "time_of_day") -> np.ndarray:
    """Map step indices to calendar slot indices.

    Args:
        steps: Step indices, any shape.
        granularity: One of :data:`SLOT_GRANULARITIES`.

    Returns:
        Non-negative slot indices, same shape as ``steps``.

    Raises:
        ValueError: If ``granularity`` is unknown, or a step is negative.
    """
    steps = np.asarray(steps, dtype=np.int64)
    if steps.size and steps.min() < 0:
        raise ValueError(f"step indices must be non-negative, got {steps.min()}")
    day = steps // STEPS_PER_DAY
    time_of_day = steps % STEPS_PER_DAY
    if granularity == "horizon":
        return np.zeros(steps.shape, dtype=np.int64)
    if granularity == "time_of_day":
        return time_of_day
    if granularity == "day_type_time_of_day":
        # Phase 4's weekend rule, unchanged: absolute_day % 7 >= 5 is a weekend.
        is_weekend = (day % 7) >= 5
        return is_weekend.astype(np.int64) * STEPS_PER_DAY + time_of_day
    raise ValueError(
        f"unknown granularity {granularity!r}; expected one of {list(SLOT_GRANULARITIES)}"
    )


def slot_labels(granularity: str) -> tuple[str, ...]:
    """Human-readable labels for each slot index at a granularity.

    Args:
        granularity: One of :data:`SLOT_GRANULARITIES`.

    Returns:
        One label per slot, in slot-index order.

    Raises:
        ValueError: If ``granularity`` is unknown.
    """
    if granularity == "horizon":
        return ("all_times",)
    if granularity == "time_of_day":
        return tuple(f"tod{index:02d}" for index in range(STEPS_PER_DAY))
    if granularity == "day_type_time_of_day":
        return tuple(
            f"{'weekend' if index >= STEPS_PER_DAY else 'weekday'}_tod{index % STEPS_PER_DAY:02d}"
            for index in range(2 * STEPS_PER_DAY)
        )
    raise ValueError(
        f"unknown granularity {granularity!r}; expected one of {list(SLOT_GRANULARITIES)}"
    )


def slot_index(
    steps: np.ndarray, granularity: str, fallback: str | None = None
) -> np.ndarray:
    """Slot indices, falling back to a coarser granularity where needed.

    Envelope cells are thin: a per-series, per-``day_type_time_of_day`` cell over 23
    calibration days holds roughly 7 weekday-equivalent observations, and a quantile from
    seven points is noise. Rather than widen the slot or silently return ``NaN``, this maps
    the requested granularity onto the next coarser one, which is exactly the fallback the
    envelope reports.

    Args:
        steps: Step indices.
        granularity: Requested granularity.
        fallback: Granularity to use instead. Defaults to the next coarser entry of
            :data:`SLOT_GRANULARITIES`, and to ``"horizon"`` when nothing is coarser.

    Returns:
        Slot indices at the requested or fallback granularity.

    Raises:
        ValueError: If either granularity is unknown.
    """
    order = list(SLOT_GRANULARITIES)
    if granularity not in order:
        raise ValueError(
            f"unknown granularity {granularity!r}; expected one of {order}"
        )
    chosen = granularity
    if fallback is not None:
        if fallback not in order:
            raise ValueError(f"unknown fallback granularity {fallback!r}")
        if order.index(fallback) > order.index(granularity):
            raise ValueError(
                f"fallback {fallback!r} is finer than the requested {granularity!r}; a "
                f"fallback must be coarser, not finer"
            )
        chosen = fallback
    return calendar_slots(steps, chosen)