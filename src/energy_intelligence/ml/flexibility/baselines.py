"""Expected-demand baselines: the reference a behavioural envelope is measured against.

A flexibility envelope needs a reference. "This load has moved 4 kW" is only meaningful
relative to *what it was expected to do*, and the choice of expectation changes the answer
substantially, so two are implemented and **selected between on held-out data** rather
than one being assumed.

``calendar``
    A same-time-of-day central value: the median realised demand at the same
    day-type and time-of-day slot over the calibration split. This is the reference that
    isolates genuine behavioural deviation - a load that is unusually high or low for that
    moment of the week.

``persistence``
    Demand at the forecast origin, carried forward. Phase 4 measured this as the strongest
    baseline at h=1 on this dataset (0.4043 kW against the tree's 0.4242), so it is not a
    strawman: it is what a practitioner would actually reach for, and it needs no fitting
    at all.

Why the difference matters
    ``calendar`` removes the predictable daily shape from the deviation, so what remains is
    harder to forecast and *behavioural*. ``persistence`` at h=1 makes the deviation
    identical to the forecast error, so what remains is *epistemic*. Phase 9 measures which
    reference gives the better-calibrated, narrower envelope and reports both; it does not
    assume the conceptually prettier one wins.

Leakage discipline
    Every statistic here is fitted from **explicitly supplied calibration rows** and never
    from the rows it is applied to. There is no default that looks at "everything", because
    the one place a default could reach the evaluation split is the one place it must not.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .demand import DemandField, calendar_slots, slot_labels

__all__ = [
    "BASELINE_METHODS",
    "FALLBACK_BASELINE",
    "BaselineFit",
    "CalendarBaseline",
    "PersistenceBaseline",
    "evaluate_baseline",
    "fit_baseline",
]

#: Every reference this phase can build. Fitted on ``CAL_FIT``, selected on ``CAL_CONF``.
BASELINE_METHODS: tuple[str, ...] = ("calendar", "persistence")

#: Used when a slot cell holds fewer observations than the configured minimum. A single
#: series-wide median is coarse but honest; a quantile from three points is not.
FALLBACK_BASELINE = "series_median"


@dataclass(frozen=True, slots=True)
class BaselineFit:
    """A fitted expected-demand reference.

    Attributes:
        method: Which baseline this is.
        granularity: Slot granularity the table is keyed by.
        table: ``[n_series, n_slots]`` expected value in kW.
        counts: ``[n_series, n_slots]`` observations behind each cell.
        series_median_kw: ``[n_series]`` fallback when a cell is too thin.
        slots_used: ``[n_series]`` how many cells were populated rather than fallen back.
        min_samples: The minimum observations a cell needed to be used.
        fit_rows: How many calibration rows were consulted.
    """

    method: str
    granularity: str
    table: np.ndarray
    counts: np.ndarray
    series_median_kw: np.ndarray
    slots_used: np.ndarray
    min_samples: int
    fit_rows: int

    def value(self, series_index: int, slot: int) -> float:
        """Expected demand for one series and slot, kW, falling back when the cell is thin."""
        index = int(series_index)
        position = int(slot)
        if 0 <= position < self.table.shape[1] and self.counts[index, position] >= self.min_samples:
            return float(self.table[index, position])
        return float(self.series_median_kw[index])

    @property
    def fallback_rate(self) -> float:
        """Share of cells that fell back to the series median."""
        total = int(self.counts.size)
        return float((self.counts < self.min_samples).sum()) / total if total else 0.0

    def describe(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "granularity": self.granularity,
            "n_slots": int(self.table.shape[1]),
            "min_samples": int(self.min_samples),
            "fit_rows": int(self.fit_rows),
            "fallback_cell_rate": self.fallback_rate,
            "table_mean_kw": float(self.table[self.counts >= self.min_samples].mean())
            if (self.counts >= self.min_samples).any()
            else None,
        }


class CalendarBaseline:
    """Same-time-of-day central value, fitted per series and calendar slot."""

    name = "calendar"

    @staticmethod
    def fit(
        field: DemandField,
        *,
        fit_steps: np.ndarray,
        series_rows: np.ndarray,
        granularity: str = "day_type_time_of_day",
        min_samples: int = 5,
    ) -> BaselineFit:
        """Fit the table from calibration steps only.

        Args:
            field: The reconstructed demand field.
            fit_steps: ``[M]`` **target step indices** on which the reference is fitted.
                These must come from the calibration split; the caller is responsible and
                nothing here can tell a calibration step from an evaluation one.
            series_rows: ``[N]`` panel row positions whose series contribute. Used so the
                table covers exactly the series the panel contains.
            granularity: Slot granularity.
            min_samples: Observations a cell needs before it is trusted.

        Returns:
            The fitted table.

        Raises:
            ValueError: If ``fit_steps`` is empty or ``min_samples`` is below 1.
        """
        fit_steps = np.asarray(fit_steps, dtype=np.int64)
        series_rows = np.asarray(series_rows, dtype=np.int64)
        if fit_steps.size == 0:
            raise ValueError("a calendar baseline needs at least one calibration step")
        if int(min_samples) < 1:
            raise ValueError(f"min_samples must be at least 1, got {min_samples}")
        if fit_steps.min() < 0 or fit_steps.max() >= field.n_steps:
            raise ValueError(
                f"fit_steps must lie in [0, {field.n_steps}), got "
                f"[{fit_steps.min()}, {fit_steps.max()}]"
            )

        slots = calendar_slots(fit_steps, granularity)
        n_slots = len(slot_labels(granularity))
        n_series = field.n_series
        table = np.zeros((n_series, n_slots), dtype=np.float64)
        counts = np.zeros((n_series, n_slots), dtype=np.int64)

        present = np.unique(series_rows)
        for index in present:
            values = field.demand_kw[int(index), fit_steps]
            for slot in np.unique(slots):
                selected = values[slots == slot]
                position = int(slot)
                table[int(index), position] = float(np.median(selected))
                counts[int(index), position] = int(selected.size)
        # The fallback median is computed over the **calibration steps only**. Taking it
        # over the whole series would let an evaluation period's own values define the
        # number a thin cell falls back to, which is leakage through the back door.
        series_median = np.array(
            [
                float(np.median(field.demand_kw[int(index), fit_steps]))
                if int(index) in set(present.tolist())
                else np.nan
                for index in range(n_series)
            ],
            dtype=np.float64,
        )
        if not np.isfinite(series_median).all():
            missing = np.flatnonzero(~np.isfinite(series_median)).tolist()
            raise ValueError(
                f"series {missing} have no calibration row, so even the fallback median is "
                f"undefined for them"
            )
        return BaselineFit(
            method=CalendarBaseline.name,
            granularity=granularity,
            table=table,
            counts=counts,
            series_median_kw=series_median,
            slots_used=(counts >= int(min_samples)).sum(axis=1).astype(np.int64),
            min_samples=int(min_samples),
            fit_rows=int(fit_steps.size),
        )


class PersistenceBaseline:
    """Demand at the forecast origin, carried forward. No fitted state at all."""

    name = "persistence"

    @staticmethod
    def fit(
        field: DemandField,
        *,
        fit_rows: np.ndarray,
        granularity: str = "horizon",
        min_samples: int = 1,
    ) -> BaselineFit:
        """Record the no-fit case.

        Persistence has nothing to estimate, so this exists only so both baselines share one
        interface and the runner can treat them identically. ``fit_rows`` is retained for
        the record: it documents which rows the method was *declared* on, which for
        persistence is every row because it depends on no statistic.

        Args:
            field: The reconstructed demand field.
            fit_rows: Calibration row positions, recorded not used.
            granularity: Recorded; persistence is horizon-independent.
            min_samples: Recorded; unused.

        Returns:
            A fit carrying no table. Read :attr:`PersistenceBaseline.name` and use
            :meth:`evaluate` rather than :meth:`BaselineFit.value`.
        """
        fit_rows = np.asarray(fit_rows, dtype=np.int64)
        return BaselineFit(
            method=PersistenceBaseline.name,
            granularity="horizon",
            table=np.zeros((field.n_series, 1), dtype=np.float64),
            counts=np.zeros((field.n_series, 1), dtype=np.int64),
            series_median_kw=np.zeros(field.n_series, dtype=np.float64),
            slots_used=np.zeros(field.n_series, dtype=np.int64),
            min_samples=int(min_samples),
            fit_rows=int(fit_rows.size),
        )

    @staticmethod
    def evaluate(field: DemandField, rows: np.ndarray) -> np.ndarray:
        """The expected value for each row, in kW.

        Args:
            field: The reconstructed demand field.
            rows: ``[M]`` panel row positions.

        Returns:
            ``[M]`` array of demand at each row's own forecast origin.
        """
        rows = np.asarray(rows, dtype=np.int64)
        return np.asarray(
            [field.origin_kw(int(row)) for row in rows], dtype=np.float64
        )


def fit_baseline(
    method: str,
    field: DemandField,
    *,
    calibration_rows: np.ndarray,
    horizons: tuple[int, ...],
    granularity: str = "day_type_time_of_day",
    min_samples: int = 5,
) -> BaselineFit:
    """Fit one baseline by name, from calibration rows only.

    Args:
        method: One of :data:`BASELINE_METHODS`.
        field: The reconstructed demand field.
        calibration_rows: ``[M]`` panel row positions on ``CAL_FIT``.
        horizons: Horizons whose **target** steps contribute to the fitted table.
        granularity: Slot granularity, for ``calendar``.
        min_samples: Observations a cell needs before it is used.

    Returns:
        The fitted baseline.

    Raises:
        ValueError: If ``method`` is unknown, or no calibration rows are supplied.

    The calendar table is fitted on the set of **target** steps the calibration rows point
    at, across every horizon. Fitting it on origins instead would leave the last ``max(h)``
    steps of each horizon under-referenced and, worse, would make the reference depend on
    which horizon was being asked about.
    """
    calibration_rows = np.asarray(calibration_rows, dtype=np.int64)
    if calibration_rows.size == 0:
        raise ValueError(
            "no calibration rows were supplied; a baseline fitted on nothing would be a "
            "number with no evidence behind it"
        )
    if method == CalendarBaseline.name:
        target_steps = np.unique(
            field.origins[calibration_rows][:, None]
            + np.asarray(horizons, dtype=np.int64)[None, :]
        )
        return CalendarBaseline.fit(
            field,
            fit_steps=target_steps,
            # Series indices, not row positions: `fit` uses this to decide which per-series
            # medians to build, and it indexes the demand matrix by it. Passing row
            # positions here would index a 40-series matrix with values up to 89,760.
            series_rows=field.series[calibration_rows],
            granularity=granularity,
            min_samples=min_samples,
        )
    if method == PersistenceBaseline.name:
        return PersistenceBaseline.fit(
            field, fit_rows=calibration_rows, min_samples=min_samples
        )
    raise ValueError(f"unknown baseline {method!r}; expected one of {list(BASELINE_METHODS)}")


def evaluate_baseline(
    fit: BaselineFit,
    field: DemandField,
    *,
    rows: np.ndarray,
    horizon_steps: int,
    granularity: str,
) -> np.ndarray:
    """Expected demand for each row at one horizon.

    The slot is derived from the **target** step ``t + h``, not the origin ``t``: the
    envelope answers "how far does demand at the predicted moment depart from what is
    expected at that moment", and keying the slot on the origin would compare a Tuesday
    evening target against a Tuesday morning reference.

    Args:
        fit: The fitted baseline.
        field: The reconstructed demand field.
        rows: ``[M]`` panel row positions.
        horizon_steps: Forecast horizon in native steps.
        granularity: Slot granularity to key on.

    Returns:
        ``[M]`` array of expected demand in kW.
    """
    rows = np.asarray(rows, dtype=np.int64)
    if fit.method == PersistenceBaseline.name:
        return PersistenceBaseline.evaluate(field, rows)
    target_steps = field.origins[rows] + int(horizon_steps)
    slots = calendar_slots(target_steps, granularity)
    series = field.series[rows]
    return np.asarray(
        [fit.value(int(series[i]), int(slots[i])) for i in range(rows.size)],
        dtype=np.float64,
    )