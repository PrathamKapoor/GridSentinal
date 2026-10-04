"""Directional behavioural envelopes: how far demand has *moved*, per direction.

The estimate this module produces
    -------------------------------
    For a series and a calendar slot, the two non-negative magnitudes such that, over the
    calibration period, realised demand stayed inside ``[expected - downward,
    expected + upward]`` for the stated fraction of observations. That is a statement about
    **behaviour**. It is deliberately constructed so it cannot be read as a capability:

    * the baseline that defines "expected" is fitted on ``CAL_FIT`` only;
    * the tail quantiles come from ``CAL_FIT`` only;
    * the result carries :attr:`FlexibilityBasis.STATISTICAL_PROXY` with
      :attr:`AuthorityLevel.NOT_CONTROLLABLE`, which
      :class:`~energy_intelligence.domain.flexibility.FlexibilityEstimate` refuses to let a
      dispatcher read as a control contract.

Why a *median* baseline and *tail* quantiles
    The central value removes the predictable daily shape, so what remains is the part a
    calendar cannot explain. The tails, rather than a standard deviation, are what a
    directional claim needs: an asymmetry between "has run high" and "has run low" is real
    and common in household demand, and a symmetric width would erase it.

Pooling across horizons
    Deviations from every horizon are pooled into one sample per (series, slot) cell and
    **one** quantile is taken from the pooled sample. They are not summed and divided: a
    cell's width must be a quantile of the deviations that land in it, not an average of
    several quantiles, which is a different and uninterpretable statistic. Pooling is
    defensible because the deviation is measured against the *target* step, so a Tuesday
    19:00 deviation is the same object whether a 15-minute or a 24-hour forecast reached it.

One-sided clamping, and why it is recorded
    ``quantile(1 - alpha)`` of the deviation can be **negative** - meaning fewer than
    ``alpha`` of calibration observations sat above the baseline. Clamping the magnitude at
    zero then puts the upper bound *at* the baseline, which is what the quantile says. This
    is not a fudge and the rate at which it fires is recorded, because a slot where the load
    has never historically run above its own median is a real and reportable finding.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .demand import DemandField, calendar_slots, slot_labels
from .baselines import BaselineFit, PersistenceBaseline, evaluate_baseline

__all__ = [
    "EnvelopeFit",
    "fit_envelope",
    "evaluate_envelope",
    "conformal_scale",
]


@dataclass(frozen=True, slots=True)
class EnvelopeFit:
    """Directional magnitudes per series and calendar slot.

    Attributes:
        method: The interval-construction procedure, recorded on every estimate derived
            from this fit.
        baseline: The fitted expected-demand reference the deviation is measured against.
            Held rather than referenced by name, because re-deriving it from a name would
            be one more place for the wrong table to be substituted.
        granularity: Slot granularity the table is keyed by.
        horizons: The horizons the leading axis of every table corresponds to, in order.
        nominal_level: Stated two-sided coverage of the resulting band.
        upward_kw: ``[n_horizons, n_series, n_slots]`` upward magnitude, kW.
        downward_kw: ``[n_horizons, n_series, n_slots]`` downward magnitude, kW.
        median_deviation_kw: ``[n_horizons, n_series, n_slots]`` median signed deviation,
            kW. Retained
            because the asymmetry between the two magnitudes is the interesting quantity
            and it cannot be recovered from them.
        counts: ``[n_series, n_slots]`` calibration deviations pooled into each cell.
        upward_clamped: ``[n_series, n_slots]`` cells whose upper quantile sat below the
            baseline and had to be clamped to zero.
        calibration_rows: How many rows the fit consulted.
    """

    method: str
    baseline: BaselineFit
    granularity: str
    horizons: tuple[int, ...]
    nominal_level: float
    upward_kw: np.ndarray
    downward_kw: np.ndarray
    median_deviation_kw: np.ndarray
    counts: np.ndarray
    upward_clamped: np.ndarray
    tail_at_saturation: np.ndarray
    calibration_rows: int

    def magnitudes(self, horizon_position: int, series_index: int, slot: int) -> tuple[float, float]:
        """``(upward, downward)`` magnitudes in kW for one horizon, series and slot.

        The horizon is part of the key rather than an afterthought because the cells are
        fitted **per horizon**. Deviation at h=1 and deviation at h=96 differ by roughly a
        factor of four on this data, so a cell shared between them would have to be wide
        enough for h=96 - which is what makes an h=1 envelope over-cover by tens of percent.
        Pooling horizons buys sample size at the cost of making per-horizon calibration
        impossible, and calibration is the property this phase exists to establish.

        Args:
            horizon_position: Index into :attr:`horizons`.
            series_index: Series index.
            slot: Calendar slot position.

        Returns:
            ``(upward, downward)`` in kW.

        Raises:
            IndexError: If any index is out of range.
        """
        column = int(horizon_position)
        index = int(series_index)
        position = int(slot)
        if not 0 <= column < self.upward_kw.shape[0]:
            raise IndexError(
                f"horizon position {column} out of range for {self.upward_kw.shape[0]} "
                f"horizons {tuple(self.horizons)}"
            )
        if not 0 <= index < self.upward_kw.shape[1]:
            raise IndexError(f"series index {index} out of range for {self.upward_kw.shape[1]}")
        if not 0 <= position < self.upward_kw.shape[2]:
            raise IndexError(
                f"slot {position} out of range for {self.granularity} "
                f"({self.upward_kw.shape[2]} slots)"
            )
        return (
            float(self.upward_kw[column, index, position]),
            float(self.downward_kw[column, index, position]),
        )

    def fallback_magnitude_kw(self, horizon_position: int) -> float:
        """The calibration-wide median magnitude at one horizon, for never-seen cells.

        Distinct from a populated cell whose magnitude is genuinely zero: "we have never
        seen this slot" and "this slot has never moved" are different facts and collapsing
        them would understate the estimate. Taken per horizon for the same reason the cells
        are.
        """
        used = self.counts[int(horizon_position)] > 0
        if not used.any():
            return 0.0
        widths = self.upward_kw[int(horizon_position)][used] + self.downward_kw[
            int(horizon_position)
        ][used]
        return float(np.median(widths))

    def describe(self) -> dict[str, Any]:
        used = self.counts > 0
        mean_up = float(self.upward_kw[used].mean()) if used.any() else None
        mean_down = float(self.downward_kw[used].mean()) if used.any() else None
        return {
            "method": self.method,
            "baseline": self.baseline.method,
            "baseline_granularity": self.baseline.granularity,
            "granularity": self.granularity,
            "horizons": [int(h) for h in self.horizons],
            "nominal_level": float(self.nominal_level),
            "calibration_rows": int(self.calibration_rows),
            "fitted_per_horizon": True,
            "cells": int(self.upward_kw.size),
            "cells_used": int(used.sum()),
            "fallback_cell_share": float(1.0 - used.sum() / used.size) if used.size else None,
            "median_observations_per_cell": float(np.median(self.counts[used]))
            if used.any()
            else None,
            "upward_clamped_cell_share": (
                float(self.upward_clamped[used].sum() / used.sum()) if used.any() else None
            ),
            "mean_upward_kw": mean_up,
            "mean_downward_kw": mean_down,
            "directional_asymmetry": (
                float(mean_up / mean_down) if mean_up is not None and mean_down else None
            ),
            "fallback_magnitude_kw_by_horizon": {
                str(horizon): self.fallback_magnitude_kw(position)
                for position, horizon in enumerate(self.horizons)
            },
            "mean_width_kw_by_horizon": {
                str(horizon): float(
                    np.mean(self.upward_kw[position][self.counts[position] > 0])
                    + np.mean(self.downward_kw[position][self.counts[position] > 0])
                )
                if np.any(self.counts[position] > 0)
                else None
                for position, horizon in enumerate(self.horizons)
            },
        }


def fit_envelope(
    *,
    field: DemandField,
    baseline: BaselineFit,
    calibration_rows: np.ndarray,
    horizons: tuple[int, ...],
    nominal_level: float = 0.90,
    min_samples: int = 5,
) -> EnvelopeFit:
    """Fit directional magnitudes from calibration rows only.

    Args:
        field: The reconstructed demand field.
        baseline: The fitted expected-demand reference.
        calibration_rows: ``[M]`` panel row positions on ``CAL_FIT``.
        horizons: Horizons whose deviations are pooled into each cell.
        nominal_level: Stated two-sided coverage.
        min_samples: Deviations a cell needs before its quantiles are used.

    Returns:
        The fitted envelope.

    Raises:
        ValueError: If no calibration rows are supplied, ``nominal_level`` is outside
            ``(0, 1)``, or no horizon is given.
    """
    calibration_rows = np.asarray(calibration_rows, dtype=np.int64)
    if calibration_rows.size == 0:
        raise ValueError("an envelope needs calibration rows; none were supplied")
    if not 0.0 < float(nominal_level) < 1.0:
        raise ValueError(f"nominal_level must lie in (0, 1), got {nominal_level}")
    if not horizons:
        raise ValueError("at least one horizon is required to fit an envelope")

    granularity = "horizon" if baseline.method == PersistenceBaseline.name else baseline.granularity
    n_slots = len(slot_labels(granularity))
    n_series = field.n_series
    # Leading horizon axis: one independent table per horizon. See `magnitudes` for why
    # sharing cells across horizons is not an acceptable simplification.
    upward = np.zeros((len(horizons), n_series, n_slots), dtype=np.float64)
    downward = np.zeros((len(horizons), n_series, n_slots), dtype=np.float64)
    median_deviation = np.zeros((len(horizons), n_series, n_slots), dtype=np.float64)
    counts = np.zeros((len(horizons), n_series, n_slots), dtype=np.int64)
    clamped = np.zeros((len(horizons), n_series, n_slots), dtype=bool)
    # A cell too small to reach the conformal rank falls back to its sample extreme.
    saturated = np.zeros((len(horizons), n_series, n_slots), dtype=bool)

    alpha = 1.0 - float(nominal_level)
    # The miss probability is split evenly between the two tails. Using the *total* miss
    # probability as the per-tail quantile would silently deliver (1 - alpha) coverage per
    # tail, i.e. 2*(1-alpha) = 80% at a nominal 90% - the band would be mislabelled and
    # would under-cover by ten points on perfectly behaved data.
    tail = alpha / 2.0
    series = field.series[calibration_rows]
    for column, horizon in enumerate(horizons):
        expected = evaluate_baseline(
            baseline,
            field,
            rows=calibration_rows,
            horizon_steps=int(horizon),
            granularity=granularity,
        )
        target_steps = field.origins[calibration_rows] + int(horizon)
        realised = np.asarray(
            [field.target_kw(int(row), int(horizon)) for row in calibration_rows],
            dtype=np.float64,
        )
        slots = calendar_slots(target_steps, granularity)
        delta = realised - expected

        # One (series, slot) -> deviations map for THIS horizon only.
        pooled: dict[tuple[int, int], list[float]] = {}
        for i in range(calibration_rows.size):
            pooled.setdefault((int(series[i]), int(slots[i])), []).append(float(delta[i]))

        for (index, position), values in pooled.items():
            sample = np.sort(np.asarray(values, dtype=np.float64))
            n = int(sample.size)
            if n < int(min_samples):
                continue
            # Conformal order statistics rather than interpolated empirical quantiles. With
            # a handful of observations per cell an interpolated quantile sits *between*
            # observed points and therefore under-covers; the rank ceil((n+1)(1-tail))
            # guarantees at least nominal coverage however small the cell is, and degrades
            # to the sample maximum when the cell is too small to promise it.
            upper_index = min(int(np.ceil((n + 1) * (1.0 - tail))), n) - 1
            lower_index = max(int(np.floor((n + 1) * tail)), 1) - 1
            upper_q = float(sample[upper_index])
            lower_q = float(sample[lower_index])
            upward[column, index, position] = max(0.0, upper_q)
            downward[column, index, position] = max(0.0, -lower_q)
            median_deviation[column, index, position] = float(np.median(sample))
            counts[column, index, position] = n
            clamped[column, index, position] = bool(upper_q < 0.0)
            saturated[column, index, position] = bool(upper_index >= n - 1 or lower_index <= 0)

    return EnvelopeFit(
        method=f"per_horizon_directional_empirical_quantile[{baseline.method}]",
        baseline=baseline,
        granularity=granularity,
        horizons=tuple(int(h) for h in horizons),
        nominal_level=float(nominal_level),
        upward_kw=upward,
        downward_kw=downward,
        median_deviation_kw=median_deviation,
        counts=counts,
        upward_clamped=clamped,
        tail_at_saturation=saturated,
        calibration_rows=int(calibration_rows.size),
    )


def evaluate_envelope(
    fit: EnvelopeFit,
    field: DemandField,
    *,
    rows: np.ndarray,
    horizon_steps: int,
) -> dict[str, np.ndarray]:
    """Apply a fitted envelope to rows, returning the band and its components.

    Args:
        fit: The fitted envelope.
        field: The reconstructed demand field.
        rows: ``[M]`` panel row positions.
        horizon_steps: Forecast horizon in native steps.

    Returns:
        ``expected_kw``, ``upward_kw``, ``downward_kw``, ``lower_kw``, ``upper_kw``,
        ``used_fallback`` and ``observed_kw``. Widths are the *raw* magnitudes; applying the
        uncertainty reliance factor is :mod:`uncertainty`'s caller, not this function's job,
        so that the un-discounted envelope is always available for comparison.
    """
    rows = np.asarray(rows, dtype=np.int64)
    granularity = fit.granularity
    try:
        column = int(fit.horizons.index(int(horizon_steps)))
    except ValueError as exc:
        raise ValueError(
            f"horizon {horizon_steps} was not fitted; this envelope covers "
            f"{list(fit.horizons)}"
        ) from exc
    target_steps = field.origins[rows] + int(horizon_steps)
    slots = calendar_slots(target_steps, granularity)
    series = field.series[rows]

    upward = np.empty(rows.size, dtype=np.float64)
    downward = np.empty(rows.size, dtype=np.float64)
    used_fallback = np.zeros(rows.size, dtype=bool)
    fallback = fit.fallback_magnitude_kw(column)
    for i in range(rows.size):
        index = int(series[i])
        position = int(slots[i])
        if fit.counts[column, index, position] > 0:
            up, down = fit.magnitudes(column, index, position)
        else:
            up = down = float(fallback)
            used_fallback[i] = True
        upward[i] = up
        downward[i] = down

    expected = evaluate_baseline(
        fit.baseline,
        field,
        rows=rows,
        horizon_steps=horizon_steps,
        granularity=granularity,
    )
    return {
        "expected_kw": expected,
        "upward_kw": upward,
        "downward_kw": downward,
        "lower_kw": expected - downward,
        "upper_kw": expected + upward,
        "used_fallback": used_fallback,
        "observed_kw": np.asarray(
            [field.target_kw(int(row), int(horizon_steps)) for row in rows],
            dtype=np.float64,
        ),
    }


def conformal_scale(
    *,
    expected_kw: np.ndarray,
    observed_kw: np.ndarray,
    upward_kw: np.ndarray,
    downward_kw: np.ndarray,
    nominal_level: float,
    tolerance: float = 1e-4,
    max_scale: float = 10.0,
) -> tuple[float, dict[str, Any]]:
    """The smallest symmetric rescale of a band that reaches ``nominal_level`` on given rows.

    This is the step that turns a *fitted* envelope into a *calibrated* one, and it is why the
    calibration split is halved rather than used whole.

    A quantile band fitted on ``CAL_FIT`` is calibrated for the spread it was fitted on. If
    the evaluation period is quieter or noisier - and on real load data it usually is - the
    same band systematically over- or under-covers, by more than the binomial noise on the
    row count can explain. On this dataset the fitted band over-covered the sealed split by
    roughly three percentage points, which at 179,520 rows is a real shift rather than
    sampling error, and no amount of re-fitting the quantiles fixes it.

    Scaling both directions by one factor and solving for the factor that reaches nominal
    coverage on ``CAL_CONF`` is the standard split-conformal correction: it adjusts the band
    to the period it is being used in, using rows that were never used to fit it. Coverage is
    monotone non-decreasing in the scale, so a bisection is exact rather than approximate.

    Args:
        expected_kw: ``[M]`` baseline value, kW.
        observed_kw: ``[M]`` realised demand, kW.
        upward_kw: ``[M]`` upward magnitudes, kW.
        downward_kw: ``[M]`` downward magnitudes, kW.
        nominal_level: The coverage to reach.
        tolerance: Bisection resolution on the scale.
        max_scale: Ceiling, so a pathological row cannot produce an unbounded band.

    Returns:
        ``(scale, report)``. The report carries the coverage before and after, so the size of
        the correction is visible in the result rather than implied by the final number.

    Raises:
        ValueError: If no rows are supplied, or a band is entirely degenerate.
    """
    expected = np.asarray(expected_kw, dtype=np.float64).ravel()
    observed = np.asarray(observed_kw, dtype=np.float64).ravel()
    up = np.asarray(upward_kw, dtype=np.float64).ravel()
    down = np.asarray(downward_kw, dtype=np.float64).ravel()
    if expected.size == 0:
        raise ValueError("conformal_scale needs at least one row")
    if not np.any(up > 0) and not np.any(down > 0):
        raise ValueError(
            "the band is degenerate - every direction is zero - so no scale factor can "
            "reach a non-zero coverage level"
        )

    def coverage(scale: float) -> float:
        lower = expected - scale * down
        upper = expected + scale * up
        return float(np.mean((observed >= lower) & (observed <= upper)))

    before = coverage(1.0)
    low, high = 0.0, float(max_scale)
    if coverage(high) < float(nominal_level):
        return float(high), {
            "scale": float(high),
            "coverage_before": before,
            "coverage_after": coverage(high),
            "nominal_level": float(nominal_level),
            "converged": False,
            "note": (
                f"even a {max_scale}x band does not reach {nominal_level:.0%} coverage on "
                f"these rows; the scale is capped rather than allowed to run away, and the "
                f"shortfall is reported"
            ),
        }
    for _ in range(200):
        mid = 0.5 * (low + high)
        if coverage(mid) >= float(nominal_level):
            high = mid
        else:
            low = mid
        if high - low <= float(tolerance):
            break
    scale = float(high)
    after = coverage(scale)
    return scale, {
        "scale": scale,
        "coverage_before": before,
        "coverage_after": after,
        "nominal_level": float(nominal_level),
        "converged": bool(abs(after - float(nominal_level)) <= 4.0 * tolerance),
        "note": (
            "fitted on the conformity split, which was not used to fit the band, and "
            "applied unchanged to the sealed test split"
        ),
    }
