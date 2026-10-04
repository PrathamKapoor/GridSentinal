"""Split conformal prediction: a coverage guarantee, and exactly what it assumes.

This is the only method in the phase that offers a *distribution-free* guarantee, and the
guarantee is worth stating precisely rather than paraphrasing, because the assumptions are
what make it either useful or worthless.

The result
----------
Given a point predictor fitted on one sample, and ``n`` calibration observations whose
residuals are **exchangeable** with the future ones, the interval

    [yhat - q, yhat + q],   q = the ceil((n+1)(1-alpha))-th smallest |residual|

satisfies

    P( y_new in interval )  >=  1 - alpha.

**The assumptions, all of which must be stated rather than implied:**

1. **Exchangeability.** Calibration residuals and future residuals come from the same
   distribution. This is the load-bearing one. It is *not* satisfied by ordinary time-series
   forecasting, where a residual from January and a residual from November are drawn from
   visibly different distributions - Phase 8's own data shows the error growing with
   horizon and shifting across the year. So the finite-sample guarantee below is reported
   as **an empirical check, not a theorem being invoked**. A time-ordered conformal method
   (adaptive or weighted) would be needed for a real guarantee, and is out of scope here.
2. **The point predictor is fixed before calibration.** A predictor re-tuned on the
   calibration set invalidates the guarantee. Enforced structurally: the calibrator is
   constructed from a frozen array of scores.
3. **Scores are computed without the target leaking into the predictor.** Enforced by
   construction: ``fit`` receives predictions and targets separately, and the predictor was
   fitted on a strictly earlier split.

The normalised form
-------------------
Dividing each residual by a positive scale ``s(x)`` before taking the quantile, then
multiplying the resulting width by ``s(x)``, preserves the guarantee **when the division is
a fixed monotone transform** - that is, when the scale does not depend on the residual.
The scale functions used here are fitted on a *different* split from the conformity scores,
so that condition holds. This is why the calibration set is split in two rather than used
whole: one part fits the scale, the other supplies the scores. Reusing one part for both
would make the scale depend on the very residuals being normalised.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from .intervals import PredictionInterval, clip_lower_at_zero

__all__ = [
    "CONFORMAL_GUARANTEE",
    "ConformalCalibrator",
    "conformal_intervals",
    "conformal_quantile_index",
    "conformal_width",
    "fit_conformal",
    "fit_conformal_quantile",
]


def conformal_quantile_index(n: int, alpha: float) -> int:
    """The 1-indexed order statistic a split-conformal interval must use.

    ``ceil((n + 1) * (1 - alpha))``, capped at ``n``.

    The ``+1`` and the ceiling are the whole point. Using the plain ``(1 - alpha)``
    quantile of ``n`` scores gives coverage that is *below* nominal for small ``n`` and
    whenever ``(n+1)(1-alpha)`` lands between two order statistics - which is most of the
    time. With 89,760 calibration rows the correction moves the index by less than one
    position, so here it barely matters; at ``n = 100`` it is the difference between a
    valid and an invalid guarantee, and the index is computed the same way regardless of
    ``n`` so that a small-``n`` experiment elsewhere in the project would be correct.

    Args:
        n: Number of conformity scores.
        alpha: Miscoverage in ``(0, 1)``.

    Returns:
        A 1-indexed position in ``1..n``.

    Raises:
        ValueError: If ``n`` is not positive or ``alpha`` is outside ``(0, 1)``.
    """
    if int(n) <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if not 0.0 < float(alpha) < 1.0:
        raise ValueError(f"alpha must lie in (0, 1), got {alpha}")
    return int(min(int(n), math.ceil((int(n) + 1) * (1.0 - float(alpha)))))


def fit_conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """The conformal quantile of conformity scores.

    Args:
        scores: Nonconformity scores from the conformity split.
        alpha: Miscoverage.

    Returns:
        The multiplier, which is the ``k``-th smallest score.

    Raises:
        ValueError: If ``scores`` is empty.
    """
    array = np.asarray(scores, dtype=np.float64).ravel()
    if array.size == 0:
        raise ValueError("cannot take a conformal quantile of zero scores")
    if not np.isfinite(array).all():
        raise ValueError("conformity scores must be finite")
    index = conformal_quantile_index(array.size, alpha)
    # ``sorted`` is stable and ``index`` is 1-indexed, so this is the index-th smallest.
    return float(np.sort(array)[index - 1])


@dataclass(frozen=True, slots=True)
class ConformalCalibrator:
    """A fitted, per-horizon conformal width multiplier.

    One multiplier per horizon, because error scales with horizon by an order of magnitude
    and a single multiplier would make the h=1 interval useless or the h=96 interval
    absent. One multiplier per **nominal level** for the same reason: a 50% interval and a
    95% interval cannot share a half-width unless the score distribution has a very
    particular shape, which is not something to assume.

    Attributes:
        multipliers: ``{horizon: {nominal_level: multiplier}}``.
        scale_name: Which scale function the scores were normalised by. Recorded so a
            report cannot attribute a normalised interval to an absolute one.
        n_scores: ``{horizon: count}`` conformity rows used.
        guarantee: The conditional coverage bound, and a statement of what it assumes.
    """

    multipliers: dict[int, dict[float, float]]
    scale_name: str
    n_scores: dict[int, int]
    guarantee: str

    @property
    def scale_is_absolute(self) -> bool:
        """Whether the scores were unnormalised, i.e. a unit scale was used."""
        return self.scale_name == "absolute"

    def width(self, horizon: int, nominal_level: float) -> float:
        """The multiplier for one horizon and level.

        Raises:
            KeyError: If that horizon or level was not calibrated.
        """
        by_level = self.multipliers.get(int(horizon))
        if by_level is None:
            raise KeyError(
                f"no conformal calibration for horizon {horizon}; have "
                f"{sorted(self.multipliers)}"
            )
        if float(nominal_level) not in by_level:
            raise KeyError(
                f"no conformal calibration at nominal level {nominal_level} for horizon "
                f"{horizon}; have {sorted(by_level)}"
            )
        return float(by_level[float(nominal_level)])

    def intervals(
        self,
        *,
        point_kw: np.ndarray,
        scale: np.ndarray | None,
        horizons: tuple[int, ...],
        nominal_levels: tuple[float, ...],
        method: str,
        clip_at_zero: bool = True,
        notes: str = "",
    ) -> dict[float, PredictionInterval]:
        """Build intervals at every horizon and nominal level.

        Args:
            point_kw: ``[N, horizons]`` point forecast, kW.
            scale: ``[N, horizons]`` positive scale, or ``None`` for absolute residuals.
            horizons: Horizon steps.
            nominal_levels: Levels to build.
            method: Procedure name, recorded on each interval.
            clip_at_zero: Raise negative lower bounds to zero.
            notes: Caveats appended to each interval.

        Returns:
            ``{nominal_level: PredictionInterval}`` with ``[N, horizons]`` bounds.

        Raises:
            ValueError: If ``scale`` has the wrong shape.
        """
        point = np.asarray(point_kw, dtype=np.float64)
        horizons = tuple(int(h) for h in horizons)
        if point.shape[1] != len(horizons):
            raise ValueError(
                f"point_kw has {point.shape[1]} horizon columns for {len(horizons)} "
                f"horizons"
            )
        if scale is None:
            scale_array = np.ones_like(point)
        else:
            scale_array = np.asarray(scale, dtype=np.float64)
            if scale_array.shape != point.shape:
                raise ValueError(
                    f"scale {scale_array.shape} must match point_kw {point.shape}"
                )
            if np.any(scale_array < 0):
                raise ValueError("a conformal scale must be non-negative")
        out: dict[float, PredictionInterval] = {}
        for nominal in nominal_levels:
            alpha = 1.0 - float(nominal)
            lower = np.empty_like(point)
            upper = np.empty_like(point)
            clipped = False
            for column, horizon in enumerate(horizons):
                multiplier = self.width(horizon, nominal)
                half = multiplier * scale_array[:, column]
                upper[:, column] = point[:, column] + half
                lower[:, column], column_clipped = clip_lower_at_zero(
                    point[:, column] - half,
                    upper[:, column],
                    enabled=clip_at_zero,
                )
                clipped = clipped or column_clipped
            out[float(nominal)] = PredictionInterval(
                lower_kw=lower,
                upper_kw=upper,
                nominal_level=float(nominal),
                method=method,
                symmetric=True,
                clipped_at_zero=clipped,
                notes=notes,
            )
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "scale_name": self.scale_name,
            "scale_is_absolute": self.scale_is_absolute,
            "multipliers": {
                str(horizon): {
                    str(level): float(value) for level, value in by_level.items()
                }
                for horizon, by_level in self.multipliers.items()
            },
            "n_scores": {str(h): int(n) for h, n in self.n_scores.items()},
            "guarantee": self.guarantee,
        }


#: What the phase claims, and does not claim, about coverage.
CONFORMAL_GUARANTEE = (
    "Split conformal gives P(y_new in interval) >= 1 - alpha when the conformity scores "
    "are exchangeable with future scores and the point predictor is fixed before "
    "calibration. Time-ordered forecasting residuals are NOT exchangeable, so the "
    "coverage reported here is an empirical measurement on held-out data, not an invoked "
    "guarantee. A time-ordered conformal method would be required for a real guarantee."
)


def fit_conformal(
    *,
    residual_kw: np.ndarray,
    scale: np.ndarray | None,
    horizons: tuple[int, ...],
    nominal_levels: tuple[float, ...],
    scale_name: str,
) -> ConformalCalibrator:
    """Fit per-horizon, per-level conformal multipliers from conformity residuals.

    Args:
        residual_kw: ``[N, horizons]`` **signed** or absolute residuals; only the absolute
            value is used. kW.
        scale: ``[N, horizons]`` positive scale the residuals were normalised by, or
            ``None`` for absolute.
        horizons: Horizon steps.
        nominal_levels: Levels to calibrate.
        scale_name: Name of the scale, recorded on the result.

    Returns:
        The calibrator.

    Raises:
        ValueError: On shape disagreement, an empty conformity set, or a non-positive scale.
    """
    residual = np.abs(np.asarray(residual_kw, dtype=np.float64))
    horizons = tuple(int(h) for h in horizons)
    if residual.shape[1] != len(horizons):
        raise ValueError(
            f"residual_kw has {residual.shape[1]} horizon columns for {len(horizons)} "
            f"horizons"
        )
    if scale is None:
        scale_array = np.ones_like(residual)
    else:
        scale_array = np.asarray(scale, dtype=np.float64)
        if scale_array.shape != residual.shape:
            raise ValueError(
                f"scale {scale_array.shape} must match residual_kw {residual.shape}"
            )
        if np.any(scale_array <= 0):
            raise ValueError(
                "a conformal scale must be strictly positive; a zero would divide the "
                "score by zero and produce an infinite width"
            )
    scores = residual / scale_array
    multipliers: dict[int, dict[float, float]] = {}
    n_scores: dict[int, int] = {}
    for column, horizon in enumerate(horizons):
        by_level: dict[float, float] = {}
        for nominal in nominal_levels:
            by_level[float(nominal)] = fit_conformal_quantile(
                scores[:, column], 1.0 - float(nominal)
            )
        multipliers[horizon] = by_level
        n_scores[horizon] = int(residual.shape[0])
    return ConformalCalibrator(
        multipliers=multipliers,
        scale_name=scale_name,
        n_scores=n_scores,
        guarantee=CONFORMAL_GUARANTEE,
    )


def conformal_width(
    calibrator: ConformalCalibrator, horizon: int, nominal_level: float
) -> float:
    """The calibrated absolute half-width multiplier for one horizon and level."""
    return calibrator.width(horizon, nominal_level)


def conformal_intervals(
    calibrator: ConformalCalibrator,
    *,
    point_kw: np.ndarray,
    scale: np.ndarray | None,
    horizons: tuple[int, ...],
    nominal_levels: tuple[float, ...],
    method: str = "split_conformal",
) -> dict[float, PredictionInterval]:
    """Convenience wrapper: fit nothing, just build intervals from a calibrator."""
    return calibrator.intervals(
        point_kw=point_kw,
        scale=scale,
        horizons=horizons,
        nominal_levels=nominal_levels,
        method=method,
    )
