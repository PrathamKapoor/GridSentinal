"""Prediction intervals: the object every method here produces, and its arithmetic.

An interval is two numbers and an assurance level. Everything else in this phase -
conformal scores, pinball quantiles, learned scale functions - is a way of choosing those
two numbers. Defining them in one place means the comparison between methods cannot be
contaminated by one method reporting a different quantity than another.

Terminology, fixed here because the distinction is easy to lose:

* **predictive uncertainty** - uncertainty about the future target given current
  information. This is what an interval quantifies.
* **prediction interval** - a range intended to contain a *future observation* at a stated
  coverage level. Not a confidence interval: a confidence interval is about a parameter or
  a mean, and the guarantees are different.
* **coverage** - the fraction of future observations that actually fall inside the interval.
  A method is *calibrated* at level ``1 - alpha`` when its measured coverage matches.

A deliberately conservative note on decomposition. Two of the methods here are built on
"the experts disagree", which is often called model or epistemic uncertainty, and the
residual-based methods are often called data or aleatoric uncertainty. **Neither label is
claimed.** Separating the two requires assumptions about the error-generating process that
this data cannot support, and Phase 6 already measured that the pretrained representation
carries almost no usable structure. The vocabulary used throughout is therefore
*predictive uncertainty*, *model disagreement* and *data variability*, and where a method
leans on disagreement it says so.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "DEFAULT_NOMINAL_LEVELS",
    "PredictionInterval",
    "empirical_quantile",
    "quantile_levels",
    "widths_from_scale",
    "lower_from_point",
    "clip_lower_at_zero",
]

#: The nominal coverage levels the phase reports.
#:
#: Four, chosen deliberately rather than swept. 50% is what an operational dashboard
#: would show as "likely range", 80% is the working default, 90% is the standard
#: statistical default and the one an energy risk assessment would cite, and 95% is where
#: interval methods start to become so wide as to be operationally useless. Reporting
#: every level from 1% to 99% would produce a large table and no additional information:
#: a method that is calibrated at 50% and 95% has told you what it does in between.
DEFAULT_NOMINAL_LEVELS: tuple[float, ...] = (0.50, 0.80, 0.90, 0.95)


def quantile_levels(nominal_levels: tuple[float, ...]) -> tuple[tuple[float, float], ...]:
    """The two tail probabilities each nominal coverage level needs.

    A nominal 90% interval is asymmetric in general: 5% of mass below the lower bound and
    5% above the upper one. That symmetry is an *assumption*, not a definition, and
    ``symmetric=False`` intervals drop it. Both are implemented; which one a method uses
    is recorded with it.

    **The split is in half.** A nominal 90% interval covers 90% of observations, so 5% lies
    below the lower bound and 5% above the upper one - not 10% below and 90% above. Taking
    ``(1 - level, level)`` as the two quantiles produces a central ``2 * level - 1``
    interval: "nominal 90%" would be an 80% interval, and "nominal 50%" would be the single
    median quantile with zero width. The error is quiet, because the symmetric methods in
    this phase are unaffected - they take the ``1 - alpha`` quantile of ``|residual|``,
    which is already correct - so only the asymmetric quantile-regression method would
    silently report coverage it had not achieved.

    Args:
        nominal_levels: Coverage levels in ``(0, 1)``.

    Returns:
        ``((alpha / 2, 1 - alpha / 2), ...)`` per level, in the order given.

    Raises:
        ValueError: If no level is supplied, or any level is outside ``(0, 1)``.
    """
    levels = tuple(float(level) for level in nominal_levels)
    if not levels:
        raise ValueError("at least one nominal level is required")
    for level in levels:
        if not 0.0 < level < 1.0:
            raise ValueError(f"nominal levels must lie in (0, 1), got {level}")
    return tuple(((1.0 - level) / 2.0, (1.0 + level) / 2.0) for level in levels)


def empirical_quantile(
    values: np.ndarray, probability: float, *, default: float = 0.0
) -> float:
    """The ``probability`` quantile of ``values`` by linear interpolation.

    Not the conformal quantile - see
    :func:`uncertainty.calibration.conformal_quantile_index` for that, which uses a
    different, finite-sample-corrected index. This one is the plain empirical quantile and
    is used where no coverage guarantee is claimed.

    Args:
        values: Sample to take the quantile of. May be any shape; it is flattened.
        probability: Quantile in ``[0, 1]``.
        default: Returned when ``values`` is empty.

    Returns:
        The quantile, as a Python float.
    """
    array = np.asarray(values, dtype=np.float64).ravel()
    if array.size == 0:
        return float(default)
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"probability must lie in [0, 1], got {probability}")
    return float(np.quantile(array, probability))


@dataclass(frozen=True, slots=True)
class PredictionInterval:
    """A half-open-free closed interval for one or many forecasts, in kW.

    Bounds are **clipped at zero on the way in** for this target, and the clip is
    recorded rather than hidden: customer demand cannot be negative, so an interval whose
    lower bound is -3 kW is stating something physically impossible while costing width.
    A lower bound of zero is still a valid bound - it just understates the nominal
    coverage by whatever mass the untruncated interval placed below zero.

    Attributes:
        lower_kw: Lower bound(s), kW.
        upper_kw: Upper bound(s), kW.
        nominal_level: Stated coverage level in ``(0, 1)``.
        method: How the bounds were produced. Recorded so a report can never present two
            different procedures under one number.
        symmetric: Whether the bounds are symmetric about the point forecast. ``False``
            means a two-sided quantile was used, which permits asymmetric intervals.
        clipped_at_zero: Whether a negative lower bound was raised to zero.
        notes: Free-form caveats, e.g. the calibration sample size.
    """

    lower_kw: np.ndarray
    upper_kw: np.ndarray
    nominal_level: float
    method: str
    symmetric: bool = True
    clipped_at_zero: bool = False
    notes: str = ""

    def __post_init__(self) -> None:
        lower = np.asarray(self.lower_kw, dtype=np.float64)
        upper = np.asarray(self.upper_kw, dtype=np.float64)
        if lower.shape != upper.shape:
            raise ValueError(
                f"lower and upper must have the same shape, got {lower.shape} and "
                f"{upper.shape}; the horizon axis is part of the shape"
            )
        if not 0.0 < float(self.nominal_level) < 1.0:
            raise ValueError(
                f"nominal_level must lie in (0, 1), got {self.nominal_level}"
            )
        if not isinstance(self.method, str) or not self.method.strip():
            raise ValueError(f"method must be a non-empty string, got {self.method!r}")
        if not np.isfinite(lower).all() or not np.isfinite(upper).all():
            raise ValueError("interval bounds must be finite")
        if np.any(lower > upper):
            offenders = int(np.count_nonzero(lower > upper))
            raise ValueError(
                f"lower_kw exceeds upper_kw on {offenders} of {lower.size} entries; an "
                f"interval with lower > upper has negative width and is not an interval"
            )

    @property
    def width_kw(self) -> np.ndarray:
        """``upper - lower``, elementwise."""
        return np.asarray(self.upper_kw, dtype=np.float64) - np.asarray(
            self.lower_kw, dtype=np.float64
        )

    @property
    def midpoint_kw(self) -> np.ndarray:
        """The interval centre, which is not the point forecast for asymmetric intervals."""
        return 0.5 * (
            np.asarray(self.lower_kw, dtype=np.float64)
            + np.asarray(self.upper_kw, dtype=np.float64)
        )

    @property
    def n_rows(self) -> int:
        return int(np.asarray(self.lower_kw).size)

    def contains(self, actual_kw: np.ndarray) -> np.ndarray:
        """Whether each realised value lies inside the closed interval.

        Args:
            actual_kw: Realised values, broadcastable to the bounds' shape.

        Returns:
            Boolean array of the bounds' shape.
        """
        actual = np.asarray(actual_kw, dtype=np.float64)
        lower = np.asarray(self.lower_kw, dtype=np.float64)
        upper = np.asarray(self.upper_kw, dtype=np.float64)
        return (actual >= lower) & (actual <= upper)

    def to_dict(self) -> dict[str, object]:
        """A JSON-friendly description. Bounds are summarised, not dumped."""
        width = self.width_kw
        return {
            "method": self.method,
            "nominal_level": float(self.nominal_level),
            "symmetric": bool(self.symmetric),
            "clipped_at_zero": bool(self.clipped_at_zero),
            "n": self.n_rows,
            "mean_width_kw": float(width.mean()) if width.size else None,
            "mean_lower_kw": float(np.asarray(self.lower_kw).mean())
            if self.n_rows
            else None,
            "mean_upper_kw": float(np.asarray(self.upper_kw).mean())
            if self.n_rows
            else None,
            "min_lower_kw": float(np.asarray(self.lower_kw).min()) if self.n_rows else None,
            "notes": self.notes,
        }


def widths_from_scale(
    *,
    point_kw: np.ndarray,
    scale: np.ndarray,
    multiplier: float,
    method: str,
    nominal_level: float,
    clip_at_zero: bool = True,
    notes: str = "",
) -> PredictionInterval:
    """A symmetric interval ``point +/- multiplier * scale``.

    The single constructor used by every symmetric method here, so that all of them
    differ only in ``scale`` and ``multiplier`` and nothing else can differ by accident.

    Args:
        point_kw: Point forecast, kW.
        scale: Non-negative width scale, same shape as the point forecast.
        multiplier: The factor the conformal or empirical quantile produced.
        method: Procedure name, recorded on the result.
        nominal_level: Stated coverage.
        clip_at_zero: Raise negative lower bounds to zero.
        notes: Caveats.

    Returns:
        The interval.

    Raises:
        ValueError: If ``scale`` is negative, shapes disagree, or ``multiplier`` is
            negative.
    """
    point = np.asarray(point_kw, dtype=np.float64)
    scale_array = np.asarray(scale, dtype=np.float64)
    if point.shape != scale_array.shape:
        raise ValueError(
            f"point {point.shape} and scale {scale_array.shape} must have the same shape"
        )
    if np.any(scale_array < 0):
        raise ValueError(
            f"a scale function must be non-negative; got a minimum of "
            f"{float(scale_array.min())}"
        )
    if float(multiplier) < 0:
        raise ValueError(f"multiplier must be non-negative, got {multiplier}")
    half = float(multiplier) * scale_array
    upper = point + half
    lower, clipped = clip_lower_at_zero(point - half, upper, enabled=clip_at_zero)
    return PredictionInterval(
        lower_kw=lower,
        upper_kw=upper,
        nominal_level=float(nominal_level),
        method=method,
        symmetric=True,
        clipped_at_zero=clipped,
        notes=notes,
    )


def clip_lower_at_zero(
    lower: np.ndarray, upper: np.ndarray, *, enabled: bool
) -> tuple[np.ndarray, bool]:
    """Raise negative lower bounds to zero, but only where an interval still remains.

    Customer demand cannot be negative, so an interval whose lower bound is -3 kW is
    stating something physically impossible while also paying for the width. Clipping it to
    zero is the right correction, and the clip is recorded on the result rather than hidden.

    The case that needs care is an interval that lies **entirely** below zero. Phase 7's
    fixed ensemble produces a handful of slightly negative point forecasts - on the sealed
    test split, 3 of 1,077,120 bounds, with a minimum point forecast of -0.67 kW - so this
    is a real case and not a hypothetical. Clipping there would return ``[0, -1]``: an
    inverted interval, or a degenerate ``[0, 0]`` that asserts a forecast of exactly zero
    with exactly no uncertainty. Both are worse than the honest answer, so those entries are
    left alone and counted. ``PredictionInterval`` then reports a small share of
    physically-impossible lower bounds, which is the true state of the forecast.

    Args:
        lower: Lower bounds.
        upper: Upper bounds.
        enabled: Whether to clip at all.

    Returns:
        ``(lower, clipped_any)``.
    """
    if not enabled:
        return np.asarray(lower, dtype=np.float64), False
    low = np.asarray(lower, dtype=np.float64)
    high = np.asarray(upper, dtype=np.float64)
    eligible = low < 0.0
    # Only clip where the upper bound is at or above zero, i.e. where a non-degenerate
    # interval survives the clip.
    low = np.where(eligible & (high >= 0.0), 0.0, low)
    return low, bool(np.any(eligible & (high >= 0.0)))


def lower_from_point(
    *,
    point_kw: np.ndarray,
    lower_delta_kw: np.ndarray,
    upper_delta_kw: np.ndarray,
    method: str,
    nominal_level: float,
    clip_at_zero: bool = True,
    notes: str = "",
) -> PredictionInterval:
    """An **asymmetric** interval from signed quantile offsets.

    Used by the direct quantile-regression method, where the residual quantiles need not
    be symmetric about zero - demand forecasts are not, because under-forecasting a peak
    costs more than over-forecasting it.

    Args:
        point_kw: Point forecast, kW.
        lower_delta_kw: ``q_alpha - point``, which is negative or zero.
        upper_delta_kw: ``q_1-alpha - point``, which is positive or zero.
        method: Procedure name.
        nominal_level: Stated coverage.
        clip_at_zero: Raise negative lower bounds to zero.
        notes: Caveats.

    Returns:
        The interval.

    Raises:
        ValueError: If shapes disagree or the lower bound ends up above the upper bound.
    """
    point = np.asarray(point_kw, dtype=np.float64)
    lower_delta = np.asarray(lower_delta_kw, dtype=np.float64)
    upper_delta = np.asarray(upper_delta_kw, dtype=np.float64)
    if point.shape != lower_delta.shape or point.shape != upper_delta.shape:
        raise ValueError(
            f"point {point.shape}, lower {lower_delta.shape} and upper "
            f"{upper_delta.shape} must all have the same shape"
        )
    lower = point + lower_delta
    upper = point + upper_delta
    lower, clipped = clip_lower_at_zero(lower, upper, enabled=clip_at_zero)
    return PredictionInterval(
        lower_kw=lower,
        upper_kw=upper,
        nominal_level=float(nominal_level),
        method=method,
        symmetric=False,
        clipped_at_zero=clipped,
        notes=notes,
    )