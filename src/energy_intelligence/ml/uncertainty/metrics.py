"""Calibration metrics. Every one defined here, none invented.

Coverage alone is not a score: a method that emits an interval from -infinity to
+infinity achieves perfect coverage and is useless. Sharpness alone is not a score either:
the tightest possible interval achieves perfect sharpness and is dishonest. So both are
reported side by side, and where a single number is genuinely wanted the **Winkler
interval score** is used, because it is the standard proper scoring rule for a single
interval and it already encodes the coverage/sharpness trade-off rather than inventing a
new weighting of it.

Definitions, in the notation used throughout:

``L, U``   interval bounds for one forecast, in kW
``y``      the realised value, in kW
``alpha``  miscoverage, ``1 - nominal_level``

* **coverage** = ``mean(L <= y <= U)``.
* **coverage error** = ``coverage - (1 - alpha)``. Signed on purpose. A method whose
  coverage error is negative is *under*-covering, which is the dangerous direction, and a
  signed number cannot hide that.
* **mean width** = ``mean(U - L)``.
* **interval score** (Winkler, ``alpha``) =
  ``(U - L) + (2/alpha) * (L - y) * 1[y < L] + (2/alpha) * (y - U) * 1[y > U]``.
  It equals the width when the interval covers, and grows superlinearly with the
  overshoot when it does not.
* **weighted interval score** (WIS) is the standard average of interval scores over
  several levels, weighting level ``alpha`` by ``alpha / 2`` and dividing by
  ``sum(alpha / 2)``. Higher ``alpha`` levels are thereby weighted by how much probability
  mass they are responsible for, which is the literature convention and is stated here so
  the number is comparable rather than merely defined.
* **sharpness ratio** = ``mean width / mean |residual|``. Dimensionless, so widths at
  h=1 and h=96 can be compared; ``None`` where the mean absolute residual is zero.
* **expected calibration error** for intervals: rows are binned by predicted width
  quantile, and within each bin the gap between nominal and empirical coverage is
  accumulated weighted by bin size. Reported alongside per-bin rows, never alone.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Mapping

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - import cycle guard for the type hint only
    from .intervals import PredictionInterval

__all__ = [
    "IntervalMetrics",
    "coverage",
    "coverage_error",
    "mean_width",
    "interval_score",
    "weighted_interval_score",
    "sharpness_ratio",
    "expected_calibration_error",
    "per_level_metrics",
]


def _as_pair(
    actual_kw: np.ndarray, lower_kw: np.ndarray, upper_kw: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    actual = np.asarray(actual_kw, dtype=np.float64)
    lower = np.asarray(lower_kw, dtype=np.float64)
    upper = np.asarray(upper_kw, dtype=np.float64)
    if lower.shape != upper.shape:
        raise ValueError(
            f"lower {lower.shape} and upper {upper.shape} must have the same shape"
        )
    try:
        actual = np.broadcast_to(actual, lower.shape)
    except ValueError as exc:
        raise ValueError(
            f"actual {actual.shape} cannot be broadcast to the bounds' shape "
            f"{lower.shape}"
        ) from exc
    if np.any(lower > upper):
        raise ValueError(
            f"{int(np.count_nonzero(lower > upper))} intervals have lower > upper"
        )
    return actual, lower, upper


def coverage(
    actual_kw: np.ndarray, lower_kw: np.ndarray, upper_kw: np.ndarray
) -> float:
    """Fraction of realised values inside the closed interval."""
    actual, lower, upper = _as_pair(actual_kw, lower_kw, upper_kw)
    if actual.size == 0:
        return float("nan")
    return float(np.mean((actual >= lower) & (actual <= upper)))


def coverage_error(observed: float, nominal_level: float) -> float:
    """``observed - nominal``. Negative means under-covering."""
    return float(observed - float(nominal_level))


def mean_width(lower_kw: np.ndarray, upper_kw: np.ndarray) -> float:
    """Mean ``upper - lower``, in kW."""
    lower, upper = _as_pair(np.zeros(1), lower_kw, upper_kw)[1:]
    if lower.size == 0:
        return float("nan")
    return float(np.mean(upper - lower))


def interval_score(
    actual_kw: np.ndarray,
    lower_kw: np.ndarray,
    upper_kw: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Winkler / interval score, elementwise.

    Args:
        actual_kw: Realised values.
        lower_kw: Lower bounds.
        upper_kw: Upper bounds.
        alpha: Miscoverage, in ``(0, 1)``.

    Returns:
        Per-element scores, the same shape as the bounds.

    Raises:
        ValueError: If ``alpha`` is outside ``(0, 1)``.
    """
    if not 0.0 < float(alpha) < 1.0:
        raise ValueError(f"alpha must lie in (0, 1), got {alpha}")
    actual, lower, upper = _as_pair(actual_kw, lower_kw, upper_kw)
    width = upper - lower
    below = (lower - actual) * (actual < lower)
    above = (actual - upper) * (actual > upper)
    return width + (2.0 / float(alpha)) * (below + above)


def weighted_interval_score(
    actual_kw: np.ndarray,
    bounds: dict[float, tuple[np.ndarray, np.ndarray]],
) -> float:
    """The WIS over several nominal levels.

    Args:
        actual_kw: Realised values.
        bounds: ``{nominal_level: (lower, upper)}``.

    Returns:
        The weighted average interval score, in kW.

    Raises:
        ValueError: If ``bounds`` is empty.
    """
    if not bounds:
        raise ValueError("weighted_interval_score needs at least one level")
    weights: list[float] = []
    scores: list[float] = []
    for nominal, (lower, upper) in sorted(bounds.items()):
        alpha = 1.0 - float(nominal)
        scores.append(float(interval_score(actual_kw, lower, upper, alpha).mean()))
        weights.append(alpha / 2.0)
    total = float(sum(weights))
    if total <= 0:  # pragma: no cover - impossible for levels in (0, 1)
        return float("nan")
    return float(np.dot(scores, weights) / total)


def sharpness_ratio(
    lower_kw: np.ndarray, upper_kw: np.ndarray, absolute_error_kw: np.ndarray
) -> float | None:
    """Mean interval width per unit of mean absolute error. Dimensionless.

    Returns:
        The ratio, or ``None`` when the mean absolute error is zero.
    """
    lower, upper = _as_pair(np.zeros(1), lower_kw, upper_kw)[1:]
    error = np.asarray(absolute_error_kw, dtype=np.float64)
    denominator = float(np.mean(error))
    if denominator <= 0.0:
        return None
    return float(np.mean(upper - lower) / denominator)


def expected_calibration_error(
    actual_kw: np.ndarray,
    lower_kw: np.ndarray,
    upper_kw: np.ndarray,
    nominal_level: float,
    *,
    scale: np.ndarray | None = None,
    bins: int = 10,
) -> dict[str, Any]:
    """Coverage error binned by predicted width, and its size-weighted mean.

    A single coverage number hides the most operationally important failure: a method can
    have correct *average* coverage while being badly over-confident on exactly the rows
    that are easy. Binning by the predicted uncertainty is how that shows up.

    Args:
        actual_kw: Realised values.
        lower_kw: Lower bounds.
        upper_kw: Upper bounds.
        nominal_level: Stated coverage.
        scale: The uncertainty scalar to bin by. Defaults to interval width. Pass the
            scale function's own output to bin by *predicted* uncertainty rather than by
            the realised interval width, which also depends on the multiplier.
        bins: Number of equal-count bins.

    Returns:
        ``{"bins": [...], "expected_calibration_error": float, ...}``.

    Raises:
        ValueError: If ``bins`` is not positive.
    """
    if int(bins) <= 0:
        raise ValueError(f"bins must be positive, got {bins}")
    actual, lower, upper = _as_pair(actual_kw, lower_kw, upper_kw)
    hit = (actual >= lower) & (actual <= upper)
    ordering = (upper - lower) if scale is None else np.broadcast_to(
        np.asarray(scale, dtype=np.float64), lower.shape
    )
    flat_order = np.asarray(ordering, dtype=np.float64).ravel()
    flat_hit = hit.ravel()
    total = flat_hit.size
    if total == 0:
        return {
            "bins": [],
            "expected_calibration_error": float("nan"),
            "nominal_level": float(nominal_level),
            "binned_by": "interval width",
            "n": 0,
        }

    edges = np.quantile(flat_order, np.linspace(0.0, 1.0, int(bins) + 1))
    edges[-1] = min(float(edges[-1]), float(flat_order.max()))
    rows: list[dict[str, Any]] = []
    accumulator = 0.0
    for index in range(int(bins)):
        low = float(edges[index])
        high = float(edges[index + 1])
        if index == int(bins) - 1:
            mask = (flat_order >= low) & (flat_order <= high)
        else:
            mask = (flat_order >= low) & (flat_order < high)
        if not np.any(mask):
            continue
        observed = float(np.mean(flat_hit[mask]))
        rows.append(
            {
                "bin": index,
                "scale_low": round(low, 6),
                "scale_high": round(high, 6),
                "rows": int(np.count_nonzero(mask)),
                "share_of_rows": float(np.mean(mask)),
                "empirical_coverage": round(observed, 6),
                "coverage_error": round(observed - float(nominal_level), 6),
            }
        )
        accumulator += abs(observed - float(nominal_level)) * np.count_nonzero(mask)
    return {
        "bins": rows,
        "expected_calibration_error": float(accumulator / total),
        "nominal_level": float(nominal_level),
        "binned_by": "predicted scale" if scale is not None else "interval width",
        "n": int(total),
    }


@dataclass(frozen=True, slots=True)
class IntervalMetrics:
    """Every calibration and sharpness measure for one method at one horizon and level.

    Attributes:
        nominal_level: Stated coverage.
        n: Rows scored.
        coverage: Measured coverage.
        coverage_error: ``coverage - nominal_level``.
        mean_width_kw: Mean interval width.
        median_width_kw: Median interval width.
        sharpness_ratio: Mean width per unit of mean absolute error.
        interval_score_kw: Winkler score at this level.
        mean_lower_kw / mean_upper_kw: Bound means, which expose a lopsided interval.
        clipped_share: Share of lower bounds raised to zero.
        calibration: The binned reliability record from
            :func:`expected_calibration_error`.
    """

    nominal_level: float
    n: int
    coverage: float
    coverage_error: float
    mean_width_kw: float
    median_width_kw: float
    sharpness_ratio: float | None
    interval_score_kw: float
    mean_lower_kw: float
    mean_upper_kw: float
    clipped_share: float
    calibration: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "nominal_level": self.nominal_level,
            "n": self.n,
            "coverage": self.coverage,
            "coverage_error": self.coverage_error,
            "mean_width_kw": self.mean_width_kw,
            "median_width_kw": self.median_width_kw,
            "sharpness_ratio": self.sharpness_ratio,
            "interval_score_kw": self.interval_score_kw,
            "mean_lower_kw": self.mean_lower_kw,
            "mean_upper_kw": self.mean_upper_kw,
            "clipped_share": self.clipped_share,
            "calibration": self.calibration,
        }

    def passes(self, tolerance: float) -> bool:
        """Whether coverage is within ``tolerance`` of nominal, in either direction.

        ``tolerance`` is an absolute coverage tolerance - 0.02 means "within two
        percentage points". It is passed in rather than defaulted so that the criterion
        appears in every artifact and cannot be changed silently.
        """
        return abs(self.coverage_error) <= float(tolerance)


def per_level_metrics(
    *,
    actual_kw: np.ndarray,
    point_kw: np.ndarray,
    bounds: Mapping[float, "PredictionInterval"],
    nominal_levels: tuple[float, ...],
    absolute_error_kw: np.ndarray,
    scale: np.ndarray | None = None,
    bins: int = 10,
) -> dict[str, IntervalMetrics]:
    """Every metric for every level of one method at one horizon.

    The coverage tolerance is **not** a parameter here. It belongs to the verdict, which
    is computed once for the whole phase from the sample size, not per cell; letting each
    call choose it would make "pass" a per-cell decision that drifts between tables.

    Args:
        actual_kw: Realised values.
        point_kw: The point forecast the intervals were built around.
        bounds: ``{nominal_level: PredictionInterval}``.
        nominal_levels: Levels to report.
        absolute_error_kw: ``|actual - point|``, for the sharpness ratio.
        scale: Predicted uncertainty scalar for binning.
        bins: Reliability bins.

    Returns:
        ``{str(nominal_level): IntervalMetrics}``.

    Raises:
        ValueError: If a requested level has no interval.
    """
    del point_kw  # The interval already carries the point it was built around.
    out: dict[str, IntervalMetrics] = {}
    for nominal in nominal_levels:
        key = float(nominal)
        interval = bounds.get(key)
        if interval is None:
            raise ValueError(
                f"no interval at nominal level {nominal}; have {sorted(bounds)}"
            )
        lower = np.asarray(interval.lower_kw, dtype=np.float64)
        upper = np.asarray(interval.upper_kw, dtype=np.float64)
        observed = coverage(actual_kw, lower, upper)
        width = upper - lower
        out[str(nominal)] = IntervalMetrics(
            nominal_level=key,
            n=int(np.asarray(actual_kw).size),
            coverage=observed,
            coverage_error=coverage_error(observed, key),
            mean_width_kw=float(width.mean()),
            median_width_kw=float(np.median(width)),
            sharpness_ratio=sharpness_ratio(lower, upper, absolute_error_kw),
            interval_score_kw=float(
                interval_score(actual_kw, lower, upper, 1.0 - key).mean()
            ),
            mean_lower_kw=float(lower.mean()),
            mean_upper_kw=float(upper.mean()),
            clipped_share=float(np.mean(lower <= 0.0)),
            calibration=expected_calibration_error(
                actual_kw,
                lower,
                upper,
                key,
                scale=scale,
                bins=bins,
            ),
        )
    return out