"""How good is a flexibility envelope? Six questions, each with a stated definition.

Definitions, stated precisely because "coverage" alone would hide the trade-off
----------------------------------------------------------------------

**Calibration** is the fraction of realised demands inside the claimed band
``[expected - downward, expected + upward]``. A calibrated envelope at nominal ``L`` achieves
``L``. Reported per direction as well, because an envelope can be calibrated overall while
being systematically wrong on one side - and for a system that wants to shed load, only the
downward side is actionable.

**Sharpness** is the mean width. Reported *relative to the signal*
(``mean_width / mean_absolute_deviation``) as well as absolute, because a wide envelope on a
wild load is not the same problem as a wide envelope on a flat one. A ratio near 1 means the
envelope is about as wide as the thing it is trying to bound, which is the regime where
sharpness matters.

**Winkler interval score** is reused from :mod:`uncertainty.metrics` rather than
reimplemented: it already scores coverage and width with published weights, and a second
implementation of the same rule would eventually disagree with the first. The band is a
prediction interval for realised demand centred on the *baseline*, not on the forecast, which
is what makes it a legitimate target for the score.

**Stability** asks whether the claimed envelope jitters more than reality. Two numbers: the
lag-1 autocorrelation of the claimed width across consecutive rows of a series (near 1 is
smooth, near 0 is jitter), and the Spearman correlation between the row-to-row change in the
claimed width and the row-to-row change in realised demand. A high second number means the
envelope moves when the world moves, which is what a downstream optimiser needs.

**Directionality** asks whether the method can tell "has run high" from "has run low". An
unconditional symmetric envelope would score the same on coverage while being unable to
answer the question at all, so the ratio of mean upward to mean downward magnitude is
reported, and it is checked against the observed asymmetry.

**Regime sensitivity** reuses Phase 4/5's tercile axes, which were derived from the target
and are therefore an *evaluation* grouping and never a feature. A method whose width does not
respond to regime is not wrong, but a downstream planner would learn nothing from it.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..uncertainty.evaluation import reliability_table
from ..uncertainty.metrics import (
    expected_calibration_error,
    interval_score,
    weighted_interval_score,
)

__all__ = [
    "directional_balance",
    "evaluate_band",
    "stability_report",
]


def evaluate_band(
    *,
    expected_kw: np.ndarray,
    observed_kw: np.ndarray,
    upward_kw: np.ndarray,
    downward_kw: np.ndarray,
    nominal_level: float,
    bands_by_level: dict[float, tuple[np.ndarray, np.ndarray]] | None = None,
    bins: int = 10,
) -> dict[str, Any]:
    """Score one envelope application against realised demand.

    Args:
        expected_kw: ``[rows]`` the baseline the band is centred on, kW.
        observed_kw: ``[rows]`` realised demand at the predicted step, kW.
        upward_kw: ``[rows]`` upward magnitudes, kW.
        downward_kw: ``[rows]`` downward magnitudes, kW.
        nominal_level: The level this band claims.
        bands_by_level: ``{nominal_level: (lower, upper)}`` for the weighted interval score,
            when several levels were estimated. The headline band's own
            ``(lower, upper)`` is always included. Supplying a scaled-down copy of the
            headline band for the other levels would produce a weighted score over bands
            that were never estimated, so the caller passes the real ones or none.
        bins: Bins for the reliability table.

    Returns:
        Coverage, per-direction coverage, sharpness, interval score and reliability.

    Raises:
        ValueError: If the arrays disagree in length, or are not finite.
    """
    expected = np.asarray(expected_kw, dtype=np.float64).ravel()
    observed = np.asarray(observed_kw, dtype=np.float64).ravel()
    up = np.asarray(upward_kw, dtype=np.float64).ravel()
    down = np.asarray(downward_kw, dtype=np.float64).ravel()
    lengths = {expected.size, observed.size, up.size, down.size}
    if len(lengths) != 1:
        raise ValueError(
            f"expected {expected.size}, observed {observed.size}, up {up.size}, down "
            f"{down.size} rows; all four must be the same length"
        )
    for name, array in (
        ("expected_kw", expected),
        ("observed_kw", observed),
        ("upward_kw", up),
        ("downward_kw", down),
    ):
        if not np.isfinite(array).all():
            raise ValueError(f"{name} contains non-finite values")
    if (up < 0).any() or (down < 0).any():
        raise ValueError("magnitudes must be non-negative")

    lower = expected - down
    upper = expected + up
    inside = (observed >= lower) & (observed <= upper)
    above = observed > expected
    below = observed < expected
    width = up + down
    deviation = np.abs(observed - expected)

    bounds: dict[float, tuple[np.ndarray, np.ndarray]] = {
        float(nominal_level): (lower, upper)
    }
    for level, pair in (bands_by_level or {}).items():
        low = np.asarray(pair[0], dtype=np.float64).ravel()
        high = np.asarray(pair[1], dtype=np.float64).ravel()
        if low.size != expected.size or high.size != expected.size:
            raise ValueError(
                f"band at level {level} has {low.size}/{high.size} rows, expected {expected.size}"
            )
        bounds[float(level)] = (low, high)
    winkler = float(
        np.mean(interval_score(observed, lower, upper, 1.0 - float(nominal_level)))
    )

    coverage = float(np.mean(inside))
    upward_rows = int(above.sum())
    downward_rows = int(below.sum())
    return {
        "nominal_level": float(nominal_level),
        "n": int(expected.size),
        "coverage": coverage,
        "coverage_error": coverage - float(nominal_level),
        "mean_width_kw": float(np.mean(width)),
        "median_width_kw": float(np.median(width)),
        "width_over_mean_absolute_deviation": (
            float(np.mean(width) / np.mean(deviation)) if deviation.mean() > 0 else None
        ),
        "mean_absolute_deviation_kw": float(np.mean(deviation)),
        "winkler_interval_score_kw": winkler,
        "weighted_interval_score_kw": float(weighted_interval_score(observed, bounds)),
        "levels_scored": sorted(bounds),
        "sharpness_ratio_vs_point_mae": (
            float(np.mean(width) / deviation.mean()) if deviation.mean() > 0 else None
        ),
        "upward": {
            "rows": upward_rows,
            "share": float(upward_rows / expected.size) if expected.size else 0.0,
            "covered": (
                float(np.mean(inside[above])) if upward_rows else None
            ),
            "conditional_coverage": (
                float(above.sum() and np.sum(inside & above) / upward_rows)
                if upward_rows
                else None
            ),
        },
        "downward": {
            "rows": downward_rows,
            "share": float(downward_rows / expected.size) if expected.size else 0.0,
            "covered": float(np.mean(inside[below])) if downward_rows else None,
            "conditional_coverage": (
                float(np.sum(inside & below) / downward_rows) if downward_rows else None
            ),
        },
        "expected_calibration_error": expected_calibration_error(
            observed, lower, upper, float(nominal_level), bins=bins
        ),
        "reliability": reliability_table(
            actual_kw=observed,
            lower_kw=lower,
            upper_kw=upper,
            nominal_level=float(nominal_level),
            uncertainty=width,
            bins=bins,
        ),
    }


def directional_balance(
    *, upward_kw: np.ndarray, downward_kw: np.ndarray
) -> dict[str, Any]:
    """Whether the envelope distinguishes the two directions.

    Args:
        upward_kw: ``[rows]`` upward magnitudes.
        downward_kw: ``[rows]`` downward magnitudes.

    Returns:
        The mean of each, their ratio, and whether the asymmetry is meaningful.

    Raises:
        ValueError: If lengths disagree.
    """
    up = np.asarray(upward_kw, dtype=np.float64).ravel()
    down = np.asarray(downward_kw, dtype=np.float64).ravel()
    if up.size != down.size:
        raise ValueError(f"upward {up.size} and downward {down.size} must be the same length")
    mean_up = float(up.mean()) if up.size else 0.0
    mean_down = float(down.mean()) if down.size else 0.0
    ratio = float(mean_up / mean_down) if mean_down > 0 else None
    return {
        "mean_upward_kw": mean_up,
        "mean_downward_kw": mean_down,
        "ratio": ratio,
        "is_symmetric": bool(ratio is not None and 0.9 <= ratio <= 1.1),
        "interpretation": (
            "the two directions carry materially different magnitudes, so a single "
            "symmetric figure would misrepresent one of them"
            if ratio is not None and (ratio < 0.9 or ratio > 1.1)
            else "the two directions are close enough that a symmetric figure would not "
            "misrepresent either"
        ),
    }


def stability_report(
    *,
    width_kw: np.ndarray,
    observed_kw: np.ndarray,
    series: np.ndarray,
    spearman,
) -> dict[str, Any]:
    """Whether the claimed envelope moves smoothly, and tracks what actually happened.

    Args:
        width_kw: ``[rows]`` claimed width per row.
        observed_kw: ``[rows]`` realised demand per row.
        series: ``[rows]`` series index per row, so differences are taken within a series and
            not across a boundary between two different customers.
        spearman: Rank-correlation callable, injected so this module is statistics-agnostic.

    Returns:
        Lag-1 width autocorrelation, the width-change/error-change correlation, and the
        share of rows where the width moved at all.

    Raises:
        ValueError: If the arrays disagree in length.
    """
    width = np.asarray(width_kw, dtype=np.float64).ravel()
    observed = np.asarray(observed_kw, dtype=np.float64).ravel()
    index = np.asarray(series, dtype=np.int64).ravel()
    if not (width.size == observed.size == index.size):
        raise ValueError(
            f"width {width.size}, observed {observed.size} and series {index.size} must be "
            f"the same length"
        )
    if width.size < 3:
        return {
            "width_lag1_autocorrelation": None,
            "width_change_vs_error_change_spearman": None,
            "note": "fewer than three rows, so no stability statistic is defined",
        }
    # Lag-1 is computed **within each series**, for the same reason the differences below
    # are. The panel is ordered series-major, so consecutive rows are usually two different
    # customers: a lag-1 taken across the raw row order measures how much one customer's
    # envelope differs from the next one's, which is a statement about the customer mix and
    # not about whether the envelope is smooth in time. On this data that mistake reported
    # an autocorrelation near 0.16 for an envelope whose width is in fact constant per
    # customer, which is exactly the kind of wrong-but-plausible number a stability check
    # exists to catch. Each series is centred by its own mean and the within-series
    # products are then pooled.
    products = 0.0
    squares = 0.0
    for value in np.unique(index):
        member = width[index == value]
        if member.size < 2:
            continue
        local = member - member.mean()
        products += float(np.sum(local[:-1] * local[1:]))
        squares += float(np.sum(local**2))
    lag1 = (products / squares) if squares > 0 else None
    # A width with no within-series variation at all is the smoothest envelope there is, so
    # it scores 1.0 rather than reporting an undefined statistic. Returning ``None`` here
    # would be read downstream as "no stability evidence", which for a constant band is the
    # opposite of the truth - it is the strongest possible stability evidence available.
    constant_within_series = squares == 0.0
    if constant_within_series:
        lag1 = 1.0
    same = index[:-1] == index[1:]
    width_change = np.abs(np.diff(width))[same]
    error_change = np.abs(np.diff(np.abs(observed)))[same]
    rho = float("nan")
    if width_change.size > 2 and width_change.std() > 0 and error_change.std() > 0:
        rho = float(spearman(width_change, error_change))
    return {
        "width_lag1_autocorrelation": lag1,
        "width_change_vs_error_change_spearman": None if rho != rho else rho,
        "mean_absolute_width_change_kw": (
            float(width_change.mean()) if width_change.size else None
        ),
        "pairs_compared": int(width_change.size),
        "share_width_changed": (
            float(np.mean(width_change > 1e-9)) if width_change.size else None
        ),
        "width_is_constant_within_series": constant_within_series,
        "lag1_computed_within_series": True,
        "interpretation": (
            "a lag-1 autocorrelation near 1 means the claimed envelope is smooth in time; "
            "near 0 means it jitters, which would make it unusable by a downstream planner "
            "even if its coverage were correct"
        ),
    }