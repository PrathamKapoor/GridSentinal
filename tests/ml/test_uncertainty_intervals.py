"""Interval arithmetic: the one place every method's bounds are built.

The properties tested here are the ones a wrong answer would violate silently. A method
whose coverage is off by a percentage point is a *result*; a method whose lower bound
exceeds its upper bound, or whose nominal level is not the coverage it achieves, is a
*bug*, and these tests are what separates the two.
"""

from __future__ import annotations

import numpy as np
import pytest

from energy_intelligence.ml.uncertainty.intervals import (
    DEFAULT_NOMINAL_LEVELS,
    PredictionInterval,
    clip_lower_at_zero,
    empirical_quantile,
    lower_from_point,
    quantile_levels,
    widths_from_scale,
)


def test_quantile_levels_splits_the_tail_in_half():
    """A nominal level's two tails must bracket exactly that level's mass.

    This is the assertion that would have caught the one-sided convention: with
    ``(1 - level, level)`` a nominal 90% interval covers 80%, and a nominal 50% interval
    collapses to the median with zero width.
    """
    for level in DEFAULT_NOMINAL_LEVELS:
        lower, upper = quantile_levels((level,))[0]
        assert lower == pytest.approx((1.0 - level) / 2.0)
        assert upper == pytest.approx((1.0 + level) / 2.0)
        assert upper - lower == pytest.approx(level)
        assert lower < upper


def test_quantile_levels_rejects_out_of_range_and_empty():
    with pytest.raises(ValueError, match="at least one"):
        quantile_levels(())
    with pytest.raises(ValueError, match=r"lie in \(0, 1\)"):
        quantile_levels((0.0,))
    with pytest.raises(ValueError, match=r"lie in \(0, 1\)"):
        quantile_levels((1.0, 1.5))


def test_empirical_quantile_matches_numpy():
    values = np.linspace(-3.0, 5.0, 101)
    assert empirical_quantile(values, 0.5) == pytest.approx(float(np.quantile(values, 0.5)))
    assert empirical_quantile(np.zeros(0), 0.9, default=-1.0) == -1.0
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        empirical_quantile(values, 1.5)


def test_widths_from_scale_is_symmetric_about_the_point():
    point = np.array([[10.0, 20.0]])
    scale = np.array([[2.0, 5.0]])
    interval = widths_from_scale(
        point_kw=point, scale=scale, multiplier=1.5, method="m", nominal_level=0.9
    )
    assert interval.symmetric
    assert np.allclose(interval.lower_kw, point - multiplier_scale(scale, 1.5))
    assert np.allclose(interval.upper_kw, point + multiplier_scale(scale, 1.5))
    assert np.allclose(interval.width_kw, multiplier_scale(scale, 3.0))


def multiplier_scale(scale: np.ndarray, multiplier: float) -> np.ndarray:
    return float(multiplier) * np.asarray(scale, dtype=np.float64)


def test_widths_from_scale_rejects_shape_and_sign_violations():
    with pytest.raises(ValueError, match="same shape"):
        widths_from_scale(
            point_kw=np.zeros((3, 2)), scale=np.ones((3, 1)), multiplier=1.0,
            method="m", nominal_level=0.9,
        )
    with pytest.raises(ValueError, match="non-negative"):
        widths_from_scale(
            point_kw=np.zeros((1, 1)), scale=np.array([[-1.0]]), multiplier=1.0,
            method="m", nominal_level=0.9,
        )
    with pytest.raises(ValueError, match="multiplier must be non-negative"):
        widths_from_scale(
            point_kw=np.zeros((1, 1)), scale=np.ones((1, 1)), multiplier=-1.0,
            method="m", nominal_level=0.9,
        )


def test_zero_clip_fires_when_the_lower_bound_would_be_negative():
    interval = widths_from_scale(
        point_kw=np.array([[0.5]]), scale=np.array([[2.0]]), multiplier=1.0,
        method="m", nominal_level=0.9,
    )
    assert interval.clipped_at_zero
    assert float(interval.lower_kw[0, 0]) == 0.0
    assert float(interval.upper_kw[0, 0]) == pytest.approx(2.5)


def test_zero_clip_leaves_an_entirely_negative_interval_alone():
    """Clipping ``[negative, negative]`` to zero would assert a zero-width certainty.

    Phase 7's fixed ensemble produces a few slightly negative point forecasts, so this is a
    real case. The honest answer keeps the interval and reports that it is unphysical,
    rather than inverting it or collapsing it to ``[0, 0]``.
    """
    interval = widths_from_scale(
        point_kw=np.array([[-0.67]]), scale=np.array([[0.2]]), multiplier=2.0,
        method="m", nominal_level=0.9,
    )
    assert float(interval.lower_kw[0, 0]) == pytest.approx(-1.07)
    assert float(interval.upper_kw[0, 0]) == pytest.approx(-0.27)
    assert float(interval.width_kw[0, 0]) > 0.0


def test_clip_lower_at_zero_is_a_no_op_when_disabled():
    lower, upper = np.array([[-3.0, 1.0]]), np.array([[5.0, 2.0]])
    kept, clipped = clip_lower_at_zero(lower, upper, enabled=False)
    assert not clipped
    assert np.allclose(kept, lower)
    fixed, clipped = clip_lower_at_zero(lower, upper, enabled=True)
    assert clipped
    assert np.allclose(fixed, [[0.0, 1.0]])


def test_prediction_interval_rejects_an_inverted_interval():
    with pytest.raises(ValueError, match="lower_kw exceeds upper_kw"):
        PredictionInterval(
            lower_kw=np.array([[2.0]]), upper_kw=np.array([[1.0]]),
            nominal_level=0.9, method="m",
        )


def test_prediction_interval_rejects_mismatched_shapes_and_bad_metadata():
    with pytest.raises(ValueError, match="same shape"):
        PredictionInterval(
            lower_kw=np.zeros((2, 2)), upper_kw=np.zeros((2, 1)),
            nominal_level=0.9, method="m",
        )
    with pytest.raises(ValueError, match="nominal_level must lie"):
        PredictionInterval(
            lower_kw=np.zeros((1, 1)), upper_kw=np.ones((1, 1)),
            nominal_level=1.0, method="m",
        )
    with pytest.raises(ValueError, match="method must be a non-empty string"):
        PredictionInterval(
            lower_kw=np.zeros((1, 1)), upper_kw=np.ones((1, 1)),
            nominal_level=0.9, method="  ",
        )
    with pytest.raises(ValueError, match="must be finite"):
        PredictionInterval(
            lower_kw=np.array([[np.nan]]), upper_kw=np.ones((1, 1)),
            nominal_level=0.9, method="m",
        )


def test_lower_from_point_allows_asymmetry_but_not_inversion():
    point = np.array([[10.0]])
    interval = lower_from_point(
        point_kw=point,
        lower_delta_kw=np.array([[-3.0]]),
        upper_delta_kw=np.array([[1.0]]),
        method="q",
        nominal_level=0.9,
    )
    assert not interval.symmetric
    assert float(interval.lower_kw[0, 0]) == pytest.approx(7.0)
    assert float(interval.upper_kw[0, 0]) == pytest.approx(11.0)
    with pytest.raises(ValueError, match="lower_kw exceeds upper_kw"):
        lower_from_point(
            point_kw=point,
            lower_delta_kw=np.array([[3.0]]),
            upper_delta_kw=np.array([[-1.0]]),
            method="q",
            nominal_level=0.9,
        )


def test_contains_and_midpoint_agree_with_the_bounds():
    interval = PredictionInterval(
        lower_kw=np.array([[1.0, 4.0]]),
        upper_kw=np.array([[3.0, 6.0]]),
        nominal_level=0.9,
        method="m",
    )
    assert interval.contains(np.array([[1.0, 6.0]])).tolist() == [[True, True]]
    assert interval.contains(np.array([[0.9, 5.9]])).tolist() == [[False, True]]
    assert interval.contains(np.array([[3.0, 6.1]])).tolist() == [[True, False]]
    assert np.allclose(interval.midpoint_kw, [[2.0, 5.0]])
    assert interval.n_rows == 2


def test_to_dict_summarises_rather_than_dumping_the_array():
    interval = widths_from_scale(
        point_kw=np.full((50, 1), 10.0), scale=np.ones((50, 1)), multiplier=2.0,
        method="m", nominal_level=0.9,
    )
    payload = interval.to_dict()
    assert payload["n"] == 50
    assert payload["mean_width_kw"] == pytest.approx(4.0)
    assert "lower_kw" not in payload


def test_a_nominal_50_interval_is_not_the_median_alone():
    """A central 50% interval needs two distinct quantiles.

    The one-sided convention mapped both tails of a nominal 50% interval onto the median,
    so the interval had no width at all - which is a bug that would have shown up only as
    an oddly narrow band in a coverage table.
    """
    grid = np.linspace(-1.0, 1.0, 1001)
    lower_q, upper_q = quantile_levels((0.5,))[0]
    low = empirical_quantile(grid, lower_q)
    high = empirical_quantile(grid, upper_q)
    assert high > low
    # A uniform grid over [-1, 1] has range 2, so a central half of its *mass* spans half
    # its range.
    assert high - low == pytest.approx(1.0, abs=0.02)