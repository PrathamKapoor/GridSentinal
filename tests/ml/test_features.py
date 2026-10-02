"""Tests for feature construction, availability policy and leakage detection.

The most important test in this file is
:func:`test_the_leakage_checker_catches_a_deliberately_leaking_feature`. A leakage
guard that cannot fail is worse than none, because it buys false confidence, so the
guard is itself tested against a feature that really does read the future.
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import build_series, build_weather

from energy_intelligence.ml.features import (
    FEATURE_CATALOGUE,
    WEATHER_AVAILABLE_THROUGH_STEP,
    assert_no_future_access,
    available_feature_names,
    build_feature_matrix,
    feature_catalogue,
)


# ======================================================================
# Catalogue
# ======================================================================


def test_every_feature_declares_its_provenance() -> None:
    for spec in feature_catalogue():
        assert spec.formula, spec.name
        assert spec.reason, spec.name
        assert spec.source, spec.name
        assert spec.availability in {"at_or_before_origin", "origin_only"}, spec.name
        assert spec.value_kind in {"HISTORICAL", "CALENDAR", "WEATHER_ACTUAL"}, spec.name


def test_only_weather_features_require_weather() -> None:
    for spec in feature_catalogue():
        assert spec.requires_weather == (spec.value_kind == "WEATHER_ACTUAL"), spec.name


def test_value_at_origin_exists_because_persistence_sees_it() -> None:
    """The feature set must not be weaker than the baseline it is compared to."""
    names = {spec.name for spec in feature_catalogue()}
    assert "value_at_origin" in names
    spec = next(s for s in feature_catalogue() if s.name == "value_at_origin")
    assert spec.formula == "y[t]"


# ======================================================================
# Availability policy
# ======================================================================


def test_weather_is_withdrawn_beyond_the_first_step() -> None:
    """Actual weather is not knowable at a 4-hour horizon; there is no forecast."""
    names, withdrawn = available_feature_names(
        lookback=672, max_horizon=4, has_weather=True
    )
    assert not any(name.endswith("_at_origin") and name.startswith("ghi") for name in names)
    reasons = dict(withdrawn)
    assert "weather FORECAST" in reasons["ghi_at_origin"]


def test_weather_is_available_for_a_15_minute_horizon() -> None:
    names, _ = available_feature_names(lookback=672, max_horizon=1, has_weather=True)
    assert "ghi_at_origin" in names
    assert "poa_irradiance_at_origin" in names


def test_weather_is_withdrawn_when_no_weather_was_extracted() -> None:
    names, withdrawn = available_feature_names(
        lookback=672, max_horizon=1, has_weather=False
    )
    reasons = dict(withdrawn)
    assert "no weather columns" in reasons["ghi_at_origin"]
    assert "ghi_at_origin" not in names


def test_features_needing_more_history_than_the_lookback_are_withdrawn() -> None:
    names, withdrawn = available_feature_names(
        lookback=96, max_horizon=1, has_weather=False
    )
    reasons = dict(withdrawn)
    assert "lag_672" in reasons
    assert "672" in reasons["lag_672"]
    assert "lag_672" not in names
    assert "lag_96" in names


def test_every_withdrawal_records_a_reason() -> None:
    _, withdrawn = available_feature_names(lookback=96, max_horizon=4, has_weather=False)
    for name, reason in withdrawn:
        assert name and reason


def test_weather_threshold_is_one_step() -> None:
    assert WEATHER_AVAILABLE_THROUGH_STEP == 1


# ======================================================================
# Formulas
# ======================================================================


def _matrix(series, origins, names, weather=None, series_row=0):
    """Build a feature matrix for one series' origins."""
    return build_feature_matrix(
        series.values,
        origins,
        lookback=672,
        feature_names=names,
        weather=weather,
        series_index=np.full(len(origins), series_row, dtype=np.int64),
    )


def test_lag_formulas_are_exact(synthetic_series) -> None:
    origins = np.array([700, 800, 900, 1000])
    names = ("value_at_origin", "lag_1", "lag_4", "lag_96")
    matrix = _matrix(synthetic_series, origins, names)
    values = synthetic_series.values
    assert np.allclose(matrix[:, 0], values[0, origins])
    assert np.allclose(matrix[:, 1], values[0, origins - 1])
    assert np.allclose(matrix[:, 2], values[0, origins - 4])
    assert np.allclose(matrix[:, 3], values[0, origins - 96])


def test_ramp_is_a_derived_difference(synthetic_series) -> None:
    origins = np.array([700, 900])
    matrix = _matrix(synthetic_series, origins, ("lag_1", "ramp_1"))
    values = synthetic_series.values[0]
    assert np.allclose(matrix[:, 1], values[origins - 1] - values[origins - 2])


def test_trailing_window_excludes_the_origin(synthetic_series) -> None:
    """A rolling mean over [t-96, t-1] must not see y[t]."""
    origins = np.array([800, 1000])
    matrix = _matrix(synthetic_series, origins, ("roll_mean_96",))
    values = synthetic_series.values[0]
    for row, origin in enumerate(origins):
        expected = values[origin - 96 : origin].mean()
        assert matrix[row, 0] == pytest.approx(expected)
        assert matrix[row, 0] != pytest.approx(values[origin - 95 : origin + 1].mean())


def test_seven_day_mean_uses_seven_points(synthetic_series) -> None:
    origins = np.array([1500, 2000])
    matrix = _matrix(synthetic_series, origins, ("roll_mean_same_hour_7d",))
    values = synthetic_series.values[0]
    for row, origin in enumerate(origins):
        expected = np.mean([values[origin - 96 * k] for k in range(1, 8)])
        assert matrix[row, 0] == pytest.approx(expected)


def test_calendar_features_are_derived_from_the_grid_index(synthetic_series) -> None:
    origins = np.array([96 * 10 + 4 * 13, 96 * 12 + 4 * 2])
    matrix = _matrix(synthetic_series, origins, ("hour_of_day", "day_of_week", "day_of_year", "is_weekend"))
    assert matrix[0, 0] == pytest.approx(13.0)
    assert matrix[0, 2] == pytest.approx(10.0)
    assert matrix[0, 3] == pytest.approx(float(matrix[0, 1] >= 5))
    assert matrix[1, 0] == pytest.approx(2.0)
    assert matrix[1, 2] == pytest.approx(12.0)


def test_weather_features_read_the_origin_instant(synthetic_series) -> None:
    weather = build_weather()
    origins = np.array([700, 900])
    names = ("ghi_at_origin", "temperature_at_origin")
    matrix = _matrix(synthetic_series, origins, names, weather=weather)
    assert np.allclose(matrix[:, 0], weather["GHI"][origins])
    assert np.allclose(matrix[:, 1], weather["Temperature"][origins])


def test_unknown_feature_is_refused(synthetic_series) -> None:
    with pytest.raises(KeyError, match="unknown feature"):
        _matrix(synthetic_series, np.array([800]), ("phase_of_the_moon",))


def test_origin_before_the_record_is_refused(synthetic_series) -> None:
    with pytest.raises(ValueError, match="earlier than the earliest offset"):
        _matrix(synthetic_series, np.array([100]), ("lag_672",))


def test_non_15_minute_grid_is_refused(synthetic_series) -> None:
    """A steps-per-day that is not a whole number of hours is refused.

    Two rejections, for two different mistakes: 18 is not a multiple of 4 at all,
    and 20 is a multiple of 4 but not a whole number of hours - in which case the
    hour-of-day feature would be a fiction.
    """
    with pytest.raises(ValueError, match="multiple of 4"):
        build_feature_matrix(
            synthetic_series.values,
            np.array([800]),
            lookback=96,
            feature_names=("hour_of_day",),
            series_index=np.zeros(1, dtype=np.int64),
            steps_per_day=18,
        )
    with pytest.raises(ValueError, match="whole number of hours"):
        build_feature_matrix(
            synthetic_series.values,
            np.array([800]),
            lookback=96,
            feature_names=("hour_of_day",),
            series_index=np.zeros(1, dtype=np.int64),
            steps_per_day=20,
        )


# ======================================================================
# The leakage guard
# ======================================================================


def test_the_guard_passes_for_the_real_feature_set(synthetic_series) -> None:
    origins = np.tile(np.arange(700, 2000, 7), synthetic_series.series_count)
    names = tuple(spec.name for spec in FEATURE_CATALOGUE if not spec.requires_weather)
    assert_no_future_access(synthetic_series.values, origins, names)


def test_the_guard_passes_with_weather(synthetic_series) -> None:
    weather = build_weather()
    origins = np.tile(np.arange(700, 2000, 7), synthetic_series.series_count)
    names = tuple(spec.name for spec in FEATURE_CATALOGUE)
    assert_no_future_access(synthetic_series.values, origins, names, weather=weather)


def test_the_guard_catches_a_deliberately_leaking_feature(synthetic_series) -> None:
    """A column that reads y[t+1] must be rejected.

    This is the test that gives the rest of the leakage machinery its meaning: if a
    future-reading feature ever slipped into the catalogue, this must fail. The
    builder is handed the matrix the guard poisoned, so reading one step ahead
    returns the poison and is detected.
    """
    origins = np.tile(np.arange(700, 2000, 7), synthetic_series.series_count)

    def leak_one_step(values, t, s):
        return values[s, t + 1]

    with pytest.raises(AssertionError, match="reads the future"):
        assert_no_future_access(
            synthetic_series.values,
            origins,
            ("lag_1", "deliberate_leak"),
            extra_builders={"deliberate_leak": leak_one_step},
        )


def test_the_guard_catches_a_leak_through_the_target_itself(synthetic_series) -> None:
    """Reading y[t+h] for h >= 1 is the classic leak; the guard must see it."""
    origins = np.tile(np.arange(700, 2000, 7), synthetic_series.series_count)

    def leak_target(values, t, s):
        return values[s, t + 96]

    with pytest.raises(AssertionError):
        assert_no_future_access(
            synthetic_series.values,
            origins,
            ("lag_1", "leaky_target"),
            extra_builders={"leaky_target": leak_target},
        )


def test_the_guard_allows_reading_the_origin_itself(synthetic_series) -> None:
    """y[t] is the current measurement and must be accepted."""

    def read_origin(values, t, s):
        return values[s, t]

    origins = np.tile(np.arange(700, 2000, 7), synthetic_series.series_count)
    assert_no_future_access(
        synthetic_series.values,
        origins,
        ("lag_1", "at_origin"),
        extra_builders={"at_origin": read_origin},
    )


def test_extra_builder_shape_is_validated(synthetic_series) -> None:
    """A builder that returns the wrong shape is refused, not broadcast silently."""
    with pytest.raises(ValueError, match="returned shape"):
        build_feature_matrix(
            synthetic_series.values,
            np.array([800, 900]),
            lookback=96,
            feature_names=("bad",),
            series_index=np.zeros(2, dtype=np.int64),
            extra_builders={"bad": lambda values, t, s: np.zeros(5)},
        )


def test_the_guard_checks_deterministically(synthetic_series) -> None:
    """Same seed, same probe positions, so a failure is reproducible."""
    origins = np.tile(np.arange(700, 3000, 3), synthetic_series.series_count)
    names = ("lag_1", "lag_96", "value_at_origin")
    assert_no_future_access(synthetic_series.values, origins, names, seed=5)
    assert_no_future_access(synthetic_series.values, origins, names, seed=5)


def test_guard_handles_an_empty_sample_list(synthetic_series) -> None:
    assert_no_future_access(synthetic_series.values, np.array([], dtype=np.int64), ("lag_1",))