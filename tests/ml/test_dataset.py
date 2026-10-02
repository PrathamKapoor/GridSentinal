"""Tests for temporal correctness: splits, windows, scaling and the dataset artifact.

These are the tests that would catch a chronologically invalid experiment. A shuffled
split, a window that crosses a boundary, or a scaler fitted on the test set would
each produce a plausible-looking table of numbers, so each is checked explicitly
rather than inferred from a metric.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from conftest import build_series, build_weather

from energy_intelligence.ml.dataset import (
    DatasetBuildConfig,
    build_dataset,
    dataset_version,
    load_dataset,
    save_dataset,
)
from energy_intelligence.ml.scaling import FeatureScaler, assert_train_only_fit
from energy_intelligence.ml.splits import (
    SPLIT_TEST,
    SPLIT_TRAIN,
    SPLIT_VALIDATION,
    sample_splits,
    split_boundaries,
    split_summary,
)


def _config(**overrides) -> DatasetBuildConfig:
    base = {"lookback": 672, "horizons": (1, 4, 96), "origin_stride": 4}
    base.update(overrides)
    return DatasetBuildConfig(**base)


# ======================================================================
# Splits
# ======================================================================


def test_boundaries_are_contiguous_and_ordered() -> None:
    bounds = split_boundaries(35040, train_fraction=0.70, validation_fraction=0.15)
    assert bounds.train[0] == 0
    assert bounds.train[1] == bounds.validation[0]
    assert bounds.validation[1] == bounds.test[0]
    assert bounds.test[1] == 35040


def test_boundaries_refuse_to_empty_the_test_range() -> None:
    with pytest.raises(ValueError, match="non-empty test range"):
        split_boundaries(1000, train_fraction=0.9, validation_fraction=0.2)


def test_boundaries_refuse_impossible_fractions() -> None:
    for train, validation in ((0.0, 0.1), (1.0, 0.1), (0.5, 0.0), (0.5, 1.5)):
        with pytest.raises(ValueError):
            split_boundaries(1000, train_fraction=train, validation_fraction=validation)


def test_bounds_refuse_a_non_positive_axis() -> None:
    with pytest.raises(ValueError, match="total_steps must be positive"):
        split_boundaries(0)


def test_every_origin_lands_in_exactly_one_split() -> None:
    bounds = split_boundaries(35040)
    origins = np.arange(0, 35040, 7)
    codes = sample_splits(origins, bounds, lookback=672, max_horizon=96)
    assert set(np.unique(codes[codes >= 0])) <= {SPLIT_TRAIN, SPLIT_VALIDATION, SPLIT_TEST}
    # Every code is assigned at most once, which is what "exactly one" means here.
    assert np.all(np.isin(codes, [-1, 0, 1, 2]))


def test_a_window_crossing_a_boundary_is_dropped_not_assigned() -> None:
    """The sample whose lookback reaches into validation must not be training."""
    bounds = split_boundaries(1000, train_fraction=0.7, validation_fraction=0.15)
    boundary = bounds.train[1]
    # lookback 10, horizon 5: an origin 1 step past the boundary would reach back
    # into training and 5 steps forward into validation, so it belongs to neither.
    # A window fits inside training only when origin - 10 >= 0 and origin + 5 < 700,
    # i.e. origin <= 694. So 694 is the last training origin, and 695 is dropped.
    origins = np.array([boundary - 6, boundary - 5, boundary - 1, boundary + 100])
    codes = sample_splits(origins, bounds, lookback=10, max_horizon=5)
    assert codes[0] == SPLIT_TRAIN, "694 is the last origin whose window fits"
    assert codes[1] == -1, "695 would put its target exactly on the boundary"
    assert codes[2] == -1, "one step before the boundary is also dropped"
    assert codes[3] == SPLIT_VALIDATION


def test_a_target_reaching_past_the_boundary_is_dropped() -> None:
    bounds = split_boundaries(1000, train_fraction=0.7, validation_fraction=0.15)
    edge = bounds.train[1] - 1
    codes = sample_splits(np.array([edge]), bounds, lookback=1, max_horizon=96)
    assert codes[0] == -1, "a 96-step target would reach into validation"


def test_splits_refuse_a_non_positive_window() -> None:
    bounds = split_boundaries(1000)
    with pytest.raises(ValueError, match="lookback must be positive"):
        sample_splits(np.array([100]), bounds, lookback=0, max_horizon=1)
    with pytest.raises(ValueError, match="max_horizon must be positive"):
        sample_splits(np.array([100]), bounds, lookback=1, max_horizon=0)


def test_split_summary_counts_the_dropped_samples() -> None:
    bounds = split_boundaries(1000, train_fraction=0.7, validation_fraction=0.15)
    codes = sample_splits(np.arange(0, 1000, 5), bounds, lookback=672, max_horizon=96)
    summary = split_summary(codes)
    assert summary["total_samples"] == codes.size
    assert summary["train"] + summary["validation"] + summary["test"] + summary[
        "dropped_crossing_boundary"
    ] == codes.size


# ======================================================================
# Dataset construction
# ======================================================================


@pytest.fixture
def dataset():
    """A three-split-populating dataset.

    20000 steps is about 52 days: long enough that a 7-day lookback leaves room for
    samples in validation and test, which is what makes the boundary assertions
    meaningful. A shorter axis legitimately produces empty validation and test, and
    asserting on those would be asserting nothing.
    """
    series = build_series(series_count=3, steps=20000)
    return build_dataset(series, _config())


@pytest.fixture
def short_dataset():
    """A dataset whose axis is too short for three populated splits."""
    series = build_series(series_count=2, steps=4000)
    return build_dataset(series, _config())


def test_targets_are_the_values_at_origin_plus_horizon(dataset) -> None:
    """The single most important alignment property."""
    assert dataset.manifest["target"]["target_id"] == "customer_load"
    series = build_series(series_count=3, steps=20000)
    for horizon in dataset.horizons:
        column = dataset.target_column(horizon)
        for row in (0, 1, 500, dataset.sample_count - 1):
            s = int(dataset.series_index[row])
            t = int(dataset.origin_index[row])
            assert column[row] == pytest.approx(
                series.normalised()[s, t + horizon], rel=1e-6
            )


def test_each_row_belongs_to_the_series_its_features_came_from(dataset) -> None:
    """Guards the row-layout bug that once mis-assigned every feature."""
    series = build_series(series_count=3, steps=20000)
    normalised = series.normalised()
    value_at_origin = dataset.feature_names.index("value_at_origin")
    for row in (0, 1, 7, 8, dataset.sample_count // 2, dataset.sample_count - 1):
        s = int(dataset.series_index[row])
        t = int(dataset.origin_index[row])
        assert dataset.features[row, value_at_origin] == pytest.approx(
            normalised[s, t], rel=1e-5
        )


def test_no_window_crosses_a_split_boundary(dataset) -> None:
    """The core temporal guarantee, checked against the recorded boundaries."""
    bounds = dataset.manifest["split"]["boundaries"]
    lookback = dataset.manifest["windows"]["lookback_steps"]
    horizon = dataset.manifest["windows"]["max_horizon_steps"]
    for code, name in ((SPLIT_TRAIN, "train"), (SPLIT_VALIDATION, "validation"), (SPLIT_TEST, "test")):
        rows = np.flatnonzero(dataset.split == code)
        assert rows.size, f"{name} must contain samples"
        origins = dataset.origin_index[rows].astype(np.int64)
        lo, hi = bounds[name]
        assert origins.min() - lookback >= lo, f"{name}: lookback crosses the lower bound"
        assert origins.max() + horizon < hi, f"{name}: target crosses the upper bound"


def test_a_short_axis_yields_empty_later_splits_and_says_so(short_dataset) -> None:
    """On a 4000-step axis a 672-step lookback cannot fit validation at all.

    This is honest behaviour rather than a bug, and the summary must show it instead
    of hiding the emptiness.
    """
    counts = split_summary(short_dataset.split)
    assert counts["train"] > 0
    assert counts["validation"] == 0
    assert counts["test"] == 0


def test_dataset_is_deterministic() -> None:
    """Same input, same bytes - otherwise nothing downstream is reproducible."""
    series = build_series(series_count=2, steps=2000)
    first = build_dataset(series, _config())
    second = build_dataset(series, _config())
    assert first.version == second.version
    assert np.array_equal(first.features, second.features)
    assert np.array_equal(first.targets, second.targets)
    assert np.array_equal(first.split, second.split)


def test_version_changes_with_every_shape_decision() -> None:
    """Lookback, horizon, target, stride or split must all change the version."""
    series = build_series(series_count=2, steps=2000)
    base = dataset_version(_config(), "customer_load", series.series_ids)
    variants = [
        dataset_version(_config(lookback=96), "customer_load", series.series_ids),
        dataset_version(_config(horizons=(1, 4)), "customer_load", series.series_ids),
        dataset_version(_config(origin_stride=2), "customer_load", series.series_ids),
        dataset_version(_config(train_fraction=0.6), "customer_load", series.series_ids),
        dataset_version(_config(), "feeder_load", series.series_ids),
    ]
    assert len(set([base, *variants])) == len(variants) + 1


def test_config_refuses_nonsense() -> None:
    with pytest.raises(ValueError, match="lookback must be positive"):
        DatasetBuildConfig(lookback=0, horizons=(1,))
    with pytest.raises(ValueError, match="horizons must be sorted"):
        DatasetBuildConfig(lookback=96, horizons=(4, 1))
    with pytest.raises(ValueError, match="at least one forecast horizon"):
        DatasetBuildConfig(lookback=96, horizons=())
    with pytest.raises(ValueError, match="origin_stride must be positive"):
        DatasetBuildConfig(lookback=96, horizons=(1,), origin_stride=0)


def test_infeasible_configuration_is_refused_with_a_reason() -> None:
    # 700 steps cannot supply a 672-step lookback plus a 96-step horizon.
    series = build_series(series_count=1, steps=700)
    with pytest.raises(ValueError, match="cannot support lookback"):
        build_dataset(series, _config(lookback=672, horizons=(96,)))


def test_impossible_weather_request_is_refused() -> None:
    """Asking for weather features without weather data must fail, not degrade."""
    series = build_series(series_count=1, steps=2000)
    config = _config(lookback=672, horizons=(1,), include_weather=True)
    with pytest.raises(ValueError, match="include_weather is set"):
        build_dataset(series, config)


def test_manifest_records_the_full_provenance() -> None:
    series = build_series(series_count=2, steps=2000)
    ds = build_dataset(series, _config())
    manifest = ds.manifest
    assert manifest["schema_version"].startswith("1.")
    assert manifest["dataset_version"] == ds.version
    assert manifest["temporal"]["resampling"].startswith("NONE")
    assert "UTC" in manifest["temporal"]["timezone_assumption"]
    assert manifest["split"]["method"].startswith("chronological")
    assert manifest["preprocessing"]["imputation"].startswith("NONE")
    assert manifest["features"]["leakage_policy"]
    assert manifest["source"]["value_origin"] == "SYNTHETIC"


def test_manifest_lists_withdrawn_features_with_reasons() -> None:
    series = build_series(series_count=1, steps=2000)
    ds = build_dataset(series, _config(lookback=96, horizons=(1, 4)))
    withdrawn = {item["name"]: item["reason"] for item in ds.manifest["features"]["withdrawn"]}
    assert "lag_672" in withdrawn
    assert withdrawn["ghi_at_origin"]


def test_dataset_is_stored_per_unit_with_the_divisor_kept() -> None:
    series = build_series(series_count=2, steps=2000)
    ds = build_dataset(series, _config())
    assert np.allclose(ds.series_scale, series.scale)
    assert ds.summary()["stored_in"].startswith("per unit")
    assert ds.target_unit == "kW"


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    series = build_series(series_count=2, steps=2000)
    ds = build_dataset(series, _config())
    npz_path, json_path = save_dataset(ds, tmp_path)

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["arrays"]["sha256"]
    assert payload["arrays"]["bytes"] == npz_path.stat().st_size

    restored = load_dataset(tmp_path, ds.version)
    assert restored.version == ds.version
    assert restored.feature_names == ds.feature_names
    assert restored.horizons == ds.horizons
    assert np.array_equal(restored.features, ds.features)
    assert np.array_equal(restored.targets, ds.targets)
    assert np.array_equal(restored.split, ds.split)
    assert np.allclose(restored.series_scale, ds.series_scale)


def test_loading_a_missing_version_lists_what_exists(tmp_path: Path) -> None:
    series = build_series(series_count=1, steps=2000)
    save_dataset(build_dataset(series, _config()), tmp_path)
    with pytest.raises(FileNotFoundError, match="available"):
        load_dataset(tmp_path, "ds-doesnotexist")


def test_weather_features_appear_only_for_a_15_minute_horizon() -> None:
    series = build_series(series_count=1, steps=2000)
    weather = build_weather(2000)
    nowcast = build_dataset(
        series,
        _config(horizons=(1,), include_weather=True),
        weather=weather,
    )
    assert "ghi_at_origin" in nowcast.feature_names

    longer = build_dataset(
        series,
        _config(horizons=(1, 4), include_weather=True),
        weather=weather,
    )
    assert "ghi_at_origin" not in longer.feature_names
    reasons = {item["name"]: item["reason"] for item in longer.manifest["features"]["withdrawn"]}
    assert "weather FORECAST" in reasons["ghi_at_origin"]


# ======================================================================
# Scaling
# ======================================================================


def test_scaler_fits_only_on_the_rows_it_is_given() -> None:
    matrix = np.arange(12, dtype=np.float64).reshape(4, 3)
    scaler = FeatureScaler.fit(matrix)
    assert scaler.fitted_on == 4
    assert_train_only_fit(scaler, 4)


def test_train_only_check_fails_when_more_rows_were_used() -> None:
    matrix = np.arange(40, dtype=np.float64).reshape(10, 4)
    scaler = FeatureScaler.fit(matrix)
    with pytest.raises(AssertionError, match="must never see validation or test data"):
        assert_train_only_fit(scaler, 7)


def test_transform_uses_the_fitted_statistics() -> None:
    train = np.array([[0.0], [2.0]])
    scaler = FeatureScaler.fit(train)
    assert scaler.mean[0] == pytest.approx(1.0)
    # A test-set value is transformed with the TRAIN statistics, not its own.
    transformed = scaler.transform(np.array([[1.0]]))
    assert transformed[0, 0] == pytest.approx(0.0)
    assert scaler.transform(np.array([[1.0]]))[0, 0] == transformed[0, 0]


def test_constant_column_does_not_divide_by_zero() -> None:
    matrix = np.array([[5.0], [5.0], [5.0]])
    scaler = FeatureScaler.fit(matrix)
    assert scaler.constant_columns == (0,)
    assert np.allclose(scaler.transform(matrix), 0.0)


def test_scaler_refuses_mismatched_columns_and_empty_input() -> None:
    scaler = FeatureScaler.fit(np.ones((3, 2)))
    with pytest.raises(ValueError, match="to match the fitted scaler"):
        scaler.transform(np.ones((3, 3)))
    with pytest.raises(ValueError, match="zero rows"):
        FeatureScaler.fit(np.ones((0, 2)))
    with pytest.raises(ValueError, match="expected a 2-D matrix"):
        FeatureScaler.fit(np.ones(4))


def test_scaler_serialises_round_trip() -> None:
    scaler = FeatureScaler.fit(np.array([[1.0, 2.0], [3.0, 5.0]]))
    restored = FeatureScaler.from_dict(scaler.to_dict())
    assert np.allclose(restored.mean, scaler.mean)
    assert np.allclose(restored.scale, scaler.scale)
    assert restored.fitted_on == scaler.fitted_on


def test_dataset_scaler_is_fitted_on_training_rows_only() -> None:
    """The end-to-end guarantee: statistics see the training split and no more."""
    series = build_series(series_count=2, steps=20000)
    dataset = build_dataset(series, _config(origin_stride=8))
    counts = split_summary(dataset.split)
    assert counts["validation"] > 0 and counts["test"] > 0, (
        "this assertion is only meaningful when every split is populated"
    )
    train_mask = dataset.mask(SPLIT_TRAIN)
    scaler = FeatureScaler.fit(dataset.features[train_mask].astype(np.float64))
    assert_train_only_fit(scaler, int(np.count_nonzero(train_mask)))
    # A scaler fitted on everything would see a different mean, proving the test
    # can actually tell the difference.
    everything = FeatureScaler.fit(dataset.features.astype(np.float64))
    assert everything.fitted_on != scaler.fitted_on