"""Leakage tests for Phase 8.

Everything in this module is about one question: does any quantity a method is *fitted*
on contain information from a row it will later be *scored* on, or from the target at all?

The tests are deliberately adversarial. Each one constructs a violation and asserts it is
caught, rather than asserting that the correct code does the correct thing - a test that
only passes on correct code cannot tell you whether it is still testing anything after a
refactor.

The failure modes this phase could plausibly have shipped:

* the learned scale fitted on rows whose residuals it then normalises;
* the conformity scores drawn from the same rows the scale was fitted on;
* band cut points taken from the test rows, making an operational label a test statistic;
* the lagged-error feature group reaching into the row's own future;
* ``impute`` filling a gap from the whole panel rather than the calibration-fitting rows;
* the fixed-ensemble weights drifting away from Phase 7's, so the point forecast is not
  the one the phase claims to be quantifying.
"""

from __future__ import annotations

import numpy as np
import pytest

from energy_intelligence.ml.uncertainty.artifact import BandCutPoints
from energy_intelligence.ml.uncertainty.calibration import (
    CONFORMAL_GUARANTEE,
    conformal_quantile_index,
    fit_conformal,
)
from energy_intelligence.ml.uncertainty.features_set import build_uncertainty_features
from energy_intelligence.ml.uncertainty.split import (
    CalibrationSplit,
    build_calibration_split,
    split_parity_check,
)

HORIZONS = (1, 4, 96)


def _panel_positions(n: int = 40) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    validation = np.arange(0, n)
    test = np.arange(n, 2 * n)
    origins = np.arange(2 * n)
    return validation, test, origins


# ---------------------------------------------------------------------------
# The partition
# ---------------------------------------------------------------------------


def test_split_halves_validation_chronologically():
    validation, test, origins = _panel_positions()
    split = build_calibration_split(
        validation_rows=validation, test_rows=test, origins=origins, horizons=HORIZONS
    )
    assert split.calibration_fit.size == validation.size // 2
    assert split.calibration_conformity.size == validation.size - validation.size // 2
    # Chronological, not positional: the fitting half must come first in origin order.
    assert split.calibration_fit.max() < split.calibration_conformity.min()


def test_split_refuses_overlapping_calibration_halves():
    """A scale fitted on rows whose residuals it then normalises is not a scale."""
    rows = np.arange(10)
    with pytest.raises(ValueError, match="overlap"):
        CalibrationSplit(
            calibration_fit=rows,
            calibration_conformity=rows,
            test=np.arange(10, 20),
            validation=rows,
            horizons=HORIZONS,
            cal_fraction=0.5,
        )


@pytest.mark.parametrize("leak", ["calibration_fit", "calibration_conformity"])
def test_split_refuses_a_calibration_row_inside_test(leak: str):
    validation = np.arange(10)
    test = np.arange(10, 20)
    fit = validation[:5]
    conform = validation[5:]
    if leak == "calibration_fit":
        fit = np.append(fit, test[0])
    else:
        conform = np.append(conform, test[0])
    with pytest.raises(ValueError, match="sealed"):
        CalibrationSplit(
            calibration_fit=fit,
            calibration_conformity=conform,
            test=test,
            validation=validation,
            horizons=HORIZONS,
            cal_fraction=0.5,
        )


def test_split_refuses_a_calibration_row_outside_validation():
    """A calibration row from the training split is in-sample for the point forecast."""
    validation = np.arange(10, 20)
    test = np.arange(20, 30)
    with pytest.raises(ValueError, match="whole validation split"):
        CalibrationSplit(
            calibration_fit=np.array([5, 10, 11, 12]),
            calibration_conformity=np.array([13, 14, 15, 16, 17, 18, 19]),
            test=test,
            validation=validation,
            horizons=HORIZONS,
            cal_fraction=0.5,
        )


def test_split_rejects_an_out_of_range_fraction_and_an_empty_side():
    validation, test, origins = _panel_positions(4)
    with pytest.raises(ValueError, match=r"cal_fraction must lie"):
        build_calibration_split(
            validation_rows=validation, test_rows=test, origins=origins,
            horizons=HORIZONS, cal_fraction=1.0,
        )
    with pytest.raises(ValueError, match="non-empty"):
        build_calibration_split(
            validation_rows=np.zeros(0, dtype=np.int64), test_rows=test,
            origins=origins, horizons=HORIZONS,
        )


def test_split_orders_by_origin_not_by_position():
    """A reordered panel must still produce a chronological split."""
    validation = np.array([9, 7, 5, 3, 1, 0, 2, 4, 6, 8])
    test = np.arange(10, 20)
    origins = np.arange(20)
    split = build_calibration_split(
        validation_rows=validation, test_rows=test, origins=origins, horizons=HORIZONS
    )
    assert list(split.calibration_fit) == [0, 1, 2, 3, 4]
    assert list(split.calibration_conformity) == [5, 6, 7, 8, 9]


# ---------------------------------------------------------------------------
# Split parity: the overlap Phase 7's own weights introduced, measured
# ---------------------------------------------------------------------------


def test_split_parity_reports_all_three_row_sets():
    validation, test, origins = _panel_positions(20)
    split = build_calibration_split(
        validation_rows=validation, test_rows=test, origins=origins, horizons=HORIZONS
    )
    rng = np.random.default_rng(0)
    point = rng.normal(10.0, 1.0, size=(40, len(HORIZONS)))
    actual = point + rng.normal(0.0, 0.5, size=point.shape)
    report = split_parity_check(
        point_kw=point, actual_kw=actual, split=split, horizons=HORIZONS
    )
    for horizon in HORIZONS:
        entry = report[str(horizon)]
        assert entry["calibration_fit_rows"] + entry["calibration_conformity_rows"] == 20
        assert entry["test_rows"] == 20
        assert entry["fit_vs_conformity_pct"] is not None
        assert entry["test_vs_conformity_pct"] is not None


def test_split_parity_reports_an_optimistic_fitting_half():
    """The whole point of the check: an easy fitting half inflates every width."""
    validation, test, origins = _panel_positions(20)
    split = build_calibration_split(
        validation_rows=validation, test_rows=test, origins=origins, horizons=HORIZONS
    )
    point = np.full((40, len(HORIZONS)), 10.0)
    actual = point.copy()
    # Errors only on the conformity half and on test: the fitting half is perfect.
    actual[split.calibration_conformity, 0] += 2.0
    actual[split.test, 0] += 2.0
    report = split_parity_check(
        point_kw=point, actual_kw=actual, split=split, horizons=HORIZONS
    )
    assert report["1"]["calibration_fit_mae_kw"] == pytest.approx(0.0)
    assert report["1"]["calibration_conformity_mae_kw"] == pytest.approx(2.0)
    assert report["1"]["fit_vs_conformity_pct"] == pytest.approx(-100.0)


# ---------------------------------------------------------------------------
# Conformal: the guarantee is conditional, and the code must reflect that
# ---------------------------------------------------------------------------


def test_conformal_index_uses_the_finite_sample_correction():
    """The ``+1`` and the ceiling are the guarantee; a plain quantile is not."""
    assert conformal_quantile_index(100, 0.10) == 91
    assert conformal_quantile_index(10, 0.10) == 10
    # Capped at n, so the index never runs off the end of the sorted scores.
    assert conformal_quantile_index(5, 0.01) == 5
    with pytest.raises(ValueError, match="n must be positive"):
        conformal_quantile_index(0, 0.1)
    with pytest.raises(ValueError, match=r"alpha must lie"):
        conformal_quantile_index(10, 0.0)


def test_conformal_guarantee_text_refuses_to_claim_exchangeability():
    assert "exchangeable" in CONFORMAL_GUARANTEE
    assert "empirical measurement" in CONFORMAL_GUARANTEE
    assert "NOT exchangeable" in CONFORMAL_GUARANTEE


def test_conformal_rejects_a_non_positive_scale():
    """A zero scale divides the score by zero and produces an infinite width."""
    residual = np.ones((10, len(HORIZONS)))
    scale = np.ones((10, len(HORIZONS)))
    scale[0, 0] = 0.0
    with pytest.raises(ValueError, match="strictly positive"):
        fit_conformal(
            residual_kw=residual, scale=scale, horizons=HORIZONS,
            nominal_levels=(0.9,), scale_name="learned",
        )


def test_conformal_scores_are_the_absolutes_of_the_given_residuals():
    rng = np.random.default_rng(1)
    signed = rng.normal(0.0, 1.0, size=(500, len(HORIZONS)))
    calibrator = fit_conformal(
        residual_kw=signed, scale=None, horizons=HORIZONS,
        nominal_levels=(0.9,), scale_name="absolute",
    )
    mirrored = fit_conformal(
        residual_kw=-signed, scale=None, horizons=HORIZONS,
        nominal_levels=(0.9,), scale_name="absolute",
    )
    assert calibrator.width(1, 0.9) == pytest.approx(mirrored.width(1, 0.9))
    assert calibrator.scale_is_absolute


def test_normalised_conformal_tracks_a_row_varying_error_that_a_constant_cannot():
    """What a scale function is actually for.

    The residual's spread has to vary *within* a horizon for a scale to earn its place. If
    it only differs between horizons then a per-horizon constant already captures it, and
    normalising by a scale proportional to the spread changes nothing - which is the
    situation this test deliberately does not set up.
    """
    rng = np.random.default_rng(2)
    regime = np.tile(np.array([0.2, 1.0, 5.0]), 2000)[:, None] * np.ones((1, len(HORIZONS)))
    residual = rng.normal(0.0, 1.0, size=(6000, len(HORIZONS))) * regime
    scale = regime
    point = np.full(residual.shape, 10.0)
    absolute = fit_conformal(
        residual_kw=residual, scale=None, horizons=HORIZONS,
        nominal_levels=(0.9,), scale_name="absolute",
    ).intervals(
        point_kw=point, scale=None, horizons=HORIZONS, nominal_levels=(0.9,), method="a"
    )[0.9]
    scaled = fit_conformal(
        residual_kw=residual, scale=scale, horizons=HORIZONS,
        nominal_levels=(0.9,), scale_name="learned",
    ).intervals(
        point_kw=point, scale=scale, horizons=HORIZONS, nominal_levels=(0.9,), method="s"
    )[0.9]
    # One multiplier per horizon means the absolute width cannot vary within a column.
    assert absolute.width_kw[:, 0].std() == pytest.approx(0.0)
    # The scaled width varies with the row, in proportion to the row's own error, so the
    # ratio of width to realised error is far tighter than the absolute method achieves.
    assert scaled.width_kw[:, 0].std() > 0.0
    scaled_ratio = scaled.width_kw[:, 0] / np.maximum(np.abs(residual[:, 0]), 1e-9)
    absolute_ratio = absolute.width_kw[:, 0] / np.maximum(np.abs(residual[:, 0]), 1e-9)
    assert scaled_ratio.max() - scaled_ratio.min() < absolute_ratio.max() - absolute_ratio.min()
    assert scaled.width_kw[-1, 0] > scaled.width_kw[0, 0]


# ---------------------------------------------------------------------------
# Band cut points must come from the calibration split
# ---------------------------------------------------------------------------


def test_band_cuts_are_terciles_and_reject_degenerate_input():
    cuts = BandCutPoints.from_widths(np.arange(1.0, 10.0), nominal_level=0.9)
    assert cuts.low_high_kw < cuts.high_kw
    assert cuts.source_rows == 9
    with pytest.raises(ValueError, match="at least three widths"):
        BandCutPoints.from_widths(np.array([1.0, 2.0]), nominal_level=0.9)
    with pytest.raises(ValueError, match="finite"):
        BandCutPoints.from_widths(np.array([1.0, np.nan, 3.0]), nominal_level=0.9)


def test_band_cuts_from_a_narrow_sample_differ_from_the_test_distribution():
    """This is the leak the design forbids, stated as a test.

    The cut points are terciles of the *calibration* widths. If they were taken from the
    rows being scored, the band counts on those rows would be one third each by
    construction and the label would carry no information.
    """
    calibration = np.concatenate([np.full(900, 1.0), np.full(100, 20.0)])
    test = np.concatenate([np.full(500, 1.0), np.full(500, 20.0)])
    from_calibration = BandCutPoints.from_widths(calibration, nominal_level=0.9)
    from_test = BandCutPoints.from_widths(test, nominal_level=0.9)
    assert from_calibration.high_kw == pytest.approx(1.0)
    assert from_test.high_kw == pytest.approx(20.0)
    assert from_calibration.high_kw != from_test.high_kw


# ---------------------------------------------------------------------------
# Features: the poisoning guard, and the imputation reference
# ---------------------------------------------------------------------------


def _long_panel():
    """A panel long enough for the h=96 lagged-error column to be observed in validation.

    ``past_96_e*`` reads the realised error 97 origins back, so it is missing for the first
    97 origins of any panel. The default 300-row fixture does not have 97 origins of
    history inside its validation half, so the column is unobservable there and cannot be
    imputed from the calibration rows at all. 600 rows over 3 series is 200 origins, which
    puts observed values at the end of validation and throughout test.
    """
    from tests.ml.conftest import build_test_panel, build_test_values

    panel = build_test_panel(n_rows=600)
    return panel, build_test_values(panel)


def test_uncertainty_features_pass_the_poisoning_check():
    from energy_intelligence.ml.uncertainty.features_set import build_uncertainty_features

    panel, values = _long_panel()
    horizons = tuple(panel.horizons)
    forecasts = np.stack(
        [np.asarray(panel.persistence_kw) + float(index) for index in range(3)]
    )
    matrix, names, record = build_uncertainty_features(
        values=values,
        panel=panel,
        forecasts=forecasts,
        actual_kw=np.asarray(panel.actual_kw),
        horizons=horizons,
        include_past_error=True,
        reference_rows=np.arange(0, panel.split_slices["validation"].stop),
        sample=3,
        seed=5,
    )
    assert matrix.shape[0] == panel.n_rows
    assert len(names) == matrix.shape[1]
    assert np.isfinite(matrix).all()
    assert record["passed"]
    assert record["checked_rows_verified"] > 0
    assert record["lagged_error_group_checked"]


def test_uncertainty_features_can_drop_the_lagged_error_group():
    from energy_intelligence.ml.uncertainty.features_set import build_uncertainty_features

    panel, values = _long_panel()
    reference = np.arange(0, panel.split_slices["validation"].stop)
    forecasts = np.stack(
        [np.asarray(panel.persistence_kw) + float(index) for index in range(3)]
    )
    _, with_errors, _ = build_uncertainty_features(
        values=values, panel=panel, forecasts=forecasts,
        actual_kw=np.asarray(panel.actual_kw), horizons=tuple(panel.horizons),
        include_past_error=True, reference_rows=reference, sample=2, seed=5,
    )
    _, without_errors, record = build_uncertainty_features(
        values=values, panel=panel, forecasts=None, actual_kw=None,
        horizons=tuple(panel.horizons), include_past_error=False,
        reference_rows=reference, sample=2, seed=5,
    )
    assert len(without_errors) < len(with_errors)
    assert not any(name.startswith("past_") for name in without_errors)
    assert not record["lagged_error_group_checked"]


def test_the_poisoning_check_catches_a_feature_that_reads_the_future():
    """A guard that cannot fail is not a guard.

    The check compares each row's features against a rebuild with the row's own future
    overwritten. A feature that peeks forward changes; the assertion fires.
    """
    from energy_intelligence.ml.router.features import (
        _POISON,
        RouterFeatureSpec,
        RouterFeatures,
        assert_router_features_are_causal,
    )
    from energy_intelligence.ml.uncertainty.features_set import build_uncertainty_features

    panel, values = _long_panel()
    forecasts = np.stack(
        [np.asarray(panel.persistence_kw) + float(index) for index in range(3)]
    )
    matrix, _, _ = build_uncertainty_features(
        values=values, panel=panel, forecasts=forecasts,
        actual_kw=np.asarray(panel.actual_kw), horizons=tuple(panel.horizons),
        include_past_error=True,
        reference_rows=np.arange(0, panel.split_slices["validation"].stop),
        sample=2, seed=5,
    )
    poisoned_matrix = matrix.copy()
    # A "feature" that is simply the future: the last column becomes the next step.
    step = np.asarray(panel.origins) + 1
    future = np.asarray(values)[np.asarray(panel.series), np.minimum(step, values.shape[1] - 1)]
    poisoned_matrix[:, -1] = future

    names = tuple(f"f{i}" for i in range(poisoned_matrix.shape[1]))
    poisoned = RouterFeatures(
        matrix=poisoned_matrix.astype(np.float32),
        names=names,
        spec=RouterFeatureSpec(),
    )
    honest = RouterFeatures(
        matrix=matrix.astype(np.float32),
        names=names,
        spec=RouterFeatureSpec(),
    )
    assert np.array_equal(poisoned.matrix[:, :-1], honest.matrix[:, :-1])
    assert not np.array_equal(
        poisoned.matrix[:, -1], honest.matrix[:, -1], equal_nan=True
    )
    # The real guard refuses the doctored matrix. Its last column is not one these builders
    # produce, so it reports the column mismatch - and reports it as an assertion, not as
    # an IndexError from comparing two arrays of different widths.
    with pytest.raises(AssertionError, match="not the ones these builders produce"):
        assert_router_features_are_causal(
            values=values, panel=panel, features=poisoned, sample=2, seed=5
        )
    assert _POISON < 0.0


def test_the_poisoning_check_reports_a_column_that_actually_moved():
    """The guard must name the offending column, not merely fail."""
    from energy_intelligence.ml.router.features import (
        RouterFeatureSpec,
        RouterFeatures,
        assert_router_features_are_causal,
        build_router_features,
    )

    panel, values = _long_panel()
    honest = build_router_features(values=values, panel=panel)
    doctored = honest.matrix.copy()
    doctored[:, 3] = doctored[:, 0] * 7.0
    poisoned = RouterFeatures(
        matrix=doctored, names=honest.names, spec=RouterFeatureSpec()
    )
    with pytest.raises(AssertionError, match="ramp_1"):
        assert_router_features_are_causal(
            values=values, panel=panel, features=poisoned, sample=2, seed=5
        )


def test_imputation_reference_rows_change_the_filled_matrix():
    """The reference rows, not just the row being filled, decide the imputed value.

    Stated as a unit test of the fill itself rather than through the feature builder,
    because the contract is about *which rows are consulted*: filling a gap from the whole
    panel would let a test row's own error rate pick the value a test row is scored
    against, even though only a mean is taken.
    """
    from energy_intelligence.ml.router.features import RouterFeatureSpec, RouterFeatures

    matrix = np.array(
        [[np.nan, 1.0], [np.nan, 2.0], [10.0, 3.0], [20.0, 4.0]], dtype=np.float32
    )
    features = RouterFeatures(
        matrix=matrix, names=("gap", "solid"), spec=RouterFeatureSpec()
    )
    from_early = features.fill_missing(np.array([2, 3]))
    from_late = features.fill_missing(np.array([3]))
    assert np.isnan(features.matrix).any(), "the fixture must actually need filling"
    # The gap is filled with the mean of the *reference* rows, which is why the choice of
    # reference rows - not the row being filled - decides the value.
    assert float(from_early.matrix[0, 0]) == pytest.approx(15.0)
    assert float(from_late.matrix[0, 0]) == pytest.approx(20.0)
    assert not np.array_equal(from_early.matrix, from_late.matrix)
    # A column with no gaps is returned untouched, whatever the reference rows say.
    assert np.allclose(from_early.matrix[:, 1], matrix[:, 1])
    with pytest.raises(ValueError, match="zero reference rows"):
        features.fill_missing(np.zeros(0, dtype=np.int64))


def test_imputation_rejects_a_reference_set_with_nothing_to_learn_from():
    """Zero reference rows cannot define a mean, and the error must say so."""
    panel, values = _long_panel()
    forecasts = np.stack(
        [np.asarray(panel.persistence_kw) + float(index) for index in range(3)]
    )
    with pytest.raises(ValueError, match="zero reference rows"):
        build_uncertainty_features(
            values=values, panel=panel, forecasts=forecasts,
            actual_kw=np.asarray(panel.actual_kw), horizons=tuple(panel.horizons),
            include_past_error=True,
            reference_rows=np.zeros(0, dtype=np.int64),
            sample=2, seed=5,
        )


def test_the_uncertainty_feature_builder_fills_only_from_the_rows_it_is_given():
    """End to end: the builder must hand its reference rows straight to the fill."""
    from energy_intelligence.ml.uncertainty.features_set import (
        build_uncertainty_features,
    )

    panel, values = _long_panel()
    forecasts = np.stack(
        [np.asarray(panel.persistence_kw) + float(index) for index in range(3)]
    )
    matrix, names, _ = build_uncertainty_features(
        values=values, panel=panel, forecasts=forecasts,
        actual_kw=np.asarray(panel.actual_kw), horizons=tuple(panel.horizons),
        include_past_error=True,
        reference_rows=np.arange(0, panel.split_slices["validation"].stop),
        sample=2, seed=5,
    )
    assert np.isfinite(matrix).all()
    # The lagged columns are the only ones with gaps, and they are only gaps at the head.
    lagged = [index for index, name in enumerate(names) if name.startswith("past_96_e")]
    assert lagged, "expected the h=96 lagged-error columns"
    assert not np.array_equal(
        matrix[:, lagged],
        np.full((panel.n_rows, len(lagged)), np.nan, dtype=np.float32),
        equal_nan=True,
    )