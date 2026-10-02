"""Tests for metrics, baselines, the experiment runner and the analysis modules.

Metrics are checked against hand-computed values rather than against themselves: a
metric library that agrees with its own reference implementation but not with
arithmetic is worse than no metric library. The zero-generation cases matter most,
because half of a PV year is exactly zero and that is where percentage metrics
usually break.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from conftest import build_series, build_weather

from energy_intelligence.ml.analysis import assign_regimes, error_analysis, regime_report
from energy_intelligence.ml.baselines.classical import CLASSICAL_MODELS, fit_classical, model_defaults
from energy_intelligence.ml.baselines.naive import NAIVE_MODELS, fit_naive, naive_forecast
from energy_intelligence.ml.baselines.neural import NEURAL_MODELS, fit_neural
from energy_intelligence.ml.dataset import DatasetBuildConfig, build_dataset
from energy_intelligence.ml.experiment import (
    ALL_MODELS,
    compare,
    run_experiment,
    verify_reproducible,
)
from energy_intelligence.ml.metrics import (
    daytime_error_mae,
    evaluate,
    mae,
    peak_error_mae,
    r2_score,
    ramp_error_mae,
    rmse,
    smape,
    zero_period_error_mae,
)


# ======================================================================
# Metrics
# ======================================================================


def test_mae_and_rmse_match_hand_arithmetic() -> None:
    actual = np.array([1.0, 2.0, 3.0])
    predicted = np.array([2.0, 2.0, 5.0])
    # |1-2| + |2-2| + |3-5| = 1 + 0 + 2 = 3, over 3 samples
    assert mae(actual, predicted) == pytest.approx(1.0)
    assert rmse(actual, predicted) == pytest.approx(np.sqrt((1 + 0 + 4) / 3))


def test_smape_is_zero_for_a_perfect_forecast() -> None:
    values = np.array([0.0, 5.0, 100.0])
    assert smape(values, values) == pytest.approx(0.0)


def test_smape_treats_zero_over_zero_as_zero() -> None:
    """The rule that makes sMAPE usable on a solar night."""
    assert smape(np.array([0.0]), np.array([0.0])) == pytest.approx(0.0)
    # A zero actual against a non-zero forecast must still be penalised fully.
    assert smape(np.array([0.0]), np.array([4.0])) == pytest.approx(200.0)


def test_smape_is_bounded_by_200_percent() -> None:
    value = smape(np.array([0.0, 10.0]), np.array([100.0, 0.0]))
    assert 0.0 <= value <= 200.0


def test_r2_is_one_for_a_perfect_forecast_and_nan_for_a_constant_target() -> None:
    values = np.array([1.0, 2.0, 3.0, 4.0])
    assert r2_score(values, values) == pytest.approx(1.0)
    assert np.isnan(r2_score(np.array([2.0, 2.0]), np.array([1.0, 3.0])))


def test_metrics_refuse_mismatched_shapes() -> None:
    for function in (mae, rmse, smape, r2_score):
        with pytest.raises(ValueError, match="shape mismatch"):
            function(np.zeros(3), np.zeros(4))


def test_metrics_on_an_empty_sample_are_nan_not_zero() -> None:
    empty = np.array([])
    assert np.isnan(mae(empty, empty))
    assert np.isnan(rmse(empty, empty))
    assert np.isnan(smape(empty, empty))


def test_peak_error_only_looks_at_the_top_decile() -> None:
    actual = np.arange(100, dtype=np.float64)
    predicted = actual.copy()
    predicted[-1] = 0.0  # only the largest value is wrong
    # The top decile of arange(100) is the 10 values 90..99; only one is wrong.
    assert peak_error_mae(actual, predicted) == pytest.approx(99.0 / 10.0)
    # A model that is only wrong in the bottom half scores zero on this metric.
    baseline = actual.copy()
    baseline[:90] = 0.0
    assert peak_error_mae(actual, baseline) == pytest.approx(0.0)


def test_zero_and_daytime_error_split_the_target() -> None:
    actual = np.array([0.0, 0.0, 10.0, 20.0])
    predicted = np.array([1.0, 2.0, 12.0, 18.0])
    assert zero_period_error_mae(actual, predicted) == pytest.approx(1.5)
    assert daytime_error_mae(actual, predicted) == pytest.approx(2.0)


def test_zero_period_error_is_none_when_there_are_no_zeros() -> None:
    assert zero_period_error_mae(np.array([1.0, 2.0]), np.array([1.0, 2.0])) is None


def test_ramp_error_needs_two_consecutive_steps() -> None:
    actual = np.array([[1.0, 3.0, 6.0]])
    predicted = np.array([[1.0, 2.0, 3.0]])
    # actual ramps: +2, +3; predicted ramps: +1, +1 -> |1| and |2| -> mean 1.5
    assert ramp_error_mae(actual, predicted) == pytest.approx(1.5)
    assert ramp_error_mae(np.array([[1.0]]), np.array([[1.0]])) is None


def test_evaluate_reports_the_sample_count_with_every_metric() -> None:
    metrics = evaluate(np.arange(50, dtype=np.float64), np.zeros(50))
    assert metrics.n == 50
    payload = metrics.to_dict()
    assert payload["mae"] == pytest.approx(np.arange(50).mean())
    assert "peak_error_mae" in payload and "ramp_error_mae" in payload


def test_metric_set_renders_without_crashing_on_missing_values() -> None:
    metrics = evaluate(np.zeros(10), np.zeros(10))
    row = metrics.render_row()
    assert "n=10" in row
    assert "n/a" in row or "R2" in row


# ======================================================================
# Naive baselines
# ======================================================================


def test_seasonal_offsets_are_exact_not_approximate() -> None:
    """For horizon h the seasonal rule reads t + h - m, not t - m."""
    series = np.arange(0, 2000, dtype=np.float64)
    horizons = (1, 4, 96)
    daily = naive_forecast(series, np.array([1000]), horizons, "naive_seasonal_day")[0]
    assert daily == pytest.approx(
        [series[1000 + 1 - 96], series[1000 + 4 - 96], series[1000 + 96 - 96]]
    )
    weekly = naive_forecast(series, np.array([1000]), horizons, "naive_seasonal_week")[0]
    assert weekly == pytest.approx(
        [series[1000 + 1 - 672], series[1000 + 4 - 672], series[1000 + 96 - 672]]
    )
    # The three horizons must give three different reads, proving the offset depends
    # on h rather than being a single lagged value reused for every horizon.
    assert len(set(daily.tolist())) == 3


def test_last_value_rule_is_deterministic_and_repeats_the_origin() -> None:
    series = np.arange(0, 100, dtype=np.float64)
    prediction = naive_forecast(series, np.array([50, 60]), (1, 4), "naive_last_value")
    assert np.allclose(prediction[:, 0], [50.0, 60.0])
    assert np.allclose(prediction[:, 1], [50.0, 60.0])


def test_drift_adds_the_recent_trend() -> None:
    series = np.arange(0, 100, dtype=np.float64)  # rising by 1 per step
    prediction = naive_forecast(series, np.array([50]), (1, 4), "naive_drift")
    assert prediction[0, 0] == pytest.approx(50.0 + 1.0)
    assert prediction[0, 1] == pytest.approx(50.0 + 2.0)


def test_a_rule_whose_offset_predates_the_record_is_refused() -> None:
    series = np.arange(0, 100, dtype=np.float64)
    with pytest.raises(ValueError, match="too early"):
        naive_forecast(series, np.array([10]), (96,), "naive_seasonal_week")


def test_unknown_naive_model_is_refused() -> None:
    with pytest.raises(KeyError, match="unknown naive model"):
        fit_naive("naive_crystal_ball", (1,))


def test_every_declared_naive_model_has_a_rule() -> None:
    for name in NAIVE_MODELS + ("naive_drift",):
        rule = fit_naive(name, (1, 4))
        assert rule.name == name and rule.description


# ======================================================================
# Classical and neural baselines
# ======================================================================


def test_classical_defaults_are_recorded_not_hidden() -> None:
    for name in CLASSICAL_MODELS:
        params = model_defaults(name)
        assert params["note"], f"{name} must explain its hyperparameters"
    with pytest.raises(KeyError, match="unknown classical model"):
        model_defaults("classical_catboost")


def test_ridge_reproduces_a_known_linear_relationship() -> None:
    x = np.linspace(-1.0, 1.0, 200).reshape(-1, 1)
    y = 3.0 * x.ravel() + 1.0
    model = fit_classical(
        "classical_ridge", x, y, feature_names=("x",), seed=1
    )
    predictions = model.predict(np.array([[0.0]]))
    assert predictions[0] == pytest.approx(1.0, abs=1e-6)
    assert model.coefficients is not None
    assert model.to_dict()["seed"] == 1


def test_classical_fit_refuses_misaligned_or_empty_input() -> None:
    with pytest.raises(ValueError, match="disagree"):
        fit_classical("classical_ridge", np.ones((5, 2)), np.ones(4), feature_names=("a", "b"), seed=1)
    with pytest.raises(ValueError, match="zero rows"):
        fit_classical("classical_ridge", np.ones((0, 2)), np.ones(0), feature_names=("a", "b"), seed=1)


def test_predict_refuses_a_different_feature_count() -> None:
    model = fit_classical("classical_ridge", np.ones((10, 2)), np.ones(10), feature_names=("a", "b"), seed=1)
    with pytest.raises(ValueError, match="was fitted on 2 features"):
        model.predict(np.ones((3, 3)))


def test_neural_baseline_is_reproducible_for_a_fixed_seed() -> None:
    x = np.linspace(-1.0, 1.0, 256).reshape(-1, 1)
    y = np.sin(3.0 * x.ravel())
    first = fit_neural("neural_mlp", x, y, seed=7, threads=2)
    second = fit_neural("neural_mlp", x, y, seed=7, threads=2)
    assert np.allclose(first.predict(x), second.predict(x))
    assert first.parameter_count() > 0
    assert first.epochs_run == 40


def test_neural_baseline_sees_the_ridge_pattern_on_a_linear_problem() -> None:
    """A sanity floor: the network must be able to fit a straight line."""
    x = np.linspace(-1.0, 1.0, 512).reshape(-1, 1)
    y = 2.0 * x.ravel() + 0.5
    model = fit_neural("neural_mlp", x, y, seed=3, threads=2)
    assert np.abs(model.predict(x) - y).mean() < 0.1


def test_gru_baseline_runs_and_reports_its_shape() -> None:
    window = np.linspace(0.0, 1.0, 200 * 8).reshape(200, 8)
    target = window[:, -1] * 2.0
    model = fit_neural("neural_gru", window, target, seed=5, threads=2)
    assert model.input_shape == (8,)
    assert model.predict(window).shape == (200,)


def test_unknown_neural_model_is_refused() -> None:
    with pytest.raises(KeyError, match="unknown neural model"):
        fit_neural("neural_transformer", np.ones((4, 2)), np.ones(4), seed=1)


# ======================================================================
# The experiment runner
# ======================================================================


@pytest.fixture
def small_dataset():
    series = build_series(series_count=2, steps=20000)
    return series, build_dataset(
        series,
        DatasetBuildConfig(lookback=672, horizons=(1, 4), origin_stride=16),
    )


def test_naive_experiment_runs_without_the_series_by_refusal(small_dataset) -> None:
    """A naive model must be given the real history or refuse to run."""
    _, dataset = small_dataset
    with pytest.raises(ValueError, match="must be given the real series history"):
        run_experiment(dataset, model="naive_last_value", seed=1)


def test_experiment_covers_every_horizon(small_dataset) -> None:
    series, dataset = small_dataset
    result = run_experiment(dataset, model="naive_last_value", seed=1, series=series.values)
    assert set(result.metrics_by_horizon) == set(dataset.horizons)
    assert result.model_family == "naive"


def test_experiment_is_reproducible(small_dataset) -> None:
    series, dataset = small_dataset
    for model in ("naive_seasonal_day", "classical_ridge"):
        ok, detail = verify_reproducible(dataset, model=model, seed=1, series=series.values)
        assert ok, detail


def test_ramp_metric_appears_only_for_consecutive_horizons(small_dataset) -> None:
    series, dataset = small_dataset
    result = run_experiment(dataset, model="naive_last_value", seed=1, series=series.values)
    # horizons are (1, 4): not consecutive, so the ramp metric must be absent.
    assert result.metrics_by_horizon[1]["ramp_error_mae"] is None
    assert result.metrics_by_horizon[4]["ramp_error_mae"] is None


def test_ramp_metric_appears_when_horizons_are_consecutive() -> None:
    series = build_series(series_count=2, steps=20000)
    dataset = build_dataset(
        series, DatasetBuildConfig(lookback=672, horizons=(1, 2, 4), origin_stride=16)
    )
    result = run_experiment(dataset, model="naive_last_value", seed=1, series=series.values)
    assert result.metrics_by_horizon[1]["ramp_error_mae"] is not None
    assert result.metrics_by_horizon[2]["ramp_error_mae"] is None


def test_metrics_are_reported_in_the_target_unit_and_in_per_unit(small_dataset) -> None:
    series, dataset = small_dataset
    result = run_experiment(dataset, model="classical_ridge", seed=1, series=series.values)
    payload = result.metrics_by_horizon[1]
    assert payload["unit"] == "kW"
    assert payload["storage"].startswith("per unit")
    # kW MAE must exceed per-unit MAE, because scales are kW, not fractions.
    assert payload["mae"] > payload["mae_per_unit"]


def test_unknown_model_lists_the_families(small_dataset) -> None:
    _, dataset = small_dataset
    with pytest.raises(KeyError, match="unknown model"):
        run_experiment(dataset, model="magic", seed=1)


def test_comparison_table_renders_all_models_and_horizons(small_dataset) -> None:
    series, dataset = small_dataset
    results = [
        run_experiment(dataset, model=name, seed=1, series=series.values)
        for name in ("naive_last_value", "classical_ridge", "neural_mlp")
    ]
    table = compare(results)
    assert "MAE" in table and "sMAPE%" in table
    for name in ("naive_last_value", "classical_ridge", "neural_mlp"):
        assert name in table
    assert len(table.splitlines()) == 1 + 1 + 3 * len(dataset.horizons)


def test_comparison_produces_no_composite_score(small_dataset) -> None:
    """A single weighted score would be an invented criterion."""
    series, dataset = small_dataset
    result = run_experiment(dataset, model="classical_ridge", seed=1, series=series.values)
    table = compare([result])
    assert "score" not in table.lower()


def test_all_declared_models_are_routable(small_dataset) -> None:
    series, dataset = small_dataset
    for model in ALL_MODELS:
        result = run_experiment(dataset, model=model, seed=1, series=series.values, threads=2)
        assert result.metrics_by_horizon
        assert result.model_family in {"naive", "classical", "neural"}


# ======================================================================
# Regimes and error analysis
# ======================================================================


def test_load_regimes_are_terciles_and_cover_every_row() -> None:
    actual = np.linspace(0.0, 1.0, 300)
    axes = assign_regimes(
        actual, target_id="customer_load", origin_index=np.arange(300)
    )
    assert {axis.name for axis in axes} == {"load_level"}
    labels = axes[0].labels
    assert set(labels) == {"low", "typical", "high"}
    assert np.all(np.isin(labels, ["low", "typical", "high"]))


def test_load_ramp_regime_needs_a_previous_value() -> None:
    actual = np.linspace(0.0, 1.0, 300)
    without = assign_regimes(actual, target_id="customer_load", origin_index=np.arange(300))
    assert {axis.name for axis in without} == {"load_level"}
    previous = np.linspace(0.0, 0.5, 300)
    with_ramp = assign_regimes(
        actual,
        target_id="customer_load",
        origin_index=np.arange(300),
        previous_actual=previous,
    )
    assert "load_ramp" in {axis.name for axis in with_ramp}


def test_solar_phases_are_derived_from_the_generation_derivative() -> None:
    actual = np.concatenate(
        [
            np.zeros(10),
            np.linspace(0.0, 100.0, 10),
            np.full(10, 100.0),
            np.linspace(100.0, 0.0, 10),
            np.zeros(10),
        ]
    )
    previous = np.concatenate([np.zeros(1), actual[:-1]])
    axes = assign_regimes(
        actual,
        target_id="pv_generation",
        origin_index=np.arange(actual.size),
        previous_actual=previous,
    )
    names = {axis.name for axis in axes}
    assert {"solar_phase", "solar_level"} <= names
    phase = next(a for a in axes if a.name == "solar_phase")
    assert set(phase.labels) >= {"night", "morning_ramp", "midday", "evening_ramp"}


def test_regime_report_quantifies_the_spread() -> None:
    """A large ratio is the evidence that regime structure exists."""
    actual = np.concatenate([np.full(50, 0.1), np.full(50, 10.0)])
    axes = assign_regimes(
        actual, target_id="customer_load", origin_index=np.arange(100)
    )
    errors = {"good": np.concatenate([np.full(50, 0.1), np.full(50, 1.0)])}
    report = regime_report(axes, errors)
    assert report["max_regime_mae_ratio"]["good"] == pytest.approx(10.0, rel=1e-6)
    assert "interpretation" in report


def test_regime_report_handles_a_flat_error_field() -> None:
    actual = np.linspace(0, 1, 30)
    axes = assign_regimes(actual, target_id="customer_load", origin_index=np.arange(30))
    report = regime_report(axes, {"flat": np.zeros(30)})
    assert report["max_regime_mae_ratio"]["flat"] is None


def test_error_analysis_finds_the_worst_rows_and_series() -> None:
    actual = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    predicted = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 20.0])
    report = error_analysis(
        actual=actual,
        predicted=predicted,
        model="test",
        origin_index=np.arange(96, 102),
        series_ids=("a", "b"),
        series_index=np.array([0, 0, 0, 1, 1, 1]),
        timestep_minutes=15,
        origin_iso="2018-01-01T00:00:00+00:00",
        quality_flags=("ok",),
    )
    assert report["worst_rows"][0]["absolute_error"] == pytest.approx(14.0)
    assert report["worst_rows"][0]["series_id"] == "b"
    assert report["worst_series"][0]["series_id"] == "b"
    assert report["error_by_hour_of_day"]
    assert report["data_quality_flags"] == ["ok"]
    json.dumps(report)