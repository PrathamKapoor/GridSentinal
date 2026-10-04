"""Phase 8 offline analysis, verdict and the runner, end to end on synthetic data.

The runner is exercised in full here - real estimator fits, real coverage arithmetic, real
artifacts - on a panel small enough to run in seconds. What the synthetic panel cannot
test is the *numbers*; what it can test, and what these tests check, is that the machinery
cannot produce a result that is internally inconsistent, that cannot be reproduced, or that
silently publishes an interval whose own calibration status reads FAIL.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from energy_intelligence.ml.uncertainty.evaluation import (
    imminent_ramp_detection,
    interval_audit,
    overconfidence_analysis,
    reliability_table,
    uncertainty_ranking,
)
from energy_intelligence.ml.uncertainty.intervals import (
    DEFAULT_NOMINAL_LEVELS,
    widths_from_scale,
)
from energy_intelligence.ml.uncertainty.measures import (
    disagreement_measures,
    spearman,
)
from energy_intelligence.ml.uncertainty.metrics import (
    coverage,
    expected_calibration_error,
    interval_score,
    mean_width,
    weighted_interval_score,
)
from energy_intelligence.ml.uncertainty.offline.analysis import (
    coverage_tolerance,
    evaluate_method,
    expert_point_forecast_table,
    final_comparison_table,
    method_table,
    regime_breakdown,
)
from energy_intelligence.ml.uncertainty.offline.verdict import (
    best_method_per_horizon,
    phase8_verdict,
)

HORIZONS = (1, 4, 96)
LEVELS = DEFAULT_NOMINAL_LEVELS


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def test_coverage_counts_the_closed_interval():
    actual = np.array([1.0, 2.0, 3.0, 4.0])
    lower = np.array([0.5, 2.5, 0.0, 10.0])
    upper = np.array([1.5, 3.5, 2.0, 11.0])
    # Only the first value lies inside its interval: the bounds are closed, so the 1.0 that
    # sits exactly on [0.5, 1.5] counts.
    assert coverage(actual, lower, upper) == pytest.approx(0.25)


def test_interval_score_penalises_both_width_and_miss():
    """A proper scoring rule: the same width costs more when it misses."""
    actual = np.array([0.0, 0.0])
    scores = interval_score(
        actual, np.array([-1.0, -1.0]), np.array([1.0, 3.0]), 0.1
    )
    assert scores[0] == pytest.approx(2.0)
    assert scores[1] > scores[0]
    wide = interval_score(
        actual, np.array([-3.0, -3.0]), np.array([3.0, 3.0]), 0.1
    )
    assert wide[0] > scores[0]


def test_wider_intervals_score_worse_when_both_cover():
    """Otherwise sharpness would be rewarded unconditionally."""
    actual = np.zeros(100)
    tight = weighted_interval_score(
        actual, {0.9: (np.full(100, -2.0), np.full(100, 2.0))}
    )
    loose = weighted_interval_score(
        actual, {0.9: (np.full(100, -5.0), np.full(100, 5.0))}
    )
    assert tight < loose


def test_expected_calibration_error_is_zero_for_a_perfect_calibration():
    """One interval covering every row at the nominal rate must score zero."""
    lower = np.full(1000, -1.0)
    upper = np.full(1000, 1.0)
    actual = np.concatenate([np.zeros(900), np.full(100, 5.0)])
    record = expected_calibration_error(actual, lower, upper, 0.9, bins=5)
    assert record["expected_calibration_error"] == pytest.approx(0.0, abs=1e-9)
    assert record["nominal_level"] == 0.9
    assert record["binned_by"] == "interval width"
    assert sum(entry["rows"] for entry in record["bins"]) == 1000


def test_expected_calibration_error_is_large_for_a_shifted_interval():
    lower = np.zeros(1000)
    upper = np.full(1000, 1.0)
    actual = np.linspace(-5.0, 5.0, 1000)
    record = expected_calibration_error(actual, lower, upper, 0.9, bins=5)
    assert record["expected_calibration_error"] > 0.5


def test_mean_width_is_the_average_span():
    assert mean_width(np.array([0.0, 2.0]), np.array([1.0, 3.0])) == pytest.approx(1.0)
    assert mean_width(np.array([0.0, 2.0]), np.array([2.0, 4.0])) == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Ranking, reliability, overconfidence
# ---------------------------------------------------------------------------


def test_uncertainty_ranking_correlates_width_with_error():
    width = np.arange(100.0)
    error = width**2
    record = uncertainty_ranking(
        uncertainty=width, absolute_error_kw=error, label="w", deciles=10
    )
    assert record["spearman"] == pytest.approx(1.0)
    assert record["top_decile_error_multiple"] > 1.0
    assert record["deciles"][-1]["uncertainty_high"] == pytest.approx(width.max())
    assert record["deciles"][-1]["error_multiple"] > 1.0


def test_uncertainty_ranking_reports_nan_for_a_constant_uncertainty():
    """A constant carries no rank information, and saying NaN beats saying 0."""
    record = uncertainty_ranking(
        uncertainty=np.ones(50), absolute_error_kw=np.arange(50.0),
        label="w", deciles=5,
    )
    assert record["spearman"] != record["spearman"]
    assert len(record["deciles"]) == 1


def test_reliability_table_bins_by_uncertainty_and_reports_coverage():
    rng = np.random.default_rng(0)
    width = rng.uniform(0.5, 4.0, 4000)
    lower = rng.normal(0.0, 1.0, 4000)
    upper = lower + width
    table = reliability_table(
        actual_kw=rng.normal(0.0, 1.0, 4000), lower_kw=lower, upper_kw=upper,
        nominal_level=0.9, uncertainty=width, bins=5,
    )
    assert table["binned_by"] == "predicted scale"
    assert sum(entry["rows"] for entry in table["bins"]) == 4000
    for entry in table["bins"]:
        assert 0.0 <= entry["empirical_coverage"] <= 1.0
        assert entry["scale_low"] <= entry["scale_high"]


def test_overconfidence_analysis_finds_the_confident_and_wrong_rows():
    """A confident row that is wrong is the failure the whole phase is for."""
    actual = np.concatenate([np.zeros(90), np.full(10, 100.0)])
    point = np.zeros(100)
    over = overconfidence_analysis(
        actual_kw=actual, point_kw=point,
        interval=widths_from_scale(
            point_kw=point[:, None], scale=np.ones((100, 1)), multiplier=1.0,
            method="m", nominal_level=0.9,
        ),
        band=np.array(["low"] * 50 + ["high"] * 50, dtype=object),
        nominal_level=0.9,
    )
    # The 10 large errors sit in the HIGH band, so they are not the dangerous class: a
    # confident row here is one the system flagged as uncertain and was right to.
    assert over["confident_and_severe_rows"] == 0
    assert over["by_band"]["high"]["severe_rows"] == 10
    assert over["by_band"]["low"]["severe_rate"] == 0.0
    assert over["by_band"]["low"]["mae_kw"] == pytest.approx(0.0)


def test_imminent_ramp_detection_compares_uncertainty_against_a_future_label():
    rng = np.random.default_rng(1)
    ramp = np.abs(rng.normal(0.0, 1.0, 2000))
    uncertainty = ramp + rng.normal(0.0, 0.1, 2000)
    record = imminent_ramp_detection(
        uncertainty=uncertainty, absolute_error_kw=ramp, future_ramp_kw=ramp,
        label="w",
    )
    assert record["spearman_uncertainty_vs_future_ramp"] > 0.9
    assert (
        record["severe_ramp_rate_in_high_uncertainty"]
        > record["severe_ramp_rate_in_low_uncertainty"]
    )
    assert record["enrichment_high_over_base"] > 1.0
    assert set(record["by_future_ramp_tercile"]) == {"low", "typical", "high"}
    assert "diagnostic" in record["note"] or "not an input" in record["note"]


# ---------------------------------------------------------------------------
# Disagreement
# ---------------------------------------------------------------------------


def test_spearman_is_rank_based_and_handles_ties():
    assert spearman(np.arange(10.0), np.arange(10.0)) == pytest.approx(1.0)
    assert spearman(np.arange(10.0), -np.arange(10.0)) == pytest.approx(-1.0)
    tied = spearman(np.array([1.0, 1.0, 2.0, 2.0]), np.array([1.0, 2.0, 3.0, 4.0]))
    assert tied == tied
    with pytest.raises(ValueError, match="aligned samples"):
        spearman(np.arange(5.0), np.arange(4.0))


def test_disagreement_measures_are_per_unit_and_shape_checked():
    forecasts = np.array(
        [[[1.0, 1.0]], [[2.0, 2.0]], [[4.0, 6.0]]]
    )
    point = np.array([[1.5, 2.0]])
    measures = disagreement_measures(
        expert_forecast_kw=forecasts, row_scale_kw=np.array([10.0]),
        horizons=(1, 2), expert_names=("a", "b", "c"), point_kw=point,
    )
    assert measures.primary.shape == (1, 2)
    assert measures["max_minus_min"][0, 0] == pytest.approx(0.3)
    assert measures.describe(np.array([0]))["max_minus_min"]["1"]["mean"] == pytest.approx(0.3)
    with pytest.raises(ValueError, match="at least two experts"):
        disagreement_measures(
            expert_forecast_kw=forecasts[:1], row_scale_kw=np.array([10.0]),
            horizons=(1, 2), expert_names=("a",), point_kw=point,
        )
    with pytest.raises(ValueError, match="must be positive"):
        disagreement_measures(
            expert_forecast_kw=forecasts, row_scale_kw=np.array([0.0]),
            horizons=(1, 2), expert_names=("a", "b", "c"), point_kw=point,
        )


# ---------------------------------------------------------------------------
# Tables and the verdict
# ---------------------------------------------------------------------------


def test_coverage_tolerance_is_the_floor_at_this_sample_size():
    small = coverage_tolerance(179520, 0.95)
    assert small["tolerance"] == pytest.approx(0.01)
    assert "floor" in small["basis"]
    tiny = coverage_tolerance(100, 0.95)
    assert tiny["tolerance"] > 0.01
    assert tiny["standard_error"] > 0.0


def test_audit_reports_coverage_width_and_a_pass_flag():
    rng = np.random.default_rng(2)
    actual = rng.normal(10.0, 1.0, size=(500, len(HORIZONS)))
    point = actual + rng.normal(0.0, 0.5, size=actual.shape)
    intervals = {
        float(level): widths_from_scale(
            point_kw=point, scale=np.ones_like(point), multiplier=2.0,
            method="m", nominal_level=level,
        )
        for level in (0.9,)
    }
    audit = interval_audit(
        method="m", actual_kw=actual, point_kw=point, bounds=intervals,
        nominal_levels=(0.9,), horizons=HORIZONS,
        disagreement={"max_minus_min": np.ones_like(point)}, tolerance=0.01,
    )
    for horizon in HORIZONS:
        entry = audit[str(horizon)]
        assert entry["levels"]["0.9"]["coverage"] > 0.9
        assert entry["point_mae_kw"] > 0.0
        assert "max_minus_min" in entry["uncertainty_vs_error"]


def test_method_table_sorts_by_interval_score_within_each_horizon():
    rows = [
        {"method": "a", "horizon": 1, "interval_score_kw": 2.0, "passes": True},
        {"method": "b", "horizon": 1, "interval_score_kw": 1.0, "passes": True},
        {"method": "c", "horizon": 4, "interval_score_kw": 0.5, "passes": False},
    ]
    ordered = best_method_per_horizon(rows, (1, 4))
    assert ordered == {1: "b", 4: "c"}
    assert best_method_per_horizon(rows, (1, 4), calibrated_only=True) == {1: "b", 4: "c"}
    uncalibrated = [
        {"method": "narrow", "horizon": 1, "interval_score_kw": 0.1, "passes": False},
        {"method": "wide", "horizon": 1, "interval_score_kw": 5.0, "passes": True},
    ]
    assert best_method_per_horizon(uncalibrated, (1,)) == {1: "narrow"}
    assert best_method_per_horizon(uncalibrated, (1,), calibrated_only=True) == {1: "wide"}


def test_final_comparison_table_labels_the_level_it_measured():
    rows = [
        {
            "method": "m", "horizon": 1, "coverage": 0.9, "coverage_error": 0.0,
            "passes": True, "mean_width_kw": 2.0, "sharpness_ratio": 3.0,
            "interval_score_kw": 1.0, "spearman_width_vs_error": 0.5,
        }
    ]
    assert "coverage @95%" in final_comparison_table(rows, [1], 0.95)
    assert "coverage @50%" in final_comparison_table(rows, [1], 0.5)
    assert "@90%" not in final_comparison_table(rows, [1], 0.95)


def test_verdict_prefers_a_calibrated_method_over_a_narrower_uncalibrated_one():
    rows = [
        {
            "method": "narrow", "horizon": h, "interval_score_kw": 1.0,
            "passes": False, "coverage": 0.5, "coverage_error": -0.45,
            "mean_width_kw": 0.5, "sharpness_ratio": 1.0, "spearman_width_vs_error": 0.9,
        }
        for h in HORIZONS
    ] + [
        {
            "method": "calibrated", "horizon": h, "interval_score_kw": 2.0,
            "passes": True, "coverage": 0.95, "coverage_error": 0.0,
            "mean_width_kw": 3.0, "sharpness_ratio": 4.0, "spearman_width_vs_error": 0.4,
        }
        for h in HORIZONS
    ]
    results = {
        name: {
            "by_horizon": {
                str(h): {
                    "metrics": {"levels": {"0.95": {"passes": any(
                        r["method"] == name and r["horizon"] == h and r["passes"]
                        for r in rows
                    )}}},
                    "ranking_by_width": {"spearman": 0.4, "top_decile_error_multiple": 1.5},
                }
                for h in HORIZONS
            }
        }
        for name in ("narrow", "calibrated")
    }
    verdict = phase8_verdict(
        rows=rows, results=results, horizons=HORIZONS, nominal_level=0.95,
        baseline_method="calibrated", tolerance=0.01,
        disagreement={"max_minus_min": {"1": 0.4, "4": 0.4, "96": 0.4}},
    )
    assert verdict["components"]["can_be_estimated"]["answer"] == "YES"
    assert verdict["components"]["can_be_estimated"]["calibrated_methods"] == ["calibrated"]
    machinery = verdict["components"]["is_worth_the_machinery"]
    assert machinery["per_horizon"]["1"]["best_calibrated_method"] == "calibrated"
    assert machinery["per_horizon"]["1"]["unconstrained_best_method"] == "narrow"
    assert machinery["per_horizon"]["1"]["unconstrained_best_is_calibrated"] is False
    assert verdict["best_method_by_horizon_unconstrained"]["1"] == "narrow"


def test_verdict_is_no_when_nothing_is_calibrated():
    rows = [
        {
            "method": "m", "horizon": h, "interval_score_kw": 1.0, "passes": False,
            "coverage": 0.5, "coverage_error": -0.45, "mean_width_kw": 1.0,
            "sharpness_ratio": 1.0, "spearman_width_vs_error": 0.0,
        }
        for h in HORIZONS
    ]
    results = {
        "m": {
            "by_horizon": {
                str(h): {
                    "metrics": {"levels": {"0.95": {"passes": False}}},
                    "ranking_by_width": {"spearman": 0.0, "top_decile_error_multiple": 1.0},
                }
                for h in HORIZONS
            }
        }
    }
    verdict = phase8_verdict(
        rows=rows, results=results, horizons=HORIZONS, nominal_level=0.95,
        baseline_method="m", tolerance=0.01, disagreement={},
    )
    assert verdict["answer"] == "NO"
    assert verdict["components"]["can_be_estimated"]["answer"] == "NO"


def test_regime_breakdown_reports_width_and_error_per_regime():
    from energy_intelligence.ml.analysis import RegimeDefinition

    actual = np.linspace(0.0, 10.0, 300)[:, None] * np.ones((1, len(HORIZONS)))
    point = actual.copy()
    interval = widths_from_scale(
        point_kw=point, scale=np.tile(np.array([[0.1, 1.0, 5.0]]), (300, 1)),
        multiplier=2.0, method="m", nominal_level=0.9,
    )
    labels = np.array(["low"] * 150 + ["high"] * 150, dtype=object)
    breakdown = regime_breakdown(
        actual_kw=actual, point_kw=point, interval=interval,
        axes=(
            RegimeDefinition(
                name="load_level", description="d", labels=labels,
                detail=np.linspace(0.0, 1.0, 300),
            ),
        ),
        horizon_index=2, nominal_level=0.9,
    )
    axis = breakdown["load_level"]
    assert set(axis["regimes"]) == {"low", "high"}
    assert axis["regimes"]["low"]["n"] == 150
    assert axis["regimes"]["low"]["share_of_rows"] == pytest.approx(0.5)
    # The zero clip is what makes the two halves differ: the low-demand half's bounds sit
    # below zero and are raised, so its interval is narrower than the high half's.
    assert axis["width_ratio_worst_to_easiest"] > 1.0
    assert axis["error_ratio_worst_to_easiest"] is None
    assert axis["width_tracks_error"] is False


def test_regime_breakdown_refuses_axes_over_a_different_row_set():
    from energy_intelligence.ml.analysis import RegimeDefinition

    actual = np.zeros((10, len(HORIZONS)))
    with pytest.raises(ValueError, match="labels for"):
        regime_breakdown(
            actual_kw=actual, point_kw=actual,
            interval=widths_from_scale(
                point_kw=actual, scale=np.ones_like(actual), multiplier=1.0,
                method="m", nominal_level=0.9,
            ),
            axes=(
                RegimeDefinition(
                    name="a", description="d",
                    labels=np.array(["low"] * 3, dtype=object),
                ),
            ),
            horizon_index=0, nominal_level=0.9,
        )


def test_expert_point_forecast_table_adds_the_ensemble_and_checks_shapes():
    forecasts = {"a": np.zeros((5, len(HORIZONS))), "b": np.ones((5, len(HORIZONS)))}
    table = expert_point_forecast_table(
        forecasts=forecasts, actual_kw=np.zeros((5, len(HORIZONS))), horizons=HORIZONS
    )
    assert table["a"]["mae_kw"]["1"] == pytest.approx(0.0)
    assert table["b"]["mae_kw"]["1"] == pytest.approx(1.0)
    assert set(table["a"]["mae_kw"]) == {"1", "4", "96"}
    with pytest.raises(ValueError, match="does not match"):
        expert_point_forecast_table(
            forecasts={"a": np.zeros((5, len(HORIZONS)))},
            actual_kw=np.zeros((4, len(HORIZONS))), horizons=HORIZONS,
        )


# ---------------------------------------------------------------------------
# The runner
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    """One full synthetic Phase 8 run, shared by the tests below."""
    from tests.ml.conftest import (
        build_test_panel,
        build_test_values,
        build_uncertainty_cache,
        build_uncertainty_config,
        write_phase7_result,
    )

    root = tmp_path_factory.mktemp("phase8")
    panel = build_test_panel(n_rows=1800)
    cache = build_uncertainty_cache(panel)
    phase7 = write_phase7_result(root / "phase7" / "stub" / "result.json")
    artifact_dir = root / "artifacts"
    result = _run(
        cache=cache,
        panel=panel,
        values=build_test_values(panel),
        artifact_dir=artifact_dir,
        phase7=phase7,
        config=build_uncertainty_config(),
    )
    return result, artifact_dir, phase7, cache, panel


def _run(*, cache, panel, values, artifact_dir, phase7, config, **overrides):
    from energy_intelligence.ml.uncertainty.offline.experiment import run_phase8

    kwargs = {
        "cache": cache,
        "panel": panel,
        "values": values,
        "artifact_dir": artifact_dir,
        "config": config,
        "experiment_id": "phase8-test",
        "dataset_version": "synthetic-v0",
        "seed": 5,
        "series_ids": tuple(f"load_s{i}" for i in range(3)),
        "origin_iso": "2025-01-01T00:00:00",
        "phase7_result_path": phase7,
        "published_point_mae": None,
        "nominal_levels": config.nominal_levels,
        "run_ablations": False,
    }
    kwargs.update(overrides)
    return run_phase8(**kwargs)


def test_runner_writes_every_artifact_it_promises(run):
    result, artifact_dir, *_ = run
    for key, path in result.artifacts.items():
        assert Path(path).is_file(), f"{key} was promised at {path}"
    assert (artifact_dir / "result.json").is_file()
    payload = json.loads(Path(result.artifacts["result"]).read_text(encoding="utf-8"))
    assert payload["verdict"]["answer"] in {"YES", "NO", "PARTIALLY"}
    assert Path(result.artifacts["summary"]).read_text(encoding="utf-8").startswith("# Phase 8")


def test_runner_halves_the_validation_split_and_leaves_test_alone(run):
    result, _, _, cache, panel = run
    validation = panel.n_rows // 2
    assert result.split["calibration_fit_rows"] == validation // 2
    assert result.split["calibration_conformity_rows"] == validation - validation // 2
    assert result.split["test_rows"] == panel.n_rows - validation
    assert result.split["calibration_fit_rows"] + result.split["calibration_conformity_rows"] == validation
    assert cache.n_rows == panel.n_rows


def test_runner_reports_split_parity_on_all_three_row_sets(run):
    result, *_ = run
    for horizon in HORIZONS:
        entry = result.split_parity[str(horizon)]
        assert entry["calibration_fit_rows"] > 0
        assert entry["calibration_conformity_rows"] > 0
        assert entry["test_rows"] > 0
        assert "fit_vs_conformity_pct" in entry
        assert "test_vs_conformity_pct" in entry


def test_runner_publishes_only_calibrated_intervals(run):
    """A published interval whose own status reads FAIL is not a result."""
    result, *_ = run
    status = result.artifact_summary["calibration_status"]
    assert len(status) == len(HORIZONS)
    for horizon, method, entry in zip(
        HORIZONS, result.artifact_summary["method"],
        (result.selection["by_horizon"][str(h)] for h in HORIZONS),
    ):
        chosen = entry["method"]
        assert method == chosen
        if entry.get("calibrated", True):
            assert status[HORIZONS.index(horizon)] == "PASS"
        levels = result.methods[chosen][str(horizon)]["metrics"]["levels"]
        assert levels[str(result.selection["level"])]["passes"] or not entry.get(
            "calibrated", True
        )


def test_runner_bands_use_per_horizon_calibration_cut_points(run):
    result, *_ = run
    counts = result.artifact_summary["band_counts"]
    for horizon in HORIZONS:
        entry = counts[str(horizon)]
        assert sum(entry.values()) == result.split["test_rows"]
        assert set(entry) == {"low", "medium", "high"}
    cuts = result.artifact_summary["band_cuts_kw"]
    assert set(cuts) == {"1", "4", "96"}
    for horizon in HORIZONS:
        low, high = cuts[str(horizon)]
        assert low < high


def test_runner_evaluates_every_method_at_every_level_and_horizon(run):
    result, *_ = run
    for name, entry in result.methods.items():
        assert set(entry) == {str(h) for h in HORIZONS}
        for horizon in HORIZONS:
            levels = entry[str(horizon)]["metrics"]["levels"]
            for level in LEVELS:
                measured = levels[str(level)]
                assert measured["coverage"] == pytest.approx(measured["coverage"])
                assert 0.0 <= measured["coverage"] <= 1.0
                assert measured["mean_width_kw"] >= 0.0
                assert measured["passes"] == (
                    abs(measured["coverage"] - level) <= result.verdict["coverage_tolerance"]
                )


def test_runner_applies_the_baseline_method_to_every_expert(run):
    result, *_ = run
    for name in ("persistence", "classical_hist_gbm", "phase5_tcn"):
        assert f"global_residual_on_{name}" in result.methods


def test_runner_records_disagreement_against_realised_error(run):
    result, *_ = run
    for measure, by_horizon in result.disagreement["spread_vs_error_spearman"].items():
        assert set(by_horizon) == {str(h) for h in HORIZONS}
        for value in by_horizon.values():
            assert -1.0 <= value <= 1.0
    assert result.disagreement["units"] == "per unit of the row's rated kW"


def test_runner_refuses_weights_that_differ_from_phase7(run):
    from tests.ml.conftest import build_uncertainty_config

    result, _, phase7, cache, panel = run
    from tests.ml.conftest import build_test_values

    drifted = build_uncertainty_config()
    object.__setattr__(drifted, "label", "uncertainty-drift")
    broken = dict(drifted.fixed_ensemble_weights)
    broken[1] = {**broken[1], "phase5_tcn": 0.90}
    object.__setattr__(drifted, "fixed_ensemble_weights", broken)
    with pytest.raises(ValueError, match="do not match Phase 7"):
        _run(
            cache=cache, panel=panel, values=build_test_values(panel),
            artifact_dir=Path(phase7).parent / "drift", phase7=phase7, config=drifted,
        )
    assert result.verdict is not None


def test_runner_reports_parity_against_a_published_reference(run):
    result, _, _, cache, panel = run
    from tests.ml.conftest import build_test_values, build_uncertainty_config

    reference = {
        name: {str(h): 0.5 + 0.1 * h for h in HORIZONS}
        for name in ("persistence", "classical_hist_gbm", "phase5_tcn", "fixed_ensemble")
    }
    parity = _run(
        cache=cache, panel=panel, values=build_test_values(panel),
        artifact_dir=Path(phase7_path(run)).parent / "parity",
        phase7=phase7_path(run), config=build_uncertainty_config(),
        published_point_mae=reference,
    ).parity
    assert parity["matches"] is False
    assert parity["max_relative_difference_pct"] > 0.0
    assert set(parity["measured_test_mae_kw"]) == {"1", "4", "96"}


def phase7_path(run) -> Path:
    return run[2]


def test_runner_is_reproducible_for_a_fixed_seed(run):
    result, artifact_dir, phase7, cache, panel = run
    from tests.ml.conftest import build_test_values, build_uncertainty_config

    again = _run(
        cache=cache, panel=panel, values=build_test_values(panel),
        artifact_dir=Path(phase7).parent / "repeat", phase7=phase7,
        config=build_uncertainty_config(), seed=5,
    )
    assert again.comparison == result.comparison
    assert again.selection == result.selection
    assert again.verdict["answer"] == result.verdict["answer"]
    assert again.artifact_summary["calibration_status"] == (
        result.artifact_summary["calibration_status"]
    )
    assert artifact_dir.is_dir()


def test_runner_records_the_data_quality_question_as_a_measurement(run):
    result, *_ = run
    assert result.data_quality["run"] is False
    assert "zero variance" in result.data_quality["reason"]
    assert result.data_quality["available_metadata"]["per_timestep_quality_flags"] is False


def test_runner_writes_a_trace_whose_rows_are_self_consistent(run):
    """Every traced line must be checkable on its own terms.

    The point of the trace is that a published interval can be audited: the origin, the
    inputs that produced the width, the outcome, and the method that was selected for that
    horizon.
    """
    result, *_ = run
    lines = Path(result.artifacts["trace"]).read_text(encoding="utf-8").strip().splitlines()
    assert lines
    horizons = {int(json.loads(line)["horizon_steps"]) for line in lines}
    assert horizons == set(HORIZONS)
    methods = {int(json.loads(line)["horizon_steps"]): json.loads(line)["method"] for line in lines}
    assert methods == {
        horizon: result.artifact_summary["method"][index]
        for index, horizon in enumerate(HORIZONS)
    }
    for line in lines:
        record = json.loads(line)
        assert record["nominal_level"] == result.selection["level"]
        assert record["calibration_status"] in {"PASS", "FAIL", "NOT_EVALUATED"}
        assert record["band"] in {"low", "medium", "high"}
        assert record["model_version"] == "phase8-test"
        assert record["width_kw"] == pytest.approx(
            record["upper_kw"] - record["lower_kw"], abs=1e-3
        )
        assert record["absolute_error_kw"] == pytest.approx(
            abs(record["actual_kw"] - record["point_kw"]), abs=1e-3
        )
        assert record["learned_scale_kw"] >= 0.0
        assert record["expert_spread_kw"] >= 0.0
        assert set(record["expert_spread_per_unit"]) == set(
            result.disagreement["spread_vs_error_spearman"]
        )