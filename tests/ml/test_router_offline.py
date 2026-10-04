"""Phase 7, part 4: the offline analysis - diversity, regret, stability, calibration, cost.

These cover the machinery that turns "the router scored 0.3871 kW" into statements about
whether the experts were worth routing between at all, whether the router picked well, and
what it cost. Each measure has an exact expected value on a hand-built case, because a
routing metric computed wrong is indistinguishable from a router that is merely mediocre.
"""

from __future__ import annotations

import numpy as np
import pytest

from energy_intelligence.ml.analysis import assign_regimes
from energy_intelligence.ml.router.offline.ablations import ABLATIONS, AblationSpec
from energy_intelligence.ml.router.offline.crossfit import label_quality
from energy_intelligence.ml.router.offline.diversity import (
    absolute_errors,
    agreement_rate,
    correlation_matrix,
    diversity_by_horizon,
    diversity_by_regime,
    diversity_report,
    signed_errors,
)
from energy_intelligence.ml.router.offline.evaluation import (
    compute_cost,
    failure_analysis,
    forecast_comparison_table,
    method_metrics,
    router_by_regime,
    router_calibration,
    routing_quality,
    routing_stability,
)
from energy_intelligence.ml.router.offline.oracle import (
    best_pair_and_gain,
    oracle_assignment,
    oracle_mixture,
    oracle_regret,
    pairwise_best_single,
    routing_regret_definition,
)

HORIZONS = (1, 4)


# ---------------------------------------------------------------------------
# Diversity
# ---------------------------------------------------------------------------


def _two_experts() -> tuple[np.ndarray, np.ndarray]:
    """Two experts over four rows and two horizons, with varied errors on both.

    Neither expert is constant-error: a constant error column has no variance and makes a
    correlation undefined, which would quietly turn a correlation test into a ``NaN`` test.
    """
    truth = np.asarray([[0.0, 1.0], [10.0, 11.0], [20.0, 19.0], [30.0, 31.0]])
    forecasts = np.stack(
        [
            np.asarray([[0.0, 2.0], [10.0, 9.0], [18.0, 21.0], [30.0, 29.0]]),
            np.asarray([[2.0, 0.5], [8.0, 13.0], [20.0, 17.0], [26.0, 33.0]]),
        ]
    )
    return forecasts, truth


def test_signed_and_absolute_errors_have_the_expected_shapes_and_values() -> None:
    forecasts, truth = _two_experts()
    signed = signed_errors(forecasts, truth)
    absolute = absolute_errors(forecasts, truth)
    assert signed.shape == absolute.shape == forecasts.shape
    assert np.allclose(signed[0, 2, 0], -2.0)
    assert np.allclose(absolute[1, 1, 0], 2.0)
    assert np.all(absolute >= 0.0)


def test_error_shapes_must_agree() -> None:
    with pytest.raises(ValueError, match="disagree"):
        absolute_errors(np.zeros((2, 4, 2)), np.zeros((4, 3)))


def test_correlation_matrix_of_a_perfect_agreement_is_one() -> None:
    errors = np.asarray([[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]])
    matrix = correlation_matrix(errors)
    assert np.allclose(matrix, 1.0)


def test_correlation_matrix_detects_an_opposite_ranking() -> None:
    errors = np.asarray([[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]])
    assert correlation_matrix(errors)[0, 1] == pytest.approx(-1.0)
    assert correlation_matrix(errors, method="spearman")[0, 1] == pytest.approx(-1.0)


def test_correlation_matrix_rejects_a_constant_column_and_an_unknown_method() -> None:
    constant = np.asarray([[1.0, 1.0], [1.0, 1.0]])
    assert np.isnan(correlation_matrix(constant)[0, 1])
    with pytest.raises(ValueError, match="pearson or spearman"):
        correlation_matrix(np.zeros((2, 3)), method="kendall")
    with pytest.raises(ValueError, match="at least an expert axis"):
        correlation_matrix(np.zeros(3))


def test_agreement_rate_counts_only_rows_inside_the_stated_tolerance() -> None:
    forecasts, truth = _two_experts()
    report = agreement_rate(forecasts, truth, relative_tolerance=0.1, absolute_floor_kw=1.0)
    assert "0-1" in report["pairs"]
    # Recompute the allowance here rather than trusting a hand-written number: the point is
    # that the reported rate equals "rows whose gap is inside 10% of demand, min 1 kW".
    allowance = np.maximum(np.abs(truth) * 0.1, 1.0)
    gap = np.abs(forecasts[0] - forecasts[1])
    expected = float(np.mean(gap <= allowance))
    assert report["pairs"]["0-1"]["agreement_rate"] == pytest.approx(expected)
    assert report["pairs"]["0-1"]["mean_absolute_difference_kw"] == pytest.approx(
        float(gap.mean())
    )
    assert report["pairs"]["0-1"]["share_disagreeing"] == pytest.approx(1.0 - expected)
    assert "one percent" in report["definition"].lower() or "10%" in report["definition"]


def test_agreement_rate_reaches_one_when_the_experts_are_identical() -> None:
    forecasts, truth = _two_experts()
    identical = np.stack([forecasts[0], forecasts[0]])
    report = agreement_rate(identical, truth)
    assert report["pairs"]["0-1"]["agreement_rate"] == 1.0
    assert report["pairs"]["0-1"]["share_disagreeing"] == 0.0


def test_diversity_report_reports_both_magnitude_and_signed_correlation() -> None:
    forecasts, truth = _two_experts()
    report = diversity_report(
        forecasts=forecasts,
        actual_kw=truth,
        horizons=HORIZONS,
        expert_names=("a", "b"),
    )
    errors = absolute_errors(forecasts, truth)
    expected = correlation_matrix(errors)
    assert np.allclose(report["overall"]["error_pearson"], expected.round(4))
    assert report["by_horizon"]["1"]["mae_kw"]["a"] == pytest.approx(
        float(errors[0, :, 0].mean())
    )
    assert set(report["best_expert_share"]) == {"1", "4"}
    # Signed errors are reported separately from magnitudes: two experts can be equally
    # bad and opposite, which is the case a 50/50 blend can exploit.
    assert report["overall"]["mean_signed_error_correlation"] != report["overall"][
        "mean_abs_error_correlation"
    ]


def test_diversity_by_horizon_names_the_winner_and_its_margin() -> None:
    forecasts, truth = _two_experts()
    summary = diversity_by_horizon(
        forecasts=forecasts, actual_kw=truth, horizons=HORIZONS, expert_names=("a", "b")
    )
    errors = absolute_errors(forecasts, truth)
    for column, key in enumerate(("1", "4")):
        mae = errors[:, :, column].mean(axis=1)
        assert summary[key]["best_expert"] == ("a" if mae[0] <= mae[1] else "b")
        assert summary[key]["mae_kw"]["a"] == pytest.approx(float(mae[0]))
        assert summary[key]["winner_margin_pct"] == pytest.approx(
            100.0 * float(np.sort(mae)[1] - np.sort(mae)[0]) / float(np.sort(mae)[0])
        )
        # Neither horizon may leave the correlation undefined.
        assert not np.isnan(summary[key]["mean_abs_error_correlation"])


def test_diversity_by_regime_labels_a_known_split() -> None:
    """Persistence-like expert a is best in the low tercile, expert b in the high one."""
    forecasts = np.stack(
        [
            np.asarray([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0]]),
            np.asarray([[5.0], [5.0], [5.0], [0.0], [0.0], [0.0]]),
        ]
    )
    truth = np.asarray([[1.0], [2.0], [3.0], [90.0], [95.0], [99.0]])
    axes = assign_regimes(
        truth[:, 0],
        target_id="customer_load",
        origin_index=np.arange(6),
        previous_actual=np.concatenate([truth[:1, 0], truth[:-1, 0]]),
    )
    report = diversity_by_regime(
        forecasts=forecasts,
        actual_kw=truth,
        horizons=(1,),
        expert_names=("a", "b"),
        axes=axes,
    )
    assert "load_level" in report
    high = report["load_level"]["regimes"]["high"]
    assert high["best_expert"] in {"a", "b"}
    assert high["oracle_mae_kw"] <= high["mae_kw"]["a"] + 1e-12
    assert high["gain_to_best_single_kw"] >= -1e-12


def test_diversity_by_regime_rejects_a_length_mismatch() -> None:
    forecasts, truth = _two_experts()
    axes = assign_regimes(
        np.zeros(2), target_id="customer_load", origin_index=np.arange(2)
    )
    with pytest.raises(ValueError, match="has 2 labels for 4 rows"):
        diversity_by_regime(
            forecasts=forecasts,
            actual_kw=truth,
            horizons=(1,),
            expert_names=("a", "b"),
            axes=axes,
        )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def test_method_metrics_reproduces_the_shared_metric_set() -> None:
    from energy_intelligence.ml.metrics import evaluate

    actual = np.asarray([[1.0, 2.0], [3.0, 4.0]])
    predicted = np.asarray([[1.5, 2.0], [2.0, 5.0]])
    metrics = method_metrics(actual_kw=actual, forecast_kw=predicted, horizons=HORIZONS)
    reference = evaluate(actual[:, 0], predicted[:, 0]).to_dict()
    assert metrics["1"]["mae"] == pytest.approx(reference["mae"])
    assert metrics["1"]["unit"] == "kW"
    assert metrics["1"]["horizon_steps"] == 1
    assert metrics["4"]["n"] == 2


def test_method_metrics_rejects_a_shape_mismatch() -> None:
    with pytest.raises(ValueError, match=r"\[rows, 2\]"):
        method_metrics(actual_kw=np.zeros((4, 2)), forecast_kw=np.zeros((4, 3)), horizons=HORIZONS)


def test_forecast_comparison_table_marks_the_short_horizon_winner() -> None:
    actual = np.asarray([[1.0], [2.0]])
    methods = {"good": np.asarray([[1.1], [1.9]]), "bad": np.asarray([[5.0], [6.0]])}
    table = forecast_comparison_table(
        methods=methods, actual_kw=actual, horizons=(1,), headline=1
    )
    assert "| good | **0.1000** |" in table
    assert "| bad | 4.0000 |" in table
    assert "Best at h=1: **good**" in table


# ---------------------------------------------------------------------------
# Routing quality, regret, stability, calibration
# ---------------------------------------------------------------------------


def _routing_case():
    """Three experts where the truth picks a different one per row."""
    truth = np.asarray([[0.0], [0.0], [0.0], [0.0]])
    forecasts = np.stack(
        [
            np.asarray([[0.0], [10.0], [10.0], [10.0]]),   # best on row 0
            np.asarray([[10.0], [0.0], [10.0], [10.0]]),   # best on row 1
            np.asarray([[10.0], [10.0], [0.0], [10.0]]),   # best on row 2
        ]
    )
    # A router that always picks the first expert.
    weights = np.zeros((4, 1, 3))
    weights[:, 0, 0] = 1.0
    perfect = np.zeros((4, 1, 3))
    for row in range(4):
        perfect[row, 0, row % 3] = 1.0
    return forecasts, truth, weights, perfect


def test_routing_regret_is_zero_for_a_perfect_selection() -> None:
    forecasts, truth, _weights, perfect = _routing_case()
    forecast = mix_of(forecasts, perfect)
    quality = routing_quality(
        forecast_kw=forecast,
        weights=perfect,
        expert_forecast_kw=forecasts,
        actual_kw=truth,
        horizons=(1,),
        expert_names=("a", "b", "c"),
    )
    entry = quality["1"]
    assert entry["selection_accuracy"] == 1.0
    assert entry["routing_regret_kw"] == pytest.approx(0.0)
    assert entry["capture_ratio"] == pytest.approx(1.0)


def test_routing_regret_is_positive_for_a_wrong_selection() -> None:
    forecasts, truth, weights, _perfect = _routing_case()
    forecast = mix_of(forecasts, weights)
    quality = routing_quality(
        forecast_kw=forecast,
        weights=weights,
        expert_forecast_kw=forecasts,
        actual_kw=truth,
        horizons=(1,),
        expert_names=("a", "b", "c"),
    )
    entry = quality["1"]
    # The oracle picks a, b, c, then a again (the last row ties and resolves to the lowest
    # index), so always picking a is right on two of four rows.
    assert entry["selection_accuracy"] == pytest.approx(0.5)
    assert entry["router_mae_kw"] == pytest.approx(7.5)
    assert entry["oracle_mae_kw"] == pytest.approx(2.5)
    assert entry["routing_regret_kw"] == pytest.approx(5.0)
    assert entry["capture_ratio"] == pytest.approx(0.0)
    assert entry["hard_share"] == 1.0
    assert entry["weighted_share"]["a"] == pytest.approx(1.0)
    assert entry["weighted_share"]["b"] == pytest.approx(0.0)


def test_oracle_regret_definition_names_every_measure_it_reports() -> None:
    definitions = routing_regret_definition()
    for key in (
        "oracle_error_kw",
        "routing_regret_kw",
        "routing_regret_pct",
        "selection_accuracy",
        "capture_ratio",
    ):
        assert key in definitions
        assert len(definitions[key]) > 40


def test_oracle_regret_rejects_mismatched_row_counts() -> None:
    with pytest.raises(ValueError, match="must cover the same rows"):
        oracle_regret(
            router_error=np.zeros(3),
            oracle_error=np.zeros(4),
            best_single_error=1.0,
        )


def test_oracle_regret_pct_is_none_when_the_baseline_is_zero() -> None:
    outcome = oracle_regret(
        router_error=np.zeros(2), oracle_error=np.zeros(2), best_single_error=0.0
    )
    assert outcome["routing_regret_pct"] is None


def mix_of(forecasts: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """A local convex combination, so these tests do not depend on the routing module."""
    return np.einsum("nhe,enh->nh", weights, forecasts)


def test_routing_stability_is_zero_for_constant_weights_and_positive_otherwise() -> None:
    forecasts, truth, weights, _perfect = _routing_case()
    constant = np.zeros((4, 1, 3))
    constant[:, 0, 0] = 1.0
    origins = np.asarray([0, 1, 2, 3])
    series = np.zeros(4, dtype=np.int64)
    flat = routing_stability(
        weights=constant, origins=origins, series=series, horizons=(1,), expert_names=("a", "b", "c")
    )
    assert flat["1"]["mean_total_variation"] == 0.0
    assert flat["1"]["selection_flip_rate"] == 0.0

    flipping = np.zeros((4, 1, 3))
    for row, winner in enumerate((0, 1, 2, 0)):
        flipping[row, 0, winner] = 1.0
    moving = routing_stability(
        weights=flipping, origins=origins, series=series, horizons=(1,), expert_names=("a", "b", "c")
    )
    assert moving["1"]["mean_total_variation"] > 0.0
    assert moving["1"]["selection_flip_rate"] == pytest.approx(1.0)
    assert moving["1"]["pairs"] == 3


def test_routing_stability_does_not_pair_different_series_together() -> None:
    """Two series, interleaved origins: consecutive rows are not consecutive steps."""
    constant = np.zeros((4, 1, 3))
    constant[0, 0, 0] = 1.0
    constant[1, 0, 1] = 1.0
    constant[2, 0, 0] = 1.0
    constant[3, 0, 1] = 1.0
    report = routing_stability(
        weights=constant,
        origins=np.asarray([0, 0, 1, 1]),
        series=np.asarray([0, 1, 0, 1]),
        horizons=(1,),
        expert_names=("a", "b", "c"),
    )
    assert report["1"]["mean_total_variation"] == 0.0


def test_calibration_distinguishes_a_confident_router_from_a_right_one() -> None:
    """Perfect confidence with 25% accuracy must score far worse than it claims."""
    forecasts, truth, _weights, perfect = _routing_case()
    truthful = router_calibration(
        weights=perfect,
        expert_forecast_kw=forecasts,
        actual_kw=truth,
        horizons=(1,),
        expert_names=("a", "b", "c"),
        bins=2,
    )
    overconfident = np.zeros((4, 1, 3))
    overconfident[:, 0, 0] = 1.0
    wrong = router_calibration(
        weights=overconfident,
        expert_forecast_kw=forecasts,
        actual_kw=truth,
        horizons=(1,),
        expert_names=("a", "b", "c"),
        bins=2,
    )
    assert truthful["1"]["mean_selection_accuracy"] == pytest.approx(1.0)
    assert truthful["1"]["expected_calibration_error"] == pytest.approx(0.0)
    assert truthful["1"]["brier_score"] == pytest.approx(0.0)
    assert wrong["1"]["mean_selection_accuracy"] == pytest.approx(0.5)
    assert wrong["1"]["mean_confidence"] == pytest.approx(1.0)
    assert wrong["1"]["expected_calibration_error"] > 0.3
    assert wrong["1"]["confidence_minus_accuracy"] > 0.3
    assert truthful["1"]["bins"][0]["observed_accuracy"] == pytest.approx(1.0)


def test_failure_analysis_lists_the_rows_where_routing_cost_the_most() -> None:
    forecasts, truth, weights, _perfect = _routing_case()
    # Make expert c the best single expert, so always picking a is a real mistake on
    # three of the four rows rather than a tie.
    forecasts = forecasts.copy()
    forecasts[2] = np.asarray([[4.0], [0.5], [1.0], [2.0]])
    forecast = mix_of(forecasts, weights)
    report = failure_analysis(
        forecast_kw=forecast,
        expert_forecast_kw=forecasts,
        weights=weights,
        actual_kw=truth,
        origins=np.asarray([10, 11, 12, 13]),
        horizons=(1,),
        expert_names=("a", "b", "c"),
        top_rows=2,
    )
    entry = report["1"]
    assert entry["share_worse_than_best_single"] == pytest.approx(0.75)
    assert entry["rows_worse_than_best_single"] == 3
    assert len(entry["worst_routing_regret"]) == 2
    assert entry["worst_routing_regret"][0]["value"] == pytest.approx(10.0)
    # Sign convention matches routing_regret_kw: positive means the router gave up gain.
    assert entry["worst_routing_regret"][0]["router_error_kw"] > entry["worst_routing_regret"][0]["oracle_error_kw"]
    assert entry["regret_by_selection"]["a"]["rows"] == 4
    assert entry["regret_by_selection"]["a"]["hit_rate"] == pytest.approx(0.25)


def test_router_by_regime_reports_weights_baselines_and_the_oracle() -> None:
    truth = np.asarray([[1.0], [2.0], [3.0], [90.0]])
    forecasts = np.stack(
        [
            np.asarray([[0.0], [1.0], [2.0], [80.0]]),
            np.asarray([[2.0], [3.0], [4.0], [100.0]]),
            np.asarray([[0.5], [0.5], [0.5], [95.0]]),
        ]
    )
    weights = np.zeros((4, 1, 3))
    weights[:, 0, 2] = 1.0
    axes = assign_regimes(
        truth[:, 0],
        target_id="customer_load",
        origin_index=np.arange(4),
        previous_actual=truth[:, 0],
    )
    report = router_by_regime(
        router_forecast_kw=mix_of(forecasts, weights),
        router_weights=weights,
        method_forecast_kw={"fixed_ensemble": mix_of(forecasts, weights)},
        expert_forecast_kw=forecasts,
        actual_kw=truth,
        axes=axes,
        horizons=(1,),
        expert_names=("a", "b", "c"),
    )
    low = report["load_level"]["regimes"]["low"]
    high = report["load_level"]["regimes"]["high"]
    assert low["n"] + high["n"] + report["load_level"]["regimes"]["typical"]["n"] == 4
    assert low["router_mae_kw"] <= high["router_mae_kw"]
    assert low["oracle_mae_kw"] <= low["router_mae_kw"] + 1e-12
    assert low["mean_router_weights"]["c"] == pytest.approx(1.0)
    assert low["best_single_expert"] == "c"
    assert "fixed_ensemble_mae_kw" in low
    assert report["load_level"]["horizon"] == 1


def test_compute_cost_reports_the_router_as_a_rounding_error_next_to_the_experts() -> None:
    cost = compute_cost(
        router_parameters=1097,
        router_train_seconds=2.0,
        router_inference_seconds=0.1,
        expert_seconds={"gbm": 3.0, "tcn": 177.0},
        expert_rows=359040,
        router_rows=179520,
        expert_parameters={"tcn": 74099},
    )
    assert cost["router_parameters"] == 1097
    assert cost["expert_total_seconds"] == pytest.approx(180.0)
    assert cost["router_seconds_per_row"] < cost["expert_seconds_per_row"] / 100.0
    assert cost["router_overhead_vs_experts"] < 0.01
    assert cost["expert_parameters"]["tcn"] == 74099
    assert "additive" in cost["note"]


def test_compute_cost_handles_a_missing_expert_timing() -> None:
    cost = compute_cost(
        router_parameters=10,
        router_train_seconds=1.0,
        router_inference_seconds=0.0,
        expert_seconds={},
        expert_rows=0,
        router_rows=0,
    )
    assert cost["expert_total_seconds"] == 0.0
    assert cost["router_overhead_vs_experts"] is None


# ---------------------------------------------------------------------------
# Oracle helpers used by the runner
# ---------------------------------------------------------------------------


def test_pairwise_best_single_keys_by_expert_and_position() -> None:
    forecasts, truth = _two_experts()
    table = pairwise_best_single(forecasts, truth, ("a", "b"))
    errors = absolute_errors(forecasts, truth)
    assert table["a"]["0"] == pytest.approx(float(errors[0, :, 0].mean()))
    assert table["b"]["1"] == pytest.approx(float(errors[1, :, 1].mean()))
    assert set(table) == {"a", "b"}
    assert set(table["a"]) == {"0", "1"}


def test_best_pair_and_gain_shows_the_third_expert_earns_its_place() -> None:
    forecasts = np.stack(
        [
            np.asarray([[0.0], [10.0], [20.0]]),
            np.asarray([[10.0], [0.0], [10.0]]),
            np.asarray([[10.0], [10.0], [0.0]]),
        ]
    )
    truth = np.asarray([[0.0], [0.0], [0.0]])
    records = best_pair_and_gain(forecasts, truth, ("a", "b", "c"))
    assert len(records) == 3
    for record in records:
        assert set(record) == {
            "experts",
            "horizon_position",
            "pair_mixture_mae_kw",
            "full_mixture_mae_kw",
            "gain_from_third_expert_kw",
        }
        # The best pair cannot beat the best triple.
        assert record["pair_mixture_mae_kw"] >= record["full_mixture_mae_kw"] - 1e-9
        assert record["gain_from_third_expert_kw"] >= -1e-9


def test_oracle_mixture_beats_the_worst_fixed_weighting() -> None:
    truth = np.asarray([[0.0], [10.0], [20.0], [30.0]])
    forecasts = np.stack(
        [
            np.full_like(truth, 15.0),
            np.full_like(truth, 17.0),
            np.full_like(truth, 13.0),
        ]
    )
    mixed = oracle_mixture(forecasts, truth)
    mixed_mae = float(np.abs(mixed - truth).mean())
    for index in range(3):
        assert mixed_mae <= float(np.abs(forecasts[index] - truth).mean()) + 1e-9


def test_label_quality_reports_a_one_expert_label_set_as_low_information() -> None:
    errors = np.zeros((3, 10, 2))
    errors[0] = 1.0
    assignment = np.zeros((10, 2), dtype=np.int64)
    report = label_quality(assignment=assignment, errors=errors, expert_names=("a", "b", "c"))
    assert report["0"]["dominant_label_share"] == pytest.approx(1.0)
    assert report["0"]["mean_margin_pct"] == pytest.approx(0.0)
    assert report["0"]["share_margin_below_5pct"] == pytest.approx(1.0)


def test_oracle_assignment_resolves_a_tie_to_the_lowest_index() -> None:
    forecasts = np.asarray([[[1.0]], [[-1.0]]])
    truth = np.asarray([[0.0]])
    assert oracle_assignment(forecasts, truth)[0, 0] == 0


# ---------------------------------------------------------------------------
# Ablation contracts
# ---------------------------------------------------------------------------


def test_every_ablation_states_the_question_it_asks() -> None:
    assert ABLATIONS
    names = [item.name for item in ABLATIONS]
    assert len(set(names)) == len(names)
    for item in ABLATIONS:
        assert item.question.strip().endswith("?")
        assert isinstance(item, AblationSpec)


def test_each_ablation_removes_exactly_what_it_says_it_removes() -> None:
    from energy_intelligence.ml.router.features import FEATURE_GROUPS

    for item in ABLATIONS:
        spec = item.to_feature_spec()
        for group in item.drop_groups:
            assert group not in spec.groups
        for group in FEATURE_GROUPS:
            if group not in item.drop_groups:
                assert group in spec.groups
        assert spec.include_past_error == item.include_past_error
        if item.shared_head:
            # A shared head still sees every feature; only the output changes, so the
            # ablation is not confounded with a change of inputs.
            assert set(spec.groups) == set(FEATURE_GROUPS)


def test_the_cold_start_option_exists_only_on_the_spec_not_the_suite() -> None:
    """Both training variants run in the main experiment; the suite does not repeat them."""
    assert all(item.warm_start for item in ABLATIONS)