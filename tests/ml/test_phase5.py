"""Tests for the Phase 5 runner, its configuration and its comparison logic.

The comparison against Phase 4 is the part most able to mislead: it must be read
from the Phase 4 artifact rather than retyped, the horizons must line up, and a
missing reference must fail loudly instead of silently dropping a baseline.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from conftest import build_series

from energy_intelligence.config.temporal import (
    TemporalExperimentConfig,
    load_temporal_config,
)
from energy_intelligence.config.schema import ConfigError, ConfigValidationError
from energy_intelligence.ml.phase5 import (
    _to_kw,
    analyse_predictions,
    load_phase4_reference,
    replay_evaluation,
    run_phase5_experiment,
)
from energy_intelligence.ml.ablation import render_ablation_table
from energy_intelligence.ml.registry import read_records
from energy_intelligence.paths import ProjectPaths
from energy_intelligence.ml.training import TrainingConfig


@pytest.fixture
def tiny_config() -> TemporalExperimentConfig:
    return TemporalExperimentConfig(
        target="customer_load",
        label="test-main",
        customer_count=2,
        commercial_fraction=0.25,
        seed=20260101,
        torch_threads=2,
        lookback_steps=96,
        token_stride=4,
        train_origin_stride=64,
        train_fraction=0.70,
        validation_fraction=0.15,
        horizons=(1, 4),
        architecture="tcn",
        cyclic_channels=True,
        delta_target=True,
        model_params={"channels": 8, "dilations": [1, 2], "hidden_size": 16,
                      "series_embedding_dim": 4},
        epochs=1,
        batch_size=256,
        learning_rate=0.003,
        weight_decay=0.0,
        loss="l1",
        huber_delta=0.05,
        patience=2,
        min_delta=1e-5,
        max_seconds=600.0,
        grad_clip=1.0,
        validation_stride=1,
        ablation_budget_epochs=1,
        ablation_budget_stride=64,
        ablation_validation_stride=4,
    )


@pytest.fixture
def phase4_reference(tmp_path: Path) -> dict:
    return {
        "naive_last_value": {1: {"mae": 0.4043}, 4: {"mae": 0.8897}, 96: {"mae": 2.4343}},
        "classical_hist_gbm": {1: {"mae": 0.4242}, 4: {"mae": 0.8579}, 96: {"mae": 1.5648}},
    }


# ======================================================================
# Unit conversion
# ======================================================================


def test_to_kw_handles_one_and_two_dimensional_input() -> None:
    """A 1-D column must not be broadcast against the scale twice."""
    values = np.array([0.5, 0.25])
    scale = np.array([10.0, 4.0])
    assert _to_kw(values, scale).tolist() == [5.0, 1.0]
    block = np.array([[0.5, 0.25], [1.0, 2.0]])
    out = _to_kw(block, scale)
    assert out.shape == block.shape
    # Each ROW carries its own scale: row 0 x10, row 1 x4.
    assert out.tolist() == [[5.0, 2.5], [4.0, 8.0]]


def test_to_kw_rejects_unexpected_rank() -> None:
    with pytest.raises(ValueError, match="1-D or 2-D"):
        _to_kw(np.zeros((2, 2, 2)), np.ones(2))


# ======================================================================
# Phase 4 reference
# ======================================================================


def test_phase4_reference_is_read_from_the_artifact(tmp_path: Path) -> None:
    payload = {
        "models": [
            {"model": "classical_hist_gbm", "metrics_by_horizon": {"1": {"mae": 0.4242}}},
            {"model": "naive_last_value", "metrics_by_horizon": {"1": {"mae": 0.4043}}},
        ]
    }
    path = tmp_path / "comparison.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    reference = load_phase4_reference(path)
    assert reference["classical_hist_gbm"][1]["mae"] == 0.4242


def test_missing_phase4_reference_fails_loudly(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Run the Phase 4 load experiment"):
        load_phase4_reference(tmp_path / "absent.json")


# ======================================================================
# Configuration
# ======================================================================


def test_shipped_temporal_config_is_valid() -> None:
    config = load_temporal_config()
    assert config.target == "customer_load"
    assert config.architecture in {"tcn", "transformer"}
    assert config.horizons == (1, 4, 96)
    assert config.token_stride < config.lookback_steps


def test_shipped_config_declares_a_question_for_every_ablation() -> None:
    config = load_temporal_config()
    assert config.ablations, "at least one ablation is expected"
    for ablation in config.ablations:
        assert ablation["name"] and ablation["question"]
        assert ablation["overrides"]


def test_config_rejects_a_non_demand_target(tmp_path: Path, tiny_config) -> None:
    body = _config_toml(tiny_config, target="battery_soc")
    path = tmp_path / "temporal.toml"
    path.write_text(body, encoding="utf-8")
    with pytest.raises(ConfigValidationError, match="demand only"):
        load_temporal_config(path)


def test_config_rejects_an_unknown_architecture(tmp_path: Path, tiny_config) -> None:
    path = tmp_path / "temporal.toml"
    path.write_text(_config_toml(tiny_config, architecture="gpt"), encoding="utf-8")
    with pytest.raises(ConfigValidationError, match="architecture must be one of"):
        load_temporal_config(path)


def test_config_rejects_a_stride_longer_than_the_window(tmp_path: Path, tiny_config) -> None:
    path = tmp_path / "temporal.toml"
    path.write_text(
        _config_toml(tiny_config, lookback_steps=96, token_stride=96), encoding="utf-8"
    )
    with pytest.raises(ConfigValidationError, match="token_stride must be shorter"):
        load_temporal_config(path)


def test_config_rejects_unknown_keys(tmp_path: Path, tiny_config) -> None:
    """An unknown TOP-LEVEL key is refused.

    It is inserted before the first table header on purpose: in TOML a bare key after
    a header belongs to that table, so appending it at the end would test nothing.
    """
    body = _config_toml(tiny_config)
    path = tmp_path / "temporal.toml"
    path.write_text("mystery = 3\n" + body, encoding="utf-8")
    with pytest.raises(ConfigValidationError, match="unknown keys"):
        load_temporal_config(path)


def test_config_rejects_a_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="temporal config not found"):
        load_temporal_config(tmp_path / "absent.toml")


def test_config_rejects_an_ablation_without_a_question(tmp_path: Path, tiny_config) -> None:
    path = tmp_path / "temporal.toml"
    path.write_text(
        _config_toml(tiny_config) + '\n[[ablations]]\nname = "x"\noverrides = { }\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="question"):
        load_temporal_config(path)


def test_config_rejects_unknown_ablation_overrides(tmp_path: Path, tiny_config) -> None:
    path = tmp_path / "temporal.toml"
    path.write_text(
        _config_toml(tiny_config)
        + '\n[[ablations]]\nname = "x"\nquestion = "q"\noverrides = { epochs = 99 }\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unknown overrides"):
        load_temporal_config(path)


def _config_toml(config: TemporalExperimentConfig, **overrides) -> str:
    payload = config.to_dict()
    payload.update(overrides)
    model_params = payload.pop("model_params")
    ablations = payload.pop("ablations", [])
    lines = [f"{key} = {json.dumps(value) if isinstance(value, (list, dict, str, bool)) else value}"
             for key, value in payload.items()]
    lines.append("[model_params]")
    lines.extend(
        f"{key} = {json.dumps(value) if isinstance(value, (list, str, bool, float, int)) else value}"
        for key, value in model_params.items()
    )
    for ablation in ablations:
        lines.append("[[ablations]]")
        lines.append(f"name = {json.dumps(ablation['name'])}")
        lines.append(f"question = {json.dumps(ablation['question'])}")
        lines.append(f"overrides = {json.dumps(ablation['overrides'])}")
    return "\n".join(lines) + "\n"


# ======================================================================
# End to end
# ======================================================================



def build_index(series, config: TemporalExperimentConfig):
    """Build the same index the runner would build for ``series``."""
    from energy_intelligence.ml.sequence import build_sequence_index

    sequence_config = config.sequence_config()
    return build_sequence_index(
        series_count=len(series.series_ids),
        total_steps=series.values.shape[1],
        config=sequence_config,
        target_id=series.spec.target_id,
        value_origin=series.value_origin,
        series_ids=series.series_ids,
        source="synthetic",
    )


# ======================================================================
# Regime analysis: each horizon must be judged on its OWN errors
# ======================================================================


def _analysis_inputs(series, index, horizons, offset: float = 8.0):
    """Predictions that are wrong by a horizon-dependent amount.

    The short-horizon column is exact and every longer column is off by a fixed
    per-unit offset, which makes any horizon mix-up visible in the regime table
    instead of subtle.
    """
    evaluated_rows = index.rows(2)
    position = {h: i for i, h in enumerate(horizons)}
    rows = index.series[evaluated_rows]
    # Paired fancy indexing: values[rows, origins] gathers one value per row.
    # values[rows][:, origins] would be an outer product, not what is wanted.
    at_origin = np.asarray(
        series.values[rows, index.origins[evaluated_rows]], dtype=np.float32
    )
    actual = np.repeat(at_origin[:, None], len(horizons), axis=1)
    predicted = actual.copy()
    for horizon in horizons[1:]:
        predicted[:, position[horizon]] += np.float32(offset)
    persistence = actual.copy()
    return evaluated_rows, actual, predicted, persistence


def _scale_bounds(series, index, evaluated_rows):
    scales = series.scale_vector[index.series[evaluated_rows]]
    return float(scales.min()), float(scales.max())


def test_each_horizon_is_analysed_against_its_own_errors(
    tiny_config, phase4_reference, tmp_path: Path
) -> None:
    """Regression test: the regime table once labelled h=1 with h=96 errors.

    Because every horizon's error array was written inside the metrics loop, the
    analysis reported per-regime MAEs roughly four times the overall figure while the
    headline metrics looked perfectly healthy. Nothing about the headline numbers
    could have caught it, so the invariant is asserted directly.
    """
    series = build_series(series_count=2, steps=1400)
    index = build_index(series, tiny_config)
    evaluated_rows, actual, predicted, persistence = _analysis_inputs(
        series, index, tiny_config.horizons
    )

    by_horizon, headline, _ = analyse_predictions(
        series=series,
        index=index,
        values=series.normalised().astype(np.float32),
        evaluated_rows=evaluated_rows,
        actual_pu=actual,
        predicted_pu=predicted,
        persistence_pu=persistence,
        row_scale=series.scale_vector[index.series[evaluated_rows]],
    )

    short = tiny_config.horizons[0]
    assert headline["horizon"] == short
    for axis in ("load_level", "load_ramp"):
        for entry in headline[axis]["regimes"].values():
            # Both models are exact at the shortest horizon here, so the headline
            # regime table must read 0.0 - proving it was built from h=1 errors and
            # not left holding the last horizon's error array.
            assert entry["persistence"] == pytest.approx(0.0, abs=1e-9)
            assert entry["phase5"] == pytest.approx(0.0, abs=1e-6)

    # The short horizon was predicted exactly, at every regime and on both axes.
    low, high = _scale_bounds(series, index, evaluated_rows)
    for axis in ("load_level", "load_ramp"):
        for entry in by_horizon[short][axis]["regimes"].values():
            assert entry["phase5"] == pytest.approx(0.0, abs=1e-6)

    # Every longer horizon carries the same 8 per-unit offset, so each regime's MAE
    # must sit between the offset scaled by the smallest and the largest row scale.
    # A tercile need not hold both series evenly, so the bounds are the invariant -
    # not a single expected value.
    for horizon, report in by_horizon.items():
        if horizon == short:
            continue
        for axis in ("load_level", "load_ramp"):
            for entry in report[axis]["regimes"].values():
                assert entry["phase5"] > low * 8.0 - 1e-6
                assert entry["phase5"] <= high * 8.0 + 1e-6


def test_regime_by_horizon_records_every_horizon_separately(
    tiny_config, phase4_reference, tmp_path: Path
) -> None:
    series = build_series(series_count=2, steps=1400)
    index = build_index(series, tiny_config)
    evaluated_rows, actual, predicted, persistence = _analysis_inputs(
        series, index, tiny_config.horizons
    )
    by_horizon, _, _ = analyse_predictions(
        series=series,
        index=index,
        values=series.normalised().astype(np.float32),
        evaluated_rows=evaluated_rows,
        actual_pu=actual,
        predicted_pu=predicted,
        persistence_pu=persistence,
        row_scale=series.scale_vector[index.series[evaluated_rows]],
    )
    assert set(by_horizon) == set(tiny_config.horizons)
    short = tiny_config.horizons[0]
    for horizon, report in by_horizon.items():
        errors = [e["phase5"] for e in report["load_level"]["regimes"].values()]
        if horizon == short:
            assert all(v == pytest.approx(0.0, abs=1e-6) for v in errors)
        else:
            assert all(v > 0.0 for v in errors)

def test_end_to_end_run_produces_every_artifact(
    tmp_path: Path, tiny_config, phase4_reference
) -> None:
    from energy_intelligence.paths import ProjectPaths

    paths = ProjectPaths.from_root(tmp_path)
    paths.ensure()
    series = build_series(series_count=2, steps=20000)

    result = run_phase5_experiment(
        series,
        sequence_config=tiny_config.sequence_config(),
        training_config=tiny_config.training_config(epochs=1),
        paths=paths,
        architecture="tcn",
        experiment_id="phase5-test",
        phase4_reference=phase4_reference,
        label="test-main",
        model_params=tiny_config.model_params,
    )

    for name in ("checkpoint.pt", "dataset_manifest.json", "training_log.jsonl",
                 "regimes.json", "error_analysis.json", "result.json"):
        assert (paths.artifacts / "phase5" / "test-main" / name).is_file(), name

    assert set(result.metrics_by_horizon) == set(tiny_config.horizons)
    assert set(result.persistence_by_horizon) == set(tiny_config.horizons)
    assert result.trained.model.parameter_count() > 0
    assert result.error_distribution["phase5"]["n"] > 0
    assert result.error_distribution["phase5"]["p99"] >= result.error_distribution["phase5"]["median"]
    assert "max_regime_mae_ratio" in result.regimes

    records = read_records(paths.experiments / "registry.jsonl")
    mine = [r for r in records if r.experiment_id == "phase5-test"]
    assert len(mine) == 1
    record = mine[0]
    assert record.model == "phase5_tcn"
    assert record.model_family == "temporal"
    assert record.hyperparameters["parameter_count"] == result.trained.model.parameter_count()
    assert "energy demand dynamics model" in " ".join(record.notes).lower()
    assert record.environment["torch"]


def test_improvement_is_reported_against_persistence_and_gbm(
    tmp_path: Path, tiny_config, phase4_reference
) -> None:
    from energy_intelligence.paths import ProjectPaths

    paths = ProjectPaths.from_root(tmp_path)
    paths.ensure()
    result = run_phase5_experiment(
        build_series(series_count=2, steps=20000),
        sequence_config=tiny_config.sequence_config(),
        training_config=tiny_config.training_config(epochs=1),
        paths=paths, architecture="tcn", experiment_id="phase5-test",
        phase4_reference=phase4_reference, label="test-imp",
        model_params=tiny_config.model_params,
    )
    for horizon in tiny_config.horizons:
        entry = result.improvement[horizon]
        assert entry["phase4_gbm_mae_kw"] == phase4_reference["classical_hist_gbm"][horizon]["mae"]
        assert "vs_persistence_pct" in entry and "vs_gbm_pct" in entry
    # The rendered table must name the baseline it is compared against.
    rendered = result.render()
    assert "phase4 GBM" in rendered and "persistence" in rendered


def test_single_horizon_config_runs_and_evaluates_one_column(
    tmp_path: Path, tiny_config, phase4_reference
) -> None:
    from energy_intelligence.paths import ProjectPaths

    paths = ProjectPaths.from_root(tmp_path)
    paths.ensure()
    result = run_phase5_experiment(
        build_series(series_count=2, steps=20000),
        sequence_config=tiny_config.sequence_config(horizons=(1,)),
        training_config=tiny_config.training_config(epochs=1),
        paths=paths, architecture="tcn", experiment_id="phase5-single",
        phase4_reference=phase4_reference, label="test-single",
        model_params=tiny_config.model_params,
    )
    assert list(result.metrics_by_horizon) == [1]
    assert result.metrics_by_horizon[1]["ramp_error_mae"] is None


def test_runner_refuses_an_empty_test_split(tiny_config, phase4_reference, tmp_path: Path) -> None:
    from energy_intelligence.paths import ProjectPaths

    paths = ProjectPaths.from_root(tmp_path)
    paths.ensure()
    config = tiny_config.sequence_config(lookback_steps=96)
    # 300 steps: the test range is 75 steps wide, which cannot hold a 96-step window
    # plus a 4-step horizon, so the test split is empty by construction.
    tiny = build_series(series_count=1, steps=300)
    with pytest.raises(ValueError, match="test split is empty"):
        run_phase5_experiment(
            tiny,
            sequence_config=config,
            training_config=TrainingConfig(epochs=1, threads=2),
            paths=paths, architecture="tcn", experiment_id="phase5-empty",
            phase4_reference=phase4_reference, label="test-empty",
        )


def test_ablation_table_renders_every_question() -> None:
    from energy_intelligence.ml.ablation import AblationOutcome

    outcome = AblationOutcome(
        name="lookback_96",
        question="Does one day suffice?",
        label="ablation-lookback_96",
        metrics={1: {"mae": 0.4}, 4: {"mae": 0.9}},
        best_epoch=2,
        best_validation_mae=0.03,
        train_seconds=120.0,
        parameters=74099,
        epochs_run=2,
        reference_name="ablation_reference",
        reference_metrics={1: {"mae": 0.42}, 4: {"mae": 0.95}},
        reference_label="ablation_reference",
        delta_vs_reference={1: -4.8, 4: -5.3},
    )
    table = render_ablation_table([outcome])
    assert "lookback_96" in table
    assert "Does one day suffice?" in table
    assert "-4.8%" in table
    assert "valid" in table


def test_phase5_records_do_not_collide_with_phase4(tmp_path: Path, tiny_config, phase4_reference) -> None:
    from energy_intelligence.paths import ProjectPaths

    paths = ProjectPaths.from_root(tmp_path)
    paths.ensure()
    run_phase5_experiment(
        build_series(series_count=2, steps=20000),
        sequence_config=tiny_config.sequence_config(),
        training_config=tiny_config.training_config(epochs=1),
        paths=paths, architecture="tcn", experiment_id="phase5-uniq",
        phase4_reference=phase4_reference, label="test-uniq",
        model_params=tiny_config.model_params,
    )
    records = read_records(paths.experiments / "registry.jsonl")
    assert len(records) == 1
    assert records[0].model_family == "temporal"
    assert records[0].dataset_version.startswith("sq-")



def write_phase4_artifact(root: Path, phase4_reference: dict) -> None:
    """Write comparison.json in the shape the Phase 4 runner actually produced.

    The simple ``phase4_reference`` fixture is the *parsed* form (model name ->
    horizon -> metrics). ``load_phase4_reference`` expects the raw artifact, which
    nests the same content under a list of per-model records.
    """
    reports = root / "artifacts" / "ml" / "load-40" / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "comparison.json").write_text(
        json.dumps(
            {
                "models": [
                    {
                        "model": name,
                        "metrics_by_horizon": {
                            str(horizon): metrics for horizon, metrics in horizons.items()
                        },
                    }
                    for name, horizons in phase4_reference.items()
                ]
            }
        ),
        encoding="utf-8",
    )

# ======================================================================
# Checkpoint replay
# ======================================================================


def test_replay_reproduces_the_recorded_metrics_and_refreshes_the_analysis(
    tiny_config, phase4_reference, tmp_path: Path
) -> None:
    """A replay must be a repair tool, not a way to publish new numbers.

    ``replay_evaluation`` re-derives the test analysis from a saved checkpoint and
    raises if the recomputed MAE differs from ``result.json``. Here the training run
    writes that file itself, so agreement is guaranteed - what is being tested is
    that the artifacts are regenerated from the checkpoint and that the guard does
    not fire on a faithful replay.
    """
    paths = ProjectPaths.from_root(tmp_path)
    write_phase4_artifact(tmp_path, phase4_reference)
    phase4_dir = paths.artifacts / "ml" / "load-40" / "reports"
    (phase4_dir / "regimes.json").write_text(
        json.dumps(
            {
                "load_level": {"regimes": {"low": {"classical_hist_gbm": {"mae": 0.08}}}},
                "load_ramp": {"regimes": {"low": {"classical_hist_gbm": {"mae": 0.07}}}},
            }
        ),
        encoding="utf-8",
    )

    series = build_series(series_count=2, steps=1400)
    written = run_phase5_experiment(
        series,
        sequence_config=tiny_config.sequence_config(),
        training_config=TrainingConfig(
            epochs=1, batch_size=256, learning_rate=0.003, max_seconds=600.0
        ),
        paths=paths,
        architecture="tcn",
        experiment_id="phase5-tcn-test",
        phase4_reference=phase4_reference,
        label=tiny_config.label,
        validation_stride=4,
        model_params=tiny_config.model_params,
    )
    root = paths.artifacts / "phase5" / tiny_config.label
    recorded = json.loads((root / "result.json").read_text(encoding="utf-8"))

    (root / "regimes.json").unlink()
    (root / "error_analysis.json").unlink()

    replayed = replay_evaluation(
        series,
        sequence_config=tiny_config.sequence_config(),
        paths=paths,
        label=tiny_config.label,
        architecture="tcn",
        experiment_id="phase5-tcn-test",
        model_params=tiny_config.model_params,
    )

    for horizon, metrics in recorded["metrics_by_horizon"].items():
        assert replayed["metrics_by_horizon"][horizon]["mae"] == pytest.approx(
            metrics["mae"], abs=1e-9
        )
    assert (root / "regimes.json").is_file()
    assert (root / "error_analysis.json").is_file()
    # Every horizon gets its own regime table, and the headline names its horizon.
    assert set(replayed["regimes_by_horizon"]) == {str(h) for h in tiny_config.horizons}
    assert replayed["regimes_headline"]["horizon"] == tiny_config.horizons[0]
    # Phase 4's regime artifact is carried through for side-by-side comparison.
    assert "classical_hist_gbm" in replayed["regimes_headline"]["phase4_reference_regimes"]["load_level"]["regimes"]["low"]
    assert written.metrics_by_horizon[tiny_config.horizons[0]]["mae"] == pytest.approx(
        replayed["metrics_by_horizon"][str(tiny_config.horizons[0])]["mae"], abs=1e-9
    )


def test_replay_refuses_a_checkpoint_whose_metrics_no_longer_hold(
    tiny_config, phase4_reference, tmp_path: Path
) -> None:
    """If the checkpoint stops reproducing its own numbers, the replay must fail.

    Without this the replay path would quietly publish a *different* result under the
    original run's name, which is the exact failure mode it exists to prevent.
    """
    paths = ProjectPaths.from_root(tmp_path)
    write_phase4_artifact(tmp_path, phase4_reference)

    series = build_series(series_count=2, steps=1400)
    run_phase5_experiment(
        series,
        sequence_config=tiny_config.sequence_config(),
        training_config=TrainingConfig(
            epochs=1, batch_size=256, learning_rate=0.003, max_seconds=600.0
        ),
        paths=paths,
        architecture="tcn",
        experiment_id="phase5-tcn-test",
        phase4_reference=phase4_reference,
        label=tiny_config.label,
        validation_stride=4,
        model_params=tiny_config.model_params,
    )
    root = paths.artifacts / "phase5" / tiny_config.label
    result_path = root / "result.json"
    tampered = json.loads(result_path.read_text(encoding="utf-8"))
    tampered["metrics_by_horizon"][str(tiny_config.horizons[0])]["mae"] += 0.5
    result_path.write_text(json.dumps(tampered), encoding="utf-8")

    with pytest.raises(AssertionError, match="does not reproduce"):
        replay_evaluation(
            series,
            sequence_config=tiny_config.sequence_config(),
            paths=paths,
            label=tiny_config.label,
            architecture="tcn",
            experiment_id="phase5-tcn-test",
            model_params=tiny_config.model_params,
        )


def test_replay_fails_loudly_without_a_checkpoint(
    tiny_config, phase4_reference, tmp_path: Path
) -> None:
    paths = ProjectPaths.from_root(tmp_path)
    with pytest.raises(FileNotFoundError, match="no checkpoint"):
        replay_evaluation(
            build_series(series_count=2, steps=1400),
            sequence_config=tiny_config.sequence_config(),
            paths=paths,
            label="absent",
            architecture="tcn",
            experiment_id="phase5-tcn-absent",
        )


# ======================================================================
# The ablation suite
# ======================================================================


def test_ablation_suite_runs_and_scores_against_its_own_reference(
    tiny_config, phase4_reference, tmp_path: Path, monkeypatch
) -> None:
    """Every ablation must be compared with a reference trained at the SAME budget.

    The reference is retrained at the ablation budget rather than reusing the main
    run's numbers, because the main run has a larger budget and comparing across
    budgets would make every ablation look bad for the wrong reason. The delta is
    also required to point the right way: a *removal* ablation that improves must
    register as a negative percentage.
    """
    from energy_intelligence.ml import ablation as ablation_module

    paths = ProjectPaths.from_root(tmp_path)
    series = build_series(series_count=2, steps=1400)
    budgets: list[tuple[str, int, int]] = []

    def fake_run(series_arg, *, sequence_config, training_config, experiment_id, **kwargs):
        budgets.append(
            (experiment_id, training_config.epochs, sequence_config.train_origin_stride)
        )
        return run_phase5_experiment(
            series_arg,
            sequence_config=sequence_config,
            training_config=training_config,
            paths=paths,
            architecture=kwargs.get("architecture", "tcn"),
            experiment_id=experiment_id,
            phase4_reference=phase4_reference,
            label=kwargs.get("label", "x"),
            validation_stride=kwargs.get("validation_stride", 1),
            model_params=kwargs.get("model_params"),
        )

    monkeypatch.setattr(ablation_module, "run_phase5_experiment", fake_run)

    suite = replace(
        tiny_config,
        ablations=[
            {
                "name": "no_cyclic",
                "question": "Do the temporal channels earn their place?",
                "overrides": {"cyclic_channels": False},
            },
            {
                "name": "lookback_96",
                "question": "Does one day carry the signal?",
                "overrides": {"lookback_steps": 96},
            },
        ],
    )
    outcomes = ablation_module.run_ablations(
        series, config=suite, paths=paths, phase4_reference=phase4_reference
    )

    assert {o.name for o in outcomes} == {"no_cyclic", "lookback_96"}
    for outcome in outcomes:
        assert outcome.question
        assert outcome.parameters > 0
        assert outcome.reference_name == "ablation_reference"
        for horizon, delta in outcome.delta_vs_reference.items():
            assert isinstance(delta, float)
            assert horizon in outcome.reference_metrics

    # The reference and both ablations all ran at the reduced budget and stride.
    assert len(budgets) == 3
    assert {epochs for _, epochs, _ in budgets} == {suite.ablation_budget_epochs}
    assert {stride for _, _, stride in budgets} == {suite.ablation_budget_stride}

    table = ablation_module.render_ablation_table(outcomes)
    assert "no_cyclic" in table and "lookback_96" in table
    assert suite.ablations[0]["question"] in table

    written = json.loads(
        (paths.artifacts / "phase5" / "ablations.json").read_text(encoding="utf-8")
    )
    assert written["budget_epochs"] == suite.ablation_budget_epochs
    assert written["reference"]["metrics"]
    assert len(written["ablations"]) == 2
    # The suite must state that its budget is not comparable with the main run.
    assert "not against the main run" in written["note"]
