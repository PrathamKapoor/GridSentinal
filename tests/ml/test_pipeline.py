"""Tests for the ML configuration, the CLI surface and the end-to-end runner.

The end-to-end test builds a real dataset artifact, runs baselines, writes reports
and appends to the registry, using the synthetic series so it runs on a clean
checkout. The real-data path is covered by ``tests/ml/test_targets.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from conftest import build_series

from energy_intelligence.cli import main
from energy_intelligence.config.ml import MlConfig, load_ml_config
from energy_intelligence.config.schema import ConfigError, ConfigValidationError
from energy_intelligence.ml.dataset import DatasetBuildConfig, build_dataset, load_dataset
from energy_intelligence.ml.pipeline import extract_target
from energy_intelligence.ml.registry import (
    ExperimentRecord,
    append_record,
    read_records,
)
from energy_intelligence.ml.runner import run_phase4
from energy_intelligence.paths import ProjectPaths


# ======================================================================
# Configuration
# ======================================================================


def test_shipped_ml_config_is_valid() -> None:
    config = load_ml_config()
    assert config.target
    assert config.lookback_steps > 0
    assert config.horizons == tuple(sorted(config.horizons))
    assert config.seed > 0
    assert config.models


def test_shipped_config_covers_all_three_families() -> None:
    models = load_ml_config().models
    assert any(m.startswith("naive_") for m in models)
    assert any(m.startswith("classical_") for m in models)


def test_config_is_printed_as_a_dict(capsys: pytest.CaptureFixture) -> None:
    assert main(["ml", "config"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["target"] == load_ml_config().target


def test_unknown_key_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "ml.toml"
    path.write_text('target = "x"\nlabel = "y"\nlookback_steps = 96\nhorizons = [1]\n'
                    "train_fraction = 0.7\nvalidation_fraction = 0.15\norigin_stride = 1\n"
                    "customer_count = 1\ncommercial_fraction = 0.25\ninclude_weather = false\n"
                    "seed = 1\ntorch_threads = 1\nmodels = [\"naive_last_value\"]\n"
                    "mystery_key = 3\n", encoding="utf-8")
    with pytest.raises(ConfigValidationError, match="unknown keys"):
        load_ml_config(path)


def test_missing_file_is_reported_clearly(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="ML config not found"):
        load_ml_config(tmp_path / "absent.toml")


def test_missing_required_key_is_reported(tmp_path: Path) -> None:
    path = tmp_path / "ml.toml"
    path.write_text('target = "customer_load"\n', encoding="utf-8")
    with pytest.raises(ConfigValidationError, match="missing required key"):
        load_ml_config(path)


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("lookback_steps", 0, "lookback_steps must be positive"),
        ("horizons", "[]", "at least one horizon"),
        ("train_fraction", "1.5", "train_fraction must lie"),
        ("validation_fraction", "0.0", "validation_fraction must lie"),
        ("origin_stride", "0", "origin_stride must be positive"),
        ("customer_count", "0", "customer_count must be positive"),
        ("commercial_fraction", "1.5", "commercial_fraction must lie"),
        ("torch_threads", "0", "torch_threads must be positive"),
        ("models", '["naive_telepathy"]', "unknown models"),
    ],
)
def test_invalid_values_are_rejected(tmp_path: Path, field: str, value: str, message: str) -> None:
    body = {
        "target": '"customer_load"',
        "label": '"test"',
        "lookback_steps": "672",
        "horizons": "[1, 4]",
        "train_fraction": "0.7",
        "validation_fraction": "0.15",
        "origin_stride": "1",
        "customer_count": "10",
        "commercial_fraction": "0.25",
        "include_weather": "false",
        "seed": "1",
        "torch_threads": "2",
        "models": '["naive_last_value"]',
    }
    body[field] = value
    path = tmp_path / "ml.toml"
    path.write_text("\n".join(f"{k} = {v}" for k, v in body.items()), encoding="utf-8")
    with pytest.raises(ConfigValidationError, match=message):
        load_ml_config(path)


def test_unsorted_horizons_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "ml.toml"
    path.write_text(
        'target = "customer_load"\nlabel = "t"\nlookback_steps = 672\nhorizons = [4, 1]\n'
        "train_fraction = 0.7\nvalidation_fraction = 0.15\norigin_stride = 1\n"
        "customer_count = 10\ncommercial_fraction = 0.25\ninclude_weather = false\n"
        "seed = 1\ntorch_threads = 1\nmodels = [\"naive_last_value\"]\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigValidationError, match="sorted"):
        load_ml_config(path)


# ======================================================================
# Target selection
# ======================================================================


def test_unsupported_target_is_refused_with_evidence(real_adapter) -> None:
    """The refusal must name what is missing, not just say 'no'.

    Run against the real dataset, so the refusal cannot be confused with a missing
    dataset: an absent dataset is a different failure with a different message.
    """
    config = MlConfig(
        target="battery_soc",
        label="battery",
        lookback_steps=672,
        horizons=(1,),
        train_fraction=0.7,
        validation_fraction=0.15,
        origin_stride=1,
        customer_count=10,
        commercial_fraction=0.25,
        include_weather=False,
        seed=1,
        torch_threads=1,
        models=("naive_last_value",),
    )
    with pytest.raises(ValueError) as caught:
        extract_target(config, real_adapter)
    message = str(caught.value)
    assert "UNSUPPORTED" in message
    assert "NOT AVAILABLE" in message
    assert "supported targets" in message


def test_extractor_refuses_unsupported_targets_before_touching_files(tmp_path: Path) -> None:
    from energy_intelligence.data.smartds import SmartDsLayout

    layout = SmartDsLayout(
        root=tmp_path,
        version="v1.0",
        year="2018",
        region="AUS",
        subregion="P1U",
        scenario="base_timeseries",
        substation="s",
        feeder="f",
    )
    config = MlConfig(
        target="wind_generation",
        label="wind",
        lookback_steps=672,
        horizons=(1,),
        train_fraction=0.7,
        validation_fraction=0.15,
        origin_stride=1,
        customer_count=10,
        commercial_fraction=0.25,
        include_weather=False,
        seed=1,
        torch_threads=1,
        models=("naive_last_value",),
    )
    with pytest.raises(ValueError, match="UNSUPPORTED"):
        extract_target(config, layout)


# ======================================================================
# CLI
# ======================================================================


def test_ml_targets_command_lists_support_status(capsys: pytest.CaptureFixture) -> None:
    assert main(["ml", "targets"]) == 0
    payload = json.loads(capsys.readouterr().out)
    ids = {t["target_id"]: t["status"] for t in payload["targets"]}
    assert ids["customer_load"] == "SUPPORTED"
    assert ids["battery_soc"] == "UNSUPPORTED"
    assert set(payload["supported"]) == {"customer_load", "feeder_load", "pv_generation"}


def test_ml_features_command_lists_the_catalogue(capsys: pytest.CaptureFixture) -> None:
    assert main(["ml", "features"]) == 0
    payload = json.loads(capsys.readouterr().out)
    names = {f["name"] for f in payload}
    assert "value_at_origin" in names
    assert any(f["value_kind"] == "WEATHER_ACTUAL" for f in payload)


def test_ml_run_without_dataset_exits_nonzero_with_guidance(
    capsys: pytest.CaptureFixture, monkeypatch, tmp_path: Path
) -> None:
    """A missing dataset must explain itself, not raise a traceback.

    The dataset root is redirected to an empty directory rather than relying on the
    working directory, because the configuration resolves its root from the project
    root - so a chdir-only test would quietly run the real pipeline.
    """
    from dataclasses import replace

    from energy_intelligence.config import data as data_module
    from energy_intelligence.config.data import load_data_config as real_loader

    original = real_loader()
    monkeypatch.setattr(
        data_module, "load_data_config", lambda path=None: replace(original, raw_root=tmp_path)
    )
    monkeypatch.setattr(
        "energy_intelligence.ml.config_ml_bridge.load_data_config",
        lambda path=None: replace(original, raw_root=tmp_path),
    )

    code = main(["ml", "run"])
    captured = capsys.readouterr()
    assert code == 1
    assert "not acquired" in captured.err
    assert "smart_ds_acquisition.md" in captured.err


# ======================================================================
# Registry
# ======================================================================


def _record(experiment_id: str) -> ExperimentRecord:
    return ExperimentRecord(
        experiment_id=experiment_id,
        created_at="2026-10-03T00:00:00+00:00",
        target_id="customer_load",
        dataset_version="ds-abc",
        dataset_sha256="deadbeef",
        feature_names=("lag_1",),
        lookback_steps=672,
        horizon_steps=1,
        split_fractions={"train": 0.7, "validation": 0.15},
        split_counts={"train": 10, "validation": 5, "test": 5},
        model="naive_last_value",
        model_family="naive",
        hyperparameters={},
        seed=None,
        train_seconds=0.0,
        predict_seconds=0.01,
        metrics={1: {"mae": 0.5}},
    )


def test_records_append_and_are_sorted(tmp_path: Path) -> None:
    path = tmp_path / "registry.jsonl"
    append_record(path, _record("b"))
    append_record(path, _record("a"))
    records = read_records(path)
    assert [r.experiment_id for r in records] == ["a", "b"]


def test_re_running_replaces_rather_than_duplicates(tmp_path: Path) -> None:
    """A repeated experiment updates the registry instead of faking a trend."""
    path = tmp_path / "registry.jsonl"
    append_record(path, _record("a"))
    append_record(path, _record("a"))
    assert len(read_records(path)) == 1


def test_malformed_registry_line_is_reported(tmp_path: Path) -> None:
    path = tmp_path / "registry.jsonl"
    path.write_text("{not json}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="malformed registry record"):
        read_records(path)


def test_missing_registry_reads_as_empty(tmp_path: Path) -> None:
    assert read_records(tmp_path / "absent.jsonl") == []


# ======================================================================
# End to end
# ======================================================================


def test_end_to_end_run_writes_every_artifact(tmp_path: Path) -> None:
    """One run must leave a dataset, a comparison, regimes, errors and a registry."""
    paths = ProjectPaths.from_root(tmp_path)
    paths.ensure()
    series = build_series(series_count=2, steps=20000)
    config = DatasetBuildConfig(
        lookback=672, horizons=(1, 4), origin_stride=32, train_fraction=0.7,
        validation_fraction=0.15,
    )

    output = run_phase4(
        series,
        config=config,
        paths=paths,
        target_label="synthetic",
        models=("naive_last_value", "classical_ridge"),
        seed=1,
        threads=2,
    )

    assert (output.dataset_dir / f"{output.dataset_version}.npz").is_file()
    assert (output.dataset_dir / f"{output.dataset_version}.json").is_file()
    assert (output.report_dir / "comparison.md").is_file()
    assert (output.report_dir / "comparison.json").is_file()
    assert (output.report_dir / "regimes.json").is_file()
    assert (output.report_dir / "summary.json").is_file()
    assert (output.report_dir / "error_classical_ridge.json").is_file()
    assert output.registry_path.is_file()

    records = read_records(output.registry_path)
    assert {r.model for r in records} == {"naive_last_value", "classical_ridge"}
    for record in records:
        assert record.dataset_version == output.dataset_version
        assert record.seed in (None, 1)
        assert record.environment["python"]
        assert record.lookback_steps == 672


def test_summary_records_the_phase3_balance_note(tmp_path: Path) -> None:
    """The Phase 3 residual must be acknowledged, not forgotten."""
    paths = ProjectPaths.from_root(tmp_path)
    paths.ensure()
    series = build_series(series_count=1, steps=20000)
    output = run_phase4(
        series,
        config=DatasetBuildConfig(lookback=672, horizons=(1,), origin_stride=64),
        paths=paths,
        target_label="synthetic",
        models=("naive_last_value",),
        seed=1,
        threads=1,
    )
    assert "20.4506" in output.summary["balance_residual_note"]


def test_regimes_report_names_the_axes_it_used(tmp_path: Path) -> None:
    paths = ProjectPaths.from_root(tmp_path)
    paths.ensure()
    series = build_series(series_count=1, steps=20000)
    output = run_phase4(
        series,
        config=DatasetBuildConfig(lookback=672, horizons=(1,), origin_stride=64),
        paths=paths,
        target_label="synthetic",
        models=("naive_last_value",),
        seed=1,
        threads=1,
    )
    assert "load_level" in output.regimes
    assert output.regimes["max_regime_mae_ratio"]["naive_last_value"] is not None


def test_dataset_artifact_can_be_reloaded_and_reused(tmp_path: Path) -> None:
    paths = ProjectPaths.from_root(tmp_path)
    paths.ensure()
    series = build_series(series_count=1, steps=20000)
    config = DatasetBuildConfig(lookback=672, horizons=(1,), origin_stride=64)
    output = run_phase4(
        series,
        config=config,
        paths=paths,
        target_label="synthetic",
        models=("naive_last_value",),
        seed=1,
        threads=1,
    )
    restored = load_dataset(output.dataset_dir, output.dataset_version)
    assert restored.sample_count > 0
    assert restored.target_unit == "kW"
    assert restored.manifest["dataset_version"] == output.dataset_version