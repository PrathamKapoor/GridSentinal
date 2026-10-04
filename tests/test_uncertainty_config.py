"""Phase 8 configuration: the loader must refuse anything that would break comparability.

The task fields are inherited, and the fixed-ensemble weights are *not* free parameters of
this phase at all. A config that moved the horizons, or that quietly re-weighted the point
forecast, would produce a table that looks like a continuation of Phase 7's and is not
comparable with it. These tests pin both.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from energy_intelligence.config.uncertainty import (
    PHASE7_FIXED_ENSEMBLE_WEIGHTS,
    UncertaintyExperimentConfig,
    load_uncertainty_config,
)

VALID_TABLE: dict[str, object] = {
    "target": "customer_load",
    "label": "uncertainty-test",
    "customer_count": 40,
    "seed": 20260101,
    "torch_threads": 8,
    "lookback_steps": 672,
    "token_stride": 4,
    "train_fraction": 0.70,
    "validation_fraction": 0.15,
    "horizons": [1, 4, 96],
    "calibration_fit_fraction": 0.5,
    "include_past_error_features": True,
    "nominal_levels": [0.50, 0.80, 0.90, 0.95],
    "band_nominal_level": 0.90,
    "scale_max_iter": 200,
    "quantile_max_iter": 200,
    "causality_sample": 32,
    "reliability_bins": 10,
    "trace_rows": 2000,
    "model_version": "phase8-uncertainty-v1",
    "run_ablations": True,
    "expert_cache": "artifacts/phase7/experts.npz",
    "phase7_result": "artifacts/phase7/router-main/result.json",
    "fixed_ensemble_weights": {
        "1": {"persistence": 0.16, "classical_hist_gbm": 0.34, "phase5_tcn": 0.50},
        "4": {"persistence": 0.00, "classical_hist_gbm": 0.34, "phase5_tcn": 0.66},
        "96": {"persistence": 0.00, "classical_hist_gbm": 0.26, "phase5_tcn": 0.74},
    },
}


def write(tmp_path: Path, table: dict[str, object], name: str = "uncertainty.toml") -> Path:
    path = tmp_path / name

    def render(value: object) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, str):
            return f'"{value}"'
        if isinstance(value, list):
            return "[" + ", ".join(render(item) for item in value) + "]"
        return str(value)

    lines: list[str] = []
    tables = {key: value for key, value in table.items() if isinstance(value, dict)}
    scalars = {key: value for key, value in table.items() if not isinstance(value, dict)}
    for key, value in scalars.items():
        lines.append(f"{key} = {render(value)}")
    for key, value in tables.items():
        lines.append("")
        lines.append(f"[{key}]")
        for row, entries in value.items():
            inner = ", ".join(f"{name} = {render(weight)}" for name, weight in entries.items())
            lines.append(f'"{row}" = {{ {inner} }}')
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_a_valid_table_loads(tmp_path: Path):
    config = load_uncertainty_config(write(tmp_path, VALID_TABLE))
    assert config.horizons == (1, 4, 96)
    assert config.nominal_levels == (0.50, 0.80, 0.90, 0.95)
    assert config.band_nominal_level == 0.90
    assert config.fixed_ensemble_weights[96]["phase5_tcn"] == pytest.approx(0.74)


def test_the_shipped_config_is_valid_and_matches_phase7():
    """The repository's own file must load and must carry Phase 7's numbers."""
    from energy_intelligence.config.loader import CONFIG_DIR

    config = load_uncertainty_config(CONFIG_DIR / "uncertainty.toml")
    for horizon, entry in PHASE7_FIXED_ENSEMBLE_WEIGHTS.items():
        assert config.fixed_ensemble_weights[int(horizon)] == pytest.approx(entry)


def test_the_shipped_config_declares_the_table_after_every_scalar(tmp_path: Path):
    """TOML scope rule: a bare key after a table header belongs to that table.

    A config that put ``expert_cache`` below ``[fixed_ensemble_weights]`` would parse
    without complaint and silently nest it under the weights, so the loader would see a
    missing required key rather than a misplaced one.
    """
    from energy_intelligence.config.loader import CONFIG_DIR

    payload = tomllib.loads(
        (CONFIG_DIR / "uncertainty.toml").read_text(encoding="utf-8")
    )
    # A bare key written below a table header becomes that table's key. So every scalar the
    # loader requires must be reachable at the top level, not nested under the weights.
    for key in ("expert_cache", "phase7_result", "horizons", "band_nominal_level"):
        assert key in payload, f"{key} was parsed inside another table"
    assert set(payload["fixed_ensemble_weights"]) == {"1", "4", "96"}


def test_missing_and_unknown_keys_are_rejected(tmp_path: Path):
    from energy_intelligence.config.schema import ConfigError, ConfigValidationError

    with pytest.raises(ConfigError, match="not found"):
        load_uncertainty_config(tmp_path / "absent.toml")

    incomplete = dict(VALID_TABLE)
    incomplete.pop("horizons")
    with pytest.raises(ConfigValidationError, match="missing required key: horizons"):
        load_uncertainty_config(write(tmp_path, incomplete))

    extra = dict(VALID_TABLE) | {"optimise_the_interval": True}
    with pytest.raises(ConfigValidationError, match="unknown keys"):
        load_uncertainty_config(write(tmp_path, extra))

    with pytest.raises(ConfigError, match="invalid TOML"):
        (tmp_path / "broken.toml").write_text("horizons = [1,", encoding="utf-8")
        load_uncertainty_config(tmp_path / "broken.toml")


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("target", "feeder_load", "inherits Phase 7's task"),
        ("customer_count", 12, "40-customer sample"),
        ("horizons", [1, 24], "inherits Phase 5's horizons"),
        ("lookback_steps", 336, "672-step window"),
        ("train_fraction", 0.6, "train_fraction of 0.70"),
        ("validation_fraction", 0.2, "validation_fraction of 0.15"),
        ("calibration_fit_fraction", 0.0, "calibration_fit_fraction must lie"),
        ("calibration_fit_fraction", 1.0, "calibration_fit_fraction must lie"),
        ("nominal_levels", [], "at least one coverage level"),
        ("nominal_levels", [0.5, 1.5], "must lie in"),
        ("nominal_levels", [0.9, 0.9], "duplicates"),
        ("band_nominal_level", 0.75, "must be one of the reported"),
        ("scale_max_iter", 0, "scale_max_iter must be positive"),
        ("quantile_max_iter", 0, "quantile_max_iter must be positive"),
        ("causality_sample", 0, "causality_sample must be positive"),
        ("reliability_bins", 1, "at least 2"),
        ("trace_rows", -1, "must not be negative"),
        ("torch_threads", 0, "torch_threads must be positive"),
        ("token_stride", 900, "shorter than lookback_steps"),
        ("model_version", "  ", "non-empty string"),
    ],
)
def test_out_of_range_fields_are_rejected(tmp_path: Path, key, value, message):
    from energy_intelligence.config.schema import ConfigValidationError

    table = dict(VALID_TABLE) | {key: value}
    with pytest.raises(ConfigValidationError, match=message):
        load_uncertainty_config(write(tmp_path, table))


def test_weights_that_differ_from_phase7_are_rejected(tmp_path: Path):
    """Phase 8 does not refit the point forecast, so it may not re-weight it."""
    from energy_intelligence.config.schema import ConfigValidationError

    drifted = {
        "1": {"persistence": 0.16, "classical_hist_gbm": 0.34, "phase5_tcn": 0.60},
        "4": {"persistence": 0.00, "classical_hist_gbm": 0.34, "phase5_tcn": 0.66},
        "96": {"persistence": 0.00, "classical_hist_gbm": 0.26, "phase5_tcn": 0.74},
    }
    with pytest.raises(ConfigValidationError, match="Phase 7 fitted"):
        load_uncertainty_config(write(tmp_path, dict(VALID_TABLE) | {"fixed_ensemble_weights": drifted}))


def test_a_renamed_expert_pool_is_rejected(tmp_path: Path):
    from energy_intelligence.config.schema import ConfigValidationError

    renamed = {
        str(h): {"persistence": 0.16, "classical_hist_gbm": 0.34, "phase5_tcn": 0.50}
        if h == "1"
        else {"persistence": 0.0, "gbm_v2": 0.34, "phase5_tcn": 0.66}
        for h in ("1", "4", "96")
    }
    with pytest.raises(ConfigValidationError, match="expert pool has changed"):
        load_uncertainty_config(write(tmp_path, dict(VALID_TABLE) | {"fixed_ensemble_weights": renamed}))


def test_weights_that_do_not_sum_to_one_are_rejected(tmp_path: Path):
    from energy_intelligence.config.schema import ConfigValidationError

    broken = {
        "1": {"persistence": 0.16, "classical_hist_gbm": 0.34, "phase5_tcn": 0.20},
        "4": {"persistence": 0.00, "classical_hist_gbm": 0.34, "phase5_tcn": 0.66},
        "96": {"persistence": 0.00, "classical_hist_gbm": 0.26, "phase5_tcn": 0.74},
    }
    with pytest.raises(ConfigValidationError, match="sum to"):
        load_uncertainty_config(write(tmp_path, dict(VALID_TABLE) | {"fixed_ensemble_weights": broken}))


def test_a_missing_horizon_in_the_weights_is_rejected(tmp_path: Path):
    from energy_intelligence.config.schema import ConfigValidationError

    partial = {
        key: value for key, value in VALID_TABLE["fixed_ensemble_weights"].items()  # type: ignore[index]
        if key != "96"
    }
    table = dict(VALID_TABLE) | {"fixed_ensemble_weights": partial}
    with pytest.raises(ConfigValidationError, match="no entry for horizon 96"):
        load_uncertainty_config(write(tmp_path, table))


def test_malformed_weight_tables_are_rejected(tmp_path: Path):
    from energy_intelligence.config.schema import ConfigValidationError

    with pytest.raises(ConfigValidationError, match="non-empty table"):
        load_uncertainty_config(write(tmp_path, dict(VALID_TABLE) | {"fixed_ensemble_weights": {}}))
    scalar = write(tmp_path, VALID_TABLE).read_text(encoding="utf-8").replace(
        '[fixed_ensemble_weights]\n"1" = { persistence = 0.16, classical_hist_gbm = 0.34, phase5_tcn = 0.5 }\n"4" = { persistence = 0.0, classical_hist_gbm = 0.34, phase5_tcn = 0.66 }\n"96" = { persistence = 0.0, classical_hist_gbm = 0.26, phase5_tcn = 0.74 }',
        '[fixed_ensemble_weights]\n"1" = 0.5\n',
    )
    assert scalar != write(tmp_path, VALID_TABLE).read_text(encoding="utf-8")
    path = tmp_path / "scalar.toml"
    path.write_text(scalar, encoding="utf-8")
    with pytest.raises(ConfigValidationError, match="must be a table of expert"):
        load_uncertainty_config(path)


def test_weight_array_orders_by_the_pool_and_reports_a_mismatch():
    config = UncertaintyExperimentConfig(
        **{
            key: value
            for key, value in VALID_TABLE.items()
            if key != "fixed_ensemble_weights"
        },
        fixed_ensemble_weights={
            int(h): dict(entry) for h, entry in VALID_TABLE["fixed_ensemble_weights"].items()  # type: ignore[index]
        },
    )
    assert config.weight_array(("phase5_tcn", "persistence", "classical_hist_gbm")) == [
        [0.50, 0.16, 0.34],
        [0.66, 0.00, 0.34],
        [0.74, 0.00, 0.26],
    ]
    with pytest.raises(ValueError, match="omit"):
        config.weight_array(("phase5_tcn", "a_new_expert", "classical_hist_gbm"))


def test_to_dict_round_trips_through_the_loader(tmp_path: Path):
    from energy_intelligence.config.loader import CONFIG_DIR

    config = load_uncertainty_config(CONFIG_DIR / "uncertainty.toml")
    payload = config.to_dict()
    assert payload["horizons"] == [1, 4, 96]
    assert payload["fixed_ensemble_weights"]["1"]["phase5_tcn"] == pytest.approx(0.5)
    assert payload["calibration_fit_fraction"] == 0.5