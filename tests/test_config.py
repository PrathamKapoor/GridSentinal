"""Configuration tests.

Covers requirements 2 and 3 of the Phase 1 brief: configuration loads
correctly, and invalid configuration is rejected.
"""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path

import pytest

from energy_intelligence.config import (
    CONFIG_FIELDS,
    MAX_SEED,
    AppConfig,
    ConfigError,
    ConfigValidationError,
    coerce_app_config,
    default_config_path,
    find_project_root,
    load_config,
    load_raw_toml,
    validate_app_config,
)

# --------------------------------------------------------------------------
# Happy path
# --------------------------------------------------------------------------


def test_valid_table_produces_a_config(config_table, tmp_paths) -> None:
    config = coerce_app_config(config_table(), project_root=tmp_paths.root)

    assert isinstance(config, AppConfig)
    assert config.seed == 7
    assert config.device == "cpu"
    assert config.experiment_name == "test-run"
    assert config.log_level == "INFO"
    assert config.is_cpu_only is True


def test_relative_paths_resolve_against_the_project_root(config_table, tmp_paths) -> None:
    config = coerce_app_config(config_table(), project_root=tmp_paths.root)

    assert config.model_path == (tmp_paths.root / "models" / "test-model").resolve()
    assert config.dataset_path == (tmp_paths.root / "data" / "processed").resolve()
    assert config.log_dir == (tmp_paths.root / "logs").resolve()


def test_absolute_paths_are_preserved(config_table, tmp_path) -> None:
    absolute = tmp_path / "elsewhere"
    config = coerce_app_config(
        config_table(output_dir=str(absolute)), project_root=tmp_path
    )
    assert config.output_dir == absolute.resolve()


def test_home_relative_paths_are_expanded(config_table, tmp_path) -> None:
    config = coerce_app_config(
        config_table(output_dir="~/some-output"), project_root=tmp_path
    )
    assert config.output_dir.is_absolute()
    assert "~" not in str(config.output_dir)
    assert config.output_dir.name == "some-output"


def test_config_is_immutable(config_table, tmp_paths) -> None:
    """A frozen config prevents mid-run mutation breaking reproducibility."""
    config = coerce_app_config(config_table(), project_root=tmp_paths.root)

    with pytest.raises(dataclasses.FrozenInstanceError):
        config.seed = 999  # type: ignore[misc]


def test_project_root_is_injected_and_absolute(config_table, tmp_paths) -> None:
    config = coerce_app_config(config_table(), project_root=tmp_paths.root)
    assert config.project_root == tmp_paths.root.resolve()


def test_as_dict_is_json_serialisable(config_table, tmp_paths) -> None:
    import json

    config = coerce_app_config(config_table(), project_root=tmp_paths.root)
    payload = json.dumps(config.as_dict())
    assert '"experiment_name": "test-run"' in payload


def test_fingerprint_is_stable_for_identical_settings(config_table, tmp_path) -> None:
    a = coerce_app_config(config_table(), project_root=tmp_path)
    b = coerce_app_config(config_table(), project_root=tmp_path)
    assert a.fingerprint() == b.fingerprint()
    assert len(a.fingerprint()) == 12


def test_fingerprint_changes_when_any_value_changes(config_table, tmp_paths) -> None:
    base = coerce_app_config(config_table(), project_root=tmp_paths.root)
    assert base.fingerprint() != coerce_app_config(
        config_table(seed=8), project_root=tmp_paths.root
    ).fingerprint()
    assert base.fingerprint() != coerce_app_config(
        config_table(log_level="DEBUG"), project_root=tmp_paths.root
    ).fingerprint()
    assert base.fingerprint() != coerce_app_config(
        config_table(experiment_name="other"), project_root=tmp_paths.root
    ).fingerprint()


def test_fingerprint_differs_between_checkouts(config_table, tmp_path) -> None:
    """Resolved paths embed the root, so two checkouts fingerprint differently.

    This is intentional: it means a fingerprint identifies one concrete
    checkout, not just a set of relative settings.
    """
    a = coerce_app_config(config_table(), project_root=tmp_path / "one")
    b = coerce_app_config(config_table(), project_root=tmp_path / "two")
    assert a.fingerprint() != b.fingerprint()


def test_device_and_log_level_are_case_normalised(config_table, tmp_paths) -> None:
    config = coerce_app_config(
        config_table(device="  CPU ", log_level="debug"), project_root=tmp_paths.root
    )
    assert config.device == "cpu"
    assert config.log_level == "DEBUG"


# --------------------------------------------------------------------------
# Rejection: unknown / missing keys
# --------------------------------------------------------------------------


def test_unknown_key_is_rejected(config_table, tmp_paths) -> None:
    with pytest.raises(ConfigValidationError) as excinfo:
        coerce_app_config(
            config_table(experimant_name="typo"), project_root=tmp_paths.root
        )
    message = str(excinfo.value)
    assert "unknown configuration key" in message
    assert "experimant_name" in message


@pytest.mark.parametrize("field", CONFIG_FIELDS)
def test_every_required_key_is_enforced(config_table, tmp_paths, remove_key, field) -> None:
    with pytest.raises(ConfigValidationError) as excinfo:
        coerce_app_config(config_table(**{field: remove_key}), project_root=tmp_paths.root)
    assert f"missing required configuration key '{field}'" in str(excinfo.value)


def test_all_errors_are_reported_together(config_table, tmp_paths) -> None:
    """A user should be able to fix every problem in one pass."""
    with pytest.raises(ConfigValidationError) as excinfo:
        coerce_app_config(
            config_table(seed=-1, device="tpu", log_level="LOUD", bogus=1),
            project_root=tmp_paths.root,
        )
    assert len(excinfo.value.errors) >= 4


def test_non_mapping_input_is_rejected(tmp_paths) -> None:
    with pytest.raises(ConfigValidationError):
        coerce_app_config(["not", "a", "table"], project_root=tmp_paths.root)  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# Rejection: field-level rules
# --------------------------------------------------------------------------


@pytest.mark.parametrize("bad_seed", [-1, MAX_SEED + 1, 1.5, "seven", True, None])
def test_invalid_seed_is_rejected(config_table, tmp_paths, bad_seed) -> None:
    with pytest.raises(ConfigValidationError) as excinfo:
        coerce_app_config(config_table(seed=bad_seed), project_root=tmp_paths.root)
    assert "seed" in str(excinfo.value)


@pytest.mark.parametrize("bad_seed", [0, 1, MAX_SEED])
def test_boundary_seeds_are_accepted(config_table, tmp_paths, bad_seed) -> None:
    config = coerce_app_config(config_table(seed=bad_seed), project_root=tmp_paths.root)
    assert config.seed == bad_seed


@pytest.mark.parametrize("bad_device", ["tpu", "gpu0", "", "cuda:9", 3])
def test_invalid_device_is_rejected(config_table, tmp_paths, bad_device) -> None:
    with pytest.raises(ConfigValidationError) as excinfo:
        coerce_app_config(config_table(device=bad_device), project_root=tmp_paths.root)
    assert "device" in str(excinfo.value)


@pytest.mark.parametrize("device", ["cpu", "cuda", "mps"])
def test_supported_devices_are_accepted(config_table, tmp_paths, device) -> None:
    config = coerce_app_config(config_table(device=device), project_root=tmp_paths.root)
    assert config.device == device


@pytest.mark.parametrize("bad_level", ["VERBOSE", "", "infoo", 7])
def test_invalid_log_level_is_rejected(config_table, tmp_paths, bad_level) -> None:
    with pytest.raises(ConfigValidationError) as excinfo:
        coerce_app_config(config_table(log_level=bad_level), project_root=tmp_paths.root)
    assert "log_level" in str(excinfo.value)


@pytest.mark.parametrize(
    "bad_name",
    [
        "",
        "Uppercase",
        "-leading-dash",
        "has space",
        "has/slash",
        "..",
        "a" * 65,
        "naïve",
        123,
    ],
)
def test_invalid_experiment_name_is_rejected(config_table, tmp_paths, bad_name) -> None:
    """Slugs are restricted because they become log file and directory names."""
    with pytest.raises(ConfigValidationError) as excinfo:
        coerce_app_config(config_table(experiment_name=bad_name), project_root=tmp_paths.root)
    assert "experiment_name" in str(excinfo.value)


@pytest.mark.parametrize("good_name", ["a", "run-1", "run_1", "run.1", "e" * 64, "phase1"])
def test_valid_experiment_names_are_accepted(config_table, tmp_paths, good_name) -> None:
    config = coerce_app_config(config_table(experiment_name=good_name), project_root=tmp_paths.root)
    assert config.experiment_name == good_name


@pytest.mark.parametrize("bad_path", ["", "   ", 5, None, ["a"]])
def test_invalid_path_is_rejected(config_table, tmp_paths, bad_path) -> None:
    with pytest.raises(ConfigValidationError) as excinfo:
        coerce_app_config(config_table(model_path=bad_path), project_root=tmp_paths.root)
    assert "model_path" in str(excinfo.value)


def test_paths_are_not_required_to_exist(config_table, tmp_paths) -> None:
    """Phase 1 creates no model or dataset.

    Requiring existence would make a valid Phase 1 config unbuildable, since
    Phases 3 and 6 own those artifacts. See decisions.md D-007.
    """
    config = coerce_app_config(
        config_table(model_path="models/does-not-exist-yet"), project_root=tmp_paths.root
    )
    assert not config.model_path.exists()
    assert config.model_path.is_absolute()


# --------------------------------------------------------------------------
# validate_app_config
# --------------------------------------------------------------------------


def test_validate_accepts_a_valid_config(config_table, tmp_paths) -> None:
    config = coerce_app_config(config_table(), project_root=tmp_paths.root)
    assert validate_app_config(config) is None


def test_validate_rejects_a_hand_built_bad_config(tmp_paths) -> None:
    """Defensive check for configs not produced by the loader."""
    bad = AppConfig(
        seed=-5,
        device="tpu",
        model_path=Path("relative/model"),
        dataset_path=tmp_paths.root / "d",
        experiment_name="Bad Name",
        log_level="LOUD",
        output_dir=tmp_paths.root / "out",
        artifact_dir=tmp_paths.root / "art",
        log_dir=tmp_paths.root / "logs",
        project_root=tmp_paths.root,
    )
    with pytest.raises(ConfigValidationError) as excinfo:
        validate_app_config(bad)
    errors = " ".join(excinfo.value.errors)
    assert "seed" in errors
    assert "device" in errors
    assert "log_level" in errors
    assert "experiment_name" in errors
    assert "model_path" in errors


# --------------------------------------------------------------------------
# Loader: files, TOML syntax, layering, environment
# --------------------------------------------------------------------------


def test_default_config_file_exists_and_is_valid() -> None:
    """The shipped configs/default.toml must satisfy the shipped schema."""
    path = default_config_path()
    assert path.is_file(), f"missing default config: {path}"
    config = load_config()
    assert config.seed == 42
    assert config.experiment_name == "phase1-baseline"


def test_test_config_file_is_valid() -> None:
    from energy_intelligence.config import CONFIG_DIR

    config = load_config(CONFIG_DIR / "test.toml")
    assert config.experiment_name == "test-run"
    assert config.log_level == "DEBUG"


def test_missing_config_file_raises(tmp_path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.toml")


def test_invalid_toml_syntax_raises_config_error(write_toml) -> None:
    path = write_toml("broken.toml", "seed = = 3\n")
    with pytest.raises(ConfigError, match="not valid TOML"):
        load_raw_toml(path)


def test_config_layering_overrides_defaults() -> None:
    from energy_intelligence.config import CONFIG_DIR

    config = load_config(CONFIG_DIR / "test.toml")
    assert config.experiment_name == "test-run"  # from test.toml
    assert config.device == "cpu"  # from test.toml


def test_partial_layer_falls_back_to_defaults(write_toml) -> None:
    path = write_toml("partial.toml", 'experiment_name = "partial-run"\n')
    config = load_config(path)
    assert config.experiment_name == "partial-run"
    assert config.seed == 42  # inherited from default.toml


def test_layer_file_with_unknown_key_is_rejected(write_toml) -> None:
    path = write_toml("unknown.toml", 'not_a_key = "x"\n')
    with pytest.raises(ConfigValidationError, match="unknown configuration key"):
        load_config(path)


def test_relative_layer_path_resolves_from_project_root() -> None:
    config = load_config("configs/test.toml")
    assert config.experiment_name == "test-run"


def test_overrides_take_priority_over_files() -> None:
    config = load_config(overrides={"seed": 999, "experiment_name": "override-run"})
    assert config.seed == 999
    assert config.experiment_name == "override-run"


def test_none_overrides_are_ignored() -> None:
    """Absent CLI flags arrive as None and must not overwrite real values."""
    config = load_config(overrides={"seed": None})
    assert config.seed == 42


def test_unknown_override_key_is_rejected() -> None:
    with pytest.raises(ConfigValidationError, match="unknown configuration key"):
        load_config(overrides={"experimant_name": "typo"})


def test_config_path_can_come_from_the_environment(monkeypatch, tmp_path) -> None:
    from energy_intelligence.config import CONFIG_DIR_ENV_VAR

    path = tmp_path / "env.toml"
    path.write_text('experiment_name = "env-run"\n', encoding="utf-8")
    monkeypatch.setenv(CONFIG_DIR_ENV_VAR, str(path))

    config = load_config()
    assert config.experiment_name == "env-run"


def test_explicit_path_beats_environment(monkeypatch, tmp_path) -> None:
    from energy_intelligence.config import CONFIG_DIR_ENV_VAR

    env_path = tmp_path / "env.toml"
    env_path.write_text('experiment_name = "env-run"\n', encoding="utf-8")
    monkeypatch.setenv(CONFIG_DIR_ENV_VAR, str(env_path))

    explicit = tmp_path / "explicit.toml"
    explicit.write_text('experiment_name = "explicit-run"\n', encoding="utf-8")

    assert load_config(explicit).experiment_name == "explicit-run"


def test_environment_is_not_mutated(monkeypatch) -> None:
    before = dict(os.environ)
    load_config()
    assert dict(os.environ) == before


# --------------------------------------------------------------------------
# Project root discovery
# --------------------------------------------------------------------------


def test_find_project_root_locates_the_checkout() -> None:
    root = find_project_root()
    assert (root / "pyproject.toml").is_file()
    assert (root / "src" / "energy_intelligence").is_dir()


def test_find_project_root_fails_outside_a_checkout(tmp_path) -> None:
    with pytest.raises(ConfigError, match="Could not locate the project root"):
        find_project_root(tmp_path)


def test_find_project_root_starts_from_a_nested_directory() -> None:
    nested = find_project_root() / "src" / "energy_intelligence" / "config"
    assert find_project_root(nested) == find_project_root()
