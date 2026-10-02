"""Tests for the Phase 2 domain configuration extension.

Phase 1 established run configuration; Phase 2 adds domain configuration in a
separate module. These tests confirm the extension is strict in the same way
Phase 1's is, and that the shipped ``configs/domain.toml`` is valid.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from energy_intelligence.config.domain import (
    VALID_TIMESTEPS_MINUTES,
    DataAnchor,
    DomainConfig,
    EnergySystemConfig,
    TimeConfig,
    default_domain_config_path,
    load_domain_config,
)
from energy_intelligence.config.schema import ConfigError, ConfigValidationError


def _valid_time(**overrides) -> dict:
    table = {"timestep_minutes": 15, "horizon_steps": 96, "origin": "2026-01-01T00:00:00+00:00"}
    table.update(overrides)
    return table


def _valid_system(**overrides) -> dict:
    table = {
        "system_id": "sys-test",
        "name": "Test",
        "description": "Test system",
        "topology_kind": "feeder",
    }
    table.update(overrides)
    return table


# ----------------------------------------------------------------------
# The shipped config
# ----------------------------------------------------------------------


def test_shipped_domain_config_is_valid() -> None:
    path = default_domain_config_path()
    assert path.is_file(), f"missing domain config: {path}"
    config = load_domain_config()
    assert config.energy_system.system_id.startswith("sys-")
    assert config.time.timestep_minutes in VALID_TIMESTEPS_MINUTES


def test_shipped_domain_config_matches_the_phase_2_decisions() -> None:
    """The chosen time base must actually be what the file declares."""
    config = load_domain_config()
    assert config.time.timestep_minutes == 15
    assert config.time.horizon_steps == 96
    assert config.time.horizon_hours == 24.0
    assert config.energy_system.topology_kind == "feeder"
    assert config.data_anchor == DataAnchor.SMART_DS


def test_domain_config_reports_a_fingerprint() -> None:
    assert len(load_domain_config().fingerprint()) == 12


def test_fingerprint_is_stable_across_loads() -> None:
    assert load_domain_config().fingerprint() == load_domain_config().fingerprint()


def test_fingerprint_changes_with_content(tmp_path: Path) -> None:
    baseline = load_domain_config().fingerprint()
    other = tmp_path / "other.toml"
    other.write_text(
        _render(_valid_time(horizon_steps=48), _valid_system(), DataAnchor.NONE),
        encoding="utf-8",
    )
    assert load_domain_config(other).fingerprint() != baseline


def test_as_dict_is_json_serialisable() -> None:
    import json

    payload = json.dumps(load_domain_config().as_dict())
    assert '"timestep_minutes": 15' in payload


# ----------------------------------------------------------------------
# TimeConfig
# ----------------------------------------------------------------------


def test_time_config_reports_horizon_hours() -> None:
    assert TimeConfig(15, 96, "2026-01-01T00:00:00+00:00").horizon_hours == 24.0
    assert TimeConfig(60, 24, "2026-01-01T00:00:00+00:00").horizon_hours == 24.0


@pytest.mark.parametrize("bad", [0, -15, 7, 90, 2.5, True])
def test_time_config_rejects_unusable_timesteps(bad: object) -> None:
    """Only values that aggregate cleanly to coarser grids are permitted."""
    with pytest.raises(ConfigValidationError, match="timestep_minutes"):
        TimeConfig(bad, 96, "2026-01-01T00:00:00+00:00")  # type: ignore[arg-type]


@pytest.mark.parametrize("timestep", sorted(VALID_TIMESTEPS_MINUTES))
def test_every_permitted_timestep_is_accepted(timestep: int) -> None:
    assert TimeConfig(timestep, 96, "2026-01-01T00:00:00+00:00").timestep_minutes == timestep


@pytest.mark.parametrize("bad", [0, -1, 1.5, True])
def test_time_config_rejects_invalid_horizon(bad: object) -> None:
    with pytest.raises(ConfigValidationError, match="horizon_steps"):
        TimeConfig(15, bad, "2026-01-01T00:00:00+00:00")  # type: ignore[arg-type]


def test_time_config_rejects_empty_origin() -> None:
    with pytest.raises(ConfigValidationError, match="origin"):
        TimeConfig(15, 96, "")


# ----------------------------------------------------------------------
# EnergySystemConfig
# ----------------------------------------------------------------------


def test_energy_system_config_accepts_valid_identity() -> None:
    config = EnergySystemConfig(**_valid_system())
    assert config.system_id == "sys-test"


def test_energy_system_config_rejects_invalid_system_id() -> None:
    with pytest.raises(ConfigValidationError, match="system_id"):
        EnergySystemConfig(**_valid_system(system_id="not-a-system-id"))


def test_energy_system_config_rejects_bad_topology_kind() -> None:
    with pytest.raises(ConfigValidationError, match="topology_kind"):
        EnergySystemConfig(**_valid_system(topology_kind="mesh"))


def test_energy_system_config_requires_name_and_description() -> None:
    with pytest.raises(ConfigValidationError, match="name"):
        EnergySystemConfig(**_valid_system(name=""))
    with pytest.raises(ConfigValidationError, match="description"):
        EnergySystemConfig(**_valid_system(description="  "))


# ----------------------------------------------------------------------
# DomainConfig
# ----------------------------------------------------------------------


def test_domain_config_rejects_unknown_anchor() -> None:
    with pytest.raises(ConfigValidationError, match="data_anchor"):
        DomainConfig(
            time=TimeConfig(15, 96, "2026-01-01T00:00:00+00:00"),
            energy_system=EnergySystemConfig(**_valid_system()),
            data_anchor="whatever",
        )


def test_domain_config_rejects_wrong_section_types() -> None:
    with pytest.raises(ConfigValidationError, match="must be a TimeConfig"):
        DomainConfig(
            time="nope",  # type: ignore[arg-type]
            energy_system=EnergySystemConfig(**_valid_system()),
        )


# ----------------------------------------------------------------------
# Loader strictness
# ----------------------------------------------------------------------


def _render(time: dict, system: dict, anchor: str = "none") -> str:
    return (
        f"data_anchor = \"{anchor}\"\n\n"
        "[time]\n"
        + "".join(f"{key} = {value!r}\n" for key, value in time.items())
        + "\n[energy_system]\n"
        + "".join(f"{key} = {value!r}\n" for key, value in system.items())
    )


def test_loader_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_domain_config(tmp_path / "absent.toml")


def test_loader_rejects_invalid_toml(tmp_path: Path) -> None:
    path = tmp_path / "bad.toml"
    path.write_text("[time\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="not valid TOML"):
        load_domain_config(path)


def test_loader_rejects_unknown_top_level_key(tmp_path: Path) -> None:
    # The unknown key must precede the first [table] header: a key written after
    # one belongs to that table and is reported by the section check instead.
    path = tmp_path / "unknown.toml"
    path.write_text(
        "mystery = 1\n" + _render(_valid_time(), _valid_system()), encoding="utf-8"
    )
    with pytest.raises(ConfigValidationError, match="unknown domain config key 'mystery'"):
        load_domain_config(path)


def test_loader_reports_unknown_section_header_as_an_unknown_key(tmp_path: Path) -> None:
    path = tmp_path / "unknown.toml"
    path.write_text(_render(_valid_time(), _valid_system()) + "\n[extra]\nx = 1\n", encoding="utf-8")
    with pytest.raises(ConfigValidationError, match="unknown domain config key 'extra'"):
        load_domain_config(path)


def test_loader_rejects_unknown_key_inside_time(tmp_path: Path) -> None:
    path = tmp_path / "unknown.toml"
    path.write_text(
        _render({**_valid_time(), "mystery": 1}, _valid_system()), encoding="utf-8"
    )
    with pytest.raises(ConfigValidationError, match=r"unknown key 'mystery' in \[time\]"):
        load_domain_config(path)


def test_loader_rejects_unknown_key_inside_energy_system(tmp_path: Path) -> None:
    path = tmp_path / "unknown.toml"
    path.write_text(
        _render(_valid_time(), {**_valid_system(), "mystery": 2}), encoding="utf-8"
    )
    with pytest.raises(ConfigValidationError, match=r"unknown key 'mystery' in \[energy_system\]"):
        load_domain_config(path)


def test_loader_rejects_missing_section(tmp_path: Path) -> None:
    path = tmp_path / "partial.toml"
    path.write_text("[time]\ntimestep_minutes = 15\nhorizon_steps = 96\norigin = 'x'\n", encoding="utf-8")
    with pytest.raises(ConfigValidationError, match=r"missing required section \[energy_system\]"):
        load_domain_config(path)


def test_loader_resolves_relative_paths_from_project_root() -> None:
    config = load_domain_config("configs/domain.toml")
    assert config.time.timestep_minutes == 15


def test_loader_accepts_a_valid_override_file(tmp_path: Path) -> None:
    path = tmp_path / "custom.toml"
    path.write_text(_render(_valid_time(horizon_steps=48), _valid_system(), "smart-ds"), encoding="utf-8")
    config = load_domain_config(path)
    assert config.time.horizon_steps == 48
    assert config.data_anchor == "smart-ds"


def test_loader_reports_section_validation_errors_with_the_file_name(tmp_path: Path) -> None:
    path = tmp_path / "bad.toml"
    path.write_text(_render(_valid_time(timestep_minutes=7), _valid_system()), encoding="utf-8")
    with pytest.raises(ConfigValidationError) as excinfo:
        load_domain_config(path)
    assert any("timestep_minutes" in error for error in excinfo.value.errors)
    assert any(str(path) in error for error in excinfo.value.errors)
