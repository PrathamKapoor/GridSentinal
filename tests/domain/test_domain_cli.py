"""CLI tests for the Phase 2 domain commands and the health-check integration.

Exit codes matter more than wording: the health check is a CI gate and the
domain commands must fail loudly on an invalid domain configuration.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from energy_intelligence.cli import build_parser, main
from energy_intelligence.domain.serialization import SCHEMA_VERSION
from energy_intelligence.health import check_domain_config


# ----------------------------------------------------------------------
# domain show / validate
# ----------------------------------------------------------------------


def test_domain_show_prints_resolved_config(capsys) -> None:
    assert main(["domain", "show"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["time"]["timestep_minutes"] == 15
    assert payload["time"]["horizon_hours"] == 24.0
    assert payload["energy_system"]["topology_kind"] == "feeder"
    assert payload["data_anchor"] == "smart-ds"


def test_domain_validate_passes_for_the_shipped_config(capsys) -> None:
    assert main(["domain", "validate"]) == 0
    output = capsys.readouterr().out
    assert "domain configuration valid" in output
    assert "fingerprint=" in output


def test_domain_validate_fails_for_a_missing_file(tmp_path: Path, capsys) -> None:
    assert main(["domain", "--domain-config", str(tmp_path / "absent.toml"), "validate"]) == 1
    assert "domain configuration error" in capsys.readouterr().err


def test_domain_validate_fails_for_an_unknown_key(tmp_path: Path, capsys) -> None:
    """An unrecognised key must fail rather than be silently ignored (D-007)."""
    path = tmp_path / "bad.toml"
    path.write_text("mystery = 1\n", encoding="utf-8")
    assert main(["domain", "--domain-config", str(path), "validate"]) == 1
    assert "mystery" in capsys.readouterr().err


def test_domain_show_fails_for_an_invalid_file(tmp_path: Path) -> None:
    path = tmp_path / "bad.toml"
    path.write_text("[time]\ntimestep_minutes = 7\n", encoding="utf-8")
    assert main(["domain", "--domain-config", str(path), "show"]) == 1


def test_domain_vocabulary_is_machine_readable(capsys) -> None:
    assert main(["domain", "vocabulary"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["schema_version"] == SCHEMA_VERSION
    assert set(payload["variable_roles"]) == {
        "action",
        "constraint",
        "derived",
        "objective",
        "observation",
    }
    assert "battery" in payload["asset_types"]
    assert payload["asset_implementations"]["battery"] == "Battery"
    assert "dispatchable" in payload["authority_levels"]
    assert "advisory_only" not in payload["controllable_authority_levels"]
    assert "soc_limited" not in payload["constraint_categories"]
    assert "state_of_charge_limits" in payload["constraint_categories"]
    assert "energy_cost" in payload["objective_categories"]
    assert "line" in payload["network_element_kinds"]
    assert "stale" in payload["quality_flags"]
    assert "simulated" in payload["state_origins"]
    assert "aleatoric" not in payload["uncertainty_kinds"]
    assert "parametric" in payload["uncertainty_kinds"]


def test_vocabulary_agrees_with_the_domain_module(capsys) -> None:
    """The CLI report must not drift from the actual enums."""
    from energy_intelligence.domain import (
        ActionType,
        AssetType,
        AuthorityLevel,
        ConstraintCategory,
        ObjectiveCategory,
    )

    main(["domain", "vocabulary"])
    payload = json.loads(capsys.readouterr().out)

    assert payload["asset_types"] == sorted(t.value for t in AssetType)
    assert payload["action_types"] == sorted(a.value for a in ActionType)
    assert payload["constraint_categories"] == sorted(c.value for c in ConstraintCategory)
    assert payload["objective_categories"] == sorted(o.value for o in ObjectiveCategory)
    assert set(payload["authority_levels"]) == {a.value for a in AuthorityLevel}


def test_domain_requires_a_subcommand() -> None:
    with pytest.raises(SystemExit) as excinfo:
        build_parser().parse_args(["domain"])
    assert excinfo.value.code != 0


# ----------------------------------------------------------------------
# Health check integration
# ----------------------------------------------------------------------


def test_domain_config_check_passes_on_this_checkout() -> None:
    result = check_domain_config()
    assert result.ok
    assert "sys-phase2-reference" in result.detail
    assert "15min x 96 steps" in result.detail
    assert "anchor=smart-ds" in result.detail


def test_domain_config_check_reports_a_missing_file(monkeypatch, tmp_path: Path) -> None:
    from energy_intelligence.config import domain as domain_config_module

    monkeypatch.setattr(
        domain_config_module, "default_domain_config_path", lambda: tmp_path / "absent.toml"
    )
    result = check_domain_config()
    assert result.failed
    assert "Config file not found" in result.detail or "not found" in result.detail
    assert result.remedy


def test_domain_config_check_reports_an_invalid_config(monkeypatch, tmp_path: Path) -> None:
    from energy_intelligence.config import domain as domain_config_module

    bad = tmp_path / "bad.toml"
    bad.write_text("[time]\ntimestep_minutes = 7\n", encoding="utf-8")
    monkeypatch.setattr(domain_config_module, "default_domain_config_path", lambda: bad)

    result = check_domain_config()
    assert result.failed
    assert result.remedy


def test_domain_config_check_detects_a_broken_time_base(monkeypatch, tmp_path: Path) -> None:
    """A config that parses but cannot build a TimeBase is not healthy."""
    from energy_intelligence.config import domain as domain_config_module

    broken = tmp_path / "broken.toml"
    broken.write_text(
        'data_anchor = "none"\n'
        "[time]\n"
        "timestep_minutes = 15\n"
        "horizon_steps = 96\n"
        'origin = "not-a-timestamp"\n'
        "\n[energy_system]\n"
        'system_id = "sys-test"\n'
        'name = "T"\n'
        'description = "D"\n'
        'topology_kind = "feeder"\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(domain_config_module, "default_domain_config_path", lambda: broken)

    result = check_domain_config()
    assert result.failed
    assert "temporal configuration is not constructible" in result.detail


def test_health_includes_the_domain_check(capsys) -> None:
    assert main(["health"]) == 0
    output = capsys.readouterr().out
    assert "domain_config" in output
    assert "HEALTHY" in output


def test_health_json_includes_the_domain_check(capsys) -> None:
    assert main(["health", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    names = {check["name"] for check in payload["checks"]}
    assert "domain_config" in names
