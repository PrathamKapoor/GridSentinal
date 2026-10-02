"""CLI tests.

The health check is a CI gate, so the exit codes matter more than the exact
wording of the output.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from energy_intelligence.cli import build_parser, main


def test_help_exits_zero(capsys) -> None:
    with pytest.raises(SystemExit) as excinfo:
        build_parser().parse_args(["--help"])
    assert excinfo.value.code == 0
    assert "energy-intel" in capsys.readouterr().out


def test_a_command_is_required() -> None:
    with pytest.raises(SystemExit) as excinfo:
        build_parser().parse_args([])
    assert excinfo.value.code != 0


def test_config_show_prints_resolved_config(capsys) -> None:
    assert main(["config", "show"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["experiment_name"] == "phase1-baseline"
    assert payload["seed"] == 42
    assert Path(payload["log_dir"]).is_absolute()


def test_config_validate_passes_for_the_shipped_config(capsys) -> None:
    assert main(["config", "validate"]) == 0
    assert "configuration valid" in capsys.readouterr().out


def test_config_validate_fails_for_a_missing_file(tmp_path, capsys) -> None:
    assert main(["--config", str(tmp_path / "nope.toml"), "config", "validate"]) == 1
    assert "configuration error" in capsys.readouterr().err


def test_config_validate_fails_for_an_invalid_file(tmp_path, capsys) -> None:
    bad = tmp_path / "bad.toml"
    bad.write_text('device = "tpu"\n', encoding="utf-8")
    assert main(["--config", str(bad), "config", "validate"]) == 1
    assert "device" in capsys.readouterr().err


def test_global_overrides_apply(capsys) -> None:
    assert main(["--seed", "123", "--experiment-name", "cli-run", "config", "show"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["seed"] == 123
    assert payload["experiment_name"] == "cli-run"


def test_invalid_experiment_name_override_is_rejected(capsys) -> None:
    assert main(["--experiment-name", "Bad Name", "config", "show"]) == 1
    assert "experiment_name" in capsys.readouterr().err


def test_named_config_file_is_accepted(capsys) -> None:
    from energy_intelligence.config import CONFIG_DIR

    assert main(["--config", "configs/test.toml", "config", "show"]) == 0
    assert json.loads(capsys.readouterr().out)["experiment_name"] == "test-run"


def test_health_command_succeeds_on_this_checkout(capsys) -> None:
    assert main(["health"]) == 0
    output = capsys.readouterr().out
    assert "HEALTHY" in output
    assert "python_version" in output


def test_health_json_output_is_parseable(capsys) -> None:
    assert main(["health", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    names = {check["name"] for check in payload["checks"]}
    assert "python_version" in names
    for check in payload["checks"]:
        assert check["status"] in {"ok", "warn", "fail"}
        assert check["detail"]


def test_health_create_dirs_flag(capsys) -> None:
    assert main(["health", "--create-dirs"]) == 0
    assert "HEALTHY" in capsys.readouterr().out


def test_health_fails_when_the_config_is_broken(tmp_path, capsys) -> None:
    """A broken config must surface as a non-zero exit, not a traceback."""
    assert main(["--config", str(tmp_path / "absent.toml"), "health"]) == 1
    assert "FAIL" in capsys.readouterr().out


def test_paths_command_reports_presence(capsys) -> None:
    assert main(["paths"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["required"]["data/raw"] is True
    assert payload["required"]["logs"] is True


def test_init_dirs_is_idempotent(capsys) -> None:
    assert main(["init-dirs"]) == 0
    first = capsys.readouterr().out
    assert "already present" in first or "created" in first

    assert main(["init-dirs"]) == 0
    assert "already present" in capsys.readouterr().out


def test_unknown_command_is_rejected() -> None:
    with pytest.raises(SystemExit) as excinfo:
        build_parser().parse_args(["not-a-command"])
    assert excinfo.value.code != 0
