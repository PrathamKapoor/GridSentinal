"""Health check tests.

Covers requirement 4 of the Phase 1 brief: a project-level basic health check
works. The health check must both pass on a sound environment and correctly
fail when a foundation is actually broken.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from energy_intelligence.health import (
    CheckResult,
    HealthReport,
    check_directories,
    check_health,
    check_logging,
    check_package_import,
    check_python_version,
    check_test_infrastructure,
    render_report,
)
from energy_intelligence.paths import ProjectPaths

ALL_CHECK_NAMES = {
    "python_version",
    "package_import",
    "config",
    "directories",
    "test_infrastructure",
    "domain_config",
    "logging",
}


# --------------------------------------------------------------------------
# Aggregate behaviour
# --------------------------------------------------------------------------


def test_health_check_reports_every_foundation() -> None:
    report = check_health()
    assert {check.name for check in report.checks} == ALL_CHECK_NAMES


def test_real_checkout_is_healthy() -> None:
    """The Phase 1 deliverable: this repository passes its own health check."""
    report = check_health()
    assert report.ok, render_report(report)
    assert report.failures() == ()
    assert report.exit_code == 0


def test_report_carries_the_loaded_config() -> None:
    report = check_health()
    assert report.config is not None
    assert report.config.experiment_name == "phase1-baseline"


def test_missing_directories_fail_the_check(tmp_path: Path) -> None:
    """A temp root with no layout must be reported, not silently passed."""
    paths = ProjectPaths.from_root(tmp_path)
    assert paths.missing()

    result = check_directories(paths)
    assert result.failed
    assert "missing" in result.detail
    assert result.remedy


def test_create_missing_dirs_recovers(tmp_path: Path) -> None:
    report = check_health(paths=ProjectPaths.from_root(tmp_path), create_missing_dirs=True)
    directories = next(c for c in report.checks if c.name == "directories")
    assert directories.ok
    assert (tmp_path / "data" / "raw").is_dir()


def test_config_failure_is_isolated_to_the_config_check(tmp_path: Path) -> None:
    """One broken foundation must not mask or crash the others."""
    report = check_health(tmp_path / "nonexistent.toml", paths=ProjectPaths.from_root(tmp_path / "x"))
    config_result = next(c for c in report.checks if c.name == "config")
    assert config_result.failed
    assert report.config is None
    assert not report.ok

    # Other checks still ran.
    assert {c.name for c in report.checks} == ALL_CHECK_NAMES
    # Logging degrades to a warning because there is no config.
    logging_result = next(c for c in report.checks if c.name == "logging")
    assert logging_result.status == "warn"


def test_check_result_status_semantics() -> None:
    assert CheckResult("a", "ok", "d").ok
    assert not CheckResult("a", "ok", "d").failed
    assert CheckResult("a", "fail", "d").failed
    assert CheckResult("a", "warn", "d").status == "warn"


def test_sort_key_puts_failures_last_for_readability() -> None:
    """render_report sorts ascending by severity so problems end up at the bottom."""
    results = [
        CheckResult("fail", "fail", ""),
        CheckResult("warn", "warn", ""),
        CheckResult("ok", "ok", ""),
    ]
    ordered = sorted(results, key=lambda item: item.sort_key)
    assert [item.name for item in ordered] == ["ok", "warn", "fail"]


# --------------------------------------------------------------------------
# Individual checks
# --------------------------------------------------------------------------


def test_python_version_check_passes_on_312_or_newer() -> None:
    result = check_python_version()
    assert result.ok
    assert "Python 3.12" in result.detail


def test_python_version_check_fails_when_minimum_is_impossible() -> None:
    result = check_python_version(minimum="99.0")
    assert result.failed
    assert "requires" in result.detail


def test_package_import_check_passes_in_the_checkout() -> None:
    result = check_package_import()
    assert result.ok
    assert "site-packages" not in result.detail


def test_directories_check_passes_for_a_complete_layout(tmp_paths) -> None:
    result = check_directories(tmp_paths)
    assert result.ok
    assert "8 directories" in result.detail


def test_test_infrastructure_check_passes_in_the_checkout() -> None:
    result = check_test_infrastructure()
    assert result.ok
    assert "pytest" in result.detail


def test_test_infrastructure_check_fails_without_a_tests_dir(tmp_path: Path) -> None:
    paths = ProjectPaths.from_root(tmp_path)
    result = check_test_infrastructure(paths)
    assert result.failed
    assert "does not exist" in result.detail


def test_test_infrastructure_check_fails_with_an_empty_tests_dir(tmp_path: Path) -> None:
    paths = ProjectPaths.from_root(tmp_path)
    (paths.root / "tests").mkdir(parents=True)
    result = check_test_infrastructure(paths)
    assert result.failed
    assert "no test_*.py files" in result.detail


def test_test_infrastructure_check_accepts_a_populated_tests_dir(tmp_path: Path) -> None:
    paths = ProjectPaths.from_root(tmp_path)
    tests_dir = paths.root / "tests"
    tests_dir.mkdir(parents=True)
    (tests_dir / "test_example.py").write_text("def test_ok():\n    assert True\n", encoding="utf-8")

    result = check_test_infrastructure(paths)
    assert result.ok
    assert "1 test module(s)" in result.detail


def test_logging_check_reports_a_writable_file(make_config) -> None:
    result = check_logging(make_config())
    assert result.ok
    assert "JSON Lines log writable" in result.detail


def test_logging_check_fails_when_the_directory_is_blocked(tmp_path: Path, make_config) -> None:
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("file in the way", encoding="utf-8")

    result = check_logging(make_config(log_dir=str(blocker / "logs")))
    assert result.failed
    assert result.remedy


def test_logging_check_warns_without_a_config() -> None:
    result = check_logging(None)
    assert result.status == "warn"


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------


def test_render_report_includes_every_check_and_a_verdict(capsys) -> None:
    report = check_health()
    output = render_report(report)

    for name in ALL_CHECK_NAMES:
        assert name in output
    assert "HEALTHY" in output
    assert "passed," in output
    assert report.config.fingerprint() in output


def test_render_report_shows_remedies_only_for_problems() -> None:
    report = HealthReport(
        checks=(
            CheckResult("good", "ok", "all good"),
            CheckResult("bad", "fail", "broken", remedy="fix the thing"),
        )
    )
    output = render_report(report)
    assert "PASS" in output
    assert "FAIL" in output
    assert "fix: fix the thing" in output
    assert "UNHEALTHY" in output
    assert output.index("FAIL") > output.index("PASS"), "failures should sort last for readability"


def test_render_report_counts_warnings_without_failing() -> None:
    report = HealthReport(
        checks=(
            CheckResult("a", "ok", ""),
            CheckResult("b", "warn", ""),
        )
    )
    output = render_report(report)
    assert "1 warning(s)" in output
    assert "HEALTHY" in output
    assert report.exit_code == 0
