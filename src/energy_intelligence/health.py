"""Phase 1 environment health check.

Verifies the five foundations Phase 1 is responsible for:

1. the interpreter satisfies ``requires-python``;
2. the ``energy_intelligence`` package imports and resolves inside this checkout;
3. configuration loads and validates;
4. the canonical project directories exist;
5. test infrastructure and a writable log sink are available.

Each check returns a :class:`CheckResult` so a failure identifies the specific
foundation that is broken rather than reporting one opaque error.

Example:
    >>> from energy_intelligence.health import check_health
    >>> report = check_health()
    >>> report.ok
    True
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import sys
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path

from .config import AppConfig, load_config
from .logging_setup import get_logger
from .paths import REQUIRED_DIRECTORIES, ProjectPaths

__all__ = [
    "CheckResult",
    "HealthReport",
    "check_health",
    "check_python_version",
    "check_package_import",
    "check_config",
    "check_directories",
    "check_test_infrastructure",
    "check_logging",
    "render_report",
]

logger = get_logger(__name__)

_SEVERITY_ORDER = {"ok": 0, "warn": 1, "fail": 2}


@dataclass(frozen=True, slots=True)
class CheckResult:
    """Outcome of a single health check."""

    name: str
    status: str
    detail: str
    remedy: str | None = None

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    @property
    def failed(self) -> bool:
        return self.status == "fail"

    @property
    def sort_key(self) -> int:
        return _SEVERITY_ORDER.get(self.status, 99)


@dataclass(frozen=True, slots=True)
class HealthReport:
    """Aggregated result of every Phase 1 health check."""

    checks: tuple[CheckResult, ...]
    config: AppConfig | None = None

    @property
    def ok(self) -> bool:
        """True only when no check failed. Warnings do not fail the report."""
        return not any(check.failed for check in self.checks)

    @property
    def exit_code(self) -> int:
        """0 when healthy, 1 otherwise. Suitable for CI gating."""
        return 0 if self.ok else 1

    def failures(self) -> tuple[CheckResult, ...]:
        return tuple(check for check in self.checks if check.failed)

    def warnings(self) -> tuple[CheckResult, ...]:
        return tuple(check for check in self.checks if check.status == "warn")


# --------------------------------------------------------------------------
# Individual checks
# --------------------------------------------------------------------------


def check_python_version(minimum: str = "3.12") -> CheckResult:
    """Verify the running interpreter meets the project's minimum version."""
    current = sys.version_info
    required = tuple(int(part) for part in minimum.split("."))

    if current[: len(required)] < required:
        return CheckResult(
            name="python_version",
            status="fail",
            detail=f"running {current.major}.{current.minor}, requires >={minimum}",
            remedy=f"Recreate the environment with Python {minimum} (see .python-version).",
        )

    return CheckResult(
        name="python_version",
        status="ok",
        detail=(
            f"Python {current.major}.{current.minor}.{current.micro} "
            f"({sys.executable})"
        ),
    )


def check_package_import() -> CheckResult:
    """Import the package and confirm it resolves inside this checkout."""
    try:
        module = importlib.import_module("energy_intelligence")
    except Exception as exc:  # noqa: BLE001 - report any import failure verbatim
        return CheckResult(
            name="package_import",
            status="fail",
            detail=f"import energy_intelligence failed: {exc}",
            remedy="Run 'uv sync' to install the project in editable mode.",
        )

    module_file = getattr(module, "__file__", None)
    if module_file is None:  # pragma: no cover - namespace package edge case
        return CheckResult(
            name="package_import",
            status="fail",
            detail="energy_intelligence has no __file__; namespace package?",
        )

    resolved = Path(module_file).resolve()
    if "site-packages" in resolved.parts:
        return CheckResult(
            name="package_import",
            status="warn",
            detail=f"importing from {resolved} (site-packages, not the checkout)",
            remedy="Run 'uv sync' to install the project in editable mode.",
        )

    version = getattr(module, "__version__", "unknown")
    return CheckResult(
        name="package_import",
        status="ok",
        detail=f"energy_intelligence {version} from {resolved.parent}",
    )


def check_config(config_path: Path | None = None) -> tuple[CheckResult, AppConfig | None]:
    """Load configuration and report the outcome.

    Returns:
        The check result and the loaded config, so the caller does not have to
        load it twice.
    """
    try:
        config = load_config(config_path)
    except Exception as exc:  # noqa: BLE001 - surface any config failure verbatim
        return (
            CheckResult(
                name="config",
                status="fail",
                detail=f"{type(exc).__name__}: {exc}",
                remedy="Check configs/default.toml, or pass --config <path>.",
            ),
            None,
        )

    return (
        CheckResult(
            name="config",
            status="ok",
            detail=(
                f"experiment={config.experiment_name} seed={config.seed} "
                f"device={config.device} log_level={config.log_level} "
                f"fingerprint={config.fingerprint()}"
            ),
        ),
        config,
    )


def check_directories(paths: ProjectPaths | None = None) -> CheckResult:
    """Verify every canonical project directory exists."""
    resolved = paths or ProjectPaths.from_root()
    missing = resolved.missing()

    if missing:
        return CheckResult(
            name="directories",
            status="fail",
            detail=f"missing: {', '.join(str(path.relative_to(resolved.root)) for path in missing)}",
            remedy="Run 'energy-intel init-dirs' to create the standard layout.",
        )

    return CheckResult(
        name="directories",
        status="ok",
        detail=f"all {len(resolved.required_directories)} directories present",
    )


def check_test_infrastructure(paths: ProjectPaths | None = None) -> CheckResult:
    """Verify pytest is installed, the tests directory exists and tests are collected.

    Args:
        paths: Layout to inspect. Defaults to the detected project root; tests
            pass a temporary layout to exercise the failure branches without
            touching the real checkout.
    """
    if importlib.util.find_spec("pytest") is None:
        return CheckResult(
            name="test_infrastructure",
            status="fail",
            detail="pytest is not installed in this environment",
            remedy="Run 'uv sync --extra dev'.",
        )

    try:
        version = metadata.version("pytest")
    except metadata.PackageNotFoundError:  # pragma: no cover - find_spec succeeded
        version = "unknown"

    resolved = paths or ProjectPaths.from_root()
    tests_dir = resolved.root / "tests"
    if not tests_dir.is_dir():
        return CheckResult(
            name="test_infrastructure",
            status="fail",
            detail=f"pytest {version} installed but {tests_dir} does not exist",
            remedy="Restore the tests/ directory from version control.",
        )

    test_files = sorted(tests_dir.glob("test_*.py"))
    if not test_files:
        return CheckResult(
            name="test_infrastructure",
            status="fail",
            detail=f"pytest {version} installed but no test_*.py files found in tests/",
            remedy="Add tests, or confirm they were not lost.",
        )

    return CheckResult(
        name="test_infrastructure",
        status="ok",
        detail=f"pytest {version}, {len(test_files)} test module(s) in tests/",
    )


def check_logging(config: AppConfig | None) -> CheckResult:
    """Verify a log file can actually be written, and emit a probe record."""
    from .logging_setup import initialize_logging, reset_logging

    if config is None:
        return CheckResult(
            name="logging",
            status="warn",
            detail="skipped: configuration failed to load, so logging cannot be configured",
        )

    try:
        log_file = initialize_logging(config, console=False, to_file=True, force=True)
    except Exception as exc:  # noqa: BLE001 - report any logging failure verbatim
        return CheckResult(
            name="logging",
            status="fail",
            detail=f"logging setup failed: {type(exc).__name__}: {exc}",
            remedy=f"Ensure the log directory is writable: {config.log_dir}",
        )

    if log_file is None:
        return CheckResult(
            name="logging",
            status="fail",
            detail=f"no log file could be opened under {config.log_dir}",
            remedy=f"Create the directory and check permissions: {config.log_dir}",
        )

    try:
        probe = get_logger("health_check")
        probe.info(
            "health check probe",
            extra={"event": "health_check_probe", "component": "health"},
        )
        for handler in logging.getLogger("energy_intelligence").handlers:
            handler.flush()
        size = log_file.stat().st_size
    except Exception as exc:  # noqa: BLE001 - report write failures verbatim
        return CheckResult(
            name="logging",
            status="fail",
            detail=f"could not write to {log_file}: {type(exc).__name__}: {exc}",
            remedy="Check free disk space and permissions on the logs directory.",
        )
    finally:
        reset_logging()

    if size == 0:
        return CheckResult(
            name="logging",
            status="fail",
            detail=f"log file {log_file} was created but stayed empty",
            remedy="Check that log level is not set above INFO and handlers are attached.",
        )

    return CheckResult(
        name="logging",
        status="ok",
        detail=f"JSON Lines log writable at {log_file} ({size} bytes after probe)",
    )


# --------------------------------------------------------------------------
# Aggregate
# --------------------------------------------------------------------------


def check_health(
    config_path: Path | str | None = None,
    *,
    paths: ProjectPaths | None = None,
    create_missing_dirs: bool = False,
) -> HealthReport:
    """Run every Phase 1 health check.

    Args:
        config_path: Optional explicit config file.
        paths: Optional custom :class:`ProjectPaths`, for testing against a
            temporary directory.
        create_missing_dirs: Create any missing canonical directories before
            checking them, instead of reporting them as failures.

    Returns:
        A :class:`HealthReport` with one result per check plus the loaded config
        when it could be loaded.
    """
    resolved_paths = paths or ProjectPaths.from_root()

    if create_missing_dirs:
        resolved_paths.ensure()

    config_result, config = check_config(Path(config_path) if config_path else None)

    checks = (
        check_python_version(),
        check_package_import(),
        config_result,
        check_directories(resolved_paths),
        check_test_infrastructure(resolved_paths),
        check_logging(config),
    )

    report = HealthReport(checks=checks, config=config)
    logger.debug(
        "health check complete",
        extra={
            "event": "health_check_complete",
            "ok": report.ok,
            "failures": len(report.failures()),
            "warnings": len(report.warnings()),
        },
    )
    return report


_STATUS_LABEL = {"ok": "PASS", "warn": "WARN", "fail": "FAIL"}


def render_report(report: HealthReport) -> str:
    """Render a :class:`HealthReport` as plain text.

    Intentionally dependency-free so it can be printed in a terminal, captured
    in a CI log, or written to a file unchanged.
    """
    lines = [
        "Energy Intelligence - Phase 1 health check",
        "=" * 52,
    ]

    for check in sorted(report.checks, key=lambda item: item.sort_key):
        lines.append(f"[{_STATUS_LABEL.get(check.status, check.status.upper()):4}] "
                     f"{check.name}")
        lines.append(f"       {check.detail}")
        if check.remedy and check.status != "ok":
            lines.append(f"       fix: {check.remedy}")

    passed = sum(1 for check in report.checks if check.status == "ok")
    warned = len(report.warnings())
    failed = len(report.failures())

    lines.append("-" * 52)
    lines.append(
        f"{passed} passed, {warned} warning(s), {failed} failed - "
        f"{'HEALTHY' if report.ok else 'UNHEALTHY'}"
    )

    if report.config is not None:
        lines.append(f"config fingerprint: {report.config.fingerprint()}")

    return "\n".join(lines)
