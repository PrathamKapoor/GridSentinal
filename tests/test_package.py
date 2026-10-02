"""Package import and public-surface tests.

Requirement 1 of the Phase 1 brief: the package can be imported.
"""

from __future__ import annotations

import importlib
import subprocess
import sys

import pytest

import energy_intelligence


def test_package_imports_and_exposes_version() -> None:
    assert energy_intelligence.__version__ == "0.1.0"


def test_package_has_a_file() -> None:
    assert energy_intelligence.__file__ is not None
    assert energy_intelligence.__file__.endswith("__init__.py")


def test_public_api_is_importable() -> None:
    for name in energy_intelligence.__all__:
        assert hasattr(energy_intelligence, name), f"__all__ advertises missing name {name!r}"


def test_submodules_import_cleanly() -> None:
    for name in (
        "energy_intelligence.cli",
        "energy_intelligence.config",
        "energy_intelligence.config.loader",
        "energy_intelligence.config.schema",
        "energy_intelligence.health",
        "energy_intelligence.logging_setup",
        "energy_intelligence.paths",
    ):
        assert importlib.import_module(name) is not None


def test_package_runs_as_a_module() -> None:
    """``python -m energy_intelligence`` must work, not just the console script."""
    result = subprocess.run(
        [sys.executable, "-m", "energy_intelligence", "--help"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert "energy-intel" in result.stdout


def test_no_ai_or_ml_dependencies_are_imported_at_import_time() -> None:
    """Phase 1 must not pull in any ML framework.

    This is a regression guard on the zero-dependency decision (D-004): if a
    future phase imports torch, transformers or similar at module scope, this
    test fails and forces the decision to be recorded deliberately.

    Runs in a subprocess so the assertion depends only on importing the
    package, not on what other tests in this session already loaded.
    """
    program = (
        "import sys, energy_intelligence;"
        "forbidden = ('torch','transformers','numpy','pandas','sklearn','scipy');"
        "leaked = [m for m in forbidden if m in sys.modules];"
        "print(','.join(leaked))"
    )
    result = subprocess.run(
        [sys.executable, "-c", program],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    leaked = result.stdout.strip()
    assert leaked == "", f"ML dependencies imported at import time: {leaked}"


def test_runtime_dependencies_are_empty() -> None:
    """Guard the declared dependency surface in pyproject.toml."""
    try:
        import tomllib
    except ModuleNotFoundError:  # pragma: no cover - requires-python is >=3.11
        pytest.skip("tomllib unavailable")

    from pathlib import Path

    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    assert data["project"]["dependencies"] == []
