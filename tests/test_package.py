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


def test_runtime_dependencies_are_exactly_the_agreed_three() -> None:
    """Guard the declared dependency surface in pyproject.toml.

    Phases 1-3 shipped with **no** runtime dependencies, and that was enforced here.
    Phase 4 is the first phase that genuinely needs a numerical stack, so the list is
    no longer empty - but it is now *exact*. The rule is unchanged in spirit: every
    dependency must be justified by a phase that uses it, so an accidental or
    convenient addition still fails this test.
    """
    try:
        import tomllib
    except ModuleNotFoundError:  # pragma: no cover - requires-python is >=3.11
        pytest.skip("tomllib unavailable")

    from pathlib import Path

    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    names = sorted(
        spec.split(">=")[0].split("==")[0].split("[")[0].strip()
        for spec in data["project"]["dependencies"]
    )
    assert names == ["numpy", "scikit-learn", "torch"], (
        "Phase 4 adds exactly three runtime dependencies (decisions.md D-055); "
        "pandas, matplotlib and any other addition must be justified first"
    )


def test_pandas_is_not_used_anywhere_in_the_package() -> None:
    """pandas was deliberately not adopted; catch an accidental import."""
    import re
    from pathlib import Path

    package = Path(__file__).resolve().parents[1] / "src" / "energy_intelligence"
    pattern = re.compile(r"^\s*(?:import pandas|from pandas)", re.MULTILINE)
    offenders = [path.name for path in package.rglob("*.py") if pattern.search(path.read_text(encoding="utf-8"))]
    assert offenders == [], f"pandas must not be imported (decisions.md D-055): {offenders}"


def test_the_domain_layer_does_not_import_any_ml_library() -> None:
    """Phase 2's contract must stay independent of the modelling stack.

    ``energy_intelligence.domain`` is the vocabulary every later phase speaks. If it
    imported numpy or torch, the domain would no longer be a dependency-free contract
    and the Phase 1/2 guarantee would quietly end.
    """
    import re
    from pathlib import Path

    domain = (
        Path(__file__).resolve().parents[1] / "src" / "energy_intelligence" / "domain"
    )
    pattern = re.compile(r"^\s*(?:import|from)\s+(numpy|torch|sklearn|scipy|pandas)\b", re.MULTILINE)
    offenders = [path.name for path in domain.rglob("*.py") if pattern.search(path.read_text(encoding="utf-8"))]
    assert offenders == [], f"the domain layer must not import an ML library: {offenders}"
