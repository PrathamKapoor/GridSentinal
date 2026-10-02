"""Shared pytest fixtures.

Design notes:

* Every test runs against a temporary directory so tests never write into the
  real repository and never depend on the real ``configs/default.toml`` values.
* ``reset_logging`` runs after every test because
  :func:`energy_intelligence.logging_setup.initialize_logging` mutates
  process-wide logger state.
* :class:`_Remove` lets a factory *delete* a key, which is how "missing required
  key" cases are expressed without duplicating the sample table.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from energy_intelligence.config import coerce_app_config
from energy_intelligence.logging_setup import reset_logging
from energy_intelligence.paths import ProjectPaths


class _Remove:
    """Sentinel: delete this key instead of setting it."""


#: A complete, valid raw configuration table. Tests override single keys.
SAMPLE_CONFIG: dict[str, Any] = {
    "seed": 7,
    "device": "cpu",
    "model_path": "models/test-model",
    "dataset_path": "data/processed",
    "experiment_name": "test-run",
    "log_level": "INFO",
    "output_dir": "artifacts",
    "artifact_dir": "artifacts",
    "log_dir": "logs",
}


def _apply(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    table = dict(base)
    for key, value in overrides.items():
        if isinstance(value, _Remove):
            table.pop(key, None)
        else:
            table[key] = value
    return table


@pytest.fixture(autouse=True)
def _isolate_logging() -> Iterator[None]:
    """Guarantee no test leaks logger handlers or run identity into the next."""
    reset_logging()
    yield
    reset_logging()


@pytest.fixture
def sample_config() -> dict[str, Any]:
    """Return a fresh copy of the valid sample configuration."""
    return dict(SAMPLE_CONFIG)


@pytest.fixture
def config_table(sample_config: dict[str, Any]):
    """Return a factory producing a raw config table with overrides applied."""
    return lambda **overrides: _apply(sample_config, overrides)


@pytest.fixture
def remove_key() -> _Remove:
    """Sentinel instance accepted by :func:`config_table` to delete a key.

    An *instance*, not the class: ``isinstance(_Remove, _Remove)`` is False, so
    passing the class would set the key instead of deleting it.
    """
    return _Remove()


@pytest.fixture
def tmp_paths(tmp_path: Path) -> ProjectPaths:
    """A :class:`ProjectPaths` rooted in ``tmp_path`` with all directories created."""
    paths = ProjectPaths.from_root(tmp_path)
    paths.ensure()
    return paths


@pytest.fixture
def make_config(tmp_paths: ProjectPaths, sample_config: dict[str, Any]):
    """Return a factory building a validated config rooted at ``tmp_path``."""

    def _make(**overrides: Any):
        return coerce_app_config(
            _apply(sample_config, overrides),
            project_root=tmp_paths.root,
            source="<test>",
        )

    return _make


@pytest.fixture
def write_toml(tmp_path: Path):
    """Return a factory writing a UTF-8 TOML file into ``tmp_path``."""

    def _write(filename: str, contents: str) -> Path:
        path = tmp_path / filename
        path.write_text(contents, encoding="utf-8")
        return path

    return _write
