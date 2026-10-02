"""Filesystem layout convention tests.

Covers the Phase 1 requirement for data / model / experiment directory
conventions.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from energy_intelligence.paths import REQUIRED_DIRECTORIES, ProjectPaths


def test_layout_is_rooted_where_requested(tmp_path: Path) -> None:
    paths = ProjectPaths.from_root(tmp_path)
    assert paths.root == tmp_path.resolve()
    assert paths.raw_data == tmp_path.resolve() / "data" / "raw"
    assert paths.processed_data == tmp_path.resolve() / "data" / "processed"
    assert paths.external_data == tmp_path.resolve() / "data" / "external"
    assert paths.models == tmp_path.resolve() / "models"
    assert paths.experiments == tmp_path.resolve() / "experiments"
    assert paths.artifacts == tmp_path.resolve() / "artifacts"
    assert paths.logs == tmp_path.resolve() / "logs"
    assert paths.configs == tmp_path.resolve() / "configs"


def test_required_directories_cover_the_documented_conventions(tmp_path: Path) -> None:
    paths = ProjectPaths.from_root(tmp_path)
    relative = {path.relative_to(paths.root).as_posix() for path in paths.required_directories}
    assert relative == set(REQUIRED_DIRECTORIES)


def test_tracked_directories_exclude_logs(tmp_path: Path) -> None:
    """logs/ is a pure output sink and is not tracked in git."""
    paths = ProjectPaths.from_root(tmp_path)
    assert paths.logs not in paths.tracked_directories
    assert paths.logs in paths.required_directories


def test_missing_reports_absent_directories(tmp_path: Path) -> None:
    paths = ProjectPaths.from_root(tmp_path)
    assert set(paths.missing()) == set(paths.required_directories)


def test_ensure_creates_everything_and_reports_what_it_made(tmp_path: Path) -> None:
    paths = ProjectPaths.from_root(tmp_path)

    created = paths.ensure()
    assert set(created) == set(paths.required_directories)
    assert paths.missing() == ()

    # Idempotent.
    assert paths.ensure() == ()


def test_ensure_is_safe_when_a_parent_is_absent(tmp_path: Path) -> None:
    paths = ProjectPaths.from_root(tmp_path / "fresh")
    paths.ensure()
    assert (tmp_path / "fresh" / "data" / "raw").is_dir()


def test_experiment_dir_is_named_by_experiment(tmp_path: Path) -> None:
    paths = ProjectPaths.from_root(tmp_path)
    assert paths.experiment_dir("run-7") == paths.experiments / "run-7"


def test_real_repository_has_the_layout() -> None:
    """A fresh clone must contain the documented directories."""
    paths = ProjectPaths.from_root()
    missing = paths.missing()
    assert missing == (), f"missing directories in the checkout: {missing}"


def test_default_paths_is_usable() -> None:
    from energy_intelligence.paths import DEFAULT_PATHS

    assert DEFAULT_PATHS.root.is_dir()
    assert DEFAULT_PATHS.required_directories


def test_paths_is_immutable(tmp_path: Path) -> None:
    paths = ProjectPaths.from_root(tmp_path)
    with pytest.raises(Exception):
        paths.root = tmp_path  # type: ignore[misc]
