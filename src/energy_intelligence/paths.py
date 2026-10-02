"""Canonical filesystem layout.

Phase 2-19 will read and write datasets, model checkpoints and experiment
artifacts. Fixing the layout now - in one place, as constants derived from the
project root - means later phases never invent their own ad-hoc paths.

Naming rules (binding for all future phases):

===========================  =========================================================
``data/raw/``                Immutable source data exactly as obtained. Never written
                             to by preprocessing.
``data/processed/``          Deterministic transforms of ``data/raw/``. Regenerable,
                             so not committed to git.
``data/external/``           Third-party or manually curated inputs not produced by
                             this project's own pipeline.
``models/``                  Model checkpoints and weights.
``experiments/``             One directory per experiment run, named by
                             ``<experiment_name>``.
``artifacts/``               Machine-readable outputs of a run: metrics, tables,
                             reports, figures.
``logs/``                    JSON Lines logs written by
                             :mod:`energy_intelligence.logging_setup`.
``configs/``                 Versioned TOML configuration.
===========================  =========================================================

Example:
    >>> from energy_intelligence.paths import ProjectPaths
    >>> paths = ProjectPaths.from_root()
    >>> paths.raw_data.parts[-2:]
    ('data', 'raw')
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import PROJECT_ROOT

__all__ = ["ProjectPaths", "REQUIRED_DIRECTORIES"]


@dataclass(frozen=True, slots=True)
class ProjectPaths:
    """Resolved locations of every project directory."""

    root: Path
    configs: Path
    logs: Path
    models: Path
    experiments: Path
    artifacts: Path
    raw_data: Path
    processed_data: Path
    external_data: Path

    @classmethod
    def from_root(cls, root: Path | str | None = None) -> ProjectPaths:
        """Build the layout from ``root``, defaulting to the detected project root."""
        base = Path(root).resolve() if root is not None else PROJECT_ROOT
        return cls(
            root=base,
            configs=base / "configs",
            logs=base / "logs",
            models=base / "models",
            experiments=base / "experiments",
            artifacts=base / "artifacts",
            raw_data=base / "data" / "raw",
            processed_data=base / "data" / "processed",
            external_data=base / "data" / "external",
        )

    @property
    def tracked_directories(self) -> tuple[Path, ...]:
        """Directories git must see so the layout survives a fresh clone.

        ``logs/`` is excluded: it is a pure output sink and is fully
        reproducible by running the project.
        """
        return (
            self.configs,
            self.models,
            self.experiments,
            self.artifacts,
            self.raw_data,
            self.processed_data,
            self.external_data,
        )

    @property
    def required_directories(self) -> tuple[Path, ...]:
        """Every directory that must exist for the project to function."""
        return self.tracked_directories + (self.logs,)

    def experiment_dir(self, experiment_name: str) -> Path:
        """Return the directory for ``experiment_name``.

        The name is assumed pre-validated by
        :func:`~energy_intelligence.config.coerce_app_config`, which restricts
        it to a slug-safe character set.
        """
        return self.experiments / experiment_name

    def missing(self) -> tuple[Path, ...]:
        """Return the required directories that do not currently exist."""
        return tuple(path for path in self.required_directories if not path.is_dir())

    def ensure(self) -> tuple[Path, ...]:
        """Create every missing required directory.

        Returns:
            The directories that were created, in creation order.
        """
        created: list[Path] = []
        for path in self.required_directories:
            if not path.is_dir():
                path.mkdir(parents=True, exist_ok=True)
                created.append(path)
        return tuple(created)


#: Convenience module-level view for callers that do not need a custom root.
DEFAULT_PATHS: ProjectPaths = ProjectPaths.from_root()

#: Directories a fresh clone must contain, relative to the project root.
REQUIRED_DIRECTORIES: tuple[str, ...] = (
    "configs",
    "data/raw",
    "data/processed",
    "data/external",
    "models",
    "experiments",
    "artifacts",
    "logs",
)
