"""The on-disk expert forecast cache.

``scripts/phase7_experts.py`` runs three models over 359,040 rows and writes the result
here. Everything downstream - diversity, oracle, router training, evaluation - reads
this file and nothing else, so every number in the phase comes from one identical set of
forecasts. Without that, "the router beat the GBM" could mean "the router and the GBM
were scored on different rows", and there is no way to tell from a table.

The cache is a convenience, not a source of truth: :func:`experts.build_experts`
reproduces it, and the experiment can be run without one.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .experts import EvaluationPanel

__all__ = ["ExpertCache", "save_expert_cache", "load_expert_cache", "panel_from_cache"]


@dataclass(frozen=True, slots=True)
class ExpertCache:
    """One set of expert forecasts plus the panel they were made on."""

    forecasts: np.ndarray
    actual_kw: np.ndarray
    persistence_kw: np.ndarray
    row_scale: np.ndarray
    origins: np.ndarray
    series: np.ndarray
    index_rows: np.ndarray
    horizons: tuple[int, ...]
    expert_names: tuple[str, ...]
    split_names: tuple[str, ...]
    split_bounds: np.ndarray
    metadata: dict[str, Any]

    @property
    def n_experts(self) -> int:
        return int(self.forecasts.shape[0])

    @property
    def n_rows(self) -> int:
        return int(self.forecasts.shape[1])

    def panel(self) -> EvaluationPanel:
        """The :class:`EvaluationPanel` view of this cache."""
        return panel_from_cache(self)

    def split_positions(self, name: str) -> np.ndarray:
        """Panel positions belonging to ``name``."""
        if name not in self.split_names:
            raise KeyError(f"unknown split {name!r}; have {list(self.split_names)}")
        index = self.split_names.index(name)
        start, stop = (int(v) for v in self.split_bounds[index])
        return np.arange(start, stop)

    def describe(self) -> dict[str, Any]:
        return {
            "n_experts": self.n_experts,
            "n_rows": self.n_rows,
            "experts": list(self.expert_names),
            "horizons": list(self.horizons),
            "splits": {
                name: int(self.split_bounds[i][1] - self.split_bounds[i][0])
                for i, name in enumerate(self.split_names)
            },
        }


def panel_from_cache(cache: ExpertCache) -> EvaluationPanel:
    """Rebuild the canonical evaluation panel from a cache."""
    split = np.empty(cache.n_rows, dtype=object)
    slices: dict[str, slice] = {}
    for index, name in enumerate(cache.split_names):
        start, stop = (int(v) for v in cache.split_bounds[index])
        split[start:stop] = name
        slices[name] = slice(start, stop)
    return EvaluationPanel(
        index_rows=cache.index_rows,
        split=split,
        split_slices=slices,
        origins=cache.origins,
        series=cache.series,
        row_scale=cache.row_scale,
        actual_kw=cache.actual_kw,
        persistence_kw=cache.persistence_kw,
        horizons=cache.horizons,
    )


def save_expert_cache(
    cache: ExpertCache, path: Path, *, metadata_path: Path | None = None
) -> Path:
    """Write the forecasts to ``path`` and the metadata beside it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        forecasts=cache.forecasts,
        actual_kw=cache.actual_kw,
        persistence_kw=cache.persistence_kw,
        row_scale=cache.row_scale,
        origins=cache.origins,
        series=cache.series,
        index_rows=cache.index_rows,
        horizons=np.asarray(cache.horizons),
        expert_names=np.asarray(cache.expert_names),
        split_slices=cache.split_bounds,
        split_names=np.asarray(cache.split_names),
    )
    target = Path(metadata_path) if metadata_path is not None else path.with_suffix(".json")
    target.write_text(json.dumps(cache.metadata, indent=2, sort_keys=True), encoding="utf-8")
    return path


def load_expert_cache(path: Path) -> ExpertCache:
    """Read an expert cache written by :func:`save_expert_cache`.

    Args:
        path: The ``.npz`` file.

    Returns:
        The cache, with the sibling ``.json`` metadata attached if it exists.

    Raises:
        ValueError: If the file is not an expert cache.
    """
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"expert cache not found: {path}")
    payload = np.load(path, allow_pickle=False)
    required = {
        "forecasts",
        "actual_kw",
        "persistence_kw",
        "row_scale",
        "origins",
        "series",
        "index_rows",
        "horizons",
        "expert_names",
        "split_slices",
        "split_names",
    }
    missing = sorted(required - set(payload.files))
    if missing:
        raise ValueError(f"{path} is not an expert cache: missing {missing}")
    metadata_path = path.with_suffix(".json")
    metadata: dict[str, Any] = {}
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return ExpertCache(
        forecasts=payload["forecasts"],
        actual_kw=payload["actual_kw"],
        persistence_kw=payload["persistence_kw"],
        row_scale=payload["row_scale"],
        origins=payload["origins"],
        series=payload["series"],
        index_rows=payload["index_rows"],
        horizons=tuple(int(h) for h in payload["horizons"]),
        expert_names=tuple(str(name) for name in payload["expert_names"]),
        split_names=tuple(str(name) for name in payload["split_names"]),
        split_bounds=payload["split_slices"],
        metadata=metadata,
    )