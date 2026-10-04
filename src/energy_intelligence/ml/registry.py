"""The experiment registry: a reproducible record of every baseline run.

What is recorded
----------------
Enough to re-run the experiment without reading the code: dataset version, target,
feature names, lookback, horizons, split fractions, model name, hyperparameters,
seed, host, library versions, training duration and the metric values.

What is deliberately **not** recorded
------------------------------------
Model weights. A fitted baseline is a derived artifact, regenerable from its
configuration, and the repository refuses to commit binaries (``.gitignore`` keeps
``models/`` and ``experiments/`` empty except for ``.gitkeep``). The checkpoint path
is recorded so a re-run can be compared against it locally.

The registry is a JSON Lines file so it appends without rewriting history, and it
lives in ``experiments/registry.jsonl``, which is the one file in that directory
that git tracks.
"""

from __future__ import annotations

import json
import platform
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

__all__ = [
    "ExperimentRecord",
    "REGISTRY_FILENAME",
    "append_record",
    "read_records",
    "environment",
    "now_iso",
]

REGISTRY_FILENAME = "registry.jsonl"


@dataclass(frozen=True, slots=True)
class ExperimentRecord:
    """One baseline experiment, in full.

    Attributes:
        experiment_id: Stable id, ``<target>-<dataset_version>-<model>-h<horizon>``.
        created_at: ISO-8601 local timestamp.
        target_id: Which declared target was forecast.
        dataset_version: The dataset artifact version trained on.
        dataset_sha256: Digest of the dataset's array file, for tamper evidence.
        feature_names: Exact feature columns, in order.
        lookback_steps: History available per sample.
        horizon_steps: Forecast horizon this result describes.
        split_fractions: Train/validation fractions actually used.
        split_counts: Rows per split actually used.
        model: Model name.
        model_family: ``naive``, ``classical`` or ``neural``.
        hyperparameters: Hyperparameters used.
        seed: Seed used, or ``None`` for a deterministic unfitted rule.
        train_seconds: Training duration, ``0.0`` for unfitted rules.
        predict_seconds: Inference duration.
        metrics: Metric values.
        artifacts: Paths written by this run.
        environment: Host and library versions.
        notes: Anything a reader must know to interpret the numbers.
    """

    experiment_id: str
    created_at: str
    target_id: str
    dataset_version: str
    dataset_sha256: str
    feature_names: tuple[str, ...]
    lookback_steps: int
    horizon_steps: int
    split_fractions: dict[str, float]
    split_counts: dict[str, int]
    model: str
    model_family: str
    hyperparameters: dict[str, Any]
    seed: int | None
    train_seconds: float
    predict_seconds: float
    metrics: dict[str, Any]
    artifacts: dict[str, str] = field(default_factory=dict)
    environment: dict[str, str] = field(default_factory=dict)
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["feature_names"] = list(self.feature_names)
        payload["notes"] = list(self.notes)
        payload["artifacts"] = {
            name: _project_relative(path) for name, path in self.artifacts.items()
        }
        return payload

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ExperimentRecord:
        data = dict(payload)
        data["feature_names"] = tuple(data.get("feature_names", ()))
        data["notes"] = tuple(data.get("notes", ()))
        return cls(**data)


def _project_relative(path: str) -> str:
    """Rewrite a path under the project root as a root-relative POSIX path.

    The registry is committed and read on other machines, so an absolute path is worse than
    useless there: it records the author's directory layout, it does not resolve, and it
    makes the file look machine-specific when the only machine-specific thing about it is
    the prefix. Paths outside the project root are left alone rather than mangled - an
    absolute path there is genuinely the only correct description.

    Args:
        path: The recorded path, absolute or already relative.

    Returns:
        The path relative to the project root when it lies inside it, else unchanged.
    """
    candidate = Path(path)
    if not candidate.is_absolute():
        return path
    try:
        from ..config.loader import PROJECT_ROOT

        return candidate.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except (ImportError, ValueError):
        return path


def environment() -> dict[str, str]:
    """Host and library versions, recorded because results depend on them."""
    versions: dict[str, str] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
    }
    for name in ("numpy", "sklearn", "torch"):
        try:
            module = __import__(name)
            versions[name] = str(getattr(module, "__version__", "unknown"))
        except Exception:  # noqa: BLE001 - a missing library is worth recording, not raising
            versions[name] = "not installed"
    return versions


def experiment_id(
    *, target_id: str, dataset_version: str, model: str, horizon: int
) -> str:
    """Deterministic id, so re-running an experiment overwrites rather than forks."""
    return f"{target_id}-{dataset_version}-{model}-h{horizon:03d}"


def append_record(path: Path, record: ExperimentRecord) -> Path:
    """Append one record, replacing any earlier record with the same id.

    Re-running an experiment should update the registry, not accumulate duplicates
    that make the history look richer than it is.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: list[ExperimentRecord] = []
    if path.is_file():
        existing = read_records(path)
    kept = [item for item in existing if item.experiment_id != record.experiment_id]
    kept.append(record)
    kept.sort(key=lambda item: item.experiment_id)
    payload = "\n".join(item.to_json() for item in kept) + "\n"
    path.write_text(payload, encoding="utf-8")
    return path


def read_records(path: Path) -> list[ExperimentRecord]:
    """Read every record. A malformed line is reported, never silently skipped."""
    if not path.is_file():
        return []
    records: list[ExperimentRecord] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(ExperimentRecord.from_dict(json.loads(line)))
        except (json.JSONDecodeError, TypeError) as exc:
            raise ValueError(f"{path}:{number}: malformed registry record: {exc}") from exc
    return records


def now_iso() -> str:
    """Local timestamp, used for records that are not part of a hash."""
    return datetime.now().astimezone().isoformat(timespec="seconds")