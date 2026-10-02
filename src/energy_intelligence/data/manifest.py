"""Dataset manifest: what was acquired, from where, and with what integrity.

Reproducibility requires answering "which bytes produced this result?" long
after the run. The manifest records the dataset identity, the exact remote
location of every acquired file, its size, and its SHA-256 digest.

Design notes
------------
* **Deterministic except for one documented field.** ``acquired_at`` is the only
  nondeterministic field; it is isolated in ``AcquisitionMetadata`` so a caller
  can compare manifests for equality while ignoring it. Everything else -- file
  list, sizes, digests, schema -- is a pure function of the downloaded bytes.
* **No secrets.** The OEDI bucket is public and unauthenticated, so no
  credentials exist to record. The manifest asserts it carries none.
* **Digests are computed, not claimed.** ``sha256`` is calculated from the bytes
  on disk. Where the authoritative source publishes its own digest, record it in
  ``source_checksums`` so a mismatch is detectable rather than assumed.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path

__all__ = [
    "FileEntry",
    "AcquisitionMetadata",
    "SchemaSummary",
    "DatasetManifest",
    "MANIFEST_VERSION",
    "compute_sha256",
]

MANIFEST_VERSION = "1.0.0-phase3"

_CHUNK = 1024 * 1024


def compute_sha256(path: Path) -> str:
    """Compute the SHA-256 digest of a file, streaming to bound memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True, order=True)
class FileEntry:
    """One acquired file.

    Attributes:
        remote_key: Key within the authoritative bucket.
        size_bytes: Size on disk.
        sha256: Digest computed from the local bytes.
        role: What the file is for, e.g. ``"loads"``, ``"profiles"``.
    """

    remote_key: str
    size_bytes: int
    sha256: str
    role: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "remote_key": self.remote_key,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "role": self.role,
        }


@dataclass(frozen=True, slots=True)
class AcquisitionMetadata:
    """Nondeterministic acquisition facts, isolated from the deterministic body.

    Attributes:
        acquired_at: ISO-8601 UTC timestamp of acquisition.
        source: Authoritative source URL prefix.
        method: How the bytes were obtained, e.g. ``"https_get"``.
        tool: The fetching tool used.
    """

    acquired_at: str
    source: str
    method: str = "https_get"
    tool: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "acquired_at": self.acquired_at,
            "source": self.source,
            "method": self.method,
            "tool": self.tool,
        }


@dataclass(frozen=True, slots=True)
class SchemaSummary:
    """Observed schema facts for one file kind.

    Attributes:
        kind: Logical file kind, e.g. ``"Loads.dss"``.
        element_classes: Element class to count.
        parameter_names: Union of parameter names seen.
        rows: Row count where meaningful.
        notes: Free-form observations, e.g. "values are per-unit".
    """

    kind: str
    element_classes: dict[str, int] = field(default_factory=dict)
    parameter_names: tuple[str, ...] = ()
    rows: int | None = None
    notes: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "element_classes": dict(sorted(self.element_classes.items())),
            "parameter_names": list(self.parameter_names),
            "rows": self.rows,
            "notes": self.notes,
        }


@dataclass(frozen=True, slots=True)
class DatasetManifest:
    """Machine-readable record of one dataset acquisition.

    Attributes:
        dataset_name: Dataset name, e.g. ``"SMART-DS"``.
        dataset_version: Version as published in ``SMART-DS_version.txt``.
        subset: Human-readable description of the acquired subset.
        acquisition: When and how the bytes were obtained.
        files: Deterministically ordered file entries.
        schema: Observed schema summaries.
        source_checksums: Digests published by the authoritative source, if any.
        license: License as stated by the authoritative source, or ``"UNKNOWN"``.
        transformation_version: Version of the raw-to-processed transform applied.
        local_root: Where the bytes live locally.
    """

    dataset_name: str
    dataset_version: str
    subset: str
    acquisition: AcquisitionMetadata
    files: tuple[FileEntry, ...]
    local_root: str
    schema: tuple[SchemaSummary, ...] = ()
    source_checksums: dict[str, str] = field(default_factory=dict)
    license: str = "UNKNOWN"
    transformation_version: str = "TBD"

    # ------------------------------------------------------------------

    @property
    def manifest_version(self) -> str:
        return MANIFEST_VERSION

    @property
    def total_bytes(self) -> int:
        return sum(entry.size_bytes for entry in self.files)

    @property
    def file_count(self) -> int:
        return len(self.files)

    def digest(self) -> str:
        """A single digest over the deterministic body.

        Excludes :attr:`acquisition` so two acquisitions of identical bytes
        produce an identical digest regardless of when they happened. This is
        the value to compare across machines and checkouts.
        """
        material = "|".join(
            [
                MANIFEST_VERSION,
                self.dataset_name,
                self.dataset_version,
                self.transformation_version,
                *(f"{e.remote_key}:{e.size_bytes}:{e.sha256}" for e in sorted(self.files)),
            ]
        )
        return hashlib.sha256(material.encode("utf-8")).hexdigest()

    def entry(self, remote_key: str) -> FileEntry | None:
        for entry in self.files:
            if entry.remote_key == remote_key:
                return entry
        return None

    def files_with_role(self, role: str) -> tuple[FileEntry, ...]:
        return tuple(entry for entry in self.files if entry.role == role)

    def to_dict(self) -> dict[str, object]:
        return {
            "_manifest_version": MANIFEST_VERSION,
            "dataset_name": self.dataset_name,
            "dataset_version": self.dataset_version,
            "subset": self.subset,
            "license": self.license,
            "local_root": self.local_root,
            "transformation_version": self.transformation_version,
            "content_digest": self.digest(),
            "acquisition": self.acquisition.to_dict(),
            "file_count": self.file_count,
            "total_bytes": self.total_bytes,
            "files": [entry.to_dict() for entry in sorted(self.files)],
            "schema": [item.to_dict() for item in self.schema],
            "source_checksums": dict(sorted(self.source_checksums.items())),
        }

    def summary_line(self) -> str:
        return (
            f"{self.dataset_name} v{self.dataset_version} | {self.file_count} files | "
            f"{self.total_bytes / (1024 * 1024):.1f} MiB | digest {self.digest()[:12]} | "
            f"{self.subset}"
        )

    def with_schema(self, schema: tuple[SchemaSummary, ...]) -> DatasetManifest:
        return replace(self, schema=schema)

    def with_transformation(self, version: str) -> DatasetManifest:
        return replace(self, transformation_version=version)

    @staticmethod
    def utc_now() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")
