"""Programmatic schema discovery.

Nothing here is hard-coded from documentation. Every reported fact is derived by
reading the real files, so a future SMART-DS revision that changes a column name
produces a different schema report rather than a silently wrong mapping.

The discovery answers the questions the Phase 3 brief asks of every relevant
file: name, type, columns, data types, row count, missing values, unique
identifiers, timestamp fields, units where determinable, and relationships.
"""

from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from .buscoords import parse_buscoords
from .dss import DssObject, DssParseError, parse_dss
from .layout import SmartDsLayout
from ..manifest import SchemaSummary

__all__ = [
    "ColumnProfile",
    "CsvSchema",
    "DssSchema",
    "ProfileSchema",
    "FeederSchema",
    "discover_csv_schema",
    "discover_dss_schema",
    "discover_profile_schema",
    "discover_buscoords_schema",
    "discover_feeder_schema",
]


@dataclass(frozen=True, slots=True)
class ColumnProfile:
    """Observed statistics for one column.

    Attributes:
        name: Column name as it appears in the file.
        inferred_type: ``"int"``, ``"float"``, ``"str"`` or ``"mixed"``.
        count: Non-empty values seen.
        missing: Empty or unparseable values seen.
        minimum: Smallest numeric value, or ``None``.
        maximum: Largest numeric value, or ``None``.
        distinct: Distinct value count, capped for reporting.
    """

    name: str
    inferred_type: str
    count: int
    missing: int
    minimum: float | None
    maximum: float | None
    distinct: int

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "inferred_type": self.inferred_type,
            "count": self.count,
            "missing": self.missing,
            "min": self.minimum,
            "max": self.maximum,
            "distinct": self.distinct,
        }


@dataclass(frozen=True, slots=True)
class CsvSchema:
    """Observed schema of a delimited text file."""

    path: str
    columns: tuple[ColumnProfile, ...]
    rows: int
    delimiter: str
    has_header: bool
    notes: str = ""

    @property
    def column_names(self) -> tuple[str, ...]:
        return tuple(column.name for column in self.columns)

    def column(self, name: str) -> ColumnProfile | None:
        for column in self.columns:
            if column.name == name:
                return column
        return None

    def to_summary(self) -> SchemaSummary:
        return SchemaSummary(
            kind=Path(self.path).name,
            element_classes={},
            parameter_names=self.column_names,
            rows=self.rows,
            notes=self.notes,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "delimiter": self.delimiter,
            "has_header": self.has_header,
            "rows": self.rows,
            "notes": self.notes,
            "columns": [column.to_dict() for column in self.columns],
        }


@dataclass(frozen=True, slots=True)
class DssSchema:
    """Observed schema of an OpenDSS element file."""

    path: str
    element_classes: dict[str, int]
    parameters_by_class: dict[str, tuple[str, ...]]
    total_elements: int
    notes: str = ""

    def to_summary(self) -> SchemaSummary:
        merged: set[str] = set()
        for names in self.parameters_by_class.values():
            merged.update(names)
        return SchemaSummary(
            kind=Path(self.path).name,
            element_classes=dict(sorted(self.element_classes.items())),
            parameter_names=tuple(sorted(merged)),
            rows=self.total_elements,
            notes=self.notes,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "total_elements": self.total_elements,
            "element_classes": dict(sorted(self.element_classes.items())),
            "parameters_by_class": {
                key: list(value) for key, value in sorted(self.parameters_by_class.items())
            },
            "notes": self.notes,
        }


@dataclass(frozen=True, slots=True)
class ProfileSchema:
    """Observed schema of a headerless per-unit load profile."""

    path: str
    rows: int
    minimum: float
    maximum: float
    is_normalised: bool
    notes: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "rows": self.rows,
            "min": self.minimum,
            "max": self.maximum,
            "is_normalised": self.is_normalised,
            "notes": self.notes,
        }


@dataclass(frozen=True, slots=True)
class FeederSchema:
    """Everything discovered about one feeder folder."""

    layout: SmartDsLayout
    loads: DssSchema | None = None
    load_shapes: DssSchema | None = None
    lines: DssSchema | None = None
    transformers: DssSchema | None = None
    pv_systems: DssSchema | None = None
    storage: DssSchema | None = None
    buscoords: dict[str, object] = field(default_factory=dict)
    metrics_rows: int = 0
    summary: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "layout": self.layout.describe(),
            "loads": self.loads.to_dict() if self.loads else None,
            "load_shapes": self.load_shapes.to_dict() if self.load_shapes else None,
            "lines": self.lines.to_dict() if self.lines else None,
            "transformers": self.transformers.to_dict() if self.transformers else None,
            "pv_systems": self.pv_systems.to_dict() if self.pv_systems else None,
            "storage": self.storage.to_dict() if self.storage else None,
            "buscoord_count": len(self.buscoords),
            "metrics_rows": self.metrics_rows,
            "notes": list(self.notes),
        }


# ----------------------------------------------------------------------
# CSV discovery
# ----------------------------------------------------------------------


def _profile_column(name: str, values: list[str]) -> ColumnProfile:
    non_empty = [v for v in values if v is not None and v.strip() != ""]
    missing = len(values) - len(non_empty)
    as_int: list[int] = []
    as_float: list[float] = []
    unparseable = 0
    for value in non_empty:
        text = value.strip()
        try:
            as_int.append(int(text))
        except ValueError:
            try:
                as_float.append(float(text))
            except ValueError:
                unparseable += 1
    numeric = [float(v) for v in as_int] + as_float
    if not non_empty:
        inferred = "empty"
    elif unparseable == 0 and as_float:
        inferred = "float"
    elif unparseable == 0 and as_int:
        inferred = "int"
    elif unparseable == len(non_empty):
        inferred = "str"
    else:
        inferred = "mixed"
    return ColumnProfile(
        name=name,
        inferred_type=inferred,
        count=len(non_empty),
        missing=missing,
        minimum=min(numeric) if numeric else None,
        maximum=max(numeric) if numeric else None,
        distinct=len({v.strip() for v in non_empty}),
    )


def discover_csv_schema(path: Path, *, sample_rows: int | None = None) -> CsvSchema:
    """Discover the schema of a CSV/TSV file from its actual content.

    Args:
        path: File to inspect.
        sample_rows: Inspect only the first N rows. ``None`` inspects all.

    Returns:
        The observed schema.
    """
    with path.open(newline="", encoding="utf-8-sig") as handle:
        sample = handle.read(8192)
        handle.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
            delimiter = dialect.delimiter
        except csv.Error:
            delimiter = ","
        reader = csv.reader(handle, delimiter=delimiter)
        rows: list[list[str]] = []
        for index, row in enumerate(reader):
            if sample_rows is not None and index >= sample_rows:
                break
            rows.append(row)

    if not rows:
        return CsvSchema(
            path=str(path), columns=(), rows=0, delimiter=delimiter, has_header=False
        )

    width = len(rows[0])
    has_header = any(
        not _try_number(cell) for cell in rows[0]
    )
    header = [cell.strip() for cell in rows[0]] if has_header else [
        f"column_{index}" for index in range(width)
    ]
    body = rows[1:] if has_header else rows

    columns = []
    for index in range(width):
        values = [row[index] if index < len(row) else "" for row in body]
        columns.append(_profile_column(header[index], values))

    notes = ""
    if not has_header:
        notes = "headerless single-column numeric series (observed: no non-numeric first row)"
    return CsvSchema(
        path=str(path),
        columns=tuple(columns),
        rows=len(body),
        delimiter=delimiter,
        has_header=has_header,
        notes=notes,
    )


def _try_number(text: str) -> float | None:
    try:
        return float(text.strip())
    except (ValueError, AttributeError):
        return None


def discover_profile_schema(path: Path) -> ProfileSchema:
    """Discover a headerless per-unit profile: row count, range, normalisation."""
    values: list[float] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            values.append(float(text))
    if not values:
        raise DssParseError(f"profile {path} contains no values")
    maximum = max(values)
    minimum = min(values)
    return ProfileSchema(
        path=str(path),
        rows=len(values),
        minimum=minimum,
        maximum=maximum,
        is_normalised=abs(maximum - 1.0) < 1e-9,
        notes="per-unit fraction of annual maximum" if abs(maximum - 1.0) < 1e-9 else "",
    )


def discover_buscoords_schema(path: Path) -> dict[str, object]:
    """Discover a ``Buscoords`` coordinate list."""
    coordinates = parse_buscoords(path.read_text(encoding="utf-8", errors="replace"))
    if not coordinates:
        raise DssParseError(f"Buscoords file {path} yielded no coordinates")
    first = next(iter(coordinates.values()))
    return {
        "path": str(path),
        "count": len(coordinates),
        "fields": ["bus", "longitude", "latitude"],
        "units": "degrees",
        "example": {"bus": first.bus, "lon": first.longitude, "lat": first.latitude},
    }


# ----------------------------------------------------------------------
# DSS discovery
# ----------------------------------------------------------------------


def discover_dss_schema(path: Path, objects: list[DssObject] | None = None) -> DssSchema:
    """Discover the schema of an OpenDSS element file.

    Args:
        path: File to inspect.
        objects: Pre-parsed objects, to avoid re-parsing large files.
    """
    if objects is None:
        objects = parse_dss(path.read_text(encoding="utf-8", errors="replace"))

    classes = Counter(obj.element_class for obj in objects)
    params: dict[str, set[str]] = {}
    for obj in objects:
        params.setdefault(obj.element_class, set()).update(obj.params)

    return DssSchema(
        path=str(path),
        element_classes=dict(classes),
        parameters_by_class={
            key: tuple(sorted(value)) for key, value in sorted(params.items())
        },
        total_elements=len(objects),
    )


# ----------------------------------------------------------------------
# Feeder-level discovery
# ----------------------------------------------------------------------


def _count_lines(path: Path) -> int:
    with path.open(encoding="utf-8", errors="replace") as handle:
        return sum(1 for line in handle if line.strip())


def discover_feeder_schema(layout: SmartDsLayout) -> FeederSchema:
    """Inspect every file relevant to one feeder folder.

    Files that do not exist for this scenario (for example ``Storage.dss`` in a
    scenario with no batteries) are reported as absent rather than assumed empty.
    """
    notes: list[str] = []

    def maybe_dss(relative: str) -> DssSchema | None:
        path = layout.feeder_file(relative)
        if not path.is_file():
            notes.append(f"{relative} absent for this scenario")
            return None
        return discover_dss_schema(path)

    loads = maybe_dss("Loads.dss")
    shapes = maybe_dss("LoadShapes.dss")
    lines = maybe_dss("Lines.dss")
    transformers = maybe_dss("Transformers.dss")
    pv = maybe_dss("PVSystems.dss")
    storage = maybe_dss("Storage.dss")

    buscoords: dict[str, object] = {}
    buscoords_path = layout.feeder_file("Buscoords.dss")
    if buscoords_path.is_file():
        try:
            buscoords = discover_buscoords_schema(buscoords_path)
        except DssParseError as exc:
            notes.append(f"Buscoords.dss unreadable: {exc}")
    else:
        notes.append("Buscoords.dss absent for this scenario")

    metrics_rows = 0
    metrics_path = layout.scenario_dir / "metrics.csv"
    if metrics_path.is_file():
        metrics_rows = max(0, _count_lines(metrics_path) - 1)
    else:
        notes.append("metrics.csv absent")

    summary: list[str] = []
    summary_path = layout.feeder_analysis_dir / "Summary_data.csv"
    if summary_path.is_file():
        with summary_path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.reader(handle)
            header = next(reader, [])
            body = next(reader, [])
        summary = [f"{name}={value}" for name, value in zip(header, body)]
    else:
        notes.append("analysis/Summary_data.csv absent")

    return FeederSchema(
        layout=layout,
        loads=loads,
        load_shapes=shapes,
        lines=lines,
        transformers=transformers,
        pv_systems=pv,
        storage=storage,
        buscoords=buscoords,
        metrics_rows=metrics_rows,
        summary=tuple(summary),
        notes=tuple(notes),
    )
