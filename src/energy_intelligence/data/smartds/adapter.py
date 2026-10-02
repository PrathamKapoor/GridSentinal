"""SMART-DS adapter: the only place that knows SMART-DS's shape.

Layering, per the Phase 3 brief:

```text
SMART-DS  ->  SmartDsAdapter  ->  NormalizedRecords  ->  EnergySystem
              (this package)      (domain-agnostic)      (Phase 2)
```

The domain model never sees a SMART-DS column name. A future dataset would add
a sibling adapter package and reuse everything above this layer.

What this adapter guarantees:

* it never invents a value;
* it never interpolates, resamples or imputes;
* it records provenance for everything it emits;
* it reports what it could not do, rather than substituting a default.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from ...domain.enums import QualityFlag
from ...domain.identifiers import AssetId, NodeId
from ...domain.quality import DataQuality
from .buscoords import BusCoordinate, load_buscoords
from .dss import DssObject, DssParseError, parse_dss
from .layout import SmartDsLayout, sanitize_identifier
from .mapping import (
    FieldMapping,
    MappingConfidence,
    mappings_for,
)
from .normalize import CustomerLoad, TimeAxis, build_time_axis, read_profile
from .schema import discover_feeder_schema, FeederSchema

__all__ = [
    "LoadCatalogue",
    "TopologyCatalogue",
    "BatteryAssetSpec",
    "DerAssetCatalogue",
    "QualityFinding",
    "QualityReport",
    "SmartDsAdapter",
]


@dataclass(frozen=True, slots=True)
class LoadCatalogue:
    """Loads discovered for one feeder, grouped by customer.

    Attributes:
        customers: One entry per customer, centre-tap pairs already grouped.
        unmapped_objects: Load objects that could not be grouped.
    """

    customers: tuple[CustomerLoad, ...]
    unmapped_objects: tuple[str, ...] = ()

    @property
    def customer_count(self) -> int:
        return len(self.customers)

    @property
    def object_count(self) -> int:
        return sum(len(customer.members) for customer in self.customers)

    @property
    def center_tap_count(self) -> int:
        return sum(1 for customer in self.customers if customer.is_center_tap)

    @property
    def total_rated_kw(self) -> float:
        return sum(customer.rated_kw for customer in self.customers)


@dataclass(frozen=True, slots=True)
class TopologyCatalogue:
    """Graph structure discovered for one feeder.

    Attributes:
        nodes: Node identifiers, one per discovered bus.
        buses: The true SMART-DS bus name for each node. Kept alongside the
            identifier because identifiers are sanitised (``_`` becomes ``-``),
            and a sanitised name is not the name in the dataset.
        edges: Directed line connections between nodes.
        coordinates: Bus coordinates, when ``Buscoords.dss`` is present.
        nominal_voltage_kv: Circuit base voltage, when stated.
        transformer_count: Number of transformer elements.
    """

    nodes: tuple[NodeId, ...]
    buses: tuple[tuple[NodeId, str], ...] = ()
    edges: tuple[tuple[NodeId, NodeId], ...] = ()
    coordinates: dict[str, BusCoordinate] = field(default_factory=dict)
    nominal_voltage_kv: float | None = None
    transformer_count: int = 0

    @property
    def node_count(self) -> int:
        return len(self.nodes)

    @property
    def edge_count(self) -> int:
        return len(self.edges)


@dataclass(frozen=True, slots=True)
class BatteryAssetSpec:
    """One battery as the dataset actually states it.

    Every field is read from ``Storage.dss`` or is a derivation of a value read
    from it, and every derivation is named. Nothing here is a default: the
    dataset states no depth of discharge, no SOC window and no separate charge
    or discharge power, so those fields are simply absent from this record rather
    than defaulted to zero.

    Attributes:
        asset_id: Domain identifier.
        node_id: Node the battery is connected to.
        name: Source element name.
        rated_power_kw: ``kWRated``, the nameplate power rating.
        nameplate_energy_kwh: ``kWhRated``, the nameplate energy rating.
        stored_energy_kwh: ``kWhStored``, a single scalar, not a series.
        initial_soc: ``kWhStored / kWhRated``; a DERIVED fraction.
        charge_efficiency: ``%EffCharge`` as a fraction.
        discharge_efficiency: ``%EffDischarge`` as a fraction.
        round_trip_efficiency: ``charge_efficiency * discharge_efficiency``.
    """

    asset_id: AssetId
    node_id: NodeId
    name: str
    rated_power_kw: float
    nameplate_energy_kwh: float
    stored_energy_kwh: float | None
    initial_soc: float | None
    charge_efficiency: float | None
    discharge_efficiency: float | None
    round_trip_efficiency: float | None


@dataclass(frozen=True, slots=True)
class DerAssetCatalogue:
    """DER assets discovered for one scenario."""

    solar: tuple[tuple[AssetId, NodeId, float], ...] = ()
    batteries: tuple[BatteryAssetSpec, ...] = ()
    ev_sites: int = 0

    @property
    def solar_count(self) -> int:
        return len(self.solar)

    @property
    def battery_count(self) -> int:
        return len(self.batteries)

    @property
    def solar_capacity_kw(self) -> float:
        return sum(capacity for _, _, capacity in self.solar)

    @property
    def battery_power_kw(self) -> float:
        return sum(spec.rated_power_kw for spec in self.batteries)

    @property
    def battery_energy_kwh(self) -> float:
        return sum(spec.nameplate_energy_kwh for spec in self.batteries)


@dataclass(frozen=True, slots=True)
class QualityFinding:
    """One data-quality observation.

    Attributes:
        subject: What was inspected.
        flag: The applicable :class:`QualityFlag`.
        detail: Human-readable explanation.
        count: How many instances, where countable.
    """

    subject: str
    flag: QualityFlag
    detail: str
    count: int = 0

    def to_dict(self) -> dict[str, object]:
        return {
            "subject": self.subject,
            "flag": self.flag.value,
            "detail": self.detail,
            "count": self.count,
        }


@dataclass(frozen=True, slots=True)
class QualityReport:
    """Aggregate data-quality findings for one adapter run."""

    findings: tuple[QualityFinding, ...] = ()
    axis: TimeAxis | None = None
    mappings: tuple[FieldMapping, ...] = ()

    @property
    def flags(self) -> frozenset[QualityFlag]:
        out: set[QualityFlag] = set()
        for finding in self.findings:
            out.add(finding.flag)
        return frozenset(out)

    @property
    def unknown_mappings(self) -> tuple[FieldMapping, ...]:
        return tuple(
            m for m in self.mappings if m.confidence is MappingConfidence.UNKNOWN
        )

    @property
    def unmapped_mappings(self) -> tuple[FieldMapping, ...]:
        return tuple(
            m for m in self.mappings if m.confidence is MappingConfidence.UNMAPPED
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "findings": [finding.to_dict() for finding in self.findings],
            "flags": sorted(flag.value for flag in self.flags),
            "unknown_mapping_count": len(self.unknown_mappings),
            "unmapped_mapping_count": len(self.unmapped_mappings),
            "axis": self.axis.describe() if self.axis else None,
        }


@dataclass
class _Parsed:
    loads: list[DssObject] = field(default_factory=list)
    shapes: dict[str, str] = field(default_factory=dict)
    lines: list[DssObject] = field(default_factory=list)
    transformers: list[DssObject] = field(default_factory=list)
    pv: list[DssObject] = field(default_factory=list)
    storage: list[DssObject] = field(default_factory=list)
    base_kv: float | None = None
    circuit_name: str | None = None
    source_bus: str | None = None


class SmartDsAdapter:
    """Reads a downloaded SMART-DS sub-region and normalises what it finds.

    Attributes:
        layout: Resolved dataset locations.
    """

    def __init__(self, layout: SmartDsLayout) -> None:
        self.layout = layout
        self._parsed: _Parsed | None = None
        self._schema: FeederSchema | None = None
        self.findings: list[QualityFinding] = []

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def _read(self, name: str) -> list[DssObject]:
        path = self.layout.feeder_file(name)
        if not path.is_file():
            return []
        try:
            return parse_dss(path.read_text(encoding="utf-8", errors="replace"))
        except DssParseError as exc:
            self.findings.append(
                QualityFinding(
                    subject=name,
                    flag=QualityFlag.INVALID_RANGE,
                    detail=f"parse failed: {exc}",
                )
            )
            return []

    def parsed(self) -> _Parsed:
        """Parse and cache the feeder's element files."""
        if self._parsed is not None:
            return self._parsed

        state = _Parsed()
        state.loads = self._read("Loads.dss")
        for obj in self._read("LoadShapes.dss"):
            mult = obj.get("mult", "")
            filename = _extract_file_reference(mult)
            if filename:
                state.shapes[obj.name] = filename
        state.lines = self._read("Lines.dss")
        state.transformers = self._read("Transformers.dss")
        state.pv = self._read("PVSystems.dss")
        state.storage = self._read("Storage.dss")

        master_path = self.layout.feeder_file("Master.dss")
        if master_path.is_file():
            for obj in parse_dss(master_path.read_text(encoding="utf-8", errors="replace")):
                if obj.element_class.lower() == "circuit":
                    state.base_kv = obj.get_float("basekV")
                    state.circuit_name = obj.name
                    # The Circuit element's bus1 *is* the point of common
                    # coupling: OpenDSS connects the source there. Reading it
                    # from the file is direct evidence; deriving it from the
                    # folder name would be a guess.
                    state.source_bus = _base_bus(obj.get("bus1"))

        self._parsed = state
        return state

    def schema(self) -> FeederSchema:
        """Discovered schema for this feeder, cached."""
        if self._schema is None:
            self._schema = discover_feeder_schema(self.layout)
        return self._schema

    # ------------------------------------------------------------------
    # Circuit boundary
    # ------------------------------------------------------------------

    def source_bus(self) -> str | None:
        """The circuit's source bus, read from ``Master.dss``.

        This is the point of common coupling with the utility grid. It is
        **read**, never inferred from the folder name: a folder name is a label,
        and using it as the PCC would be a guess dressed as a fact.

        Returns ``None`` when ``Master.dss`` declares no ``Circuit`` element or no
        ``bus1``, so a caller can report an unresolved boundary instead of
        inventing one.
        """
        return self.parsed().source_bus

    def circuit_name(self) -> str | None:
        """The ``Circuit`` element name, e.g. ``feeder_p1udt12703-p1uhs0_1247x``."""
        return self.parsed().circuit_name

    def profile_filename(self, loadshape_name: str) -> str | None:
        """The profile file a ``Loadshape`` refers to, or ``None``.

        Shape-to-file resolution lives here so no caller re-implements the
        ``mult = (file=...)`` convention. Phase 4 needs it to accumulate the
        feeder total without materialising every customer series (D-057).
        """
        return self.parsed().shapes.get(loadshape_name)

    # ------------------------------------------------------------------
    # Time
    # ------------------------------------------------------------------

    def time_axis(self) -> TimeAxis:
        """Build the native time axis.

        Raises:
            DssParseError: If no loadshape declares ``npts``, which is the only
                place the dataset states its vector length.
        """
        for obj in self._read("LoadShapes.dss"):
            npts = obj.get_int("npts")
            if npts:
                return build_time_axis(npts, year=int(self.layout.year))
        raise DssParseError(
            "no Loadshape declared npts; the dataset's vector length is unknown "
            "and must not be assumed"
        )

    # ------------------------------------------------------------------
    # Loads
    # ------------------------------------------------------------------

    def loads(
        self,
        *,
        indices: tuple[int, ...] | None = None,
        names: tuple[str, ...] | None = None,
    ) -> LoadCatalogue:
        """Group Load objects into customers with kilowatt series.

        Centre-tap pairs (names ending ``_1``/``_2``) are grouped into one
        customer and **summed**, which the empirical verification established is
        the correct reconstruction of customer total demand.

        Args:
            indices: Grid indices to materialise. ``None`` materialises every
                point of every profile, which is expensive; tests and validation
                pass a small subset.
            names: Restrict the result to these **base** customer names (the name
                with the centre-tap suffix stripped). ``None`` returns every
                customer. Added in Phase 4 so an ML dataset can cover a declared
                subset without materialising 1,871 whole-year series; it selects
                customers and changes no semantics (D-057).
        """
        state = self.parsed()
        wanted = set(names) if names is not None else None
        profile_cache: dict[str, tuple[float, ...]] = {}

        def profile(name: str) -> tuple[float, ...] | None:
            filename = state.shapes.get(name)
            if filename is None:
                return None
            if filename not in profile_cache:
                path = self.layout.profiles_dir / filename
                if not path.is_file():
                    self.findings.append(
                        QualityFinding(
                            subject=f"profiles/{filename}",
                            flag=QualityFlag.MISSING,
                            detail=f"loadshape {name!r} references a file that is not present",
                        )
                    )
                    return None
                profile_cache[filename] = read_profile(path)
            return profile_cache[filename]

        grouped: dict[str, list[DssObject]] = {}
        unmapped: list[str] = []
        for obj in state.loads:
            grouped.setdefault(obj.name, []).append(obj)

        by_base: dict[str, list[DssObject]] = {}
        for name, members in grouped.items():
            base = name[:-2] if name.endswith(("_1", "_2")) else name
            by_base.setdefault(base, []).extend(members)

        customers: list[CustomerLoad] = []
        for base in sorted(by_base):
            if wanted is not None and base not in wanted:
                continue
            members = tuple(sorted(by_base[base], key=lambda o: o.name))
            shape_name = members[0].get("yearly")
            if shape_name is None:
                unmapped.extend(m.name for m in members)
                continue
            values = profile(shape_name)
            if values is None:
                unmapped.extend(m.name for m in members)
                continue

            rated = 0.0
            usable = len(values)
            for member in members:
                rated += member.get_float("kW", 0.0) or 0.0

            if indices is None:
                series = tuple(
                    sum((member.get_float("kW", 0.0) or 0.0) * values[i] for member in members)
                    for i in range(usable)
                )
            else:
                series = tuple(
                    sum(
                        (member.get_float("kW", 0.0) or 0.0) * values[i]
                        for member in members
                        if 0 <= i < usable
                    )
                    for i in indices
                )

            quality = (
                DataQuality(flags=frozenset({QualityFlag.OK}))
                if rated >= 0 and min(series) >= -1e-9
                else DataQuality(flags=frozenset({QualityFlag.INVALID_RANGE}))
            )
            customers.append(
                CustomerLoad(
                    name=base,
                    members=members,
                    series=series,
                    rated_kw=rated,
                    quality=quality,
                    bus=_base_bus(members[0].get("bus1")),
                )
            )

        self.findings.append(
            QualityFinding(
                subject="Loads.dss",
                flag=QualityFlag.OK,
                detail=(
                    f"{state.loads.__len__()} Load objects grouped into "
                    f"{len(customers)} customers"
                    + (f" (filtered to {len(names)} requested)" if names is not None else "")
                ),
                count=len(customers),
            )
        )
        return LoadCatalogue(tuple(customers), tuple(unmapped))

    # ------------------------------------------------------------------
    # Topology
    # ------------------------------------------------------------------

    def topology(self) -> TopologyCatalogue:
        """Build the node graph from Lines.dss and Buscoords.dss."""
        state = self.parsed()
        base_buses: set[str] = set()
        edges: list[tuple[NodeId, NodeId]] = []

        for line in state.lines:
            bus1 = _base_bus(line.get("bus1"))
            bus2 = _base_bus(line.get("bus2"))
            if not bus1 or not bus2:
                continue
            base_buses.add(bus1)
            base_buses.add(bus2)
            edges.append((self.layout.feeder_node_id(bus1), self.layout.feeder_node_id(bus2)))

        coordinates: dict[str, BusCoordinate] = {}
        buscoords_path = self.layout.feeder_file("Buscoords.dss")
        if buscoords_path.is_file():
            try:
                coordinates = load_buscoords(buscoords_path)
            except DssParseError as exc:
                self.findings.append(
                    QualityFinding(
                        subject="Buscoords.dss",
                        flag=QualityFlag.INVALID_RANGE,
                        detail=str(exc),
                    )
                )
        else:
            self.findings.append(
                QualityFinding(
                    subject="Buscoords.dss",
                    flag=QualityFlag.MISSING,
                    detail="no node coordinates available for this feeder",
                )
            )

        # An element may sit on a bus with no line incident (a lone transformer
        # secondary, or a device behind a service transformer). Include every bus
        # any discovered element connects to, so the graph covers every asset
        # point. Otherwise an asset would reference a node the topology does not
        # declare, which the domain model rejects.
        for obj in (*state.loads, *state.pv, *state.storage):
            bus = _base_bus(obj.get("bus1"))
            if bus:
                base_buses.add(bus)

        return TopologyCatalogue(
            nodes=tuple(sorted((self.layout.feeder_node_id(b) for b in base_buses), key=str)),
            buses=tuple(
                (self.layout.feeder_node_id(bus), bus) for bus in sorted(base_buses)
            ),
            edges=tuple(edges),
            coordinates=coordinates,
            nominal_voltage_kv=state.base_kv,
            transformer_count=len(state.transformers),
        )

    # ------------------------------------------------------------------
    # DER assets
    # ------------------------------------------------------------------

    def der_assets(self) -> DerAssetCatalogue:
        """Discover PV, battery and EV assets for this scenario."""
        state = self.parsed()

        solar = []
        for obj in state.pv:
            if obj.element_class.lower() != "pvsystem":
                continue
            bus = _base_bus(obj.get("bus1"))
            if not bus:
                continue
            capacity = obj.get_float("Pmpp", 0.0) or 0.0
            solar.append(
                (
                    AssetId(f"asset-smartds-pv-{sanitize_identifier(obj.name)}"),
                    self.layout.feeder_node_id(bus),
                    capacity,
                )
            )

        batteries: list[BatteryAssetSpec] = []
        for obj in state.storage:
            if obj.element_class.lower() != "storage":
                continue
            bus = _base_bus(obj.get("bus1"))
            if not bus:
                continue
            rated_power = obj.get_float("kWRated")
            nameplate_energy = obj.get_float("kWhRated")
            stored_energy = obj.get_float("kWhStored")
            charge_efficiency = _as_fraction(obj.get_float("%EffCharge"))
            discharge_efficiency = _as_fraction(obj.get_float("%EffDischarge"))
            batteries.append(
                BatteryAssetSpec(
                    asset_id=AssetId(f"asset-smartds-bess-{sanitize_identifier(obj.name)}"),
                    node_id=self.layout.feeder_node_id(bus),
                    name=obj.name,
                    rated_power_kw=rated_power or 0.0,
                    nameplate_energy_kwh=nameplate_energy or 0.0,
                    stored_energy_kwh=stored_energy,
                    initial_soc=_initial_soc(obj),
                    charge_efficiency=charge_efficiency,
                    discharge_efficiency=discharge_efficiency,
                    round_trip_efficiency=(
                        charge_efficiency * discharge_efficiency
                        if charge_efficiency is not None and discharge_efficiency is not None
                        else None
                    ),
                )
            )

        ev_sites = 0
        placement = self.layout.placement_file("ev_residential", "L")
        if placement.is_file():
            import json

            try:
                payload = json.loads(placement.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                self.findings.append(
                    QualityFinding(
                        subject="ev_residential=L.json",
                        flag=QualityFlag.INVALID_RANGE,
                        detail=f"malformed placement JSON: {exc}",
                    )
                )
            else:
                ev_sites = _count_placement_sites(payload)

        if batteries:
            self.findings.append(
                QualityFinding(
                    subject="Storage.dss",
                    flag=QualityFlag.MISSING,
                    detail=(
                        "battery dispatch, state-of-charge series, depth-of-discharge "
                        "(usable energy), the SOC window and separate charge/discharge "
                        "power limits are UNAVAILABLE; only kWRated, kWhRated, "
                        "%EffCharge, %EffDischarge and one kWhStored scalar exist"
                    ),
                    count=len(batteries),
                )
            )

        return DerAssetCatalogue(
            solar=tuple(solar),
            batteries=tuple(batteries),
            ev_sites=ev_sites,
        )

    # ------------------------------------------------------------------
    def quality_report(self) -> QualityReport:
        """Consolidate findings and mappings into one report."""
        return QualityReport(
            findings=tuple(self.findings),
            axis=self.time_axis(),
            mappings=mappings_for(""),
        )


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _base_bus(raw: str | None) -> str | None:
    """Strip OpenDSS phase suffixes from a bus reference.

    ``"p1ulv279.1.2"`` -> ``"p1ulv279"``. Phase 2 has no per-phase node, so the
    node is the base bus name.
    """
    if not raw:
        return None
    head = raw.split(".")[0].strip()
    return head or None


def _extract_file_reference(mult: str | None) -> str | None:
    """Pull the *filename* out of ``mult = (file=../../../../../profiles/x.csv)``.

    The reference observed in SMART-DS is a path **relative to the feeder
    folder**, so it must be reduced to its basename before being resolved
    against ``profiles_dir``. Using the raw path would resolve outside the
    dataset and silently fail to find every profile.
    """
    if not mult:
        return None
    start = mult.find("file=")
    if start < 0:
        return None
    rest = mult[start + 5 :]
    close = rest.find(")")
    token = (rest[:close] if close >= 0 else rest.split()[0]).strip()
    if not token:
        return None
    return PurePosixPath(token).name


def _as_fraction(percent: float | None) -> float | None:
    """Convert a percentage parameter to a fraction.

    OpenDSS states efficiencies as ``%EffCharge=95.0``, meaning 0.95. Returns
    ``None`` when the source does not state it, because a defaulted efficiency
    would silently change every energy calculation downstream.
    """
    if percent is None:
        return None
    return percent / 100.0


def _initial_soc(obj: DssObject) -> float | None:
    """Initial state of charge from the single ``kWhStored`` scalar.

    Returns ``None`` when the ratio cannot be formed. A missing or zero capacity
    must not produce an invented SOC, so the caller records it as unknown
    instead.
    """
    stored = obj.get_float("kWhStored")
    rated = obj.get_float("kWhRated")
    if stored is None or rated is None or rated <= 0:
        return None
    return stored / rated


def _count_placement_sites(payload: object) -> int:
    """Count element names across a placement JSON, whatever its nesting.

    The documented shape is a base object keyed by feeder or substation, whose
    values are lists of element names. The count is robust to the nesting so a
    format change degrades the count rather than crashing.
    """
    total = 0
    if isinstance(payload, dict):
        for value in payload.values():
            total += _count_placement_sites(value)
    elif isinstance(payload, list):
        for item in payload:
            if isinstance(item, str):
                total += 1
            else:
                total += _count_placement_sites(item)
    return total
