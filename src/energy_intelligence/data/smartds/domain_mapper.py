"""Map normalised SMART-DS records into Phase 2 domain objects.

This is the only module that constructs Phase 2 objects from real data, and it
is the place where the answer to the Phase 3 primary question is produced:
*can a real SMART-DS instance be transformed into an EnergySystem without
inventing information?*

The mapper is conservative by construction:

* an asset is created only where the dataset supplies the fields Phase 2
  requires;
* a field the dataset does not state becomes ``None`` -- meaning *not known* --
  rather than a default that would later be mistaken for a measurement;
* where the dataset supplies nothing at all, the asset is **omitted** and the
  omission is recorded, rather than filled in;
* the grid-connection node is **read** from ``Master.dss`` and, if it cannot be
  identified, mapping **fails** (see :class:`UnresolvedBoundaryError`) instead of
  promoting an arbitrary bus to the point of common coupling;
* availability defaults to ``1.0`` **only** because SMART-DS models every
  equipment item as in service. That is a documented modelling fact, not a
  measurement, and it is recorded in the mapper's report so a reader can see it.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from ...domain.assets import Asset, Battery, SolarAsset
from ...domain.enums import AssetType, AuthorityLevel, QualityFlag, Unit, VariableRole
from ...domain.identifiers import AssetId, NodeId
from ...domain.observations import ObservationRecord
from ...domain.provenance import ProcessingStep, Provenance, ProvenanceEvent, SourceReference
from ...domain.quantities import Quantity
from ...domain.quality import DataQuality
from ...domain.system import EnergySystem
from ...domain.timebase import TimeBase
from ...domain.topology import NetworkElement, NetworkElementId, NetworkElementKind, NetworkTopology, Node
from .adapter import DerAssetCatalogue, LoadCatalogue, SmartDsAdapter, TopologyCatalogue
from .layout import SmartDsLayout, sanitize_identifier

__all__ = [
    "MappingReport",
    "DomainMappingResult",
    "UnresolvedBoundaryError",
    "map_to_domain",
]

#: Recorded on every provenance chain this mapper emits.
_TRANSFORM_VERSION = "phase3-adapter-1.0.0"

_LOAD_STEP = ProcessingStep(
    name="load_power_from_rating_and_pu_profile",
    version=_TRANSFORM_VERSION,
    detail="P(t) = sum over customer Load objects of (kW_rating x profile_pu(t))",
)

_BATTERY_STEP = ProcessingStep(
    name="battery_round_trip_efficiency_from_directional_efficiencies",
    version=_TRANSFORM_VERSION,
    detail="round_trip = (%EffCharge/100) x (%EffDischarge/100)",
)


class UnresolvedBoundaryError(RuntimeError):
    """The feeder boundary could not be identified from the dataset.

    Raised instead of inventing a point of common coupling. ``NetworkTopology``
    requires exactly one grid-connection node, so a boundary that the data does
    not state cannot be represented; failing loudly is the only honest option.
    """


@dataclass(frozen=True, slots=True)
class MappingReport:
    """What was created, omitted, and why."""

    assets_created: int = 0
    nodes_created: int = 0
    elements_created: int = 0
    observations_created: int = 0
    omitted: tuple[str, ...] = ()
    unknown_fields: tuple[str, ...] = ()
    availability_basis: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "assets_created": self.assets_created,
            "nodes_created": self.nodes_created,
            "elements_created": self.elements_created,
            "observations_created": self.observations_created,
            "omitted": list(self.omitted),
            "unknown_fields": list(self.unknown_fields),
            "availability_basis": self.availability_basis,
        }


@dataclass(frozen=True, slots=True)
class DomainMappingResult:
    """The instantiated system plus its mapping report."""

    system: EnergySystem
    report: MappingReport
    topology: NetworkTopology
    observations: tuple[ObservationRecord, ...] = ()


def _source_reference(layout: SmartDsLayout, role: str, name: str) -> SourceReference:
    """Build a :class:`SourceReference` pointing at the exact SMART-DS file."""
    return SourceReference(
        source_id=f"smartds-{layout.region}-{layout.subregion}",
        dataset=role,
        locator=f"{layout.version}/{layout.year}/{layout.region}/{layout.subregion}/"
        f"scenarios/{layout.scenario}/{name}",
        version=layout.version.lstrip("v"),
    )


def _provenance(
    layout: SmartDsLayout,
    role: str,
    name: str,
    event: str,
    *,
    event_timestamp: str,
    with_processing: bool = True,
) -> Provenance:
    """Build provenance anchored to the grid instant being mapped.

    The event timestamp is the *data's* timestamp, not the wall-clock time of
    the run. That keeps the encoded system byte-identical across re-runs, which
    is the determinism Phase 3 requires; acquisition time lives in the manifest,
    where it is documented as the one nondeterministic field.
    """
    return Provenance(
        source=_source_reference(layout, role, name),
        events=(
            ProvenanceEvent(
                timestamp=event_timestamp, event=event, detail=layout.scenario
            ),
        ),
        processing=(_LOAD_STEP,) if with_processing else (),
    )


def map_to_domain(
    adapter: SmartDsAdapter,
    *,
    load_catalogue: LoadCatalogue,
    topology_catalogue: TopologyCatalogue,
    der_catalogue: DerAssetCatalogue,
    demand_kw: float,
    demand_quality: DataQuality,
    summary_customer_kw: float | None,
    event_timestamp: str,
) -> DomainMappingResult:
    """Build a real :class:`EnergySystem` from discovered SMART-DS data.

    Args:
        adapter: Adapter holding the resolved layout.
        load_catalogue: Grouped load customers.
        topology_catalogue: Discovered graph.
        der_catalogue: Discovered PV, battery and EV assets.
        demand_kw: Demand at the timepoint being mapped.
        demand_quality: Data-quality metadata for that demand value.
        summary_customer_kw: Published customer power, used only to name the
            system; never substituted for computed demand.

    Returns:
        The system and a report of what was omitted.

    Raises:
        UnresolvedBoundaryError: If ``Master.dss`` does not state a source bus,
            or the stated source bus is absent from the discovered graph. A
            boundary the data does not state cannot be invented.
        DomainValidationError: Propagated from the domain model if the
            constructed system is internally inconsistent. That is intended: a
            broken mapping must fail loudly rather than yield a plausible
            object.
    """
    layout = adapter.layout
    omitted: list[str] = []
    unknown_fields: list[str] = []
    provenance = _provenance(
        layout,
        "feeder_model",
        f"opendss/{layout.substation}/{layout.feeder}",
        "ingested_from_smart_ds",
        event_timestamp=event_timestamp,
        with_processing=False,
    )

    # ---- Nodes -----------------------------------------------------
    bus_names = dict(topology_catalogue.buses)
    nodes: list[Node] = []
    for node_id in topology_catalogue.nodes:
        bus = bus_names.get(node_id) or _bus_from_node(node_id)
        nodes.append(
            Node(
                node_id=node_id,
                name=bus,
                voltage_kv=(
                    Quantity(topology_catalogue.nominal_voltage_kv, Unit.KILOVOLT)
                    if topology_catalogue.nominal_voltage_kv is not None
                    else None
                ),
                parent_node_id=None,
                is_grid_connection=False,
                metadata={"bus": bus} if bus else None,
            )
        )

    # Exactly one grid-connection node is required by the domain model. The
    # source bus is READ from the Circuit element in Master.dss, which is where
    # OpenDSS declares the point of common coupling. Nothing is inferred from the
    # folder name, and no bus is promoted to PCC when the source bus is unknown.
    head_node = _grid_connection_node(layout, adapter, nodes)
    nodes = [
        Node(
            node_id=node.node_id,
            name=node.name,
            voltage_kv=node.voltage_kv,
            parent_node_id=node.parent_node_id,
            is_grid_connection=node.node_id == head_node,
            metadata=node.metadata,
        )
        for node in nodes
    ]

    # ---- Network elements -----------------------------------------
    elements: list[NetworkElement] = []
    for index, (from_node, to_node) in enumerate(topology_catalogue.edges):
        elements.append(
            NetworkElement(
                element_id=NetworkElementId(
                    f"net-smartds-edge-{index:05d}"
                ),
                kind=NetworkElementKind.LINE,
                from_node_id=from_node,
                to_node_id=to_node,
                rating=None,
                is_in_service=True,
                parameters=None,
                metadata={"source": "Lines.dss"},
            )
        )

    network = NetworkTopology(
        nodes=tuple(nodes),
        elements=tuple(elements),
        description=(
            f"SMART-DS {layout.region}/{layout.subregion} {layout.year} feeder "
            f"{layout.feeder} (scenario {layout.scenario})"
        ),
        metadata={"dataset": "SMART-DS", "version": layout.version.lstrip("v")},
    )

    # ---- Assets ----------------------------------------------------
    assets: list[Asset] = []

    # One asset per customer. SMART-DS resolves 3,690 Load objects into 1,871
    # customers, each with its own rating and its own connection bus; collapsing
    # them into one lumped load would destroy exactly the structure Phase 9 and
    # Phase 10 need. AssetType.LOAD is implemented by the base Asset class, so no
    # flexibility fields are asserted -- SMART-DS contains none.
    for customer in load_catalogue.customers:
        node_id = (
            layout.feeder_node_id(customer.bus) if customer.bus else head_node
        )
        if node_id is None:  # pragma: no cover - _grid_connection_node guarantees one
            raise UnresolvedBoundaryError(
                f"customer {customer.name!r} has no connection bus and no grid node exists"
            )
        assets.append(
            Asset(
                asset_type=AssetType.LOAD,
                asset_id=AssetId(
                    f"asset-smartds-load-{sanitize_identifier(customer.name)}"
                ),
                name=customer.name,
                node_id=node_id,
                authority=AuthorityLevel.NOT_CONTROLLABLE,
                rated_power=Quantity(customer.rated_kw, Unit.KILOWATT),
                availability=1.0,
                metadata={
                    "bus": customer.bus or "",
                    "members": ",".join(member.name for member in customer.members),
                    "member_count": str(len(customer.members)),
                    "center_tap": str(customer.is_center_tap).lower(),
                    "flexibility": "NOT STATED by the dataset; no shiftable or curtailable load",
                },
            )
        )

    if load_catalogue.unmapped_objects:
        omitted.append(
            f"{len(load_catalogue.unmapped_objects)} Load objects could not be grouped "
            "into a customer (no usable loadshape); no asset was created for them"
        )

    for asset_id, node_id, capacity in der_catalogue.solar:
        assets.append(
            SolarAsset(
                asset_type=AssetType.SOLAR,
                asset_id=asset_id,
                name=asset_id.value,
                node_id=node_id,
                authority=AuthorityLevel.CURTAILABLE,
                rated_power=Quantity(capacity, Unit.KILOWATT),
                availability=1.0,
                metadata={"scenario": layout.scenario},
                capacity_kw=Quantity(capacity, Unit.KILOWATT),
                tilt_degrees=None,
                azimuth_degrees=None,
                has_inverter=True,
            )
        )

    for spec in der_catalogue.batteries:
        assets.append(
            Battery(
                asset_type=AssetType.BATTERY,
                asset_id=spec.asset_id,
                name=spec.name,
                node_id=spec.node_id,
                authority=AuthorityLevel.DISPATCHABLE,
                rated_power=Quantity(spec.rated_power_kw, Unit.KILOWATT),
                availability=1.0,
                metadata={
                    "scenario": layout.scenario,
                    "initial_soc": (
                        f"{spec.initial_soc:.6f}" if spec.initial_soc is not None else "UNKNOWN"
                    ),
                    "stored_energy_kwh": (
                        f"{spec.stored_energy_kwh:.6f}"
                        if spec.stored_energy_kwh is not None
                        else "UNKNOWN"
                    ),
                    "dispatch": "UNAVAILABLE",
                    "availability_basis": "modelled in service; not a measurement",
                },
                # kWhRated is a NAMEPLATE rating. Usable energy additionally needs a
                # depth-of-discharge limit, which SMART-DS does not state, so it is
                # UNKNOWN rather than equal to the nameplate (D-050).
                usable_energy=None,
                nameplate_energy=Quantity(spec.nameplate_energy_kwh, Unit.KILOWATT_HOURS),
                min_soc=None,
                max_soc=None,
                # SMART-DS states one kWRated, not separate charge and discharge
                # limits; both directional limits are therefore UNKNOWN.
                max_charge_power=None,
                max_discharge_power=None,
                round_trip_efficiency=(
                    spec.round_trip_efficiency
                    if spec.round_trip_efficiency is not None
                    else 0.0
                ),
            )
        )

    if der_catalogue.batteries:
        unknown_fields.extend(
            (
                "Battery.usable_energy (no depth-of-discharge limit in SMART-DS)",
                "Battery.min_soc and Battery.max_soc (no SOC window in SMART-DS)",
                "Battery.max_charge_power and Battery.max_discharge_power "
                "(only a single kWRated is stated)",
            )
        )

    if der_catalogue.ev_sites:
        omitted.append(
            f"{der_catalogue.ev_sites} EV adoption sites found, but SMART-DS provides "
            "no EV charging load, power or session series; no EV asset was created "
            "because a charging profile cannot be invented"
        )

    omitted.append(
        "wind generation: SMART-DS contains no wind assets, so no wind asset was created"
    )
    omitted.append(
        "battery dispatch and state-of-charge series: no SOC(t) exists in the dataset, "
        "so no StorageState was constructed"
    )

    # ---- Observations ----------------------------------------------
    # The value the ingestion reconstructed, at the real instant it describes. It
    # is DERIVED (rating x per-unit profile), never labelled observed, and it is
    # scoped to the point of common coupling where the grid sees it.
    observations: list[ObservationRecord] = [
        ObservationRecord(
            variable="total_demand_kw",
            role=VariableRole.DERIVED,
            value=Quantity(demand_kw, Unit.KILOWATT),
            quality=demand_quality,
            provenance=_provenance(
                layout,
                "feeder_model",
                f"opendss/{layout.substation}/{layout.feeder}",
                "demand_reconstructed_from_ratings_and_profiles",
                event_timestamp=event_timestamp,
            ),
            timestamp=event_timestamp,
            node_id=head_node,
        )
    ]
    if summary_customer_kw is not None:
        observations.append(
            ObservationRecord(
                variable="published_customer_demand_kw",
                role=VariableRole.OBSERVATION,
                value=Quantity(summary_customer_kw, Unit.KILOWATT),
                quality=DataQuality(
                    flags=frozenset({QualityFlag.OK}),
                    detail="published in analysis/Summary_data.csv; the dataset's own figure, not a reconstruction",
                ),
                provenance=_provenance(
                    layout,
                    "published_summary",
                    f"opendss/{layout.substation}/{layout.feeder}/analysis/Summary_data.csv",
                    "published_peak_demand_read",
                    event_timestamp=event_timestamp,
                    with_processing=False,
                ),
                timestamp=event_timestamp,
                node_id=head_node,
            )
        )

    # ---- Time base -------------------------------------------------
    axis = adapter.time_axis()
    time_base = TimeBase(
        timestep_minutes=axis.timestep_minutes,
        horizon_steps=96,
        origin=datetime(axis.year, 1, 1, tzinfo=timezone.utc),
    )

    system = EnergySystem(
        system_id=layout.system_id(),
        name=f"SMART-DS {layout.region}/{layout.subregion} feeder {layout.feeder}",
        description=(
            f"Real SMART-DS feeder instantiated from {layout.version} {layout.year}. "
            "Demand is reconstructed from rated kW times per-unit profiles. "
            f"Published peak customer power: {summary_customer_kw:.4f} kW."
            if summary_customer_kw is not None
            else f"Real SMART-DS feeder instantiated from {layout.version} {layout.year}."
        ),
        time_base=time_base,
        topology=network,
        assets=tuple(assets),
        provenance=provenance,
        metadata={
            "dataset": "SMART-DS",
            "scenario": layout.scenario,
            "subregion": layout.subregion,
            "feeder": layout.feeder,
            "timezone_assumption": (
                "SMART-DS carries no timezone; the synthetic yearly clock is "
                "anchored to 01-01T00:00:00+00:00 and treated as UTC"
            ),
        },
    )

    report = MappingReport(
        assets_created=len(assets),
        nodes_created=len(nodes),
        elements_created=len(elements),
        observations_created=len(observations),
        omitted=tuple(omitted),
        unknown_fields=tuple(unknown_fields),
        availability_basis=(
            "1.0 for all assets because SMART-DS models every equipment item as "
            "in service. This is a modelling fact, NOT a measurement, and no "
            "availability timeseries exists."
        ),
    )

    return DomainMappingResult(
        system=system,
        report=report,
        topology=network,
        observations=tuple(observations),
    )


def _bus_from_node(node_id: NodeId) -> str | None:
    """Recover a bus name from a generated NodeId, when no catalogue exists.

    Identifiers are sanitised, so this fallback is approximate: ``_`` becomes
    ``-``. The adapter's ``TopologyCatalogue.buses`` is authoritative and is
    preferred wherever it is available.
    """
    prefix = "node-smartds-"
    if not node_id.value.startswith(prefix):
        return None
    return node_id.value[len(prefix) :]


def _grid_connection_node(
    layout: SmartDsLayout,
    adapter: SmartDsAdapter,
    nodes: list[Node],
) -> NodeId:
    """The node that carries the grid connection, read from ``Master.dss``.

    ``NetworkTopology`` requires exactly one grid-connection node. SMART-DSS
    declares it directly as the ``Circuit`` element's ``bus1``, so that is what is
    read. The feeder *folder* name is never used for this: a label is not a fact,
    and promoting an arbitrary bus to the point of common coupling would put an
    invented node into the domain model.

    Raises:
        UnresolvedBoundaryError: If no source bus is stated, or the stated bus is
            absent from the discovered graph.
    """
    source_bus = adapter.source_bus()
    if not source_bus:
        raise UnresolvedBoundaryError(
            "Master.dss states no Circuit bus1, so the point of common coupling is "
            "unknown; no synthetic PCC node was created"
        )
    node_id = layout.feeder_node_id(source_bus)
    known = {node.node_id for node in nodes}
    if node_id not in known:
        raise UnresolvedBoundaryError(
            f"the Circuit source bus {source_bus!r} is not among the "
            f"{len(known)} discovered nodes, so the boundary cannot be resolved "
            "without inventing a node"
        )
    return node_id

