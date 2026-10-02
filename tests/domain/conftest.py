"""Shared fixtures for the Phase 2 energy-domain tests.

Every fixture builds a small but *complete* reference environment. Tests then
vary exactly one aspect, which keeps failures attributable to a single cause.

The reference system is deliberately minimal: two nodes (a point of common
coupling and one feeder), one battery, one solar array. No size is hard-coded
anywhere in the domain model, and these counts are test fixtures only.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from energy_intelligence.domain import (
    Action,
    ActionId,
    ActionType,
    AssetId,
    AssetType,
    AuthorityLevel,
    Battery,
    Constraint,
    ConstraintCategory,
    ConstraintId,
    DataQuality,
    DemandState,
    EnergyState,
    EnergySystem,
    EVState,
    FlexibleLoad,
    GridState,
    NetworkElement,
    NetworkElementKind,
    NetworkElementId,
    NetworkTopology,
    Node,
    NodeId,
    NodePower,
    Objective,
    ObjectiveCategory,
    ObjectiveId,
    ProcessingStep,
    Provenance,
    ProvenanceEvent,
    Quantity,
    QualityFlag,
    RenewableState,
    SolarAsset,
    SourceReference,
    StateOrigin,
    StorageState,
    SystemId,
    TimeBase,
    Unit,
    UncertaintyEstimate,
    UncertaintyKind,
    VariableRole,
)

UTC = timezone.utc
ORIGIN = datetime(2026, 1, 1, tzinfo=UTC)


@pytest.fixture
def clean_quality() -> DataQuality:
    return DataQuality(flags=frozenset({QualityFlag.OK}))


@pytest.fixture
def provenance() -> Provenance:
    return Provenance(
        source=SourceReference(
            source_id="src-test",
            dataset="load_profile",
            locator="data/raw/test/load.csv",
            version="v1",
            checksum="deadbeef",
        ),
        events=(
            ProvenanceEvent("2026-01-01T00:00:00+00:00", "ingested", "test fixture"),
        ),
    )


@pytest.fixture
def processed_provenance(provenance: Provenance) -> Provenance:
    """Provenance that includes a processing step, for round-trip coverage."""
    return Provenance(
        source=provenance.source,
        events=provenance.events,
        processing=(ProcessingStep("resample_mean", "v1", "15-minute mean"),),
    )


@pytest.fixture
def time_base() -> TimeBase:
    return TimeBase(timestep_minutes=15, horizon_steps=96, origin=ORIGIN)


@pytest.fixture
def pcc_node() -> Node:
    return Node(
        node_id=NodeId("node-pcc"),
        name="Point of common coupling",
        voltage_kv=Quantity(11.0, Unit.KILOVOLT),
        is_grid_connection=True,
    )


@pytest.fixture
def feeder_node() -> Node:
    return Node(
        node_id=NodeId("node-feeder-a"),
        name="Feeder A head",
        voltage_kv=Quantity(11.0, Unit.KILOVOLT),
        parent_node_id=NodeId("node-pcc"),
    )


@pytest.fixture
def topology(pcc_node: Node, feeder_node: Node) -> NetworkTopology:
    return NetworkTopology(
        nodes=(pcc_node, feeder_node),
        elements=(
            NetworkElement(
                element_id=NetworkElementId("net-feeder-a-l1"),
                kind=NetworkElementKind.LINE,
                from_node_id=NodeId("node-pcc"),
                to_node_id=NodeId("node-feeder-a"),
                rating=Quantity(2000.0, Unit.KILOWATT),
                parameters={"resistance_ohm": 0.35, "reactance_ohm": 0.12},
            ),
        ),
        description="Two-node reference feeder",
    )


@pytest.fixture
def aggregate_topology(pcc_node: Node) -> NetworkTopology:
    """A single lumped node, to prove the aggregate case is representable."""
    return NetworkTopology(nodes=(pcc_node,), description="Lumped aggregate")


@pytest.fixture
def battery() -> Battery:
    return Battery(
        asset_type=AssetType.BATTERY,
        asset_id=AssetId("asset-battery-1"),
        name="Feeder A battery",
        node_id=NodeId("node-feeder-a"),
        authority=AuthorityLevel.DISPATCHABLE,
        rated_power=Quantity(250.0, Unit.KILOWATT),
        availability=1.0,
        metadata={"chemistry": "lfp"},
        usable_energy=Quantity(500.0, Unit.KILOWATT_HOURS),
        min_soc=0.1,
        max_soc=0.95,
        max_charge_power=Quantity(250.0, Unit.KILOWATT),
        max_discharge_power=Quantity(250.0, Unit.KILOWATT),
        round_trip_efficiency=0.92,
    )


@pytest.fixture
def solar() -> SolarAsset:
    return SolarAsset(
        asset_type=AssetType.SOLAR,
        asset_id=AssetId("asset-solar-1"),
        name="Feeder A rooftop PV",
        node_id=NodeId("node-feeder-a"),
        authority=AuthorityLevel.CURTAILABLE,
        rated_power=Quantity(400.0, Unit.KILOWATT),
        availability=1.0,
        metadata=None,
        capacity_kw=Quantity(400.0, Unit.KILOWATT),
        tilt_degrees=None,
        azimuth_degrees=None,
        has_inverter=True,
    )


@pytest.fixture
def advisory_ev() -> FlexibleLoad:
    """An asset the system may reason about but never dispatch."""
    return FlexibleLoad(
        asset_type=AssetType.FLEXIBLE_LOAD,
        asset_id=AssetId("asset-load-advisory"),
        name="Customer plant (advisory only)",
        node_id=NodeId("node-feeder-a"),
        authority=AuthorityLevel.ADVISORY_ONLY,
        rated_power=Quantity(150.0, Unit.KILOWATT),
        availability=1.0,
        metadata=None,
        min_power=Quantity(60.0, Unit.KILOWATT),
        max_power=Quantity(150.0, Unit.KILOWATT),
        shiftable_energy=0.4,
        priority=3,
        is_hvac=False,
    )


@pytest.fixture
def system(
    time_base: TimeBase,
    topology: NetworkTopology,
    battery: Battery,
    solar: SolarAsset,
    provenance: Provenance,
) -> EnergySystem:
    return EnergySystem(
        system_id=SystemId("sys-reference"),
        name="Reference environment",
        description="Two-node renewable-integrated DER environment",
        time_base=time_base,
        topology=topology,
        assets=(battery, solar),
        provenance=provenance,
    )


@pytest.fixture
def balanced_state(
    provenance: Provenance,
    clean_quality: DataQuality,
    battery: Battery,
    solar: SolarAsset,
) -> EnergyState:
    """A state whose per-node power balance closes within tolerance."""
    return EnergyState(
        timestamp="2026-01-01T00:00:00+00:00",
        origin=StateOrigin.OBSERVED,
        node_power=(
            NodePower(
                node_id=NodeId("node-pcc"),
                generation_kw=Quantity(0.0, Unit.KILOWATT),
                load_kw=Quantity(0.0, Unit.KILOWATT),
                storage_kw=Quantity(0.0, Unit.KILOWATT),
                grid_kw=Quantity(0.0, Unit.KILOWATT),
            ),
            NodePower(
                node_id=NodeId("node-feeder-a"),
                generation_kw=Quantity(300.0, Unit.KILOWATT),
                load_kw=Quantity(500.0, Unit.KILOWATT),
                storage_kw=Quantity(0.0, Unit.KILOWATT),
                grid_kw=Quantity(200.0, Unit.KILOWATT),
            ),
        ),
        demand=DemandState(
            total_demand_kw=Quantity(500.0, Unit.KILOWATT),
            quality=clean_quality,
            by_category={"residential": 300.0, "commercial": 200.0},
        ),
        grid=GridState(
            import_kw=Quantity(200.0, Unit.KILOWATT),
            export_kw=Quantity(0.0, Unit.KILOWATT),
            quality=clean_quality,
        ),
        provenance=provenance,
        storage=(
            StorageState(
                asset_id=battery.asset_id,
                state_of_charge=0.6,
                charge_power_kw=Quantity(0.0, Unit.KILOWATT),
                discharge_power_kw=Quantity(0.0, Unit.KILOWATT),
                available_energy=Quantity(300.0, Unit.KILOWATT_HOURS),
                quality=clean_quality,
            ),
        ),
        renewables=(
            RenewableState(
                asset_id=solar.asset_id,
                available_power=Quantity(320.0, Unit.KILOWATT),
                actual_power=Quantity(300.0, Unit.KILOWATT),
                quality=clean_quality,
            ),
        ),
        timebase_id="tb-15min",
    )


@pytest.fixture
def discharge_action(provenance: Provenance, battery: Battery) -> Action:
    return Action(
        action_id=ActionId("act-discharge-1"),
        action_type=ActionType.BATTERY_DISCHARGE,
        target_asset_id=battery.asset_id,
        setpoint=Quantity(100.0, Unit.KILOWATT),
        role=VariableRole.ACTION,
        issued_under=AuthorityLevel.DISPATCHABLE,
        start_step=92,
        duration_steps=4,
        provenance=provenance,
        rationale="Discharge into the evening peak",
    )


@pytest.fixture
def constraint(provenance: Provenance, battery: Battery) -> Constraint:
    return Constraint(
        constraint_id=ConstraintId("con-soc-window"),
        category=ConstraintCategory.STATE_OF_CHARGE_LIMITS,
        description="Battery state of charge stays within its usable window",
        provenance=provenance,
        applies_to_asset_ids=(battery.asset_id,),
        lower_bound=0.1,
        upper_bound=0.95,
    )


@pytest.fixture
def objective(provenance: Provenance) -> Objective:
    return Objective(
        objective_id=ObjectiveId("obj-energy-cost"),
        category=ObjectiveCategory.ENERGY_COST,
        measurement="Sum of grid import at the tariff across the horizon",
        why_it_matters="Energy cost is the primary economic driver for a DER operator",
        provenance=provenance,
    )


# ----------------------------------------------------------------------
# Round-trip case factory
# ----------------------------------------------------------------------


@pytest.fixture
def make_case():
    """Return a factory that builds one named domain object from fixtures.

    Used by the parametrized serialization round-trip test so every serialisable
    type is exercised without duplicating fixture wiring per type.
    """

    def _make(case: str, request) -> object:
        builders = {
            "battery": lambda r: r.getfixturevalue("battery"),
            "solar": lambda r: r.getfixturevalue("solar"),
            "topology": lambda r: r.getfixturevalue("topology"),
            "pcc_node": lambda r: r.getfixturevalue("pcc_node"),
            "element": lambda r: r.getfixturevalue("topology").elements[0],
            "state": lambda r: r.getfixturevalue("balanced_state"),
            "action": lambda r: r.getfixturevalue("discharge_action"),
            "forecast": _make_forecast,
            "constraint": lambda r: r.getfixturevalue("constraint"),
            "objective": lambda r: r.getfixturevalue("objective"),
            "observation": _make_observation,
            "system": lambda r: r.getfixturevalue("system"),
        }
        return builders[case](request)

    return _make


def _make_forecast(request):
    from energy_intelligence.domain import Forecast, ForecastId, VariableRole

    return Forecast(
        forecast_id=ForecastId("fc-roundtrip"),
        variable="total_demand",
        role=VariableRole.OBSERVATION,
        target_time="2026-01-01T06:00:00+00:00",
        issued_at="2026-01-01T00:00:00+00:00",
        horizon_steps=24,
        value=Quantity(650.0, Unit.KILOWATT),
        provenance=request.getfixturevalue("provenance"),
        uncertainty=UncertaintyEstimate(
            kind=UncertaintyKind.PARAMETRIC,
            unit=Unit.KILOWATT,
            method="quantile",
            relative_std=0.12,
        ),
    )


def _make_observation(request):
    from energy_intelligence.domain import ObservationRecord, VariableRole

    return ObservationRecord(
        variable="battery_state_of_charge",
        role=VariableRole.OBSERVATION,
        value=Quantity(0.6, Unit.FRACTION),
        quality=request.getfixturevalue("clean_quality"),
        provenance=request.getfixturevalue("provenance"),
        timestamp="2026-01-01T00:00:00+00:00",
        asset_id=AssetId("asset-battery-1"),
    )
