"""Tests for topology, assets, state, actions, constraints and objectives."""

from __future__ import annotations

import pytest

from energy_intelligence.domain import (
    ACTION_REQUIREMENTS,
    Action,
    ActionId,
    ActionType,
    Asset,
    AssetId,
    AssetType,
    AuthorityLevel,
    Battery,
    Constraint,
    ConstraintCategory,
    ConstraintId,
    DataQuality,
    DemandState,
    DomainValidationError,
    EVCharger,
    EVState,
    FlexibleLoad,
    GridState,
    NetworkElement,
    NetworkElementId,
    NetworkElementKind,
    NetworkTopology,
    Node,
    NodeId,
    NodePower,
    Objective,
    ObjectiveCategory,
    ObjectiveId,
    ObservationRecord,
    POWER_BALANCE_TOLERANCE,
    Quantity,
    QualityFlag,
    RenewableState,
    SolarAsset,
    StateOrigin,
    StorageState,
    Unit,
    VariableRole,
    WindAsset,
    asset_class_for,
    objective_metadata,
    required_authority_for,
    sign_constraint_for,
    sufficient_authorities_for,
    unit_for_action,
    unit_for_constraint,
)

KW = Unit.KILOWATT
KWH = Unit.KILOWATT_HOURS


def kw(value: float) -> Quantity:
    return Quantity(value, KW)


# ======================================================================
# Topology
# ======================================================================


def test_topology_accepts_a_valid_two_node_feeder(topology: NetworkTopology) -> None:
    assert len(topology.nodes) == 2
    assert len(topology.elements) == 1
    assert not topology.is_aggregate
    assert topology.has_node(NodeId("node-feeder-a"))
    assert topology.node(NodeId("node-pcc")).is_grid_connection


def test_aggregate_topology_is_recognised(aggregate_topology: NetworkTopology) -> None:
    """A lumped system must be identifiable as such, not silently treated as a feeder."""
    assert aggregate_topology.is_aggregate


def test_topology_requires_at_least_one_node() -> None:
    with pytest.raises(DomainValidationError, match="at least one node"):
        NetworkTopology(nodes=())


def test_topology_requires_a_grid_connection_node(feeder_node: Node) -> None:
    """The system boundary must state where it meets the utility."""
    non_grid = Node(
        node_id=NodeId("node-orphan"), name="Orphan", parent_node_id=NodeId("node-pcc")
    )
    with pytest.raises(DomainValidationError, match="is_grid_connection"):
        NetworkTopology(nodes=(non_grid,))
    assert feeder_node.is_grid_connection is False


def test_grid_connection_node_must_not_have_a_parent(pcc_node: Node) -> None:
    downstream = Node(
        node_id=NodeId("node-second-pcc"),
        name="Second PCC",
        parent_node_id=NodeId("node-pcc"),
        is_grid_connection=True,
    )
    with pytest.raises(DomainValidationError, match="must not have a parent"):
        NetworkTopology(nodes=(pcc_node, downstream))


def test_topology_rejects_duplicate_node_ids(pcc_node: Node) -> None:
    with pytest.raises(DomainValidationError, match="duplicate node_id"):
        NetworkTopology(nodes=(pcc_node, pcc_node))


def test_topology_rejects_unknown_parent(pcc_node: Node, feeder_node: Node) -> None:
    """A dangling parent reference would break every downstream graph walk."""
    with pytest.raises(DomainValidationError, match="unknown parent"):
        NetworkTopology(nodes=(feeder_node,))



def test_node_cannot_be_its_own_parent() -> None:
    with pytest.raises(DomainValidationError, match="own parent"):
        Node(
            node_id=NodeId("node-loop"),
            name="Loop",
            parent_node_id=NodeId("node-loop"),
            is_grid_connection=True,
        )


def test_node_rejects_non_positive_voltage() -> None:
    with pytest.raises(DomainValidationError, match="must be positive"):
        Node(node_id=NodeId("node-a"), name="A", voltage_kv=Quantity(0.0, Unit.KILOVOLT))


def test_node_rejects_unknown_id_type() -> None:
    with pytest.raises(DomainValidationError, match="must be a NodeId"):
        Node(node_id="node-a", name="A")  # type: ignore[arg-type]


def test_element_rejects_unknown_endpoint(topology: NetworkTopology) -> None:
    """An element pointing at a node that does not exist is unresolvable."""
    broken = NetworkElement(
        element_id=NetworkElementId("net-dangling"),
        kind=NetworkElementKind.LINE,
        from_node_id=NodeId("node-pcc"),
        to_node_id=NodeId("node-does-not-exist"),
    )
    with pytest.raises(DomainValidationError, match="unknown to_node_id"):
        NetworkTopology(nodes=topology.nodes, elements=(broken,))


def test_element_cannot_connect_a_node_to_itself() -> None:
    with pytest.raises(DomainValidationError, match="to itself"):
        NetworkElement(
            element_id=NetworkElementId("net-self"),
            kind=NetworkElementKind.LINE,
            from_node_id=NodeId("node-a"),
            to_node_id=NodeId("node-a"),
        )


def test_element_rejects_non_numeric_parameters() -> None:
    with pytest.raises(DomainValidationError, match="must be a real number"):
        NetworkElement(
            element_id=NetworkElementId("net-bad"),
            kind=NetworkElementKind.LINE,
            from_node_id=NodeId("node-a"),
            to_node_id=NodeId("node-b"),
            parameters={"resistance_ohm": "low"},
        )


def test_element_rejects_duplicate_ids(topology: NetworkTopology) -> None:
    with pytest.raises(DomainValidationError, match="duplicate element_id"):
        NetworkTopology(nodes=topology.nodes, elements=topology.elements * 2)


def test_topology_node_lookup_returns_none_for_unknown(pcc_node: Node) -> None:
    topology = NetworkTopology(nodes=(pcc_node,))
    assert topology.node(NodeId("node-absent")) is None


# ======================================================================
# Assets
# ======================================================================


def test_asset_requires_explicit_availability_and_metadata() -> None:
    """These are inputs that materially affect flexibility, so they are required."""
    with pytest.raises(TypeError):
        Asset(  # type: ignore[call-arg]
            asset_type=AssetType.SUBSTATION,
            asset_id=AssetId("asset-sub-1"),
            name="Substation",
            node_id=NodeId("node-pcc"),
            authority=AuthorityLevel.NOT_CONTROLLABLE,
            rated_power=kw(1000.0),
        )


def test_asset_availability_is_validated() -> None:
    for bad in (-0.1, 1.1):
        with pytest.raises(DomainValidationError, match=r"availability must lie in \[0, 1\]"):
            Asset(
                asset_type=AssetType.SUBSTATION,
                asset_id=AssetId("asset-sub-1"),
                name="Substation",
                node_id=NodeId("node-pcc"),
                authority=AuthorityLevel.NOT_CONTROLLABLE,
                rated_power=kw(1000.0),
                availability=bad,
                metadata=None,
            )


def test_asset_available_power_scales_with_availability(battery: Battery) -> None:
    assert battery.available_power.value == battery.rated_power.value


def test_asset_controllability_follows_authority(battery: Battery, advisory_ev: FlexibleLoad) -> None:
    assert battery.is_controllable
    assert not advisory_ev.is_controllable


def test_battery_rejects_soc_bound_above_one(battery: Battery) -> None:
    """An SOC window above 1.0 is physically impossible."""
    with pytest.raises(DomainValidationError, match=r"max_soc must lie in \[0, 1\]"):
        _replace_battery(battery, state=None, min_soc=0.1, max_soc=1.5)


def test_battery_rejects_soc_bound_below_zero(battery: Battery) -> None:
    with pytest.raises(DomainValidationError, match=r"min_soc must lie in \[0, 1\]"):
        _replace_battery(battery, state=None, min_soc=-0.1, max_soc=0.9)


def test_battery_rejects_inverted_soc_window(battery: Battery) -> None:
    with pytest.raises(DomainValidationError, match="must not exceed max_soc"):
        _replace_battery(battery, state=None, min_soc=0.8, max_soc=0.2)


def test_battery_rejects_non_positive_efficiency(battery: Battery) -> None:
    with pytest.raises(DomainValidationError, match=r"round_trip_efficiency must lie in \(0, 1\]"):
        _replace_battery(battery, state=None, min_soc=0.1, max_soc=0.9, eff=0.0)


def test_battery_rejects_negative_usable_energy(battery: Battery) -> None:
    """Negative capacity is impossible."""
    with pytest.raises(DomainValidationError, match="usable_energy must be non-negative"):
        _replace_battery(battery, state=None, min_soc=0.1, max_soc=0.9, energy=Quantity(-1.0, KWH))


def test_battery_exposes_usable_window(battery: Battery) -> None:
    assert battery.usable_soc_window == (0.1, 0.95)


def test_battery_nameplate_duration(battery: Battery) -> None:
    """500 kWh at 250 kW is 2 hours; kWh over kW is a duration."""
    assert battery.nameplate_duration_hours == 2.0


def test_battery_nameplate_duration_undefined_for_incompatible_units(battery: Battery) -> None:
    """Phase 2 performs no unit conversion, so it must decline rather than guess."""
    mismatched = _replace_battery(battery, state=None, min_soc=0.1, max_soc=0.9, power=Quantity(0.25, Unit.MEGAWATT))
    assert mismatched.nameplate_duration_hours is None


def test_battery_nameplate_duration_undefined_for_zero_power(battery: Battery) -> None:
    empty = _replace_battery(battery, state=None, min_soc=0.1, max_soc=0.9, power=kw(0.0))
    assert empty.nameplate_duration_hours is None


def _replace_battery(
    battery: Battery,
    *,
    state: object,
    min_soc: float,
    max_soc: float,
    energy: Quantity | None = None,
    power: Quantity | None = None,
    eff: float | None = None,
) -> Battery:
    return Battery(
        asset_type=battery.asset_type,
        asset_id=battery.asset_id,
        name=battery.name,
        node_id=battery.node_id,
        authority=battery.authority,
        rated_power=power if power is not None else battery.rated_power,
        availability=battery.availability,
        metadata=None,
        usable_energy=energy if energy is not None else battery.usable_energy,
        min_soc=min_soc,
        max_soc=max_soc,
        max_charge_power=battery.max_charge_power,
        max_discharge_power=battery.max_discharge_power,
        round_trip_efficiency=eff if eff is not None else battery.round_trip_efficiency,
    )


def test_solar_rejects_negative_capacity(solar: SolarAsset) -> None:
    with pytest.raises(DomainValidationError, match="capacity_kw must be non-negative"):
        SolarAsset(
            asset_type=solar.asset_type,
            asset_id=solar.asset_id,
            name=solar.name,
            node_id=solar.node_id,
            authority=solar.authority,
            rated_power=kw(400.0),
            availability=1.0,
            metadata=None,
            capacity_kw=kw(-1.0),
        )


def test_solar_rejects_out_of_range_tilt(solar: SolarAsset) -> None:
    with pytest.raises(DomainValidationError, match="tilt_degrees must lie in"):
        SolarAsset(
            asset_type=solar.asset_type,
            asset_id=solar.asset_id,
            name=solar.name,
            node_id=solar.node_id,
            authority=solar.authority,
            rated_power=kw(400.0),
            availability=1.0,
            metadata=None,
            capacity_kw=kw(400.0),
            tilt_degrees=Quantity(95.0, Unit.DIMENSIONLESS_RATIO),
        )


def test_wind_rejects_non_monotonic_speeds() -> None:
    with pytest.raises(DomainValidationError, match="cut_in < rated < cut_out"):
        WindAsset(
            asset_type=AssetType.WIND,
            asset_id=AssetId("asset-wind-1"),
            name="Wind",
            node_id=NodeId("node-feeder-a"),
            authority=AuthorityLevel.CURTAILABLE,
            rated_power=kw(2000.0),
            availability=1.0,
            metadata=None,
            cut_in_speed=Quantity(12.0, Unit.WIND_SPEED),
            rated_speed=Quantity(3.0, Unit.WIND_SPEED),
            cut_out_speed=Quantity(25.0, Unit.WIND_SPEED),
        )


def test_wind_accepts_monotonic_speeds() -> None:
    asset = WindAsset(
        asset_type=AssetType.WIND,
        asset_id=AssetId("asset-wind-1"),
        name="Wind",
        node_id=NodeId("node-feeder-a"),
        authority=AuthorityLevel.CURTAILABLE,
        rated_power=kw(2000.0),
        availability=1.0,
        metadata=None,
        cut_in_speed=Quantity(3.0, Unit.WIND_SPEED),
        rated_speed=Quantity(12.0, Unit.WIND_SPEED),
        cut_out_speed=Quantity(25.0, Unit.WIND_SPEED),
    )
    assert asset.cut_in_speed.value == 3.0


def test_ev_charger_rejects_zero_connectors() -> None:
    with pytest.raises(DomainValidationError, match="at least 1"):
        EVCharger(
            asset_type=AssetType.EV_CHARGER,
            asset_id=AssetId("asset-ev-1"),
            name="EV",
            node_id=NodeId("node-feeder-a"),
            authority=AuthorityLevel.SCHEDULEABLE,
            rated_power=kw(22.0),
            availability=1.0,
            metadata=None,
            min_return_soc=0.8,
            num_connectors=0,
        )


def test_flexible_load_rejects_inverted_power_bounds() -> None:
    with pytest.raises(DomainValidationError, match="must not exceed max_power"):
        FlexibleLoad(
            asset_type=AssetType.FLEXIBLE_LOAD,
            asset_id=AssetId("asset-load-1"),
            name="Load",
            node_id=NodeId("node-feeder-a"),
            authority=AuthorityLevel.SCHEDULEABLE,
            rated_power=kw(100.0),
            availability=1.0,
            metadata=None,
            min_power=kw(80.0),
            max_power=kw(50.0),
            shiftable_energy=0.2,
            priority=1,
            is_hvac=False,
        )


def test_flexible_load_rejects_mismatched_power_units() -> None:
    with pytest.raises(DomainValidationError, match="differs from"):
        FlexibleLoad(
            asset_type=AssetType.FLEXIBLE_LOAD,
            asset_id=AssetId("asset-load-1"),
            name="Load",
            node_id=NodeId("node-feeder-a"),
            authority=AuthorityLevel.SCHEDULEABLE,
            rated_power=kw(100.0),
            availability=1.0,
            metadata=None,
            min_power=Quantity(10.0, Unit.MEGAWATT),
            max_power=kw(50.0),
            shiftable_energy=0.2,
            priority=1,
            is_hvac=False,
        )


def test_asset_class_for_covers_every_asset_type() -> None:
    """Catalogue drift between the enum and the class hierarchy must fail."""
    for asset_type in AssetType:
        cls = asset_class_for(asset_type)
        assert issubclass(cls, Asset)


def test_asset_class_for_rejects_non_enum() -> None:
    with pytest.raises(DomainValidationError, match="must be an AssetType"):
        asset_class_for("battery")  # type: ignore[arg-type]


def test_asset_types_do_not_all_carry_every_field(solar: SolarAsset) -> None:
    """A solar asset must not carry battery fields, and vice versa."""
    assert not hasattr(solar, "usable_energy")
    assert not hasattr(solar, "min_soc")


# ======================================================================
# State
# ======================================================================


def test_balanced_state_validates(system, balanced_state: EnergyState) -> None:
    system.validate_state(balanced_state)
    assert balanced_state.is_decision_grade
    assert balanced_state.origin is StateOrigin.OBSERVED


def test_node_power_residual_is_computed(balanced_state: EnergyState) -> None:
    for node in balanced_state.node_power:
        assert node.is_balanced
        assert abs(node.residual_kw) <= POWER_BALANCE_TOLERANCE


def test_node_power_detects_imbalance() -> None:
    """An unbalanced node is the signature of a sign error or missing asset."""
    node = NodePower(
        node_id=NodeId("node-feeder-a"),
        generation_kw=kw(300.0),
        load_kw=kw(500.0),
        storage_kw=kw(0.0),
        grid_kw=kw(0.0),  # 200 kW short
    )
    assert not node.is_balanced
    assert node.residual_kw == pytest.approx(-200.0)


def test_node_power_uses_explicit_unaccounted_residual() -> None:
    node = NodePower(
        node_id=NodeId("node-a"),
        generation_kw=kw(0.0),
        load_kw=kw(100.0),
        storage_kw=kw(0.0),
        grid_kw=kw(100.0),
        unaccounted_kw=kw(5.0),
    )
    assert node.residual_kw == pytest.approx(5.0)
    assert not node.is_balanced


def test_node_power_rejects_negative_generation() -> None:
    with pytest.raises(DomainValidationError, match="generation_kw must be non-negative"):
        NodePower(
            node_id=NodeId("node-a"),
            generation_kw=kw(-1.0),
            load_kw=kw(0.0),
            storage_kw=kw(0.0),
            grid_kw=kw(0.0),
        )


def test_node_power_reports_unit_errors() -> None:
    node = NodePower(
        node_id=NodeId("node-a"),
        generation_kw=Quantity(0.0, Unit.MEGAWATT),
        load_kw=kw(0.0),
        storage_kw=kw(0.0),
        grid_kw=kw(0.0),
    )
    errors = node.unit_errors()
    assert len(errors) == 1
    assert "must be expressed in kW" in errors[0]


def test_storage_state_rejects_soc_outside_unit_interval(
    battery: Battery, clean_quality: DataQuality
) -> None:
    """SOC > 1 and SOC < 0 are both physically impossible."""
    for bad in (1.01, -0.01):
        with pytest.raises(DomainValidationError, match=r"state_of_charge must lie in \[0, 1\]"):
            StorageState(
                asset_id=battery.asset_id,
                state_of_charge=bad,
                charge_power_kw=kw(0.0),
                discharge_power_kw=kw(0.0),
                available_energy=Quantity(300.0, KWH),
                quality=clean_quality,
            )


def test_storage_state_rejects_simultaneous_charge_and_discharge(
    battery: Battery, clean_quality: DataQuality
) -> None:
    with pytest.raises(DomainValidationError, match="cannot charge and discharge"):
        StorageState(
            asset_id=battery.asset_id,
            state_of_charge=0.5,
            charge_power_kw=kw(10.0),
            discharge_power_kw=kw(10.0),
            available_energy=Quantity(300.0, KWH),
            quality=clean_quality,
        )


def test_storage_state_net_power_sign(battery: Battery, clean_quality: DataQuality) -> None:
    charging = StorageState(
        asset_id=battery.asset_id,
        state_of_charge=0.5,
        charge_power_kw=kw(50.0),
        discharge_power_kw=kw(0.0),
        available_energy=Quantity(300.0, KWH),
        quality=clean_quality,
    )
    assert charging.net_power_kw == -50.0


def test_renewable_state_curtailment_fraction(solar: SolarAsset, clean_quality: DataQuality) -> None:
    state = RenewableState(
        asset_id=solar.asset_id,
        available_power=kw(400.0),
        actual_power=kw(300.0),
        quality=clean_quality,
    )
    assert state.curtailment_kw == pytest.approx(100.0)
    assert state.curtailment_fraction == pytest.approx(0.25)


def test_renewable_state_zero_available_means_no_curtailment(
    solar: SolarAsset, clean_quality: DataQuality
) -> None:
    state = RenewableState(
        asset_id=solar.asset_id,
        available_power=kw(0.0),
        actual_power=kw(0.0),
        quality=clean_quality,
    )
    assert state.curtailment_fraction == 0.0


def test_demand_state_rejects_category_sum_mismatch(clean_quality: DataQuality) -> None:
    """A category breakdown that does not sum to the total is a silent data bug."""
    with pytest.raises(DomainValidationError, match="must sum to total_demand_kw"):
        DemandState(
            total_demand_kw=kw(500.0),
            quality=clean_quality,
            by_category={"residential": 100.0},
        )


def test_demand_state_accepts_matching_categories(clean_quality: DataQuality) -> None:
    state = DemandState(
        total_demand_kw=kw(500.0),
        quality=clean_quality,
        by_category={"residential": 300.0, "commercial": 200.0},
    )
    assert sum(state.by_category.values()) == pytest.approx(500.0)


def test_ev_state_rejects_deferrable_above_charging(clean_quality: DataQuality) -> None:
    with pytest.raises(DomainValidationError, match="must not exceed charging_power_kw"):
        EVState(
            connected_vehicles=5,
            available_connectors=3,
            charging_power_kw=kw(50.0),
            deferrable_power_kw=kw(80.0),
            quality=clean_quality,
        )


def test_grid_state_net_flow(clean_quality: DataQuality) -> None:
    state = GridState(import_kw=kw(100.0), export_kw=kw(30.0), quality=clean_quality)
    assert state.net_kw == pytest.approx(70.0)


def test_grid_state_rejects_non_positive_voltage(clean_quality: DataQuality) -> None:
    with pytest.raises(DomainValidationError, match="must be positive"):
        GridState(
            import_kw=kw(0.0),
            export_kw=kw(0.0),
            quality=clean_quality,
            voltage_by_node={"node-pcc": Quantity(-1.0, Unit.VOLT)},
        )


def test_state_quality_flags_union(balanced_state: EnergyState, battery: Battery) -> None:
    assert balanced_state.quality_flags() == {QualityFlag.OK}


def test_state_is_not_decision_grade_when_any_component_is_degraded(
    provenance, clean_quality: DataQuality, battery: Battery, solar: SolarAsset
) -> None:
    """A single degraded component must disqualify the whole state by default."""
    degraded = DataQuality(flags=frozenset({QualityFlag.STALE}))
    state = _state_with_quality(provenance, clean_quality, battery, solar, storage_quality=degraded)
    assert not state.is_decision_grade
    assert QualityFlag.STALE in state.quality_flags()


def _state_with_quality(provenance, clean_quality, battery, solar, *, storage_quality):
    return __import__(
        "energy_intelligence.domain", fromlist=["EnergyState"]
    ).EnergyState(
        timestamp="2026-01-01T00:00:00+00:00",
        origin=StateOrigin.OBSERVED,
        node_power=(
            NodePower(
                node_id=NodeId("node-pcc"),
                generation_kw=kw(0.0),
                load_kw=kw(0.0),
                storage_kw=kw(0.0),
                grid_kw=kw(0.0),
            ),
            NodePower(
                node_id=NodeId("node-feeder-a"),
                generation_kw=kw(300.0),
                load_kw=kw(500.0),
                storage_kw=kw(0.0),
                grid_kw=kw(200.0),
            ),
        ),
        demand=DemandState(total_demand_kw=kw(500.0), quality=clean_quality),
        grid=GridState(import_kw=kw(200.0), export_kw=kw(0.0), quality=clean_quality),
        provenance=provenance,
        storage=(
            StorageState(
                asset_id=battery.asset_id,
                state_of_charge=0.6,
                charge_power_kw=kw(0.0),
                discharge_power_kw=kw(0.0),
                available_energy=Quantity(300.0, KWH),
                quality=storage_quality,
            ),
        ),
        renewables=(
            RenewableState(
                asset_id=solar.asset_id,
                available_power=kw(320.0),
                actual_power=kw(300.0),
                quality=clean_quality,
            ),
        ),
    )


def test_state_requires_a_timestamp(balanced_state: EnergyState) -> None:
    import dataclasses

    with pytest.raises(DomainValidationError, match="timestamp"):
        dataclasses.replace(balanced_state, timestamp="")


def test_state_origin_is_explicit_and_typed(balanced_state: EnergyState) -> None:
    """Observed, simulated and predicted must be distinguishable."""
    assert isinstance(balanced_state.origin, StateOrigin)
    assert balanced_state.origin.value == "observed"


# ======================================================================
# Actions
# ======================================================================


def test_valid_action_is_constructed(discharge_action: Action) -> None:
    assert discharge_action.end_step == 95
    assert discharge_action.is_executable
    assert discharge_action.covers_step(92)
    assert not discharge_action.covers_step(91)
    assert not discharge_action.covers_step(96)


def test_action_rejects_observation_role(provenance, battery: Battery) -> None:
    """The action/observation boundary, enforced at construction."""
    with pytest.raises(DomainValidationError, match="not valid on an Action"):
        Action(
            action_id=ActionId("act-bad"),
            action_type=ActionType.BATTERY_DISCHARGE,
            target_asset_id=battery.asset_id,
            setpoint=kw(10.0),
            role=VariableRole.OBSERVATION,
            issued_under=AuthorityLevel.DISPATCHABLE,
            start_step=0,
            duration_steps=1,
            provenance=provenance,
        )


def test_action_rejects_advisory_authority(provenance, battery: Battery) -> None:
    """Advisory output is not an action."""
    with pytest.raises(DomainValidationError, match="not a controllable authority level"):
        Action(
            action_id=ActionId("act-bad"),
            action_type=ActionType.BATTERY_DISCHARGE,
            target_asset_id=battery.asset_id,
            setpoint=kw(10.0),
            role=VariableRole.ACTION,
            issued_under=AuthorityLevel.ADVISORY_ONLY,
            start_step=0,
            duration_steps=1,
            provenance=provenance,
        )


def test_action_rejects_wrong_unit(provenance, battery: Battery) -> None:
    with pytest.raises(DomainValidationError, match="must be expressed in kW"):
        Action(
            action_id=ActionId("act-bad"),
            action_type=ActionType.BATTERY_DISCHARGE,
            target_asset_id=battery.asset_id,
            setpoint=Quantity(10.0, Unit.DEGREE_CELSIUS),
            role=VariableRole.ACTION,
            issued_under=AuthorityLevel.DISPATCHABLE,
            start_step=0,
            duration_steps=1,
            provenance=provenance,
        )


def test_action_rejects_negative_non_negative_setpoint(provenance, battery: Battery) -> None:
    with pytest.raises(DomainValidationError, match="must be non-negative"):
        Action(
            action_id=ActionId("act-bad"),
            action_type=ActionType.BATTERY_DISCHARGE,
            target_asset_id=battery.asset_id,
            setpoint=kw(-10.0),
            role=VariableRole.ACTION,
            issued_under=AuthorityLevel.DISPATCHABLE,
            start_step=0,
            duration_steps=1,
            provenance=provenance,
        )


def test_action_rejects_zero_setpoint_for_signed_actions(provenance) -> None:
    """A zero setpoint is the absence of an action, not an action."""
    with pytest.raises(DomainValidationError, match="must be non-zero"):
        Action(
            action_id=ActionId("act-zero"),
            action_type=ActionType.FLEXIBLE_LOAD_SHIFT,
            target_asset_id=AssetId("asset-load-1"),
            setpoint=kw(0.0),
            role=VariableRole.ACTION,
            issued_under=AuthorityLevel.SCHEDULEABLE,
            start_step=0,
            duration_steps=1,
            provenance=provenance,
        )


def test_action_rejects_non_positive_duration(provenance, battery: Battery) -> None:
    with pytest.raises(DomainValidationError, match="at least 1"):
        Action(
            action_id=ActionId("act-bad"),
            action_type=ActionType.BATTERY_DISCHARGE,
            target_asset_id=battery.asset_id,
            setpoint=kw(10.0),
            role=VariableRole.ACTION,
            issued_under=AuthorityLevel.DISPATCHABLE,
            start_step=0,
            duration_steps=0,
            provenance=provenance,
        )


def test_action_validates_against_system(discharge_action: Action, system) -> None:
    system.validate_actions((discharge_action,))


def test_action_rejects_unknown_target_asset(discharge_action: Action, system) -> None:
    import dataclasses

    orphan = dataclasses.replace(
        discharge_action, target_asset_id=AssetId("asset-does-not-exist")
    )
    with pytest.raises(DomainValidationError, match="unknown asset"):
        system.validate_actions((orphan,))


def test_advisory_asset_cannot_be_dispatched(
    provenance, advisory_ev: FlexibleLoad, system, time_base, topology, battery, solar
) -> None:
    """The tiered authority model, enforced end to end."""
    from energy_intelligence.domain import EnergySystem

    assert not advisory_ev.is_controllable

    # The advisory asset must actually belong to the system for this to be a
    # test of authority rather than of asset lookup.
    owning_system = EnergySystem(
        system_id=system.system_id,
        name=system.name,
        description=system.description,
        time_base=time_base,
        topology=topology,
        assets=system.assets + (advisory_ev,),
        provenance=provenance,
    )
    action = Action(
        action_id=ActionId("act-advisory"),
        action_type=ActionType.FLEXIBLE_LOAD_SHIFT,
        target_asset_id=advisory_ev.asset_id,
        setpoint=kw(-20.0),
        role=VariableRole.ACTION,
        issued_under=AuthorityLevel.SCHEDULEABLE,
        start_step=0,
        duration_steps=1,
        provenance=provenance,
    )
    with pytest.raises(DomainValidationError, match="cannot be dispatched"):
        owning_system.validate_actions((action,))


def test_scheduling_authority_does_not_permit_curtailment(provenance, solar: SolarAsset) -> None:
    """Curtailment is a direct output reduction, not a shift in time.

    An earlier draft ranked authority levels and implied that scheduling
    authority could curtail. That implication is not real, so sufficiency is
    now an explicit per-action-type set.
    """
    action = Action(
        action_id=ActionId("act-curtail"),
        action_type=ActionType.RENEWABLE_CURTAILMENT,
        target_asset_id=solar.asset_id,
        setpoint=kw(50.0),
        role=VariableRole.ACTION,
        issued_under=AuthorityLevel.SCHEDULEABLE,
        start_step=0,
        duration_steps=1,
        provenance=provenance,
    )
    assert not action.has_sufficient_authority()

    curtailment = Action(
        action_id=ActionId("act-curtail-ok"),
        action_type=ActionType.RENEWABLE_CURTAILMENT,
        target_asset_id=solar.asset_id,
        setpoint=kw(50.0),
        role=VariableRole.ACTION,
        issued_under=AuthorityLevel.CURTAILABLE,
        start_step=0,
        duration_steps=1,
        provenance=provenance,
    )
    assert curtailment.has_sufficient_authority()


def test_dispatchable_authority_satisfies_schedule_and_curtail(provenance) -> None:
    """Full dispatch authority implies the weaker capabilities."""
    shift = Action(
        action_id=ActionId("act-strong-shift"),
        action_type=ActionType.FLEXIBLE_LOAD_SHIFT,
        target_asset_id=AssetId("asset-load-1"),
        setpoint=kw(-20.0),
        role=VariableRole.ACTION,
        issued_under=AuthorityLevel.DISPATCHABLE,
        start_step=0,
        duration_steps=1,
        provenance=provenance,
    )
    assert shift.has_sufficient_authority()

    assert sufficient_authorities_for(ActionType.BATTERY_DISCHARGE) == frozenset(
        {AuthorityLevel.DISPATCHABLE}
    )
    assert sufficient_authorities_for(ActionType.RENEWABLE_CURTAILMENT) == frozenset(
        {AuthorityLevel.DISPATCHABLE, AuthorityLevel.CURTAILABLE}
    )


def test_every_action_type_maps_to_a_required_authority() -> None:
    """Catalogue drift between ActionType and the requirement table must fail."""
    for action_type in ActionType:
        assert action_type in ACTION_REQUIREMENTS


def test_every_action_type_has_a_unit_and_sign_rule() -> None:
    for action_type in ActionType:
        assert isinstance(unit_for_action(action_type), Unit)
        assert sign_constraint_for(action_type) in {"non_negative", "signed", "any"}


def test_unit_and_authority_helpers_reject_non_enums() -> None:
    with pytest.raises(DomainValidationError):
        unit_for_action("battery_charge")  # type: ignore[arg-type]
    with pytest.raises(DomainValidationError):
        required_authority_for("battery_charge")  # type: ignore[arg-type]


# ======================================================================
# Constraints
# ======================================================================


def test_constraint_defaults_unit_to_category_unit(constraint: Constraint) -> None:
    assert constraint.unit is Unit.FRACTION
    assert constraint.unit == unit_for_constraint(ConstraintCategory.STATE_OF_CHARGE_LIMITS)


def test_constraint_rejects_wrong_unit(provenance, battery: Battery) -> None:
    with pytest.raises(DomainValidationError, match="must be expressed in fraction"):
        Constraint(
            constraint_id=ConstraintId("con-bad"),
            category=ConstraintCategory.STATE_OF_CHARGE_LIMITS,
            description="SOC bounds",
            provenance=provenance,
            applies_to_asset_ids=(battery.asset_id,),
            lower_bound=0.1,
            unit=KW,
        )


def test_constraint_requires_a_scope(provenance) -> None:
    """An unscoped constraint cannot be checked against anything."""
    with pytest.raises(DomainValidationError, match="at least one asset or node"):
        Constraint(
            constraint_id=ConstraintId("con-unscoped"),
            category=ConstraintCategory.POWER_BALANCE,
            description="System power balance",
            provenance=provenance,
        )


def test_constraint_rejects_inverted_bounds(provenance, battery: Battery) -> None:
    with pytest.raises(DomainValidationError, match="exceeds upper_bound"):
        Constraint(
            constraint_id=ConstraintId("con-bad"),
            category=ConstraintCategory.STATE_OF_CHARGE_LIMITS,
            description="SOC bounds",
            provenance=provenance,
            applies_to_asset_ids=(battery.asset_id,),
            lower_bound=0.9,
            upper_bound=0.1,
        )


def test_physics_dependent_constraints_are_flagged(provenance, battery: Battery) -> None:
    """Voltage limits exist in the vocabulary but need a power-flow solver."""
    voltage = Constraint(
        constraint_id=ConstraintId("con-voltage"),
        category=ConstraintCategory.VOLTAGE_LIMITS,
        description="Node voltage stays within statutory limits",
        provenance=provenance,
        applies_to_node_ids=(NodeId("node-feeder-a"),),
        lower_bound=0.95,
        upper_bound=1.05,
        unit=Unit.VOLT,
    )
    assert voltage.needs_power_flow
    assert voltage.unit is Unit.VOLT


def test_no_constraint_is_enforceable_in_phase_2(constraint: Constraint) -> None:
    """Phase 2 declares; it does not enforce. Stated honestly on the object."""
    assert constraint.is_enforceable_in_phase_2 is False


# ======================================================================
# Objectives
# ======================================================================


def test_objective_direction_and_unit_come_from_the_category(objective: Objective) -> None:
    assert objective.direction.value == "minimize"
    assert objective.unit == objective_metadata(objective.category).unit


def test_objective_has_no_weight_field() -> None:
    """Pre-emptive weighting would hide the trade-offs. Enforced structurally."""
    assert "weight" not in Objective.__dataclass_fields__


def test_objective_conflicts_are_symmetric(objective: Objective, provenance) -> None:
    from energy_intelligence.domain import Objective as O

    utilisation = O(
        objective_id=ObjectiveId("obj-renewable-util"),
        category=ObjectiveCategory.RENEWABLE_UTILIZATION,
        measurement="Share of available renewable energy that is used",
        why_it_matters="Renewable integration is a primary project goal",
        provenance=provenance,
    )
    assert objective.conflicts(utilisation)
    assert utilisation.conflicts(objective)


def test_energy_cost_and_peak_demand_conflict(objective: Objective, provenance) -> None:
    """Shaving peak costs money, so the two objectives genuinely compete.

    Recording this is the point of the conflict table: it stops Phase 10 from
    treating the objective set as independent.
    """
    peak = Objective(
        objective_id=ObjectiveId("obj-peak"),
        category=ObjectiveCategory.PEAK_DEMAND,
        measurement="Maximum demand over the horizon",
        why_it_matters="Peak management reduces network loading",
        provenance=provenance,
    )
    assert objective.conflicts(peak)
    assert peak.conflicts(objective)


def test_objectives_of_the_same_category_do_not_conflict(objective: Objective, provenance) -> None:
    from energy_intelligence.domain import Objective as O

    second_cost = O(
        objective_id=ObjectiveId("obj-energy-cost-2"),
        category=ObjectiveCategory.ENERGY_COST,
        measurement="Same objective expressed against a different tariff window",
        why_it_matters="A second cost view of the same quantity",
        provenance=provenance,
    )
    assert not objective.conflicts(second_cost)


def test_unmet_demand_and_constraint_violation_do_not_conflict(objective: Objective, provenance) -> None:
    """Two independent safety measures; neither trades against the other."""
    from energy_intelligence.domain import Objective as O

    unmet = O(
        objective_id=ObjectiveId("obj-unmet"),
        category=ObjectiveCategory.UNMET_DEMAND,
        measurement="Energy of demand not served",
        why_it_matters="Unmet demand is the worst possible outcome",
        provenance=provenance,
    )
    violation = O(
        objective_id=ObjectiveId("obj-violation"),
        category=ObjectiveCategory.CONSTRAINT_VIOLATION,
        measurement="Count of violated constraints over the horizon",
        why_it_matters="Constraint violations indicate an infeasible or unsafe plan",
        provenance=provenance,
    )
    assert not unmet.conflicts(violation)
    assert not violation.conflicts(unmet)
    assert unmet.conflicts_with == ()


def test_every_objective_category_has_metadata() -> None:
    for category in ObjectiveCategory:
        assert objective_metadata(category).direction is not None


def test_objective_rejects_non_numeric_target(objective: Objective, provenance) -> None:
    import dataclasses

    with pytest.raises(DomainValidationError, match="target_value"):
        dataclasses.replace(objective, target_value="high")


# ======================================================================
# Observations - the action/observation boundary
# ======================================================================


def test_observation_accepts_observation_role(provenance, battery: Battery, clean_quality) -> None:
    record = ObservationRecord(
        variable="battery_state_of_charge",
        role=VariableRole.OBSERVATION,
        value=Quantity(0.6, Unit.FRACTION),
        quality=clean_quality,
        provenance=provenance,
        timestamp="2026-01-01T00:00:00+00:00",
        asset_id=battery.asset_id,
    )
    assert record.is_actionable_input
    assert not record.is_missing


def test_observation_rejects_action_role(provenance, battery: Battery, clean_quality) -> None:
    """The mirror of the Action check: an action variable cannot be observed."""
    with pytest.raises(DomainValidationError, match="not valid on an ObservationRecord"):
        ObservationRecord(
            variable="battery_charge_power",
            role=VariableRole.ACTION,
            value=kw(10.0),
            quality=clean_quality,
            provenance=provenance,
            timestamp="2026-01-01T00:00:00+00:00",
            asset_id=battery.asset_id,
        )


def test_observation_rejects_constraint_and_objective_roles(
    provenance, battery: Battery, clean_quality
) -> None:
    for role in (VariableRole.CONSTRAINT, VariableRole.OBJECTIVE):
        with pytest.raises(DomainValidationError, match="not valid on an ObservationRecord"):
            ObservationRecord(
                variable="soc_upper_bound",
                role=role,
                value=Quantity(0.95, Unit.FRACTION),
                quality=clean_quality,
                provenance=provenance,
                timestamp="2026-01-01T00:00:00+00:00",
                asset_id=battery.asset_id,
            )


def test_observation_must_be_scoped(provenance, clean_quality) -> None:
    """An unscoped observation cannot be attributed to part of the system."""
    with pytest.raises(DomainValidationError, match="asset_id, a node_id"):
        ObservationRecord(
            variable="total_demand",
            role=VariableRole.OBSERVATION,
            value=kw(100.0),
            quality=clean_quality,
            provenance=provenance,
            timestamp="2026-01-01T00:00:00+00:00",
        )


def test_observation_rejects_non_conforming_variable_name(
    provenance, battery: Battery, clean_quality
) -> None:
    for bad in ("Total Demand", "demand-kw", "demand!"):
        with pytest.raises(DomainValidationError, match="must match"):
            ObservationRecord(
                variable=bad,
                role=VariableRole.OBSERVATION,
                value=kw(1.0),
                quality=clean_quality,
                provenance=provenance,
                timestamp="2026-01-01T00:00:00+00:00",
                asset_id=battery.asset_id,
            )


def test_observation_reports_missing_and_actionability(
    provenance, battery: Battery
) -> None:
    missing = ObservationRecord(
        variable="battery_state_of_charge",
        role=VariableRole.OBSERVATION,
        value=Quantity(0.6, Unit.FRACTION),
        quality=DataQuality(flags=frozenset({QualityFlag.MISSING})),
        provenance=provenance,
        timestamp="2026-01-01T00:00:00+00:00",
        asset_id=battery.asset_id,
    )
    assert missing.is_missing
    assert not missing.is_actionable_input
