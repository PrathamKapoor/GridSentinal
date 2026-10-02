"""Tests for EnergySystem cross-entity validation, forecasts and serialization."""

from __future__ import annotations

import dataclasses
import json

import pytest

from energy_intelligence.domain import (
    SCHEMA_VERSION,
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
    EnergyState,
    EnergySystem,
    Forecast,
    ForecastId,
    GridState,
    NetworkTopology,
    Node,
    NodeId,
    NodePower,
    ObservationRecord,
    Objective,
    ObjectiveCategory,
    ObjectiveId,
    Quantity,
    QualityFlag,
    SerializationError,
    StateOrigin,
    SystemId,
    UncertaintyEstimate,
    UncertaintyKind,
    Unit,
    UncertaintyEstimate,
    UncertaintyKind,
    VariableRole,
    decode,
    encode,
    to_dict,
)

KW = Unit.KILOWATT
KWH = Unit.KILOWATT_HOURS


def kw(value: float) -> Quantity:
    return Quantity(value, KW)


# ======================================================================
# EnergySystem referential integrity
# ======================================================================


def test_system_exposes_lookups(system: EnergySystem, battery: Battery) -> None:
    assert system.asset(battery.asset_id) is battery
    assert system.asset(AssetId("asset-absent")) is None
    assert battery.asset_id in system.asset_ids
    assert len(system.assets_at_node(NodeId("node-feeder-a"))) == 2
    assert len(system.controllable_assets()) == 2


def test_system_is_not_aggregate(
    system: EnergySystem, aggregate_topology, time_base, battery, provenance
) -> None:
    assert not system.is_aggregate_topology
    # A lumped system holds one node, so the battery must attach to that node.
    on_pcc = dataclasses.replace(battery, node_id=NodeId("node-pcc"))
    lumped = EnergySystem(
        system_id=SystemId("sys-lumped"),
        name="Lumped",
        description="Single node",
        time_base=time_base,
        topology=aggregate_topology,
        assets=(on_pcc,),
        provenance=provenance,
    )
    assert lumped.is_aggregate_topology


def test_system_rejects_asset_on_unknown_node(time_base, topology, provenance, battery) -> None:
    """A dangling asset reference breaks every downstream per-node calculation."""
    stranded = dataclasses.replace(battery, node_id=NodeId("node-nowhere"))
    with pytest.raises(DomainValidationError, match="not in the topology"):
        EnergySystem(
            system_id=SystemId("sys-bad"),
            name="Bad",
            description="Bad",
            time_base=time_base,
            topology=topology,
            assets=(stranded,),
            provenance=provenance,
        )


def test_system_rejects_duplicate_asset_ids(time_base, topology, provenance, battery) -> None:
    with pytest.raises(DomainValidationError, match="duplicate asset_id"):
        EnergySystem(
            system_id=SystemId("sys-bad"),
            name="Bad",
            description="Bad",
            time_base=time_base,
            topology=topology,
            assets=(battery, battery),
            provenance=provenance,
        )


def test_system_rejects_constraint_referencing_unknown_asset(
    system: EnergySystem, constraint: Constraint
) -> None:
    orphan = dataclasses.replace(
        constraint, applies_to_asset_ids=(AssetId("asset-does-not-exist"),)
    )
    with pytest.raises(DomainValidationError, match="unknown asset"):
        dataclasses.replace(system, constraints=(orphan,))


def test_system_rejects_constraint_referencing_unknown_node(
    system: EnergySystem, constraint: Constraint
) -> None:
    orphan = dataclasses.replace(constraint, applies_to_asset_ids=(), applies_to_node_ids=(NodeId("node-nope"),))
    with pytest.raises(DomainValidationError, match="unknown node"):
        dataclasses.replace(system, constraints=(orphan,))


def test_system_rejects_duplicate_objectives(system: EnergySystem, objective: Objective) -> None:
    with pytest.raises(DomainValidationError, match="duplicate objective_id"):
        dataclasses.replace(system, objectives=(objective, objective))


def test_system_accepts_valid_constraints_and_objectives(
    system: EnergySystem, constraint: Constraint, objective: Objective
) -> None:
    enriched = dataclasses.replace(system, constraints=(constraint,), objectives=(objective,))
    assert len(enriched.constraints) == 1
    assert len(enriched.objectives) == 1


def test_system_controllable_capacity_excludes_non_kw(system: EnergySystem) -> None:
    """Phase 2 performs no unit conversion, so non-kW ratings are skipped."""
    assert system.controllable_capacity_kw == pytest.approx(650.0)


def test_system_rejects_wrong_field_types(system: EnergySystem) -> None:
    with pytest.raises(DomainValidationError, match="assets must be a tuple"):
        dataclasses.replace(system, assets=[1, 2, 3])
    with pytest.raises(DomainValidationError, match="must be a TimeBase"):
        dataclasses.replace(system, time_base=None)
    with pytest.raises(DomainValidationError, match="must be a Provenance"):
        dataclasses.replace(system, provenance=None)


# ======================================================================
# State / action validation against the system
# ======================================================================


def test_state_must_cover_every_node(system: EnergySystem, balanced_state: EnergyState) -> None:
    """A state that omits a node would silently hide that node's power."""
    partial = dataclasses.replace(
        balanced_state, node_power=balanced_state.node_power[:1]
    )
    with pytest.raises(DomainValidationError, match="does not report power at node"):
        system.validate_state(partial)


def test_state_must_not_report_unknown_nodes(
    system: EnergySystem, balanced_state: EnergyState
) -> None:
    extra = NodePower(
        node_id=NodeId("node-ghost"),
        generation_kw=kw(0.0),
        load_kw=kw(0.0),
        storage_kw=kw(0.0),
        grid_kw=kw(0.0),
    )
    with pytest.raises(DomainValidationError, match="unknown node"):
        system.validate_state(dataclasses.replace(balanced_state, node_power=balanced_state.node_power + (extra,)))


def test_state_must_not_report_storage_for_a_non_battery(
    system: EnergySystem, balanced_state: EnergyState, solar
) -> None:
    """Reporting a battery state for an asset the system does not own as a battery
    is the failure that would corrupt every flexibility figure downstream."""
    impostor = dataclasses.replace(balanced_state.storage[0], asset_id=solar.asset_id)
    with pytest.raises(DomainValidationError, match="does not own as a battery"):
        system.validate_state(dataclasses.replace(balanced_state, storage=(impostor,)))


def test_state_must_not_report_renewables_for_a_battery(
    system: EnergySystem, balanced_state: EnergyState, battery
) -> None:
    impostor = dataclasses.replace(balanced_state.renewables[0], asset_id=battery.asset_id)
    with pytest.raises(DomainValidationError, match="solar or wind"):
        system.validate_state(dataclasses.replace(balanced_state, renewables=(impostor,)))


def test_state_construction_rejects_unbalanced_node(provenance, clean_quality) -> None:
    """Balance is enforced at construction, not only when validated against a system."""
    with pytest.raises(DomainValidationError, match="power balance .* does not close"):
        EnergyState(
            timestamp="2026-01-01T00:00:00+00:00",
            origin=StateOrigin.OBSERVED,
            node_power=(
                NodePower(
                    node_id=NodeId("node-pcc"),
                    generation_kw=kw(0.0),
                    load_kw=kw(500.0),
                    storage_kw=kw(0.0),
                    grid_kw=kw(0.0),
                ),
            ),
            demand=DemandState(total_demand_kw=kw(500.0), quality=clean_quality),
            grid=GridState(import_kw=kw(0.0), export_kw=kw(0.0), quality=clean_quality),
            provenance=provenance,
        )


def test_state_construction_rejects_duplicate_nodes(provenance, clean_quality) -> None:
    node = NodePower(
        node_id=NodeId("node-pcc"),
        generation_kw=kw(0.0),
        load_kw=kw(0.0),
        storage_kw=kw(0.0),
        grid_kw=kw(0.0),
    )
    with pytest.raises(DomainValidationError, match="duplicate node"):
        EnergyState(
            timestamp="2026-01-01T00:00:00+00:00",
            origin=StateOrigin.OBSERVED,
            node_power=(node, node),
            demand=DemandState(total_demand_kw=kw(0.0), quality=clean_quality),
            grid=GridState(import_kw=kw(0.0), export_kw=kw(0.0), quality=clean_quality),
            provenance=provenance,
        )


def test_state_rejects_non_kw_units_at_construction(provenance, clean_quality) -> None:
    """A per-node power balance stated in MW would be silently wrong."""
    with pytest.raises(DomainValidationError, match="must be expressed in kW"):
        EnergyState(
            timestamp="2026-01-01T00:00:00+00:00",
            origin=StateOrigin.OBSERVED,
            node_power=(
                NodePower(
                    node_id=NodeId("node-pcc"),
                    generation_kw=Quantity(0.0, Unit.MEGAWATT),
                    load_kw=kw(0.0),
                    storage_kw=kw(0.0),
                    grid_kw=kw(0.0),
                ),
            ),
            demand=DemandState(total_demand_kw=kw(0.0), quality=clean_quality),
            grid=GridState(import_kw=kw(0.0), export_kw=kw(0.0), quality=clean_quality),
            provenance=provenance,
        )


def test_actions_against_system_reports_every_failure(system: EnergySystem, discharge_action: Action) -> None:
    from energy_intelligence.domain import validate_actions_against_system

    orphans = (
        dataclasses.replace(discharge_action, target_asset_id=AssetId("asset-nope-1")),
        dataclasses.replace(discharge_action, action_id=ActionId("act-two"), target_asset_id=AssetId("asset-nope-2")),
    )
    with pytest.raises(DomainValidationError) as excinfo:
        validate_actions_against_system(system, orphans)
    assert len(excinfo.value.errors) == 2


def test_system_validates_observations(system: EnergySystem, provenance, battery) -> None:
    good = ObservationRecord(
        variable="battery_state_of_charge",
        role=VariableRole.OBSERVATION,
        value=Quantity(0.6, Unit.FRACTION),
        quality=DataQuality(flags=frozenset({QualityFlag.OK})),
        provenance=provenance,
        timestamp="2026-01-01T00:00:00+00:00",
        asset_id=battery.asset_id,
    )
    system.validate_observations((good,))

    bad_asset = dataclasses.replace(good, asset_id=AssetId("asset-unknown"))
    bad_node = dataclasses.replace(good, asset_id=None, node_id=NodeId("node-unknown"))
    with pytest.raises(DomainValidationError) as excinfo:
        system.validate_observations((bad_asset, bad_node))
    assert len(excinfo.value.errors) == 2


def test_system_validates_forecast_horizon(system: EnergySystem, provenance) -> None:
    inside = Forecast(
        forecast_id=ForecastId("fc-1"),
        variable="total_demand",
        role=VariableRole.OBSERVATION,
        target_time="2026-01-02T00:00:00+00:00",
        issued_at="2026-01-01T00:00:00+00:00",
        horizon_steps=96,
        value=kw(500.0),
        provenance=provenance,
    )
    system.validate_forecasts((inside,))

    beyond = dataclasses.replace(inside, horizon_steps=1000)
    with pytest.raises(DomainValidationError, match="beyond the system's configured horizon"):
        system.validate_forecasts((beyond,))


# ======================================================================
# Forecast
# ======================================================================


def test_valid_forecast(provenance) -> None:
    forecast = Forecast(
        forecast_id=ForecastId("fc-demand-1"),
        variable="total_demand",
        role=VariableRole.OBSERVATION,
        target_time="2026-01-01T06:00:00+00:00",
        issued_at="2026-01-01T00:00:00+00:00",
        horizon_steps=24,
        value=kw(650.0),
        provenance=provenance,
    )
    assert not forecast.has_uncertainty
    assert not forecast.is_quantified_uncertainty


def test_forecast_rejects_issuance_after_the_target_time(provenance) -> None:
    """A forecast stamped at or after its target time is hindsight, not a forecast."""
    with pytest.raises(DomainValidationError, match="must be strictly before"):
        Forecast(
            forecast_id=ForecastId("fc-bad"),
            variable="total_demand",
            role=VariableRole.OBSERVATION,
            target_time="2026-01-01T00:00:00+00:00",
            issued_at="2026-01-01T06:00:00+00:00",
            horizon_steps=24,
            value=kw(650.0),
            provenance=provenance,
        )


def test_forecast_rejects_invalid_timestamps(provenance) -> None:
    with pytest.raises(DomainValidationError, match="not valid ISO-8601"):
        Forecast(
            forecast_id=ForecastId("fc-bad"),
            variable="total_demand",
            role=VariableRole.OBSERVATION,
            target_time="not-a-time",
            issued_at="2026-01-01T00:00:00+00:00",
            horizon_steps=24,
            value=kw(650.0),
            provenance=provenance,
        )


def test_forecast_rejects_action_role(provenance) -> None:
    with pytest.raises(DomainValidationError, match="role must be 'observation'"):
        Forecast(
            forecast_id=ForecastId("fc-bad"),
            variable="battery_charge_power",
            role=VariableRole.ACTION,
            target_time="2026-01-01T06:00:00+00:00",
            issued_at="2026-01-01T00:00:00+00:00",
            horizon_steps=24,
            value=kw(50.0),
            provenance=provenance,
        )


def test_forecast_uncertainty_must_match_value_unit(provenance) -> None:
    """Uncertainty in the wrong unit would silently misstate the forecast."""
    with pytest.raises(DomainValidationError, match="must match the forecast value unit"):
        Forecast(
            forecast_id=ForecastId("fc-bad"),
            variable="total_demand",
            role=VariableRole.OBSERVATION,
            target_time="2026-01-01T06:00:00+00:00",
            issued_at="2026-01-01T00:00:00+00:00",
            horizon_steps=24,
            value=kw(650.0),
            provenance=provenance,
            uncertainty=UncertaintyEstimate(
                kind=UncertaintyKind.MODEL, unit=Unit.MEGAWATT, relative_std=0.1
            ),
        )


def test_forecast_with_quantified_uncertainty(provenance) -> None:
    forecast = Forecast(
        forecast_id=ForecastId("fc-unc"),
        variable="total_demand",
        role=VariableRole.OBSERVATION,
        target_time="2026-01-01T06:00:00+00:00",
        issued_at="2026-01-01T00:00:00+00:00",
        horizon_steps=24,
        value=kw(650.0),
        provenance=provenance,
        uncertainty=UncertaintyEstimate(
            kind=UncertaintyKind.MODEL,
            unit=KW,
            method="quantile",
            relative_std=0.15,
        ),
    )
    assert forecast.has_uncertainty
    assert forecast.is_quantified_uncertainty
    assert forecast.uncertainty.is_method_selected


# ======================================================================
# Serialization
# ======================================================================


ROUND_TRIP_CASES = [
    "battery",
    "solar",
    "topology",
    "pcc_node",
    "element",
    "state",
    "action",
    "forecast",
    "constraint",
    "objective",
    "observation",
    "system",
]


@pytest.mark.parametrize("case", ROUND_TRIP_CASES)
def test_every_domain_type_round_trips(case: str, request, make_case) -> None:
    """Round-trip is enforced per type, not assumed."""
    obj = make_case(case, request)
    payload = encode(obj)
    restored = decode(payload)
    assert restored == obj, f"{case} changed across a round-trip"
    # Determinism: re-encoding the restored object is byte-identical.
    assert encode(restored) == payload, f"{case} encoding is not deterministic"


def test_encoding_is_deterministic_and_versioned(system: EnergySystem) -> None:
    first = encode(system)
    second = encode(system)
    assert first == second
    payload = json.loads(first)
    assert payload["_schema_version"] == SCHEMA_VERSION
    assert json.dumps(payload, sort_keys=True) == json.dumps(json.loads(second), sort_keys=True)


def test_encoding_produces_compact_deterministic_json(system: EnergySystem) -> None:
    assert ", " not in encode(system)
    assert '": ' not in encode(system)


def test_decoding_rejects_a_foreign_schema_version(system: EnergySystem) -> None:
    payload = json.loads(encode(system))
    payload["_schema_version"] = "1.0.0-old"
    with pytest.raises(SerializationError, match="does not match"):
        decode(json.dumps(payload))


def test_decoding_requires_a_schema_version(system: EnergySystem) -> None:
    payload = json.loads(encode(system))
    del payload["_schema_version"]
    with pytest.raises(SerializationError, match="no _schema_version"):
        decode(json.dumps(payload))


def test_decoding_rejects_invalid_json() -> None:
    with pytest.raises(SerializationError, match="not valid JSON"):
        decode("{not json")


def test_decoding_rejects_a_non_object() -> None:
    with pytest.raises(SerializationError, match="expected a JSON object"):
        decode("[1, 2, 3]")


def test_decoding_rejects_an_undiscoverable_payload() -> None:
    with pytest.raises(SerializationError, match="could not determine domain type"):
        decode(json.dumps({"_schema_version": SCHEMA_VERSION, "mystery": 1}))


def test_decoding_rejects_missing_required_fields() -> None:
    """A required field must fail loudly rather than defaulting silently."""
    # Node requires both node_id and name.
    with pytest.raises(Exception):
        decode(json.dumps({"_schema_version": SCHEMA_VERSION, "node_id": "node-pcc"}))

    # Battery requires an identity and a name. usable_energy and
    # round_trip_efficiency are deliberately NOT in this list: D-050 makes them
    # nullable, because a source may state only a nameplate rating. Absent and
    # explicit-null mean the same thing for those two, and both mean "unknown".
    with pytest.raises(Exception):
        decode(
            json.dumps(
                {
                    "_schema_version": SCHEMA_VERSION,
                    "asset_type": "battery",
                    "name": "b",
                    "node_id": "node-pcc",
                    "authority": "dispatchable",
                    "rated_power": {"value": 1.0, "unit": "kW"},
                    "availability": 1.0,
                }
            )
        )


def test_battery_round_trips_with_unknown_fields() -> None:
    """A battery built from a nameplate-only source must survive a round trip.

    This is the SMART-DS case: ``kWhRated`` present, usable energy, SOC window
    and directional power limits absent (D-050).
    """
    battery = Battery(
        asset_type=AssetType.BATTERY,
        asset_id=AssetId("asset-bess-1"),
        name="bess1",
        node_id=NodeId("node-pcc"),
        authority=AuthorityLevel.DISPATCHABLE,
        rated_power=Quantity(8.0, Unit.KILOWATT),
        availability=1.0,
        metadata={"dispatch": "UNAVAILABLE"},
        usable_energy=None,
        nameplate_energy=Quantity(16.0, Unit.KILOWATT_HOURS),
        min_soc=None,
        max_soc=None,
        max_charge_power=None,
        max_discharge_power=None,
        round_trip_efficiency=0.9025,
    )

    decoded = decode(encode(battery))

    assert isinstance(decoded, Battery)
    assert decoded.usable_energy is None
    assert decoded.nameplate_energy == Quantity(16.0, Unit.KILOWATT_HOURS)
    assert decoded.min_soc is None and decoded.max_soc is None
    assert decoded.max_charge_power is None and decoded.max_discharge_power is None
    assert decoded.round_trip_efficiency == pytest.approx(0.9025)
    assert decoded.capacity_known is False


def test_decoding_rejects_unknown_enum_member(battery: Battery) -> None:
    payload = json.loads(encode(battery))
    payload["asset_type"] = "nuclear"
    with pytest.raises((SerializationError, ValueError)):
        decode(json.dumps(payload))


def test_decoding_rejects_unknown_unit(battery: Battery) -> None:
    payload = json.loads(encode(battery))
    payload["rated_power"]["unit"] = "horsepower"
    with pytest.raises(SerializationError, match="unknown unit"):
        decode(json.dumps(payload))


def test_decoding_rejects_non_numeric_quantity(battery: Battery) -> None:
    payload = json.loads(encode(battery))
    payload["rated_power"]["value"] = "250"
    with pytest.raises(SerializationError, match="must be a number"):
        decode(json.dumps(payload))


def test_encoding_rejects_unserialisable_content(provenance) -> None:
    """A field the encoder cannot represent must fail loudly."""

    @dataclasses.dataclass(frozen=True)
    class WithCallable:
        value: object

    with pytest.raises(SerializationError, match="cannot serialize"):
        to_dict(WithCallable(value=lambda: None))


def test_to_dict_requires_an_object() -> None:
    with pytest.raises(SerializationError, match="expected an object at the top level"):
        to_dict([1, 2, 3])


def test_encoding_emits_every_declared_field(battery: Battery) -> None:
    """No hidden fields: the emitted keys are exactly the dataclass fields."""
    payload = to_dict(battery)
    assert set(payload) == set(battery.__dataclass_fields__)
