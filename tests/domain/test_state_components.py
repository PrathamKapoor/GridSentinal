"""Focused tests for the remaining EnergyState component validation branches.

State is the most heavily validated object in the domain model, so its error
paths are tested explicitly rather than left to incidental coverage.
"""

from __future__ import annotations

import pytest

from energy_intelligence.domain import (
    AssetId,
    DataQuality,
    DemandState,
    DomainValidationError,
    EnergyState,
    EVState,
    GridState,
    NodeId,
    NodePower,
    Quantity,
    QualityFlag,
    StateOrigin,
    StorageState,
    Unit,
)

KW = Unit.KILOWATT
KWH = Unit.KILOWATT_HOURS
CLEAN = DataQuality(flags=frozenset({QualityFlag.OK}))


def kw(value: float) -> Quantity:
    return Quantity(value, KW)


def kwh(value: float) -> Quantity:
    return Quantity(value, KWH)


# ----------------------------------------------------------------------
# DemandState
# ----------------------------------------------------------------------


def test_demand_state_requires_a_quantity() -> None:
    with pytest.raises(DomainValidationError, match="total_demand_kw must be a Quantity"):
        DemandState(total_demand_kw=500.0, quality=CLEAN)  # type: ignore[arg-type]


def test_demand_state_rejects_negative_total() -> None:
    with pytest.raises(DomainValidationError, match="total_demand_kw must be non-negative"):
        DemandState(total_demand_kw=kw(-1.0), quality=CLEAN)


def test_demand_state_requires_a_quality_object() -> None:
    with pytest.raises(DomainValidationError, match="quality must be a DataQuality"):
        DemandState(total_demand_kw=kw(1.0), quality="ok")  # type: ignore[arg-type]


def test_demand_state_rejects_non_mapping_categories() -> None:
    with pytest.raises(DomainValidationError, match="by_category must be a dict"):
        DemandState(total_demand_kw=kw(1.0), quality=CLEAN, by_category=[("a", 1.0)])  # type: ignore[arg-type]


def test_demand_state_rejects_non_numeric_category() -> None:
    with pytest.raises(DomainValidationError, match="must be a real number"):
        DemandState(total_demand_kw=kw(1.0), quality=CLEAN, by_category={"residential": "high"})


def test_demand_state_rejects_negative_category() -> None:
    with pytest.raises(DomainValidationError, match="must be non-negative"):
        DemandState(total_demand_kw=kw(1.0), quality=CLEAN, by_category={"residential": -1.0})


def test_demand_state_categories_are_optional() -> None:
    assert DemandState(total_demand_kw=kw(1.0), quality=CLEAN).by_category is None


# ----------------------------------------------------------------------
# EVState
# ----------------------------------------------------------------------


def test_ev_state_rejects_non_integer_counts() -> None:
    with pytest.raises(DomainValidationError, match="connected_vehicles must be an int"):
        EVState(
            connected_vehicles=1.5,  # type: ignore[arg-type]
            available_connectors=1,
            charging_power_kw=kw(0.0),
            deferrable_power_kw=kw(0.0),
            quality=CLEAN,
        )


def test_ev_state_rejects_negative_counts() -> None:
    with pytest.raises(DomainValidationError, match="available_connectors must be non-negative"):
        EVState(
            connected_vehicles=1,
            available_connectors=-1,
            charging_power_kw=kw(0.0),
            deferrable_power_kw=kw(0.0),
            quality=CLEAN,
        )


def test_ev_state_rejects_non_quantity_powers() -> None:
    with pytest.raises(DomainValidationError, match="charging_power_kw must be a Quantity"):
        EVState(
            connected_vehicles=1,
            available_connectors=1,
            charging_power_kw=7.0,  # type: ignore[arg-type]
            deferrable_power_kw=kw(0.0),
            quality=CLEAN,
        )


def test_ev_state_rejects_negative_power() -> None:
    with pytest.raises(DomainValidationError, match="deferrable_power_kw must be non-negative"):
        EVState(
            connected_vehicles=1,
            available_connectors=1,
            charging_power_kw=kw(0.0),
            deferrable_power_kw=kw(-1.0),
            quality=CLEAN,
        )


# ----------------------------------------------------------------------
# GridState
# ----------------------------------------------------------------------


def test_grid_state_rejects_negative_flows() -> None:
    with pytest.raises(DomainValidationError, match="import_kw must be non-negative"):
        GridState(import_kw=kw(-1.0), export_kw=kw(0.0), quality=CLEAN)


def test_grid_state_requires_quality() -> None:
    with pytest.raises(DomainValidationError, match="quality must be a DataQuality"):
        GridState(import_kw=kw(0.0), export_kw=kw(0.0), quality=None)  # type: ignore[arg-type]


def test_grid_state_rejects_non_positive_frequency() -> None:
    with pytest.raises(DomainValidationError, match="frequency_hz must be positive"):
        GridState(
            import_kw=kw(0.0),
            export_kw=kw(0.0),
            quality=CLEAN,
            frequency_hz=Quantity(0.0, Unit.HERTZ),
        )


def test_grid_state_rejects_non_quantity_frequency() -> None:
    with pytest.raises(DomainValidationError, match="frequency_hz must be a Quantity or None"):
        GridState(import_kw=kw(0.0), export_kw=kw(0.0), quality=CLEAN, frequency_hz=50.0)  # type: ignore[arg-type]


def test_grid_state_rejects_non_mapping_voltages() -> None:
    with pytest.raises(DomainValidationError, match="voltage_by_node must be a dict"):
        GridState(
            import_kw=kw(0.0),
            export_kw=kw(0.0),
            quality=CLEAN,
            voltage_by_node=[11.0],  # type: ignore[arg-type]
        )


def test_grid_state_rejects_non_quantity_voltage() -> None:
    with pytest.raises(DomainValidationError, match="must be a Quantity"):
        GridState(
            import_kw=kw(0.0),
            export_kw=kw(0.0),
            quality=CLEAN,
            voltage_by_node={"node-pcc": 11.0},  # type: ignore[dict-item]
        )


def test_grid_state_accepts_measured_voltage() -> None:
    """Voltage is an observation with its own unit, never a derived quantity."""
    state = GridState(
        import_kw=kw(0.0),
        export_kw=kw(0.0),
        quality=CLEAN,
        frequency_hz=Quantity(50.01, Unit.HERTZ),
        voltage_by_node={"node-pcc": Quantity(11.02, Unit.KILOVOLT)},
    )
    assert state.frequency_hz.value == 50.01
    assert state.voltage_by_node["node-pcc"].unit is Unit.KILOVOLT


# ----------------------------------------------------------------------
# StorageState / NodePower type guards
# ----------------------------------------------------------------------


def test_storage_state_requires_an_asset_id() -> None:
    with pytest.raises(DomainValidationError, match="asset_id must be an AssetId"):
        StorageState(
            asset_id="asset-x",  # type: ignore[arg-type]
            state_of_charge=0.5,
            charge_power_kw=kw(0.0),
            discharge_power_kw=kw(0.0),
            available_energy=kwh(1.0),
            quality=CLEAN,
        )


def test_storage_state_rejects_non_numeric_soc() -> None:
    with pytest.raises(DomainValidationError, match="state_of_charge must be a real number"):
        StorageState(
            asset_id=AssetId("asset-b-1"),
            state_of_charge="60%",  # type: ignore[arg-type]
            charge_power_kw=kw(0.0),
            discharge_power_kw=kw(0.0),
            available_energy=kwh(1.0),
            quality=CLEAN,
        )


def test_storage_state_rejects_non_quantity_energy() -> None:
    with pytest.raises(DomainValidationError, match="available_energy must be a Quantity"):
        StorageState(
            asset_id=AssetId("asset-b-1"),
            state_of_charge=0.5,
            charge_power_kw=kw(0.0),
            discharge_power_kw=kw(0.0),
            available_energy=300.0,  # type: ignore[arg-type]
            quality=CLEAN,
        )


def test_node_power_requires_quantities() -> None:
    with pytest.raises(DomainValidationError, match="generation_kw must be a Quantity"):
        NodePower(
            node_id=NodeId("node-a"),
            generation_kw=1.0,  # type: ignore[arg-type]
            load_kw=kw(0.0),
            storage_kw=kw(0.0),
            grid_kw=kw(0.0),
        )


def test_node_power_requires_an_id() -> None:
    with pytest.raises(DomainValidationError, match="node_id must be a NodeId"):
        NodePower(
            node_id="node-a",  # type: ignore[arg-type]
            generation_kw=kw(0.0),
            load_kw=kw(0.0),
            storage_kw=kw(0.0),
            grid_kw=kw(0.0),
        )


def test_node_power_rejects_non_quantity_unaccounted() -> None:
    with pytest.raises(DomainValidationError, match="unaccounted_kw must be a Quantity or None"):
        NodePower(
            node_id=NodeId("node-a"),
            generation_kw=kw(0.0),
            load_kw=kw(0.0),
            storage_kw=kw(0.0),
            grid_kw=kw(0.0),
            unaccounted_kw=5.0,  # type: ignore[arg-type]
        )


# ----------------------------------------------------------------------
# EnergyState structural guards
# ----------------------------------------------------------------------


def _minimal_state(**overrides) -> EnergyState:
    node = NodePower(
        node_id=NodeId("node-a"),
        generation_kw=kw(0.0),
        load_kw=kw(0.0),
        storage_kw=kw(0.0),
        grid_kw=kw(0.0),
    )
    fields = {
        "timestamp": "2026-01-01T00:00:00+00:00",
        "origin": StateOrigin.OBSERVED,
        "node_power": (node,),
        "demand": DemandState(total_demand_kw=kw(0.0), quality=CLEAN),
        "grid": GridState(import_kw=kw(0.0), export_kw=kw(0.0), quality=CLEAN),
        "provenance": None,
    }
    fields.update(overrides)
    return EnergyState(**fields)


def test_state_requires_provenance() -> None:
    with pytest.raises(DomainValidationError, match="provenance must be a Provenance"):
        _minimal_state()


def test_state_requires_a_valid_origin(provenance) -> None:
    with pytest.raises(DomainValidationError, match="origin must be a StateOrigin"):
        _minimal_state(provenance=provenance, origin="observed")  # type: ignore[arg-type]


def test_state_requires_node_power(provenance) -> None:
    with pytest.raises(DomainValidationError, match="node_power must be a non-empty tuple"):
        _minimal_state(provenance=provenance, node_power=())


def test_state_rejects_non_tuple_node_power(provenance) -> None:
    """A list of otherwise-valid nodes is rejected; node_power must be a tuple."""
    with pytest.raises(DomainValidationError, match="node_power must be a non-empty tuple"):
        _minimal_state(
            provenance=provenance,
            node_power=[
                NodePower(
                    node_id=NodeId("node-a"),
                    generation_kw=kw(0.0),
                    load_kw=kw(0.0),
                    storage_kw=kw(0.0),
                    grid_kw=kw(0.0),
                )
            ],  # type: ignore[arg-type]
        )


def test_state_rejects_wrong_node_power_type(provenance) -> None:
    with pytest.raises(DomainValidationError, match="every entry in node_power must be"):
        _minimal_state(provenance=provenance, node_power=("node-a",))  # type: ignore[arg-type]


def test_state_rejects_wrong_component_types(provenance) -> None:
    with pytest.raises(DomainValidationError, match="demand must be a DemandState"):
        _minimal_state(provenance=provenance, demand=kw(0.0))  # type: ignore[arg-type]
    with pytest.raises(DomainValidationError, match="grid must be a GridState"):
        _minimal_state(provenance=provenance, grid=kw(0.0))  # type: ignore[arg-type]


def test_state_rejects_non_tuple_storage(provenance) -> None:
    with pytest.raises(DomainValidationError, match="storage must be a tuple"):
        _minimal_state(provenance=provenance, storage=[])  # type: ignore[arg-type]


def test_state_rejects_duplicate_assets(provenance) -> None:
    storage = StorageState(
        asset_id=AssetId("asset-b-1"),
        state_of_charge=0.5,
        charge_power_kw=kw(0.0),
        discharge_power_kw=kw(0.0),
        available_energy=kwh(1.0),
        quality=CLEAN,
    )
    with pytest.raises(DomainValidationError, match="duplicate asset in storage"):
        _minimal_state(provenance=provenance, storage=(storage, storage))


def test_state_rejects_wrong_ev_type(provenance) -> None:
    with pytest.raises(DomainValidationError, match="ev must be an EVState or None"):
        _minimal_state(provenance=provenance, ev=kw(0.0))  # type: ignore[arg-type]


@pytest.mark.parametrize("origin", list(StateOrigin))
def test_every_state_origin_is_accepted(provenance, origin: StateOrigin) -> None:
    """Observed, simulated and predicted must all be representable and distinct."""
    state = _minimal_state(provenance=provenance, origin=origin)
    assert state.origin is origin


def test_state_origins_are_distinguishable() -> None:
    values = {origin.value for origin in StateOrigin}
    assert values == {"observed", "simulated", "predicted", "assimilated"}


def test_state_timebase_id_is_optional(provenance) -> None:
    assert _minimal_state(provenance=provenance).timebase_id == ""


def test_state_optional_components_default_empty(provenance) -> None:
    state = _minimal_state(provenance=provenance)
    assert state.storage == ()
    assert state.renewables == ()
    assert state.ev is None
