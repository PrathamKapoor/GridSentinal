"""Deterministic, versioned serialization.

Requirements from the Phase 2 brief, and how each is met:

deterministic
    :func:`encode` produces byte-identical output for identical input. Enums
    serialise as their ``value``, mappings are emitted in sorted key order, and
    no timestamp or random value is introduced by the encoder itself.

explicit schema
    Every payload carries :data:`SCHEMA_VERSION`. A reader that meets a version
    it does not understand raises rather than guessing.

no hidden fields
    The encoder walks :func:`dataclasses.fields` and emits every declared field.
    There is no private attribute to smuggle state through, and the field list is
    the single source of truth shared by encoding and decoding.

versionable
    :data:`SCHEMA_VERSION` is bumped when the shape changes, and a payload
    records the version it was written with.

round-trip
    :func:`decode` reconstructs the exact object. This is enforced by tests for
    every serialisable type, not assumed.

Format is JSON. Chosen because it is inspectable, dependency-free, and diffable
in git -- which matters for a research artifact whose reproducibility depends on
being able to see exactly what changed. No database is introduced (D-036).
"""

from __future__ import annotations

import json
from dataclasses import fields, is_dataclass
from datetime import date, datetime
from enum import Enum
from typing import Any, Mapping

from .assets import ASSET_TYPES, Asset, Battery, EVCharger, FlexibleLoad, SolarAsset, WindAsset
from .actions import Action
from .constraints import Constraint
from .errors import DomainError, DomainValidationError
from .forecasts import Forecast
from .identifiers import (
    ActionId,
    AssetId,
    ConstraintId,
    ForecastId,
    NetworkElementId,
    NodeId,
    ObjectiveId,
    SystemId,
)
from .observations import ObservationRecord
from .objectives import Objective
from .provenance import ProcessingStep, Provenance, ProvenanceEvent, SourceReference
from .quantities import Quantity
from .quality import DataQuality
from .state import EnergyState
from .system import EnergySystem
from .timebase import TimeBase
from .topology import NetworkElement, NetworkTopology, Node
from .uncertainty import UncertaintyEstimate

__all__ = [
    "SCHEMA_VERSION",
    "encode",
    "decode",
    "to_dict",
    "from_dict",
    "SerializationError",
    "JSON_SEPARATORS",
]

#: Bumped whenever the emitted shape changes. 2.1.0 adds ``Battery.nameplate_energy``
#: and allows ``usable_energy`` / ``min_soc`` / ``max_soc`` to be ``null`` (D-050).
#: A 2.0.0 payload still decodes: every added read is nullable.
SCHEMA_VERSION = "2.1.0-phase3"

#: Compact and fully deterministic: no incidental whitespace, stable key order.
JSON_SEPARATORS = (",", ":")


class SerializationError(DomainError):
    """Raised when a payload cannot be encoded or decoded.

    Inherits from :class:`~energy_intelligence.domain.errors.DomainError` so a
    caller can catch one family of domain failure.
    """


#: Identifier classes, resolved after their module is fully imported.
_IDENTIFIER_TYPES: tuple[type, ...] = (
    SystemId,
    AssetId,
    NodeId,
    NetworkElementId,
    ConstraintId,
    ObjectiveId,
    ActionId,
    ForecastId,
)


# ----------------------------------------------------------------------
# Encoding
# ----------------------------------------------------------------------


def _encode_value(value: Any) -> Any:
    """Recursively convert a domain object into JSON-native types."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Quantity):
        return {"value": value.value, "unit": value.unit.value}
    if isinstance(value, _IDENTIFIER_TYPES):
        return value.value
    if isinstance(value, DataQuality):
        return {
            "flags": sorted(flag.value for flag in value.flags),
            "detail": value.detail,
        }
    if isinstance(value, UncertaintyEstimate):
        # Handled before the dataclass branch so the ``None`` fields are always
        # emitted, keeping the payload shape explicit and stable.
        return {
            "kind": value.kind.value,
            "unit": value.unit.value,
            "method": value.method,
            "lower_bound": value.lower_bound,
            "upper_bound": value.upper_bound,
            "standard_deviation": value.standard_deviation,
            "relative_std": value.relative_std,
            "notes": value.notes,
        }
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _encode_value(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, Mapping):
        return {
            str(key): _encode_value(item)
            for key, item in sorted(value.items(), key=lambda kv: str(kv[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_encode_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [_encode_value(item) for item in sorted(value, key=str)]
    raise SerializationError(
        f"cannot serialize value of type {type(value).__name__}: {value!r}"
    )


def to_dict(obj: Any) -> dict[str, Any]:
    """Convert a domain object into a plain, JSON-native dictionary."""
    encoded = _encode_value(obj)
    if not isinstance(encoded, dict):
        raise SerializationError(f"expected an object at the top level, got {type(encoded).__name__}")
    return encoded


def encode(obj: Any) -> str:
    """Serialize a domain object to a deterministic JSON string.

    The top-level ``_schema_version`` key is added by this function and is not a
    dataclass field, which is why it is namespaced with an underscore.
    """
    payload = to_dict(obj)
    body = {"_schema_version": SCHEMA_VERSION, **payload}
    return json.dumps(body, separators=JSON_SEPARATORS, sort_keys=True, allow_nan=False)


# ----------------------------------------------------------------------
# Decoding
# ----------------------------------------------------------------------


def _require(payload: Mapping[str, Any], key: str, expected: type, context: str) -> Any:
    if key not in payload:
        raise SerializationError(f"{context}: missing required field {key!r}")
    value = payload[key]
    if not isinstance(value, expected):
        raise SerializationError(
            f"{context}: field {key!r} must be {expected.__name__}, got {type(value).__name__}"
        )
    return value


def _quantity(payload: Any, context: str) -> Quantity:
    from .enums import Unit

    if not isinstance(payload, Mapping):
        raise SerializationError(f"{context}: expected a quantity object, got {type(payload).__name__}")
    raw_value = payload.get("value")
    raw_unit = payload.get("unit")
    if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
        raise SerializationError(f"{context}: quantity.value must be a number, got {raw_value!r}")
    if not isinstance(raw_unit, str):
        raise SerializationError(f"{context}: quantity.unit must be a string, got {raw_unit!r}")
    try:
        unit = Unit(raw_unit)
    except ValueError as exc:
        raise SerializationError(f"{context}: unknown unit {raw_unit!r}") from exc
    return Quantity(value=float(raw_value), unit=unit)


def _optional_quantity(payload: Any, context: str) -> Quantity | None:
    return None if payload is None else _quantity(payload, context)


def _provenance(payload: Any, context: str) -> Provenance:
    from .enums import QualityFlag

    if not isinstance(payload, Mapping):
        raise SerializationError(f"{context}: expected a provenance object")

    source_payload = _require(payload, "source", Mapping, context)
    source = SourceReference(
        source_id=_require(source_payload, "source_id", str, f"{context}.source"),
        dataset=_require(source_payload, "dataset", str, f"{context}.source"),
        locator=_require(source_payload, "locator", str, f"{context}.source"),
        version=_require(source_payload, "version", str, f"{context}.source"),
        checksum=source_payload.get("checksum"),
    )

    events = tuple(
        ProvenanceEvent(
            timestamp=_require(item, "timestamp", str, f"{context}.events"),
            event=_require(item, "event", str, f"{context}.events"),
            detail=item.get("detail", ""),
        )
        for item in payload.get("events", ())
    )
    processing = tuple(
        ProcessingStep(
            name=_require(item, "name", str, f"{context}.processing"),
            version=_require(item, "version", str, f"{context}.processing"),
            detail=item.get("detail", ""),
        )
        for item in payload.get("processing", ())
    )
    return Provenance(source=source, events=events, processing=processing)


def _quality(payload: Any, context: str) -> DataQuality:
    from .enums import QualityFlag

    if not isinstance(payload, Mapping):
        raise SerializationError(f"{context}: expected a quality object")
    flags = payload.get("flags")
    if not isinstance(flags, list):
        raise SerializationError(f"{context}: quality.flags must be a list")
    try:
        members = frozenset(QualityFlag(item) for item in flags)
    except ValueError as exc:
        raise SerializationError(f"{context}: unknown quality flag in {flags!r}") from exc
    return DataQuality(flags=members, detail=payload.get("detail", ""))


def _uncertainty(payload: Any, context: str) -> UncertaintyEstimate | None:
    from .enums import UncertaintyKind, Unit

    if payload is None:
        return None
    if not isinstance(payload, Mapping):
        raise SerializationError(f"{context}: expected an uncertainty object")
    try:
        kind = UncertaintyKind(payload.get("kind"))
    except ValueError as exc:
        raise SerializationError(f"{context}: unknown uncertainty kind") from exc
    try:
        unit = Unit(payload.get("unit"))
    except ValueError as exc:
        raise SerializationError(f"{context}: unknown uncertainty unit") from exc
    return UncertaintyEstimate(
        kind=kind,
        unit=unit,
        method=payload.get("method", "TBD"),
        lower_bound=payload.get("lower_bound"),
        upper_bound=payload.get("upper_bound"),
        standard_deviation=payload.get("standard_deviation"),
        relative_std=payload.get("relative_std"),
        notes=payload.get("notes", ""),
    )


def _assets(payload: Any, context: str) -> tuple[Asset, ...]:
    from .enums import AssetType, AuthorityLevel

    result: list[Asset] = []
    for item in payload:
        asset_type = AssetType(item["asset_type"])
        cls = ASSET_TYPES[asset_type]
        common = {
            "asset_id": AssetId(item["asset_id"]),
            "name": item["name"],
            "node_id": NodeId(item["node_id"]),
            "authority": AuthorityLevel(item["authority"]),
            "rated_power": _quantity(item["rated_power"], f"{context}.rated_power"),
            "availability": item["availability"],
            "metadata": item.get("metadata"),
        }
        if cls is Asset:
            result.append(Asset(asset_type=asset_type, **common))
        elif cls is SolarAsset:
            result.append(
                SolarAsset(
                    asset_type=asset_type,
                    capacity_kw=_quantity(item["capacity_kw"], f"{context}.capacity_kw"),
                    tilt_degrees=_optional_quantity(item.get("tilt_degrees"), f"{context}.tilt"),
                    azimuth_degrees=_optional_quantity(
                        item.get("azimuth_degrees"), f"{context}.azimuth"
                    ),
                    has_inverter=item.get("has_inverter", False),
                    **common,
                )
            )
        elif cls is WindAsset:
            result.append(
                WindAsset(
                    asset_type=asset_type,
                    cut_in_speed=_optional_quantity(item.get("cut_in_speed"), f"{context}.cut_in"),
                    rated_speed=_optional_quantity(item.get("rated_speed"), f"{context}.rated"),
                    cut_out_speed=_optional_quantity(item.get("cut_out_speed"), f"{context}.cut_out"),
                    **common,
                )
            )
        elif cls is Battery:
            result.append(
                Battery(
                    asset_type=asset_type,
                    usable_energy=_optional_quantity(
                        item.get("usable_energy"), f"{context}.usable_energy"
                    ),
                    min_soc=item.get("min_soc"),
                    max_soc=item.get("max_soc"),
                    max_charge_power=_optional_quantity(
                        item.get("max_charge_power"), f"{context}.max_charge"
                    ),
                    max_discharge_power=_optional_quantity(
                        item.get("max_discharge_power"), f"{context}.max_discharge"
                    ),
                    round_trip_efficiency=item.get("round_trip_efficiency"),
                    nameplate_energy=_optional_quantity(
                        item.get("nameplate_energy"), f"{context}.nameplate_energy"
                    ),
                    **common,
                )
            )
        elif cls is EVCharger:
            result.append(
                EVCharger(
                    asset_type=asset_type,
                    min_return_soc=item["min_return_soc"],
                    num_connectors=item["num_connectors"],
                    **common,
                )
            )
        else:  # FlexibleLoad
            result.append(
                FlexibleLoad(
                    asset_type=asset_type,
                    min_power=_optional_quantity(item.get("min_power"), f"{context}.min_power"),
                    max_power=_optional_quantity(item.get("max_power"), f"{context}.max_power"),
                    shiftable_energy=item["shiftable_energy"],
                    priority=item["priority"],
                    is_hvac=item.get("is_hvac", False),
                    **common,
                )
            )
    return tuple(result)


def _constraints(payload: Any, context: str) -> tuple[Constraint, ...]:
    from .enums import ConstraintCategory, Unit

    return tuple(
        Constraint(
            constraint_id=ConstraintId(item["constraint_id"]),
            category=ConstraintCategory(item["category"]),
            description=item["description"],
            provenance=_provenance(item["provenance"], f"{context}.provenance"),
            applies_to_asset_ids=tuple(AssetId(item) for item in item.get("applies_to_asset_ids", ())),
            applies_to_node_ids=tuple(NodeId(item) for item in item.get("applies_to_node_ids", ())),
            lower_bound=item.get("lower_bound"),
            upper_bound=item.get("upper_bound"),
            unit=Unit(item["unit"]) if item.get("unit") is not None else None,
            requires_power_flow=item.get("requires_power_flow", False),
        )
        for item in payload
    )


def _objectives(payload: Any, context: str) -> tuple[Objective, ...]:
    from .enums import ObjectiveCategory

    return tuple(
        Objective(
            objective_id=ObjectiveId(item["objective_id"]),
            category=ObjectiveCategory(item["category"]),
            measurement=item["measurement"],
            why_it_matters=item["why_it_matters"],
            provenance=_provenance(item["provenance"], f"{context}.provenance"),
            target_value=item.get("target_value"),
        )
        for item in payload
    )


def from_dict(payload: Mapping[str, Any]) -> Any:
    """Reconstruct a domain object from its dictionary form.

    Dispatch is by discriminating field. Order matters: an asset payload also
    carries ``node_id`` (the node it attaches to), so the asset check must
    precede the bare-node check or every asset would decode as a ``Node``.
    """
    if not isinstance(payload, Mapping):
        raise SerializationError(f"expected an object, got {type(payload).__name__}")

    if "system_id" in payload:
        return _energy_system(payload)
    if "forecast_id" in payload:
        return _forecast(payload)
    if "action_id" in payload:
        return _action(payload)
    if "constraint_id" in payload:
        return _constraints([payload], "constraint")[0]
    if "objective_id" in payload:
        return _objectives([payload], "objective")[0]
    if "node_power" in payload:
        return _energy_state(payload)
    if "variable" in payload and "quality" in payload:
        return _observation(payload)
    if "element_id" in payload:
        return _network_element(payload)
    # Every asset payload carries asset_type; a bare Node never does.
    if "asset_type" in payload:
        return _assets([payload], "asset")[-1]
    if "nodes" in payload:
        return _topology(payload)
    if "node_id" in payload:
        return _node(payload)
    raise SerializationError(
        "could not determine domain type from payload keys: "
        f"{sorted(str(key) for key in payload)}"
    )


def decode(text: str) -> Any:
    """Deserialize a JSON string produced by :func:`encode`.

    Raises:
        SerializationError: On invalid JSON, an unrecognised schema version, or
            a malformed payload.
    """
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SerializationError(f"payload is not valid JSON: {exc}") from exc

    if not isinstance(payload, Mapping):
        raise SerializationError(f"expected a JSON object, got {type(payload).__name__}")

    version = payload.get("_schema_version")
    if version is None:
        raise SerializationError(
            "payload has no _schema_version; it was not produced by encode()"
        )
    if version != SCHEMA_VERSION:
        raise SerializationError(
            f"payload schema version {version!r} does not match {SCHEMA_VERSION!r}"
        )

    body = {key: value for key, value in payload.items() if key != "_schema_version"}
    return from_dict(body)


# ----------------------------------------------------------------------
# Type-specific decoders
# ----------------------------------------------------------------------


def _node(payload: Mapping[str, Any]) -> Node:
    from .enums import NetworkElementKind  # noqa: F401 - kept for symmetry

    return Node(
        node_id=NodeId(payload["node_id"]),
        name=payload["name"],
        voltage_kv=_optional_quantity(payload.get("voltage_kv"), "node.voltage_kv"),
        parent_node_id=NodeId(payload["parent_node_id"])
        if payload.get("parent_node_id") is not None
        else None,
        is_grid_connection=payload.get("is_grid_connection", False),
        metadata=payload.get("metadata"),
    )


def _network_element(payload: Mapping[str, Any]) -> NetworkElement:
    from .enums import NetworkElementKind

    return NetworkElement(
        element_id=NetworkElementId(payload["element_id"]),
        kind=NetworkElementKind(payload["kind"]),
        from_node_id=NodeId(payload["from_node_id"]),
        to_node_id=NodeId(payload["to_node_id"]),
        rating=_optional_quantity(payload.get("rating"), "element.rating"),
        is_in_service=payload.get("is_in_service", True),
        parameters=payload.get("parameters"),
        metadata=payload.get("metadata"),
    )


def _topology(payload: Mapping[str, Any]) -> NetworkTopology:
    return NetworkTopology(
        nodes=tuple(_node(item) for item in payload["nodes"]),
        elements=tuple(_network_element(item) for item in payload.get("elements", ())),
        description=payload.get("description", ""),
        metadata=payload.get("metadata"),
    )


def _energy_state(payload: Mapping[str, Any]) -> EnergyState:
    from .enums import StateOrigin

    return EnergyState(
        timestamp=payload["timestamp"],
        origin=StateOrigin(payload["origin"]),
        node_power=tuple(_node_power(item) for item in payload["node_power"]),
        demand=_demand(payload["demand"]),
        grid=_grid(payload["grid"]),
        provenance=_provenance(payload["provenance"], "state.provenance"),
        storage=tuple(_storage(item) for item in payload.get("storage", ())),
        renewables=tuple(_renewable(item) for item in payload.get("renewables", ())),
        ev=_ev(payload["ev"]) if payload.get("ev") is not None else None,
        timebase_id=payload.get("timebase_id", ""),
    )


def _node_power(payload: Mapping[str, Any]) -> Any:
    from .state import NodePower

    return NodePower(
        node_id=NodeId(payload["node_id"]),
        generation_kw=_quantity(payload["generation_kw"], "node_power.generation_kw"),
        load_kw=_quantity(payload["load_kw"], "node_power.load_kw"),
        storage_kw=_quantity(payload["storage_kw"], "node_power.storage_kw"),
        grid_kw=_quantity(payload["grid_kw"], "node_power.grid_kw"),
        unaccounted_kw=_optional_quantity(payload.get("unaccounted_kw"), "node_power.unaccounted"),
    )


def _demand(payload: Mapping[str, Any]) -> Any:
    from .state import DemandState

    return DemandState(
        total_demand_kw=_quantity(payload["total_demand_kw"], "demand.total_demand_kw"),
        quality=_quality(payload["quality"], "demand.quality"),
        by_category=payload.get("by_category"),
    )


def _grid(payload: Mapping[str, Any]) -> Any:
    from .state import GridState

    voltages = payload.get("voltage_by_node")
    return GridState(
        import_kw=_quantity(payload["import_kw"], "grid.import_kw"),
        export_kw=_quantity(payload["export_kw"], "grid.export_kw"),
        quality=_quality(payload["quality"], "grid.quality"),
        frequency_hz=_optional_quantity(payload.get("frequency_hz"), "grid.frequency_hz"),
        voltage_by_node=(
            {key: _quantity(value, f"grid.voltage_by_node[{key}]") for key, value in voltages.items()}
            if voltages
            else None
        ),
    )


def _storage(payload: Mapping[str, Any]) -> Any:
    from .state import StorageState

    return StorageState(
        asset_id=AssetId(payload["asset_id"]),
        state_of_charge=payload["state_of_charge"],
        charge_power_kw=_quantity(payload["charge_power_kw"], "storage.charge_power_kw"),
        discharge_power_kw=_quantity(
            payload["discharge_power_kw"], "storage.discharge_power_kw"
        ),
        available_energy=_quantity(payload["available_energy"], "storage.available_energy"),
        quality=_quality(payload["quality"], "storage.quality"),
    )


def _renewable(payload: Mapping[str, Any]) -> Any:
    from .state import RenewableState

    return RenewableState(
        asset_id=AssetId(payload["asset_id"]),
        available_power=_quantity(payload["available_power"], "renewable.available_power"),
        actual_power=_quantity(payload["actual_power"], "renewable.actual_power"),
        quality=_quality(payload["quality"], "renewable.quality"),
    )


def _ev(payload: Mapping[str, Any]) -> Any:
    from .state import EVState

    return EVState(
        connected_vehicles=payload["connected_vehicles"],
        available_connectors=payload["available_connectors"],
        charging_power_kw=_quantity(payload["charging_power_kw"], "ev.charging_power_kw"),
        deferrable_power_kw=_quantity(payload["deferrable_power_kw"], "ev.deferrable_power_kw"),
        quality=_quality(payload["quality"], "ev.quality"),
    )


def _energy_system(payload: Mapping[str, Any]) -> EnergySystem:
    from .timebase import TimeBase as _TimeBase

    return EnergySystem(
        system_id=SystemId(payload["system_id"]),
        name=payload["name"],
        description=payload["description"],
        time_base=_TimeBase(
            timestep_minutes=payload["time_base"]["timestep_minutes"],
            horizon_steps=payload["time_base"]["horizon_steps"],
            origin=datetime.fromisoformat(payload["time_base"]["origin"]),
        ),
        topology=_topology(payload["topology"]),
        assets=_assets(payload["assets"], "system.assets"),
        provenance=_provenance(payload["provenance"], "system.provenance"),
        constraints=_constraints(payload.get("constraints", ()), "system.constraints"),
        objectives=_objectives(payload.get("objectives", ()), "system.objectives"),
        metadata=payload.get("metadata"),
    )


def _action(payload: Mapping[str, Any]) -> Action:
    from .enums import ActionType, AuthorityLevel, VariableRole

    return Action(
        action_id=ActionId(payload["action_id"]),
        action_type=ActionType(payload["action_type"]),
        target_asset_id=AssetId(payload["target_asset_id"]),
        setpoint=_quantity(payload["setpoint"], "action.setpoint"),
        role=VariableRole(payload["role"]),
        issued_under=AuthorityLevel(payload["issued_under"]),
        start_step=payload["start_step"],
        duration_steps=payload["duration_steps"],
        provenance=_provenance(payload["provenance"], "action.provenance"),
        rationale=payload.get("rationale", ""),
    )


def _forecast(payload: Mapping[str, Any]) -> Forecast:
    from .enums import VariableRole

    return Forecast(
        forecast_id=ForecastId(payload["forecast_id"]),
        variable=payload["variable"],
        role=VariableRole(payload["role"]),
        target_time=payload["target_time"],
        issued_at=payload["issued_at"],
        horizon_steps=payload["horizon_steps"],
        value=_quantity(payload["value"], "forecast.value"),
        provenance=_provenance(payload["provenance"], "forecast.provenance"),
        uncertainty=_uncertainty(payload.get("uncertainty"), "forecast.uncertainty"),
    )


def _observation(payload: Mapping[str, Any]) -> ObservationRecord:
    from .enums import VariableRole

    return ObservationRecord(
        variable=payload["variable"],
        role=VariableRole(payload["role"]),
        value=_quantity(payload["value"], "observation.value"),
        quality=_quality(payload["quality"], "observation.quality"),
        provenance=_provenance(payload["provenance"], "observation.provenance"),
        timestamp=payload["timestamp"],
        asset_id=AssetId(payload["asset_id"]) if payload.get("asset_id") is not None else None,
        node_id=NodeId(payload["node_id"]) if payload.get("node_id") is not None else None,
    )
