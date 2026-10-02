"""Energy domain model: the formal contract every later phase depends on.

Defines **what energy system is being modelled**, not how software is organised.
Phases 1-20 all consume this contract; nothing in it presumes a model, an
optimizer, a simulator or an agent exists.

What is defined
    * :class:`~energy_intelligence.domain.enums.VariableRole` - the boundary
      between what is observed and what is commanded, enforced at construction.
    * :class:`~energy_intelligence.domain.assets.Asset` and subtypes - typed
      assets with explicit identity and a declared control authority.
    * :class:`~energy_intelligence.domain.topology.NetworkTopology` - a
      configurable node graph; physics deferred to Phase 11.
    * :class:`~energy_intelligence.domain.state.EnergyState` - system state with
      enforced per-node power balance and an explicit origin.
    * :class:`~energy_intelligence.domain.actions.Action` - the control space,
      gated by authority.
    * :class:`~energy_intelligence.domain.constraints.Constraint` and
      :class:`~energy_intelligence.domain.objectives.Objective` - declared, not
      enforced and not weighted.
    * :class:`~energy_intelligence.domain.uncertainty.UncertaintyEstimate` - the
      container Phase 8 will fill in.
    * :mod:`~energy_intelligence.domain.provenance` - mandatory traceability on
      every datum.

What is deliberately absent
    Any optimizer, forecast model, agent, digital twin or UI. Those belong to
    Phases 4-20.

Example:
    >>> from energy_intelligence.domain import Battery, Quantity, Unit
    >>> Battery(
    ...     asset_type=AssetType.BATTERY,
    ...     asset_id=AssetId("asset-battery-1"),
    ...     name="Feeder A battery",
    ...     node_id=NodeId("node-feeder-a"),
    ...     authority=AuthorityLevel.DISPATCHABLE,
    ...     rated_power=Quantity(250.0, Unit.KILOWATT),
    ...     availability=1.0,
    ...     metadata=None,
    ...     usable_energy=Quantity(500.0, Unit.KILOWATT_HOURS),
    ...     min_soc=0.1,
    ...     max_soc=0.95,
    ...     max_charge_power=Quantity(250.0, Unit.KILOWATT),
    ...     max_discharge_power=Quantity(250.0, Unit.KILOWATT),
    ...     round_trip_efficiency=0.92,
    ... ).usable_soc_window
    (0.1, 0.95)
"""

from __future__ import annotations

from .actions import (
    ACTION_REQUIREMENTS,
    Action,
    required_authority_for,
    sign_constraint_for,
    sufficient_authorities_for,
    unit_for_action,
)
from .assets import (
    ASSET_TYPES,
    Asset,
    Battery,
    EVCharger,
    FlexibleLoad,
    SolarAsset,
    WindAsset,
    asset_class_for,
)
from .constraints import CONSTRAINT_UNITS, Constraint, unit_for_constraint
from .enums import (
    CONTROLLABLE_AUTHORITY_LEVELS,
    ActionType,
    AssetType,
    AuthorityLevel,
    ConstraintCategory,
    NetworkElementKind,
    ObjectiveCategory,
    ObjectiveDirection,
    QualityFlag,
    StateOrigin,
    UncertaintyKind,
    Unit,
    VariableRole,
)
from .errors import DomainError, DomainValidationError
from .forecasts import Forecast
from .identifiers import (
    ActionId,
    AssetId,
    ConstraintId,
    ForecastId,
    Identifier,
    NetworkElementId,
    NodeId,
    ObjectiveId,
    SystemId,
)
from .observations import ALLOWED_OBSERVATION_ROLES, ObservationRecord
from .objectives import OBJECTIVE_METADATA, Objective, objective_metadata
from .provenance import ProcessingStep, Provenance, ProvenanceEvent, SourceReference
from .quantities import Quantity
from .quality import CLEAN_QUALITY, DataQuality
from .serialization import SCHEMA_VERSION, SerializationError, decode, encode, to_dict
from .state import (
    POWER_BALANCE_TOLERANCE,
    DemandState,
    EnergyState,
    EVState,
    GridState,
    NodePower,
    RenewableState,
    StorageState,
)
from .system import (
    EnergySystem,
    validate_actions_against_system,
    validate_state_against_system,
)
from .timebase import DEFAULT_HORIZON_STEPS, DEFAULT_TIMESTEP_MINUTES, TimeBase
from .topology import NetworkElement, NetworkTopology, Node
from .uncertainty import TBD_METHOD, UncertaintyEstimate

__all__ = [
    # errors
    "DomainError",
    "DomainValidationError",
    # vocabulary
    "ActionType",
    "AssetType",
    "AuthorityLevel",
    "ConstraintCategory",
    "NetworkElementKind",
    "ObjectiveCategory",
    "ObjectiveDirection",
    "QualityFlag",
    "StateOrigin",
    "UncertaintyKind",
    "Unit",
    "VariableRole",
    "CONTROLLABLE_AUTHORITY_LEVELS",
    # identifiers
    "ActionId",
    "AssetId",
    "ConstraintId",
    "ForecastId",
    "Identifier",
    "NetworkElementId",
    "NodeId",
    "ObjectiveId",
    "SystemId",
    # values
    "Quantity",
    "DataQuality",
    "CLEAN_QUALITY",
    "Provenance",
    "ProvenanceEvent",
    "ProcessingStep",
    "SourceReference",
    "TimeBase",
    "DEFAULT_TIMESTEP_MINUTES",
    "DEFAULT_HORIZON_STEPS",
    "UncertaintyEstimate",
    "TBD_METHOD",
    # topology
    "Node",
    "NetworkElement",
    "NetworkTopology",
    # assets
    "Asset",
    "Battery",
    "EVCharger",
    "FlexibleLoad",
    "SolarAsset",
    "WindAsset",
    "ASSET_TYPES",
    "asset_class_for",
    # state
    "DemandState",
    "EnergyState",
    "EVState",
    "GridState",
    "NodePower",
    "RenewableState",
    "StorageState",
    "POWER_BALANCE_TOLERANCE",
    # control
    "Action",
    "ACTION_REQUIREMENTS",
    "required_authority_for",
    "sufficient_authorities_for",
    "unit_for_action",
    "sign_constraint_for",
    "Constraint",
    "CONSTRAINT_UNITS",
    "unit_for_constraint",
    "Objective",
    "OBJECTIVE_METADATA",
    "objective_metadata",
    # observations and forecasts
    "ObservationRecord",
    "ALLOWED_OBSERVATION_ROLES",
    "Forecast",
    # root
    "EnergySystem",
    "validate_state_against_system",
    "validate_actions_against_system",
    # serialization
    "SCHEMA_VERSION",
    "SerializationError",
    "encode",
    "decode",
    "to_dict",
]

DOMAIN_SCHEMA_VERSION = SCHEMA_VERSION
