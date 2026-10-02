"""EnergySystem: the root aggregate, and cross-entity validation.

:class:`EnergySystem` is the single object every later phase starts from. It owns
the topology, the assets, the declared constraints and objectives, and the
temporal grid, and it is where **referential integrity** is enforced.

Most validation is local and already happens at construction: a
:class:`~energy_intelligence.domain.topology.NetworkTopology` cannot reference an
unknown node, a :class:`~energy_intelligence.domain.quantities.Quantity` cannot
hold NaN. What cannot be checked locally is anything involving two entities, and
that is exactly what :meth:`EnergySystem.validate` covers:

* every asset's node exists in the topology;
* no duplicate asset identifiers;
* every constraint's asset and node references resolve;
* every objective is unique;
* a state reported for this system covers exactly the system's nodes;
* a state reported for this system does not report assets the system does not own.

That last pair is the one that matters most: without it a system could report a
battery's state while owning no battery, and every downstream number would be
wrong in a way nothing would flag.
"""

from __future__ import annotations

from dataclasses import dataclass

from .actions import Action
from .assets import Asset
from .constraints import Constraint
from .errors import DomainValidationError
from .forecasts import Forecast
from .identifiers import SystemId
from .observations import ObservationRecord
from .objectives import Objective
from .provenance import Provenance
from .state import EnergyState
from .timebase import TimeBase
from .topology import NetworkTopology

__all__ = ["EnergySystem", "validate_state_against_system", "validate_actions_against_system"]


@dataclass(frozen=True, slots=True)
class EnergySystem:
    """A renewable-integrated distribution-level environment with DERs.

    Nothing about the *size* of this system is hard-coded. Node count, feeder
    count, asset count and every rating come from configuration, so the same type
    represents a single lumped node and a large multi-feeder system.

    Attributes:
        system_id: Unique identifier for this system boundary.
        name: Human-readable name.
        description: What this boundary includes and excludes.
        time_base: The temporal grid.
        topology: Nodes and network elements.
        assets: Every asset, each attached to a node.
        constraints: Declared constraints. May be empty.
        objectives: Declared objectives. May be empty.
        provenance: Where this system definition came from. Mandatory.
        metadata: Free-form, non-authoritative extras.
    """

    system_id: SystemId
    name: str
    description: str
    time_base: TimeBase
    topology: NetworkTopology
    assets: tuple[Asset, ...]
    provenance: Provenance
    constraints: tuple[Constraint, ...] = ()
    objectives: tuple[Objective, ...] = ()
    metadata: dict[str, str] | None = None

    def __post_init__(self) -> None:
        errors: list[str] = []

        if not isinstance(self.system_id, SystemId):
            errors.append(f"system_id must be a SystemId, got {type(self.system_id).__name__}")
        for field_name in ("name", "description"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                errors.append(f"{field_name} must be a non-empty string, got {value!r}")
        if not isinstance(self.time_base, TimeBase):
            errors.append(f"time_base must be a TimeBase, got {type(self.time_base).__name__}")
        if not isinstance(self.topology, NetworkTopology):
            errors.append(f"topology must be a NetworkTopology, got {type(self.topology).__name__}")
        if not isinstance(self.provenance, Provenance):
            errors.append(f"provenance must be a Provenance, got {type(self.provenance).__name__}")

        for field_name, expected in (
            ("assets", Asset),
            ("constraints", Constraint),
            ("objectives", Objective),
        ):
            value = getattr(self, field_name)
            if not isinstance(value, tuple):
                errors.append(f"{field_name} must be a tuple, got {type(value).__name__}")
            elif not all(isinstance(item, expected) for item in value):
                errors.append(f"every entry in {field_name} must be a {expected.__name__}")

        if errors:
            raise DomainValidationError(errors, subject="EnergySystem")

        errors = _referential_errors(self)
        if errors:
            raise DomainValidationError(errors, subject="EnergySystem")

    # ------------------------------------------------------------------
    # Lookups
    # ------------------------------------------------------------------

    @property
    def asset_ids(self) -> frozenset:
        return frozenset(asset.asset_id for asset in self.assets)

    def asset(self, asset_id) -> Asset | None:
        for candidate in self.assets:
            if candidate.asset_id == asset_id:
                return candidate
        return None

    def assets_at_node(self, node_id) -> tuple[Asset, ...]:
        return tuple(asset for asset in self.assets if asset.node_id == node_id)

    def controllable_assets(self) -> tuple[Asset, ...]:
        return tuple(asset for asset in self.assets if asset.is_controllable)

    def node_ids(self) -> frozenset:
        return self.topology.node_ids

    @property
    def is_aggregate_topology(self) -> bool:
        """Whether this system is modelled as a single lumped node.

        Surfaced rather than hidden, because a conclusion drawn on an aggregate
        system cannot support per-location claims.
        """
        return self.topology.is_aggregate

    @property
    def controllable_capacity_kw(self) -> float:
        """Total controllable rated power, in kW.

        Only assets whose rated power is expressed in kW contribute; anything in
        another unit is skipped rather than converted, because Phase 2 performs
        no unit conversion.
        """
        return sum(
            asset.rated_power.value
            for asset in self.assets
            if asset.is_controllable and asset.rated_power.unit.value == "kW"
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_state(self, state: EnergyState) -> None:
        """Check a state against this system's topology and asset inventory."""
        validate_state_against_system(self, state)

    def validate_actions(self, actions: tuple[Action, ...]) -> None:
        """Check a set of actions against this system's assets."""
        validate_actions_against_system(self, actions)

    def validate_observations(self, observations: tuple[ObservationRecord, ...]) -> None:
        """Check that every observation references a known asset or node."""
        errors: list[str] = []
        for observation in observations:
            if observation.asset_id is not None and not self._has_asset(observation.asset_id):
                errors.append(
                    f"observation of {observation.variable!r} references unknown "
                    f"asset {observation.asset_id}"
                )
            if observation.node_id is not None and not self.topology.has_node(
                observation.node_id
            ):
                errors.append(
                    f"observation of {observation.variable!r} references unknown "
                    f"node {observation.node_id}"
                )
        if errors:
            raise DomainValidationError(errors, subject="EnergySystem.validate_observations")

    def validate_forecasts(self, forecasts: tuple[Forecast, ...]) -> None:
        """Check that forecasts sit inside the system's declared horizon.

        A forecast beyond the horizon belongs to a different planning problem
        than the one this system was configured for.
        """
        errors: list[str] = []
        horizon = self.time_base.horizon_steps
        for forecast in forecasts:
            if forecast.horizon_steps > horizon:
                errors.append(
                    f"forecast {forecast.forecast_id} has horizon {forecast.horizon_steps} "
                    f"steps, beyond the system's configured horizon of {horizon}"
                )
        if errors:
            raise DomainValidationError(errors, subject="EnergySystem.validate_forecasts")

    def _has_asset(self, asset_ref) -> bool:
        return any(asset.asset_id == asset_ref for asset in self.assets)


# ----------------------------------------------------------------------
# Module-level helpers, reusable independently of the class
# ----------------------------------------------------------------------


def validate_state_against_system(system: EnergySystem, state: EnergyState) -> None:
    """Verify a state covers exactly this system's nodes and assets.

    Raises:
        DomainValidationError: On any mismatch.
    """
    errors: list[str] = []

    if not isinstance(state, EnergyState):
        raise DomainValidationError(
            [f"state must be an EnergyState, got {type(state).__name__}"],
            subject="validate_state_against_system",
        )

    system_nodes = system.node_ids()
    state_nodes = {item.node_id for item in state.node_power}

    for missing in sorted(system_nodes - state_nodes, key=str):
        errors.append(f"state does not report power at node {missing}")
    for extra in sorted(state_nodes - system_nodes, key=str):
        errors.append(f"state reports power at unknown node {extra}")

    battery_ids = {
        asset.asset_id
        for asset in system.assets
        if asset.asset_type.value == "battery"
    }
    for item in state.storage:
        if item.asset_id not in battery_ids:
            errors.append(
                f"state reports storage for {item.asset_id}, which the system does "
                "not own as a battery asset"
            )

    renewable_ids = {
        asset.asset_id
        for asset in system.assets
        if asset.asset_type.value in ("solar", "wind")
    }
    for item in state.renewables:
        if item.asset_id not in renewable_ids:
            errors.append(
                f"state reports renewable output for {item.asset_id}, which the "
                "system does not own as a solar or wind asset"
            )

    if errors:
        raise DomainValidationError(errors, subject="validate_state_against_system")


def validate_actions_against_system(system: EnergySystem, actions: tuple[Action, ...]) -> None:
    """Verify every action targets a known asset with sufficient authority.

    Raises:
        DomainValidationError: Collecting every offending action, not just the
            first, so a whole batch can be corrected in one pass.
    """
    errors: list[str] = []
    for action in actions:
        try:
            action.validate_against_system(system.assets)
        except DomainValidationError as exc:
            errors.extend(exc.errors)
    if errors:
        raise DomainValidationError(errors, subject="validate_actions_against_system")


def _referential_errors(system: EnergySystem) -> list[str]:
    """Check every cross-entity reference in a system."""
    errors: list[str] = []

    seen_assets: set = set()
    for asset in system.assets:
        if asset.asset_id in seen_assets:
            errors.append(f"duplicate asset_id: {asset.asset_id}")
        seen_assets.add(asset.asset_id)
        if not system.topology.has_node(asset.node_id):
            errors.append(
                f"asset {asset.asset_id} attaches to node {asset.node_id}, "
                "which is not in the topology"
            )

    system_nodes = system.node_ids()
    system_assets = system.asset_ids

    seen_constraints: set = set()
    for constraint in system.constraints:
        if constraint.constraint_id in seen_constraints:
            errors.append(f"duplicate constraint_id: {constraint.constraint_id}")
        seen_constraints.add(constraint.constraint_id)
        for asset_ref in constraint.applies_to_asset_ids:
            if asset_ref not in system_assets:
                errors.append(
                    f"constraint {constraint.constraint_id} references unknown "
                    f"asset {asset_ref}"
                )
        for node_ref in constraint.applies_to_node_ids:
            if node_ref not in system_nodes:
                errors.append(
                    f"constraint {constraint.constraint_id} references unknown "
                    f"node {node_ref}"
                )

    seen_objectives: set = set()
    for objective in system.objectives:
        if objective.objective_id in seen_objectives:
            errors.append(f"duplicate objective_id: {objective.objective_id}")
        seen_objectives.add(objective.objective_id)

    return errors
