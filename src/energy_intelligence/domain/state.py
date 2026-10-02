"""Energy state.

Represents what the system knows about the energy environment at one instant.

Two decisions are load-bearing here:

1. **Per-node power balance is enforced.** The chosen topology model is
   node-granular, so :class:`NodePower` must close to within a tolerance at each
   node, not merely system-wide. This is what distinguishes a distribution model
   from a lumped one, and it is the invariant Phase 11's digital twin will
   simulate against.

2. **State origin is explicit.** :class:`~energy_intelligence.domain.enums.StateOrigin`
   distinguishes observed, simulated, predicted and assimilated states. The
   Phase 2 brief requires these not to be interchangeable, because the
   project's central research signal is comparing a learned model's prediction
   against a simulation against reality. If origin were implicit, that comparison
   would be meaningless.

No power flow is solved. Voltage and current appear as *observations* only.
"""

from __future__ import annotations

from dataclasses import dataclass

from .enums import QualityFlag, StateOrigin
from .errors import DomainValidationError
from .identifiers import AssetId, NodeId
from .provenance import Provenance
from .quality import DataQuality
from .quantities import Quantity

__all__ = [
    "NodePower",
    "StorageState",
    "RenewableState",
    "DemandState",
    "EVState",
    "GridState",
    "EnergyState",
    "POWER_BALANCE_TOLERANCE",
]

#: Per-node balance tolerance in kW. Deliberately generous enough for
#: unmodelled losses at a distribution feeder while still catching a sign error,
#: a double count, or a missing asset. Phase 11 may tighten this once a
#: simulator defines the loss model.
POWER_BALANCE_TOLERANCE = 0.5


@dataclass(frozen=True, slots=True)
class NodePower:
    """Power balance at one node.

    Sign convention, applied uniformly: generation and grid import are positive,
    load and grid export are negative. Storage is a signed injection.

    Attributes:
        node_id: The node this balance describes.
        generation_kw: Total renewable generation at the node.
        load_kw: Total demand at the node.
        storage_kw: Net storage injection: positive discharging into the node,
            negative charging from it.
        grid_kw: Positive importing from the utility, negative exporting to it.
        unaccounted_kw: Explicit residual when the reported figures do not close.
            Recorded rather than hidden, so Phase 3's aggregation can be audited.
    """

    node_id: NodeId
    generation_kw: Quantity
    load_kw: Quantity
    storage_kw: Quantity
    grid_kw: Quantity
    unaccounted_kw: Quantity | None = None

    def __post_init__(self) -> None:
        errors: list[str] = []
        if not isinstance(self.node_id, NodeId):
            errors.append(f"node_id must be a NodeId, got {type(self.node_id).__name__}")

        for field_name in ("generation_kw", "load_kw", "storage_kw", "grid_kw"):
            value = getattr(self, field_name)
            if not isinstance(value, Quantity):
                errors.append(f"{field_name} must be a Quantity, got {type(value).__name__}")

        for field_name in ("generation_kw", "load_kw"):
            value = getattr(self, field_name)
            if isinstance(value, Quantity) and value.value < 0:
                errors.append(f"{field_name} must be non-negative, got {value.value}")

        if self.unaccounted_kw is not None and not isinstance(self.unaccounted_kw, Quantity):
            errors.append("unaccounted_kw must be a Quantity or None")

        if errors:
            raise DomainValidationError(errors, subject="NodePower")

    @property
    def total_injection_kw(self) -> float:
        """Sum of all injections at the node."""
        return (
            self.generation_kw.value
            + self.storage_kw.value
            + self.grid_kw.value
            - self.load_kw.value
        )

    @property
    def residual_kw(self) -> float:
        """Signed power-balance residual in kW.

        Should be within :data:`POWER_BALANCE_TOLERANCE` of zero.
        """
        if self.unaccounted_kw is not None:
            return self.total_injection_kw + self.unaccounted_kw.value
        return self.total_injection_kw

    @property
    def is_balanced(self) -> bool:
        return abs(self.residual_kw) <= POWER_BALANCE_TOLERANCE

    def unit_errors(self) -> list[str]:
        """Report non-conforming units, if any."""
        problems = []
        for field_name in ("generation_kw", "load_kw", "storage_kw", "grid_kw"):
            value = getattr(self, field_name)
            if isinstance(value, Quantity) and value.unit.value != "kW":
                problems.append(
                    f"{field_name} must be expressed in kW, got {value.unit.value}"
                )
        return problems


@dataclass(frozen=True, slots=True)
class StorageState:
    """Observed state of one battery.

    Attributes:
        asset_id: The battery.
        state_of_charge: Stored energy as a fraction of usable capacity, in
            ``[0, 1]``.
        charge_power_kw: Charging power, non-negative.
        discharge_power_kw: Discharging power, non-negative.
        available_energy: Energy that can still be delivered above the minimum
            state of charge. Derived from the asset; recorded so downstream
            phases need not recompute it inconsistently.
        quality: Data-quality metadata.
    """

    asset_id: AssetId
    state_of_charge: float
    charge_power_kw: Quantity
    discharge_power_kw: Quantity
    available_energy: Quantity
    quality: DataQuality

    def __post_init__(self) -> None:
        errors: list[str] = []
        if not isinstance(self.asset_id, AssetId):
            errors.append(f"asset_id must be an AssetId, got {type(self.asset_id).__name__}")

        if isinstance(self.state_of_charge, bool) or not isinstance(self.state_of_charge, (int, float)):
            errors.append(f"state_of_charge must be a real number, got {self.state_of_charge!r}")
        elif not 0.0 <= self.state_of_charge <= 1.0:
            errors.append(
                f"state_of_charge must lie in [0, 1], got {self.state_of_charge}"
            )

        for field_name in ("charge_power_kw", "discharge_power_kw", "available_energy"):
            value = getattr(self, field_name)
            if not isinstance(value, Quantity):
                errors.append(f"{field_name} must be a Quantity, got {type(value).__name__}")
                continue
            if value.value < 0:
                errors.append(f"{field_name} must be non-negative, got {value.value}")

        if isinstance(self.charge_power_kw, Quantity) and isinstance(
            self.discharge_power_kw, Quantity
        ):
            if self.charge_power_kw.value > 0 and self.discharge_power_kw.value > 0:
                errors.append(
                    "a battery cannot charge and discharge simultaneously; "
                    f"got charge={self.charge_power_kw.value}, "
                    f"discharge={self.discharge_power_kw.value}"
                )

        if not isinstance(self.quality, DataQuality):
            errors.append(f"quality must be a DataQuality, got {type(self.quality).__name__}")

        if errors:
            raise DomainValidationError(errors, subject="StorageState")

    @property
    def net_power_kw(self) -> float:
        """Signed net injection: positive discharging, negative charging."""
        return self.discharge_power_kw.value - self.charge_power_kw.value


@dataclass(frozen=True, slots=True)
class RenewableState:
    """Observed state of one renewable generator.

    Attributes:
        asset_id: The generator.
        available_power: Power the resource could produce right now.
        actual_power: Power actually produced.
        curtailment_kw: Available minus actual; non-negative.
        quality: Data-quality metadata.
    """

    asset_id: AssetId
    available_power: Quantity
    actual_power: Quantity
    quality: DataQuality

    def __post_init__(self) -> None:
        errors: list[str] = []
        if not isinstance(self.asset_id, AssetId):
            errors.append(f"asset_id must be an AssetId, got {type(self.asset_id).__name__}")
        for field_name in ("available_power", "actual_power"):
            value = getattr(self, field_name)
            if not isinstance(value, Quantity):
                errors.append(f"{field_name} must be a Quantity, got {type(value).__name__}")
            elif value.value < 0:
                errors.append(f"{field_name} must be non-negative, got {value.value}")
        if not isinstance(self.quality, DataQuality):
            errors.append(f"quality must be a DataQuality, got {type(self.quality).__name__}")
        if errors:
            raise DomainValidationError(errors, subject="RenewableState")

    @property
    def curtailment_kw(self) -> float:
        return self.available_power.value - self.actual_power.value

    @property
    def curtailment_fraction(self) -> float:
        """Fraction of available renewable power not used, in ``[0, 1]``."""
        if self.available_power.value <= 0:
            return 0.0
        return max(0.0, min(1.0, self.curtailment_kw / self.available_power.value))


@dataclass(frozen=True, slots=True)
class DemandState:
    """Observed demand.

    Attributes:
        total_demand_kw: Aggregate demand.
        by_category: Demand split by category, e.g. ``{"residential": 120.0,
            "commercial": 80.0}``. Values are kW.
        quality: Data-quality metadata.
    """

    total_demand_kw: Quantity
    quality: DataQuality
    by_category: dict[str, float] | None = None

    def __post_init__(self) -> None:
        errors: list[str] = []
        if not isinstance(self.total_demand_kw, Quantity):
            errors.append(f"total_demand_kw must be a Quantity, got {type(self.total_demand_kw).__name__}")
        elif self.total_demand_kw.value < 0:
            errors.append(f"total_demand_kw must be non-negative, got {self.total_demand_kw.value}")
        if not isinstance(self.quality, DataQuality):
            errors.append(f"quality must be a DataQuality, got {type(self.quality).__name__}")
        if self.by_category is not None:
            if not isinstance(self.by_category, dict):
                errors.append(
                    f"by_category must be a dict or None, got {type(self.by_category).__name__}"
                )
            else:
                all_numeric = True
                for key, value in self.by_category.items():
                    if isinstance(value, bool) or not isinstance(value, (int, float)):
                        errors.append(f"by_category[{key!r}] must be a real number, got {value!r}")
                        all_numeric = False
                    elif value < 0:
                        errors.append(f"by_category[{key!r}] must be non-negative, got {value}")
                # Only reconcile the total once every component is known numeric;
                # summing mixed types would raise TypeError instead of reporting
                # the validation failure.
                if all_numeric and isinstance(self.total_demand_kw, Quantity) and self.by_category:
                    total = sum(self.by_category.values())
                    if abs(total - self.total_demand_kw.value) > 1e-6:
                        errors.append(
                            "by_category must sum to total_demand_kw; got "
                            f"{total} vs {self.total_demand_kw.value}"
                        )
        if errors:
            raise DomainValidationError(errors, subject="DemandState")


@dataclass(frozen=True, slots=True)
class EVState:
    """Observed state of EV charging.

    Attributes:
        connected_vehicles: Vehicles currently connected.
        available_connectors: Connectors not in use.
        charging_power_kw: Total charging power.
        deferrable_power_kw: Charging power that could be deferred without
            violating a departure deadline. Derived; recorded so downstream
            phases consume one consistent figure.
        quality: Data-quality metadata.
    """

    connected_vehicles: int
    available_connectors: int
    charging_power_kw: Quantity
    deferrable_power_kw: Quantity
    quality: DataQuality

    def __post_init__(self) -> None:
        errors: list[str] = []
        for field_name in ("connected_vehicles", "available_connectors"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int):
                errors.append(f"{field_name} must be an int, got {type(value).__name__}")
            elif value < 0:
                errors.append(f"{field_name} must be non-negative, got {value}")

        for field_name in ("charging_power_kw", "deferrable_power_kw"):
            value = getattr(self, field_name)
            if not isinstance(value, Quantity):
                errors.append(f"{field_name} must be a Quantity, got {type(value).__name__}")
            elif value.value < 0:
                errors.append(f"{field_name} must be non-negative, got {value.value}")

        if isinstance(self.charging_power_kw, Quantity) and isinstance(
            self.deferrable_power_kw, Quantity
        ):
            if self.deferrable_power_kw.value > self.charging_power_kw.value:
                errors.append(
                    "deferrable_power_kw must not exceed charging_power_kw; got "
                    f"{self.deferrable_power_kw.value} vs {self.charging_power_kw.value}"
                )

        if not isinstance(self.quality, DataQuality):
            errors.append(f"quality must be a DataQuality, got {type(self.quality).__name__}")

        if errors:
            raise DomainValidationError(errors, subject="EVState")


@dataclass(frozen=True, slots=True)
class GridState:
    """Observed state at the utility connection.

    Voltage and frequency are recorded as observations with their own quality
    flags. They are **not** derived: Phase 2 performs no power flow, so a
    modelled voltage would be a fabrication.

    Attributes:
        import_kw: Power drawn from the utility, non-negative.
        export_kw: Power delivered to the utility, non-negative.
        frequency_hz: Observed system frequency.
        voltage_by_node: Observed voltage at each measured node.
        quality: Data-quality metadata for the connection-level figures.
    """

    import_kw: Quantity
    export_kw: Quantity
    quality: DataQuality
    frequency_hz: Quantity | None = None
    voltage_by_node: dict[str, Quantity] | None = None

    def __post_init__(self) -> None:
        errors: list[str] = []
        for field_name in ("import_kw", "export_kw"):
            value = getattr(self, field_name)
            if not isinstance(value, Quantity):
                errors.append(f"{field_name} must be a Quantity, got {type(value).__name__}")
            elif value.value < 0:
                errors.append(f"{field_name} must be non-negative, got {value.value}")
        if not isinstance(self.quality, DataQuality):
            errors.append(f"quality must be a DataQuality, got {type(self.quality).__name__}")
        if self.frequency_hz is not None:
            if not isinstance(self.frequency_hz, Quantity):
                errors.append("frequency_hz must be a Quantity or None")
            elif self.frequency_hz.value <= 0:
                errors.append(f"frequency_hz must be positive, got {self.frequency_hz.value}")
        if self.voltage_by_node is not None:
            if not isinstance(self.voltage_by_node, dict):
                errors.append("voltage_by_node must be a dict or None")
            else:
                for node_ref, value in self.voltage_by_node.items():
                    if not isinstance(value, Quantity):
                        errors.append(
                            f"voltage_by_node[{node_ref!r}] must be a Quantity"
                        )
                    elif value.value <= 0:
                        errors.append(
                            f"voltage_by_node[{node_ref!r}] must be positive, got {value.value}"
                        )
        if errors:
            raise DomainValidationError(errors, subject="GridState")

    @property
    def net_kw(self) -> float:
        """Signed grid flow: positive importing, negative exporting."""
        return self.import_kw.value - self.export_kw.value


@dataclass(frozen=True, slots=True)
class EnergyState:
    """The complete system state at one instant.

    Attributes:
        timestamp: ISO-8601 instant, aligned to the governing :class:`TimeBase`.
        origin: Whether this state was observed, simulated, predicted or
            assimilated. Mandatory, and never inferred.
        node_power: Per-node power balance, one entry per node.
        demand: Aggregate demand.
        storage: Per-battery state. May be empty if no batteries are modelled.
        renewables: Per-generator state. May be empty.
        ev: EV charging state. May be ``None``.
        grid: Utility connection state.
        provenance: Where this state came from. Mandatory.
        timebase_id: Identifier of the governing :class:`TimeBase`, so a state can
            be checked against the grid it claims to sit on.
    """

    timestamp: str
    origin: StateOrigin
    node_power: tuple[NodePower, ...]
    demand: DemandState
    grid: GridState
    provenance: Provenance
    storage: tuple[StorageState, ...] = ()
    renewables: tuple[RenewableState, ...] = ()
    ev: EVState | None = None
    timebase_id: str = ""

    def __post_init__(self) -> None:
        errors: list[str] = []

        if not isinstance(self.timestamp, str) or not self.timestamp.strip():
            errors.append(f"timestamp must be a non-empty ISO-8601 string, got {self.timestamp!r}")
        if not isinstance(self.origin, StateOrigin):
            errors.append(f"origin must be a StateOrigin, got {type(self.origin).__name__}")
        if not isinstance(self.provenance, Provenance):
            errors.append(f"provenance must be a Provenance, got {type(self.provenance).__name__}")
        if not isinstance(self.demand, DemandState):
            errors.append(f"demand must be a DemandState, got {type(self.demand).__name__}")
        if not isinstance(self.grid, GridState):
            errors.append(f"grid must be a GridState, got {type(self.grid).__name__}")

        if not isinstance(self.node_power, tuple) or not self.node_power:
            errors.append(
                "node_power must be a non-empty tuple; a state must report power "
                "at every modelled node"
            )
        elif not all(isinstance(item, NodePower) for item in self.node_power):
            errors.append("every entry in node_power must be a NodePower")
        else:
            seen: set[NodeId] = set()
            for item in self.node_power:
                if item.node_id in seen:
                    errors.append(f"duplicate node in node_power: {item.node_id}")
                seen.add(item.node_id)
                errors.extend(item.unit_errors())
                if not item.is_balanced:
                    errors.append(
                        f"power balance at {item.node_id} does not close: residual "
                        f"{item.residual_kw:.4f} kW exceeds tolerance "
                        f"{POWER_BALANCE_TOLERANCE} kW"
                    )

        for field_name, expected in (("storage", StorageState), ("renewables", RenewableState)):
            value = getattr(self, field_name)
            if not isinstance(value, tuple):
                errors.append(f"{field_name} must be a tuple, got {type(value).__name__}")
                continue
            if not all(isinstance(item, expected) for item in value):
                errors.append(f"every entry in {field_name} must be a {expected.__name__}")
            seen_assets: set[AssetId] = set()
            for item in value:
                if item.asset_id in seen_assets:
                    errors.append(f"duplicate asset in {field_name}: {item.asset_id}")
                seen_assets.add(item.asset_id)

        if self.ev is not None and not isinstance(self.ev, EVState):
            errors.append(f"ev must be an EVState or None, got {type(self.ev).__name__}")

        if errors:
            raise DomainValidationError(errors, subject="EnergyState")

    @property
    def is_decision_grade(self) -> bool:
        """Whether every component of this state is clean enough to act on.

        Conservative: any missing, estimated, interpolated, stale, out-of-range,
        outlier or sensor-error component disqualifies the whole state. Phase 13
        may relax this per-component; the strict default is the safe one.
        """
        components = [self.demand.quality, self.grid.quality]
        components.extend(item.quality for item in self.storage)
        components.extend(item.quality for item in self.renewables)
        if self.ev is not None:
            components.append(self.ev.quality)
        return all(component.is_usable_for_decision for component in components)

    def quality_flags(self) -> set[QualityFlag]:
        """Union of every quality flag present in the state."""
        flags: set[QualityFlag] = set()
        for component in (self.demand.quality, self.grid.quality):
            flags |= component.flags
        for item in self.storage:
            flags |= item.quality.flags
        for item in self.renewables:
            flags |= item.quality.flags
        if self.ev is not None:
            flags |= self.ev.quality.flags
        return flags
