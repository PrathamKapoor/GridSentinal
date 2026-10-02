"""Typed asset model.

Design principle, taken from the Phase 2 brief: an asset has explicit identity,
and **specific asset types add only their own fields**. A solar asset does not
carry a state-of-charge field, and a battery does not carry an irradiance field.
That is why this is a shallow hierarchy with one concrete class per asset type
rather than one ``Asset`` class with every optional field on it.

Concrete classes per :class:`~energy_intelligence.domain.enums.AssetType`:

======================  ==================================================
:class:`Asset`         Infrastructure: grid connection, substation, load.
:class:`SolarAsset`    Capacity, tilt, azimuth, inverter rating.
:class:`WindAsset`     Rated power, cut-in/cut-out/rated speeds.
:class:`Battery`       Usable energy, SOC bounds, charge/discharge limits,
                       round-trip efficiency.
:class:`EVCharger`     Max power, minimum state of charge on return.
:class:`FlexibleLoad`  Min/max power, shiftable energy, priority.
======================  ==================================================

Every asset declares an :class:`~energy_intelligence.domain.enums.AuthorityLevel`,
which determines whether an action may target it (D-028). No bus count, feeder
count or asset count is hard-coded anywhere: those are configuration.
"""

from __future__ import annotations

from dataclasses import dataclass

from .enums import AssetType, AuthorityLevel, Unit
from .errors import DomainValidationError
from .identifiers import AssetId, NodeId
from .quantities import Quantity

__all__ = [
    "Asset",
    "SolarAsset",
    "WindAsset",
    "Battery",
    "EVCharger",
    "FlexibleLoad",
    "ASSET_TYPES",
    "asset_class_for",
]

#: (energy unit, power unit) pairs whose ratio is a duration in hours.
#: Anything outside this table yields ``None`` rather than a wrong number.
_DURATION_PAIRS: frozenset[tuple[Unit, Unit]] = frozenset(
    {
        (Unit.KILOWATT_HOURS, Unit.KILOWATT),
        (Unit.MEGAWATT_HOURS, Unit.MEGAWATT),
    }
)


@dataclass(frozen=True, slots=True)
class Asset:
    """Base asset: identity, location, authority, availability.

    ``availability`` and ``metadata`` are deliberately **required** rather than
    defaulted. A dataclass subclass cannot introduce a required field once the
    base declares defaults, so defaulting them here would force every
    type-specific field (``capacity_kw``, ``usable_energy``, ...) to be
    expressed as ``None``-with-a-default, which misrepresents them as optional
    to both readers and type checkers. Making them required keeps every field's
    optionality honest and stops ``availability`` -- an input that materially
    affects any flexibility calculation -- from being forgotten.

    Attributes:
        asset_id: Unique identifier.
        asset_type: Which kind of asset this is.
        name: Human-readable label.
        node_id: Node this asset attaches to.
        authority: How much control the system has over this asset.
        rated_power: Nameplate power rating.
        availability: Fraction of rated capability currently available, in
            ``[0, 1]``. ``1.0`` means fully available; this is an *input*, not a
            prediction.
        metadata: Free-form, non-authoritative extras.
    """

    asset_type: AssetType
    asset_id: AssetId
    name: str
    node_id: NodeId
    authority: AuthorityLevel
    rated_power: Quantity
    availability: float
    metadata: dict[str, str] | None

    def __post_init__(self) -> None:
        errors: list[str] = []
        if not isinstance(self.asset_type, AssetType):
            errors.append(f"asset_type must be an AssetType, got {type(self.asset_type).__name__}")
        if not isinstance(self.asset_id, AssetId):
            errors.append(f"asset_id must be an AssetId, got {type(self.asset_id).__name__}")
        if not isinstance(self.name, str) or not self.name.strip():
            errors.append(f"name must be a non-empty string, got {self.name!r}")
        if not isinstance(self.node_id, NodeId):
            errors.append(f"node_id must be a NodeId, got {type(self.node_id).__name__}")
        if not isinstance(self.authority, AuthorityLevel):
            errors.append(
                f"authority must be an AuthorityLevel, got {type(self.authority).__name__}"
            )
        if not isinstance(self.rated_power, Quantity):
            errors.append(f"rated_power must be a Quantity, got {type(self.rated_power).__name__}")
        elif self.rated_power.value < 0:
            errors.append(f"rated_power must be non-negative, got {self.rated_power.value}")

        if isinstance(self.availability, bool) or not isinstance(self.availability, (int, float)):
            errors.append(
                f"availability must be a real number, got {type(self.availability).__name__}"
            )
        elif not 0.0 <= self.availability <= 1.0:
            errors.append(f"availability must lie in [0, 1], got {self.availability}")

        if self.metadata is not None and not isinstance(self.metadata, dict):
            errors.append(f"metadata must be a dict or None, got {type(self.metadata).__name__}")

        if errors:
            raise DomainValidationError(errors, subject=type(self).__name__)

    @property
    def is_controllable(self) -> bool:
        """Whether *some* action may be issued against this asset."""
        from .enums import CONTROLLABLE_AUTHORITY_LEVELS

        return self.authority in CONTROLLABLE_AUTHORITY_LEVELS

    @property
    def available_power(self) -> Quantity:
        """Rated power scaled by availability.

        Derived, not observed: it is a function of a declared rating and a
        declared availability factor, both of which are inputs.
        """
        return Quantity(value=self.rated_power.value * self.availability, unit=self.rated_power.unit)


@dataclass(frozen=True, slots=True)
class SolarAsset(Asset):
    """Photovoltaic generator.

    Attributes:
        capacity_kw: Nameplate DC capacity.
        tilt_degrees: Panel tilt from horizontal, for irradiance-to-power
            conversion in a later phase.
        azimuth_degrees: Panel orientation, degrees clockwise from north.
        has_inverter: Whether an inverter model applies.
    """

    capacity_kw: Quantity
    tilt_degrees: Quantity | None = None
    azimuth_degrees: Quantity | None = None
    has_inverter: bool = False

    def __post_init__(self) -> None:
        errors: list[str] = []
        if not isinstance(self.capacity_kw, Quantity):
            errors.append(f"capacity_kw must be a Quantity, got {type(self.capacity_kw).__name__}")
        elif self.capacity_kw.value < 0:
            errors.append(f"capacity_kw must be non-negative, got {self.capacity_kw.value}")

        for field_name, bound in (("tilt_degrees", 90.0), ("azimuth_degrees", 360.0)):
            value = getattr(self, field_name)
            if value is None:
                continue
            if not isinstance(value, Quantity):
                errors.append(f"{field_name} must be a Quantity or None")
                continue
            if not 0.0 <= value.value < bound:
                errors.append(f"{field_name} must lie in [0, {bound:g}), got {value.value}")

        if errors:
            raise DomainValidationError(errors, subject="SolarAsset")


@dataclass(frozen=True, slots=True)
class WindAsset(Asset):
    """Wind turbine or wind farm.

    Attributes:
        cut_in_speed: Below this wind speed the turbine produces nothing.
        rated_speed: Wind speed at which rated power is reached.
        cut_out_speed: Above this the turbine shuts down for safety.
    """

    cut_in_speed: Quantity | None = None
    rated_speed: Quantity | None = None
    cut_out_speed: Quantity | None = None

    def __post_init__(self) -> None:
        errors: list[str] = []
        speeds = {
            "cut_in_speed": self.cut_in_speed,
            "rated_speed": self.rated_speed,
            "cut_out_speed": self.cut_out_speed,
        }
        for field_name, value in speeds.items():
            if value is None:
                continue
            if not isinstance(value, Quantity):
                errors.append(f"{field_name} must be a Quantity or None")
                continue
            if value.value < 0:
                errors.append(f"{field_name} must be non-negative, got {value.value}")

        if all(speeds[name] is not None for name in ("cut_in_speed", "rated_speed", "cut_out_speed")):
            if not (self.cut_in_speed.value < self.rated_speed.value < self.cut_out_speed.value):
                errors.append(
                    "speeds must satisfy cut_in < rated < cut_out; got "
                    f"{self.cut_in_speed.value}, {self.rated_speed.value}, "
                    f"{self.cut_out_speed.value}"
                )

        if errors:
            raise DomainValidationError(errors, subject="WindAsset")


@dataclass(frozen=True, slots=True)
class Battery(Asset):
    """Battery energy storage.

    Attributes:
        usable_energy: Deliverable energy at the depth-of-discharge limit.
        min_soc: Lowest permitted state of charge, as a fraction.
        max_soc: Highest permitted state of charge, as a fraction.
        max_charge_power: Maximum charge power.
        max_discharge_power: Maximum discharge power.
        round_trip_efficiency: Fraction of stored energy recoverable, in
            ``(0, 1]``.
    """

    usable_energy: Quantity
    min_soc: float
    max_soc: float
    max_charge_power: Quantity | None
    max_discharge_power: Quantity | None
    round_trip_efficiency: float

    def __post_init__(self) -> None:
        errors: list[str] = []

        if not isinstance(self.usable_energy, Quantity):
            errors.append(f"usable_energy must be a Quantity, got {type(self.usable_energy).__name__}")
        elif self.usable_energy.value < 0:
            errors.append(
                f"usable_energy must be non-negative, got {self.usable_energy.value}"
            )

        for field_name in ("min_soc", "max_soc"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                errors.append(f"{field_name} must be a real number, got {value!r}")
            elif not 0.0 <= value <= 1.0:
                errors.append(f"{field_name} must lie in [0, 1], got {value}")

        if isinstance(self.min_soc, (int, float)) and isinstance(self.max_soc, (int, float)):
            if self.min_soc > self.max_soc:
                errors.append(
                    f"min_soc {self.min_soc} must not exceed max_soc {self.max_soc}"
                )

        if (
            isinstance(self.round_trip_efficiency, bool)
            or not isinstance(self.round_trip_efficiency, (int, float))
        ):
            errors.append(
                f"round_trip_efficiency must be a real number, got {self.round_trip_efficiency!r}"
            )
        elif not 0.0 < self.round_trip_efficiency <= 1.0:
            errors.append(
                "round_trip_efficiency must lie in (0, 1]; got "
                f"{self.round_trip_efficiency}"
            )

        for field_name in ("max_charge_power", "max_discharge_power"):
            value = getattr(self, field_name)
            if value is None:
                continue
            if not isinstance(value, Quantity):
                errors.append(f"{field_name} must be a Quantity or None")
                continue
            if value.value < 0:
                errors.append(f"{field_name} must be non-negative, got {value.value}")

        if errors:
            raise DomainValidationError(errors, subject="Battery")

    @property
    def usable_soc_window(self) -> tuple[float, float]:
        """The usable state-of-charge window, as ``(min, max)``."""
        return (self.min_soc, self.max_soc)

    @property
    def nameplate_duration_hours(self) -> float | None:
        """Usable energy divided by rated power: hours of rated discharge.

        ``None`` unless the energy and power units form a dimensionally
        compatible pair. This is a dimensional consistency check, not a unit
        conversion -- Phase 2 deliberately performs no conversion (D-039).

        Note that comparing the two ``unit`` values directly would always fail,
        because ``kWh`` is never equal to ``kW``; the pairing below is what makes
        the ratio meaningful.
        """
        pair = (self.usable_energy.unit, self.rated_power.unit)
        if pair not in _DURATION_PAIRS or self.rated_power.value <= 0:
            return None
        return self.usable_energy.value / self.rated_power.value


@dataclass(frozen=True, slots=True)
class EVCharger(Asset):
    """Electric vehicle charger or aggregated charging point.

    Attributes:
        min_return_soc: State of charge a connected vehicle requires on
            return, as a fraction. This is the constraint that makes EV
            flexibility non-trivial: energy must be returned.
        num_connectors: Number of simultaneous connections supported.
    """

    min_return_soc: float
    num_connectors: int

    def __post_init__(self) -> None:
        errors: list[str] = []
        if isinstance(self.min_return_soc, bool) or not isinstance(self.min_return_soc, (int, float)):
            errors.append(f"min_return_soc must be a real number, got {self.min_return_soc!r}")
        elif not 0.0 <= self.min_return_soc <= 1.0:
            errors.append(f"min_return_soc must lie in [0, 1], got {self.min_return_soc}")

        if isinstance(self.num_connectors, bool) or not isinstance(self.num_connectors, int):
            errors.append(f"num_connectors must be an int, got {type(self.num_connectors).__name__}")
        elif self.num_connectors < 1:
            errors.append(f"num_connectors must be at least 1, got {self.num_connectors}")

        if errors:
            raise DomainValidationError(errors, subject="EVCharger")


@dataclass(frozen=True, slots=True)
class FlexibleLoad(Asset):
    """Shiftable or curtailable load.

    Attributes:
        min_power: Minimum power the load must draw.
        max_power: Maximum power the load may draw.
        shiftable_energy: Energy that may be moved within the scheduling
            window, as a fraction of consumption. ``0.0`` means fully
            non-shiftable.
        priority: Relative importance; higher means less deferrable. Used by
            Phase 10 to break ties when comfort cannot be preserved.
        is_hvac: Whether this is thermal load, which carries comfort coupling.
    """

    min_power: Quantity | None
    max_power: Quantity | None
    shiftable_energy: float
    priority: int
    is_hvac: bool

    def __post_init__(self) -> None:
        errors: list[str] = []

        if isinstance(self.shiftable_energy, bool) or not isinstance(
            self.shiftable_energy, (int, float)
        ):
            errors.append(f"shiftable_energy must be a real number, got {self.shiftable_energy!r}")
        elif not 0.0 <= self.shiftable_energy <= 1.0:
            errors.append(f"shiftable_energy must lie in [0, 1], got {self.shiftable_energy}")

        if isinstance(self.priority, bool) or not isinstance(self.priority, int):
            errors.append(f"priority must be an int, got {type(self.priority).__name__}")
        elif self.priority < 0:
            errors.append(f"priority must be non-negative, got {self.priority}")

        for field_name in ("min_power", "max_power"):
            value = getattr(self, field_name)
            if value is None:
                continue
            if not isinstance(value, Quantity):
                errors.append(f"{field_name} must be a Quantity or None")
                continue
            if value.value < 0:
                errors.append(f"{field_name} must be non-negative, got {value.value}")

        if self.min_power is not None and self.max_power is not None:
            if self.min_power.unit != self.max_power.unit:
                errors.append(
                    f"min_power unit {self.min_power.unit} differs from "
                    f"max_power unit {self.max_power.unit}"
                )
            elif self.min_power.value > self.max_power.value:
                errors.append(
                    f"min_power {self.min_power.value} must not exceed "
                    f"max_power {self.max_power.value}"
                )

        if errors:
            raise DomainValidationError(errors, subject="FlexibleLoad")


#: Maps each asset type to its implementing class. Used by serialization to
#: reconstruct the correct concrete type, and by validation to check that a
#: declared asset type matches its class.
ASSET_TYPES: dict[AssetType, type[Asset]] = {
    AssetType.GRID_CONNECTION: Asset,
    AssetType.SUBSTATION: Asset,
    AssetType.LOAD: Asset,
    AssetType.SOLAR: SolarAsset,
    AssetType.WIND: WindAsset,
    AssetType.BATTERY: Battery,
    AssetType.EV_CHARGER: EVCharger,
    AssetType.EV_FLEET: EVCharger,
    AssetType.FLEXIBLE_LOAD: FlexibleLoad,
    AssetType.HVAC: FlexibleLoad,
}


def asset_class_for(asset_type: AssetType) -> type[Asset]:
    """Return the implementing class for ``asset_type``.

    Raises:
        DomainValidationError: If the type has no implementing class, which
            would mean the asset catalogue and the class hierarchy disagree.
    """
    if not isinstance(asset_type, AssetType):
        raise DomainValidationError(
            [f"asset_type must be an AssetType, got {type(asset_type).__name__}"],
            subject="asset_class_for",
        )
    try:
        return ASSET_TYPES[asset_type]
    except KeyError as exc:  # pragma: no cover - guards catalogue drift
        raise DomainValidationError(
            [f"no asset class registered for {asset_type.value!r}"],
            subject="asset_class_for",
        ) from exc
