"""Constraint categories.

Phase 2 **declares** constraints; it does not enforce them and does not commit to
an algebraic formulation. That restraint is deliberate: the formulation depends
on the optimizer choice, which the brief places in Phase 10, and a formulation
written in Phase 2 would prejudge it. See ``decisions.md`` (D-030).

What Phase 2 does fix is the *category vocabulary* and, critically, the binding
references. A constraint that names an asset or node which does not exist is
rejected here, so the constraint set cannot silently reference a phantom
resource -- a failure mode that would otherwise surface much later as an
infeasible or, worse, a wrongly feasible optimisation.

Physics-dependent categories (``voltage_limits``, ``thermal_loading_limits``)
exist in the vocabulary because the chosen topology model admits them, but they
are declared, not computed: Phase 2 runs no power flow.
"""

from __future__ import annotations

from dataclasses import dataclass

from .enums import ConstraintCategory, Unit
from .errors import DomainValidationError
from .identifiers import AssetId, ConstraintId, NodeId
from .provenance import Provenance

__all__ = ["Constraint", "CONSTRAINT_UNITS", "unit_for_constraint"]

#: Unit in which each category's bound is naturally expressed. Recorded so a
#: later phase does not have to infer it, and so Phase 10 can reject a bound
#: expressed in the wrong unit at the boundary.
CONSTRAINT_UNITS: dict[ConstraintCategory, Unit] = {
    ConstraintCategory.POWER_BALANCE: Unit.KILOWATT,
    ConstraintCategory.STATE_OF_CHARGE_LIMITS: Unit.FRACTION,
    ConstraintCategory.CHARGE_DISCHARGE_POWER_LIMITS: Unit.KILOWATT,
    ConstraintCategory.RENEWABLE_AVAILABILITY: Unit.KILOWATT,
    ConstraintCategory.FLEXIBLE_LOAD_BOUNDS: Unit.KILOWATT,
    ConstraintCategory.GRID_CONNECTION_CAPACITY: Unit.KILOWATT,
    ConstraintCategory.VOLTAGE_LIMITS: Unit.VOLT,
    ConstraintCategory.THERMAL_LOADING_LIMITS: Unit.AMPERE,
    ConstraintCategory.MINIMUM_UP_TIME: Unit.MINUTES,
    ConstraintCategory.RAMP_RATE_LIMITS: Unit.KILOWATT,
    ConstraintCategory.EV_DEPARTURE_DEADLINE: Unit.FRACTION,
    ConstraintCategory.DEMAND_FLEXIBILITY_LIMIT: Unit.KILOWATT,
}

#: Categories that cannot be evaluated without a power-flow solver. Declared so
#: a caller can tell a declared-but-unenforceable constraint from an immediately
#: actionable one.
PHYSICS_DEPENDENT: frozenset[ConstraintCategory] = frozenset(
    {
        ConstraintCategory.VOLTAGE_LIMITS,
        ConstraintCategory.THERMAL_LOADING_LIMITS,
    }
)


def unit_for_constraint(category: ConstraintCategory) -> Unit:
    """Return the natural unit for a constraint category."""
    if not isinstance(category, ConstraintCategory):
        raise DomainValidationError(
            [f"category must be a ConstraintCategory, got {type(category).__name__}"],
            subject="unit_for_constraint",
        )
    return CONSTRAINT_UNITS[category]


@dataclass(frozen=True, slots=True)
class Constraint:
    """One declared constraint on the system or one of its assets.

    Asset and node references are *typed* identifiers, not bare strings. An
    earlier revision used ``str`` here while everything else used ``AssetId``;
    because ``AssetId`` is not a ``str`` subclass, every cross-reference silently
    failed to resolve. Typing them removes that whole class of bug.

    Attributes:
        constraint_id: Unique identifier.
        category: Which kind of constraint this is.
        description: Human-readable statement of the constraint in words.
        applies_to_asset_ids: Assets this constrains. May be empty for
            system-wide constraints such as power balance.
        applies_to_node_ids: Nodes this constrains. May be empty.
        lower_bound: Optional lower bound value.
        upper_bound: Optional upper bound value.
        unit: Unit both bounds are expressed in. Must match the category's
            natural unit.
        requires_power_flow: True when enforcement needs a power-flow solver,
            which Phase 2 does not provide.
        provenance: Why this constraint exists. Mandatory.
    """

    constraint_id: ConstraintId
    category: ConstraintCategory
    description: str
    provenance: Provenance
    applies_to_asset_ids: tuple[AssetId, ...] = ()
    applies_to_node_ids: tuple[NodeId, ...] = ()
    lower_bound: float | None = None
    upper_bound: float | None = None
    unit: Unit | None = None
    requires_power_flow: bool = False

    def __post_init__(self) -> None:
        errors: list[str] = []

        if not isinstance(self.constraint_id, ConstraintId):
            errors.append(
                f"constraint_id must be a ConstraintId, got {type(self.constraint_id).__name__}"
            )
        if not isinstance(self.category, ConstraintCategory):
            errors.append(f"category must be a ConstraintCategory, got {type(self.category).__name__}")
        if not isinstance(self.description, str) or not self.description.strip():
            errors.append(f"description must be a non-empty string, got {self.description!r}")
        if not isinstance(self.provenance, Provenance):
            errors.append(f"provenance must be a Provenance, got {type(self.provenance).__name__}")

        for field_name, expected in (
            ("applies_to_asset_ids", AssetId),
            ("applies_to_node_ids", NodeId),
        ):
            value = getattr(self, field_name)
            if not isinstance(value, tuple):
                errors.append(f"{field_name} must be a tuple, got {type(value).__name__}")
            elif not all(isinstance(item, expected) for item in value):
                errors.append(
                    f"every entry in {field_name} must be an {expected.__name__}"
                )

        if self.lower_bound is not None and isinstance(self.lower_bound, (int, float)):
            if self.upper_bound is not None and self.lower_bound > self.upper_bound:
                errors.append(
                    f"lower_bound {self.lower_bound} exceeds upper_bound {self.upper_bound}"
                )

        expected_unit = CONSTRAINT_UNITS.get(self.category)
        if self.unit is None:
            # Defaulted rather than required: an absent unit means "the
            # category's natural unit", which keeps construction ergonomic
            # without losing the mapping.
            object.__setattr__(self, "unit", expected_unit)
        elif expected_unit is not None and self.unit != expected_unit:
            errors.append(
                f"{self.category.value} bounds must be expressed in "
                f"{expected_unit.value}, got {self.unit.value}"
            )

        if not self.applies_to_asset_ids and not self.applies_to_node_ids:
            errors.append(
                "a constraint must apply to at least one asset or node; an "
                "unscoped constraint cannot be checked against anything"
            )

        if errors:
            raise DomainValidationError(errors, subject="Constraint")

    @property
    def is_bounded(self) -> bool:
        return self.lower_bound is not None or self.upper_bound is not None

    @property
    def is_enforceable_in_phase_2(self) -> bool:
        """Whether Phase 2 tooling could evaluate this constraint.

        Always False for physics-dependent categories, and False for every
        category in practice: Phase 2 declares, Phase 10 evaluates. Exposed so
        that a caller can report this honestly rather than implying enforcement.
        """
        return False

    @property
    def needs_power_flow(self) -> bool:
        if self.requires_power_flow:
            return True
        return self.category in PHYSICS_DEPENDENT
