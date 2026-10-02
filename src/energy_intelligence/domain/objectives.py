"""Energy objectives.

Phase 2 defines the objective **categories** and, importantly, the conflicts
between them. It deliberately does **not** combine them into a weighted sum.

Why no weights: choosing weights is a policy decision that encodes a preference
between, say, cost and comfort. Making that choice in Phase 2 would either bake
in an arbitrary answer or hide the trade-off inside a scalar that nobody can
audit. Instead each objective declares its direction, its measurement and its
conflicts, and Phase 10 decides how to trade them off. See ``decisions.md`` (D-031).

Consequence: this module has no ``weight`` field, and no class represents a
scalarised objective. If a later phase needs one, adding it is a recorded
decision, not an incidental edit.

None of these are generic ML metrics. Every one is an energy-system outcome:
what was paid, what peaked, what was spilled, what degraded, what was shed.
"""

from __future__ import annotations

from dataclasses import dataclass

from .enums import ObjectiveCategory, ObjectiveDirection, Unit
from .errors import DomainValidationError
from .identifiers import ObjectiveId
from .provenance import Provenance

__all__ = ["Objective", "OBJECTIVE_METADATA", "ObjectiveMetadata", "objective_metadata"]

#: Static, non-negotiable metadata per category.
#:
#: Keeping direction and unit here rather than on each instance means two
#: objectives of the same category cannot disagree about what "better" means.
#: The conflict list is the substantive content: it is what stops Phase 10 from
#: treating the objective set as independent.
OBJECTIVE_METADATA: dict[ObjectiveCategory, "ObjectiveMetadata"] = {}


@dataclass(frozen=True, slots=True)
class ObjectiveMetadata:
    """Fixed facts about an objective category.

    Attributes:
        direction: Whether lower or higher is better.
        unit: Unit the objective is measured in.
        conflicts_with: Categories that compete with this one. Bidirectional.
    """

    direction: ObjectiveDirection
    unit: Unit
    conflicts_with: tuple[ObjectiveCategory, ...] = ()


OBJECTIVE_METADATA.update(
    {
        ObjectiveCategory.ENERGY_COST: ObjectiveMetadata(
            direction=ObjectiveDirection.MINIMIZE,
            unit=Unit.COUNT,  # currency placeholder; Phase 3 sets the real unit
            conflicts_with=(
                ObjectiveCategory.RENEWABLE_UTILIZATION,
                ObjectiveCategory.BATTERY_DEGRADATION,
            ),
        ),
        ObjectiveCategory.PEAK_DEMAND: ObjectiveMetadata(
            direction=ObjectiveDirection.MINIMIZE,
            unit=Unit.KILOWATT,
            conflicts_with=(
                ObjectiveCategory.ENERGY_COST,
                ObjectiveCategory.LOAD_SHED_AMOUNT,
            ),
        ),
        ObjectiveCategory.RENEWABLE_CURTAILMENT: ObjectiveMetadata(
            direction=ObjectiveDirection.MINIMIZE,
            unit=Unit.KILOWATT_HOURS,
            conflicts_with=(ObjectiveCategory.ENERGY_COST,),
        ),
        ObjectiveCategory.RENEWABLE_UTILIZATION: ObjectiveMetadata(
            direction=ObjectiveDirection.MAXIMIZE,
            unit=Unit.PERCENT,
            conflicts_with=(ObjectiveCategory.ENERGY_COST, ObjectiveCategory.PEAK_DEMAND),
        ),
        ObjectiveCategory.BATTERY_DEGRADATION: ObjectiveMetadata(
            direction=ObjectiveDirection.MINIMIZE,
            unit=Unit.COUNT,
            conflicts_with=(ObjectiveCategory.ENERGY_COST, ObjectiveCategory.RENEWABLE_UTILIZATION),
        ),
        ObjectiveCategory.CONSTRAINT_VIOLATION: ObjectiveMetadata(
            direction=ObjectiveDirection.MINIMIZE,
            unit=Unit.COUNT,
            conflicts_with=(),
        ),
        ObjectiveCategory.GRID_IMPORT_PEAK: ObjectiveMetadata(
            direction=ObjectiveDirection.MINIMIZE,
            unit=Unit.KILOWATT,
            conflicts_with=(ObjectiveCategory.ENERGY_COST,),
        ),
        ObjectiveCategory.LOAD_SHED_AMOUNT: ObjectiveMetadata(
            direction=ObjectiveDirection.MINIMIZE,
            unit=Unit.KILOWATT_HOURS,
            conflicts_with=(ObjectiveCategory.ENERGY_COST, ObjectiveCategory.RENEWABLE_UTILIZATION),
        ),
        ObjectiveCategory.UNMET_DEMAND: ObjectiveMetadata(
            direction=ObjectiveDirection.MINIMIZE,
            unit=Unit.KILOWATT_HOURS,
            conflicts_with=(),
        ),
        ObjectiveCategory.EMISSIONS: ObjectiveMetadata(
            direction=ObjectiveDirection.MINIMIZE,
            unit=Unit.COUNT,
            conflicts_with=(ObjectiveCategory.ENERGY_COST,),
        ),
    }
)


def objective_metadata(category: ObjectiveCategory) -> ObjectiveMetadata:
    """Return the fixed metadata for an objective category."""
    if not isinstance(category, ObjectiveCategory):
        raise DomainValidationError(
            [f"category must be an ObjectiveCategory, got {type(category).__name__}"],
            subject="objective_metadata",
        )
    return OBJECTIVE_METADATA[category]


@dataclass(frozen=True, slots=True)
class Objective:
    """One declared energy objective.

    Deliberately has **no** ``weight`` field. See the module docstring.

    Attributes:
        objective_id: Unique identifier.
        category: Which objective this is.
        measurement: How this objective is computed, in words. Left explicit
            because the measurement is not always obvious: "peak demand" could
            mean feeder peak, system peak, or coincident peak, and the three lead
            to different optimisations.
        why_it_matters: Plain-language justification.
        provenance: Why this objective was declared. Mandatory.
        target_value: Optional target, in ``unit``. A target is not a weight: it
            states a desired outcome without implying how hard to push for it.
    """

    objective_id: ObjectiveId
    category: ObjectiveCategory
    measurement: str
    why_it_matters: str
    provenance: Provenance
    target_value: float | None = None

    def __post_init__(self) -> None:
        errors: list[str] = []
        if not isinstance(self.objective_id, ObjectiveId):
            errors.append(
                f"objective_id must be an ObjectiveId, got {type(self.objective_id).__name__}"
            )
        if not isinstance(self.category, ObjectiveCategory):
            errors.append(f"category must be an ObjectiveCategory, got {type(self.category).__name__}")
        for field_name in ("measurement", "why_it_matters"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                errors.append(f"{field_name} must be a non-empty string, got {value!r}")
        if not isinstance(self.provenance, Provenance):
            errors.append(f"provenance must be a Provenance, got {type(self.provenance).__name__}")

        if self.target_value is not None:
            if isinstance(self.target_value, bool) or not isinstance(
                self.target_value, (int, float)
            ):
                errors.append(f"target_value must be a real number, got {self.target_value!r}")

        if errors:
            raise DomainValidationError(errors, subject="Objective")

    @property
    def direction(self) -> ObjectiveDirection:
        return objective_metadata(self.category).direction

    @property
    def unit(self) -> Unit:
        return objective_metadata(self.category).unit

    @property
    def conflicts_with(self) -> tuple[ObjectiveCategory, ...]:
        return objective_metadata(self.category).conflicts_with

    def conflicts(self, other: Objective) -> bool:
        """Whether this objective and ``other`` compete.

        Symmetric: a conflict declared in either direction counts, so adding an
        objective in a different order cannot change the answer.
        """
        if not isinstance(other, Objective):
            raise DomainValidationError(
                [f"other must be an Objective, got {type(other).__name__}"],
                subject="Objective.conflicts",
            )
        if other.category == self.category:
            return False
        return other.category in self.conflicts_with or self.category in other.conflicts_with
