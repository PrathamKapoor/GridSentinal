"""Physical quantities with explicit units.

A bare ``float`` cannot express *what* a number means. In an energy system that
is a correctness hazard rather than a style issue: 0.5 means half a megawatt or
half a state of charge depending on context, and a bug that confuses them is
silent. Every numeric value in the domain model is therefore a
:class:`Quantity` carrying its :class:`~energy_intelligence.domain.enums.Unit`.

This is deliberately strict: mixing units raises rather than silently
coercing. Conversion between units belongs to whichever phase first needs it
(the optimization phase), not to the data model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .enums import Unit
from .errors import DomainValidationError

__all__ = ["Quantity"]


@dataclass(frozen=True, slots=True)
class Quantity:
    """A finite, non-negative-or-signed numeric value with a unit.

    Attributes:
        value: The magnitude. May be negative for signed quantities such as
            net power flow or temperature.
        unit: The unit the magnitude is expressed in.
    """

    value: float
    unit: Unit

    def __post_init__(self) -> None:
        errors: list[str] = []
        if isinstance(self.value, bool) or not isinstance(self.value, (int, float)):
            errors.append(
                f"value must be a real number, got {type(self.value).__name__}: {self.value!r}"
            )
        elif math.isnan(self.value):
            errors.append("value must not be NaN")
        elif math.isinf(self.value):
            errors.append("value must be finite")

        if not isinstance(self.unit, Unit):
            errors.append(
                f"unit must be a Unit member, got {type(self.unit).__name__}: {self.unit!r}"
            )

        if errors:
            raise DomainValidationError(errors, subject="Quantity")

    @property
    def is_zero(self) -> bool:
        return self.value == 0.0

    def same_unit_as(self, other: Quantity) -> bool:
        """Return whether two quantities are directly comparable.

        No implicit conversion is performed, so this must be checked explicitly
        before comparing magnitudes.
        """
        return self.unit == other.unit

    def require_same_unit(self, other: Quantity, context: str = "") -> None:
        """Raise if two quantities are not in the same unit."""
        if not self.same_unit_as(other):
            suffix = f" ({context})" if context else ""
            raise DomainValidationError(
                [f"unit mismatch{suffix}: {self.unit.value} vs {other.unit.value}"],
                subject="Quantity",
            )

    def __str__(self) -> str:
        return f"{self.value:g} {self.unit.value}"
