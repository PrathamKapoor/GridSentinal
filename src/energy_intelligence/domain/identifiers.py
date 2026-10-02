"""Typed identifiers for domain entities.

Identifiers are validated on construction rather than being bare strings, so a
typo in an asset or node reference fails immediately instead of silently
creating an unresolvable reference during Phase 10 topology validation.

Each identifier kind has its own prefix and character class. Prefixes make
cross-entity reference mistakes (``NodeId("asset-7")``) impossible to write, and
are readable in logs and serialized JSON without extra labelling.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import ClassVar

from .errors import DomainValidationError

__all__ = [
    "Identifier",
    "AssetId",
    "NodeId",
    "NetworkElementId",
    "ConstraintId",
    "ObjectiveId",
    "ActionId",
    "ForecastId",
    "SystemId",
]

# {prefix}-{slug}; slug starts alphanumeric, then lowercase/digits/-/_ up to 63.
_ID_PATTERN = r"^{prefix}-[a-z0-9][a-z0-9_-]{{0,63}}$"


@dataclass(frozen=True, slots=True)
class Identifier:
    """Base class for a validated domain identifier.

    Subclasses override :attr:`PATTERN`. Instances are frozen, hashable and
    compare by value, so they can be used directly as dictionary keys when
    indexing assets by identifier.
    """

    PATTERN: ClassVar[re.Pattern[str]] = re.compile(r"^id-[a-z0-9][a-z0-9_-]{0,63}$")

    value: str

    def __post_init__(self) -> None:
        if not isinstance(self.value, str):
            raise DomainValidationError(
                [f"identifier must be a string, got {type(self.value).__name__}"],
                subject=type(self).__name__,
            )
        if not self.PATTERN.match(self.value):
            raise DomainValidationError(
                [
                    f"identifier {self.value!r} does not match "
                    f"{type(self).__name__}.PATTERN ({self.PATTERN.pattern})"
                ],
                subject=type(self).__name__,
            )

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True)
class AssetId(Identifier):
    """Identifies one energy asset (solar, battery, EV charger, load, ...)."""

    PATTERN: ClassVar[re.Pattern[str]] = re.compile(_ID_PATTERN.format(prefix="asset"))


@dataclass(frozen=True, slots=True)
class NodeId(Identifier):
    """Identifies one node in the network topology."""

    PATTERN: ClassVar[re.Pattern[str]] = re.compile(_ID_PATTERN.format(prefix="node"))


@dataclass(frozen=True, slots=True)
class NetworkElementId(Identifier):
    """Identifies one network element (line, transformer, switch, ...)."""

    PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        _ID_PATTERN.format(prefix="net")
    )


@dataclass(frozen=True, slots=True)
class ConstraintId(Identifier):
    """Identifies one declared constraint."""

    PATTERN: ClassVar[re.Pattern[str]] = re.compile(_ID_PATTERN.format(prefix="con"))


@dataclass(frozen=True, slots=True)
class ObjectiveId(Identifier):
    """Identifies one declared objective."""

    PATTERN: ClassVar[re.Pattern[str]] = re.compile(_ID_PATTERN.format(prefix="obj"))


@dataclass(frozen=True, slots=True)
class ActionId(Identifier):
    """Identifies one candidate action."""

    PATTERN: ClassVar[re.Pattern[str]] = re.compile(_ID_PATTERN.format(prefix="act"))


@dataclass(frozen=True, slots=True)
class ForecastId(Identifier):
    """Identifies one forecast issuance."""

    PATTERN: ClassVar[re.Pattern[str]] = re.compile(_ID_PATTERN.format(prefix="fc"))


@dataclass(frozen=True, slots=True)
class SystemId(Identifier):
    """Identifies the energy system as a whole."""

    PATTERN: ClassVar[re.Pattern[str]] = re.compile(_ID_PATTERN.format(prefix="sys"))
