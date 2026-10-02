"""Observations and the action/observation boundary.

This module enforces the Phase 2 requirement that a variable cannot accidentally
be both an uncontrolled observation and a control action. The mechanism is
:class:`~energy_intelligence.domain.enums.VariableRole`:

* :class:`ObservationRecord` accepts only ``OBSERVATION`` and ``DERIVED`` roles.
* :class:`energy_intelligence.domain.actions.Action` accepts only the ``ACTION``
  role.

An :class:`ObservationRecord` carrying ``VariableRole.ACTION`` is therefore
rejected at construction, so the mix-up cannot reach a later phase. See
``decisions.md`` (D-034).

The concrete variable catalogue is in ``docs/energy_system_spec.md``. This module
is deliberately *generic*: it does not enumerate variables, because the set of
variables a future system tracks must remain data-driven rather than closed at
Phase 2.
"""

from __future__ import annotations

from dataclasses import dataclass

from .enums import QualityFlag, VariableRole
from .errors import DomainValidationError
from .identifiers import AssetId, NodeId
from .provenance import Provenance
from .quality import DataQuality
from .quantities import Quantity

__all__ = [
    "ObservationRecord",
    "ALLOWED_OBSERVATION_ROLES",
    "OBSERVATION_ONLY_ROLES",
]

#: Roles an observation may declare. Excludes ACTION, CONSTRAINT and OBJECTIVE.
ALLOWED_OBSERVATION_ROLES: frozenset[VariableRole] = frozenset(
    {VariableRole.OBSERVATION, VariableRole.DERIVED}
)

#: Roles that describe something the system reads rather than sets.
OBSERVATION_ONLY_ROLES: frozenset[VariableRole] = frozenset(
    {VariableRole.OBSERVATION, VariableRole.DERIVED}
)


@dataclass(frozen=True, slots=True)
class ObservationRecord:
    """One measured or inferred quantity at one instant.

    Attributes:
        variable: Name of the variable, e.g. ``"battery_state_of_charge"``.
        role: Must be ``OBSERVATION`` or ``DERIVED``.
        value: The magnitude and its unit.
        quality: Data-quality metadata. Mandatory.
        provenance: Where the value came from. Mandatory and non-optional, so it
            cannot silently disappear.
        timestamp: ISO-8601 instant the value describes.
        asset_id: Asset the value belongs to, if it is asset-scoped. Typed, not
            a bare string, so it cannot silently fail to resolve against
            :class:`~energy_intelligence.domain.assets.Asset`.
        node_id: Node the value belongs to, if it is node-scoped.
    """

    variable: str
    role: VariableRole
    value: Quantity
    quality: DataQuality
    provenance: Provenance
    timestamp: str
    asset_id: AssetId | None = None
    node_id: NodeId | None = None

    def __post_init__(self) -> None:
        errors: list[str] = []

        if not isinstance(self.variable, str) or not self.variable.strip():
            errors.append(f"variable must be a non-empty string, got {self.variable!r}")
        elif not self.variable.replace("_", "").isalnum():
            # Guards against typos and stray punctuation entering the catalogue.
            errors.append(
                f"variable {self.variable!r} must match [a-z0-9_]+ (lowercase, "
                "digits and underscores only)"
            )

        if not isinstance(self.role, VariableRole):
            errors.append(f"role must be a VariableRole, got {type(self.role).__name__}")
        elif self.role not in ALLOWED_OBSERVATION_ROLES:
            errors.append(
                f"role {self.role.value!r} is not valid on an ObservationRecord; "
                f"observations carry only {sorted(role.value for role in ALLOWED_OBSERVATION_ROLES)}. "
                "An action variable belongs in actions.Action."
            )

        if not isinstance(self.value, Quantity):
            errors.append(f"value must be a Quantity, got {type(self.value).__name__}")
        if not isinstance(self.quality, DataQuality):
            errors.append(f"quality must be a DataQuality, got {type(self.quality).__name__}")
        if not isinstance(self.provenance, Provenance):
            errors.append(f"provenance must be a Provenance, got {type(self.provenance).__name__}")
        if not isinstance(self.timestamp, str) or not self.timestamp.strip():
            errors.append(f"timestamp must be a non-empty ISO-8601 string, got {self.timestamp!r}")

        for field_name, expected in (("asset_id", AssetId), ("node_id", NodeId)):
            value = getattr(self, field_name)
            if value is not None and not isinstance(value, expected):
                errors.append(
                    f"{field_name} must be None or an {expected.__name__}, "
                    f"got {type(value).__name__}: {value!r}"
                )

        if self.asset_id is None and self.node_id is None:
            errors.append(
                "an observation must be scoped to an asset_id, a node_id, or both; "
                "an unscoped observation cannot be attributed to part of the system"
            )

        if errors:
            raise DomainValidationError(errors, subject="ObservationRecord")

    @property
    def is_actionable_input(self) -> bool:
        """Whether a decision may rely on this observation.

        Delegates to :attr:`DataQuality.is_usable_for_decision` so that the
        quality policy has exactly one definition and cannot drift between
        modules.
        """
        return self.quality.is_usable_for_decision

    @property
    def is_missing(self) -> bool:
        return self.quality.has_flag(QualityFlag.MISSING)


def assert_role_is_action(role: VariableRole) -> None:
    """Raise unless ``role`` is ``ACTION``.

    Used by ``actions.Action`` so the boundary rule is stated once.
    """
    if not isinstance(role, VariableRole):
        raise DomainValidationError(
            [f"role must be a VariableRole, got {type(role).__name__}"], subject="Action"
        )
    if role is not VariableRole.ACTION:
        raise DomainValidationError(
            [
                f"role {role.value!r} is not valid on an Action; actions carry only "
                f"{VariableRole.ACTION.value!r}. An observed variable belongs in "
                "ObservationRecord."
            ],
            subject="Action",
        )
