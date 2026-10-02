"""Candidate actions.

The control half of the contract. Every action declares:

* its :class:`~energy_intelligence.domain.enums.VariableRole`, which is
  constrained to ``ACTION`` and cannot be an observation role;
* the :class:`~energy_intelligence.domain.enums.AuthorityLevel` under which it is
  issued, which must be sufficient for the target asset;
* a setpoint with a unit and a magnitude range that depends on the action type.

The authority check is the enforcement point for the chosen control-authority
model: an action may only target an asset whose authority permits it, so an
``ADVISORY_ONLY`` asset cannot be dispatched. ``validate_against_system`` performs
that cross-check against real assets; the constructor enforces everything that
can be known without the system.

No optimizer lives here. This module defines *what can be commanded*, not *what
should be commanded*. See ``decisions.md`` (D-028, D-035).
"""

from __future__ import annotations

from dataclasses import dataclass

from .enums import (
    CONTROLLABLE_AUTHORITY_LEVELS,
    ActionType,
    AuthorityLevel,
    Unit,
    VariableRole,
)
from .errors import DomainValidationError
from .identifiers import ActionId, AssetId
from .observations import assert_role_is_action
from .provenance import Provenance
from .quantities import Quantity

__all__ = [
    "Action",
    "ACTION_REQUIREMENTS",
    "required_authority_for",
    "unit_for_action",
    "sign_constraint_for",
]

#: Authority levels sufficient to perform each action type.
#:
#: This is the enforcement table for the tiered control-authority model. An
#: action type absent from this mapping cannot be authorised at all.
#:
#: These are *sets*, not a ranked order. An earlier draft encoded them as a
#: lattice (DISPATCHABLE > SCHEDULEABLE > CURTAILABLE), which was wrong: it
#: silently asserted that scheduling authority implies authority to curtail
#: generation. Curtailing a solar array is a direct output reduction, not a
#: shift in time, so it needs its own capability. Listing sufficiency explicitly
#: makes each implication a stated decision instead of a numerical accident.
ACTION_REQUIREMENTS: dict[ActionType, frozenset[AuthorityLevel]] = {
    ActionType.BATTERY_CHARGE: frozenset({AuthorityLevel.DISPATCHABLE}),
    ActionType.BATTERY_DISCHARGE: frozenset({AuthorityLevel.DISPATCHABLE}),
    ActionType.GRID_IMPORT_LIMIT: frozenset({AuthorityLevel.DISPATCHABLE}),
    ActionType.EV_CHARGING: frozenset(
        {AuthorityLevel.DISPATCHABLE, AuthorityLevel.SCHEDULEABLE}
    ),
    ActionType.EV_DEFER_CHARGING: frozenset(
        {AuthorityLevel.DISPATCHABLE, AuthorityLevel.SCHEDULEABLE}
    ),
    ActionType.FLEXIBLE_LOAD_SHIFT: frozenset(
        {AuthorityLevel.DISPATCHABLE, AuthorityLevel.SCHEDULEABLE}
    ),
    ActionType.HVAC_SETPOINT_ADJUSTMENT: frozenset(
        {AuthorityLevel.DISPATCHABLE, AuthorityLevel.SCHEDULEABLE}
    ),
    ActionType.DEMAND_RESPONSE: frozenset(
        {AuthorityLevel.DISPATCHABLE, AuthorityLevel.SCHEDULEABLE}
    ),
    ActionType.RENEWABLE_CURTAILMENT: frozenset(
        {AuthorityLevel.DISPATCHABLE, AuthorityLevel.CURTAILABLE}
    ),
}

#: Expected unit for each action type's setpoint.
_ACTION_UNITS: dict[ActionType, Unit] = {
    ActionType.BATTERY_CHARGE: Unit.KILOWATT,
    ActionType.BATTERY_DISCHARGE: Unit.KILOWATT,
    ActionType.EV_CHARGING: Unit.KILOWATT,
    ActionType.EV_DEFER_CHARGING: Unit.KILOWATT,
    ActionType.FLEXIBLE_LOAD_SHIFT: Unit.KILOWATT,
    ActionType.HVAC_SETPOINT_ADJUSTMENT: Unit.DEGREE_CELSIUS,
    ActionType.RENEWABLE_CURTAILMENT: Unit.KILOWATT,
    ActionType.DEMAND_RESPONSE: Unit.KILOWATT,
    ActionType.GRID_IMPORT_LIMIT: Unit.KILOWATT,
}

#: Actions whose setpoint is a non-negative magnitude.
_NON_NEGATIVE_ACTIONS: frozenset[ActionType] = frozenset(
    {
        ActionType.BATTERY_CHARGE,
        ActionType.BATTERY_DISCHARGE,
        ActionType.EV_CHARGING,
        ActionType.EV_DEFER_CHARGING,
        ActionType.RENEWABLE_CURTAILMENT,
        ActionType.DEMAND_RESPONSE,
    }
)

#: Actions whose setpoint may be signed, because "more" and "less" are both
#: meaningful (a load shift, an import limit, a temperature setpoint).
_SIGNED_ACTIONS: frozenset[ActionType] = frozenset(
    {
        ActionType.FLEXIBLE_LOAD_SHIFT,
        ActionType.HVAC_SETPOINT_ADJUSTMENT,
        ActionType.GRID_IMPORT_LIMIT,
    }
)


def sufficient_authorities_for(action_type: ActionType) -> frozenset[AuthorityLevel]:
    """Return every authority level under which ``action_type`` may be issued."""
    if not isinstance(action_type, ActionType):
        raise DomainValidationError(
            [f"action_type must be an ActionType, got {type(action_type).__name__}"],
            subject="sufficient_authorities_for",
        )
    return ACTION_REQUIREMENTS[action_type]


def required_authority_for(action_type: ActionType) -> AuthorityLevel:
    """Return the *narrowest* authority level sufficient for ``action_type``.

    Used in error messages to name the minimum the caller must obtain. Ordering
    within :data:`ACTION_REQUIREMENTS` is a declaration of narrowness, not an
    implication: see the note on that mapping.
    """
    sufficient = sufficient_authorities_for(action_type)
    if AuthorityLevel.DISPATCHABLE in sufficient:
        return AuthorityLevel.DISPATCHABLE
    if AuthorityLevel.SCHEDULEABLE in sufficient:
        return AuthorityLevel.SCHEDULEABLE
    return AuthorityLevel.CURTAILABLE


def unit_for_action(action_type: ActionType) -> Unit:
    """Return the unit an action's setpoint must be expressed in."""
    if not isinstance(action_type, ActionType):
        raise DomainValidationError(
            [f"action_type must be an ActionType, got {type(action_type).__name__}"],
            subject="unit_for_action",
        )
    return _ACTION_UNITS[action_type]


def sign_constraint_for(action_type: ActionType) -> str:
    """Return ``"non_negative"``, ``"signed"`` or ``"any"`` for an action type."""
    if action_type in _NON_NEGATIVE_ACTIONS:
        return "non_negative"
    if action_type in _SIGNED_ACTIONS:
        return "signed"
    return "any"


@dataclass(frozen=True, slots=True)
class Action:
    """One candidate control action against one asset.

    Attributes:
        action_id: Unique identifier.
        action_type: What is being commanded.
        target_asset_id: Asset being commanded.
        setpoint: Commanded magnitude, in the unit required by ``action_type``.
        role: Always ``ACTION``. Supplied explicitly so the action/observation
            boundary is visible on every instance rather than implied.
        issued_under: Authority level the issuing system holds over the target.
            Must be a controllable level.
        start_step: Grid step index at which the action takes effect, relative
            to the governing :class:`~energy_intelligence.domain.timebase.TimeBase`.
        duration_steps: How many grid steps the action holds for. At least one.
        provenance: Why this action exists. Mandatory.
        rationale: Human-readable justification.
    """

    action_id: ActionId
    action_type: ActionType
    target_asset_id: AssetId
    setpoint: Quantity
    role: VariableRole
    issued_under: AuthorityLevel
    start_step: int
    duration_steps: int
    provenance: Provenance
    rationale: str = ""

    def __post_init__(self) -> None:
        errors: list[str] = []

        if not isinstance(self.action_id, ActionId):
            errors.append(f"action_id must be an ActionId, got {type(self.action_id).__name__}")
        if not isinstance(self.action_type, ActionType):
            errors.append(f"action_type must be an ActionType, got {type(self.action_type).__name__}")
        if not isinstance(self.target_asset_id, AssetId):
            errors.append(
                f"target_asset_id must be an AssetId, got {type(self.target_asset_id).__name__}"
            )
        if not isinstance(self.setpoint, Quantity):
            errors.append(f"setpoint must be a Quantity, got {type(self.setpoint).__name__}")

        # The action/observation boundary, enforced here.
        try:
            assert_role_is_action(self.role)
        except DomainValidationError as exc:
            errors.extend(exc.errors)

        if not isinstance(self.issued_under, AuthorityLevel):
            errors.append(
                f"issued_under must be an AuthorityLevel, got {type(self.issued_under).__name__}"
            )
        elif self.issued_under not in CONTROLLABLE_AUTHORITY_LEVELS:
            errors.append(
                f"issued_under {self.issued_under.value!r} is not a controllable "
                f"authority level; an Action cannot be issued under "
                f"{sorted(level.value for level in CONTROLLABLE_AUTHORITY_LEVELS)}. "
                "Advisory output is not an Action."
            )

        if isinstance(self.start_step, bool) or not isinstance(self.start_step, int):
            errors.append(f"start_step must be an int, got {type(self.start_step).__name__}")
        if isinstance(self.duration_steps, bool) or not isinstance(self.duration_steps, int):
            errors.append(f"duration_steps must be an int, got {type(self.duration_steps).__name__}")
        elif self.duration_steps < 1:
            errors.append(f"duration_steps must be at least 1, got {self.duration_steps}")

        if not isinstance(self.provenance, Provenance):
            errors.append(f"provenance must be a Provenance, got {type(self.provenance).__name__}")

        if errors:
            raise DomainValidationError(errors, subject="Action")

        # Unit and sign checks are only meaningful once action_type and setpoint
        # are known well-formed, so they run after the structural pass.
        errors = []

        expected_unit = unit_for_action(self.action_type)
        if isinstance(self.setpoint, Quantity) and self.setpoint.unit != expected_unit:
            errors.append(
                f"{self.action_type.value} setpoint must be expressed in "
                f"{expected_unit.value}, got {self.setpoint.unit.value}"
            )

        constraint = sign_constraint_for(self.action_type)
        if isinstance(self.setpoint, Quantity):
            if constraint == "non_negative" and self.setpoint.value < 0:
                errors.append(
                    f"{self.action_type.value} setpoint must be non-negative, got "
                    f"{self.setpoint.value}"
                )
            elif constraint == "signed" and self.setpoint.value == 0:
                errors.append(
                    f"{self.action_type.value} setpoint must be non-zero; a zero "
                    "setpoint is the absence of an action, not an action"
                )

        if errors:
            raise DomainValidationError(errors, subject="Action")

    @property
    def end_step(self) -> int:
        """Last grid step the action applies to, inclusive of ``start_step``."""
        return self.start_step + self.duration_steps - 1

    @property
    def is_executable(self) -> bool:
        """Whether this action may be carried out.

        Phase 2 can only answer the authority half of this question. Full
        assurance also requires forecast confidence, constraint margins and
        adversarial results, none of which exist yet -- that is Phase 13.
        """
        return self.issued_under in CONTROLLABLE_AUTHORITY_LEVELS

    def covers_step(self, step: int) -> bool:
        return self.start_step <= step <= self.end_step

    def validate_against_system(self, assets) -> None:
        """Check this action against a system's assets.

        Args:
            assets: Iterable of :class:`~energy_intelligence.domain.assets.Asset`.

        Raises:
            DomainValidationError: If the target asset is unknown, or the
                issuing authority is insufficient for the target asset's declared
                authority.
        """
        from .assets import Asset

        errors: list[str] = []
        by_id = {asset.asset_id: asset for asset in assets}

        target = by_id.get(self.target_asset_id)
        if target is None:
            known = ", ".join(sorted(str(key) for key in by_id)) or "<none>"
            errors.append(
                f"action {self.action_id} targets unknown asset {self.target_asset_id}; "
                f"known assets: {known}"
            )
        else:
            if not isinstance(target, Asset):  # pragma: no cover - defensive
                errors.append(f"target {self.target_asset_id} is not an Asset")
            else:
                if not target.is_controllable:
                    errors.append(
                        f"asset {target.asset_id} has authority "
                        f"{target.authority.value!r} and cannot be dispatched"
                    )
                elif not self.has_sufficient_authority():
                    required = required_authority_for(self.action_type)
                    errors.append(
                        f"action {self.action_id} is issued under "
                        f"{self.issued_under.value!r} but {self.action_type.value} "
                        f"requires one of "
                        f"{sorted(level.value for level in sufficient_authorities_for(self.action_type))} "
                        f"(narrowest: {required.value!r}); target asset "
                        f"{target.asset_id} has authority {target.authority.value!r}"
                    )

        if errors:
            raise DomainValidationError(errors, subject=f"Action[{self.action_id}]")

    def has_sufficient_authority(self) -> bool:
        """Whether the issuing authority is sufficient for this action type.

        Membership test against the declared sufficiency set. There is
        deliberately no ordering comparison on :class:`AuthorityLevel`: as a
        ``StrEnum`` its members compare alphabetically, so ``>=`` would encode
        "curtailable" as more powerful than "scheduleable", which is meaningless.
        """
        return self.issued_under in sufficient_authorities_for(self.action_type)
