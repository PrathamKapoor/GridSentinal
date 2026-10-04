"""Flexibility representation: the contract Phase 9 and every later phase depend on.

What is decided here
    A typed statement of **how much response capability can reasonably be associated
    with part of the system, in which direction, on what evidence, with how much
    uncertainty and for how long**. :class:`FlexibilityEstimate` is that statement.

What this module deliberately does not do
    It does not compute anything, does not import :mod:`energy_intelligence.ml`, and
    does not know what a battery is. Phase 9's estimators live in
    :mod:`energy_intelligence.ml.flexibility` and produce these objects; Phase 10's
    optimizer consumes them. Neither direction is visible from here.

The distinction this contract exists to enforce
    ------------------------------------------------------

    A system that has observed a load move by 4 kW has learned something about
    **behaviour**. It has not learned that it can *command* 4 kW of reduction, and
    on most utility data it never will: no dispatcher, no setpoint, no contract and
    no measurement of response exists. Those are two different claims and confusing
    them is the specific failure this module makes structurally impossible.

    The enforcement is in :meth:`FlexibilityEstimate.__post_init__`, not in prose:

    * ``basis=PHYSICAL`` requires an authority the system actually holds. A dataset
      that states only a nameplate rating cannot support a physical claim.
    * ``basis=STATISTICAL_PROXY`` is **refused** a controllable authority, in every
      mode including scenario mode. A behavioural statistic is not a control
      contract, and the moment a proxy is allowed to say ``DISPATCHABLE`` the
      distinction is gone.
    * ``basis=UNKNOWN`` must carry **no magnitude**. An unknown with a number is a
      fabricated capability wearing an honest label.
    * A controllable authority requires ``PHYSICAL`` or ``ASSUMED`` evidence - never
      a proxy and never an unknown.

    Together these mean a downstream consumer can branch on
    :attr:`FlexibilityEstimate.is_dispatchable` and be right.

Sign conventions, stated because three of them exist in this project
    ----------------------------------------------------------

    :class:`~energy_intelligence.domain.assets.FlexibleLoad` and
    :class:`~energy_intelligence.domain.actions.Action` already commit to a
    convention and this module follows it rather than inventing a fourth:

    * **Here**: magnitude is a non-negative count of kW, and
      :class:`FlexibilityDirection` says which way net demand would move. Positive
      magnitude, explicit direction, no sign on the number.
    * :class:`ActionType`.FLEXIBLE_LOAD_SHIFT` carries a **signed load** setpoint,
      so :meth:`FlexibilityEstimate.as_signed_setpoint` returns ``-magnitude`` for
      ``DOWNWARD`` and ``+magnitude`` for ``UPWARD``.
    * :class:`~energy_intelligence.domain.state.NodePower` takes **load as negative**,
      so :meth:`FlexibilityEstimate.as_node_power_delta` returns the opposite sign
      again.

    Getting this wrong is not a cosmetic error: it would make a load-reduction
    action read as a load increase at the optimiser.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


from .enums import (
    CONTROLLABLE_AUTHORITY_LEVELS,
    AuthorityLevel,
    FlexibilityBasis,
    FlexibilityDirection,
    Unit,
    VariableRole,
)
from .errors import DomainValidationError
from .identifiers import AssetId, NodeId
from .provenance import Provenance
from .uncertainty import UncertaintyEstimate

__all__ = [
    "AggregationLevel",
    "FlexibilityEstimate",
    "FlexibilityEnvelope",
    "FLEXIBILITY_SCHEMA_VERSION",
]

#: Version of the flexibility payload shape. Separate from the domain-wide
#: ``serialization.SCHEMA_VERSION`` on purpose: that contract is strict equality and
#: is asserted by a Phase 3 fidelity test, so widening it for a Phase 9 type would
#: invalidate payloads Phase 3 already wrote. Flexibility carries its own version and
#: its own round-trip, exactly as Phase 8's ``ProbabilisticForecastBatch`` does.
FLEXIBILITY_SCHEMA_VERSION = "1.0.0-phase9"


class AggregationLevel(StrEnum):
    """What a flexibility figure is an aggregate of.

    Declared as its own axis rather than inferred from the target identifier, because
    the difference is the one a later optimiser must not get wrong: a group-level
    figure is *not* the sum of its members' capabilities. See
    :class:`~energy_intelligence.ml.flexibility.aggregation` for the measured
    diversification this axis exists to keep honest.
    """

    ASSET = "asset"
    GROUP = "group"
    FEEDER = "feeder"
    SYSTEM = "system"


def _as_float_grid(value: Any, name: str) -> tuple[tuple[int, ...], list[float]]:
    """Read a 1- or 2-dimensional numeric grid without importing a numeric library.

    The domain layer carries this project's hard rule that it depends on no ML library -
    ``tests/test_package.py`` enforces it, and rightly so: a domain object that needs numpy
    to validate itself cannot be used by the plain-Python consumers the rule exists to
    protect. Magnitudes arrive here as nested sequences (lists, tuples, or any array-like a
    caller cares to pass), so they are read structurally rather than coerced.

    Args:
        value: A sequence of numbers, or of equal-length sequences of numbers.
        name: Field name, for the error message.

    Returns:
        ``(shape, flat)`` where ``shape`` is ``(rows,)`` or ``(rows, columns)`` and ``flat``
        is row-major.

    Raises:
        TypeError: If the value is not a numeric grid, or is ragged.
    """
    try:
        rows = list(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be a sequence of numbers, got {type(value).__name__}") from exc

    flat: list[float] = []
    if not rows:
        return (0,), flat

    first = rows[0]
    if isinstance(first, (list, tuple)) or (
        hasattr(first, "__len__") and not isinstance(first, (str, bytes))
    ):
        width: int | None = None
        for index, row in enumerate(rows):
            try:
                cells = list(row)
            except TypeError as exc:
                raise TypeError(
                    f"{name} row {index} must be a sequence of numbers, got "
                    f"{type(row).__name__}"
                ) from exc
            if width is None:
                width = len(cells)
            elif len(cells) != width:
                raise TypeError(
                    f"{name} is ragged: row 0 has {width} entries and row {index} has "
                    f"{len(cells)}"
                )
            for cell in cells:
                flat.append(_as_finite_float(cell, name))
        return (len(rows), width or 0), flat

    for index, cell in enumerate(rows):
        flat.append(_as_finite_float(cell, name))
    return (len(rows),), flat


def _as_finite_float(value: Any, name: str) -> float:
    """One cell as a finite float.

    Raises:
        TypeError: If the cell is not a real number, or is not finite.
    """
    if isinstance(value, bool):
        raise TypeError(f"{name} contains a boolean; magnitudes must be real numbers")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"{name} contains {value!r}, which is not a real number"
        ) from exc
    if number != number or number in (float("inf"), float("-inf")):
        raise TypeError(f"{name} must be finite, got {value!r}")
    return number

@dataclass(frozen=True, slots=True)
class FlexibilityEstimate:
    """One flexibility statement about part of the system.

    Attributes:
        target: What the claim is about. An :class:`AssetId` for a single asset; a
            free-form scope string (``"group-40-customers"``) for an aggregate.
        direction: Which way net demand would move.
        magnitude_kw: Non-negative magnitude in kW, or ``None`` when ``basis`` is
            ``UNKNOWN``.
        basis: What kind of evidence supports the number. This is the field that
            decides whether it may be acted on.
        authority: How much control the system holds over the target.
            :attr:`is_dispatchable` is derived from it and ``basis`` together.
        confidence: Probability-like score in ``[0, 1]``, or ``None`` meaning *not
            characterised*. ``None`` is a legal value rather than a default of 1.0,
            because "we did not measure confidence" and "we are certain" must not
            look the same.
        uncertainty: Optional :class:`UncertaintyEstimate`. Carried rather than
            flattened into a scalar so the interval's own method name survives.
        timestamp: ISO-8601 instant the estimate is valid **from**.
        horizon_steps: Forecast horizon this applies to, or ``None`` for a
            horizon-independent statement.
        validity_steps: How many grid steps the estimate holds for. ``None`` means
            valid only at ``timestamp``.
        aggregation: Whether this is a single asset or an aggregate.
        role: Always :attr:`VariableRole`.DERIVED - flexibility is the domain's own
            worked example of a derived variable.
        provenance: Mandatory. A flexibility claim without provenance is a guess
            with a number attached.
        method: How it was computed, or the reason it is unknown.
        limitations: Explicit caveats, carried rather than left in a document. A
            consumer reading only this object still learns what it is not.
        asset_id: Typed asset reference when the target is a single asset.
        node_id: Typed node reference when one applies.
    """

    target: str
    direction: FlexibilityDirection
    basis: FlexibilityBasis
    authority: AuthorityLevel
    magnitude_kw: float | None
    unit: Unit
    timestamp: str
    provenance: Provenance
    method: str
    confidence: float | None = None
    uncertainty: UncertaintyEstimate | None = None
    horizon_steps: int | None = None
    validity_steps: int | None = None
    aggregation: AggregationLevel = AggregationLevel.ASSET
    role: VariableRole = VariableRole.DERIVED
    limitations: tuple[str, ...] = ()
    asset_id: AssetId | None = None
    node_id: NodeId | None = None

    def __post_init__(self) -> None:
        errors: list[str] = []

        if not isinstance(self.target, str) or not self.target.strip():
            errors.append(f"target must be a non-empty string, got {self.target!r}")
        elif not self.target.replace("-", "").replace("_", "").isalnum():
            errors.append(
                f"target {self.target!r} must match [a-z0-9_-]+ so it can be referenced "
                f"by an identifier-shaped name"
            )
        for name, expected in (
            ("direction", FlexibilityDirection),
            ("basis", FlexibilityBasis),
            ("authority", AuthorityLevel),
            ("unit", Unit),
            ("aggregation", AggregationLevel),
            ("role", VariableRole),
        ):
            value = getattr(self, name)
            if not isinstance(value, expected):
                errors.append(f"{name} must be a {expected.__name__}, got {type(value).__name__}")
        if isinstance(self.role, VariableRole) and self.role is not VariableRole.DERIVED:
            errors.append(
                f"a flexibility estimate is a derived variable, so role must be "
                f"{VariableRole.DERIVED.value!r}, got "
                f"{getattr(self.role, 'value', self.role)!r}"
            )
        if isinstance(self.unit, Unit) and self.unit is not Unit.KILOWATT:
            errors.append(
                f"flexibility magnitude is a power, so unit must be {Unit.KILOWATT.value!r} "
                f"to match ConstraintCategory.DEMAND_FLEXIBILITY_LIMIT; got "
                f"{self.unit.value!r}"
            )
        if not isinstance(self.authority, AuthorityLevel):
            pass  # already reported above
        elif self.authority in CONTROLLABLE_AUTHORITY_LEVELS:
            if self.basis is FlexibilityBasis.STATISTICAL_PROXY:
                errors.append(
                    "a statistical proxy may not claim a controllable authority. The "
                    "figure describes how a load has behaved, not what an operator can "
                    "command; allowing DISPATCHABLE here is the precise confusion this "
                    "contract exists to prevent. Use basis=PHYSICAL with real capability "
                    "metadata, or basis=ASSUMED inside an explicit scenario."
                )
            if self.basis is FlexibilityBasis.UNKNOWN:
                errors.append(
                    "an unknown cannot be paired with a controllable authority; the two "
                    "statements contradict each other"
                )
        if isinstance(self.basis, FlexibilityBasis) and self.basis is FlexibilityBasis.UNKNOWN and (
            self.magnitude_kw is not None
        ):
            errors.append(
                f"basis=UNKNOWN must carry no magnitude, got {self.magnitude_kw!r}. An "
                f"unknown with a number is a fabricated capability wearing an honest label"
            )
        if self.magnitude_kw is not None:
            if isinstance(self.magnitude_kw, bool) or not isinstance(
                self.magnitude_kw, (int, float)
            ):
                errors.append(
                    f"magnitude_kw must be a real number or None, got {self.magnitude_kw!r}"
                )
            else:
                if self.magnitude_kw != self.magnitude_kw:
                    errors.append("magnitude_kw must not be NaN")
                if self.magnitude_kw < 0:
                    errors.append(
                        f"magnitude_kw must be non-negative; direction carries the sign, "
                        f"got {self.magnitude_kw}"
                    )
        if self.confidence is not None:
            if isinstance(self.confidence, bool) or not isinstance(self.confidence, (int, float)):
                errors.append(
                    f"confidence must be a real number in [0, 1] or None, got {self.confidence!r}"
                )
            elif not 0.0 <= self.confidence <= 1.0:
                errors.append(f"confidence must lie in [0, 1], got {self.confidence}")
        if isinstance(self.timestamp, str) and not self.timestamp.strip():
            errors.append("timestamp must be a non-empty ISO-8601 string")
        if not isinstance(self.provenance, Provenance):
            errors.append(
                f"provenance must be a Provenance, got {type(self.provenance).__name__}; a "
                f"flexibility claim without provenance is a guess with a number attached"
            )
        if not isinstance(self.method, str) or not self.method.strip():
            errors.append(f"method must be a non-empty string, got {self.method!r}")
        for name in ("horizon_steps", "validity_steps"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int)):
                errors.append(f"{name} must be an int or None, got {value!r}")
            elif isinstance(value, int) and value <= 0:
                errors.append(f"{name} must be positive when set, got {value}")
        if isinstance(self.uncertainty, UncertaintyEstimate) is False and (
            self.uncertainty is not None
        ):
            errors.append(
                f"uncertainty must be an UncertaintyEstimate or None, got "
                f"{type(self.uncertainty).__name__}"
            )
        if not isinstance(self.limitations, tuple) or not all(
            isinstance(item, str) and item.strip() for item in self.limitations
        ):
            errors.append("limitations must be a tuple of non-empty strings")
        for name, expected in (("asset_id", AssetId), ("node_id", NodeId)):
            value = getattr(self, name)
            if value is not None and not isinstance(value, expected):
                errors.append(
                    f"{name} must be None or an {expected.__name__}, got "
                    f"{type(value).__name__}"
                )
        if isinstance(self.aggregation, AggregationLevel) and (
            self.aggregation is not AggregationLevel.ASSET
        ) and self.asset_id is not None:
            errors.append(
                f"an aggregate (aggregation={self.aggregation.value!r}) must not also name "
                f"a single asset_id; the two describe different scopes"
            )

        if errors:
            raise DomainValidationError(errors, subject="FlexibilityEstimate")

    # -- derived properties ------------------------------------------------

    @property
    def is_dispatchable(self) -> bool:
        """Whether a decision may act on this figure as a capability.

        Requires **both** a controllable authority and evidence that is not a proxy and
        not an unknown. Every statistical or unknown figure answers ``False`` here, which
        is the property Phase 10 will branch on.
        """
        return (
            self.authority in CONTROLLABLE_AUTHORITY_LEVELS
            and self.basis in (FlexibilityBasis.PHYSICAL, FlexibilityBasis.ASSUMED)
        )

    @property
    def is_quantified(self) -> bool:
        """Whether a magnitude is attached at all."""
        return self.magnitude_kw is not None

    @property
    def is_observational(self) -> bool:
        """Whether the figure describes observed behaviour rather than capability."""
        return self.basis is FlexibilityBasis.STATISTICAL_PROXY

    @property
    def basis_qualifier(self) -> str:
        """Short human label used by the CLI and reports so the basis is never implicit."""
        return {
            FlexibilityBasis.PHYSICAL: "PHYSICAL",
            FlexibilityBasis.STATISTICAL_PROXY: "STATISTICAL_PROXY",
            FlexibilityBasis.ASSUMED: "ASSUMED",
            FlexibilityBasis.UNKNOWN: "UNKNOWN",
        }[self.basis]

    def as_signed_setpoint(self) -> float | None:
        """The magnitude as an :class:`ActionType`.FLEXIBLE_LOAD_SHIFT signed setpoint.

        That action is signed in *load*, so ``DOWNWARD`` (shed power) is **negative**
        there while its magnitude here stays positive. ``None`` when unquantified.
        """
        if self.magnitude_kw is None:
            return None
        return -self.magnitude_kw if self.direction is FlexibilityDirection.DOWNWARD else self.magnitude_kw

    def as_node_power_delta(self) -> float | None:
        """The magnitude as a :class:`NodePower` delta, whose load is negative.

        ``DOWNWARD`` therefore becomes **positive** here: shedding load moves a node's
        net power toward zero, which is an increase in that convention. The third sign
        the project uses, and the one most easily got wrong.
        """
        if self.magnitude_kw is None:
            return None
        return self.magnitude_kw if self.direction is FlexibilityDirection.DOWNWARD else -self.magnitude_kw

    # -- serialisation -----------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """A JSON-ready dictionary carrying every declared field."""
        return {
            "_flexibility_schema_version": FLEXIBILITY_SCHEMA_VERSION,
            "target": self.target,
            "direction": self.direction.value,
            "basis": self.basis.value,
            "authority": self.authority.value,
            "magnitude_kw": None if self.magnitude_kw is None else float(self.magnitude_kw),
            "unit": self.unit.value,
            "timestamp": self.timestamp,
            "method": self.method,
            "confidence": None if self.confidence is None else float(self.confidence),
            "uncertainty": (
                None
                if self.uncertainty is None
                else {
                    "kind": self.uncertainty.kind.value,
                    "unit": self.uncertainty.unit.value,
                    "method": self.uncertainty.method,
                    "lower_bound": self.uncertainty.lower_bound,
                    "upper_bound": self.uncertainty.upper_bound,
                    "standard_deviation": self.uncertainty.standard_deviation,
                    "relative_std": self.uncertainty.relative_std,
                    "notes": self.uncertainty.notes,
                }
            ),
            "horizon_steps": self.horizon_steps,
            "validity_steps": self.validity_steps,
            "aggregation": self.aggregation.value,
            "role": self.role.value,
            "limitations": list(self.limitations),
            "asset_id": None if self.asset_id is None else self.asset_id.value,
            "node_id": None if self.node_id is None else self.node_id.value,
            "is_dispatchable": self.is_dispatchable,
            "provenance": {
                "source": {
                    "source_id": self.provenance.source.source_id,
                    "dataset": self.provenance.source.dataset,
                    "locator": self.provenance.source.locator,
                    "version": self.provenance.source.version,
                    "checksum": self.provenance.source.checksum,
                },
                "events": [
                    {"timestamp": e.timestamp, "event": e.event, "detail": e.detail}
                    for e in self.provenance.events
                ],
                "processing": [
                    {"name": s.name, "version": s.version, "detail": s.detail}
                    for s in self.provenance.processing
                ],
            },
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FlexibilityEstimate":
        """Rebuild an estimate from :meth:`to_dict`.

        Raises:
            DomainValidationError: If the schema version is wrong or a field is malformed.
        """
        from .provenance import ProcessingStep, ProvenanceEvent, SourceReference

        version = payload.get("_flexibility_schema_version")
        if version != FLEXIBILITY_SCHEMA_VERSION:
            raise DomainValidationError(
                [
                    f"flexibility payload schema version {version!r} does not match "
                    f"{FLEXIBILITY_SCHEMA_VERSION!r}"
                ],
                subject="FlexibilityEstimate",
            )
        raw_uncertainty = payload.get("uncertainty")
        uncertainty = (
            None
            if raw_uncertainty is None
            else UncertaintyEstimate(
                kind=UncertaintyKind(raw_uncertainty["kind"]),
                unit=Unit(raw_uncertainty["unit"]),
                method=raw_uncertainty["method"],
                lower_bound=raw_uncertainty.get("lower_bound"),
                upper_bound=raw_uncertainty.get("upper_bound"),
                standard_deviation=raw_uncertainty.get("standard_deviation"),
                relative_std=raw_uncertainty.get("relative_std"),
                notes=raw_uncertainty.get("notes", ""),
            )
        )
        raw_provenance = payload["provenance"]
        source = raw_provenance["source"]
        return cls(
            target=payload["target"],
            direction=FlexibilityDirection(payload["direction"]),
            basis=FlexibilityBasis(payload["basis"]),
            authority=AuthorityLevel(payload["authority"]),
            magnitude_kw=payload.get("magnitude_kw"),
            unit=Unit(payload["unit"]),
            timestamp=payload["timestamp"],
            method=payload["method"],
            confidence=payload.get("confidence"),
            uncertainty=uncertainty,
            horizon_steps=payload.get("horizon_steps"),
            validity_steps=payload.get("validity_steps"),
            aggregation=AggregationLevel(payload["aggregation"]),
            role=VariableRole(payload["role"]),
            limitations=tuple(payload.get("limitations", ())),
            asset_id=None if payload.get("asset_id") is None else AssetId(payload["asset_id"]),
            node_id=None if payload.get("node_id") is None else NodeId(payload["node_id"]),
            provenance=Provenance(
                source=SourceReference(
                    source_id=source["source_id"],
                    dataset=source["dataset"],
                    locator=source["locator"],
                    version=source["version"],
                    checksum=source.get("checksum"),
                ),
                events=tuple(
                    ProvenanceEvent(
                        timestamp=e["timestamp"], event=e["event"], detail=e.get("detail", "")
                    )
                    for e in raw_provenance.get("events", ())
                ),
                processing=tuple(
                    ProcessingStep(
                        name=s["name"], version=s["version"], detail=s.get("detail", "")
                    )
                    for s in raw_provenance.get("processing", ())
                ),
            ),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    def render(self) -> str:
        """The conceptual output, as text, with the basis stated before the number."""
        magnitude = (
            "not quantified" if self.magnitude_kw is None else f"{self.magnitude_kw:.4f} kW"
        )
        dispatch = "DISPATCHABLE" if self.is_dispatchable else "NOT DISPATCHABLE"
        lines = [
            f"Target        : {self.target} ({self.aggregation.value})",
            f"Direction     : {self.direction.value} (net demand "
            f"{'decreases' if self.direction is FlexibilityDirection.DOWNWARD else 'increases'})",
            f"Magnitude     : {magnitude}",
            f"Basis         : {self.basis_qualifier}",
            f"Authority     : {self.authority.value}",
            f"Status        : {dispatch}",
            f"Method        : {self.method}",
            f"Timestamp     : {self.timestamp}",
        ]
        if self.confidence is not None:
            lines.append(f"Confidence    : {self.confidence:.3f}")
        if self.uncertainty is not None and self.uncertainty.is_quantified:
            lines.append(f"Uncertainty   : {self.uncertainty}")
        for limitation in self.limitations:
            lines.append(f"Limitation    : {limitation}")
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class FlexibilityEnvelope:
    """A directional pair for one target and horizon, as a block rather than per row.

    Phase 9 estimates an envelope for many rows at once. Shipping one
    :class:`FlexibilityEstimate` per row would mean 179,520 objects per horizon with
    identical provenance, and - worse - it would make it easy to publish the block while
    losing track of which rows it covers. This holds the arrays plus the block-level
    metadata, and :meth:`at` produces the per-row contract object on demand.

    Attributes:
        target: What the claim is about.
        aggregation: Scope of the block.
        basis: Evidence class, shared by every row in the block.
        authority: Control authority, shared by every row.
        method: How the block was computed.
        nominal_level: Stated coverage of the envelope, when it is probabilistic.
        upward_kw: ``[rows]`` or ``[rows, horizons]`` upward magnitudes, kW.
        downward_kw: Matched downward magnitudes, kW, non-negative.
        reliability: ``[rows]`` or ``[rows, horizons]`` reliance factors in ``[0, 1]``,
            or ``None`` when no uncertainty conditioning was applied.
        timestamps: ISO-8601 instant each row is valid from.
        provenance: Mandatory.
        limitations: Block-level caveats.
    """

    target: str
    aggregation: AggregationLevel
    basis: FlexibilityBasis
    authority: AuthorityLevel
    method: str
    upward_kw: Any
    downward_kw: Any
    provenance: Provenance
    timestamps: tuple[str, ...] = ()
    nominal_level: float | None = None
    reliability: Any = None
    horizons: tuple[int, ...] = ()
    limitations: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        errors: list[str] = []
        try:
            up_shape, up_flat = _as_float_grid(self.upward_kw, "upward_kw")
            down_shape, down_flat = _as_float_grid(self.downward_kw, "downward_kw")
        except TypeError as exc:
            raise DomainValidationError([str(exc)], subject="FlexibilityEnvelope") from exc
        if up_shape != down_shape:
            errors.append(
                f"upward_kw {up_shape} and downward_kw {down_shape} must have the same shape"
            )
        elif any(value < 0 for value in up_flat) or any(value < 0 for value in down_flat):
            errors.append("magnitudes must be non-negative; direction carries the sign")
        if self.reliability is not None:
            try:
                rel_shape, rel_flat = _as_float_grid(self.reliability, "reliability")
            except TypeError as exc:
                raise DomainValidationError([str(exc)], subject="FlexibilityEnvelope") from exc
            if rel_shape != up_shape:
                errors.append(
                    f"reliability {rel_shape} must match the magnitude shape {up_shape}"
                )
            elif any(value < 0.0 or value > 1.0 for value in rel_flat):
                errors.append(
                    f"reliability must lie in [0, 1], got "
                    f"[{min(rel_flat)}, {max(rel_flat)}]"
                )
        if self.timestamps and len(self.timestamps) != up_shape[0]:
            errors.append(
                f"timestamps holds {len(self.timestamps)} entries for {up_shape[0]} rows"
            )
        if self.nominal_level is not None and not 0.0 < float(self.nominal_level) < 1.0:
            errors.append(f"nominal_level must lie in (0, 1), got {self.nominal_level}")
        if not isinstance(self.provenance, Provenance):
            errors.append(f"provenance must be a Provenance, got {type(self.provenance).__name__}")
        if not isinstance(self.method, str) or not self.method.strip():
            errors.append(f"method must be a non-empty string, got {self.method!r}")
        if self.basis is FlexibilityBasis.UNKNOWN and up_flat:
            errors.append(
                "a basis=UNKNOWN envelope may not carry magnitudes; use "
                "FlexibilityEstimate with magnitude_kw=None instead"
            )
        if errors:
            raise DomainValidationError(errors, subject="FlexibilityEnvelope")

    @property
    def n_rows(self) -> int:
        rows = self.upward_kw
        try:
            return len(rows)
        except TypeError:
            raise DomainValidationError(
                ["upward_kw must be a sequence"], subject="FlexibilityEnvelope"
            ) from None

    @property
    def is_dispatchable(self) -> bool:
        return (
            self.authority in CONTROLLABLE_AUTHORITY_LEVELS
            and self.basis in (FlexibilityBasis.PHYSICAL, FlexibilityBasis.ASSUMED)
        )

    def to_dict(self) -> dict[str, Any]:
        """A JSON-ready dictionary carrying every declared contract field.

        The magnitude **arrays** are not in here: they are per-row numeric data whose size
        depends on the panel, so they are published beside this as an NPZ and referenced by
        name. What belongs in a contract object is the claim - basis, authority, level,
        horizons, provenance, limitations - and ``is_dispatchable`` in particular is
        serialised explicitly so that a deserialised envelope cannot quietly recover a
        control authority it was not allowed to hold.
        """
        return {
            "_flexibility_schema_version": FLEXIBILITY_SCHEMA_VERSION,
            "target": self.target,
            "aggregation": self.aggregation.value,
            "basis": self.basis.value,
            "authority": self.authority.value,
            "method": self.method,
            "nominal_level": (
                None if self.nominal_level is None else float(self.nominal_level)
            ),
            "horizons": [int(h) for h in self.horizons],
            "rows": int(self.upward_kw.shape[0]),
            "upward_kw_shape": list(self.upward_kw.shape),
            "downward_kw_shape": list(self.downward_kw.shape),
            "is_dispatchable": self.is_dispatchable,
            "limitations": list(self.limitations),
            "provenance": {
                "source": self.provenance.source.source_id,
                "dataset": self.provenance.source.dataset,
                "locator": self.provenance.source.locator,
                "version": self.provenance.source.version,
                "events": [event.event for event in self.provenance.events],
                "processing": [step.name for step in self.provenance.processing],
            },
        }

    def at(self, row: int, horizon_index: int | None = None) -> FlexibilityEstimate:
        """One row as the per-row domain contract object.

        Args:
            row: Row position within the block.
            horizon_index: Column when the block is two-dimensional, else ``None``.

        Returns:
            The estimate for that row and column.

        Raises:
            IndexError: If the row or column is out of range.
        """
        if not 0 <= int(row) < self.n_rows:
            raise IndexError(f"row {row} out of range for {self.n_rows} rows")

        def pick(array: Any) -> float:
            # A one-dimensional block indexes by row; a two-dimensional one indexes by row
            # then column. Read structurally, because this layer imports no numeric library.
            first = list(array)[int(row)]
            if isinstance(first, (list, tuple)) or (
                hasattr(first, "__len__") and not isinstance(first, (str, bytes))
            ):
                cells = list(first)
                index = int(horizon_index or 0)
                if not 0 <= index < len(cells):
                    raise IndexError(
                        f"horizon index {index} out of range for {len(cells)} columns"
                    )
                return float(cells[index])
            return float(first)

        timestamp = self.timestamps[int(row)] if self.timestamps else ""
        return FlexibilityEstimate(
            target=self.target,
            direction=FlexibilityDirection.UPWARD,
            basis=self.basis,
            authority=self.authority,
            magnitude_kw=pick(up),
            unit=Unit.KILOWATT,
            timestamp=timestamp,
            provenance=self.provenance,
            method=self.method,
            horizon_steps=(
                self.horizons[int(horizon_index or 0)] if self.horizons else None
            ),
            validity_steps=None,
            aggregation=self.aggregation,
            uncertainty=None,
            limitations=self.limitations,
        )