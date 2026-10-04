"""Uncertainty representation.

Uncertainty is first-class in this project: the whole architecture depends on
knowing *how uncertain* a decision is, not merely what it predicts. Phase 2
therefore defines the container now and deliberately defers the mathematics.

What is decided
    A forecast or a measurement may carry an :class:`UncertaintyEstimate`.
    Its :class:`~energy_intelligence.domain.enums.UncertaintyKind` distinguishes
    measurement uncertainty (sensor error), model uncertainty (epistemic,
    reducible with more data), parametric uncertainty (aleatoric, irreducible
    scenario spread) and combined. ``UNKNOWN`` is a legal value so a model can
    honestly declare that uncertainty has not yet been characterised.

What is NOT decided
    The numeric representation. A later phase (Phase 4 baselines, Phase 8
    uncertainty) must select a method -- quantiles, ensembles, conformal
    prediction, and so on. Until then :attr:`UncertaintyEstimate.method` is
    recorded as ``"TBD"`` and no interval semantics are assumed.

Why this matters
    Neither source repository models uncertainty. The Guardrailed
    forecasting repository contains no reference to quantiles, conformal
    prediction, ensembles or aleatoric/epistemic decomposition. Getting the
    container right while the methodology is still open is what allows Phase 8 to
    choose a method without changing every domain object. See
    ``decisions.md`` (D-033).
"""

from __future__ import annotations

from dataclasses import dataclass

from .enums import UncertaintyKind, Unit
from .errors import DomainValidationError

__all__ = ["UncertaintyEstimate", "UNKNOWN_METHOD", "TBD_METHOD"]

#: Placeholder until Phase 4/8 selects a method. Recorded explicitly rather than
#: left blank so it is visible that the choice is outstanding.
TBD_METHOD = "TBD"


@dataclass(frozen=True, slots=True)
class UncertaintyEstimate:
    """A quantitative or qualitative statement of uncertainty.

    Exactly one **representation** may be supplied: a prediction interval
    (``lower_bound`` and ``upper_bound`` together, which is one quantity) **or** a scalar
    summary (``standard_deviation`` or ``relative_std``). Supplying an interval *and* a
    scalar is rejected rather than silently accepted, because the relationship between a
    width and a standard deviation depends on the distributional assumption the interval's
    method made. Passing ``None`` for all four with a non-``UNKNOWN`` kind is also
    rejected: an uncertainty that asserts nothing is not an uncertainty estimate.

    Phase 2 additionally forbade the paired interval, which left this container unable to
    hold the one representation Phase 8 exists to produce. Repaired in Phase 8; see
    ``decisions.md`` D-099.

    Attributes:
        kind: Which source of uncertainty this describes.
        method: How it was computed. ``"TBD"`` until Phase 8 selects a method.
        lower_bound: Optional lower bound, in ``unit``.
        upper_bound: Optional upper bound, in ``unit``.
        standard_deviation: Optional standard deviation, in ``unit``.
        relative_std: Optional coefficient of variation (dimensionless).
        notes: Free-form caveats, e.g. assumptions or sample size.
    """

    kind: UncertaintyKind
    unit: Unit
    method: str = TBD_METHOD
    lower_bound: float | None = None
    upper_bound: float | None = None
    standard_deviation: float | None = None
    relative_std: float | None = None
    notes: str = ""

    def __post_init__(self) -> None:
        errors: list[str] = []

        if not isinstance(self.kind, UncertaintyKind):
            errors.append(f"kind must be an UncertaintyKind, got {type(self.kind).__name__}")
        if not isinstance(self.unit, Unit):
            errors.append(f"unit must be a Unit, got {type(self.unit).__name__}")
        if not isinstance(self.method, str) or not self.method.strip():
            errors.append(f"method must be a non-empty string, got {self.method!r}")

        supplied = [
            name
            for name, value in (
                ("lower_bound", self.lower_bound),
                ("upper_bound", self.upper_bound),
                ("standard_deviation", self.standard_deviation),
                ("relative_std", self.relative_std),
            )
            if value is not None
        ]

        if self.kind is not UncertaintyKind.UNKNOWN and not supplied:
            errors.append(
                f"an UncertaintyEstimate of kind {getattr(self.kind, 'value', self.kind)!r} "
                "must quantify something; supply lower_bound/upper_bound/"
                "standard_deviation/relative_std, or use kind=UNKNOWN to declare it "
                "unquantified"
            )

        # A *paired* lower/upper is ONE quantity - a prediction interval - and is allowed.
        # Phase 2 counted the two bounds separately, which made a prediction interval the
        # one representation the container could not hold, and Phase 8 exists to produce
        # one. Mixing an interval with a scalar summary is still refused, because how a
        # width relates to a standard deviation depends on the distributional assumption
        # the interval's method made - which is exactly what its `method` names. See
        # decisions.md D-099.
        has_interval = self.lower_bound is not None or self.upper_bound is not None
        has_scalar = (
            self.standard_deviation is not None or self.relative_std is not None
        )
        if has_interval and has_scalar:
            errors.append(
                "supply either a prediction interval (lower_bound and upper_bound) or a "
                "scalar summary (standard_deviation or relative_std), not both; how an "
                "interval relates to a standard deviation depends on the distributional "
                "assumption the interval's method makes"
            )
        if has_interval and len(supplied) == 1:
            missing = (
                "upper_bound" if self.lower_bound is not None else "lower_bound"
            )
            errors.append(
                f"an interval needs both bounds; {missing} is missing"
            )

        for name in ("lower_bound", "upper_bound", "standard_deviation", "relative_std"):
            value = getattr(self, name)
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                errors.append(f"{name} must be a real number, got {value!r}")
                continue
            if value != value:  # NaN
                errors.append(f"{name} must not be NaN")
            if name in ("standard_deviation", "relative_std") and value < 0:
                errors.append(f"{name} must be non-negative, got {value}")

        if (
            self.lower_bound is not None
            and self.upper_bound is not None
            and self.lower_bound > self.upper_bound
        ):
            errors.append(
                f"lower_bound {self.lower_bound} exceeds upper_bound {self.upper_bound}"
            )

        if errors:
            raise DomainValidationError(errors, subject="UncertaintyEstimate")

    @property
    def is_quantified(self) -> bool:
        """Whether a numeric summary is attached."""
        return self.kind is not UncertaintyKind.UNKNOWN

    @property
    def is_method_selected(self) -> bool:
        """Whether a computation method has been chosen yet."""
        return self.method != TBD_METHOD

    def __str__(self) -> str:
        if not self.is_quantified:
            return f"UncertaintyEstimate({self.kind.value}, unquantified)"
        return f"UncertaintyEstimate({self.kind.value}, {self.method})"
