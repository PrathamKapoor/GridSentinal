"""Data-quality representation.

Phase 2 makes the domain model *capable* of expressing data quality. It does not
detect any of these conditions: outlier detection, staleness detection and sensor
fault diagnosis all belong to later phases (the Anomaly Agent, Phase 14). What is
defined here is the vocabulary and the invariants, so that a detected condition
has somewhere to be recorded when it is eventually detected.

This is the one area where SAT-SA is a *negative* reference: it has no staleness
detector, no sensor-error flag and no estimated/imputed marker, and its own
research notes acknowledge that gap. Those omissions are exactly the ones worth
closing here. See ``decisions.md`` (D-029).
"""

from __future__ import annotations

from dataclasses import dataclass

from .enums import QualityFlag
from .errors import DomainValidationError

__all__ = ["DataQuality", "CLEAN_QUALITY"]


@dataclass(frozen=True, slots=True)
class DataQuality:
    """Quality metadata attached to a measurement or state.

    Flags are a set, not a single value, because conditions genuinely co-occur:
    a reading can be both stale and interpolated.

    Invariants:
        * At least one flag is required; "no information" is not the same as "clean".
        * :attr:`QualityFlag.OK` cannot be combined with any other flag. Mixing
          "ok" with "missing" would be self-contradictory and would let a
          degraded reading pass a naive ``OK in flags`` check.

    Attributes:
        flags: The applicable quality conditions.
        detail: Free-form elaboration for the human reading a report.
    """

    flags: frozenset[QualityFlag]
    detail: str = ""

    def __post_init__(self) -> None:
        errors: list[str] = []

        if not isinstance(self.flags, frozenset):
            errors.append(
                f"flags must be a frozenset, got {type(self.flags).__name__}; "
                "use frozenset({QualityFlag.OK}) rather than a set so quality "
                "metadata is hashable"
            )
        elif not self.flags:
            errors.append(
                "at least one flag is required; absence of flags means unknown "
                "quality, which must be stated explicitly rather than implied"
            )
        else:
            invalid = self.flags - frozenset(QualityFlag)
            if invalid:
                errors.append(f"unknown QualityFlag member(s): {sorted(invalid)}")
            if QualityFlag.OK in self.flags and len(self.flags) > 1:
                errors.append(
                    f"{QualityFlag.OK.value!r} cannot be combined with other flags; "
                    f"got {sorted(flag.value for flag in self.flags)}"
                )

        if errors:
            raise DomainValidationError(errors, subject="DataQuality")

    @property
    def is_clean(self) -> bool:
        """True only when explicitly flagged OK and nothing else."""
        return self.flags == frozenset({QualityFlag.OK})

    @property
    def has(self) -> bool:
        """True when at least one adverse condition applies."""
        return not self.is_clean

    def has_flag(self, flag: QualityFlag) -> bool:
        return flag in self.flags

    @property
    def is_usable_for_decision(self) -> bool:
        """Whether a decision may rely on this value directly.

        Conservative by design. Any adverse condition -- missing, stale, out of
        range, outlier, sensor error, estimated or interpolated -- disqualifies
        the value. Staleness is included deliberately: an out-of-date state of
        charge is precisely the input that would produce a confidently wrong
        dispatch decision, and "it was fine a few steps ago" is not a safe basis
        for acting on a time-coupled system.

        Interpolation and estimation are not distinguished from outright defects
        here, so that a later decision-assurance policy can choose to admit them
        under a tightened assurance level. This property is the strict default.
        """
        blocking = frozenset(QualityFlag) - {QualityFlag.OK}
        return not (self.flags & blocking)

    def with_flag(self, flag: QualityFlag) -> DataQuality:
        """Return a copy with one more flag, dropping OK if it is now mixed."""
        flags = set(self.flags)
        flags.discard(QualityFlag.OK)
        flags.add(flag)
        return DataQuality(flags=frozenset(flags), detail=self.detail)

    def __str__(self) -> str:
        return f"DataQuality({', '.join(sorted(flag.value for flag in self.flags))})"

#: The all-clear. Defined after the class so construction order is correct.
#: Shared so comparisons need no allocation.
CLEAN_QUALITY = DataQuality(flags=frozenset({QualityFlag.OK}))