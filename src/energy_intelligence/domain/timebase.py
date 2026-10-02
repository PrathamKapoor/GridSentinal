"""Temporal model.

Chosen: **15-minute native timestep, 96-step (24-hour) day-ahead horizon, with
hourly derivable by aggregation.** See ``decisions.md`` (D-027).

Rationale, recorded so it can be revisited rather than inherited:

* Battery cycling and EV charging are only meaningfully schedulable at
  sub-hourly resolution. A 1-hour battery cycle cannot express most economically
  or grid-service-useful schedules, so hourly would under-resolve the very
  flexibility the project is about.
* PV ramp is materially non-representative inside an hour on a distribution
  feeder, where cloud transients drive the net-load ramps that motivate the
  study.
* 15 minutes is the resolution used by distribution operators and by the
  SMART-DS representability anchor, so the model is not fighting its data.

The values are *not* hard-coded: :class:`TimeBase` is constructed from
configuration, so selecting 5-minute or hourly data later requires no code
change.

Timestamps are timezone-aware UTC. Naive datetimes are rejected, because a naive
timestamp in an energy system is ambiguous across a daylight-saving transition
and silently corrupts any time-series join.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from .errors import DomainValidationError

__all__ = ["TimeBase", "DEFAULT_TIMESTEP_MINUTES", "DEFAULT_HORIZON_STEPS", "UTC"]

UTC = timezone.utc

#: Recorded as the project default; overridable via configuration.
DEFAULT_TIMESTEP_MINUTES = 15
DEFAULT_HORIZON_STEPS = 96

#: Steps per aggregation period, keyed by the divisor used to derive one.
_MINUTES_PER_HOUR = 60
_SECONDS_PER_HOUR = 3600


@dataclass(frozen=True, slots=True, order=True)
class TimeBase:
    """The temporal grid the energy system is discretised onto.

    Attributes:
        timestep_minutes: Native resolution. One grid step.
        horizon_steps: Number of steps the system plans/forecasts ahead.
        origin: Datetime the grid is anchored to; every step index is relative
            to this, which makes grid arithmetic independent of calendar
            alignment.
    """

    timestep_minutes: int
    horizon_steps: int
    origin: datetime

    def __post_init__(self) -> None:
        errors: list[str] = []

        if isinstance(self.timestep_minutes, bool) or not isinstance(self.timestep_minutes, int):
            errors.append(
                f"timestep_minutes must be an int, got {type(self.timestep_minutes).__name__}"
            )
        elif self.timestep_minutes <= 0:
            errors.append(f"timestep_minutes must be positive, got {self.timestep_minutes}")

        if isinstance(self.horizon_steps, bool) or not isinstance(self.horizon_steps, int):
            errors.append(
                f"horizon_steps must be an int, got {type(self.horizon_steps).__name__}"
            )
        elif self.horizon_steps <= 0:
            errors.append(f"horizon_steps must be positive, got {self.horizon_steps}")

        if not isinstance(self.origin, datetime):
            errors.append(f"origin must be a datetime, got {type(self.origin).__name__}")
        elif self.origin.tzinfo is None or self.origin.utcoffset() is None:
            errors.append(
                "origin must be timezone-aware; a naive datetime is ambiguous "
                "across daylight-saving transitions"
            )

        if errors:
            raise DomainValidationError(errors, subject="TimeBase")

    # ------------------------------------------------------------------
    # Grid arithmetic
    # ------------------------------------------------------------------

    @property
    def timestep(self) -> timedelta:
        """Length of one grid step."""
        return timedelta(minutes=self.timestep_minutes)

    @property
    def horizon_duration(self) -> timedelta:
        """Total time spanned by the planning/forecast horizon."""
        return self.timestep * self.horizon_steps

    def step_time(self, index: int) -> datetime:
        """Return the wall-clock time of grid step ``index``.

        ``index`` may be negative (history) or >= 0 (forecast).
        """
        if isinstance(index, bool) or not isinstance(index, int):
            raise DomainValidationError(
                [f"step index must be an int, got {type(index).__name__}"], subject="TimeBase"
            )
        return self.origin + self.timestep * index

    def step_index(self, moment: datetime) -> int:
        """Return the grid index of ``moment``.

        Raises:
            DomainValidationError: If ``moment`` is not exactly aligned to the
                grid. Silent rounding of an off-grid timestamp would shift the
                whole system by a partial step, so misalignment is an error.
        """
        if not isinstance(moment, datetime):
            raise DomainValidationError(
                [f"moment must be a datetime, got {type(moment).__name__}"], subject="TimeBase"
            )
        if moment.tzinfo is None or moment.utcoffset() is None:
            raise DomainValidationError(
                ["moment must be timezone-aware"], subject="TimeBase"
            )

        offset = moment - self.origin
        seconds = offset.total_seconds()
        step_seconds = self.timestep.total_seconds()
        quotient = seconds / step_seconds

        if abs(quotient - round(quotient)) > 1e-9:
            raise DomainValidationError(
                [
                    f"{moment.isoformat()} is not aligned to the {self.timestep_minutes}-minute "
                    f"grid anchored at {self.origin.isoformat()}"
                ],
                subject="TimeBase",
            )
        return int(round(quotient))

    def is_aligned(self, moment: datetime) -> bool:
        """Whether ``moment`` falls exactly on a grid step."""
        try:
            self.step_index(moment)
        except DomainValidationError:
            return False
        return True

    def steps_per_hour(self) -> int:
        """Steps in one hour, or ``None`` when the timestep does not divide it."""
        if _MINUTES_PER_HOUR % self.timestep_minutes:
            return None
        return _MINUTES_PER_HOUR // self.timestep_minutes

    def aggregate_to_hourly(self) -> bool:
        """Whether an hourly view can be derived by integer aggregation.

        With the chosen 15-minute timestep this is True (4 steps/hour). With, for
        example, a 7-minute timestep it is False, and Phase 3 must resample
        rather than aggregate.
        """
        return self.steps_per_hour() is not None

    @property
    def horizon_hours(self) -> float:
        """Horizon length expressed in hours, for reporting."""
        return self.horizon_duration.total_seconds() / _SECONDS_PER_HOUR

    def __str__(self) -> str:
        return (
            f"TimeBase({self.timestep_minutes}min x {self.horizon_steps} steps "
            f"= {self.horizon_hours:g}h)"
        )
