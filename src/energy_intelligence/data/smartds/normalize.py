"""Normalisation of raw SMART-DS records into domain-ready series.

Three things happen here, each recorded as provenance:

1. **Per-unit to kilowatts.** SMART-DS load profiles are a *fraction of the
   annual maximum*, not power. Multiplying by the ``Loads.dss`` kW rating is
   what produces kilowatts, and that multiplication is the single most important
   derivation in the whole pipeline.

2. **Index to timestamp.** The dataset carries time only as a positional index
   into a 35040-point yearly vector. Timestamps must be *constructed*.

3. **Quality annotation.** Nothing is filled in, imputed or interpolated.

Timestamp semantics -- an explicit assumption, not a discovered fact
-------------------------------------------------------------------------
SMART-DS contains **no timezone information whatsoever**. Observed time is a
position in a synthetic 35040-point yearly vector referenced by
``Solve mode=yearly stepsize=15m number=35040``. There is no UTC offset, no
local time, and no daylight-saving handling: OpenDSS's yearly mode uses a
continuous synthetic clock.

The Phase 2 domain requires timezone-aware datetimes and rejects naive ones.
These requirements cannot both be satisfied without a choice, so the choice is
made explicitly here, recorded in provenance, and is overridable:

    The synthetic clock is anchored to ``<year>-01-01T00:00:00+00:00`` and the
    resulting timestamps are treated as UTC.

Why this is acceptable rather than a silent lie: the whole dataset is internally
consistent on one synthetic clock, so anchoring it anywhere preserves every
*relative* time relationship -- which is what forecasting, flexibility and
balance validation depend on. What it does not preserve is true wall-clock local
time (Austin is CST/CDT). This is recorded as a known limitation in the gaps
report, not hidden.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from ...domain.quality import DataQuality
from .dss import DssObject

__all__ = [
    "GRID_POINTS_365_DAYS",
    "TimeAxis",
    "LoadPoint",
    "CustomerLoad",
    "TimezoneAssumption",
    "aggregate",
    "build_time_axis",
]

#: 365 days x 96 steps of 15 minutes, confirmed from three independent sources.
GRID_POINTS_365_DAYS = 35_040

#: The single documented, overridable assumption about time.
TIMEZONE_ASSUMPTION = (
    "SMART-DS provides no timezone. The synthetic yearly clock used by "
    "'Solve mode=yearly stepsize=15m' is anchored to 01-01T00:00:00+00:00 and "
    "treated as UTC. Relative timing is preserved exactly; true local wall-clock "
    "time is NOT, because the dataset carries no UTC offset."
)

TimezoneAssumption = TIMEZONE_ASSUMPTION


@dataclass(frozen=True, slots=True)
class TimeAxis:
    """The dataset's native time grid, mapped onto absolute timestamps.

    Attributes:
        year: Calendar year the vector covers.
        timestep_minutes: Native resolution, 15.
        points: Number of timepoints.
        assumed_utc: Whether the synthetic clock is being treated as UTC.
    """

    year: int
    timestep_minutes: int
    points: int
    assumed_utc: bool = True

    def __post_init__(self) -> None:
        if self.timestep_minutes <= 0:
            raise ValueError(f"timestep_minutes must be positive, got {self.timestep_minutes}")
        if self.points <= 0:
            raise ValueError(f"points must be positive, got {self.points}")
        expected_days = self.points / (24 * 60 / self.timestep_minutes)
        if expected_days > 366.5:
            raise ValueError(
                f"{self.points} points at {self.timestep_minutes} min implies "
                f"{expected_days:.1f} days, which is more than a year"
            )

    @property
    def origin(self) -> datetime:
        return datetime(self.year, 1, 1, tzinfo=timezone.utc)

    def timestamp(self, index: int) -> datetime:
        """Absolute timestamp of grid ``index``. Negative indices are history."""
        return self.origin + timedelta(minutes=self.timestep_minutes * index)

    def index(self, timestamp: datetime) -> int:
        """Inverse of :meth:`timestamp`, exact only on grid points."""
        offset = timestamp - self.origin
        steps = offset.total_seconds() / (self.timestep_minutes * 60)
        rounded = round(steps)
        if abs(steps - rounded) > 1e-9:
            raise ValueError(
                f"{timestamp.isoformat()} is not aligned to the "
                f"{self.timestep_minutes}-minute grid"
            )
        return int(rounded)

    @property
    def days(self) -> float:
        return self.points * self.timestep_minutes / (60 * 24)

    @property
    def matches_domain_grid(self) -> bool:
        """Whether this axis is directly usable by the Phase 2 time base."""
        return self.timestep_minutes == 15 and self.points % 96 == 0

    def describe(self) -> dict[str, object]:
        return {
            "year": self.year,
            "timestep_minutes": self.timestep_minutes,
            "points": self.points,
            "days": self.days,
            "assumed_utc": self.assumed_utc,
            "timezone_assumption": TIMEZONE_ASSUMPTION if self.assumed_utc else "",
            "matches_domain_grid": self.matches_domain_grid,
        }


def build_time_axis(points: int, *, year: int, timestep_minutes: int = 15) -> TimeAxis:
    """Build a :class:`TimeAxis`, validating the point count."""
    return TimeAxis(
        year=year, timestep_minutes=timestep_minutes, points=points, assumed_utc=True
    )


@dataclass(frozen=True, slots=True)
class LoadPoint:
    """One load's power at one grid index.

    Attributes:
        index: Position in the yearly vector.
        kw: Real power in kilowatts.
    """

    index: int
    kw: float

    @property
    def kvar(self) -> float:
        return 0.0


@dataclass(frozen=True, slots=True)
class CustomerLoad:
    """A customer's real power series, already in kilowatts.

    Attributes:
        name: Base name with the centre-tap suffix stripped, e.g.
            ``"load_p1ulv279"``.
        members: The OpenDSS objects that make up this customer.
        series: Power by grid index, in kW.
        rated_kw: Sum of member ratings, i.e. the annual maximum customer draw.
        quality: Data-quality assessment for the series.
        bus: The customer's connection bus with any OpenDSS phase suffix removed,
            or ``None`` when the dataset states no bus.
    """

    name: str
    members: tuple[DssObject, ...]
    series: tuple[float, ...]
    rated_kw: float
    quality: DataQuality
    bus: str | None = None

    @property
    def is_center_tap(self) -> bool:
        return len(self.members) == 2

    @property
    def is_commercial(self) -> bool:
        """Whether the customer's loadshape name marks it commercial.

        The convention is the dataset's own: a shape named ``com_kw_*`` belongs to
        a commercial customer and ``res_kw_*`` to a residential one. Phase 3 uses
        the same prefix test for its commercial/residential balance identities, so
        classification cannot drift between phases.
        """
        if not self.members:
            return False
        shape = self.members[0].get("yearly", "")
        return shape.startswith("com")

    @property
    def peak_kw(self) -> float:
        return max(self.series) if self.series else 0.0

    def value_at(self, index: int) -> float:
        return self.series[index]


def aggregate(series: tuple[float, ...], indices: tuple[int, ...]) -> float:
    """Sum selected indices, skipping out-of-range indices."""
    total = 0.0
    length = len(series)
    for index in indices:
        if 0 <= index < length:
            total += series[index]
    return total


def read_profile(path: Path) -> tuple[float, ...]:
    """Read a headerless single-column per-unit profile.

    Raises:
        ValueError: If any line is not a real number. A malformed profile must
            not be silently truncated.
    """
    values: list[float] = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                values.append(float(text))
            except ValueError as exc:
                raise ValueError(
                    f"{path.name} line {number}: not a real number: {text!r}"
                ) from exc
    return tuple(values)
