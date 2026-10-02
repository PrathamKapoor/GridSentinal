"""Bus coordinate parsing.

``Buscoords.dss`` is **not** a normal OpenDSS directive file. Despite the
extension it contains no ``New``/``Set`` directives at all -- observed content is
a bare whitespace-separated list:

```text
p1udm971 -97.71529442807753 30.43181335686635
```

i.e. ``<bus_name> <longitude> <latitude>``. This was confirmed by reading the
real file, not assumed from the extension. Parsing it with a generic ``.dss``
parser silently yields zero elements, which would make every node appear to have
no location -- a silent data loss rather than an error.

Coordinates are **degrees** (longitude, latitude) per SMART-DS documentation and
per the observed magnitude (~-97.7, ~30.4 for Austin, Texas).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from .dss import DssParseError

__all__ = ["BusCoordinate", "parse_buscoords", "BUS_LON_RANGE", "BUS_LAT_RANGE"]

#: Plausibility bounds for degrees. Used to reject a mis-parsed column rather
#: than accepting longitude 30.4 as a latitude.
BUS_LON_RANGE = (-180.0, 180.0)
BUS_LAT_RANGE = (-90.0, 90.0)


@dataclass(frozen=True, slots=True)
class BusCoordinate:
    """Geographic position of one bus.

    Attributes:
        bus: Bus name, e.g. ``"p1udm971"``.
        longitude: Degrees east-positive.
        latitude: Degrees north-positive.
    """

    bus: str
    longitude: float
    latitude: float

    def __post_init__(self) -> None:
        if not self.bus.strip():
            raise DssParseError("bus coordinate has an empty bus name")
        for name, value, bounds in (
            ("longitude", self.longitude, BUS_LON_RANGE),
            ("latitude", self.latitude, BUS_LAT_RANGE),
        ):
            if math.isnan(value) or math.isinf(value):
                raise DssParseError(f"{self.bus} {name} is not finite: {value!r}")
            low, high = bounds
            if not low <= value <= high:
                raise DssParseError(
                    f"{self.bus} {name} {value!r} is outside the plausible range "
                    f"[{low}, {high}]; the file columns may be mis-ordered"
                )


def parse_buscoords(text: str) -> dict[str, BusCoordinate]:
    """Parse a ``Buscoords.dss`` coordinate list.

    Args:
        text: Full file contents.

    Returns:
        Bus name to :class:`BusCoordinate`.

    Raises:
        DssParseError: If a line does not have exactly three fields, or a
            coordinate is out of range. Comment lines are skipped.
    """
    out: dict[str, BusCoordinate] = {}
    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith(("!", "//", "Set", "Buscoords")):
            continue
        parts = line.split()
        if len(parts) != 3:
            raise DssParseError(
                f"Buscoords line {number}: expected '<bus> <lon> <lat>', got {len(parts)} "
                f"field(s): {line!r}"
            )
        bus, lon_raw, lat_raw = parts
        try:
            longitude = float(lon_raw)
            latitude = float(lat_raw)
        except ValueError as exc:
            raise DssParseError(
                f"Buscoords line {number}: non-numeric coordinate in {line!r}"
            ) from exc

        # Constructed outside the numeric try/except: DssParseError subclasses
        # ValueError, so catching ValueError here would swallow the more
        # specific out-of-range message.
        coordinate = BusCoordinate(bus=bus, longitude=longitude, latitude=latitude)
        if bus in out:
            raise DssParseError(f"Buscoords line {number}: duplicate bus {bus!r}")
        out[bus] = coordinate
    return out


def load_buscoords(path: Path) -> dict[str, BusCoordinate]:
    """Read and parse a ``Buscoords.dss`` from disk."""
    return parse_buscoords(path.read_text(encoding="utf-8", errors="replace"))
