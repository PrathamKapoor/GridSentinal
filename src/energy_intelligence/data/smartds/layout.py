"""SMART-DS dataset layout.

Encapsulates *where things are* in the dataset, so that no other module needs to
know the directory convention. This is the only module that knows that the
dataset is organised as
``<version>/<year>/<region>/<subregion>/scenarios/<scenario>/...``.

The layout facts below were confirmed by listing the authoritative OEDI bucket
and by downloading real files, not assumed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from ...domain.identifiers import NodeId, SystemId
from ...domain.errors import DomainValidationError

__all__ = [
    "OEDI_BUCKET_URL",
    "SMART_DS_VERSION",
    "REGIONS",
    "SCENARIO_RE",
    "SmartDsLayout",
    "sanitize_identifier",
    "validate_scenario_name",
    "KNOWN_SCENARIOS",
]

#: Authoritative public endpoint. SMART-DS is distributed through the OEDI data
#: lake as an open, unauthenticated S3 bucket; there is no mirror in use.
OEDI_BUCKET_URL = "https://oedi-data-lake.s3.amazonaws.com/"

#: Dataset version this adapter targets. Confirmed by reading the shipped
#: ``SMART-DS_version.txt`` inside a scenario folder, whose entire content is
#: "1.0".
SMART_DS_VERSION = "1.0"

#: Regions present in v1.0, confirmed by listing the bucket.
REGIONS: frozenset[str] = frozenset({"AUS", "GSO", "SFO"})

SCENARIO_RE = re.compile(
    r"^(?P<solar>base|none|low|medium|high|extreme)"
    r"(?:_solar_(?P<solar_level>none|low|medium|high|extreme))?"
    r"(?:_batteries_(?P<battery_level>none|low|high))?"
    r"(?:_(?P<mode>timeseries|peak))?$"
)

#: Scenario directory names observed in the bucket listing for AUS/P1U/2018.
KNOWN_SCENARIOS: tuple[str, ...] = (
    "base_timeseries",
    "solar_extreme_batteries_high_timeseries",
    "solar_extreme_batteries_low_timeseries",
    "solar_extreme_batteries_none_timeseries",
    "solar_high_batteries_high_timeseries",
    "solar_high_batteries_low_timeseries",
    "solar_high_batteries_none_timeseries",
    "solar_low_batteries_high_timeseries",
    "solar_low_batteries_low_timeseries",
    "solar_low_batteries_none_timeseries",
    "solar_medium_batteries_high_timeseries",
    "solar_medium_batteries_low_timeseries",
    "solar_medium_batteries_none_timeseries",
    "solar_none_batteries_high_timeseries",
    "solar_none_batteries_low_timeseries",
)

_BARE_NAME_RE = re.compile(r"^[-a-z0-9]+$")


def sanitize_identifier(raw: str, max_length: int = 60) -> str:
    """Convert a SMART-DS name into a valid Phase 2 identifier body.

    SMART-DS names contain characters the Phase 2 identifier pattern forbids.
    Lowercasing and collapsing disallowed runs into single hyphens keeps the
    mapping auditable: the result is a deterministic function of the input.

    Args:
        raw: Source name.
        max_length: Cap on the produced body length.

    Raises:
        DomainValidationError: If nothing usable survives sanitisation.
    """
    lowered = raw.strip().lower()
    cleaned = re.sub(r"[^a-z0-9]+", "-", lowered).strip("-")
    cleaned = re.sub(r"-{2,}", "-", cleaned)
    if not cleaned:
        raise DomainValidationError(
            [f"name {raw!r} contains no identifier-safe characters"], subject="sanitize_identifier"
        )
    if not cleaned[0].isalnum():
        cleaned = f"x{cleaned}"
    return cleaned[:max_length].rstrip("-")


def validate_scenario_name(name: str) -> str:
    """Validate a scenario directory name against the observed grammar."""
    if name not in KNOWN_SCENARIOS:
        raise DomainValidationError(
            [
                f"unknown SMART-DS scenario {name!r}",
                f"known scenarios: {list(KNOWN_SCENARIOS)}",
            ],
            subject="validate_scenario_name",
        )
    return name


@dataclass(frozen=True, slots=True)
class SmartDsLayout:
    """Resolved locations within one SMART-DS sub-region.

    Attributes:
        root: Local root of the downloaded dataset.
        version: Dataset version directory, e.g. ``"v1.0"``.
        year: Year directory, e.g. ``"2018"``.
        region: Region code, e.g. ``"AUS"``.
        subregion: Sub-region, e.g. ``"P1U"``.
        scenario: Scenario directory name.
        substation: Sub-station folder name.
        feeder: Feeder folder name.
    """

    root: Path
    version: str
    year: str
    region: str
    subregion: str
    scenario: str
    substation: str
    feeder: str

    # ------------------------------------------------------------------
    # Directory builders
    # ------------------------------------------------------------------

    @property
    def version_dir(self) -> Path:
        return self.root / self.version

    @property
    def subregion_dir(self) -> Path:
        return self.version_dir / self.year / self.region / self.subregion

    @property
    def profiles_dir(self) -> Path:
        """Load and solar profile files, shared across scenarios in a sub-region."""
        return self.subregion_dir / "profiles"

    @property
    def solar_data_dir(self) -> Path:
        return self.subregion_dir / "solar_data"

    @property
    def load_data_dir(self) -> Path:
        return self.subregion_dir / "load_data"

    @property
    def placements_dir(self) -> Path:
        return self.version_dir / "placements" / self.region / self.subregion

    @property
    def user_guide_dir(self) -> Path:
        return self.version_dir / "User_Guide"

    @property
    def scenario_dir(self) -> Path:
        return self.subregion_dir / "scenarios" / self.scenario

    @property
    def opendss_dir(self) -> Path:
        return self.scenario_dir / "opendss"

    @property
    def substation_dir(self) -> Path:
        return self.opendss_dir / self.substation

    @property
    def feeder_dir(self) -> Path:
        return self.substation_dir / self.feeder

    @property
    def feeder_analysis_dir(self) -> Path:
        return self.feeder_dir / "analysis"

    def feeder_file(self, name: str) -> Path:
        return self.feeder_dir / name

    def placement_file(self, scenario_key: str, penetration: str) -> Path:
        """Return the placement JSON path, e.g. ``battery_customer=L.json``."""
        if penetration not in {"L", "M", "H", "E"}:
            raise DomainValidationError(
                [f"penetration must be one of L, M, H, E; got {penetration!r}"],
                subject="SmartDsLayout.placement_file",
            )
        return self.placements_dir / f"{scenario_key}={penetration}.json"

    # ------------------------------------------------------------------
    # Naming
    # ------------------------------------------------------------------

    def system_id(self) -> SystemId:
        """A stable ``SystemId`` for this sub-region."""
        return SystemId(
            f"sys-smartds-{sanitize_identifier(self.region)}-{sanitize_identifier(self.subregion)}"
        )

    def feeder_node_id(self, bus: str) -> NodeId:
        """Map a SMART-DS bus name to a Phase 2 :class:`NodeId`."""
        return NodeId(f"node-smartds-{sanitize_identifier(bus)}")

    def s3_prefix(self) -> str:
        """The authoritative remote prefix, excluding the bucket host."""
        return (
            f"SMART-DS/{self.version}/{self.year}/{self.region}/{self.subregion}"
        )

    def remote_key(self, relative: str) -> str:
        """The full S3 key for a path relative to the version directory.

        ``remote_key("2018/AUS/P1U/profiles/x.csv")`` -> the complete key. The
        version prefix is included because a key without it does not resolve in
        the bucket, and a locator that cannot be fetched is not a locator.
        """
        return f"SMART-DS/{self.version}/{relative}"

    def source_url(self) -> str:
        """The authoritative base URL for this selection, ready to concatenate."""
        return f"{OEDI_BUCKET_URL}SMART-DS/{self.version}/"

    def describe(self) -> dict[str, str]:
        return {
            "version": self.version,
            "year": self.year,
            "region": self.region,
            "subregion": self.subregion,
            "scenario": self.scenario,
            "substation": self.substation,
            "feeder": self.feeder,
        }

    def __str__(self) -> str:
        return (
            f"SmartDsLayout({self.region}/{self.subregion} {self.year} "
            f"{self.scenario} {self.substation}/{self.feeder})"
        )
