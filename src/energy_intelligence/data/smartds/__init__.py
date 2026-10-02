"""SMART-DS adapter package.

The only layer that knows SMART-DS's file layout, column names and semantics.
Nothing above this package refers to a SMART-DS identifier; a future dataset
would add a sibling adapter and reuse the domain layer unchanged.

Modules
-------

``layout``       dataset paths and identifier sanitisation
``dss``          generic OpenDSS directive parser (dataset-agnostic)
``buscoords``    the bare coordinate-list format that ``.dss`` parsing cannot read
``schema``       programmatic schema discovery
``mapping``      semantic field mapping with explicit confidence
``normalize``    per-unit to kilowatts, index to timestamp, time axis
``adapter``      facade composing parse + normalise + quality
``domain_mapper`` normalised records to Phase 2 domain objects
``balance``      energy balance validation against a fixed tolerance
``pipeline``     end-to-end orchestration and artifact production
"""

from __future__ import annotations

from .adapter import (
    BatteryAssetSpec,
    DerAssetCatalogue,
    LoadCatalogue,
    QualityFinding,
    QualityReport,
    SmartDsAdapter,
    TopologyCatalogue,
)
from .balance import (
    TOLERANCE_KW,
    BalanceReport,
    BalanceSample,
    PublishedSummary,
    read_summary_data,
    validate_feeder_balance,
)
from .buscoords import BusCoordinate, load_buscoords, parse_buscoords
from .domain_mapper import (
    DomainMappingResult,
    MappingReport,
    UnresolvedBoundaryError,
    map_to_domain,
)
from .dss import DssObject, DssParseError, parse_dss
from .layout import (
    KNOWN_SCENARIOS,
    OEDI_BUCKET_URL,
    REGIONS,
    SMART_DS_VERSION,
    SmartDsLayout,
    sanitize_identifier,
)
from .mapping import (
    SMART_DS_MAPPINGS,
    FieldMapping,
    MappingConfidence,
    ValueOrigin,
    mapping_table,
)
from .normalize import (
    GRID_POINTS_365_DAYS,
    TIMEZONE_ASSUMPTION,
    CustomerLoad,
    TimeAxis,
    build_time_axis,
    read_profile,
)
from .pipeline import IngestionResult, FILE_ROLES, build_manifest, run_ingestion
from .schema import (
    ColumnProfile,
    CsvSchema,
    DssSchema,
    FeederSchema,
    ProfileSchema,
    discover_buscoords_schema,
    discover_csv_schema,
    discover_dss_schema,
    discover_feeder_schema,
    discover_profile_schema,
)

__all__ = [
    # layout
    "OEDI_BUCKET_URL",
    "SMART_DS_VERSION",
    "REGIONS",
    "KNOWN_SCENARIOS",
    "SmartDsLayout",
    "sanitize_identifier",
    # dss
    "DssObject",
    "DssParseError",
    "parse_dss",
    "BusCoordinate",
    "parse_buscoords",
    "load_buscoords",
    # schema
    "ColumnProfile",
    "CsvSchema",
    "DssSchema",
    "ProfileSchema",
    "FeederSchema",
    "discover_csv_schema",
    "discover_dss_schema",
    "discover_profile_schema",
    "discover_buscoords_schema",
    "discover_feeder_schema",
    # mapping
    "FieldMapping",
    "MappingConfidence",
    "ValueOrigin",
    "SMART_DS_MAPPINGS",
    "mapping_table",
    # normalize
    "TimeAxis",
    "CustomerLoad",
    "build_time_axis",
    "read_profile",
    "GRID_POINTS_365_DAYS",
    "TIMEZONE_ASSUMPTION",
    # adapter
    "SmartDsAdapter",
    "LoadCatalogue",
    "TopologyCatalogue",
    "DerAssetCatalogue",
    "BatteryAssetSpec",
    "QualityFinding",
    "QualityReport",
    # domain mapping
    "DomainMappingResult",
    "MappingReport",
    "UnresolvedBoundaryError",
    "map_to_domain",
    # balance
    "TOLERANCE_KW",
    "BalanceSample",
    "BalanceReport",
    "PublishedSummary",
    "read_summary_data",
    "validate_feeder_balance",
    # pipeline
    "IngestionResult",
    "build_manifest",
    "run_ingestion",
    "FILE_ROLES",
]
