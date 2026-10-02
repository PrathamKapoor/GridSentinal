"""Data ingestion: raw external energy data into the Phase 2 domain model.

Phase 3 is a reality-validation phase. This package answers one question with
evidence: *can a real dataset be transformed into our domain abstraction without
inventing information or silently changing its meaning?*

Layering
--------

```text
raw SMART-DS files  ->  data.smartds  ->  Phase 2 domain objects
                         (adapter)         (frozen, validated)
```

``data.smartds`` holds every dataset-specific fact. The domain model holds none.
A future dataset adds a sibling adapter package and reuses the domain layer
unchanged.

Guarantees
----------

* No value is invented, defaulted or imputed.
* No timestamp is silently rounded; off-grid input raises.
* Every emitted domain object carries mandatory provenance.
* Anything that could not be established is reported as ``UNKNOWN`` rather than
  substituted.
* Derived values are marked derived and never called observed.
"""

from __future__ import annotations

from .manifest import (
    MANIFEST_VERSION,
    AcquisitionMetadata,
    DatasetManifest,
    FileEntry,
    SchemaSummary,
    compute_sha256,
)
from .smartds import (
    GRID_POINTS_365_DAYS,
    KNOWN_SCENARIOS,
    OEDI_BUCKET_URL,
    REGIONS,
    SMART_DS_MAPPINGS,
    SMART_DS_VERSION,
    TOLERANCE_KW,
    TIMEZONE_ASSUMPTION,
    BalanceReport,
    BusCoordinate,
    ColumnProfile,
    CsvSchema,
    CustomerLoad,
    DerAssetCatalogue,
    DssObject,
    DssParseError,
    DssSchema,
    FieldMapping,
    FeederSchema,
    IngestionResult,
    LoadCatalogue,
    MappingConfidence,
    MappingReport,
    ProfileSchema,
    PublishedSummary,
    QualityFinding,
    QualityReport,
    SmartDsAdapter,
    SmartDsLayout,
    TimeAxis,
    TopologyCatalogue,
    ValueOrigin,
    build_manifest,
    build_time_axis,
    discover_buscoords_schema,
    discover_csv_schema,
    discover_dss_schema,
    discover_feeder_schema,
    discover_profile_schema,
    load_buscoords,
    mapping_table,
    map_to_domain,
    parse_buscoords,
    parse_dss,
    read_profile,
    read_summary_data,
    run_ingestion,
    sanitize_identifier,
    validate_feeder_balance,
)

__all__ = [
    "MANIFEST_VERSION",
    "AcquisitionMetadata",
    "DatasetManifest",
    "FileEntry",
    "SchemaSummary",
    "compute_sha256",
    # smartds re-exports
    "BalanceReport",
    "BusCoordinate",
    "ColumnProfile",
    "CsvSchema",
    "CustomerLoad",
    "DerAssetCatalogue",
    "DssObject",
    "DssParseError",
    "DssSchema",
    "FieldMapping",
    "FeederSchema",
    "IngestionResult",
    "LoadCatalogue",
    "MappingConfidence",
    "MappingReport",
    "ProfileSchema",
    "PublishedSummary",
    "QualityFinding",
    "QualityReport",
    "SmartDsAdapter",
    "SmartDsLayout",
    "TimeAxis",
    "TopologyCatalogue",
    "ValueOrigin",
    "build_manifest",
    "build_time_axis",
    "discover_buscoords_schema",
    "discover_csv_schema",
    "discover_dss_schema",
    "discover_feeder_schema",
    "discover_profile_schema",
    "load_buscoords",
    "mapping_table",
    "map_to_domain",
    "parse_buscoords",
    "parse_dss",
    "read_profile",
    "read_summary_data",
    "run_ingestion",
    "sanitize_identifier",
    "validate_feeder_balance",
    "GRID_POINTS_365_DAYS",
    "KNOWN_SCENARIOS",
    "OEDI_BUCKET_URL",
    "REGIONS",
    "SMART_DS_MAPPINGS",
    "SMART_DS_VERSION",
    "TOLERANCE_KW",
    "TIMEZONE_ASSUMPTION",
]
