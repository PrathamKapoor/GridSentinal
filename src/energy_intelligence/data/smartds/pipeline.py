"""End-to-end SMART-DS ingestion pipeline.

Composes the adapter stages into one reproducible run and produces every
Phase 3 artifact: manifest, schema report, mapping report, quality report and
balance validation report.

```text
SmartDsLayout
      |
      v
SmartDsAdapter ......... parse .dss, discover schema, build time axis
      |
      +--> LoadCatalogue ........ customer-grouped kW series
      +--> TopologyCatalogue .... node graph + coordinates
      +--> DerAssetCatalogue .... PV / battery / EV discovery
      |
      v
map_to_domain() ......... Phase 2 EnergySystem
      |
      v
validate_feeder_balance() ... balance report against a FIXED tolerance
```

Every stage is separately callable and separately testable; the pipeline only
sequences them.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ...domain.quality import DataQuality
from ...domain.enums import QualityFlag
from ..manifest import (
    AcquisitionMetadata,
    DatasetManifest,
    FileEntry,
)
from .adapter import SmartDsAdapter
from .balance import (
    BalanceReport,
    PublishedSummary,
    read_summary_data,
    validate_feeder_balance,
)
from .domain_mapper import DomainMappingResult, MappingReport, map_to_domain
from .layout import SmartDsLayout

__all__ = ["IngestionResult", "run_ingestion", "build_manifest", "FILE_ROLES"]

#: Files the pipeline considers part of the acquired subset, with their role.
#: Roles feed the manifest so a reader can tell what each file is for.
FILE_ROLES: dict[str, str] = {
    "User_Guide/Readme.md": "documentation",
    "metrics.csv": "feeder_metrics",
    "Master.dss": "network_model",
    "Loads.dss": "loads",
    "LoadShapes.dss": "loadshapes",
    "Lines.dss": "network_model",
    "LineCodes.dss": "network_model",
    "Transformers.dss": "network_model",
    "Buscoords.dss": "coordinates",
    "Capacitors.dss": "network_model",
    "PVSystems.dss": "solar",
    "Storage.dss": "storage",
    "analysis/Summary_data.csv": "published_summary",
}


@dataclass(frozen=True, slots=True)
class IngestionResult:
    """Everything one ingestion run produced."""

    layout: SmartDsLayout
    manifest: DatasetManifest
    schema: dict[str, object]
    quality: dict[str, object]
    mapping: MappingReport
    mapping_result: DomainMappingResult
    balance: BalanceReport
    reconstruction: dict[str, float]


def build_manifest(
    layout: SmartDsLayout,
    *,
    acquired_at: str,
    source_url: str,
    tool: str = "",
) -> DatasetManifest:
    """Walk the acquired subset and build a checksummed manifest.

    Only files that exist are recorded; a missing file is reported by the schema
    report rather than invented here.
    """
    from ..manifest import compute_sha256

    entries: list[FileEntry] = []
    roots = [
        (layout.subregion_dir, layout.subregion_dir),
        (layout.version_dir / "User_Guide", layout.version_dir),
        (layout.version_dir / "placements", layout.version_dir),
    ]
    seen: set[Path] = set()
    for base, _ in roots:
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*")):
            if not path.is_file() or path in seen:
                continue
            seen.add(path)
            relative = path.relative_to(layout.version_dir).as_posix()
            role = next(
                (value for name, value in FILE_ROLES.items() if relative.endswith(name)),
                "",
            )
            entries.append(
                FileEntry(
                    remote_key=layout.remote_key(relative),
                    size_bytes=path.stat().st_size,
                    sha256=compute_sha256(path),
                    role=role or "unclassified",
                )
            )

    entries.sort(key=lambda item: item.remote_key)
    subset = (
        f"{layout.region}/{layout.subregion} {layout.year} {layout.scenario} "
        f"{layout.substation}/{layout.feeder}"
    )
    return DatasetManifest(
        dataset_name="SMART-DS",
        dataset_version=layout.version.lstrip("v"),
        subset=subset,
        acquisition=AcquisitionMetadata(
            acquired_at=acquired_at,
            source=source_url,
            method="https_get",
            tool=tool,
        ),
        files=tuple(entries),
        local_root=str(layout.subregion_dir),
    )


def run_ingestion(
    layout: SmartDsLayout,
    *,
    peak_only: bool = True,
) -> IngestionResult:
    """Run the full pipeline for one feeder.

    Args:
        layout: Resolved dataset locations.
        peak_only: When True, only the peak timepoint is materialised, which is
            what balance validation needs. When False, whole-year series are
            built for every customer.

    Returns:
        The complete :class:`IngestionResult`.

    Raises:
        DssParseError: If the dataset cannot be parsed faithfully.
        DomainValidationError: If the constructed domain objects are inconsistent.
    """
    adapter = SmartDsAdapter(layout)
    axis = adapter.time_axis()

    summary_path = layout.feeder_analysis_dir / "Summary_data.csv"
    summary: PublishedSummary | None = None
    if summary_path.is_file():
        summary = read_summary_data(summary_path)
        indices: tuple[int, ...] | None = (summary.peak_index,)
    else:
        indices = (0,)

    loads = adapter.loads(indices=indices)
    topology_catalogue = adapter.topology()
    der_catalogue = adapter.der_assets()

    index = indices[0] if indices else 0
    total_kw = sum(customer.value_at(0) for customer in loads.customers)
    commercial_kw = sum(
        customer.value_at(0)
        for customer in loads.customers
        if customer.members and customer.members[0].get("yearly", "").startswith("com")
    )
    residential_kw = total_kw - commercial_kw

    demand_quality = DataQuality(flags=frozenset({QualityFlag.OK}))

    mapping_result = map_to_domain(
        adapter,
        load_catalogue=loads,
        topology_catalogue=topology_catalogue,
        der_catalogue=der_catalogue,
        demand_kw=total_kw,
        demand_quality=demand_quality,
        summary_customer_kw=summary.customer_power_kw if summary else None,
        event_timestamp=axis.timestamp(index).isoformat(),
    )

    if summary is not None:
        balance = validate_feeder_balance(
            summary=summary,
            reconstructed_customer_kw=total_kw,
            reconstructed_commercial_kw=commercial_kw,
            reconstructed_residential_kw=residential_kw,
            node_count=topology_catalogue.node_count,
            timestamp_count=1,
        )
    else:
        balance = validate_feeder_balance(
            summary=_empty_summary(),
            reconstructed_customer_kw=total_kw,
            node_count=topology_catalogue.node_count,
        )

    manifest = build_manifest(
        layout,
        acquired_at=DatasetManifest.utc_now(),
        source_url=layout.source_url(),
    )
    schema = adapter.schema()
    schema_payload = schema.to_dict()
    schema_payload["time_axis"] = axis.describe()

    # Each discovered DssSchema already knows how to summarise itself.
    schema_summaries = tuple(
        part.to_summary()
        for part in (
            schema.loads,
            schema.lines,
            schema.transformers,
            schema.pv_systems,
            schema.storage,
        )
        if part is not None
    )
    manifest = manifest.with_schema(schema_summaries).with_transformation(
        "phase3-adapter-1.0.0"
    )

    quality_payload = adapter.quality_report().to_dict()
    quality_payload["load_customers"] = loads.customer_count
    quality_payload["load_objects"] = loads.object_count
    quality_payload["center_tap_customers"] = loads.center_tap_count
    quality_payload["total_rated_kw"] = loads.total_rated_kw
    quality_payload["solar_count"] = der_catalogue.solar_count
    quality_payload["battery_count"] = der_catalogue.battery_count
    quality_payload["ev_sites"] = der_catalogue.ev_sites

    return IngestionResult(
        layout=layout,
        manifest=manifest,
        schema=schema_payload,
        quality=quality_payload,
        mapping=mapping_result.report,
        mapping_result=mapping_result,
        balance=balance,
        reconstruction={
            "grid_index": float(index),
            "timestamp": axis.timestamp(index).isoformat(),
            "total_kw": total_kw,
            "commercial_kw": commercial_kw,
            "residential_kw": residential_kw,
        },
    )


def _empty_summary() -> PublishedSummary:
    """A summary of all-unknown values, used when no Summary_data.csv exists."""
    return PublishedSummary(
        losses_kw=float("nan"),
        circuit_power_kw=float("nan"),
        losses_percent=float("nan"),
        customer_power_kw=float("nan"),
        commercial_kw=float("nan"),
        residential_kw=float("nan"),
        peak_day=1,
        peak_hour=0,
        peak_minute=0,
    )
