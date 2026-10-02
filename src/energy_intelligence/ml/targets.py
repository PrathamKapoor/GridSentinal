"""Forecasting targets: what SMART-DS actually supports, and what it does not.

This module is the Phase 4 answer to the target question, and it is written
*before* any model, because choosing a target from what a dataset contains is the
only way to avoid inventing a research problem.

Each candidate is described by the fields the phase brief requires:

``Target`` ``Source`` ``Unit`` ``Resolution`` ``Coverage`` ``Missingness``
``Asset granularity`` ``Forecast feasibility`` ``Status``

Statuses are deliberately narrow:

``SUPPORTED``
    A real, complete time series exists at the native resolution.
``PARTIALLY_SUPPORTED``
    A real series exists but it is not the quantity we would ideally forecast
    (a different location, a coarser granularity, or missing a dimension).
``UNSUPPORTED``
    No series exists. The absence is recorded, never filled.

Every entry carries ``evidence`` naming the file and the count that was actually
observed, because a status without evidence is an assertion.

Extraction
----------
:func:`load_target`, :func:`feeder_load_target` and :func:`solar_target` read the
*same* real files the Phase 3 adapter reads. Nothing here re-implements SMART-DS
semantics: customer series come from
:meth:`~energy_intelligence.data.smartds.adapter.SmartDsAdapter.loads`, and the
time axis comes from the Phase 3 :class:`~energy_intelligence.data.smartds.TimeAxis`,
so Phase 4 cannot silently introduce a second resampling or derivation rule.

Values keep the Phase 3 fidelity vocabulary: load series are ``DERIVED`` (rated kW
times a per-unit profile), the solar series is ``OBSERVED`` (it is a published
column of a real file), and every sample carries the provenance of its source.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..data.smartds.adapter import SmartDsAdapter
from ..data.smartds.layout import SmartDsLayout
from ..domain.enums import Unit
from ..domain.provenance import Provenance, ProvenanceEvent, SourceReference
from ..domain.quality import DataQuality
from ..domain.enums import QualityFlag

__all__ = [
    "TargetStatus",
    "TargetSpec",
    "TARGET_REGISTRY",
    "TargetSeries",
    "solar_data_file",
    "load_target",
    "feeder_load_target",
    "solar_target",
    "target_spec",
    "supported_targets",
    "select_customer_sample",
]


class TargetStatus:
    """The three allowed target statuses, as plain strings for serialisation."""

    SUPPORTED = "SUPPORTED"
    PARTIALLY_SUPPORTED = "PARTIALLY_SUPPORTED"
    UNSUPPORTED = "UNSUPPORTED"


@dataclass(frozen=True, slots=True)
class TargetSpec:
    """One forecasting candidate and its measured properties."""

    target_id: str
    description: str
    source: str
    unit: str
    resolution_minutes: int | None
    coverage: str
    missingness: str
    asset_granularity: str
    forecast_feasibility: str
    status: str
    evidence: str

    def to_dict(self) -> dict[str, object]:
        return {
            "target_id": self.target_id,
            "description": self.description,
            "source": self.source,
            "unit": self.unit,
            "resolution_minutes": self.resolution_minutes,
            "coverage": self.coverage,
            "missingness": self.missingness,
            "asset_granularity": self.asset_granularity,
            "forecast_feasibility": self.forecast_feasibility,
            "status": self.status,
            "evidence": self.evidence,
        }


#: Every candidate considered in Phase 4, including the ones that do not exist.
TARGET_REGISTRY: tuple[TargetSpec, ...] = (
    TargetSpec(
        target_id="customer_load",
        description="Individual customer real power demand.",
        source="Loads.dss kW rating x profiles/<customer>_pu.csv (Phase 3 derivation)",
        unit="kW",
        resolution_minutes=15,
        coverage="35040 points = 365 days x 96 steps, 2018, complete",
        missingness="none observed; every profile file is present and full length",
        asset_granularity="per customer (1871 customers: 1819 centre-tap pairs + 52 singles)",
        forecast_feasibility=(
            "high: strong daily and weekly seasonality, no missing intervals, and the "
            "Phase 3 reconstruction reproduces the dataset's own published figures"
        ),
        status=TargetStatus.SUPPORTED,
        evidence=(
            "1871 customers resolved from 3690 Load objects; profiles/ holds 341 CSV "
            "files of 35040 values each; residential reconstruction matches the "
            "published peak to 8 decimal places"
        ),
    ),
    TargetSpec(
        target_id="feeder_load",
        description="Feeder-level total real power demand (all customers summed).",
        source="Sum over every Load object of kW x profile (Phase 3 derivation)",
        unit="kW",
        resolution_minutes=15,
        coverage="35040 points = 365 days x 96 steps, 2018, complete",
        missingness="none observed",
        asset_granularity="one series per feeder",
        forecast_feasibility=(
            "high, but smoother than the individual series: aggregation averages out "
            "customer-level noise, so it is the easier of the two load targets"
        ),
        status=TargetStatus.SUPPORTED,
        evidence=(
            "reconstructed peak 15070.2676 kW at grid index 19553 "
            "(2018-07-23T16:15Z) against the dataset's published 15090.7182 kW; the "
            "-20.4506 kW difference is the Phase 3 balance residual and is retained"
        ),
    ),
    TargetSpec(
        target_id="pv_generation",
        description="Realised DC generation of a 1000 kW photovoltaic array.",
        source=(
            "solar_data/AUS_30.3459_-97.8095_25_180_full.csv column "
            "'kW Generated (1000 kW Array)'"
        ),
        unit="kW",
        resolution_minutes=15,
        coverage="35040 points = 365 days x 96 steps, 2018, complete",
        missingness="none; 52.2% of values are exactly 0.0 (night), which is physical, not missing",
        asset_granularity=(
            "one series, for ONE array at one location/orientation "
            "(30.3459 N, 97.8095 W, tilt 25 deg, azimuth 180 deg), scaled to 1000 kW"
        ),
        forecast_feasibility=(
            "high and physically meaningful: a deterministic daily cycle with real "
            "weather covariates (GHI, PoA irradiance, DNI, DHI, temperature, wind) "
            "shipped alongside it in the same file"
        ),
        status=TargetStatus.PARTIALLY_SUPPORTED,
        evidence=(
            "35040 rows with columns DNI, DHI, Wind Speed, Temperature, GHI, "
            "PoA Irradiance, kW Generated; range 0.0-834.05 kW, mean 159.73 kW. It is "
            "NOT the feeder PV fleet: the 1216 PVSystem objects reference 20 irradiance "
            "loadshapes (AUS_30.4196_-97.8095_<tilt>_<azimuth>) that are absent from "
            "the dataset, so feeder PV output is NOT derivable (G-03)"
        ),
    ),
    TargetSpec(
        target_id="net_load",
        description="Grid net load = demand minus local renewable generation.",
        source="would require customer/feeder load AND that feeder's PV generation",
        unit="kW",
        resolution_minutes=None,
        coverage="NOT AVAILABLE",
        missingness="n/a",
        asset_granularity="n/a",
        forecast_feasibility=(
            "UNSUPPORTED for this feeder. The base_timeseries scenario has no PV at "
            "all, and the PV scenarios' irradiance shapes are absent, so combining our "
            "load series with the one shipped solar file would fabricate a net load for "
            "an array that is not on this feeder"
        ),
        status=TargetStatus.UNSUPPORTED,
        evidence=(
            "base_timeseries contains no PVSystems.dss; the 20 referenced irradiance "
            "loadshapes have no backing file anywhere in the dataset"
        ),
    ),
    TargetSpec(
        target_id="battery_soc",
        description="Battery state of charge, and therefore charge/discharge power.",
        source="Storage.dss: kWhStored is a single scalar, not a series",
        unit="fraction / kWh",
        resolution_minutes=None,
        coverage="NOT AVAILABLE",
        missingness="n/a",
        asset_granularity="would be per battery (93 exist)",
        forecast_feasibility=(
            "UNSUPPORTED and underivable: no SOC(t), no dispatch column, no storage "
            "loadshape. Differencing a single scalar yields nothing (G-01)"
        ),
        status=TargetStatus.UNSUPPORTED,
        evidence=(
            "93/93 Storage objects inspected: State=IDLING, kWhStored=8.0 exactly once, "
            "no SOC column anywhere in Storage.dss or LoadShapes.dss"
        ),
    ),
    TargetSpec(
        target_id="battery_power",
        description="Battery charge or discharge power.",
        source="none",
        unit="kW",
        resolution_minutes=None,
        coverage="NOT AVAILABLE",
        missingness="n/a",
        asset_granularity="would be per battery",
        forecast_feasibility="UNSUPPORTED: see battery_soc",
        status=TargetStatus.UNSUPPORTED,
        evidence="same 93 Storage objects; no dispatch and no directional limit published",
    ),
    TargetSpec(
        target_id="ev_demand",
        description="EV charging demand.",
        source="placements/ev_residential=L.json (locations only)",
        unit="kW",
        resolution_minutes=None,
        coverage="NOT AVAILABLE",
        missingness="n/a",
        asset_granularity="would be per charger",
        forecast_feasibility=(
            "UNSUPPORTED: 2991 adoption sites are named but no charging profile, power "
            "or session series exists (G-04)"
        ),
        status=TargetStatus.UNSUPPORTED,
        evidence="placement JSON carries element names only; no time series file exists",
    ),
    TargetSpec(
        target_id="wind_generation",
        description="Wind generation.",
        source="none",
        unit="kW",
        resolution_minutes=None,
        coverage="NOT AVAILABLE",
        missingness="n/a",
        asset_granularity="n/a",
        forecast_feasibility=(
            "UNSUPPORTED: SMART-DS v1.0 contains no wind assets. Wind speed appears in "
            "the solar file as a weather covariate, but no turbine exists to convert it"
        ),
        status=TargetStatus.UNSUPPORTED,
        evidence="no wind asset in any acquired scenario; metrics.csv reports 0 wind capacity",
    ),
)


def target_spec(target_id: str) -> TargetSpec:
    """Look up a target by id.

    Raises:
        KeyError: If the id is not in the registry, so an experiment can never
            silently run against a target nobody declared.
    """
    for spec in TARGET_REGISTRY:
        if spec.target_id == target_id:
            return spec
    raise KeyError(
        f"unknown target {target_id!r}; declared targets: "
        f"{[spec.target_id for spec in TARGET_REGISTRY]}"
    )


def supported_targets() -> tuple[TargetSpec, ...]:
    """Targets that are at least partially supported, i.e. usable in Phase 4."""
    return tuple(
        spec
        for spec in TARGET_REGISTRY
        if spec.status != TargetStatus.UNSUPPORTED
    )


@dataclass(frozen=True, slots=True)
class TargetSeries:
    """One extracted target: a matrix of series on the native time axis.

    Attributes:
        spec: The declared target this series realises.
        values: ``(n_series, n_steps)`` in ``spec.unit``. No resampling, ever.
        series_ids: Stable identifier per row of ``values``.
        origin: ISO-8601 instant of index 0.
        timestep_minutes: Native step, 15.
        provenance: Where the bytes came from.
        quality: Quality metadata carried through from Phase 3.
        value_origin: ``DERIVED`` or ``OBSERVED``, never conflated.
        scale: Per-series normalising quantity, in ``spec.unit``. For load this is
            the customer's rated kW; for PV it is the array rating the column name
            states. It is a **physical quantity known for the whole horizon**, not a
            fitted statistic, so dividing by it leaks nothing.
        scale_source: Where ``scale`` came from, for the manifest.
        notes: Free-form provenance notes, e.g. cross-phase consistency checks.
    """

    spec: TargetSpec
    values: np.ndarray
    series_ids: tuple[str, ...]
    origin: str
    timestep_minutes: int
    provenance: Provenance
    quality: DataQuality
    value_origin: str
    scale: np.ndarray | None = None
    scale_source: str = ""
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def series_count(self) -> int:
        return int(self.values.shape[0])

    @property
    def step_count(self) -> int:
        return int(self.values.shape[1])

    @property
    def scale_vector(self) -> np.ndarray:
        """Per-series normalising quantity, defaulting to 1.0 when not applicable."""
        if self.scale is None:
            return np.ones(self.series_count, dtype=np.float64)
        return np.asarray(self.scale, dtype=np.float64)

    def normalised(self) -> np.ndarray:
        """The series in units of :attr:`scale`.

        Per-unit load is how SMART-DS itself stores its profiles, so this is the
        dataset's own convention rather than an imported one. It also removes the
        scale imbalance between a 1.4 kW household and a 388 kW commercial customer,
        which otherwise lets one pooled model ignore the majority of series.
        """
        scale = self.scale_vector
        return self.values / scale[:, None]

    def timestamps(self, count: int | None = None) -> list[str]:
        """ISO-8601 timestamps for the first ``count`` steps (diagnostics only)."""
        from datetime import datetime, timedelta, timezone

        start = datetime.fromisoformat(self.origin)
        limit = self.step_count if count is None else count
        return [
            (start + timedelta(minutes=self.timestep_minutes * i)).isoformat()
            for i in range(limit)
        ]

    def to_dict(self) -> dict[str, object]:
        return {
            "target_id": self.spec.target_id,
            "unit": self.spec.unit,
            "value_origin": self.value_origin,
            "status": self.spec.status,
            "series_count": self.series_count,
            "step_count": self.step_count,
            "timestep_minutes": self.timestep_minutes,
            "origin": self.origin,
            "min": float(self.values.min()) if self.values.size else None,
            "max": float(self.values.max()) if self.values.size else None,
            "mean": float(self.values.mean()) if self.values.size else None,
            "zeros": int(np.count_nonzero(self.values == 0.0)),
            "source": self.provenance.source.locator,
            "notes": list(self.notes),
        }


def _provenance(layout: SmartDsLayout, locator: str, event: str, origin: str) -> Provenance:
    return Provenance(
        source=SourceReference(
            source_id=f"smartds-{layout.region}-{layout.subregion}",
            dataset="ml_target",
            locator=f"{layout.version}/{locator}",
            version=layout.version.lstrip("v"),
        ),
        events=(ProvenanceEvent(timestamp=origin, event=event, detail=layout.scenario),),
    )


def load_target(adapter: SmartDsAdapter, names: tuple[str, ...]) -> TargetSeries:
    """Extract whole-year series for the named customers.

    Args:
        adapter: Phase 3 adapter over the acquired dataset.
        names: Base customer names to extract. An explicit subset is required:
            materialising all 1,871 customers for a year is 65.6M values, which is a
            scale problem, not a baseline problem (D-058).

    Returns:
        A :class:`TargetSeries` with one row per requested customer.
    """
    if not names:
        raise ValueError("load_target requires at least one customer name")

    axis = adapter.time_axis()
    catalogue = adapter.loads(indices=None, names=tuple(sorted(names)))
    missing = sorted(set(names) - {customer.name for customer in catalogue.customers})
    if missing:
        raise ValueError(
            f"requested customers could not be resolved from Loads.dss: {missing[:5]}"
            + (" ..." if len(missing) > 5 else "")
        )

    values = np.asarray([customer.series for customer in catalogue.customers], dtype=np.float64)
    rated = np.asarray([customer.rated_kw for customer in catalogue.customers], dtype=np.float64)
    layout = adapter.layout
    return TargetSeries(
        spec=target_spec("customer_load"),
        values=values,
        series_ids=tuple(customer.name for customer in catalogue.customers),
        origin=axis.timestamp(0).isoformat(),
        timestep_minutes=axis.timestep_minutes,
        provenance=_provenance(
            layout,
            f"{layout.year}/{layout.region}/{layout.subregion}/scenarios/"
            f"{layout.scenario}/opendss/{layout.substation}/{layout.feeder}",
            "customer_series_derived_from_rating_and_profile",
            axis.timestamp(0).isoformat(),
        ),
        quality=DataQuality(flags=frozenset({QualityFlag.OK})),
        value_origin="DERIVED",
        scale=rated,
        scale_source=(
            "each customer's summed kW rating from Loads.dss; a physical quantity "
            "known for the whole forecast horizon, so per-unit scaling leaks nothing"
        ),
        notes=(
            "P(t) = sum over the customer's Load objects of (kW x profile_pu(t)); the "
            "Phase 3 derivation, unchanged",
            f"{values.shape[0]} customers x {values.shape[1]} steps",
        ),
    )


def feeder_load_target(adapter: SmartDsAdapter) -> TargetSeries:
    """Extract the whole-year feeder-total demand series.

    This is the same arithmetic Phase 3 performs per customer, accumulated per
    *profile* instead of per customer so that all 3,690 Load objects can be summed
    without holding 1,871 whole-year series in memory. The cross-phase consistency
    note records the check that proves the two agree.
    """
    from ..data.smartds.normalize import read_profile

    layout = adapter.layout
    axis = adapter.time_axis()
    state = adapter.parsed()

    rated_by_profile: dict[str, float] = {}
    for obj in state.loads:
        shape = obj.get("yearly")
        if not shape:
            continue
        filename = adapter.profile_filename(shape)
        if filename is None:
            continue
        rated_by_profile[filename] = rated_by_profile.get(filename, 0.0) + (
            obj.get_float("kW", 0.0) or 0.0
        )

    if not rated_by_profile:
        raise ValueError("no Load object resolved to a profile; cannot build a feeder series")

    total = np.zeros(axis.points, dtype=np.float64)
    for filename, rated in sorted(rated_by_profile.items()):
        path = layout.profiles_dir / filename
        if not path.is_file():
            continue
        values = np.asarray(read_profile(path), dtype=np.float64)
        usable = min(values.size, axis.points)
        if usable < axis.points:
            raise ValueError(
                f"profile {filename} has {values.size} points but the declared grid "
                f"has {axis.points}; padding would be an invented value"
            )
        total += rated * values

    return TargetSeries(
        spec=target_spec("feeder_load"),
        values=total.reshape(1, -1),
        series_ids=(f"{layout.substation}/{layout.feeder}",),
        scale=np.asarray([float(sum(rated_by_profile.values()))]),
        scale_source="sum of every connected Load object's kW rating",
        origin=axis.timestamp(0).isoformat(),
        timestep_minutes=axis.timestep_minutes,
        provenance=_provenance(
            layout,
            f"{layout.year}/{layout.region}/{layout.subregion}/scenarios/"
            f"{layout.scenario}/opendss/{layout.substation}/{layout.feeder}/Loads.dss",
            "feeder_series_summed_over_all_load_objects",
            axis.timestamp(0).isoformat(),
        ),
        quality=DataQuality(flags=frozenset({QualityFlag.OK})),
        value_origin="DERIVED",
        notes=(
            "sum over all Load objects of (kW x profile_pu(t)), accumulated per profile",
            "cross-phase check: index 19553 must equal the Phase 3 reconstruction "
            "15070.2676 kW (test asserts agreement to 1e-6)",
        ),
    )


def solar_data_file(layout: SmartDsLayout) -> Path:
    """The single solar file shipped with the acquired sub-region.

    Returns:
        The path of the one ``solar_data/*.csv`` present.

    Raises:
        FileNotFoundError: If none is present, so an experiment can never silently
            substitute another location's file.
    """
    candidates = sorted(layout.solar_data_dir.glob("*.csv"))
    if not candidates:
        raise FileNotFoundError(
            f"no solar_data/*.csv under {layout.solar_data_dir}; the dataset ships "
            "none for this sub-region and PV must not be fabricated"
        )
    if len(candidates) > 1:
        raise FileNotFoundError(
            f"{len(candidates)} solar files present ({[c.name for c in candidates]}); "
            "Phase 4 supports exactly one and will not guess which to use"
        )
    return candidates[0]


#: Column of the solar file that is the realised generation of a 1000 kW array.
SOLAR_GENERATION_COLUMN = "kW Generated (1000 kW Array)"

#: Nameplate of the array the column reports, stated in the column name itself.
PV_ARRAY_RATING_KW = 1000.0

#: Real weather covariates shipped in the same file. Names are the file's own.
SOLAR_WEATHER_COLUMNS: tuple[str, ...] = (
    "DNI",
    "DHI",
    "GHI",
    "PoA Irradiance (W/m^2)",
    "Wind Speed",
    "Temperature",
)


def solar_target(layout: SmartDsLayout) -> tuple[TargetSeries, dict[str, np.ndarray]]:
    """Extract the shipped PV generation series and its weather covariates.

    Returns:
        The generation :class:`TargetSeries` and a mapping of weather column name to
        its own array, all on the identical native time axis.

    Raises:
        FileNotFoundError: If the file or the generation column is absent.
        ValueError: If the row count does not match the declared 35,040-point grid,
            because padding or truncating would silently misalign every timestamp.
    """
    path = solar_data_file(layout)
    generation: list[float] = []
    weather: dict[str, list[float]] = {name: [] for name in SOLAR_WEATHER_COLUMNS}
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing_columns = [
            name for name in (SOLAR_GENERATION_COLUMN, *SOLAR_WEATHER_COLUMNS)
            if name not in (reader.fieldnames or ())
        ]
        if missing_columns:
            raise ValueError(
                f"{path.name} is missing required columns {missing_columns}; the "
                "target must not be reconstructed from other columns"
            )
        for row in reader:
            generation.append(float(row[SOLAR_GENERATION_COLUMN]))
            for name in SOLAR_WEATHER_COLUMNS:
                weather[name].append(float(row[name]))

    from datetime import datetime, timezone

    values = np.asarray(generation, dtype=np.float64).reshape(1, -1)
    expected = 365 * 96
    if values.shape[1] != expected:
        raise ValueError(
            f"{path.name} has {values.shape[1]} rows, expected {expected} for a "
            "15-minute 365-day grid; interpolation is not permitted"
        )

    weather_arrays = {name: np.asarray(col, dtype=np.float64) for name, col in weather.items()}
    series = TargetSeries(
        spec=target_spec("pv_generation"),
        values=values,
        series_ids=(path.stem,),
        scale=np.asarray([PV_ARRAY_RATING_KW]),
        scale_source=(
            "the 1000 kW array rating stated in the column name "
            "'kW Generated (1000 kW Array)'"
        ),
        origin=datetime(2018, 1, 1, tzinfo=timezone.utc).isoformat(),
        timestep_minutes=15,
        provenance=_provenance(
            layout,
            f"{layout.year}/{layout.region}/{layout.subregion}/solar_data/{path.name}",
            "pv_generation_column_read",
            datetime(2018, 1, 1, tzinfo=timezone.utc).isoformat(),
        ),
        quality=DataQuality(
            flags=frozenset({QualityFlag.OK}),
            detail=(
                "published column of a real SMART-DS file; exact zeros are physical "
                "night-time values, not missing data"
            ),
        ),
        value_origin="OBSERVED",
        notes=(
            f"one 1000 kW array at {path.stem.split('_', 1)[1]}; NOT the feeder PV fleet",
            f"{float(values.min()):.3f}-{float(values.max()):.3f} kW, "
            f"{float(values.mean()):.3f} kW mean, "
            f"{100.0 * float(np.mean(values == 0.0)):.1f}% exact zeros",
        ),
    )
    return series, weather_arrays


def select_customer_sample(
    adapter: SmartDsAdapter,
    *,
    count: int,
    seed: int,
    commercial_fraction: float = 0.25,
) -> tuple[tuple[str, ...], dict[str, object]]:
    """Choose a reproducible, stratified subset of customers.

    Sampling the first N customer names in sorted order would take a contiguous run
    of one neighbourhood, which biases both class mix and rated power. This selects
    **stratified by class and rated-power decile**, deterministically from ``seed``,
    and reports the composition so the sample's limits are visible in the dataset
    manifest rather than implied.

    Args:
        adapter: Adapter over the acquired feeder.
        count: Total customers to select.
        seed: Seed for the deterministic selection.
        commercial_fraction: Share of the sample that must be commercial.

    Returns:
        The selected base customer names, sorted, plus a composition report.

    Raises:
        ValueError: If ``count`` exceeds the number of available customers.
    """
    catalogue = adapter.loads(indices=(0,))
    customers = catalogue.customers
    if count > len(customers):
        raise ValueError(
            f"requested {count} customers but only {len(customers)} exist in this feeder"
        )

    commercial = [c for c in customers if c.is_commercial]
    residential = [c for c in customers if not c.is_commercial]
    wanted_commercial = min(len(commercial), int(round(count * commercial_fraction)))
    wanted_residential = min(len(residential), count - wanted_commercial)

    rng = np.random.default_rng(seed)

    def pick(pool: list, how_many: int) -> list:
        """Take ``how_many`` spread across rated-power deciles, deterministically."""
        if how_many <= 0:
            return []
        ordered = sorted(pool, key=lambda c: (c.rated_kw, c.name))
        strata = np.array_split(np.arange(len(ordered)), 10)
        per_stratum = max(1, how_many // 10)
        chosen: list = []
        for stratum in strata:
            if len(chosen) >= how_many or stratum.size == 0:
                break
            take = min(per_stratum, how_many - len(chosen), stratum.size)
            picked = rng.choice(stratum, size=take, replace=False)
            chosen.extend(ordered[int(i)] for i in sorted(picked))
        if len(chosen) < how_many:  # top up from what is left, still deterministically
            remaining = [c for c in ordered if c not in chosen]
            extra = rng.choice(
                len(remaining), size=how_many - len(chosen), replace=False
            )
            chosen.extend(remaining[int(i)] for i in sorted(extra))
        return chosen

    selected = pick(commercial, wanted_commercial) + pick(residential, wanted_residential)
    selected.sort(key=lambda c: c.name)
    report = {
        "requested": count,
        "selected": len(selected),
        "seed": seed,
        "commercial_requested_fraction": commercial_fraction,
        "commercial_available": len(commercial),
        "residential_available": len(residential),
        "selected_commercial": sum(1 for c in selected if c.is_commercial),
        "selected_residential": sum(1 for c in selected if not c.is_commercial),
        "selected_rated_kw_min": min(c.rated_kw for c in selected),
        "selected_rated_kw_max": max(c.rated_kw for c in selected),
        "population_rated_kw_min": min(c.rated_kw for c in customers),
        "population_rated_kw_max": max(c.rated_kw for c in customers),
        "method": "stratified by class and rated-power decile, numpy default_rng(seed)",
    }
    return tuple(c.name for c in selected), report


def _profile_filename(reference: str | None) -> str | None:
    """Reduce a ``file=...`` loadshape reference to its basename.

    Retained for callers holding a raw reference; the adapter already resolves
    loadshape names, and Phase 4 uses
    :meth:`~energy_intelligence.data.smartds.adapter.SmartDsAdapter.profile_filename`
    so the convention is implemented once.
    """
    if not reference:
        return None
    start = reference.find("file=")
    if start < 0:
        return None
    rest = reference[start + 5 :]
    close = rest.find(")")
    token = (rest[:close] if close >= 0 else rest.split()[0]).strip()
    return Path(token).name or None


def unit_for(target_id: str) -> Unit:
    """The Phase 2 unit a target is measured in, for provenance records."""
    mapping = {"customer_load": Unit.KILOWATT, "feeder_load": Unit.KILOWATT}
    return mapping.get(target_id, Unit.KILOWATT)