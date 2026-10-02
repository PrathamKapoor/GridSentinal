"""Semantic mapping from SMART-DS fields to the Phase 2 energy domain.

Every row here is a claim about what a SMART-DS field means. Claims are graded
by ``MappingConfidence`` and, where they are not ``DIRECT``, the transformation
is stated explicitly so it can be audited or rejected.

Confidence vocabulary
---------------------

``DIRECT``
    The source field already carries the domain quantity, in a compatible unit,
    with no interpretation. Example: ``Total peak time real customer power (kW)``
    is demand in kW.

``DERIVED``
    The domain quantity is computed from the source field by an arithmetic
    relation documented by the authoritative source **and** verified against
    published values. Example: load power ``kW(t) = Loads.dss kW x
    profile_pu(t)``. This is never called "measured".

``PARTIAL``
    Related but not equivalent: some aspect is missing, aggregated away, or
    requires an assumption to complete.

``UNKNOWN``
    Meaning could not be established from the data or documentation.

``UNMAPPED``
    The source field exists and is understood, but the Phase 2 domain has no
    corresponding concept. Recorded so information is not silently dropped.

Nothing is upgraded to ``DIRECT`` on the basis of a similar-looking name. The
centre-tap load question in particular was resolved empirically, not by
assuming the documentation's wording applied.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

__all__ = [
    "MappingConfidence",
    "ValueOrigin",
    "FieldMapping",
    "SMART_DS_MAPPINGS",
    "mappings_for",
    "unmapped_fields",
    "mapping_table",
]


class MappingConfidence(StrEnum):
    """Confidence that a SMART-DS field means what the domain field means."""

    DIRECT = "direct"
    DERIVED = "derived"
    PARTIAL = "partial"
    UNKNOWN = "unknown"
    UNMAPPED = "unmapped"


class ValueOrigin(StrEnum):
    """How a domain value came into being.

    Distinct from :class:`MappingConfidence`: confidence is about the *mapping*,
    origin is about the *value*. A derived value with a verified mapping is still
    ``DERIVED`` in origin.
    """

    OBSERVED = "observed"
    DERIVED = "derived"
    ESTIMATED = "estimated"
    SIMULATED = "simulated"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class FieldMapping:
    """One documented source-to-domain mapping.

    Attributes:
        source: File and field, e.g. ``"Loads.dss:kW"``.
        source_description: What the source field means, per documentation or
            observation.
        source_unit: Unit as published, or ``"UNKNOWN"``.
        domain_field: The Phase 2 concept, or ``"-"`` when unmapped.
        domain_meaning: What the domain field means.
        transformation: The exact arithmetic, or ``"none"`` for a direct map.
        information_loss: What is lost, or ``"none"``.
        confidence: Grading of the mapping.
        origin: How the resulting value comes into being.
        evidence: What this claim rests on.
    """

    source: str
    source_description: str
    source_unit: str
    domain_field: str
    domain_meaning: str
    transformation: str
    information_loss: str
    confidence: MappingConfidence
    origin: ValueOrigin
    evidence: str

    def to_dict(self) -> dict[str, str]:
        return {
            "source": self.source,
            "source_description": self.source_description,
            "source_unit": self.source_unit,
            "domain_field": self.domain_field,
            "domain_meaning": self.domain_meaning,
            "transformation": self.transformation,
            "information_loss": self.information_loss,
            "confidence": self.confidence.value,
            "origin": self.origin.value,
            "evidence": self.evidence,
        }


_EVIDENCE_USER_GUIDE = "SMART-DS User_Guide v1.0 (OEDI User_Guide/Readme.md)"
_EVIDENCE_OBSERVED = "observed directly in the downloaded file"


SMART_DS_MAPPINGS: tuple[FieldMapping, ...] = (
    # ---------------- Demand ----------------
    FieldMapping(
        source="Loads.dss:kW",
        source_description=(
            "For timeseries scenarios, the maximum power drawn by the load at any "
            "time during the year. Scaled by the referenced loadshape multiplier "
            "at each timepoint."
        ),
        source_unit="kW",
        domain_field="load.rated_power_kw",
        domain_meaning="Peak real power the customer can draw over the year.",
        transformation="none (stored as a rating, not a timepoint value)",
        information_loss="none",
        confidence=MappingConfidence.DIRECT,
        origin=ValueOrigin.OBSERVED,
        evidence=_EVIDENCE_USER_GUIDE,
    ),
    FieldMapping(
        source="Loads.dss:kW x profiles/<shape>_pu.csv",
        source_description=(
            "Annual maximum kW multiplied by the per-unit load multiplier at the "
            "timepoint."
        ),
        source_unit="kW",
        domain_field="demand.total_demand_kw (time series)",
        domain_meaning="Real power demand at a timepoint.",
        transformation="P(t) = kW x profile_pu(t)",
        information_loss=(
            "None for demand. Verified: summing over all Load objects reproduces the "
            "published residential peak power exactly (ratio 1.00000000)."
        ),
        confidence=MappingConfidence.DERIVED,
        origin=ValueOrigin.DERIVED,
        evidence=(
            "User_Guide derivation rule, then empirically verified against "
            "Summary_data.csv: residential 12380.5126 kW computed vs 12380.5126 kW "
            "published (error 0.0000 kW)."
        ),
    ),
    FieldMapping(
        source="Loads.dss:kvar",
        source_description="Annual maximum reactive power for the load.",
        source_unit="kvar",
        domain_field="grid.reactive_power_var",
        domain_meaning="Reactive power at the connection.",
        transformation="none",
        information_loss=(
            "Reactive power has no dedicated Phase 2 state field; "
            "GridState does not carry reactive power. See gaps report."
        ),
        confidence=MappingConfidence.PARTIAL,
        origin=ValueOrigin.OBSERVED,
        evidence=_EVIDENCE_USER_GUIDE,
    ),
    FieldMapping(
        source="Loads.dss:model",
        source_description="OpenDSS load model. All observed values are 1 (constant PQ).",
        source_unit="n/a",
        domain_field="-",
        domain_meaning="Phase 2 has no ZIP/power-factor modelling concept.",
        transformation="none",
        information_loss="Constant-PQ assumption of the source is not represented.",
        confidence=MappingConfidence.UNMAPPED,
        origin=ValueOrigin.OBSERVED,
        evidence=f"{_EVIDENCE_OBSERVED}: 3690/3690 loads are model=1",
    ),
    FieldMapping(
        source="Loads.dss:Phases",
        source_description="Phase count: 1 or 3. Observed 3638 single-phase, 52 three-phase.",
        source_unit="count",
        domain_field="asset.metadata.phases",
        domain_meaning="Recorded as asset metadata; Phase 2 has no per-phase concept.",
        transformation="none",
        information_loss="Phase-level power is not modelled by Phase 2.",
        confidence=MappingConfidence.PARTIAL,
        origin=ValueOrigin.OBSERVED,
        evidence=_EVIDENCE_OBSERVED,
    ),
    FieldMapping(
        source="Loads.dss:name suffix _1 / _2",
        source_description=(
            "Centre-tap customers are represented by two Load objects, one per "
            "active line. Both carry the SAME kW, each representing half the "
            "customer on its own line."
        ),
        source_unit="n/a",
        domain_field="load.customer grouping",
        domain_meaning=(
            "One Phase 2 load asset per customer, requiring the pair to be summed."
        ),
        transformation="P_customer(t) = kW_1 x profile(t) + kW_2 x profile(t)",
        information_loss=(
            "The per-phase split is discarded; Phase 2 has no per-phase node."
        ),
        confidence=MappingConfidence.DERIVED,
        origin=ValueOrigin.DERIVED,
        evidence=(
            "Empirically resolved. Deduping the pair gave 9157.14 kW (-39.3%); "
            "summing both members gave 15070.27 kW (-0.14%) against a published "
            "15090.72 kW, with residential matching to 8 decimal places. "
            "The documentation's 'multiply by 0.5' guidance applies to driving an "
            "OpenDSS solve, not to reconstructing customer total demand."
        ),
    ),
    FieldMapping(
        source="Summary_data.csv:Total peak time real customer power (kW)",
        source_description="Published aggregate customer real power at the peak timepoint.",
        source_unit="kW",
        domain_field="demand.total_demand_kw (single timepoint)",
        domain_meaning="Aggregate demand at one instant.",
        transformation="none",
        information_loss="Only one timepoint; not a series.",
        confidence=MappingConfidence.DIRECT,
        origin=ValueOrigin.OBSERVED,
        evidence=_EVIDENCE_OBSERVED,
    ),
    FieldMapping(
        source="Summary_data.csv:Total peak time real losses (kW)",
        source_description="Published real losses in the feeder at the peak timepoint.",
        source_unit="kW",
        domain_field="node_power.unaccounted_kw",
        domain_meaning=(
            "Explicit residual when reported node figures do not close. Phase 2 has "
            "no dedicated network-loss concept (physics deferred, D-023)."
        ),
        transformation="residual = supplied - consumed",
        information_loss=(
            "Losses are a single aggregate value with no per-element or per-node "
            "attribution, and no timeseries."
        ),
        confidence=MappingConfidence.PARTIAL,
        origin=ValueOrigin.OBSERVED,
        evidence=(
            f"{_EVIDENCE_OBSERVED}: 501.672 kW = 3.2175% of customer power."
        ),
    ),
    # ---------------- Renewables ----------------
    FieldMapping(
        source="PVSystems.dss:Pmpp",
        source_description="Nameplate maximum power of the PV installation.",
        source_unit="kW",
        domain_field="solar.capacity_kw",
        domain_meaning="Rated DC capacity of the array.",
        transformation="none",
        information_loss="none",
        confidence=MappingConfidence.DIRECT,
        origin=ValueOrigin.OBSERVED,
        evidence=f"{_EVIDENCE_OBSERVED}: 1216 PVSystem objects in the PV scenario.",
    ),
    FieldMapping(
        source="PVSystems.dss:yearly (irradiance loadshape)",
        source_description=(
            "Reference to a loadshape supplying plane-of-array irradiance. The shape "
            "is defined outside the feeder folder, in the sub-region level "
            "LoadShapes.dss."
        ),
        source_unit="kW/m^2 (profiles) or W/m^2 (solar_data)",
        domain_field="renewable.actual_power",
        domain_meaning="Real power generated by the array.",
        transformation=(
            "P(t) = Pmpp x (irradiance(t) / 1000) using OpenDSS Model=1 scaling"
        ),
        information_loss=(
            "The feeder-level LoadShapes.dss observed contains no AUS_* irradiance "
            "shapes, so the linkage cannot be resolved from the feeder folder alone. "
            "Derivation NOT implemented in Phase 3."
        ),
        confidence=MappingConfidence.PARTIAL,
        origin=ValueOrigin.DERIVED,
        evidence=(
            "User_Guide states profiles hold PoA irradiance in kW/m^2 while "
            "solar_data holds W/m^2; both observed at 15-minute, 35040 rows."
        ),
    ),
    FieldMapping(
        source="solar_data/*.csv:kW Generated (1000 kW Array)",
        source_description=(
            "PVWatts output for a 1 MW reference array at the named tilt/azimuth."
        ),
        source_unit="kW per 1000 kW array",
        domain_field="solar.generation_derivation",
        domain_meaning="A per-configuration generation factor for a reference array.",
        transformation="P(t) = Pmpp x kW_generated_1000(t) / 1000",
        information_loss=(
            "Aggregate weather location, not per-array; arrays sharing a location "
            "and orientation share a factor."
        ),
        confidence=MappingConfidence.PARTIAL,
        origin=ValueOrigin.DERIVED,
        evidence=(
            f"{_EVIDENCE_OBSERVED}: max 834.050 kW per 1000 kW array, "
            "PoA irradiance max 1119.0 W/m^2."
        ),
    ),
    FieldMapping(
        source="solar_data/*.csv:PoA Irradiance (W/m^2)",
        source_description="Plane-of-array irradiance from NSRDB, 15-minute.",
        source_unit="W/m^2",
        domain_field="weather.irradiance",
        domain_meaning="Irradiance incident on the array plane.",
        transformation="none",
        information_loss="none",
        confidence=MappingConfidence.DIRECT,
        origin=ValueOrigin.OBSERVED,
        evidence=f"{_EVIDENCE_OBSERVED}: 35040 rows, range 0.0-1119.0.",
    ),
    FieldMapping(
        source="solar_data/*.csv:DNI, DHI, GHI",
        source_description="Solar irradiance components from NSRDB at half-hour "
        "resolution, interpolated to 15 minutes.",
        source_unit="W/m^2",
        domain_field="weather.irradiance_components",
        domain_meaning="Direct normal, diffuse horizontal and global horizontal.",
        transformation="none",
        information_loss=(
            "Source is half-hourly interpolated to 15 minutes, so the finest real "
            "information is 30 minutes."
        ),
        confidence=MappingConfidence.DIRECT,
        origin=ValueOrigin.OBSERVED,
        evidence=_EVIDENCE_USER_GUIDE,
    ),
    FieldMapping(
        source="solar_data/*.csv:Temperature, Wind Speed",
        source_description="Ambient conditions used by PVWatts.",
        source_unit="degC, m/s",
        domain_field="weather",
        domain_meaning="Ambient weather state.",
        transformation="none",
        information_loss="none",
        confidence=MappingConfidence.DIRECT,
        origin=ValueOrigin.OBSERVED,
        evidence=_EVIDENCE_OBSERVED,
    ),
    # ---------------- Storage ----------------
    FieldMapping(
        source="Storage.dss:kWRated",
        source_description="Rated real power of the battery.",
        source_unit="kW",
        domain_field="battery.rated_power",
        domain_meaning="Nameplate power rating.",
        transformation="none",
        information_loss=(
            "A single rating does not distinguish charge from discharge limits, so "
            "battery.max_charge_power and battery.max_discharge_power are both None "
            "(D-050); SMART-DS publishes no directional limit."
        ),
        confidence=MappingConfidence.DIRECT,
        origin=ValueOrigin.OBSERVED,
        evidence=f"{_EVIDENCE_OBSERVED}: 93 Storage objects, all kWRated=8.0.",
    ),
    FieldMapping(
        source="Storage.dss:kWhRated",
        source_description="Rated energy capacity of the battery.",
        source_unit="kWh",
        domain_field="battery.nameplate_energy",
        domain_meaning="Nameplate energy capacity, not deliverable energy.",
        transformation="none",
        information_loss=(
            "User Guide states capacity is sized at twice the kW rating; no depth of "
            "discharge limit is published, so battery.usable_energy is None (UNKNOWN) "
            "rather than equal to the nameplate (D-050)."
        ),
        confidence=MappingConfidence.PARTIAL,
        origin=ValueOrigin.OBSERVED,
        evidence=f"{_EVIDENCE_OBSERVED}: all kWhRated=16.0 with kWRated=8.0.",
    ),
    FieldMapping(
        source="Storage.dss:kWhStored",
        source_description=(
            "A single initial stored-energy scalar for the OpenDSS model. NOT a time "
            "series."
        ),
        source_unit="kWh",
        domain_field="storage.state_of_charge (initial only)",
        domain_meaning="Fraction of capacity stored.",
        transformation="SOC0 = kWhStored / kWhRated",
        information_loss=(
            "Only one value exists for the whole year. No SOC(t) series, therefore no "
            "power derivative and NO dispatch derivation is possible."
        ),
        confidence=MappingConfidence.PARTIAL,
        origin=ValueOrigin.OBSERVED,
        evidence=(
            f"{_EVIDENCE_OBSERVED}: every Storage object carries kWhStored=8.0 "
            "exactly once; there is no SOC column and no storage loadshape."
        ),
    ),
    FieldMapping(
        source="Storage.dss:%EffCharge / %EffDischarge",
        source_description="Round-trip efficiency components.",
        source_unit="percent",
        domain_field="battery.round_trip_efficiency",
        domain_meaning="Fraction of stored energy recoverable.",
        transformation="eta = (EffCharge/100) x (EffDischarge/100) = 0.9025",
        information_loss=(
            "Components are given, not their product; the product is derived. A cycle "
            "efficiency is not a per-direction efficiency, and SMART-DS publishes no "
            "directional power limit to apply it to."
        ),
        confidence=MappingConfidence.DERIVED,
        origin=ValueOrigin.DERIVED,
        evidence=f"{_EVIDENCE_OBSERVED}: all 95.0 / 95.0.",
    ),
    FieldMapping(
        source="Storage.dss:State",
        source_description="OpenDSS control state. Every observed value is IDLING.",
        source_unit="n/a",
        domain_field="battery.availability",
        domain_meaning="Fraction of rated capability available.",
        transformation="none",
        information_loss=(
            "IDLING is a static modelling state, not an availability signal. No "
            "outage or derate information exists."
        ),
        confidence=MappingConfidence.UNKNOWN,
        origin=ValueOrigin.UNKNOWN,
        evidence=f"{_EVIDENCE_OBSERVED}: 93/93 Storage objects are State=IDLING.",
    ),
    FieldMapping(
        source="Storage.dss:dispatch / charge / discharge power (timeseries)",
        source_description="Does not exist in the dataset.",
        source_unit="n/a",
        domain_field="storage.charge_power_kw / discharge_power_kw",
        domain_meaning="Observed battery power at a timepoint.",
        transformation="UNAVAILABLE - no derivation justified",
        information_loss="Total. Battery dispatch cannot be derived without SOC(t).",
        confidence=MappingConfidence.UNKNOWN,
        origin=ValueOrigin.UNKNOWN,
        evidence=(
            "Exhaustive inspection of Storage.dss and LoadShapes.dss: no storage "
            "loadshape, no SOC column, no dispatch column, no discharge curve."
        ),
    ),
    # ---------------- EV ----------------
    FieldMapping(
        source="placements/ev_residential=*.json, ev_commercial=*.json",
        source_description=(
            "Lists which customer loads are selected for EV adoption, at a given "
            "penetration level."
        ),
        source_unit="n/a",
        domain_field="ev.connected_vehicles",
        domain_meaning="Number of connected vehicles.",
        transformation="none",
        information_loss=(
            "LOCATIONS ONLY. There is no EV charging load, power, energy demand or "
            "session timeseries anywhere in the dataset. A count of adopted sites is "
            "not a charging profile."
        ),
        confidence=MappingConfidence.PARTIAL,
        origin=ValueOrigin.OBSERVED,
        evidence=(
            f"{_EVIDENCE_USER_GUIDE} describes these purely as 'charging locations'; "
            "no EV timeseries folder exists in the sub-region layout."
        ),
    ),
    # ---------------- Topology ----------------
    FieldMapping(
        source="Lines.dss:bus1 / bus2",
        source_description="Endpoints of each line, giving the network graph.",
        source_unit="n/a",
        domain_field="network element from_node / to_node",
        domain_meaning="Directed connection between two nodes.",
        transformation="none",
        information_loss=(
            "Bus names carry phase suffixes (.1.2.3) which Phase 2 has no concept of; "
            "the node is taken as the base bus name."
        ),
        confidence=MappingConfidence.DIRECT,
        origin=ValueOrigin.OBSERVED,
        evidence=f"{_EVIDENCE_OBSERVED}: 4555 Line objects with explicit bus1/bus2.",
    ),
    FieldMapping(
        source="Lines.dss:Length / Units",
        source_description="Line length. Units observed as 'km' in the source.",
        source_unit="km",
        domain_field="network element length",
        domain_meaning="Physical line length.",
        transformation="none",
        information_loss="Phase 2 has no length field; not yet mapped.",
        confidence=MappingConfidence.UNMAPPED,
        origin=ValueOrigin.OBSERVED,
        evidence=f"{_EVIDENCE_OBSERVED}: Units=km Length=0.0168...",
    ),
    FieldMapping(
        source="Transformers.dss",
        source_description="Distribution transformers with kV and kVA ratings.",
        source_unit="kV, kVA",
        domain_field="network element (transformer)",
        domain_meaning="Voltage transformation and its rating.",
        transformation="none",
        information_loss="No timeseries; rating only.",
        confidence=MappingConfidence.DIRECT,
        origin=ValueOrigin.OBSERVED,
        evidence=f"{_EVIDENCE_OBSERVED}: 642 Transformer objects.",
    ),
    FieldMapping(
        source="Buscoords.dss",
        source_description=(
            "Bare '<bus> <longitude> <latitude>' lines with no directives. Parsing "
            "this file with a generic .dss parser yields zero elements."
        ),
        source_unit="degrees",
        domain_field="node.metadata.latitude / longitude",
        domain_meaning="Geographic position of a node.",
        transformation="none",
        information_loss="Phase 2 has no lat/lon field; recorded as asset metadata.",
        confidence=MappingConfidence.DIRECT,
        origin=ValueOrigin.OBSERVED,
        evidence=f"{_EVIDENCE_OBSERVED}: 5194 coordinate rows.",
    ),
    FieldMapping(
        source="Master.dss:Circuit bus1",
        source_description=(
            "The Circuit element's bus1, which is where OpenDSS connects the source "
            "and therefore the point of common coupling with the utility grid."
        ),
        source_unit="bus name",
        domain_field="topology.grid_connection_node (is_grid_connection=True)",
        domain_meaning="The single node through which the grid supplies the feeder.",
        transformation="phase suffix stripped; bus name kept verbatim in the node name",
        information_loss=(
            "NONE. This is read, not inferred: the feeder folder name "
            "'p1uhs0_1247--p1udt12703' does NOT contain the source bus "
            "'p1udt12703-p1uhs0_1247x', so deriving it from the folder would be a "
            "guess (D-051)."
        ),
        confidence=MappingConfidence.DIRECT,
        origin=ValueOrigin.OBSERVED,
        evidence=(
            f"{_EVIDENCE_OBSERVED}: Master.dss declares "
            "'New Circuit.feeder_p1udt12703-p1uhs0_1247x bus1=p1udt12703-p1uhs0_1247x', "
            "and that bus appears as a Line terminal in Lines.dss."
        ),
    ),
    FieldMapping(
        source="Master.dss:basekV",
        source_description="Nominal base voltage of the feeder circuit.",
        source_unit="kV",
        domain_field="node.voltage_kv",
        domain_meaning="Nominal voltage level of a node.",
        transformation="none",
        information_loss="one value for the circuit, not per node.",
        confidence=MappingConfidence.PARTIAL,
        origin=ValueOrigin.OBSERVED,
        evidence=f"{_EVIDENCE_OBSERVED}: basekV=12.47.",
    ),
    FieldMapping(
        source="Master.dss:Solve mode=yearly stepsize=15m number=35040",
        source_description=(
            "The circuit's own solve configuration: 15-minute steps over 35040 "
            "timepoints."
        ),
        source_unit="minutes",
        domain_field="time_base.timestep_minutes",
        domain_meaning="Native resolution of the dataset.",
        transformation="none",
        information_loss=(
            "The load profiles themselves are native 15-minute (interval=0.25 h), so "
            "NO resampling to the Phase 2 15-minute grid is required."
        ),
        confidence=MappingConfidence.DIRECT,
        origin=ValueOrigin.OBSERVED,
        evidence=(
            "Confirmed three independent ways: Master.dss stepsize=15m number=35040; "
            "LoadShapes npts=35040 interval=0.25; profile files contain exactly "
            "35040 values; solar_data contains 35040 data rows."
        ),
    ),
    FieldMapping(
        source="grid import / export flow (timeseries)",
        source_description=(
            "Not provided. Only a single aggregate circuit power value exists, at "
            "the published peak timepoint."
        ),
        source_unit="kW",
        domain_field="grid.import_kw / export_kw",
        domain_meaning="Power exchanged with the utility.",
        transformation="UNAVAILABLE without a power-flow solve",
        information_loss=(
            "Total. Per-node balance therefore cannot be evaluated over time; only "
            "the published peak-timepoint aggregate is checkable."
        ),
        confidence=MappingConfidence.UNKNOWN,
        origin=ValueOrigin.UNKNOWN,
        evidence=(
            "Sub-region layout contains profiles, solar_data, load_data, "
            "load_curves, scenarios and placements. None provides grid flow. The "
            "only circuit-level figure is Summary_data.csv, one row."
        ),
    ),
    FieldMapping(
        source="wind generation",
        source_description="Absent. SMART-DS contains no wind assets.",
        source_unit="n/a",
        domain_field="wind.*",
        domain_meaning="Wind generation state.",
        transformation="UNAVAILABLE",
        information_loss="Wind is in the Phase 2 domain but not in this dataset.",
        confidence=MappingConfidence.UNKNOWN,
        origin=ValueOrigin.UNKNOWN,
        evidence=(
            f"{_EVIDENCE_OBSERVED}: no wind folder in the sub-region layout; "
            "metrics.csv has PV and Battery columns only."
        ),
    ),
)


def mappings_for(source_prefix: str) -> tuple[FieldMapping, ...]:
    """Return every mapping whose source path starts with ``source_prefix``."""
    return tuple(m for m in SMART_DS_MAPPINGS if m.source.startswith(source_prefix))


def unmapped_fields() -> tuple[FieldMapping, ...]:
    """Mappings explicitly recorded as understood but not carried into the domain.

    These are the fields at risk of being silently dropped, so they are kept
    addressable rather than discarded.
    """
    return tuple(m for m in SMART_DS_MAPPINGS if m.confidence is MappingConfidence.UNMAPPED)


def unknown_fields() -> tuple[FieldMapping, ...]:
    """Mappings that could not be established from the data."""
    return tuple(m for m in SMART_DS_MAPPINGS if m.confidence is MappingConfidence.UNKNOWN)


def mapping_table() -> tuple[dict[str, str], ...]:
    """All mappings as dicts, for report generation."""
    return tuple(mapping.to_dict() for mapping in SMART_DS_MAPPINGS)
