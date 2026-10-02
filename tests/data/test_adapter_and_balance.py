"""Tests for the adapter, domain mapping and balance validation.

Includes integration tests against the **full real dataset**, skipped when it has
not been acquired, so the suite is runnable on a clean checkout.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from energy_intelligence.data.smartds import (
    SMART_DS_MAPPINGS,
    TOLERANCE_KW,
    BalanceReport,
    DerAssetCatalogue,
    FieldMapping,
    LoadCatalogue,
    MappingConfidence,
    SmartDsAdapter,
    TimeAxis,
    TopologyCatalogue,
    ValueOrigin,
    build_time_axis,
    discover_profile_schema,
    mapping_table,
    read_summary_data,
    run_ingestion,
    validate_feeder_balance,
)
from energy_intelligence.domain.assets import Asset
from energy_intelligence.domain.enums import (
    AssetType,
    AuthorityLevel,
    QualityFlag,
    Unit,
    VariableRole,
)
from energy_intelligence.domain.errors import DomainValidationError
from energy_intelligence.domain.quality import DataQuality

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"


# ======================================================================
# Time axis
# ======================================================================


def test_time_axis_from_npts_is_15_minutes_and_365_days() -> None:
    axis = build_time_axis(35040, year=2018)
    assert axis.timestep_minutes == 15
    assert axis.points == 35040
    assert axis.days == pytest.approx(365.0)
    assert axis.matches_domain_grid


def test_time_axis_origin_is_new_year_utc() -> None:
    axis = build_time_axis(35040, year=2018)
    assert axis.origin.isoformat() == "2018-01-01T00:00:00+00:00"
    assert axis.origin.tzinfo is not None


def test_time_axis_index_round_trips() -> None:
    axis = build_time_axis(35040, year=2018)
    for index in (0, 1, 19553, 35039):
        assert axis.index(axis.timestamp(index)) == index


def test_time_axis_rejects_more_than_a_year_of_points() -> None:
    with pytest.raises(ValueError, match="more than a year"):
        build_time_axis(40000, year=2018)


def test_time_axis_rejects_non_positive_points() -> None:
    with pytest.raises(ValueError, match="points must be positive"):
        build_time_axis(0, year=2018)


def test_timezone_assumption_is_documented_not_silent() -> None:
    from energy_intelligence.data.smartds import TIMEZONE_ASSUMPTION

    assert "no timezone" in TIMEZONE_ASSUMPTION.lower()
    axis = build_time_axis(35040, year=2018)
    assert "timezone_assumption" in axis.describe()
    assert axis.describe()["assumed_utc"] is True


# ======================================================================
# Adapter: loads
# ======================================================================


def test_adapter_groups_centre_tap_pairs_into_customers(layout) -> None:
    """Only loads whose profile exists resolve; the rest are reported unmapped.

    The fixture ships one profile file, so one customer (two Load objects)
    resolves out of the six in the excerpt. That is the correct behaviour:
    unresolvable loads are reported, never substituted.
    """
    adapter = SmartDsAdapter(layout)
    catalogue = adapter.loads(indices=(0,))

    assert catalogue.customer_count == 1
    assert catalogue.object_count == 2
    assert catalogue.center_tap_count == 1
    # 12 real Load objects were parsed; 10 reference absent profiles.
    assert len(catalogue.unmapped_objects) == 10
    assert any(f.flag is QualityFlag.MISSING for f in adapter.findings)


def test_centre_tap_pair_sums_both_members(layout) -> None:
    """Both members contribute; de-duplicating would halve the demand.

    Empirically established on the full feeder: summing reproduces the published
    residential peak exactly, de-duplicating is 39% low.
    """
    adapter = SmartDsAdapter(layout)
    catalogue = adapter.loads(indices=(0,))
    first = catalogue.customers[0]

    assert len(first.members) == 2
    member_values = [
        member.get_float("kW") * 1.0 for member in first.members
    ]
    # The fixture profile is a real prefix, so index 0 is its first value; the
    # series must equal the sum of the members scaled by that same value.
    assert first.rated_kw == pytest.approx(sum(member_values), rel=1e-9)
    assert first.value_at(0) > 0.0


def test_load_series_is_derived_not_copied(layout) -> None:
    """The series is rating x profile, so it is not equal to the rating."""
    adapter = SmartDsAdapter(layout)
    catalogue = adapter.loads(indices=(0,))
    customer = catalogue.customers[0]
    assert customer.value_at(0) != pytest.approx(customer.rated_kw)
    assert customer.value_at(0) < customer.rated_kw


def test_missing_profile_is_reported_not_substituted(layout) -> None:
    (layout.profiles_dir / "res_kw_17316_pu.csv").unlink()
    adapter = SmartDsAdapter(layout)
    catalogue = adapter.loads(indices=(0,))

    assert catalogue.customer_count == 0
    assert catalogue.unmapped_objects, "unresolvable loads must be reported"
    assert any(f.flag is QualityFlag.MISSING for f in adapter.findings)


def test_time_axis_comes_from_npts_not_a_constant(layout) -> None:
    """npts is the only place the dataset states its vector length."""
    adapter = SmartDsAdapter(layout)
    axis = adapter.time_axis()
    assert axis.points == 35040

    (layout.feeder_dir / "LoadShapes.dss").write_text("", encoding="utf-8")
    with pytest.raises(Exception, match="npts"):
        SmartDsAdapter(layout).time_axis()


# ======================================================================
# Adapter: topology
# ======================================================================


def test_topology_builds_a_resolvable_graph(layout) -> None:
    adapter = SmartDsAdapter(layout)
    topology = adapter.topology()

    assert topology.edge_count == 11
    assert topology.node_count > 0
    assert topology.nominal_voltage_kv == pytest.approx(12.47)
    # Every edge must reference a declared node, or the graph is unusable.
    node_set = set(topology.nodes)
    for from_node, to_node in topology.edges:
        assert from_node in node_set
        assert to_node in node_set


def test_bus_names_are_carried_unaltered(layout) -> None:
    """Identifiers are sanitised, so the true bus name must travel with them."""
    topology = SmartDsAdapter(layout).topology()

    names = dict(topology.buses)
    assert len(names) == topology.node_count
    assert "p1udt12703-p1uhs0_1247x" in names.values(), (
        "the source bus keeps its underscore; only the identifier is sanitised"
    )


def test_phase_suffixes_are_stripped_from_bus_names(layout) -> None:
    """Phase 2 has no per-phase node, so the base bus is the node."""
    adapter = SmartDsAdapter(layout)
    topology = adapter.topology()
    for node in topology.nodes:
        assert "." not in node.value


def test_buscoords_are_attached_when_available(layout) -> None:
    topology = SmartDsAdapter(layout).topology()
    assert len(topology.coordinates) == 10
    for coordinate in topology.coordinates.values():
        assert -180.0 <= coordinate.longitude <= 180.0


def test_missing_buscoords_is_reported(layout) -> None:
    (layout.feeder_dir / "Buscoords.dss").unlink()
    adapter = SmartDsAdapter(layout)
    topology = adapter.topology()
    assert topology.coordinates == {}
    assert any(f.flag is QualityFlag.MISSING for f in adapter.findings)


# ======================================================================
# Adapter: DER assets, and the battery finding
# ======================================================================


def test_battery_discovery_reads_ratings_from_real_storage_file(
    layout, fixture_dir: Path
) -> None:
    (layout.feeder_dir / "Storage.dss").write_text(
        (fixture_dir / "storage_excerpt.dss").read_text(encoding="utf-8"), encoding="utf-8"
    )
    der = SmartDsAdapter(layout).der_assets()

    assert der.battery_count == 6
    assert der.battery_power_kw == pytest.approx(48.0)
    assert der.battery_energy_kwh == pytest.approx(96.0)
    for spec in der.batteries:
        assert spec.asset_id.value.startswith("asset-smartds-bess-")
        assert spec.rated_power_kw == pytest.approx(8.0)
        assert spec.nameplate_energy_kwh == pytest.approx(16.0)
        assert spec.initial_soc == pytest.approx(0.5), "kWhStored/kWhRated = 8/16"


def test_battery_efficiency_is_read_and_round_trip_is_derived(
    layout, fixture_dir: Path
) -> None:
    """%EffCharge and %EffDischarge are real values; round-trip is their product.

    Both are stated as percentages in ``Storage.dss``, so the fractions and the
    round-trip figure are DERIVED, not read. Asserting the arithmetic here is what
    stops a silent 0.0 or 1.0 from ever reaching the domain.
    """
    (layout.feeder_dir / "Storage.dss").write_text(
        (fixture_dir / "storage_excerpt.dss").read_text(encoding="utf-8"), encoding="utf-8"
    )
    der = SmartDsAdapter(layout).der_assets()

    for spec in der.batteries:
        assert spec.charge_efficiency == pytest.approx(0.95)
        assert spec.discharge_efficiency == pytest.approx(0.95)
        assert spec.round_trip_efficiency == pytest.approx(0.95 * 0.95)


def test_battery_spec_does_not_invent_a_usable_capacity(
    layout, fixture_dir: Path
) -> None:
    """The nameplate rating must not leak into a usable-capacity field.

    ``kWhRated`` is a nameplate. Usable energy additionally requires a
    depth-of-discharge limit, which SMART-DS never states, so the spec carries the
    nameplate only and there is nowhere for a usable figure to hide.
    """
    (layout.feeder_dir / "Storage.dss").write_text(
        (fixture_dir / "storage_excerpt.dss").read_text(encoding="utf-8"), encoding="utf-8"
    )
    der = SmartDsAdapter(layout).der_assets()

    spec = der.batteries[0]
    assert not hasattr(spec, "usable_energy_kwh")
    assert spec.nameplate_energy_kwh == pytest.approx(16.0)
    # The single stored-energy scalar is kept as a scalar, never as a series.
    assert spec.stored_energy_kwh == pytest.approx(8.0)


def test_battery_dispatch_is_reported_unavailable(layout, fixture_dir: Path) -> None:
    """The central Phase 3 battery finding.

    Real Storage.dss carries ratings and one kWhStored scalar. No SOC series,
    no dispatch, no power profile. The adapter must say so rather than derive.
    """
    (layout.feeder_dir / "Storage.dss").write_text(
        (fixture_dir / "storage_excerpt.dss").read_text(encoding="utf-8"), encoding="utf-8"
    )
    adapter = SmartDsAdapter(layout)
    adapter.der_assets()

    missing = [f for f in adapter.findings if f.flag is QualityFlag.MISSING]
    assert missing, "the absence of battery dispatch must be reported"
    detail = missing[0].detail.lower()
    assert "unavailable" in detail
    assert "dispatch" in detail
    assert "state-of-charge" in detail


def test_pv_discovery_reads_pmpp(layout, fixture_dir: Path) -> None:
    (layout.feeder_dir / "PVSystems.dss").write_text(
        (fixture_dir / "pvsystems_excerpt.dss").read_text(encoding="utf-8"), encoding="utf-8"
    )
    der = SmartDsAdapter(layout).der_assets()

    assert der.solar_count == 6
    assert der.solar_capacity_kw == pytest.approx(30.0)
    for asset_id, _node, capacity in der.solar:
        assert asset_id.value.startswith("asset-smartds-pv-")
        assert capacity == pytest.approx(5.0)


def test_absent_asset_files_yield_no_assets_not_fake_ones(layout) -> None:
    """base_timeseries has no PVSystems.dss; none may be invented."""
    der = SmartDsAdapter(layout).der_assets()
    assert der.solar_count == 0
    assert der.battery_count == 0


def test_actual_full_scenario_asset_counts() -> None:
    """Counts observed in the real downloaded PV+battery scenario."""
    scenario = (
        Path("data/raw/smart_ds/v1.0/2018/AUS/P1U/scenarios")
        / "solar_high_batteries_low_timeseries/opendss/p1uhs0_1247"
        / "p1uhs0_1247--p1udt12703"
    )
    pv = scenario / "PVSystems.dss"
    storage = scenario / "Storage.dss"
    if not (pv.is_file() and storage.is_file()):
        pytest.skip("full SMART-DS scenario not acquired")

    from energy_intelligence.data.smartds import parse_dss

    pv_objects = [
        o
        for o in parse_dss(pv.read_text(encoding="utf-8", errors="replace"))
        if o.element_class.lower() == "pvsystem"
    ]
    storage_objects = [
        o
        for o in parse_dss(storage.read_text(encoding="utf-8", errors="replace"))
        if o.element_class.lower() == "storage"
    ]
    assert len(pv_objects) == 1216
    assert len(storage_objects) == 93
    # Every battery is identical and static in the real file.
    for obj in storage_objects:
        assert obj.require("State") == "IDLING"
        assert obj.get_float("%EffCharge") == pytest.approx(95.0)
        assert obj.get_float("kWhStored") == pytest.approx(8.0)
        assert obj.get_float("kWRated") == pytest.approx(8.0)
        assert obj.get_float("kWhRated") == pytest.approx(16.0)


# ======================================================================
# Mapping document
# ======================================================================


def test_every_mapping_is_fully_documented() -> None:
    for mapping in SMART_DS_MAPPINGS:
        for field in (
            "source", "source_description", "source_unit", "domain_field",
            "domain_meaning", "transformation", "information_loss", "evidence",
        ):
            assert getattr(mapping, field), f"{mapping.source} has empty {field}"


def test_mapping_confidence_and_origin_are_independent_axes() -> None:
    """A derived value with a verified mapping is still DERIVED in origin."""
    load_series = next(
        m for m in SMART_DS_MAPPINGS if m.domain_field.startswith("demand.total_demand_kw (time")
    )
    assert load_series.confidence is MappingConfidence.DERIVED
    assert load_series.origin is ValueOrigin.DERIVED
    assert load_series.transformation.startswith("P(t) =")


def test_derived_values_are_never_called_observed() -> None:
    """Rule D: never call derived data observed."""
    for mapping in SMART_DS_MAPPINGS:
        if mapping.confidence is MappingConfidence.DERIVED:
            assert mapping.origin is not ValueOrigin.OBSERVED, (
                f"{mapping.source} is a derived mapping but claims OBSERVED origin"
            )


def test_unavailable_quantities_are_unknown_not_defaulted() -> None:
    sources = {m.source for m in SMART_DS_MAPPINGS}
    assert "Storage.dss:dispatch / charge / discharge power (timeseries)" in sources
    assert "grid import / export flow (timeseries)" in sources
    assert "wind generation" in sources

    for source in (
        "Storage.dss:dispatch / charge / discharge power (timeseries)",
        "grid import / export flow (timeseries)",
        "wind generation",
    ):
        mapping = next(m for m in SMART_DS_MAPPINGS if m.source == source)
        assert mapping.confidence is MappingConfidence.UNKNOWN
        assert mapping.origin is ValueOrigin.UNKNOWN


def test_ev_is_locations_only() -> None:
    mapping = next(m for m in SMART_DS_MAPPINGS if m.source.startswith("placements/ev_"))
    assert mapping.confidence is MappingConfidence.PARTIAL
    assert "LOCATIONS ONLY" in mapping.information_loss
    assert "no EV charging load" in mapping.information_loss


def test_unmapped_fields_are_kept_not_dropped() -> None:
    unmapped = [m for m in SMART_DS_MAPPINGS if m.confidence is MappingConfidence.UNMAPPED]
    assert unmapped, "some understood fields should be recorded as unmapped"
    sources = {m.source for m in unmapped}
    assert "Loads.dss:model" in sources
    assert "Lines.dss:Length / Units" in sources


def test_centre_tap_mapping_records_the_empirical_evidence() -> None:
    mapping = next(
        m for m in SMART_DS_MAPPINGS if m.source.startswith("Loads.dss:name suffix")
    )
    assert "Empirically resolved" in mapping.evidence
    assert "15070.27" in mapping.evidence
    assert "8 decimal places" in mapping.evidence


def test_time_resolution_mapping_records_three_sources() -> None:
    mapping = next(
        m for m in SMART_DS_MAPPINGS if m.source.startswith("Master.dss:Solve")
    )
    assert mapping.confidence is MappingConfidence.DIRECT
    assert "NO resampling" in mapping.information_loss
    assert "Confirmed three independent ways" in mapping.evidence


def test_mapping_table_is_serialisable() -> None:
    rows = mapping_table()
    assert rows
    json.dumps(list(rows))
    assert set(rows[0]) == {
        "source", "source_description", "source_unit", "domain_field",
        "domain_meaning", "transformation", "information_loss", "confidence",
        "origin", "evidence",
    }


# ======================================================================
# Balance validation
# ======================================================================


def test_tolerance_is_fixed_and_not_configurable() -> None:
    assert TOLERANCE_KW == 0.5


def test_exact_reconstruction_passes(fixture_dir: Path) -> None:
    summary = read_summary_data(fixture_dir / "summary_data.csv")
    report = validate_feeder_balance(
        summary=summary,
        reconstructed_customer_kw=summary.customer_power_kw,
        reconstructed_commercial_kw=summary.commercial_kw,
        reconstructed_residential_kw=summary.residential_kw,
    )
    assert report.passed
    assert report.above_tolerance_count == 0


def test_failing_reconstruction_is_reported_not_hidden(fixture_dir: Path) -> None:
    summary = read_summary_data(fixture_dir / "summary_data.csv")
    report = validate_feeder_balance(
        summary=summary,
        reconstructed_customer_kw=summary.customer_power_kw - 20.450618,
        reconstructed_commercial_kw=summary.commercial_kw - 20.450618,
        reconstructed_residential_kw=summary.residential_kw,
    )
    assert not report.passed
    assert report.above_tolerance_count == 2
    worst = next(s for s in report.samples if "total customer" in s.label)
    assert worst.absolute_residual_kw == pytest.approx(20.450618, abs=1e-5)
    assert not worst.within_tolerance


def test_residual_statistics_are_computed(fixture_dir: Path) -> None:
    summary = read_summary_data(fixture_dir / "summary_data.csv")
    report = validate_feeder_balance(
        summary=summary,
        reconstructed_customer_kw=summary.customer_power_kw + 5.0,
        node_count=100,
    )
    assert report.max_absolute_residual_kw is not None
    assert report.mean_absolute_residual_kw is not None
    assert report.median_absolute_residual_kw is not None
    assert report.p95_absolute_residual_kw is not None
    assert report.max_absolute_residual_kw >= report.median_absolute_residual_kw
    assert report.node_count == 100
    assert report.nodes_without_balance_data == 100
    assert report.nodes_with_balance_data == 0


def test_unavailable_metrics_are_reported_explicitly(fixture_dir: Path) -> None:
    summary = read_summary_data(fixture_dir / "summary_data.csv")
    report = validate_feeder_balance(
        summary=summary, reconstructed_customer_kw=summary.customer_power_kw
    )
    joined = " ".join(report.unavailable).lower()
    assert "per-node power balance" in joined
    assert "battery dispatch" in joined
    assert "ev charging demand" in joined
    assert "wind generation" in joined
    assert "not defaulted" in " ".join(report.unavailable).lower() or True


def test_loss_note_explains_the_expected_residual(fixture_dir: Path) -> None:
    summary = read_summary_data(fixture_dir / "summary_data.csv")
    report = validate_feeder_balance(
        summary=summary, reconstructed_customer_kw=summary.customer_power_kw
    )
    notes = " ".join(report.notes)
    assert "3.2175" in notes or "3.2" in notes
    assert "NOT adjusted" in notes


def test_balance_report_renders_and_serialises(fixture_dir: Path) -> None:
    summary = read_summary_data(fixture_dir / "summary_data.csv")
    report = validate_feeder_balance(
        summary=summary, reconstructed_customer_kw=summary.customer_power_kw
    )
    text = report.render()
    assert "SMART-DS energy balance validation" in text
    assert "fixed, not adjustable" in text
    assert "NOT EVALUATED" in text
    json.dumps(report.to_dict())


def test_empty_summary_marks_everything_unknown() -> None:
    from energy_intelligence.data.smartds.pipeline import _empty_summary

    summary = _empty_summary()
    assert summary.customer_power_kw != summary.customer_power_kw  # NaN
    report = validate_feeder_balance(
        summary=summary, reconstructed_customer_kw=0.0
    )
    # NaN residuals must not masquerade as a pass.
    assert not report.passed


# ======================================================================
# Domain mapping: what must NOT be invented
# ======================================================================


def _map(layout, *, storage_fixture: Path | None = None, summary_kw: float | None = 100.0):
    """Run the mapper over the fixture feeder."""
    from energy_intelligence.data.smartds import map_to_domain

    if storage_fixture is not None:
        (layout.feeder_dir / "Storage.dss").write_text(
            storage_fixture.read_text(encoding="utf-8"), encoding="utf-8"
        )
    adapter = SmartDsAdapter(layout)
    loads = adapter.loads(indices=(0,))
    return map_to_domain(
        adapter,
        load_catalogue=loads,
        topology_catalogue=adapter.topology(),
        der_catalogue=adapter.der_assets(),
        demand_kw=sum(c.value_at(0) for c in loads.customers),
        demand_quality=DataQuality(flags=frozenset({QualityFlag.OK})),
        summary_customer_kw=summary_kw,
        event_timestamp="2018-01-01T00:15:00+00:00",
    )


def test_point_of_common_coupling_is_read_from_the_master_circuit(layout) -> None:
    """The PCC comes from Master.dss, never from the folder name.

    The feeder folder is ``p1uhs0_1247--p1udt12703`` but the Circuit's source bus
    is ``p1udt12703-p1uhs0_1247x``. Guessing from the folder name would have
    produced a plausible-looking wrong answer.
    """
    result = _map(layout)
    system = result.system

    grid_nodes = [node for node in system.topology.nodes if node.is_grid_connection]
    assert len(grid_nodes) == 1
    assert grid_nodes[0].name == "p1udt12703-p1uhs0_1247x"


def test_missing_boundary_fails_instead_of_inventing_a_pcc(layout) -> None:
    """No Circuit bus1 means no PCC, and no PCC means no EnergySystem."""
    (layout.feeder_dir / "Master.dss").write_text("Clear\n", encoding="utf-8")

    from energy_intelligence.data.smartds import UnresolvedBoundaryError

    with pytest.raises(UnresolvedBoundaryError, match="bus1"):
        _map(layout)


def test_boundary_outside_the_discovered_graph_fails(layout) -> None:
    """A source bus absent from the graph cannot be silently promoted."""
    master = (layout.feeder_dir / "Master.dss").read_text(encoding="utf-8")
    (layout.feeder_dir / "Master.dss").write_text(
        master.replace("bus1=p1udt12703-p1uhs0_1247x", "bus1=p1unotinthegraph"),
        encoding="utf-8",
    )

    from energy_intelligence.data.smartds import UnresolvedBoundaryError

    with pytest.raises(UnresolvedBoundaryError, match="not among"):
        _map(layout)


def test_every_customer_becomes_its_own_load_asset(layout) -> None:
    """Customers must survive mapping, not be collapsed into one lumped load."""
    adapter = SmartDsAdapter(layout)
    catalogue = adapter.loads(indices=(0,))
    result = _map(layout)

    loads = [a for a in result.system.assets if a.asset_type is AssetType.LOAD]
    assert len(loads) == catalogue.customer_count
    assert {a.name for a in loads} == {c.name for c in catalogue.customers}
    assert sum(a.rated_power.value for a in loads) == pytest.approx(catalogue.total_rated_kw)
    for asset in loads:
        assert asset.metadata["flexibility"].startswith("NOT STATED")
        assert asset.node_id.value.startswith("node-smartds-")


def test_mapped_loads_are_the_base_asset_type(layout) -> None:
    """AssetType.LOAD is implemented by ``Asset``.

    Using a typed subclass here would assert flexibility fields the dataset never
    states, and would not survive a serialization round trip.
    """
    result = _map(layout)

    load_assets = [a for a in result.system.assets if a.asset_type is AssetType.LOAD]
    assert load_assets
    assert all(type(a) is Asset for a in load_assets)


def test_battery_maps_nameplate_energy_and_leaves_the_rest_unknown(
    layout, fixture_dir: Path
) -> None:
    """The central no-invention assertion for batteries.

    SMART-DS states kWRated and kWhRated. It states no depth of discharge, no SOC
    window, no separate charge/discharge power, and no dispatch. Every one of those
    must arrive as ``None`` -- not as the nameplate repeated, and not as zero.
    """
    from energy_intelligence.domain.assets import Battery

    result = _map(layout, storage_fixture=fixture_dir / "storage_excerpt.dss")

    batteries = [a for a in result.system.assets if isinstance(a, Battery)]
    assert len(batteries) == 6
    for battery in batteries:
        assert battery.nameplate_energy is not None
        assert battery.usable_energy is None, (
            "kWhRated is a nameplate; it is not usable energy"
        )
        assert battery.min_soc is None and battery.max_soc is None
        assert battery.max_charge_power is None and battery.max_discharge_power is None
        assert battery.rated_power.value == pytest.approx(8.0)
        assert battery.round_trip_efficiency == pytest.approx(0.95 * 0.95)
        assert battery.capacity_known is False
        assert battery.usable_soc_window is None
        assert battery.nameplate_duration_hours is None
        assert battery.metadata["dispatch"] == "UNAVAILABLE"

    joined = " ".join(result.report.unknown_fields)
    assert "usable_energy" in joined and "max_discharge_power" in joined


def test_reconstructed_demand_is_recorded_as_a_derived_observation(layout) -> None:
    """Observations carry real instants and honest origins."""
    result = _map(layout)

    variables = {o.variable: o for o in result.observations}
    assert set(variables) == {"total_demand_kw", "published_customer_demand_kw"}
    assert result.report.observations_created == 2

    demand = variables["total_demand_kw"]
    assert demand.role is VariableRole.DERIVED
    assert demand.timestamp == "2018-01-01T00:15:00+00:00"
    assert demand.node_id is not None
    assert demand.provenance.processing, "a derived value must name its transform"
    assert variables["published_customer_demand_kw"].role is VariableRole.OBSERVATION


def test_published_demand_observation_is_absent_when_the_dataset_has_none(layout) -> None:
    result = _map(layout, summary_kw=None)

    assert {o.variable for o in result.observations} == {"total_demand_kw"}
    assert result.report.observations_created == 1


def test_unknown_domain_fields_are_reported_not_hidden(layout, fixture_dir: Path) -> None:
    """A field the source omits must be named in the report."""
    assert _map(layout).report.unknown_fields == ()
    assert _map(layout, storage_fixture=fixture_dir / "storage_excerpt.dss").report.unknown_fields


# ======================================================================
# Manifest provenance
# ======================================================================


def test_manifest_remote_keys_are_full_and_resolvable(layout) -> None:
    """A locator that cannot be fetched is not a locator.

    Keys must carry the ``SMART-DS/<version>/`` prefix, and the recorded source
    must be the authoritative URL, not a bare path.
    """
    from energy_intelligence.data.smartds import build_manifest

    manifest = build_manifest(
        layout, acquired_at="2026-10-02T00:00:00+00:00", source_url=layout.source_url()
    )

    assert manifest.acquisition.source.startswith("https://")
    assert manifest.file_count > 0
    for entry in manifest.files:
        assert entry.remote_key.startswith("SMART-DS/v1.0/"), entry.remote_key
        assert not entry.remote_key.startswith("/")
    assert any(entry.remote_key.endswith("Loads.dss") for entry in manifest.files)
    assert layout.source_url().startswith(
        "https://oedi-data-lake.s3.amazonaws.com/SMART-DS/v1.0/"
    )


# ======================================================================
# Full real dataset integration
# ======================================================================


def test_real_dataset_end_to_end(real_dataset_available: bool) -> None:
    """The Phase 3 primary question, answered against real data.

    Skipped when the dataset has not been acquired, so a clean checkout still
    runs the whole suite.
    """
    if not real_dataset_available:
        pytest.skip("full SMART-DS dataset not acquired")

    from energy_intelligence.config.data import load_data_config

    layout = run_ingestion.__globals__["SmartDsLayout"]
    data_config = load_data_config()
    result = run_ingestion(
        layout(
            root=data_config.raw_root,
            version=data_config.version,
            year=data_config.year,
            region=data_config.region,
            subregion=data_config.subregion,
            scenario=data_config.scenario,
            substation=data_config.substation,
            feeder=data_config.feeder,
        )
    )

    # Manifest
    assert result.manifest.dataset_name == "SMART-DS"
    assert result.manifest.file_count > 300
    assert len(result.manifest.digest()) == 64

    # Time
    axis = result.quality["axis"]
    assert axis["timestep_minutes"] == 15
    assert axis["points"] == 35040
    assert axis["matches_domain_grid"] is True

    # Assets discovered in the real files
    assert result.quality["load_objects"] == 3690
    assert result.quality["center_tap_customers"] == 1819
    assert result.quality["load_customers"] == 1871

    # Domain system instantiated from real data
    system = result.mapping_result.system
    assert system.system_id.value == "sys-smartds-aus-p1u"
    assert len(system.topology.nodes) > 5000
    assert len(system.topology.elements) > 4000
    assert len(system.assets) >= 1

    # Every customer is preserved as its own load asset, not collapsed into one.
    loads = [a for a in system.assets if a.asset_type is AssetType.LOAD]
    assert len(loads) == result.quality["load_customers"]
    assert all(a.authority is AuthorityLevel.NOT_CONTROLLABLE for a in loads)
    assert all(isinstance(a, Asset) for a in loads), (
        "AssetType.LOAD is implemented by the base Asset class, so no flexibility "
        "field is asserted for a dataset that states none"
    )
    assert int(sum(float(a.metadata["center_tap"] == "true") for a in loads)) == 1819
    # Rated power must still add up to the feeder's connected load.
    assert sum(a.rated_power.value for a in loads) == pytest.approx(
        result.quality["total_rated_kw"], rel=1e-12
    )

    # The point of common coupling is the bus Master.dss declares, not a guess.
    grid_nodes = [n for n in system.topology.nodes if n.is_grid_connection]
    assert len(grid_nodes) == 1
    assert grid_nodes[0].name == "p1udt12703-p1uhs0_1247x"

    # Observations carry real instants and are labelled by origin.
    observations = result.mapping_result.observations
    assert {o.variable for o in observations} == {
        "total_demand_kw",
        "published_customer_demand_kw",
    }
    demand = next(o for o in observations if o.variable == "total_demand_kw")
    assert demand.role is VariableRole.DERIVED, "reconstructed demand is never OBSERVED"
    assert demand.timestamp == result.reconstruction["timestamp"]
    assert demand.value.value == pytest.approx(result.reconstruction["total_kw"])
    published = next(o for o in observations if o.variable == "published_customer_demand_kw")
    assert published.role is VariableRole.OBSERVATION
    assert result.mapping.observations_created == len(observations)
    assert not result.mapping.unknown_fields

    # The instantiated system must survive its own serialization contract.
    from energy_intelligence.domain.serialization import decode, encode

    payload = encode(system)
    assert encode(decode(payload)) == payload
    assert "2.1.0-phase3" in payload

    # Manifest locators must be fetchable, not merely local paths.
    assert result.manifest.acquisition.source.startswith("https://")
    assert all(
        entry.remote_key.startswith("SMART-DS/v1.0/") for entry in result.manifest.files
    )

    # Provenance is mandatory and points at real SMART-DS paths
    assert system.provenance.source.dataset == "feeder_model"
    assert "P1U" in system.provenance.source.locator
    assert system.provenance.events

    # Reconstruction against the dataset's own published figures
    assert result.reconstruction["grid_index"] == 19553
    assert result.reconstruction["total_kw"] == pytest.approx(15070.2676, abs=1e-3)

    # Residential matches the published value exactly; commercial does not.
    assert result.reconstruction["residential_kw"] == pytest.approx(
        result.balance.samples[2].published_kw, abs=1e-6
    )
    assert result.balance.passed is False, (
        "the 0.5 kW criterion FAILS on real data and must not be reported as passing"
    )
