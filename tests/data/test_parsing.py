"""Tests for parsing, discovery and manifest construction.

Every fixture here is a verbatim excerpt of real SMART-DS v1.0 text, so these
tests assert the parser handles the real format rather than an invented one.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from energy_intelligence.data.manifest import (
    AcquisitionMetadata,
    DatasetManifest,
    FileEntry,
    compute_sha256,
)
from energy_intelligence.data.smartds import (
    DssParseError,
    discover_buscoords_schema,
    discover_csv_schema,
    discover_dss_schema,
    discover_feeder_schema,
    discover_profile_schema,
    parse_buscoords,
    parse_dss,
    read_profile,
    read_summary_data,
    sanitize_identifier,
)

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"


# ======================================================================
# OpenDSS parsing
# ======================================================================


def test_parses_real_load_directives(fixture_dir: Path) -> None:
    objects = parse_dss((fixture_dir / "loads_excerpt.dss").read_text(encoding="utf-8"))
    assert len(objects) == 12
    assert all(obj.element_class == "Load" for obj in objects)

    first = objects[0]
    assert first.name == "load_p1ulv279_1"
    assert first.require("conn") == "wye"
    assert first.require("yearly") == "res_kw_17316_pu"
    assert first.require("bus1") == "p1ulv279.1"
    assert first.get_float("kW") == pytest.approx(5.262356427925992)
    assert first.get_int("Phases") == 1
    assert first.get_float("model") == 1.0


def test_centre_tap_pairs_carry_identical_ratings(fixture_dir: Path) -> None:
    """Observed in real data: both halves of a pair carry the same kW.

    This is why summing the pair is correct and de-duplicating is not.
    """
    objects = parse_dss((fixture_dir / "loads_excerpt.dss").read_text(encoding="utf-8"))
    pairs: dict[str, set[float]] = {}
    for obj in objects:
        if obj.name.endswith(("_1", "_2")):
            pairs.setdefault(obj.name[:-2], set()).add(obj.get_float("kW"))
    assert pairs, "fixture should contain centre-tap pairs"
    for base, ratings in pairs.items():
        assert len(ratings) == 1, f"{base} members disagree on kW: {ratings}"


def test_parses_real_loadshape_with_nested_file_reference(fixture_dir: Path) -> None:
    """``mult = (file=../../x.csv)`` contains '=' inside a value.

    A naive whitespace split would corrupt this, so the parser scans for
    key boundaries instead.
    """
    objects = parse_dss(
        (fixture_dir / "loadshapes_excerpt.dss").read_text(encoding="utf-8")
    )
    assert objects
    shape = objects[0]
    assert shape.element_class == "Loadshape"
    assert shape.get_int("npts") == 35040
    assert shape.get_float("interval") == 0.25
    assert "res_kw_17316_pu.csv" in shape.require("mult")


def test_parses_real_circuit_definition(fixture_dir: Path) -> None:
    objects = parse_dss((fixture_dir / "master_excerpt.dss").read_text(encoding="utf-8"))
    circuits = [o for o in objects if o.element_class.lower() == "circuit"]
    assert circuits
    assert circuits[0].get_float("basekV") == pytest.approx(12.47)
    assert circuits[0].get_float("pu") == pytest.approx(1.03)


def test_parses_real_lines_with_phase_suffixed_buses(fixture_dir: Path) -> None:
    objects = parse_dss((fixture_dir / "lines_excerpt.dss").read_text(encoding="utf-8"))
    lines = [o for o in objects if o.element_class == "Line"]
    # 10 leading lines plus the line incident to the Circuit source bus, which the
    # extraction script appends so the excerpt has a resolvable boundary.
    assert len(lines) == 11
    first = lines[0]
    assert "." in first.require("bus1")
    assert first.get("enabled") == "y"
    assert first.get("Linecode")


def test_excerpt_contains_the_circuit_source_bus(fixture_dir: Path) -> None:
    """The Circuit's bus1 must be present as a line terminal in the excerpt."""
    master = parse_dss((fixture_dir / "master_excerpt.dss").read_text(encoding="utf-8"))
    circuit = next(o for o in master if o.element_class.lower() == "circuit")
    source_bus = circuit.require("bus1").split(".")[0]

    lines = parse_dss((fixture_dir / "lines_excerpt.dss").read_text(encoding="utf-8"))
    terminals = {
        value.split(".")[0]
        for line in lines
        for key in ("bus1", "bus2")
        if (value := line.get(key))
    }
    assert source_bus in terminals


def test_non_element_directives_are_skipped() -> None:
    text = """
Clear
New Circuit.test bus1=b1 basekV=12.47
Redirect Loads.dss
Set Voltagebases=[12.47]
Solve mode=yearly stepsize=15m number=35040
"""
    objects = parse_dss(text)
    assert len(objects) == 1
    assert objects[0].qualified_name == "Circuit.test"


def test_elements_split_without_blank_lines() -> None:
    text = "New Load.a bus1=b1 kW=1\nNew Load.b bus1=b2 kW=2\n"
    objects = parse_dss(text)
    assert [o.name for o in objects] == ["a", "b"]


def test_parameter_order_is_preserved(fixture_dir: Path) -> None:
    objects = parse_dss((fixture_dir / "loads_excerpt.dss").read_text(encoding="utf-8"))
    first = objects[0]
    assert list(first.order)[:4] == ["conn", "bus1", "kV", "Vminpu"]


def test_lookup_is_case_insensitive_but_preserves_source(fixture_dir: Path) -> None:
    obj = parse_dss((fixture_dir / "loads_excerpt.dss").read_text(encoding="utf-8"))[0]
    assert obj.get("kw") == obj.get("kW")
    assert "kW" in obj.params


def test_missing_required_parameter_raises() -> None:
    obj = parse_dss("New Load.a bus1=b1 kW=1")[0]
    with pytest.raises(DssParseError, match="missing required parameter"):
        obj.require("kvar")


def test_non_numeric_value_raises_rather_than_defaulting() -> None:
    obj = parse_dss("New Load.a bus1=b1 kW=large")[0]
    with pytest.raises(DssParseError, match="not a real number"):
        obj.get_float("kW")


def test_element_name_must_be_qualified() -> None:
    with pytest.raises(DssParseError, match="not of the form Class.name"):
        parse_dss("New Loadonlyname bus1=b1")


def test_empty_element_directive_raises() -> None:
    with pytest.raises(DssParseError, match="empty element directive"):
        parse_dss("New\n")


# ======================================================================
# Buscoords - a format that is NOT a .dss directive file
# ======================================================================


def test_buscoords_parses_real_rows(fixture_dir: Path) -> None:
    coordinates = parse_buscoords(
        (fixture_dir / "buscoords_excerpt.dss").read_text(encoding="utf-8")
    )
    assert len(coordinates) == 10
    first = next(iter(coordinates.values()))
    assert -180.0 <= first.longitude <= 180.0
    assert -90.0 <= first.latitude <= 90.0


def test_buscoords_yields_nothing_through_the_generic_parser(fixture_dir: Path) -> None:
    """Why a dedicated parser exists.

    Passing the real coordinate file to the generic .dss parser yields zero
    elements, so every node would silently appear to have no location.
    """
    assert parse_dss((fixture_dir / "buscoords_excerpt.dss").read_text(encoding="utf-8")) == []


def test_buscoords_rejects_wrong_field_count() -> None:
    with pytest.raises(DssParseError, match="expected '<bus> <lon> <lat>'"):
        parse_buscoords("p1udm971 -97.71\n")


def test_buscoords_rejects_out_of_range_coordinate() -> None:
    """Latitude 200 is impossible, so the columns must not be transposed silently."""
    with pytest.raises(DssParseError, match="outside the plausible range"):
        parse_buscoords("bus1 -97.7 200.0\n")


def test_buscoords_rejects_duplicate_bus() -> None:
    with pytest.raises(DssParseError, match="duplicate bus"):
        parse_buscoords("bus1 -97.7 30.4\nbus1 -97.7 30.4\n")


def test_buscoords_schema_reports_units(fixture_dir: Path) -> None:
    schema = discover_buscoords_schema(fixture_dir / "buscoords_excerpt.dss")
    assert schema["count"] == 10
    assert schema["units"] == "degrees"
    assert schema["fields"] == ["bus", "longitude", "latitude"]


# ======================================================================
# Profiles
# ======================================================================


def test_reads_real_profile_values(fixture_dir: Path) -> None:
    values = read_profile(fixture_dir / "profile_res_kw_17316.csv")
    assert len(values) == 96
    assert all(0.0 <= v <= 1.0 for v in values)


def test_profile_fixture_is_a_prefix_not_a_normalised_series(
    fixture_dir: Path,
) -> None:
    """The fixture is a 96-value prefix, so it must NOT be asserted normalised.

    Per-unit normalisation (max == 1.0) is a property of the full 35040-point
    year. Asserting it on a truncated fixture would be a false test.
    """
    schema = discover_profile_schema(fixture_dir / "profile_res_kw_17316.csv")
    assert schema.rows == 96
    assert schema.is_normalised is False


def test_profile_rejects_non_numeric_line(tmp_path: Path) -> None:
    bad = tmp_path / "bad.csv"
    bad.write_text("0.1\nnot-a-number\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not a real number"):
        read_profile(bad)


# ======================================================================
# CSV schema discovery
# ======================================================================


def test_discovers_real_solar_columns(fixture_dir: Path) -> None:
    schema = discover_csv_schema(fixture_dir / "solar_excerpt.csv")
    assert schema.has_header
    assert schema.rows == 4
    assert "PoA Irradiance (W/m^2)" in schema.column_names
    assert "kW Generated (1000 kW Array)" in schema.column_names
    poa = schema.column("PoA Irradiance (W/m^2)")
    assert poa is not None
    assert poa.inferred_type == "float"


def test_discovers_summary_data(fixture_dir: Path) -> None:
    schema = discover_csv_schema(fixture_dir / "summary_data.csv")
    assert schema.rows == 1
    assert "Total peak time real customer power (kW)" in schema.column_names


def test_headerless_profile_is_detected_as_such(fixture_dir: Path) -> None:
    schema = discover_csv_schema(fixture_dir / "profile_res_kw_17316.csv")
    assert schema.has_header is False
    assert "headerless" in schema.notes


# ======================================================================
# Summary data
# ======================================================================


def test_reads_published_summary(fixture_dir: Path) -> None:
    summary = read_summary_data(fixture_dir / "summary_data.csv")
    assert summary.customer_power_kw == pytest.approx(15090.718224264258)
    assert summary.residential_kw == pytest.approx(12380.512586099816)
    assert summary.commercial_kw == pytest.approx(2710.2056381643924)
    assert summary.losses_kw == pytest.approx(501.67236241331824)
    assert summary.circuit_power_kw == pytest.approx(15591.956006078866)
    assert summary.losses_percent == pytest.approx(3.217507554650169)


def test_published_peak_index_lands_on_the_grid(fixture_dir: Path) -> None:
    summary = read_summary_data(fixture_dir / "summary_data.csv")
    index = summary.peak_index
    assert index == (204 - 1) * 96 + 16 * 4 + 1
    assert 0 <= index < 35040


def test_published_class_split_is_internally_consistent(fixture_dir: Path) -> None:
    summary = read_summary_data(fixture_dir / "summary_data.csv")
    assert summary.published_split_consistent


# ======================================================================
# Discovery
# ======================================================================


def test_discovers_dss_schema_from_real_files(fixture_dir: Path) -> None:
    schema = discover_dss_schema(fixture_dir / "loads_excerpt.dss")
    assert schema.total_elements == 12
    assert schema.element_classes == {"Load": 12}
    assert "kW" in schema.parameters_by_class["Load"]
    assert "yearly" in schema.parameters_by_class["Load"]


def test_feeder_schema_reports_absent_files(layout, fixture_dir: Path) -> None:
    """PVSystems.dss is absent from base_timeseries and must be reported absent.

    Reporting absence explicitly is what stops a missing asset class from
    looking like a system with no PV.
    """
    (layout.feeder_dir / "PVSystems.dss").write_text(
        (fixture_dir / "pvsystems_excerpt.dss").read_text(encoding="utf-8"), encoding="utf-8"
    )
    schema = discover_feeder_schema(layout)
    assert schema.loads is not None
    assert schema.loads.total_elements == 12
    assert schema.pv_systems is not None
    assert schema.pv_systems.total_elements == 6
    assert schema.transformers is None
    assert any("Transformers.dss absent" in note for note in schema.notes)


# ======================================================================
# Manifest
# ======================================================================


def test_compute_sha256_is_stable(tmp_path: Path) -> None:
    path = tmp_path / "f.bin"
    path.write_bytes(b"energy-intelligence")
    first = compute_sha256(path)
    assert first == compute_sha256(path)
    assert len(first) == 64


def _manifest(tmp_path: Path) -> DatasetManifest:
    a = tmp_path / "a.dss"
    b = tmp_path / "b.csv"
    a.write_text("New Load.x bus1=b1 kW=1\n", encoding="utf-8")
    b.write_text("1,2\n", encoding="utf-8")
    return DatasetManifest(
        dataset_name="SMART-DS",
        dataset_version="1.0",
        subset="test",
        acquisition=AcquisitionMetadata(
            acquired_at="2026-10-02T00:00:00+00:00", source="https://example.invalid"
        ),
        files=(
            FileEntry("a.dss", a.stat().st_size, compute_sha256(a), "loads"),
            FileEntry("b.csv", b.stat().st_size, compute_sha256(b), ""),
        ),
        local_root=str(tmp_path),
    )


def test_manifest_digest_excludes_acquisition_time(tmp_path: Path) -> None:
    """Two acquisitions of identical bytes must agree on the content digest."""
    first = _manifest(tmp_path)
    later = DatasetManifest(
        dataset_name=first.dataset_name,
        dataset_version=first.dataset_version,
        subset=first.subset,
        acquisition=AcquisitionMetadata(
            acquired_at="2030-01-01T00:00:00+00:00", source="https://example.invalid"
        ),
        files=first.files,
        local_root=first.local_root,
    )
    assert first.digest() == later.digest()


def test_manifest_digest_changes_when_content_changes(tmp_path: Path) -> None:
    base = _manifest(tmp_path)
    (tmp_path / "a.dss").write_text("New Load.x bus1=b1 kW=2\n", encoding="utf-8")
    changed = DatasetManifest(
        dataset_name=base.dataset_name,
        dataset_version=base.dataset_version,
        subset=base.subset,
        acquisition=base.acquisition,
        files=(
            FileEntry(
                "a.dss",
                (tmp_path / "a.dss").stat().st_size,
                compute_sha256(tmp_path / "a.dss"),
                "loads",
            ),
            base.files[1],
        ),
        local_root=base.local_root,
    )
    assert changed.digest() != base.digest()


def test_manifest_is_json_serialisable_and_has_no_secrets(tmp_path: Path) -> None:
    import json

    payload = _manifest(tmp_path).to_dict()
    text = json.dumps(payload)
    assert "_manifest_version" in payload
    assert payload["file_count"] == 2
    for forbidden in ("password", "token", "secret", "credential"):
        assert forbidden not in text.lower()


def test_manifest_lookups(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    assert manifest.entry("a.dss") is not None
    assert manifest.entry("missing") is None
    assert len(manifest.files_with_role("loads")) == 1
    assert manifest.total_bytes > 0


# ======================================================================
# Identifiers
# ======================================================================


@pytest.mark.parametrize(
    ("raw", "expected_prefix"),
    [
        ("p1uhs0_1247--p1udt12703", "p1uhs0-1247-p1udt12703"),
        ("load_p1ulv279_1", "load-p1ulv279-1"),
        ("p1udm971", "p1udm971"),
    ],
)
def test_sanitize_identifier(raw: str, expected_prefix: str) -> None:
    assert sanitize_identifier(raw).startswith(expected_prefix)


def test_sanitize_identifier_rejects_unusable_input() -> None:
    with pytest.raises(Exception, match="identifier-safe"):
        sanitize_identifier("///")
