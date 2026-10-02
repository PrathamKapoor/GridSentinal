"""Tests for target extraction and the target registry.

The registry is the phase's answer to "what can actually be forecast", so most of
these tests are about the *refusals*: an unsupported target must not produce data,
a missing solar column must not be substituted, and an early origin must not be
allowed to read before the start of the record.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from energy_intelligence.data.smartds import SmartDsAdapter
from energy_intelligence.ml.targets import (
    PV_ARRAY_RATING_KW,
    TARGET_REGISTRY,
    TargetStatus,
    feeder_load_target,
    load_target,
    select_customer_sample,
    solar_data_file,
    solar_target,
    supported_targets,
    target_spec,
)

from conftest import build_series


# ======================================================================
# The registry itself
# ======================================================================


def test_every_required_field_is_filled_for_every_target() -> None:
    """A status without evidence is an assertion, so every field is checked."""
    for spec in TARGET_REGISTRY:
        for field in (
            "target_id",
            "description",
            "source",
            "unit",
            "coverage",
            "missingness",
            "asset_granularity",
            "forecast_feasibility",
            "status",
            "evidence",
        ):
            assert getattr(spec, field), f"{spec.target_id} has an empty {field}"


def test_statuses_are_from_the_allowed_set() -> None:
    allowed = {
        TargetStatus.SUPPORTED,
        TargetStatus.PARTIALLY_SUPPORTED,
        TargetStatus.UNSUPPORTED,
    }
    for spec in TARGET_REGISTRY:
        assert spec.status in allowed


def test_unsupported_targets_are_the_ones_with_no_series() -> None:
    """The four absent quantities must all be marked unsupported, with reasons."""
    unsupported = {
        spec.target_id for spec in TARGET_REGISTRY if spec.status == TargetStatus.UNSUPPORTED
    }
    assert {"net_load", "battery_soc", "battery_power", "ev_demand", "wind_generation"} <= unsupported


def test_supported_targets_are_only_the_extractable_ones() -> None:
    ids = {spec.target_id for spec in supported_targets()}
    assert ids == {"customer_load", "feeder_load", "pv_generation"}


def test_pv_is_only_partially_supported_and_says_why() -> None:
    spec = target_spec("pv_generation")
    assert spec.status == TargetStatus.PARTIALLY_SUPPORTED
    assert "NOT the feeder PV fleet" in spec.evidence
    assert "G-03" in spec.evidence


def test_zero_minutes_is_recorded_for_absent_series() -> None:
    """A target with no series must not claim a resolution."""
    for spec in TARGET_REGISTRY:
        if spec.status == TargetStatus.UNSUPPORTED:
            assert spec.resolution_minutes is None, spec.target_id
            assert spec.coverage == "NOT AVAILABLE", spec.target_id


def test_unknown_target_raises_with_the_declared_list() -> None:
    with pytest.raises(KeyError, match="declared targets"):
        target_spec("wind_power_at_height")


# ======================================================================
# Per-unit normalisation
# ======================================================================


def test_series_are_stored_per_unit_and_invertible() -> None:
    series = build_series(series_count=3)
    normalised = series.normalised()
    assert normalised.shape == series.values.shape
    assert np.allclose(normalised * series.scale[:, None], series.values)
    assert np.allclose(series.scale_vector, series.scale)


def test_scale_defaults_to_ones_when_absent() -> None:
    series = build_series(series_count=2)
    bare = type(series)(
        spec=series.spec,
        values=series.values,
        series_ids=series.series_ids,
        origin=series.origin,
        timestep_minutes=series.timestep_minutes,
        provenance=series.provenance,
        quality=series.quality,
        value_origin=series.value_origin,
    )
    assert np.allclose(bare.scale_vector, 1.0)
    assert np.allclose(bare.normalised(), bare.values)


# ======================================================================
# Refusals
# ======================================================================


def test_load_target_requires_at_least_one_name(synthetic_series) -> None:
    adapter = SmartDsAdapter.__new__(SmartDsAdapter)
    with pytest.raises(ValueError, match="at least one customer name"):
        load_target(adapter, ())


def test_solar_refuses_when_no_file_is_present(tmp_path: Path, monkeypatch) -> None:
    """No solar file means no PV, not a fabricated zero series."""
    from energy_intelligence.data.smartds import SmartDsLayout

    layout = SmartDsLayout(
        root=tmp_path,
        version="v1.0",
        year="2018",
        region="AUS",
        subregion="P1U",
        scenario="base_timeseries",
        substation="s",
        feeder="f",
    )
    with pytest.raises(FileNotFoundError, match="no solar_data"):
        solar_target(layout)


def test_solar_refuses_when_two_files_are_present(tmp_path: Path) -> None:
    """Ambiguity is a refusal, not a coin flip."""
    from energy_intelligence.data.smartds import SmartDsLayout

    layout = SmartDsLayout(
        root=tmp_path,
        version="v1.0",
        year="2018",
        region="AUS",
        subregion="P1U",
        scenario="base_timeseries",
        substation="s",
        feeder="f",
    )
    layout.solar_data_dir.mkdir(parents=True)
    (layout.solar_data_dir / "a.csv").write_text("x\n", encoding="utf-8")
    (layout.solar_data_dir / "b.csv").write_text("x\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="2 solar files"):
        solar_data_file(layout)


def test_solar_rejects_a_wrong_length_file(tmp_path: Path) -> None:
    """A short file must fail loudly rather than being padded or truncated."""
    from energy_intelligence.data.smartds import SmartDsLayout

    layout = SmartDsLayout(
        root=tmp_path,
        version="v1.0",
        year="2018",
        region="AUS",
        subregion="P1U",
        scenario="base_timeseries",
        substation="s",
        feeder="f",
    )
    layout.solar_data_dir.mkdir(parents=True)
    header = (
        "Year,Month,Day,Hour,Minute,DNI,DHI,Wind Speed,Temperature,GHI,"
        "PoA Irradiance (W/m^2),kW Generated (1000 kW Array)"
    )
    (layout.solar_data_dir / "short.csv").write_text(
        header + "\n2018,1,1,0,0,0,0,1,1,0,0,0\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="interpolation is not permitted"):
        solar_target(layout)


def test_solar_rejects_a_missing_generation_column(tmp_path: Path) -> None:
    """The target must not be reconstructed from some other column."""
    from energy_intelligence.data.smartds import SmartDsLayout

    layout = SmartDsLayout(
        root=tmp_path,
        version="v1.0",
        year="2018",
        region="AUS",
        subregion="P1U",
        scenario="base_timeseries",
        substation="s",
        feeder="f",
    )
    layout.solar_data_dir.mkdir(parents=True)
    (layout.solar_data_dir / "nogen.csv").write_text(
        "Year,Month,Day,Hour,Minute,GHI\n2018,1,1,0,0,0.0\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="missing required columns"):
        solar_target(layout)


# ======================================================================
# Real data
# ======================================================================


def test_real_customer_extraction_is_complete_and_clean(real_adapter) -> None:
    adapter = SmartDsAdapter(real_adapter)
    names, report = select_customer_sample(adapter, count=8, seed=20260101)
    series = load_target(adapter, names)

    assert series.values.shape == (8, 35040)
    assert series.series_ids == tuple(sorted(names))
    assert np.all(np.isfinite(series.values))
    assert series.values.min() > 0.0
    assert series.value_origin == "DERIVED"
    assert series.spec.unit == "kW"


def test_feeder_extraction_matches_the_phase3_reconstruction(real_adapter) -> None:
    """Cross-phase consistency: the Phase 4 aggregate must equal Phase 3's.

    Phase 3 summed 1,871 per-customer series and reported 15070.2676 kW at the
    published peak. Phase 4 accumulates per profile instead, to avoid holding every
    series in memory. If the two ever disagree, one of them is wrong.
    """
    series = feeder_load_target(SmartDsAdapter(real_adapter))
    value = float(series.values[0, 19553])
    assert value == pytest.approx(15070.267606731582, abs=1e-6)
    # And the Phase 3 balance residual is preserved, not quietly closed.
    assert value - 15090.7182242643 == pytest.approx(-20.450618, abs=1e-4)


def test_feeder_scale_is_the_connected_rating(real_adapter) -> None:
    series = feeder_load_target(SmartDsAdapter(real_adapter))
    # Total rated kW across the feeder is far larger than the peak demand, which is
    # what a low load factor looks like.
    assert float(series.scale[0]) > 1000.0
    assert float(series.values.max()) < float(series.scale[0])


def test_real_solar_target_matches_the_published_column(real_adapter) -> None:
    series, weather = solar_target(real_adapter)
    assert series.values.shape == (1, 35040)
    assert series.value_origin == "OBSERVED"
    assert float(series.scale[0]) == PV_ARRAY_RATING_KW
    zeros = int(np.count_nonzero(series.values == 0.0))
    assert zeros == 18304, "the real file has 52.2% exact zeros at night"
    assert set(weather) == {
        "DNI",
        "DHI",
        "GHI",
        "PoA Irradiance (W/m^2)",
        "Wind Speed",
        "Temperature",
    }
    assert all(v.size == 35040 for v in weather.values())


def test_real_pv_shape_is_capped_at_rating(real_adapter) -> None:
    """Generation cannot exceed the array rating; a breach would mean a units bug."""
    series, _ = solar_target(real_adapter)
    assert float(series.values.max()) <= PV_ARRAY_RATING_KW


def test_customer_sampling_is_stratified_and_reproducible(real_adapter) -> None:
    adapter = SmartDsAdapter(real_adapter)
    first, report = select_customer_sample(adapter, count=12, seed=20260101)
    second, _ = select_customer_sample(adapter, count=12, seed=20260101)
    assert first == second, "the same seed must give the same sample"
    other, _ = select_customer_sample(adapter, count=12, seed=999)
    assert other != first, "a different seed must give a different sample"
    assert report["selected_commercial"] > 0, "the minority class must be represented"
    assert report["commercial_available"] == 34
    assert len(set(first)) == len(first), "no duplicates"


def test_sampling_more_than_exists_is_refused(real_adapter) -> None:
    adapter = SmartDsAdapter(real_adapter)
    with pytest.raises(ValueError, match="only 1871 exist"):
        select_customer_sample(adapter, count=5000, seed=1)


def test_unknown_customer_name_is_refused(real_adapter) -> None:
    adapter = SmartDsAdapter(real_adapter)
    with pytest.raises(ValueError, match="could not be resolved"):
        load_target(adapter, ("load_does_not_exist",))