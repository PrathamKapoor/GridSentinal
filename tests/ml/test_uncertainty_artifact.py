"""The published artifact: validation, band assignment, round-trip and domain conversion.

``ProbabilisticForecast`` is the phase's contract with everything downstream, so the tests
here are about the contract rather than about calibration. An artifact that stores the
bounds and drops the calibration status would force a consumer to reconstruct the label,
and the reconstruction would be theirs rather than the experiment's - so every field the
object validates is asserted to survive a write and read.
"""

from __future__ import annotations

import numpy as np
import pytest

from energy_intelligence.domain.enums import UncertaintyKind
from energy_intelligence.domain.forecasts import Forecast
from energy_intelligence.ml.uncertainty.artifact import (
    CALIBRATION_FAIL,
    CALIBRATION_PASS,
    CALIBRATION_UNKNOWN,
    ProbabilisticForecast,
    ProbabilisticForecastBatch,
    UncertaintyBand,
    forecast_to_domain,
    interval_to_estimate,
    load_batch,
    save_batch,
)
from energy_intelligence.ml.uncertainty.intervals import PredictionInterval

HORIZONS = (1, 4, 96)
LEVEL = 0.90


def make_batch(n_rows: int = 12, *, width_scale: float = 1.0) -> ProbabilisticForecastBatch:
    rng = np.random.default_rng(4)
    point = rng.uniform(5.0, 50.0, size=(n_rows, len(HORIZONS)))
    half = width_scale * rng.uniform(0.5, 3.0, size=(n_rows, len(HORIZONS)))
    return ProbabilisticForecastBatch(
        timestamps=tuple(f"2025-01-01T{index // 4:02d}:{(index % 4) * 15:02d}:00" for index in range(n_rows)),
        asset_ids=tuple(f"asset-load-{index}" for index in range(n_rows)),
        horizon_steps=HORIZONS,
        point_kw=point,
        lower_kw=point - half,
        upper_kw=point + half,
        nominal_level=LEVEL,
        method=("conformal_state", "conformal_state", "conformal_dispersion"),
        model_version="phase8-test-v1",
        calibration_status=(CALIBRATION_PASS, CALIBRATION_PASS, CALIBRATION_FAIL),
        coverage_error=(0.002, -0.003, -0.02),
        band_cuts=((1.0, 3.0), (1.0, 3.0), (1.0, 3.0)),
        data_quality=tuple(("ok",) for _ in range(n_rows)),
        notes="fixture batch",
    )


# ---------------------------------------------------------------------------
# ProbabilisticForecast: the per-row contract
# ---------------------------------------------------------------------------


def test_forecast_rejects_an_asset_id_outside_the_domain_pattern():
    with pytest.raises(ValueError, match="asset-"):
        ProbabilisticForecast(
            timestamp="2025-01-01T00:00:00", asset_id="load_p1", horizon_steps=1,
            point_kw=1.0, lower_kw=0.5, upper_kw=1.5, nominal_level=0.9, method="m",
        )


def test_forecast_accepts_a_prefixed_dataset_series_id():
    """The dataset's own ids are not domain ids; the prefix is what makes them usable."""
    forecast = ProbabilisticForecast(
        timestamp="2025-01-01T00:00:00", asset_id="asset-load_p1ulv14763",
        horizon_steps=96, point_kw=1.0, lower_kw=0.5, upper_kw=1.5,
        nominal_level=0.9, method="m",
    )
    assert forecast.width_kw == pytest.approx(1.0)


def test_forecast_rejects_bad_metadata():
    common = {
        "timestamp": "2025-01-01T00:00:00",
        "asset_id": "asset-a",
        "horizon_steps": 1,
        "point_kw": 1.0,
        "lower_kw": 0.5,
        "upper_kw": 1.5,
        "nominal_level": 0.9,
        "method": "m",
    }
    with pytest.raises(ValueError, match="lower_kw 2.0 exceeds upper_kw"):
        ProbabilisticForecast(**{**common, "lower_kw": 2.0})
    with pytest.raises(ValueError, match="must not be NaN"):
        ProbabilisticForecast(**{**common, "point_kw": float("nan")})
    with pytest.raises(ValueError, match="horizon_steps must be a positive int"):
        ProbabilisticForecast(**{**common, "horizon_steps": 0})
    with pytest.raises(ValueError, match="calibration_status must be one of"):
        ProbabilisticForecast(**{**common, "calibration_status": "MAYBE"})
    with pytest.raises(ValueError, match="band must be an UncertaintyBand"):
        ProbabilisticForecast(**{**common, "band": "low"})
    with pytest.raises(ValueError, match="uncertainty_kind must be an UncertaintyKind"):
        ProbabilisticForecast(**{**common, "uncertainty_kind": "combined"})


def test_render_states_the_calibration_status_and_not_just_the_interval():
    forecast = ProbabilisticForecast(
        timestamp="2025-01-01T00:00:00", asset_id="asset-a", horizon_steps=1,
        point_kw=12.5, lower_kw=9.0, upper_kw=16.0, nominal_level=0.9,
        calibration_status=CALIBRATION_PASS, coverage_error=-0.004,
        band=UncertaintyBand.HIGH, method="conformal_state",
    )
    text = forecast.render()
    assert "12.50 kW" in text
    assert "[9.00, 16.00] kW" in text
    assert "PASS" in text and "-0.004" in text
    assert "HIGH" in text
    assert forecast.to_dict()["calibration_status"] == CALIBRATION_PASS
    unmeasured = ProbabilisticForecast(
        timestamp="2025-01-01T00:00:00", asset_id="asset-a", horizon_steps=1,
        point_kw=1.0, lower_kw=0.5, upper_kw=1.5, nominal_level=0.9, method="m",
    )
    assert unmeasured.calibration_status == CALIBRATION_UNKNOWN
    assert "not evaluated" in unmeasured.render()


# ---------------------------------------------------------------------------
# The batch
# ---------------------------------------------------------------------------


def test_batch_requires_one_method_name_per_horizon():
    """One name for all three horizons would credit a horizon with a method not chosen for it."""
    batch = make_batch()
    with pytest.raises(ValueError, match="one interval procedure per horizon"):
        ProbabilisticForecastBatch(
            timestamps=batch.timestamps,
            asset_ids=batch.asset_ids,
            horizon_steps=batch.horizon_steps,
            point_kw=batch.point_kw,
            lower_kw=batch.lower_kw,
            upper_kw=batch.upper_kw,
            nominal_level=LEVEL,
            method=("conformal_state",),
            model_version="v1",
            band_cuts=((1.0, 3.0),) * 3,
        )


def test_batch_requires_one_band_cut_pair_per_horizon():
    """Widths differ 5x across these horizons; one shared pair would say nothing."""
    batch = make_batch()
    with pytest.raises(ValueError, match="one ascending"):
        ProbabilisticForecastBatch(
            timestamps=batch.timestamps,
            asset_ids=batch.asset_ids,
            horizon_steps=batch.horizon_steps,
            point_kw=batch.point_kw,
            lower_kw=batch.lower_kw,
            upper_kw=batch.upper_kw,
            nominal_level=LEVEL,
            method=batch.method,
            model_version="v1",
            band_cuts=((1.0, 3.0), (1.0, 3.0)),
        )


def test_batch_rejects_misaligned_arrays_and_counts():
    batch = make_batch()
    with pytest.raises(ValueError, match=r"point/lower/upper"):
        ProbabilisticForecastBatch(
            timestamps=batch.timestamps, asset_ids=batch.asset_ids,
            horizon_steps=batch.horizon_steps, point_kw=batch.point_kw[:, :1],
            lower_kw=batch.lower_kw, upper_kw=batch.upper_kw, nominal_level=LEVEL,
            method=batch.method, model_version="v1",
            band_cuts=((1.0, 3.0),) * 3,
        )
    with pytest.raises(ValueError, match="asset ids"):
        ProbabilisticForecastBatch(
            timestamps=batch.timestamps, asset_ids=batch.asset_ids[:2],
            horizon_steps=batch.horizon_steps, point_kw=batch.point_kw,
            lower_kw=batch.lower_kw, upper_kw=batch.upper_kw, nominal_level=LEVEL,
            method=batch.method, model_version="v1",
            band_cuts=((1.0, 3.0),) * 3,
        )
    with pytest.raises(ValueError, match="calibration_status must be empty"):
        ProbabilisticForecastBatch(
            timestamps=batch.timestamps, asset_ids=batch.asset_ids,
            horizon_steps=batch.horizon_steps, point_kw=batch.point_kw,
            lower_kw=batch.lower_kw, upper_kw=batch.upper_kw, nominal_level=LEVEL,
            method=batch.method, model_version="v1",
            calibration_status=(CALIBRATION_PASS,),
            band_cuts=((1.0, 3.0),) * 3,
        )


def test_batch_bands_come_from_the_right_horizons_cut_points():
    """A cut pair of (100, 200) on h=96 must not be read against h=1's widths."""
    batch = make_batch(n_rows=9)
    wide_cuts = ProbabilisticForecastBatch(
        timestamps=batch.timestamps, asset_ids=batch.asset_ids,
        horizon_steps=batch.horizon_steps, point_kw=batch.point_kw,
        lower_kw=batch.lower_kw, upper_kw=batch.upper_kw, nominal_level=LEVEL,
        method=batch.method, model_version="v1",
        band_cuts=((1.0, 3.0), (1.0, 3.0), (100.0, 200.0)),
    )
    # Every width is below 6 kW, so against h=96's own (100, 200) cut points all rows are LOW.
    assert all(band is UncertaintyBand.LOW for band in wide_cuts.bands(2))
    # The same widths against h=1's (1, 3) split the rows.
    assert len(set(wide_cuts.bands(0))) > 1


def test_row_exposes_the_per_horizon_method_and_status():
    batch = make_batch()
    first = batch.row(0, 2)
    assert first.method == "conformal_dispersion"
    assert first.calibration_status == CALIBRATION_FAIL
    assert first.coverage_error == pytest.approx(-0.02)
    assert first.horizon_steps == 96
    assert isinstance(first.band, UncertaintyBand)
    second = batch.row(0, 0)
    assert second.method == "conformal_state"
    assert second.calibration_status == CALIBRATION_PASS


def test_batch_width_kw_rejects_an_out_of_range_column():
    with pytest.raises(IndexError, match="horizon_index"):
        make_batch().width_kw(7)
    with pytest.raises(IndexError, match="horizon_index"):
        make_batch().bands(7)


def test_batch_to_dict_reports_every_per_horizon_field():
    payload = make_batch().to_dict()
    assert payload["method"] == ["conformal_state", "conformal_state", "conformal_dispersion"]
    assert payload["calibration_status"] == ["PASS", "PASS", "FAIL"]
    assert set(payload["band_cuts_kw"]) == {"1", "4", "96"}
    assert set(payload["mean_width_kw"]) == {"1", "4", "96"}
    assert set(payload["band_counts"]) == {"1", "4", "96"}


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------


def test_batch_survives_a_write_and_read(tmp_path):
    batch = make_batch(n_rows=25)
    path = save_batch(batch, tmp_path / "forecasts.npz")
    assert path.is_file()
    assert path.with_suffix(".json").is_file()
    restored = load_batch(path)
    assert restored.timestamps == batch.timestamps
    assert restored.asset_ids == batch.asset_ids
    assert restored.horizon_steps == batch.horizon_steps
    assert restored.method == batch.method
    assert restored.model_version == batch.model_version
    assert restored.calibration_status == batch.calibration_status
    assert restored.band_cuts == batch.band_cuts
    assert restored.data_quality == batch.data_quality
    assert restored.notes == batch.notes
    for name in ("point_kw", "lower_kw", "upper_kw"):
        assert np.allclose(getattr(restored, name), getattr(batch, name))
    for column in range(len(HORIZONS)):
        assert np.array_equal(restored.bands(column), batch.bands(column))


def test_load_batch_reports_a_missing_file_and_a_foreign_archive(tmp_path):
    from energy_intelligence.ml.uncertainty.artifact import load_batch as load

    with pytest.raises(FileNotFoundError):
        load(tmp_path / "absent.npz")
    foreign = tmp_path / "foreign.npz"
    np.savez_compressed(foreign, point_kw=np.zeros((2, 2)))
    with pytest.raises(ValueError, match="not written by save_batch"):
        load(foreign)


def test_save_batch_keeps_a_missing_coverage_error_missing(tmp_path):
    """``None`` must survive as ``None``, not come back as a NaN that prints as a number."""
    batch = make_batch()
    stripped = ProbabilisticForecastBatch(
        timestamps=batch.timestamps, asset_ids=batch.asset_ids,
        horizon_steps=batch.horizon_steps, point_kw=batch.point_kw,
        lower_kw=batch.lower_kw, upper_kw=batch.upper_kw, nominal_level=LEVEL,
        method=batch.method, model_version=batch.model_version,
        calibration_status=batch.calibration_status,
        band_cuts=batch.band_cuts,
    )
    restored = load_batch(save_batch(stripped, tmp_path / "stripped.npz"))
    assert restored.coverage_error == ()
    assert restored.row(0, 0).calibration_status == CALIBRATION_PASS
    assert restored.row(0, 0).coverage_error is None


# ---------------------------------------------------------------------------
# The join to the Phase 2 domain
# ---------------------------------------------------------------------------


def test_interval_to_estimate_carries_a_paired_interval():
    """The domain container had to be repaired to hold this; the repair is exercised here."""
    interval = PredictionInterval(
        lower_kw=np.array([[9.0]]), upper_kw=np.array([[16.0]]),
        nominal_level=0.9, method="conformal_state",
    )
    estimate = interval_to_estimate(
        interval=interval, kind=UncertaintyKind.COMBINED
    )
    assert estimate.lower_bound == pytest.approx(9.0)
    assert estimate.upper_bound == pytest.approx(16.0)
    assert estimate.method == "conformal_state"
    with pytest.raises(ValueError, match="single forecast"):
        interval_to_estimate(
            interval=PredictionInterval(
                lower_kw=np.zeros((2, 1)), upper_kw=np.ones((2, 1)),
                nominal_level=0.9, method="m",
            )
        )


def test_forecast_to_domain_produces_a_populated_forecast():
    forecast = make_batch().row(3, 1)
    domain = forecast_to_domain(forecast)
    assert isinstance(domain, Forecast)
    assert domain.uncertainty is not None
    assert domain.uncertainty.lower_bound == pytest.approx(forecast.lower_kw)
    assert domain.uncertainty.upper_bound == pytest.approx(forecast.upper_kw)
    assert domain.uncertainty.method == forecast.method
    assert domain.horizon_steps == forecast.horizon_steps
    assert float(domain.value.value) == pytest.approx(forecast.point_kw)
    assert domain.provenance.processing[0].name == "interval_construction"
    assert "PASS" in domain.provenance.processing[0].detail


def test_forecast_to_domain_refuses_a_forecast_issued_after_it_describes():
    forecast = make_batch().row(0, 0)
    with pytest.raises(ValueError, match="strictly before"):
        forecast_to_domain(
            forecast, issued_at="2030-01-01T00:00:00"
        )


def test_forecast_to_domain_defaults_the_issuance_to_the_horizon():
    forecast = make_batch().row(0, 2)
    domain = forecast_to_domain(forecast)
    assert domain.issued_at < domain.target_time
    assert domain.forecast_id.value.startswith("fc-load-")