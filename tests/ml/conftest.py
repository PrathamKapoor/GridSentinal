"""Fixtures for the Phase 4 tests.

Most tests here use a **synthetic** series rather than the real dataset. That is a
deliberate exception to the Phase 3 rule, and the reason is specific: a temporal
test needs to know the answer. "Is index 500 of this series 7.25?" is only
checkable if the series is constructed to be checkable, and a real SMART-DS series
would force every assertion into a tolerance.

What is **not** relaxed: anything that could hide a fidelity problem. Tests that
concern the real files - target extraction, the cross-phase consistency check, the
PV column contract - use the acquired dataset and skip when it is absent, exactly
as Phase 3 does.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from energy_intelligence.data.smartds import SmartDsLayout
from energy_intelligence.domain.enums import QualityFlag
from energy_intelligence.domain.provenance import Provenance, ProvenanceEvent, SourceReference
from energy_intelligence.domain.quality import DataQuality
from energy_intelligence.ml.targets import TargetSeries, target_spec


def _provenance() -> Provenance:
    return Provenance(
        source=SourceReference(
            source_id="synthetic",
            dataset="ml_target",
            locator="synthetic/for-tests",
            version="0",
        ),
        events=(
            ProvenanceEvent(
                timestamp="2018-01-01T00:00:00+00:00",
                event="synthetic_series_constructed_for_tests",
                detail="not SMART-DS data",
            ),
        ),
    )


def build_series(
    *,
    series_count: int = 3,
    steps: int = 4000,
    target_id: str = "customer_load",
    seed: int = 7,
    with_daily_cycle: bool = True,
) -> TargetSeries:
    """A synthetic series with a known, checkable structure.

    ``value(t) = level + daily_shape * sin(2*pi*t/96)`` with a small deterministic
    noise term, so a test can assert both the shape and the exactness of lags.

    Args:
        series_count: Number of rows.
        steps: Length of the time axis.
        target_id: Which declared target to label the series with.
        seed: Seed for the noise term.
        with_daily_cycle: When False, produces a flat series, which is the easiest
            possible case for persistence.

    Returns:
        The synthetic :class:`TargetSeries`.
    """
    t = np.arange(steps, dtype=np.float64)
    rng = np.random.default_rng(seed)
    rows = []
    levels = []
    for index in range(series_count):
        level = 0.2 + 0.1 * index
        levels.append(level)
        if with_daily_cycle:
            shape = 0.1 * np.sin(2.0 * np.pi * t / 96.0 + index)
        else:
            shape = np.zeros_like(t)
        noise = rng.normal(0.0, 0.002, size=steps)
        rows.append(level + shape + noise)
    values = np.stack(rows, axis=0)
    scale = np.asarray([1.0 + 0.5 * i for i in range(series_count)], dtype=np.float64)
    return TargetSeries(
        spec=target_spec(target_id),
        values=values * scale[:, None],
        series_ids=tuple(f"synthetic-{i}" for i in range(series_count)),
        origin=datetime(2018, 1, 1, tzinfo=timezone.utc).isoformat(),
        timestep_minutes=15,
        provenance=_provenance(),
        quality=DataQuality(
            flags=frozenset({QualityFlag.OK}),
            detail="synthetic fixture, not SMART-DS",
        ),
        value_origin="SYNTHETIC",
        scale=scale,
        scale_source="synthetic test scale",
        notes=("synthetic series constructed by tests/ml/conftest.py",),
    )


def build_weather(steps: int = 4000, seed: int = 11) -> dict[str, np.ndarray]:
    """Synthetic weather arrays shaped like the real solar file's columns."""
    t = np.arange(steps, dtype=np.float64)
    rng = np.random.default_rng(seed)
    ghi = np.clip(800.0 * np.sin(np.pi * (t % 96) / 96.0), 0.0, None)
    return {
        "DNI": ghi * 0.9 + rng.normal(0, 5.0, steps),
        "DHI": ghi * 0.2 + rng.normal(0, 3.0, steps),
        "GHI": ghi,
        "PoA Irradiance (W/m^2)": ghi * 1.05,
        "Wind Speed": np.abs(rng.normal(4.0, 1.5, steps)),
        "Temperature": 15.0 + 10.0 * np.sin(2 * np.pi * t / (96 * 365)),
    }


@pytest.fixture
def synthetic_series() -> TargetSeries:
    """A three-series synthetic target."""
    return build_series()


@pytest.fixture
def real_dataset_available() -> bool:
    """Whether the acquired SMART-DS subset is present locally."""
    return Path("data/raw/smart_ds/v1.0/2018/AUS/P1U/profiles").is_dir() and Path(
        "data/raw/smart_ds/v1.0/2018/AUS/P1U/scenarios/base_timeseries/opendss"
        "/p1uhs0_1247/p1uhs0_1247--p1udt12703/Loads.dss"
    ).is_file()


@pytest.fixture
def real_adapter(real_dataset_available: bool) -> SmartDsLayout:
    """The real feeder layout, skipping when the dataset is absent."""
    if not real_dataset_available:
        pytest.skip("full SMART-DS dataset not acquired")
    from energy_intelligence.config.data import load_data_config

    config = load_data_config()
    return SmartDsLayout(
        root=config.raw_root,
        version=config.version,
        year=config.year,
        region=config.region,
        subregion=config.subregion,
        scenario=config.scenario,
        substation=config.substation,
        feeder=config.feeder,
    )