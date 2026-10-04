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

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from energy_intelligence.data.smartds import SmartDsLayout
from energy_intelligence.domain.enums import QualityFlag
from energy_intelligence.domain.provenance import Provenance, ProvenanceEvent, SourceReference
from energy_intelligence.domain.quality import DataQuality
from energy_intelligence.ml.router.experts import (
    EvaluationPanel,
    ExpertForecast,
    ExpertPool,
)
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


# ---------------------------------------------------------------------------
# Phase 7: the expert pool's stubs and its hand-built panel
# ---------------------------------------------------------------------------
#
# The experts here are stubs with known closed-form behaviour rather than the real
# persistence/GBM/TCN. "Does the router combine three experts correctly" has an
# exact answer when each expert is a closed form; real models would turn every
# assertion into a tolerance and hide the arithmetic under test.

HORIZONS: tuple[int, ...] = (1, 4, 96)
SERIES_COUNT = 3
ORIGIN_BASE = 200
#: Unit stride. The lagged-error feature reaches back ``h + 1`` steps, so at h=96 it looks
#: 97 origins into the past. A panel on a coarser stride simply has no predecessor for the
#: short horizons and the feature would be untestable rather than wrong.
ORIGIN_STEP = 1


class StubExpert:
    """An expert whose forecast is a known closed form of the panel it is asked about.

    ``forecast[row, column] = y(t) + offset + 0.5 * column``, where ``y(t)`` is the row's
    own demand at the origin. Distinct per expert and in the panel's own kW scale, so a
    mixture is verifiable to the last decimal and a per-unit normalisation is testable.
    """

    def __init__(self, panel: "EvaluationPanel", name: str, offset: float) -> None:
        self.name = name
        self.offset = float(offset)
        self._panel = panel
        self.calls: list[int] = []

    def predict(self, positions: np.ndarray) -> ExpertForecast:
        positions = np.asarray(positions, dtype=np.int64)
        self.calls.append(int(positions.size))
        base = self._panel.persistence_kw[positions, 0][:, None] + 0.5 * np.arange(
            len(HORIZONS)
        )[None, :]
        return ExpertForecast(
            name=self.name,
            forecast_kw=base + self.offset,
            horizons=HORIZONS,
            provenance={"stub": True, "offset": self.offset},
        )


class ConstantExpert:
    """Returns a fixed array regardless of position, for interface-error tests."""

    def __init__(self, name: str, value: float = 1.0) -> None:
        self.name = name
        self.value = float(value)

    def predict(self, positions: np.ndarray) -> ExpertForecast:
        positions = np.asarray(positions, dtype=np.int64)
        return ExpertForecast(
            name=self.name,
            forecast_kw=np.full((positions.size, len(HORIZONS)), self.value),
            horizons=HORIZONS,
        )


def _panel_values(series_count: int, steps: int) -> np.ndarray:
    """The shared per-unit series both the panel and the ``values`` fixture come from.

    They must be the same array. The causality check recomputes the realised targets from
    a (poisoned) copy of these values and compares them, so a fixture whose panel targets
    were invented separately would make every lagged column look like a leak.
    """
    t = np.arange(steps, dtype=np.float64)
    rows = [
        0.5 + 0.2 * np.sin(2.0 * np.pi * t / 96.0 + index) + 0.001 * t / 100.0
        for index in range(series_count)
    ]
    return np.stack(rows, axis=0).astype(np.float32)


def build_test_panel(
    *,
    n_rows: int = 300,
    series_count: int = SERIES_COUNT,
    steps: int = 1200,
) -> EvaluationPanel:
    """A panel in (origin, series) order, which the lagged-error lookup requires.

    ``actual_kw`` is ``values[series, origin + horizon] * rated_kw`` for the series from
    :func:`_panel_values`, which is the invariant the leakage check depends on.
    """
    origins: list[int] = []
    series: list[int] = []
    for step in range(n_rows // series_count):
        for index in range(series_count):
            origins.append(ORIGIN_BASE + step * ORIGIN_STEP)
            series.append(index)
    origins_arr = np.asarray(origins, dtype=np.int64)
    series_arr = np.asarray(series, dtype=np.int64)
    n = origins_arr.size
    scale = np.asarray([10.0 + 5.0 * i for i in range(series_count)], dtype=np.float64)[series_arr]
    per_unit = _panel_values(series_count, steps)
    horizon_steps = np.asarray(HORIZONS, dtype=np.int64)
    actual = (
        per_unit[series_arr[:, None], origins_arr[:, None] + horizon_steps[None, :]]
        * scale[:, None]
    ).astype(np.float64)
    # Persistence is ``y(t)`` for every horizon, so one column repeated. Making it vary by
    # horizon here would hide the exact property several tests assert.
    at_origin = per_unit[series_arr, origins_arr].astype(np.float64) * scale
    half = n // 2
    return EvaluationPanel(
        index_rows=np.arange(n, dtype=np.int64),
        split=np.asarray(["validation"] * half + ["test"] * (n - half), dtype=object),
        split_slices={"validation": slice(0, half), "test": slice(half, n)},
        origins=origins_arr,
        series=series_arr,
        row_scale=scale,
        actual_kw=actual,
        persistence_kw=np.repeat(at_origin[:, None], len(HORIZONS), axis=1),
        horizons=HORIZONS,
    )


def build_test_values(panel: EvaluationPanel, *, steps: int = 1200) -> np.ndarray:
    """The per-unit series this panel's targets were built from."""
    return _panel_values(int(panel.series.max()) + 1, steps)


def build_stub_pool(panel: EvaluationPanel) -> ExpertPool:
    """Persistence plus two stubs, in the pool's real reporting order."""
    from energy_intelligence.ml.router.experts import PersistenceExpert

    return ExpertPool(
        [
            PersistenceExpert(panel),
            StubExpert(panel, "classical_hist_gbm", 1.0),
            StubExpert(panel, "phase5_tcn", 2.0),
        ],
        HORIZONS,
    )


def build_forecasts(panel: EvaluationPanel, n_experts: int = 3) -> np.ndarray:
    """A ``[n_experts, N, horizons]`` block with known, distinct, panel-scaled forecasts."""
    base = panel.persistence_kw[:, 0][:, None] + 0.5 * np.arange(len(HORIZONS))[None, :]
    return np.stack([base + 1.0 * index for index in range(n_experts)], axis=0)


@pytest.fixture
def panel() -> EvaluationPanel:
    return build_test_panel()


@pytest.fixture
def values(panel: EvaluationPanel) -> np.ndarray:
    return build_test_values(panel)


@pytest.fixture
def forecasts(panel: EvaluationPanel) -> np.ndarray:
    return build_forecasts(panel)


@pytest.fixture
def stub_pool(panel: EvaluationPanel) -> ExpertPool:
    return build_stub_pool(panel)


@pytest.fixture
def train_rows(panel: EvaluationPanel) -> np.ndarray:
    return np.arange(0, panel.split_slices["validation"].stop)


@pytest.fixture
def test_rows(panel: EvaluationPanel) -> np.ndarray:
    return np.arange(panel.split_slices["test"].start, panel.n_rows)


@pytest.fixture
def stub_expert(panel: EvaluationPanel):
    """Factory for an expert with known closed-form behaviour.

    Returns a callable ``(name, offset) -> StubExpert``. A factory rather than a fixed
    instance because most tests need several experts with different offsets, and several
    need deliberately misbehaving ones.
    """
    return lambda name, offset: StubExpert(panel, name, offset)


@pytest.fixture
def constant_expert():
    """Factory for an expert that ignores its input and returns a constant."""
    return ConstantExpert


@pytest.fixture
def expert_pool():
    """Factory for an :class:`ExpertPool` over any list of experts."""
    return ExpertPool


# ---------------------------------------------------------------------------
# Phase 8: the uncertainty cache and the Phase 7 artifact it recovers weights from
# ---------------------------------------------------------------------------
#
# The uncertainty runner reads two things it cannot compute for itself: Phase 7's
# fixed-ensemble weights and its published test MAE. Both are faked here from the same
# stub experts the Phase 7 tests use, so a Phase 8 test can assert that a weight mismatch
# is *caught* without depending on a 179,520-row real run.

#: Phase 7's fitted fixed-ensemble weights, matching the stub pool's names.
PHASE7_STUB_WEIGHTS: dict[str, dict[str, float]] = {
    "1": {"persistence": 0.16, "classical_hist_gbm": 0.34, "phase5_tcn": 0.50},
    "4": {"persistence": 0.00, "classical_hist_gbm": 0.34, "phase5_tcn": 0.66},
    "96": {"persistence": 0.00, "classical_hist_gbm": 0.26, "phase5_tcn": 0.74},
}

EXPERT_NAMES: tuple[str, ...] = ("persistence", "classical_hist_gbm", "phase5_tcn")


def build_uncertainty_cache(panel: EvaluationPanel, *, seed: int = 3):
    """An :class:`ExpertCache` whose forecasts are the stub pool's, plus small noise.

    The noise is what makes the calibration problem non-trivial: without it every method
    would produce a near-zero-width interval and the coverage assertions would be testing
    nothing.
    """
    from energy_intelligence.ml.router.cache import ExpertCache

    rng = np.random.default_rng(seed)
    forecasts = build_forecasts(panel)
    noise = np.stack(
        [
            rng.normal(0.0, 0.10 * panel.row_scale[:, None], size=forecasts.shape[1:])
            for _ in forecasts
        ]
    )
    half = panel.split_slices["validation"].stop
    bounds = np.array([[0, half], [half, panel.n_rows]], dtype=np.int64)
    return ExpertCache(
        forecasts=forecasts + noise,
        actual_kw=np.asarray(panel.actual_kw, dtype=np.float64),
        persistence_kw=np.asarray(panel.persistence_kw, dtype=np.float64),
        row_scale=np.asarray(panel.row_scale, dtype=np.float64),
        origins=np.asarray(panel.origins, dtype=np.int64),
        series=np.asarray(panel.series, dtype=np.int64),
        index_rows=np.asarray(panel.index_rows, dtype=np.int64),
        horizons=HORIZONS,
        expert_names=EXPERT_NAMES,
        split_names=("validation", "test"),
        split_bounds=bounds,
        metadata={"synthetic": True, "index_version": "synthetic-v0"},
    )


def write_phase7_result(
    path: Path, *, weights: dict[str, dict[str, float]] | None = None
) -> Path:
    """Write a minimal Phase 7 ``result.json`` carrying ``weights``.

    Only ``routing.fixed_ensemble_weights`` and ``methods`` are populated, because those
    are the two parts the Phase 8 runner reads.
    """
    recorded = weights if weights is not None else PHASE7_STUB_WEIGHTS
    payload = {
        "experiment_id": "phase7-stub",
        "dataset_version": "synthetic-v0",
        "routing": {"fixed_ensemble_weights": recorded},
        "methods": {
            name: {str(h): {"mae": 0.5 + 0.1 * h} for h in HORIZONS}
            for name in (*EXPERT_NAMES, "fixed_ensemble")
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def build_uncertainty_config(**overrides):
    """A valid :class:`UncertaintyExperimentConfig`, with small budgets for speed."""
    from energy_intelligence.config.uncertainty import UncertaintyExperimentConfig

    base = {
        "target": "customer_load",
        "label": "uncertainty-test",
        "customer_count": 40,
        "seed": 20260101,
        "torch_threads": 1,
        "lookback_steps": 672,
        "token_stride": 4,
        "train_fraction": 0.70,
        "validation_fraction": 0.15,
        "horizons": HORIZONS,
        "fixed_ensemble_weights": {
            int(h): dict(entry) for h, entry in PHASE7_STUB_WEIGHTS.items()
        },
        "calibration_fit_fraction": 0.5,
        "include_past_error_features": True,
        "nominal_levels": (0.50, 0.80, 0.90, 0.95),
        "band_nominal_level": 0.90,
        "scale_max_iter": 10,
        "quantile_max_iter": 10,
        "causality_sample": 3,
        "reliability_bins": 4,
        "trace_rows": 3,
        "model_version": "phase8-test",
        "run_ablations": False,
    }
    base.update(overrides)
    return UncertaintyExperimentConfig(**base)


@pytest.fixture
def uncertainty_cache(panel: EvaluationPanel):
    return build_uncertainty_cache(panel)


@pytest.fixture
def uncertainty_values(panel: EvaluationPanel) -> np.ndarray:
    return build_test_values(panel)


@pytest.fixture
def uncertainty_config():
    return build_uncertainty_config()


@pytest.fixture
def phase7_result(tmp_path: Path) -> Path:
    return write_phase7_result(tmp_path / "phase7" / "stub" / "result.json")