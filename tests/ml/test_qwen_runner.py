"""End-to-end tests for the Phase 6 runner and the feature cache.

The runner holds the verdict, the regime analysis and the artifact writing, and until
now it had only ever been exercised by a five-hour script. These tests drive it on a
synthetic series with a stub backbone, so the logic is checked without a 3.44 GB
download.

The stub backbone is deliberately **not** a language model. What is under test here is
the plumbing - slicing, scaling, paired statistics, verdict wording, artifacts - and a
real trunk would make every test slow while testing the same plumbing.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from conftest import build_series

from energy_intelligence.ml.qwen.experiment import (
    run_phase6_experiment,
)
from energy_intelligence.ml.qwen.features import (
    FeatureCache,
    extract_and_cache,
    select_subsample,
)
from energy_intelligence.ml.qwen.probe import ProbeTrainingConfig
from energy_intelligence.ml.qwen.representation import EnergyRepresentation
from energy_intelligence.ml.sequence import SequenceConfig, build_sequence_index, gather_sequences

HORIZONS = (1, 4, 96)
FEATURE_NAMES = (
    "hour_of_day", "day_of_week", "day_of_year", "is_weekend", "value_at_origin",
    "lag_1", "lag_4", "lag_96", "lag_672", "roll_mean_96", "roll_std_96",
    "ramp_1", "roll_mean_same_hour_7d",
)


# 35,040 native steps is what the real dataset has. A validation segment must be
# wider than lookback + max_horizon for the split to contain any origins at all:
# 15% of 12,000 = 1,800 > 672 + 96, whereas 15% of 4,000 = 600 does not, and the
# index correctly refuses to place a window that would straddle a boundary.
STEPS = 12_000


def _index(series):
    return build_sequence_index(
        series_count=len(series.series_ids),
        total_steps=series.values.shape[1],
        config=SequenceConfig(
            lookback_steps=672, token_stride=4, horizons=HORIZONS, train_origin_stride=8
        ),
        target_id=series.spec.target_id,
        value_origin=series.value_origin,
        series_ids=series.series_ids,
        source="synthetic",
    )


def _fake_cache(series, index, subsample, *, layers=(4, 8), seed: int = 3):
    """A feature cache built from the windows themselves.

    Not a model - a deterministic function of the input window, which is exactly what
    the runner needs in order to be exercised end to end. Targets and persistence come
    from the real gatherer, so the evaluation path is genuine.
    """
    rows = np.concatenate([subsample.train, subsample.validation, subsample.test])
    windows, targets, persistence = gather_sequences(
        series.normalised().astype(np.float32), index, rows
    )
    flat = windows.transpose(0, 2, 1).reshape(windows.shape[0], -1)
    rng = np.random.default_rng(seed)
    features = {
        layer: (flat[:, :64] + rng.normal(scale=0.01, size=(flat.shape[0], 64))).astype(
            np.float32
        )
        for layer in layers
    }
    return FeatureCache(
        version="fc-test",
        layers=tuple(layers),
        features=features,
        targets=targets,
        persistence=persistence,
        origins=index.origins[rows],
        series=index.series[rows],
        row_scale=series.scale_vector[index.series[rows]],
    )


def _phase4_features(series, index, rows):
    from energy_intelligence.ml.dataset import build_feature_matrix

    matrix = build_feature_matrix(
        series.normalised().astype(np.float32),
        index.origins[rows],
        lookback=index.config.lookback_steps,
        feature_names=FEATURE_NAMES,
        series_index=index.series[rows],
    )
    return matrix, FEATURE_NAMES


def _run_empty(tmp_path: Path):
    """Drive the runner with a validation slice of zero rows."""
    series = build_series(series_count=4, steps=STEPS)
    index = _index(series)
    subsample = select_subsample(
        index, train_rows=400, validation_rows=200, test_rows=200
    )
    cache = _fake_cache(series, index, subsample)
    empty = FeatureCache(
        version=cache.version,
        layers=cache.layers,
        features=cache.features,
        targets=cache.targets,
        persistence=cache.persistence,
        origins=cache.origins,
        series=cache.series,
        row_scale=cache.row_scale,
    )
    # Shrink the validation block to nothing by lying about the split sizes.
    original = select_subsample
    try:
        import energy_intelligence.ml.qwen.experiment as experiment_module

        slices = experiment_module._split_slices
        experiment_module._split_slices = lambda sub: {
            "train": slice(0, 400),
            "validation": slice(400, 400),
            "test": slice(400, 800),
        }
        return run_phase6_experiment(
            experiment_id="phase6-empty",
            index=index,
            values=series.normalised().astype(np.float32),
            scale_vector=series.scale_vector,
            series_ids=series.series_ids,
            series_origin_iso=series.origin,
            cache=empty,
            subsample=subsample,
            feature_layer=8,
            representation_version="er-test",
            horizons=HORIZONS,
            probe_layers=(0,),
            basis="pca",
            basis_hidden_size=32,
            series_embedding_dim=4,
            training=ProbeTrainingConfig(epochs=1, torch_threads=2, seed=1, batch_size=64),
            phase4_reference={h: {"mae": 0.5} for h in HORIZONS},
            phase5_reference={h: {"mae": 0.5} for h in HORIZONS},
            phase4_features=_phase4_features(series, index, subsample.train),
            scaler=None,
            phase5_checkpoint=None,
            artifact_root=tmp_path / "qwen-empty",
            backbone_kind="pretrained",
        )
    finally:
        experiment_module._split_slices = slices
        del original


def _run(tmp_path: Path, *, train_rows=400, val_rows=200, test_rows=200, **overrides):
    series = build_series(series_count=4, steps=STEPS)
    index = _index(series)
    subsample = select_subsample(
        index, train_rows=train_rows, validation_rows=val_rows, test_rows=test_rows
    )
    cache = _fake_cache(series, index, subsample)
    kwargs = dict(
        experiment_id="phase6-test",
        index=index,
        values=series.normalised().astype(np.float32),
        scale_vector=series.scale_vector,
        series_ids=series.series_ids,
        series_origin_iso=series.origin,
        cache=cache,
        subsample=subsample,
        feature_layer=8,
        representation_version="er-test",
        horizons=HORIZONS,
        probe_layers=(0,),
        basis="pca",
        basis_hidden_size=32,
        series_embedding_dim=4,
        training=ProbeTrainingConfig(
            epochs=3, patience=3, torch_threads=2, seed=1, batch_size=64
        ),
        phase4_reference={h: {"mae": 0.5} for h in HORIZONS},
        phase5_reference={h: {"mae": 0.5} for h in HORIZONS},
        phase4_features=_phase4_features(series, index, subsample.train),
        scaler=None,
        phase5_checkpoint=None,
        artifact_root=tmp_path / "qwen-test",
        backbone_kind="pretrained",
    )
    kwargs.update(overrides)
    return run_phase6_experiment(**kwargs), series, index, subsample


# ======================================================================
# End to end
# ======================================================================


def test_the_runner_produces_a_complete_result_and_artifacts(tmp_path: Path) -> None:
    result, _series, index, subsample = _run(tmp_path)

    assert set(result.metrics_by_horizon) == set(HORIZONS)
    for horizon in HORIZONS:
        metrics = result.metrics_by_horizon[horizon]
        assert metrics["n"] == subsample.test.size
        assert np.isfinite(metrics["mae"]) and metrics["mae"] >= 0
        # R^2 has no lower bound: a model worse than predicting the mean scores
        # negative without being broken, so only finiteness is asserted here.
        assert np.isfinite(metrics["r2"])

    root = tmp_path / "qwen-test"
    assert (root / "result.json").is_file()
    assert (root / "summary.md").is_file()
    assert (root / "probe.pt").is_file()

    payload = json.loads((root / "result.json").read_text(encoding="utf-8"))
    assert payload["dataset_version"] == index.version
    assert payload["subsample_version"] == subsample.version()
    assert "verdict" in payload and payload["verdict"]


def test_the_runner_records_provenance_for_reproducibility(tmp_path: Path) -> None:
    """Section 38: every field needed to repeat the experiment must be recorded."""
    result, _s, index, subsample = _run(tmp_path)
    assert result.dataset_version == index.version
    assert result.subsample_version == subsample.version()
    assert result.representation_version == "er-test"
    compute = result.compute
    assert compute["backbone_parameters"] == 1_720_574_976
    assert compute["backbone_trainable_parameters"] == 0
    assert compute["backbone_frozen_parameters"] == 1_720_574_976
    assert compute["probe_parameters"] > 0
    assert compute["dtype"] == "float32"
    assert compute["head_training_seconds"] >= 0
    assert result.probe["seed"] if "seed" in result.probe else True


def test_the_runner_reports_a_refitted_gbm_on_the_same_rows(tmp_path: Path) -> None:
    result, _s, _i, subsample = _run(tmp_path)
    entry = result.improvement[HORIZONS[0]]
    assert "classical_hist_gbm" in entry["vs"]
    assert entry["vs"]["classical_hist_gbm"]["baseline_mae_kw"] > 0
    assert entry["vs"]["persistence"]["baseline_mae_kw"] > 0
    # The GBM must be close to, not wildly different from, persistence: a gross
    # mismatch here is what a per-unit/kW unit bug looks like.
    ratio = (
        entry["vs"]["classical_hist_gbm"]["baseline_mae_kw"]
        / entry["vs"]["persistence"]["baseline_mae_kw"]
    )
    assert 0.2 < ratio < 5.0, f"GBM/persistence ratio {ratio} suggests a unit bug"
    assert entry["subsample_absolute_mae_is_not_the_published_mae"] is True


def test_the_runner_reports_regimes_for_every_model(tmp_path: Path) -> None:
    result, _s, _i, _sub = _run(tmp_path)
    for axis in ("load_level", "load_ramp"):
        regimes = result.regime[axis]["regimes"]
        assert regimes
        for entry in regimes.values():
            assert entry["n"] > 0
            assert "qwen_probe" in entry
            assert "persistence" in entry
            assert entry["classical_hist_gbm"] >= 0
    assert "not a deployable routing signal" in result.regime["note"]


def test_the_runner_reports_an_error_distribution_for_every_model(tmp_path: Path) -> None:
    result, _s, _i, _sub = _run(tmp_path)
    distributions = result.errors["distribution_by_model"]
    assert "qwen_probe" in distributions
    assert "persistence" in distributions
    assert "classical_hist_gbm" in distributions
    for stats in distributions.values():
        assert stats["median"] <= stats["p90"] <= stats["p99"] <= stats["max"]
        assert stats["mae"] >= 0
        assert 0.0 <= stats["share_above_1kw"] <= 1.0


def test_the_runner_measures_the_representation_at_every_cached_layer(tmp_path: Path) -> None:
    result, _s, _i, _sub = _run(tmp_path)
    assert set(result.diagnostics) == {"layer_4", "layer_8"}
    for stats in result.diagnostics.values():
        assert stats["hidden_norm_mean"] > 0
        assert 1.0 <= stats["participation_ratio"] <= 2048.0


def test_the_runner_survives_a_missing_phase5_checkpoint(tmp_path: Path) -> None:
    """The TCN comparison is valuable, not mandatory; its absence must not crash."""
    result, _s, _i, _sub = _run(tmp_path, phase5_checkpoint=tmp_path / "absent.pt")
    assert "phase5_tcn" not in result.improvement[HORIZONS[0]]["vs"]
    assert (tmp_path / "qwen-test" / "result.json").is_file()


def test_the_runner_rejects_a_probe_that_was_never_selected(tmp_path: Path) -> None:
    """A zero-epoch run must not silently report untrained weights as a measurement."""
    with pytest.raises(ValueError, match="epochs must be at least 1"):
        _run(tmp_path, training=ProbeTrainingConfig(epochs=0, torch_threads=2))


def test_the_runner_rejects_an_empty_training_or_validation_split(tmp_path: Path) -> None:
    """An empty split must fail loudly rather than produce a meaningless MAE."""
    with pytest.raises(ValueError, match="training and validation rows"):
        _run_empty(tmp_path)


# ======================================================================
# The verdict wording
# ======================================================================


@pytest.mark.parametrize(
    "improvements,expected_fragment",
    [
        ([10.0, 10.0, 10.0], "beats GBM at every horizon"),
        ([-10.0, -10.0, -10.0], "loses to GBM at every horizon"),
        ([10.0, 10.0, -10.0], "beats GBM at 2 of 3 horizons"),
    ],
)
def test_the_verdict_counts_horizons_rather_than_averaging(
    improvements, expected_fragment
) -> None:
    """Phase 5 already established that a single averaged number hides the answer."""

    def block(value):
        return {
            "classical_hist_gbm": {
                "improvement_pct": value,
                "paired": {"excludes_zero": True},
            }
        }

    payload = {
        horizon: {"vs": block(value), "paired_vs_gbm": {"excludes_zero": True}}
        for horizon, value in zip(HORIZONS, improvements)
    }
    verdict = __import__(
        "energy_intelligence.ml.qwen.experiment", fromlist=["_verdict"]
    )._verdict(payload, HORIZONS)
    assert expected_fragment in verdict
    for horizon in HORIZONS:
        assert f"h={horizon}" in verdict


# ======================================================================
# The feature cache
# ======================================================================


class _StubBackbone(nn.Module):
    """A deterministic stand-in for the trunk, for testing the cache plumbing."""

    def __init__(self, hidden: int = 2048, depth: int = 5) -> None:
        super().__init__()
        self.depth = depth
        self.project = nn.Linear(hidden, hidden, bias=False)
        with torch.no_grad():
            self.project.weight.copy_(torch.eye(hidden) * 0.5)

    def forward(self, inputs_embeds=None, output_hidden_states=False, use_cache=False):
        hidden = inputs_embeds
        states = [hidden]
        for _ in range(self.depth):
            hidden = self.project(hidden)
            states.append(hidden)
        return type("Out", (), {"hidden_states": tuple(states), "logits": None})()


class _StubCheckpoint:
    root = Path("unused")

    def verify(self, **kwargs):
        return {}


def test_extraction_caches_to_disk_and_reuses_it(tmp_path: Path) -> None:
    """The cache is not a convenience: without it every head run re-pays the trunk."""
    series = build_series(series_count=2, steps=STEPS)
    index = _index(series)
    subsample = select_subsample(
        index, train_rows=32, validation_rows=16, test_rows=16
    )
    representation = EnergyRepresentation(patch_size=8, hidden_size=2048)
    out = tmp_path / "features.npz"

    calls = {"n": 0}

    def counting_load(*args, **kwargs):
        calls["n"] += 1
        return _StubBackbone()

    import energy_intelligence.ml.qwen.features as features_module

    original = features_module._load_backbone
    features_module._load_backbone = counting_load
    try:
        first = extract_and_cache(
            checkpoint=_StubCheckpoint(),
            index=index,
            values=series.normalised().astype(np.float32),
            scale_vector=series.scale_vector,
            subsample=subsample,
            representation=representation,
            layers=(0, 5),
            aggregate="last",
            backbone_kind="random",
            out_path=out,
            batch_size=8,
            dtype=torch.float32,
            torch_threads=2,
        )
        assert calls["n"] == 1
        assert out.is_file()
        second = extract_and_cache(
            checkpoint=_StubCheckpoint(),
            index=index,
            values=series.normalised().astype(np.float32),
            scale_vector=series.scale_vector,
            subsample=subsample,
            representation=representation,
            layers=(0, 5),
            aggregate="last",
            backbone_kind="random",
            out_path=out,
            batch_size=8,
            dtype=torch.float32,
            torch_threads=2,
        )
    finally:
        features_module._load_backbone = original

    assert calls["n"] == 1, "the second extraction should have hit the cache"
    assert first.version == second.version
    for layer in (0, 5):
        assert np.array_equal(first.features[layer], second.features[layer])
    assert first.targets.shape == (subsample.train.size + subsample.validation.size + subsample.test.size, 3)


def test_the_cache_key_changes_with_the_backbone_kind(tmp_path: Path) -> None:
    """A pretrained and a random cache must never overwrite each other."""
    from energy_intelligence.ml.qwen.features import _cache_key

    representation = EnergyRepresentation(patch_size=8)
    base = dict(
        subsample_version="ss-1",
        representation=representation,
        in_channels=6,
        layers=(0, 28),
        aggregate="last",
    )
    pretrained = _cache_key(backbone="pretrained", **base)
    random = _cache_key(backbone="random", **base)
    assert pretrained != random
    assert _cache_key(backbone="pretrained", **base) == pretrained


def test_extraction_pools_the_requested_aggregation(tmp_path: Path) -> None:
    """`last` and `mean` must not silently return the same thing."""
    series = build_series(series_count=2, steps=STEPS)
    index = _index(series)
    subsample = select_subsample(index, train_rows=16, validation_rows=8, test_rows=8)

    import energy_intelligence.ml.qwen.features as features_module

    original = features_module._load_backbone
    features_module._load_backbone = lambda *a, **k: _StubBackbone()
    try:
        common = dict(
            checkpoint=_StubCheckpoint(),
            index=index,
            values=series.normalised().astype(np.float32),
            scale_vector=series.scale_vector,
            subsample=subsample,
            representation=EnergyRepresentation(patch_size=8, hidden_size=2048),
            layers=(5,),
            backbone_kind="random",
            batch_size=8,
            dtype=torch.float32,
            torch_threads=2,
        )
        last = extract_and_cache(aggregate="last", out_path=tmp_path / "a.npz", **common)
        mean = extract_and_cache(aggregate="mean", out_path=tmp_path / "b.npz", **common)
    finally:
        features_module._load_backbone = original

    assert not np.allclose(last.features[5], mean.features[5])


def test_extraction_refuses_an_unknown_aggregation(tmp_path: Path) -> None:
    series = build_series(series_count=2, steps=STEPS)
    index = _index(series)
    subsample = select_subsample(index, train_rows=8, validation_rows=4, test_rows=4)

    import energy_intelligence.ml.qwen.features as features_module

    original = features_module._load_backbone
    features_module._load_backbone = lambda *a, **k: _StubBackbone()
    try:
        with pytest.raises(ValueError, match="unknown aggregate"):
            extract_and_cache(
                checkpoint=_StubCheckpoint(),
                index=index,
                values=series.normalised().astype(np.float32),
                scale_vector=series.scale_vector,
                subsample=subsample,
                representation=EnergyRepresentation(patch_size=8, hidden_size=2048),
                layers=(5,),
                aggregate="max",
                backbone_kind="random",
                out_path=tmp_path / "c.npz",
                batch_size=8,
                dtype=torch.float32,
                torch_threads=2,
            )
    finally:
        features_module._load_backbone = original
