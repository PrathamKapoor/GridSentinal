"""Tests for the Phase 6 configuration, the subsample and the paired statistics.

The configuration tests are strict on purpose. Phase 6's whole result rests on being
comparable to Phase 4 and Phase 5, and the easiest way to break that silently is to
change the horizons, the split or the target in a config file and let the comparison
carry on regardless.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from conftest import build_series

from energy_intelligence.config.qwen import QwenExperimentConfig, load_qwen_config
from energy_intelligence.config.schema import ConfigError, ConfigValidationError
from energy_intelligence.ml.qwen.baselines import (
    bootstrap_summary,
    gbm_baseline,
    paired_bootstrap,
    persistence_baseline,
)
from energy_intelligence.ml.qwen.experiment import representation_diagnostics
from energy_intelligence.ml.qwen.features import select_subsample
from energy_intelligence.ml.sequence import SequenceConfig, build_sequence_index

CONFIG_DIR = Path("configs")


def vars_of(sequence_config):
    import dataclasses

    return {f.name: getattr(sequence_config, f.name) for f in dataclasses.fields(sequence_config)}


def _index(series, **overrides):
    config = SequenceConfig(**overrides)
    return build_sequence_index(
        series_count=len(series.series_ids),
        total_steps=series.values.shape[1],
        config=config,
        target_id=series.spec.target_id,
        value_origin=series.value_origin,
        series_ids=series.series_ids,
        source="synthetic",
    )


def _write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "qwen.toml"
    path.write_text(body, encoding="utf-8")
    return path


BASE = """
target = "customer_load"
label = "t"
seed = 1
horizons = [1, 4, 96]
lookback_steps = 672
token_stride = 4
patch_size = 8
"""


# ======================================================================
# The shipped configuration
# ======================================================================


def test_the_shipped_qwen_config_loads_and_keeps_the_phase4_and_5_task() -> None:
    config = load_qwen_config()
    assert config.target == "customer_load"
    assert tuple(config.horizons) == (1, 4, 96)
    assert config.delta_target is True
    assert config.customer_count == 40
    # The sequence contract must be Phase 5's, or the rows are not the same rows.
    sequence = config.sequence_config()
    assert sequence.lookback_steps == 672
    assert sequence.token_stride == 4
    assert sequence.delta_target is True


def test_the_shipped_config_inherits_the_phase5_sequence_version() -> None:
    """The strongest possible statement that Phase 6 reuses Phase 5's rows."""
    from energy_intelligence.ml.sequence import sequence_version

    from energy_intelligence.ml.sequence import sequence_version

    config = load_qwen_config()
    series_ids = tuple(f"customer-{i}" for i in range(config.customer_count))
    # sq-437c16b5d925 is the version Phase 5 committed its metrics under. Matching it
    # is what makes "identical rows" a checked fact rather than an intention.
    assert sequence_version(config.sequence_config(), "customer_load", series_ids) == (
        "sq-437c16b5d925"
    )


def test_every_shipped_ablation_declares_a_question() -> None:
    """An ablation without a stated question is a run for its own sake (D-087)."""
    config = load_qwen_config()
    assert config.ablations, "the phase should declare its ablations"
    for ablation in config.ablations:
        assert ablation["name"].strip()
        assert len(ablation["question"].strip()) > 20
        assert "overrides" in ablation


def test_the_random_backbone_control_is_declared() -> None:
    """Without it, a good probe result would be uninterpretable."""
    config = load_qwen_config()
    assert config.random_backbone is True
    names = {a["name"] for a in config.ablations}
    assert "random_backbone" in names


def test_the_loss_matches_phase5s_l1_objective() -> None:
    """Optimising one thing and reporting another hides peak under-weighting."""
    assert load_qwen_config().loss == "l1"


# ======================================================================
# Strictness
# ======================================================================


def test_an_unknown_key_is_rejected_rather_than_ignored(tmp_path: Path) -> None:
    """D-007: a typo must not silently disable a setting."""
    with pytest.raises(ConfigValidationError, match="unknown key"):
        load_qwen_config(_write(tmp_path, BASE + 'typo_here = 3\n'))


def test_a_missing_required_key_is_reported(tmp_path: Path) -> None:
    with pytest.raises(ConfigValidationError, match="missing required key"):
        load_qwen_config(_write(tmp_path, 'target = "customer_load"\nlabel = "t"\n'))


def test_a_missing_file_fails_loudly(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        load_qwen_config(tmp_path / "absent.toml")


def test_changing_the_target_is_refused(tmp_path: Path) -> None:
    """A different target would invalidate every comparison in the phase."""
    with pytest.raises(ConfigValidationError, match="same demand target"):
        load_qwen_config(_write(tmp_path, BASE.replace('"customer_load"', '"pv_generation"')))


@pytest.mark.parametrize("horizons", ["[1, 4]", "[1, 4, 96, 288]", "[4, 1, 96]"])
def test_horizons_may_not_be_redefined(tmp_path: Path, horizons: str) -> None:
    """Section 18: the horizons stay h=1/4/96 or the task has changed silently."""
    with pytest.raises(ConfigValidationError, match="horizons"):
        load_qwen_config(_write(tmp_path, BASE.replace("[1, 4, 96]", horizons)))


def test_a_patch_size_that_does_not_divide_the_window_is_refused(tmp_path: Path) -> None:
    """A ragged tail would be dropped without notice."""
    with pytest.raises(ConfigValidationError, match="patch_size"):
        load_qwen_config(_write(tmp_path, BASE.replace("patch_size = 8", "patch_size = 5")))


def test_a_feature_layer_outside_the_cache_is_refused(tmp_path: Path) -> None:
    """Reading an uncached layer would silently score nothing."""
    with pytest.raises(ConfigValidationError, match="feature_layer"):
        load_qwen_config(
            _write(tmp_path, BASE + "feature_layers = [8, 28]\nfeature_layer = 16\n")
        )


def test_changing_the_loss_needs_its_own_decision(tmp_path: Path) -> None:
    with pytest.raises(ConfigValidationError, match="L1 objective"):
        load_qwen_config(_write(tmp_path, BASE + 'loss = "huber"\n'))


def test_an_unknown_aggregation_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ConfigValidationError, match="aggregate"):
        load_qwen_config(_write(tmp_path, BASE + 'aggregate = "max"\n'))


def test_an_ablation_without_a_question_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ConfigValidationError, match="question"):
        load_qwen_config(
            _write(
                tmp_path,
                BASE + '[[ablations]]\nname = "x"\noverrides = { patch_size = 4 }\n',
            )
        )


def test_an_ablation_with_an_unknown_override_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ConfigValidationError, match="unknown override"):
        load_qwen_config(
            _write(
                tmp_path,
                BASE
                + '[[ablations]]\nname = "x"\nquestion = "Does this do the thing we care about?"\n'
                + 'overrides = { not_a_setting = 1 }\n',
            )
        )


def test_a_non_positive_subsample_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ConfigValidationError, match="train_rows"):
        load_qwen_config(_write(tmp_path, BASE + "train_rows = 0\n"))


# ======================================================================
# The subsample
# ======================================================================


def test_the_subsample_is_balanced_across_series() -> None:
    """An unbalanced subsample would silently weight one customer over another."""
    series = build_series(series_count=6, steps=4000)
    index = _index(series, lookback_steps=192, token_stride=4, horizons=(1, 4, 96))
    subsample = select_subsample(index, train_rows=600, validation_rows=300, test_rows=300)
    counts = np.bincount(index.series[subsample.test], minlength=6)
    assert counts.min() == counts.max()
    assert counts.sum() == subsample.test.size


def test_the_subsample_is_evenly_spaced_in_time() -> None:
    """Clustering at the start of the split would not be representative of it."""
    series = build_series(series_count=3, steps=4000)
    index = _index(series, lookback_steps=192, token_stride=4, horizons=(1, 4, 96))
    subsample = select_subsample(index, train_rows=300, validation_rows=150, test_rows=150)
    for series_id in np.unique(index.series)[:5]:
        rows = subsample.test[index.series[subsample.test] == series_id]
        origins = np.sort(index.origins[rows])
        assert origins.size > 2
        gaps = np.diff(origins)
        # Even spacing means the gap spread is small relative to the mean.
        assert gaps.max() <= 3 * max(1, gaps.mean())


def test_the_subsample_is_deterministic_across_calls() -> None:
    """Reruns must select the same rows without a seed being stored elsewhere."""
    series = build_series(series_count=3, steps=4000)
    index = _index(series, lookback_steps=192, token_stride=4, horizons=(1, 4, 96))
    a = select_subsample(index, train_rows=300, validation_rows=150, test_rows=150)
    b = select_subsample(index, train_rows=300, validation_rows=150, test_rows=150)
    assert np.array_equal(a.test, b.test)
    assert a.version() == b.version()


def test_the_subsample_never_crosses_a_split_boundary() -> None:
    series = build_series(series_count=3, steps=4000)
    index = _index(series, lookback_steps=192, token_stride=4, horizons=(1, 4, 96))
    subsample = select_subsample(index, train_rows=300, validation_rows=150, test_rows=150)
    for name, code in (("train", 0), ("validation", 1), ("test", 2)):
        rows = subsample.split(name)
        assert np.all(index.split[rows] == code), f"{name} leaked across splits"


def test_the_subsample_version_changes_with_the_rows() -> None:
    series = build_series(series_count=3, steps=4000)
    index = _index(series, lookback_steps=192, token_stride=4, horizons=(1, 4, 96))
    a = select_subsample(index, train_rows=300, validation_rows=150, test_rows=150)
    b = select_subsample(index, train_rows=300, validation_rows=150, test_rows=159)
    assert a.test.size != b.test.size
    assert a.version() != b.version()


def test_the_selector_quantises_to_whole_rows_per_series() -> None:
    """Requested counts are approximate; the artifact records what was actually used.

    With 3 series, asking for 150 or 151 test rows both yield 50 per series. Silently
    returning "151" would be a small lie in a provenance record, so the summary
    reports the realised size instead.
    """
    series = build_series(series_count=3, steps=4000)
    index = _index(series, lookback_steps=192, token_stride=4, horizons=(1, 4, 96))
    a = select_subsample(index, train_rows=300, validation_rows=150, test_rows=150)
    b = select_subsample(index, train_rows=300, validation_rows=150, test_rows=151)
    assert a.test.size == b.test.size == 150
    summary = a.summary(index)
    assert summary["rows"]["test"] == a.test.size
    assert summary["series_per_split"]["test"] == 3


# ======================================================================
# Paired statistics
# ======================================================================


def test_the_paired_bootstrap_detects_a_real_difference() -> None:
    rng = np.random.default_rng(0)
    a = rng.normal(loc=1.0, scale=0.5, size=2000)
    b = rng.normal(loc=1.2, scale=0.5, size=2000)
    stats = paired_bootstrap(a, b, samples=500, seed=1)
    # A is better, so mean(a) - mean(b) is negative.
    assert stats["mean_difference"] == pytest.approx(-0.2, abs=0.05)
    assert stats["excludes_zero"] is True
    # A real difference puts the whole interval on one side of zero.
    assert stats["ci_high"] < 0


def test_the_paired_bootstrap_reports_no_difference_when_there_is_none() -> None:
    rng = np.random.default_rng(1)
    a = rng.normal(loc=1.0, scale=0.5, size=2000)
    b = a.copy()
    stats = paired_bootstrap(a, b, samples=500, seed=2)
    assert stats["mean_difference"] == pytest.approx(0.0, abs=1e-12)
    assert stats["excludes_zero"] is False


def test_the_paired_bootstrap_is_reproducible() -> None:
    rng = np.random.default_rng(2)
    a = rng.normal(size=800)
    b = rng.normal(size=800)
    first = paired_bootstrap(a, b, samples=300, seed=7)
    second = paired_bootstrap(a, b, samples=300, seed=7)
    assert first == second


def test_the_paired_bootstrap_refuses_misaligned_rows() -> None:
    with pytest.raises(ValueError, match="aligned"):
        paired_bootstrap(np.zeros(10), np.zeros(9))


def test_the_paired_bootstrap_needs_more_than_one_row() -> None:
    with pytest.raises(ValueError, match="at least two rows"):
        paired_bootstrap(np.zeros(1), np.zeros(1))


def test_the_bootstrap_summary_scores_the_reference_against_itself() -> None:
    rng = np.random.default_rng(3)
    errors = {
        "classical_hist_gbm": rng.normal(loc=1.0, size=500),
        "qwen_probe": rng.normal(loc=1.5, size=500),
    }
    summary = bootstrap_summary(errors, reference="classical_hist_gbm", samples=300)
    assert summary["classical_hist_gbm"]["vs_reference"] is None
    assert summary["qwen_probe"]["mae"] > summary["classical_hist_gbm"]["mae"]
    assert summary["qwen_probe"]["relative_pct"] > 0


def test_the_bootstrap_summary_requires_the_reference_to_exist() -> None:
    with pytest.raises(KeyError, match="reference"):
        bootstrap_summary({"a": np.zeros(5)}, reference="b")


# ======================================================================
# Baselines
# ======================================================================


def test_the_persistence_baseline_is_exactly_the_origin_value() -> None:
    actual = np.array([[1.0, 2.0, 3.0]])
    persistence = np.array([[1.0, 1.1, 1.2]])
    result = persistence_baseline(actual, persistence)
    assert result.predicted.tolist() == persistence.tolist()
    assert result.detail["fitted"] is False


def test_the_refitted_gbm_is_trained_in_per_unit_space() -> None:
    """Fitting on kW targets against per-unit features produced a 14.6 kW train MAE.

    The guard is that the estimator sees per-unit features and per-unit targets, and
    that its per-unit error is far below the per-unit persistence error.
    """
    rng = np.random.default_rng(4)
    rows = 3000
    origin = rng.normal(loc=0.3, scale=0.1, size=rows)
    features = np.column_stack([origin, origin * 1.01, rng.normal(size=rows)])
    # Targets are a smooth function of the features plus small noise, so a fitted
    # model must clearly beat "repeat the origin".
    smooth = 0.6 * origin + 0.4 * features[:, 1] + 0.5 * features[:, 2] * 0.01
    targets = (smooth + rng.normal(scale=0.001, size=rows))[:, None]
    horizons = (1,)
    predictions, detail = gbm_baseline(
        train_features=features,
        train_targets=targets,
        predict_features={"test": features},
        horizons=horizons,
        feature_names=("value_at_origin", "lag_1", "noise"),
        seed=1,
    )
    assert detail["fitted"] is True
    assert len(detail["feature_names"]) == 3
    assert predictions["test"].shape == (rows, 1)
    mae = float(np.mean(np.abs(predictions["test"] - targets)))
    persistence_mae = float(np.mean(np.abs(origin[:, None] - targets)))
    assert mae < 0.5 * persistence_mae, (mae, persistence_mae)


# ======================================================================
# Representation diagnostics
# ======================================================================


def test_representation_diagnostics_detect_a_collapsed_representation() -> None:
    """A rank-1 feature set must not score like a full-width one."""
    rng = np.random.default_rng(5)
    width = 64
    direction = rng.normal(size=(1, width))
    # Vary along one direction only: that is what "collapsed onto a few dimensions"
    # means. Rows that are all IDENTICAL would be a different construction, and
    # centring removes their variance entirely.
    collapsed = rng.normal(size=(500, 1)) * direction + rng.normal(scale=1e-6, size=(500, width))
    collapsed_stats = representation_diagnostics(collapsed)
    assert collapsed_stats["participation_ratio"] < 5.0

    rich = rng.normal(size=(500, width))
    rich_stats = representation_diagnostics(rich)
    assert rich_stats["participation_ratio"] > 0.5 * width


def test_representation_diagnostics_are_finite_on_real_shaped_input() -> None:
    rng = np.random.default_rng(6)
    stats = representation_diagnostics(rng.normal(size=(200, 64)))
    for key in (
        "hidden_norm_mean",
        "hidden_norm_std",
        "participation_ratio",
        "mean_pairwise_cosine_abs",
        "featurewise_cv_mean",
    ):
        assert np.isfinite(stats[key]), key
