"""Tests for the temporal architectures and the training protocol.

The properties that matter, and that are easy to get wrong:

* a causal convolution must not see its own future - verified by perturbing the last
  input tokens and asserting the earlier representations are unchanged;
* multi-horizon alignment must be exact, with one head per declared horizon;
* checkpoint selection must come from validation, and the training function must have
  no way to reach the test split;
* scaling must be fitted on training rows only.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from conftest import build_series

from energy_intelligence.ml.scaling import FeatureScaler, assert_train_only_fit
from energy_intelligence.ml.sequence import SequenceConfig, build_sequence_index, gather_sequences
from energy_intelligence.ml.temporal import (
    ARCHITECTURE_NAMES,
    ARCHITECTURES,
    build_model,
    model_defaults,
)
from energy_intelligence.ml.training import (
    TrainingConfig,
    evaluate_checkpoint,
    fit_temporal_model,
    predict_series,
)


@pytest.fixture
def tiny_index():
    """A small, real-shaped sequence index over a synthetic series."""
    series = build_series(series_count=2, steps=20000)
    config = SequenceConfig(
        lookback_steps=96, token_stride=4, horizons=(1, 4), train_origin_stride=64
    )
    index = build_sequence_index(
        series_count=2,
        total_steps=20000,
        config=config,
        target_id="customer_load",
        value_origin="DERIVED",
        series_ids=series.series_ids,
        source="synthetic",
    )
    return series, index


def _scaler_for(series, index) -> FeatureScaler:
    rows = index.rows(0)[:256]
    blocks = []
    for start in range(0, rows.size, 128):
        raw, _, _ = gather_sequences(
            series.normalised().astype(np.float32), index, rows[start : start + 128]
        )
        blocks.append(raw.transpose(0, 2, 1).reshape(raw.shape[0], -1).astype(np.float64))
    return FeatureScaler.fit(np.concatenate(blocks, axis=0))


# ======================================================================
# Architecture
# ======================================================================


def test_only_two_architectures_are_offered() -> None:
    assert set(ARCHITECTURE_NAMES) == {ARCHITECTURES.TCN, ARCHITECTURES.TRANSFORMER}


def test_defaults_declare_a_reason_and_stay_small() -> None:
    for name in ARCHITECTURE_NAMES:
        params = model_defaults(name)
        assert params["note"], name
        assert params["hidden_size"] <= 128, "Phase 5 must stay small"
    with pytest.raises(KeyError, match="unknown architecture"):
        model_defaults("transformer-xl")


def test_model_is_a_real_module_with_reachable_parameters() -> None:
    """The optimiser must be able to see the parameters."""
    model = build_model(ARCHITECTURES.TCN, in_channels=6, horizons=(1, 4, 96), series_count=2)
    assert isinstance(model, torch.nn.Module)
    total = sum(p.numel() for p in model.parameters())
    assert total == model.parameter_count()
    assert model.trainable_parameter_count() == total
    assert 10_000 < total < 2_000_000, "a baseline, not a foundation model"


def test_forward_shape_matches_the_horizon_list() -> None:
    for architecture in ARCHITECTURE_NAMES:
        model = build_model(architecture, in_channels=6, horizons=(1, 4, 96), series_count=3)
        out = model(torch.randn(8, 6, 168), torch.tensor([0, 1, 2, 0, 1, 2, 0, 1]))
        assert out.shape == (8, 3), architecture
        assert model.horizon_count == 3


def test_heads_are_separate_per_horizon_but_share_one_trunk() -> None:
    model = build_model(ARCHITECTURES.TCN, in_channels=6, horizons=(1, 4, 96), series_count=2)
    assert len(model.heads) == 3
    trunk_ids = {id(p) for p in model.trunk.parameters()}
    head_ids = {id(p) for p in model.heads.parameters()}
    assert trunk_ids, "the shared trunk must have parameters"
    assert not (trunk_ids & head_ids), "trunk and heads must not share parameters"


def test_head_order_matches_the_horizon_order() -> None:
    """Head i must be horizon i, which is what makes multi-horizon alignment checkable."""
    model = build_model(ARCHITECTURES.TCN, in_channels=2, horizons=(1, 4, 96), series_count=2)
    model.eval()
    x = torch.randn(4, 2, 168)
    series_ids = torch.tensor([0, 1, 0, 1])
    baseline = model(x, series_ids)
    # Nudging only the head for the last horizon must leave the others alone.
    with torch.no_grad():
        model.heads[-1].bias.add_(1.0)
        nudged = model(x, series_ids)
    assert torch.allclose(baseline[:, :-1], nudged[:, :-1])
    assert not torch.allclose(baseline[:, -1], nudged[:, -1])


def test_unknown_architecture_is_refused() -> None:
    with pytest.raises(KeyError, match="unknown architecture"):
        build_model("deepseek", in_channels=6, horizons=(1,), series_count=2)


def test_series_embedding_is_optional_and_tolerates_missing_ids() -> None:
    without = build_model(ARCHITECTURES.TCN, in_channels=6, horizons=(1,), series_count=None)
    assert without.series_embedding is None
    with_embedding = build_model(ARCHITECTURES.TCN, in_channels=6, horizons=(1,), series_count=5)
    assert with_embedding.series_embedding is not None
    x = torch.randn(4, 6, 168)
    assert with_embedding(x, torch.tensor([0, 1, 2, 3])).shape == (4, 1)
    assert with_embedding(x).shape == (4, 1)


def test_tcn_convolution_padding_never_reaches_forward_in_time() -> None:
    """The convolution itself must be causal.

    GroupNorm mixes across positions *inside* the window, so this is asserted on the
    convolution stack rather than on the normalised output. It is still the property
    that matters: no post-origin value can enter, because the window ends at the
    origin (proved in tests/ml/test_sequence.py).
    """
    from energy_intelligence.ml.temporal import _ResidualBlock

    block = _ResidualBlock(channels=2, kernel_size=3, dilation=2, dropout=0.0)
    block.eval()
    x = torch.zeros(1, 2, 24)
    with torch.no_grad():
        # conv1 is left-padded only; conv2 is 1x1, so the pair is causal.
        a = block.conv1(torch.nn.functional.pad(x, (block.padding, 0)))
        perturbed = x.clone()
        perturbed[0, :, -1] = 99.0
        b = block.conv1(torch.nn.functional.pad(perturbed, (block.padding, 0)))
    assert torch.allclose(a[:, :, :-1], b[:, :, :-1], atol=1e-6)
    assert not torch.allclose(a[:, :, -1], b[:, :, -1])


def test_encoder_output_responds_to_the_window_contents() -> None:
    """The encoder is not constant: a different window must give a different output."""
    model = build_model(ARCHITECTURES.TCN, in_channels=2, horizons=(1,), series_count=2)
    model.eval()
    with torch.no_grad():
        a = model.encoder(torch.zeros(1, 2, 24))
        b = model.encoder(torch.ones(1, 2, 24))
    assert not torch.allclose(a, b)


def test_transformer_patches_the_sequence_and_stays_small() -> None:
    model = build_model(
        ARCHITECTURES.TRANSFORMER,
        in_channels=6,
        horizons=(1, 4, 96),
        series_count=4,
        params={**model_defaults(ARCHITECTURES.TRANSFORMER), "patch_tokens": 8},
    )
    assert model(torch.randn(4, 6, 168), torch.tensor([0, 1, 2, 3])).shape == (4, 3)
    assert 10_000 < model.parameter_count() < 2_000_000


def test_config_dict_records_what_was_built() -> None:
    model = build_model(ARCHITECTURES.TCN, in_channels=6, horizons=(1, 4), series_count=2)
    payload = model.config_dict()
    assert payload["architecture"] == ARCHITECTURES.TCN
    assert payload["horizon_count"] == 2
    assert payload["channels"] == 48
    assert payload["series_embedding_dim"] == 8


def test_repr_mentions_the_architecture() -> None:
    model = build_model(ARCHITECTURES.TCN, in_channels=6, horizons=(1,), series_count=2)
    assert "tcn" in repr(model)


# ======================================================================
# Training protocol
# ======================================================================


def test_training_config_rejects_nonsense() -> None:
    with pytest.raises(ValueError, match="epochs must be positive"):
        TrainingConfig(epochs=0)
    with pytest.raises(ValueError, match="batch_size must be positive"):
        TrainingConfig(batch_size=0)
    with pytest.raises(ValueError, match="learning_rate must be positive"):
        TrainingConfig(learning_rate=0.0)
    with pytest.raises(ValueError, match="loss must be"):
        TrainingConfig(loss="cosine")
    with pytest.raises(ValueError, match="huber_delta must be positive"):
        TrainingConfig(loss="huber", huber_delta=0.0)
    with pytest.raises(ValueError, match="patience must not be negative"):
        TrainingConfig(patience=-1)
    with pytest.raises(ValueError, match="threads must be positive"):
        TrainingConfig(threads=0)
    with pytest.raises(ValueError, match="max_seconds must be positive"):
        TrainingConfig(max_seconds=0.0)


def test_one_training_step_changes_the_parameters(tiny_index, tmp_path: Path) -> None:
    series, index = tiny_index
    scaler = _scaler_for(series, index)
    model = build_model(ARCHITECTURES.TCN, in_channels=index.config.channel_count,
                        horizons=index.config.horizons, series_count=2)
    before = [p.detach().clone() for p in model.parameters()]
    trained = fit_temporal_model(
        model, series.normalised().astype(np.float32), index,
        config=TrainingConfig(epochs=1, batch_size=128, threads=2),
        scaler=scaler, checkpoint_path=tmp_path / "one.pt",
    )
    assert any(
        not torch.allclose(b, a) for b, a in zip(before, model.parameters(), strict=True)
    ), "an epoch passed without changing a single parameter"
    assert trained.checkpoint_path.is_file()
    assert trained.train_seconds > 0
    assert trained.best_epoch == 1


def test_checkpoint_is_written_and_reloads_identically(tiny_index, tmp_path: Path) -> None:
    series, index = tiny_index
    values = series.normalised().astype(np.float32)
    scaler = _scaler_for(series, index)
    trained = fit_temporal_model(
        build_model(ARCHITECTURES.TCN, in_channels=index.config.channel_count,
                    horizons=index.config.horizons, series_count=2),
        values, index,
        config=TrainingConfig(epochs=2, batch_size=128, threads=2),
        scaler=scaler, checkpoint_path=tmp_path / "best.pt",
    )
    payload = torch.load(trained.checkpoint_path, weights_only=False)
    assert payload["parameter_count"] == trained.model.parameter_count()
    assert payload["sequence_config"]["lookback_steps"] == index.config.lookback_steps
    assert "training_config" in payload

    reloaded = build_model(ARCHITECTURES.TCN, in_channels=index.config.channel_count,
                           horizons=index.config.horizons, series_count=2)
    reloaded.load_state_dict(payload["state_dict"])
    rows = index.rows(2)[:32]
    original = predict_series(trained.model, values, index, rows, scaler=scaler)[1]
    restored = predict_series(reloaded, values, index, rows, scaler=scaler)[1]
    assert np.allclose(original, restored)


def test_checkpoint_selection_records_the_validation_best_epoch(tiny_index, tmp_path: Path) -> None:
    series, index = tiny_index
    scaler = _scaler_for(series, index)
    trained = fit_temporal_model(
        build_model(ARCHITECTURES.TCN, in_channels=index.config.channel_count,
                    horizons=index.config.horizons, series_count=2),
        series.normalised().astype(np.float32), index,
        config=TrainingConfig(epochs=3, batch_size=256, threads=2, patience=5),
        scaler=scaler, checkpoint_path=tmp_path / "sel.pt",
    )
    best = min(trained.history, key=lambda h: h.validation_mae_per_unit)
    assert best.epoch == trained.best_epoch
    assert best.is_best
    assert trained.best_validation_mae == pytest.approx(best.validation_mae_per_unit)


def test_validation_stride_only_affects_selection_rows(tiny_index, tmp_path: Path) -> None:
    """A stride changes which rows score the epochs, never the reported test set."""
    series, index = tiny_index
    scaler = _scaler_for(series, index)
    values = series.normalised().astype(np.float32)
    trained = fit_temporal_model(
        build_model(ARCHITECTURES.TCN, in_channels=index.config.channel_count,
                    horizons=index.config.horizons, series_count=2),
        values, index,
        config=TrainingConfig(epochs=1, batch_size=256, threads=2),
        scaler=scaler, checkpoint_path=tmp_path / "vs.pt", validation_stride=4,
    )
    assert trained.validation_rows == len(index.rows(1)[::4])
    _, _, _, rows = evaluate_checkpoint(trained, values, index, split_code=2, scaler=scaler)
    assert rows.size == index.rows(2).size, "the test set must never be strided"


def test_early_stopping_respects_patience(tiny_index, tmp_path: Path) -> None:
    series, index = tiny_index
    scaler = _scaler_for(series, index)
    trained = fit_temporal_model(
        build_model(ARCHITECTURES.TCN, in_channels=index.config.channel_count,
                    horizons=index.config.horizons, series_count=2),
        series.normalised().astype(np.float32), index,
        # A huge learning rate makes validation plateau, so patience fires.
        config=TrainingConfig(epochs=8, batch_size=256, learning_rate=5.0, patience=1, threads=2),
        scaler=scaler, checkpoint_path=tmp_path / "es.pt",
    )
    assert trained.stopped_early
    assert len(trained.history) < 8


def test_time_cap_stops_training(tiny_index, tmp_path: Path) -> None:
    series, index = tiny_index
    scaler = _scaler_for(series, index)
    trained = fit_temporal_model(
        build_model(ARCHITECTURES.TCN, in_channels=index.config.channel_count,
                    horizons=index.config.horizons, series_count=2),
        series.normalised().astype(np.float32), index,
        config=TrainingConfig(epochs=50, batch_size=64, patience=99, max_seconds=0.01, threads=2),
        scaler=scaler, checkpoint_path=tmp_path / "cap.pt",
    )
    assert trained.stopped_early
    assert len(trained.history) <= 2


def test_training_refuses_an_empty_split(tiny_index, tmp_path: Path) -> None:
    import dataclasses

    series, index = tiny_index
    scaler = _scaler_for(series, index)
    model = build_model(ARCHITECTURES.TCN, in_channels=index.config.channel_count,
                        horizons=index.config.horizons, series_count=2)
    broken = dataclasses.replace(index, split=np.zeros_like(index.split))
    with pytest.raises(ValueError, match="validation split is empty"):
        fit_temporal_model(model, series.normalised().astype(np.float32), broken,
                           config=TrainingConfig(epochs=1, threads=2), scaler=scaler,
                           checkpoint_path=tmp_path / "x.pt")


def test_scaling_is_fitted_on_training_rows(tiny_index) -> None:
    series, index = tiny_index
    scaler = _scaler_for(series, index)
    assert_train_only_fit(scaler, scaler.fitted_on)
    with pytest.raises(AssertionError):
        assert_train_only_fit(scaler, scaler.fitted_on + 1)


def test_predictions_are_deterministic_in_eval_mode(tiny_index, tmp_path: Path) -> None:
    series, index = tiny_index
    values = series.normalised().astype(np.float32)
    scaler = _scaler_for(series, index)
    trained = fit_temporal_model(
        build_model(ARCHITECTURES.TCN, in_channels=index.config.channel_count,
                    horizons=index.config.horizons, series_count=2),
        values, index,
        config=TrainingConfig(epochs=1, batch_size=256, threads=2),
        scaler=scaler, checkpoint_path=tmp_path / "det.pt",
    )
    rows = index.rows(2)[:64]
    first = predict_series(trained.model, values, index, rows, scaler=scaler)[1]
    second = predict_series(trained.model, values, index, rows, scaler=scaler)[1]
    assert np.array_equal(first, second)


def test_evaluate_checkpoint_returns_every_test_row(tiny_index, tmp_path: Path) -> None:
    series, index = tiny_index
    values = series.normalised().astype(np.float32)
    scaler = _scaler_for(series, index)
    trained = fit_temporal_model(
        build_model(ARCHITECTURES.TCN, in_channels=index.config.channel_count,
                    horizons=index.config.horizons, series_count=2),
        values, index,
        config=TrainingConfig(epochs=1, batch_size=256, threads=2),
        scaler=scaler, checkpoint_path=tmp_path / "ev.pt",
    )
    actual, predicted, persistence, rows = evaluate_checkpoint(
        trained, values, index, split_code=2, scaler=scaler
    )
    assert actual.shape == predicted.shape == persistence.shape
    assert actual.shape == (rows.size, len(index.config.horizons))
    assert np.isfinite(predicted).all()


def test_delta_formulation_differs_from_persistence(tiny_index, tmp_path: Path) -> None:
    """With delta_target the output must not equal persistence.

    Otherwise the model is predicting nothing and the reported numbers would be a
    copy of the baseline's.
    """
    series, index = tiny_index
    values = series.normalised().astype(np.float32)
    scaler = _scaler_for(series, index)
    trained = fit_temporal_model(
        build_model(ARCHITECTURES.TCN, in_channels=index.config.channel_count,
                    horizons=index.config.horizons, series_count=2),
        values, index,
        config=TrainingConfig(epochs=1, batch_size=256, threads=2),
        scaler=scaler, checkpoint_path=tmp_path / "d.pt",
    )
    rows = index.rows(2)[:64]
    _, predicted, persistence = predict_series(trained.model, values, index, rows, scaler=scaler)
    assert not np.allclose(predicted, persistence)


def test_training_is_reproducible_for_a_fixed_seed(tiny_index, tmp_path: Path) -> None:
    series, index = tiny_index
    values = series.normalised().astype(np.float32)
    scaler = _scaler_for(series, index)
    predictions = []
    for run in range(2):
        torch.manual_seed(0)
        trained = fit_temporal_model(
            build_model(ARCHITECTURES.TCN, in_channels=index.config.channel_count,
                        horizons=index.config.horizons, series_count=2),
            values, index,
            config=TrainingConfig(epochs=1, batch_size=256, threads=2, seed=11),
            scaler=scaler, checkpoint_path=tmp_path / f"r{run}.pt",
        )
        predictions.append(
            predict_series(trained.model, values, index, index.rows(2)[:32], scaler=scaler)[1]
        )
    assert np.allclose(predictions[0], predictions[1], atol=1e-6), (
        "the same seed produced different weights"
    )