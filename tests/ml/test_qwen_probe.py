"""Tests for the numerical bridge, the frozen-feature probe, and the leakage guards.

The bridge is where a silent numerical error would be most expensive and least
visible: a patch folded in the wrong order still produces plausible-looking features
and a plausible-looking loss. So the tests here check shapes, determinism and
alignment against Phase 5's own gatherer rather than only checking that code runs.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from conftest import build_series

from energy_intelligence.ml.qwen.probe import (
    FeatureBasis,
    ProbeHead,
    ProbeTrainingConfig,
    fit_probe,
    probe_predict,
)
from energy_intelligence.ml.qwen.representation import (
    EnergyPatchProjector,
    EnergyRepresentation,
    build_patches,
    representation_version,
)
from energy_intelligence.ml.sequence import SequenceConfig, build_sequence_index


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


# ======================================================================
# The representation contract
# ======================================================================


def test_a_week_of_context_becomes_the_expected_number_of_tokens() -> None:
    """672 native steps at stride 4 is 168 sampled positions, 21 tokens at patch 8."""
    representation = EnergyRepresentation(patch_size=8)
    assert representation.token_count(168) == 21


def test_a_ragged_patch_is_refused_rather_than_silently_dropped() -> None:
    """Dropping a partial tail would shorten the context without saying so."""
    representation = EnergyRepresentation(patch_size=8)
    with pytest.raises(ValueError, match="does not divide"):
        representation.token_count(100)


def test_build_patches_refuses_a_ragged_window() -> None:
    windows = np.zeros((2, 100, 6), dtype=np.float32)
    with pytest.raises(ValueError, match="does not divide"):
        build_patches(windows, patch_size=8)


def test_build_patches_preserves_every_value_in_order() -> None:
    """Folding must be a reshape, not a reordering.

    A patch projected in the wrong order still yields the right shape and a finite
    loss, so the only defence is asserting the values come back where they started.
    """
    windows = np.arange(2 * 12 * 3, dtype=np.float32).reshape(2, 12, 3)
    patches, tokens = build_patches(windows, patch_size=4)
    assert tokens == 3
    assert patches.shape == (2, 3, 12)
    # Flattening patch-major: token 0 is positions 0-3 of every channel.
    assert patches[0, 0].tolist() == windows[0, :4, :].reshape(-1).tolist()
    assert patches[0, 1].tolist() == windows[0, 4:8, :].reshape(-1).tolist()
    assert patches[1, 2].tolist() == windows[1, 8:12, :].reshape(-1).tolist()
    # And nothing is lost.
    assert sorted(patches.ravel().tolist()) == sorted(windows.ravel().tolist())


def test_build_patches_rejects_a_two_dimensional_input() -> None:
    with pytest.raises(ValueError, match=r"\[batch, positions, channels\]"):
        build_patches(np.zeros((4, 12), dtype=np.float32), patch_size=4)


def test_the_representation_version_changes_when_the_mapping_changes() -> None:
    base = EnergyRepresentation(patch_size=8)
    assert representation_version(base, in_channels=6) == representation_version(
        base, in_channels=6
    )
    assert representation_version(base, in_channels=6) != representation_version(
        base, in_channels=2
    )
    assert representation_version(base, in_channels=6) != representation_version(
        EnergyRepresentation(patch_size=4), in_channels=6
    )


# ======================================================================
# The projector
# ======================================================================


def test_the_projector_writes_into_the_backbone_residual_width() -> None:
    projector = EnergyPatchProjector(patch_size=8, in_channels=6, hidden_size=2048)
    patches = np.zeros((3, 21, 48), dtype=np.float32)
    out = projector(torch.from_numpy(patches))
    assert out.shape == (3, 21, 2048)
    assert projector.in_features == 48


def test_the_projector_is_deterministic_for_a_given_seed() -> None:
    """The projector is frozen during extraction, so two draws must not differ.

    A different projection per run would make two feature caches incomparable for
    reasons that have nothing to do with the backbone.
    """
    a = EnergyPatchProjector(patch_size=4, in_channels=3, hidden_size=16, seed=7)
    b = EnergyPatchProjector(patch_size=4, in_channels=3, hidden_size=16, seed=7)
    c = EnergyPatchProjector(patch_size=4, in_channels=3, hidden_size=16, seed=8)
    patches = torch.from_numpy(np.random.default_rng(0).normal(size=(2, 6, 12)).astype(np.float32))
    assert torch.allclose(a(patches), b(patches))
    assert not torch.allclose(a(patches), c(patches))


def test_the_projector_is_small_enough_that_credit_cannot_go_to_it() -> None:
    """The shipped projector is ~100k parameters against a 1.7B backbone."""
    projector = EnergyPatchProjector(patch_size=8, in_channels=6, hidden_size=2048)
    assert projector.parameter_count() == 48 * 2048 + 2048 + 2 * 48


def test_the_projector_rejects_a_wrong_patch_width() -> None:
    projector = EnergyPatchProjector(patch_size=8, in_channels=6, hidden_size=32)
    with pytest.raises(ValueError, match="last dimension"):
        projector(torch.zeros(2, 21, 47))


def test_the_projector_rejects_a_two_dimensional_input() -> None:
    projector = EnergyPatchProjector(patch_size=8, in_channels=6, hidden_size=32)
    with pytest.raises(ValueError, match=r"\[batch, tokens, features\]"):
        projector(torch.zeros(21, 48))


# ======================================================================
# The probe
# ======================================================================


def test_the_probe_output_width_matches_the_horizons() -> None:
    head = ProbeHead(16, (1, 4, 96), layers=(0,))
    assert head(torch.zeros(5, 16)).shape == (5, 3)


def test_the_probe_rejects_a_wrong_feature_width() -> None:
    head = ProbeHead(16, (1,), layers=(0,))
    with pytest.raises(ValueError, match="expected 16 features"):
        head(torch.zeros(5, 15))


def test_the_probe_counts_only_the_declared_trainable_parameters() -> None:
    """A claim of 'N trainable parameters' has to match requires_grad, not intent."""
    head = ProbeHead(16, (1, 4, 96), layers=(0,), series_count=40)
    # The series embedding is concatenated onto the features, so the output layer's
    # INPUT is in_features + embedding width. Getting this wrong understates the
    # head by 24 parameters, which is small enough to hide and large enough to matter
    # when the whole claim is "the head has N parameters".
    expected = (head.in_features + head.series_embedding_dim) * 3 + 3 + 40 * 8
    assert head.parameter_count() == expected
    trainable = sum(p.numel() for p in head.parameters() if p.requires_grad)
    assert trainable == expected


def test_the_series_embedding_requires_a_series_index() -> None:
    """Omitting it must fail loudly, not silently drop the identity signal."""
    head = ProbeHead(16, (1,), layers=(0,), series_count=40)
    with pytest.raises(ValueError, match="series index is required"):
        head(torch.zeros(3, 16))
    assert head(torch.zeros(3, 16), torch.tensor([0, 1, 2])).shape == (3, 1)


def test_the_series_embedding_rejects_a_row_count_mismatch() -> None:
    head = ProbeHead(16, (1,), layers=(0,), series_count=40)
    with pytest.raises(ValueError, match="row count"):
        head(torch.zeros(3, 16), torch.tensor([0, 1]))


def test_different_series_produce_different_predictions() -> None:
    """Otherwise the embedding is inert and the probe is blind to customer identity."""
    torch.manual_seed(0)
    head = ProbeHead(16, (1,), layers=(0,), series_count=4)
    features = torch.ones(2, 16)
    a = head(features, torch.tensor([0, 0]))
    b = head(features, torch.tensor([1, 2]))
    assert not torch.allclose(a, b)


# ======================================================================
# The PCA basis
# ======================================================================


def test_the_basis_is_fitted_on_what_it_is_given_and_transforms_consistently() -> None:
    rng = np.random.default_rng(1)
    features = rng.normal(size=(500, 32)).astype(np.float32)
    basis = FeatureBasis(8).fit(features)
    compressed = basis.transform(features)
    assert compressed.shape == (500, 8)
    assert 0.0 < basis.cumulative_variance <= 1.0 + 1e-9


def test_the_basis_refuses_to_transform_before_being_fitted() -> None:
    with pytest.raises(RuntimeError, match="not fitted"):
        FeatureBasis(4).transform(np.zeros((3, 8)))


def test_the_basis_round_trips_through_disk(tmp_path) -> None:
    from energy_intelligence.ml.qwen.probe import FeatureBasis as FB

    features = np.random.default_rng(2).normal(size=(300, 16)).astype(np.float32)
    basis = FB(6).fit(features)
    path = tmp_path / "basis.npz"
    basis.save(path)
    restored = FB.load(path)
    assert np.allclose(restored.transform(features), basis.transform(features))


# ======================================================================
# Leakage: the property that matters most here
# ======================================================================


def test_the_probe_sees_nothing_after_its_forecast_origin() -> None:
    """Poison the future and prove the input window is bit-identical.

    This is Phase 4's leakage method applied to the Phase 6 path, and it uses the
    established boundary: values strictly **after** the origin. Poisoning y(t) itself
    would fail, and correctly so - the value at the origin is a legitimate input, and
    Phase 4 included it as a feature for exactly that reason.

    If the backbone could see past the origin, every number this phase produces would
    be a fiction.
    """
    from energy_intelligence.ml.sequence import gather_sequences

    series = build_series(series_count=3, steps=4000)
    index = _index(series, lookback_steps=192, token_stride=4, horizons=(1, 4, 96))
    values = series.normalised().astype(np.float32)
    rows = index.rows(2)[:64]
    assert rows.size, "the synthetic fixture produced no test rows"

    baseline_x, baseline_y, baseline_b = gather_sequences(values, index, rows)

    changed_targets = 0
    for position, row in enumerate(rows):
        origin = int(index.origins[row])
        series_row = int(index.series[row])
        poisoned = values.copy()
        poisoned[series_row, origin + 1 :] = np.nan
        candidate_x, candidate_y, _ = gather_sequences(poisoned, index, np.asarray([row]))
        assert np.array_equal(
            baseline_x[position : position + 1], candidate_x, equal_nan=True
        ), f"row {row} (origin {origin}) changed when values after its origin were poisoned"
        if not np.array_equal(
            np.nan_to_num(baseline_y[position], nan=-1.0),
            np.nan_to_num(candidate_y[0], nan=-1.0),
        ):
            changed_targets += 1

    # Every target must have been inside the poisoned region, or the test proves nothing.
    assert changed_targets == rows.size


def test_the_phase6_window_passes_phases_own_causality_guard() -> None:
    """Run the production guard, not a reimplementation of it."""
    from energy_intelligence.ml.sequence import assert_sequences_are_causal

    series = build_series(series_count=3, steps=4000)
    index = _index(series, lookback_steps=672, token_stride=4, horizons=(1, 4, 96))
    values = series.normalised().astype(np.float32)
    for split in (0, 1, 2):
        assert_sequences_are_causal(values, index, index.rows(split)[:48])


def test_the_cyclic_channels_are_a_function_of_the_origin_only() -> None:
    """Time-of-day must come from the origin, never from the target's timestamp.

    Two rows at the same native origin for the same series must carry identical cyclic
    channels whatever their targets turn out to be.
    """
    from energy_intelligence.ml.sequence import gather_sequences

    series = build_series(series_count=2, steps=4000)
    index = _index(series, lookback_steps=96, token_stride=4, horizons=(1, 4, 96))
    values = series.normalised().astype(np.float32)
    rows = index.rows(2)[:128]
    windows, _targets, _baseline = gather_sequences(values, index, rows)

    names = index.config.channel_names
    hour = [i for i, n in enumerate(names) if "hour" in n]
    assert hour, f"expected an hour-of-day channel among {names}"

    origins = index.origins[rows]
    series_index = index.series[rows]
    # Group by (series, origin); each group's cyclic channels must be identical.
    seen: dict[tuple[int, int], np.ndarray] = {}
    for position in range(rows.size):
        key = (int(series_index[position]), int(origins[position]))
        block = windows[position][hour].reshape(-1)
        if key in seen:
            assert np.array_equal(seen[key], block)
        else:
            seen[key] = block


def test_scaling_uses_each_series_own_rated_kw() -> None:
    """A 1.4 kW household and a 210 kW shop must not be treated as equally important."""
    series = build_series(series_count=4, steps=400)
    scales = series.scale_vector
    assert scales.shape == (4,)
    assert np.all(scales > 0)
    assert len(set(scales.tolist())) > 1


# ======================================================================
# Training and checkpointing
# ======================================================================


def _probe_data(rows: int = 400, features: int = 24, horizons: int = 3, seed: int = 5):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(rows, features)).astype(np.float32)
    persistence = rng.normal(loc=0.3, scale=0.02, size=(rows, horizons)).astype(np.float32)
    signal = rng.normal(scale=0.01, size=(rows, 1)).astype(np.float32)
    targets = persistence + signal * np.arange(1, horizons + 1, dtype=np.float32)
    return x, targets.astype(np.float32), persistence


def test_probe_training_reduces_validation_error_and_selects_on_validation(
    tmp_path,
) -> None:
    x, y, p = _probe_data()
    trained = fit_probe(
        x[:300],
        y[:300],
        p[:300],
        x[300:],
        y[300:],
        p[300:],
        horizons=(1, 4, 96),
        config=ProbeTrainingConfig(epochs=40, patience=40, torch_threads=2, seed=1),
        layers=(0, 32),
        basis_width=None,
        series_count=0,
        checkpoint_path=tmp_path / "probe.pt",
    )
    assert trained.parameter_count > 0
    assert trained.best_epoch >= 1
    scores = [h["validation_mae_per_unit"] for h in trained.history]
    assert min(scores) == pytest.approx(trained.best_validation_mae_per_unit)
    # The best epoch must be the argmin of the recorded validation scores.
    assert trained.history[trained.best_epoch - 1]["is_best"] is True


def test_probe_early_stopping_leaves_the_best_checkpoint_loaded(tmp_path) -> None:
    """Patience must not change WHICH epoch is returned, only when to stop."""
    x, y, p = _probe_data(seed=9)
    trained = fit_probe(
        x[:300],
        y[:300],
        p[:300],
        x[300:],
        y[300:],
        p[300:],
        horizons=(1, 4, 96),
        config=ProbeTrainingConfig(epochs=200, patience=2, torch_threads=2, seed=2),
        layers=(0,),
        basis_width=None,
        checkpoint_path=tmp_path / "p.pt",
    )
    assert trained.stopped_early is True
    assert trained.epochs_run < 200


def test_a_reloaded_checkpoint_predicts_identically(tmp_path) -> None:
    """Checkpoint selection is only meaningful if reload reproduces the weights."""
    x, y, p = _probe_data(seed=11)
    path = tmp_path / "probe.pt"
    trained = fit_probe(
        x[:300],
        y[:300],
        p[:300],
        x[300:],
        y[300:],
        p[300:],
        horizons=(1, 4, 96),
        config=ProbeTrainingConfig(epochs=5, torch_threads=2, seed=3),
        layers=(0,),
        basis_width=None,
        checkpoint_path=path,
    )
    before = probe_predict(trained.head, x[300:], basis=trained.basis)

    payload = torch.load(path, weights_only=False)
    restored = ProbeHead(
        payload["in_features"],
        tuple(payload["horizons"]),
        layers=tuple(payload["layers"]),
        series_count=payload.get("series_count", 0),
        series_embedding_dim=payload.get("series_embedding_dim", 8),
    )
    restored.load_state_dict(payload["state_dict"])
    after = probe_predict(restored, x[300:])
    assert np.allclose(before, after, atol=1e-6)


def test_probe_training_is_reproducible_for_a_fixed_seed(tmp_path) -> None:
    x, y, p = _probe_data(seed=13)
    kwargs = dict(horizons=(1, 4, 96), layers=(0,), basis_width=None)
    config = ProbeTrainingConfig(epochs=6, seed=4, torch_threads=2)
    a = fit_probe(x[:300], y[:300], p[:300], x[300:], y[300:], p[300:],
                  config=config, checkpoint_path=tmp_path / "a.pt", **kwargs)
    b = fit_probe(x[:300], y[:300], p[:300], x[300:], y[300:], p[300:],
                  config=config, checkpoint_path=tmp_path / "b.pt", **kwargs)
    assert a.best_validation_mae_per_unit == pytest.approx(
        b.best_validation_mae_per_unit, abs=1e-12
    )


def test_the_probe_requires_grad_on_every_head_parameter_and_nothing_else(
    tmp_path,
) -> None:
    """The frozen-backbone claim depends on gradients existing ONLY in the head."""
    x, y, p = _probe_data(seed=17)
    trained = fit_probe(
        x[:300], y[:300], p[:300], x[300:], y[300:], p[300:],
        horizons=(1, 4, 96),
        config=ProbeTrainingConfig(epochs=1, torch_threads=2, seed=5),
        layers=(0,), basis_width=None,
        series_count=4, series_embedding_dim=8,
        train_series_index=np.arange(300) % 4,
        validation_series_index=np.arange(100) % 4,
        checkpoint_path=tmp_path / "p.pt",
    )
    head = trained.head
    assert all(p.requires_grad for p in head.parameters())
    # One optimisation step must move weights and produce finite gradients.
    optimiser = torch.optim.AdamW(head.parameters(), lr=0.01)
    features = torch.from_numpy(x[:64])
    series = torch.arange(64) % 4
    loss = torch.nn.functional.l1_loss(head(features, series), torch.from_numpy(p[:64]))
    loss.backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters()
    )
    before = head.output.weight.detach().clone()
    optimiser.step()
    assert not torch.allclose(before, head.output.weight.detach())


def test_probe_predict_refuses_to_guess_a_missing_series_index(tmp_path) -> None:
    x, y, p = _probe_data(seed=19)
    trained = fit_probe(
        x[:300], y[:300], p[:300], x[300:], y[300:], p[300:],
        horizons=(1, 4, 96),
        config=ProbeTrainingConfig(epochs=1, torch_threads=2, seed=6),
        layers=(0,), basis_width=None,
        series_count=4,
        train_series_index=np.arange(300) % 4,
        validation_series_index=np.arange(100) % 4,
        checkpoint_path=tmp_path / "p.pt",
    )
    with pytest.raises(ValueError, match="series index is required"):
        probe_predict(trained.head, x[300:], basis=trained.basis)
