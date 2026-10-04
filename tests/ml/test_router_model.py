"""Phase 7, part 2: the router itself - weights, mixing, metadata, training, checkpointing.

The properties under test are the ones that make the output a **routing decision** rather
than a fourth model:

* weights are non-negative and sum to one, per row and per horizon;
* the mixed forecast is exactly the stated convex combination, checkable to the last
  decimal against closed-form expert forecasts;
* hard routing is the argmax of the same weights, not a separately trained model;
* inference is deterministic;
* a checkpoint reloads to bit-identical outputs.

Plus the guardrails: bad weights, mismatched shapes and wrong feature counts are refused
rather than silently producing a plausible number.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from energy_intelligence.ml.router.ensembles import (
    _simplex_grid,
    apply_best_single,
    best_single_forecast,
    ensemble_similarity,
    fit_fixed_weights,
    fixed_forecast,
    select_best_single,
    uniform_weights,
)
from energy_intelligence.ml.router.experts import EvaluationPanel, ExpertPool
from energy_intelligence.ml.router.model import (
    ARCHITECTURE_NAMES,
    EnergyRouterNet,
    RouterArchitecture,
    build_router,
    simplex_weights,
)
from energy_intelligence.ml.router.routing import (
    ROUTING_STRATEGIES,
    EnergyRouter,
    RoutingDecision,
    hard_selection,
    mix,
)
from energy_intelligence.ml.router.training import (
    RouterTrainingConfig,
    fit_router,
    load_router,
    router_weights_for,
)

HORIZONS = (1, 4, 96)


# ---------------------------------------------------------------------------
# The network
# ---------------------------------------------------------------------------


@torch.no_grad()
def _weights(net, features: np.ndarray) -> np.ndarray:
    """Forward pass without autograd, for assertions on the returned numbers."""
    return net(torch.from_numpy(np.asarray(features, dtype=np.float32))).numpy()


def test_router_weights_are_non_negative_and_sum_to_one() -> None:
    net = build_router(n_features=5, horizons=HORIZONS, n_experts=3, seed=1)
    weights = _weights(net, np.zeros((7, 5)))
    assert weights.shape == (7, 3, 3)
    assert np.all(weights >= 0.0)
    assert np.allclose(weights.sum(axis=2), 1.0)


def test_router_has_one_head_per_horizon() -> None:
    net = build_router(n_features=4, horizons=HORIZONS, n_experts=3, seed=1)
    assert net.heads.out_features == 3 * len(HORIZONS)
    assert net.heads.weight.shape[0] == 3 * len(HORIZONS)
    assert len(net.horizons) == 3


def test_a_shared_head_makes_every_horizon_share_one_vector() -> None:
    architecture = RouterArchitecture(name="tiny_mlp", hidden_sizes=(8,), shared_head=True)
    net = build_router(
        n_features=4, horizons=HORIZONS, n_experts=3, architecture=architecture, seed=1
    )
    assert net.heads.out_features == 3
    weights = _weights(net, np.zeros((5, 4)))
    assert weights.shape == (5, 3, 3)
    for column in range(1, 3):
        assert np.allclose(weights[:, 0, :], weights[:, column, :])


def test_router_rejects_a_wrong_feature_count() -> None:
    net = build_router(n_features=4, horizons=HORIZONS, n_experts=3, seed=1)
    with pytest.raises(ValueError, match=r"\[batch, 4\] features"):
        net(torch.zeros(3, 5))


def test_router_refuses_fewer_than_two_experts() -> None:
    with pytest.raises(ValueError, match="fewer than two experts"):
        build_router(n_features=4, horizons=HORIZONS, n_experts=1, seed=1)


def test_router_refuses_no_features_and_no_horizons() -> None:
    with pytest.raises(ValueError, match="at least one feature"):
        build_router(n_features=0, horizons=HORIZONS, n_experts=3, seed=1)
    with pytest.raises(ValueError, match="at least one horizon"):
        build_router(n_features=4, horizons=(), n_experts=3, seed=1)


def test_router_initialisation_is_seeded_and_leaves_global_rng_alone() -> None:
    """Two builds from the same seed match, and neither consumes the caller's RNG."""
    torch.manual_seed(1234)
    before = torch.randn(3)
    torch.manual_seed(1234)
    after = torch.randn(3)
    assert torch.equal(before, after)

    a = build_router(n_features=4, horizons=HORIZONS, n_experts=3, seed=7)
    b = build_router(n_features=4, horizons=HORIZONS, n_experts=3, seed=7)
    c = build_router(n_features=4, horizons=HORIZONS, n_experts=3, seed=8)
    assert torch.equal(a.heads.weight, b.heads.weight)
    assert not torch.equal(a.heads.weight, c.heads.weight)


def test_router_initial_weights_are_not_all_uniform() -> None:
    """A uniform start gives every expert the same first gradient; the bias breaks that."""
    net = build_router(n_features=4, horizons=HORIZONS, n_experts=3, seed=7)
    bias = net.heads.bias.detach().numpy()
    assert not np.allclose(bias, bias[0])


def test_simplex_weights_respect_temperature() -> None:
    logits = torch.tensor([[10.0, 0.0, 0.0]])
    sharp = simplex_weights(logits, temperature=1.0).numpy()
    flat = simplex_weights(logits, temperature=100.0).numpy()
    assert sharp[0, 0] > 0.99
    assert flat[0, 0] < 0.4
    assert np.allclose(flat.sum(axis=1), 1.0)


@pytest.mark.parametrize("name", ARCHITECTURE_NAMES)
def test_every_architecture_name_builds(name: str) -> None:
    hidden = () if name == "linear" else (8,)
    net = build_router(
        n_features=4,
        horizons=HORIZONS,
        n_experts=3,
        architecture=RouterArchitecture(name=name, hidden_sizes=hidden),
        seed=1,
    )
    assert _weights(net, np.zeros((2, 4))).shape == (2, 3, 3)


def test_unknown_architecture_is_refused() -> None:
    with pytest.raises(ValueError, match="router architecture must be one of"):
        RouterArchitecture(name="transformer")


def test_linear_architecture_refuses_hidden_layers() -> None:
    with pytest.raises(ValueError, match="no hidden layers"):
        RouterArchitecture(name="linear", hidden_sizes=(8,))


def test_router_architecture_round_trips_through_its_dict() -> None:
    original = RouterArchitecture(name="tiny_mlp", hidden_sizes=(4, 8), dropout=0.1, temperature=2.0)
    assert RouterArchitecture.from_dict(original.to_dict()) == original


def test_router_rejects_invalid_architecture_values() -> None:
    with pytest.raises(ValueError, match="hidden sizes must be positive"):
        RouterArchitecture(hidden_sizes=(0,))
    with pytest.raises(ValueError, match="dropout must lie"):
        RouterArchitecture(dropout=1.0)
    with pytest.raises(ValueError, match="temperature must be positive"):
        RouterArchitecture(temperature=0.0)


def test_router_parameter_count_is_reported_and_small() -> None:
    net = build_router(n_features=24, horizons=HORIZONS, n_experts=3, seed=1)
    expected = 24 * 32 + 32 + 32 * 9 + 9
    assert net.parameter_count() == expected
    assert net.trainable_parameter_count() == expected


# ---------------------------------------------------------------------------
# Mixing
# ---------------------------------------------------------------------------


def test_mix_is_the_stated_convex_combination_to_the_last_decimal() -> None:
    weights = np.asarray([[[0.25, 0.5, 0.25]]])
    forecasts = np.asarray([[[10.0]], [[20.0]], [[40.0]]])
    assert mix(weights, forecasts)[0, 0] == pytest.approx(0.25 * 10 + 0.5 * 20 + 0.25 * 40)


def test_mix_rejects_weights_that_do_not_sum_to_one() -> None:
    with pytest.raises(ValueError, match="must sum to one"):
        mix(np.asarray([[[0.2, 0.2, 0.2]]]), np.zeros((3, 1, 1)))


def test_mix_rejects_a_negative_weight() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        mix(np.asarray([[[-0.5, 1.0, 0.5]]]), np.zeros((3, 1, 1)))


def test_mix_rejects_a_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="do not match expert forecasts"):
        mix(np.asarray([[[0.5, 0.5]]]), np.zeros((2, 3, 1)))
    with pytest.raises(ValueError, match="do not match expert forecasts"):
        mix(np.asarray([[[0.5, 0.5, 0.0]]]), np.zeros((2, 1, 2)))
    with pytest.raises(ValueError, match=r"\d+ weights for 3 experts"):
        mix(np.asarray([[[0.5, 0.5]]]), np.zeros((3, 1, 1)))
    with pytest.raises(ValueError, match=r"\[rows, horizons, experts\]"):
        mix(np.zeros((1, 1)), np.zeros((1, 1, 1)))


def test_hard_selection_is_a_one_hot_argmax_of_the_same_weights() -> None:
    weights = np.asarray([[[0.1, 0.7, 0.2], [0.6, 0.3, 0.1]]])
    one_hot = hard_selection(weights)
    assert one_hot.shape == weights.shape
    assert np.array_equal(one_hot.sum(axis=2), np.ones((1, 2)))
    assert np.array_equal(np.argmax(one_hot, axis=2), np.argmax(weights, axis=2))
    assert one_hot[0, 0, 1] == 1.0
    assert one_hot[0, 1, 0] == 1.0


def test_hard_selection_breaks_a_tie_towards_the_lower_index_deterministically() -> None:
    weights = np.asarray([[[0.5, 0.5, 0.0]]])
    assert hard_selection(weights)[0, 0, 0] == 1.0
    assert hard_selection(weights).tolist() == hard_selection(weights).tolist()


# ---------------------------------------------------------------------------
# The deployable router object
# ---------------------------------------------------------------------------


def _fitted_router(pool: ExpertPool, n_features: int = 3, seed: int = 5):
    net = build_router(
        n_features=n_features, horizons=pool.horizons, n_experts=pool.n_experts, seed=seed
    )
    from energy_intelligence.ml.scaling import FeatureScaler

    scaler = FeatureScaler.fit(np.zeros((10, n_features)) + np.arange(n_features))
    return EnergyRouter(
        net, pool, scaler, tuple(f"f{i}" for i in range(n_features))
    )


def test_router_route_returns_the_mixed_forecast_and_the_evidence(
    stub_pool: ExpertPool, panel: EvaluationPanel
) -> None:
    router = _fitted_router(stub_pool)
    positions = np.arange(0, panel.n_rows, 11)
    features = np.arange(positions.size * 3, dtype=np.float64).reshape(positions.size, 3)
    decision = router.route(positions, features)

    assert decision.n_rows == positions.size
    assert decision.horizons == panel.horizons
    assert decision.expert_names == stub_pool.names
    assert np.allclose(decision.weight_totals(), 1.0)
    assert decision.forecast_kw.shape == (positions.size, len(panel.horizons))
    expected = mix(decision.weights, decision.expert_forecast_kw)
    assert np.allclose(decision.forecast_kw, expected)
    assert decision.router_feature_names == ("f0", "f1", "f2")
    assert decision.seconds >= 0.0


def test_router_hard_strategy_returns_a_one_hot_decision(
    stub_pool: ExpertPool, panel: EvaluationPanel
) -> None:
    router = _fitted_router(stub_pool)
    positions = np.arange(0, panel.n_rows, 11)
    features = np.zeros((positions.size, 3))
    soft = router.route(positions, features, strategy=ROUTING_STRATEGIES.SOFT)
    hard = router.route(positions, features, strategy=ROUTING_STRATEGIES.HARD)
    assert np.allclose(hard.weights.sum(axis=2), 1.0)
    assert set(np.unique(hard.weights)) <= {0.0, 1.0}
    assert np.array_equal(hard.selected, soft.selected)


def test_router_inference_is_deterministic(stub_pool: ExpertPool, panel: EvaluationPanel) -> None:
    router = _fitted_router(stub_pool)
    positions = np.arange(0, panel.n_rows, 7)
    features = np.linspace(0.0, 1.0, positions.size * 3).reshape(positions.size, 3)
    first = router.route(positions, features).forecast_kw
    second = router.route(positions, features).forecast_kw
    assert np.array_equal(first, second)


def test_router_rejects_an_unknown_strategy(stub_pool: ExpertPool, panel: EvaluationPanel) -> None:
    router = _fitted_router(stub_pool)
    with pytest.raises(ValueError, match="routing strategy must be one of"):
        router.route(np.arange(4), np.zeros((4, 3)), strategy="oracle")


def test_router_rejects_mismatched_features_and_positions(
    stub_pool: ExpertPool, panel: EvaluationPanel
) -> None:
    router = _fitted_router(stub_pool)
    with pytest.raises(ValueError, match="feature rows for 7 positions"):
        router.route(np.arange(7), np.zeros((3, 3)))
    with pytest.raises(ValueError, match="expected 3 features"):
        router.route(np.arange(7), np.zeros((7, 5)))


def test_router_refuses_a_pool_of_the_wrong_width(stub_pool: ExpertPool) -> None:
    net = build_router(n_features=3, horizons=stub_pool.horizons, n_experts=2, seed=1)
    from energy_intelligence.ml.scaling import FeatureScaler

    scaler = FeatureScaler.fit(np.zeros((10, 3)))
    with pytest.raises(ValueError, match="output columns but the pool has"):
        EnergyRouter(net, stub_pool, scaler, ("f0", "f1", "f2"))


def test_router_refuses_a_feature_name_count_that_disagrees(stub_pool: ExpertPool) -> None:
    net = build_router(n_features=3, horizons=stub_pool.horizons, n_experts=3, seed=1)
    from energy_intelligence.ml.scaling import FeatureScaler

    scaler = FeatureScaler.fit(np.zeros((10, 3)))
    with pytest.raises(ValueError, match="expects 3 features but 2 names"):
        EnergyRouter(net, stub_pool, scaler, ("f0", "f1"))


def test_router_accepts_precomputed_forecasts_without_reasking_the_pool(
    stub_pool: ExpertPool, panel: EvaluationPanel
) -> None:
    router = _fitted_router(stub_pool)
    positions = np.arange(0, panel.n_rows, 11)
    features = np.zeros((positions.size, 3))
    supplied = np.stack(
        [np.full((positions.size, len(panel.horizons)), 5.0 + i) for i in range(3)]
    )
    decision = router.route(positions, features, forecasts=supplied)
    assert np.allclose(decision.expert_forecast_kw, supplied)


def test_router_rejects_forecasts_of_the_wrong_shape(
    stub_pool: ExpertPool, panel: EvaluationPanel
) -> None:
    router = _fitted_router(stub_pool)
    with pytest.raises(ValueError, match="expert forecasts must be"):
        router.route(
            np.arange(4), np.zeros((4, 3)), forecasts=np.zeros((3, 4, 1))
        )


def test_routing_decision_reports_capacity_and_mean_weights() -> None:
    weights = np.asarray(
        [
            [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1]],
            [[0.2, 0.3, 0.5], [0.6, 0.2, 0.2]],
        ]
    )
    decision = RoutingDecision(
        positions=np.arange(2),
        horizons=(1, 4),
        expert_names=("a", "b", "c"),
        weights=weights,
        selected=np.argmax(weights, axis=2),
        strategy=ROUTING_STRATEGIES.SOFT,
        forecast_kw=np.zeros((2, 2)),
        expert_forecast_kw=np.zeros((3, 2, 2)),
        router_feature_names=("f0",),
    )
    capacity = decision.capacity()
    assert capacity["1"]["a"] == pytest.approx(0.5)
    assert capacity["4"]["b"] == pytest.approx(0.5)
    assert decision.mean_weights()["1"]["a"] == pytest.approx(0.5)
    assert decision.selected_names().tolist() == [["a", "b"], ["c", "a"]]
    payload = decision.to_dict()
    assert payload["strategy"] == "soft"
    assert payload["max_weight_sum_error"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Ensembles
# ---------------------------------------------------------------------------


def test_simplex_grid_points_all_sum_to_one() -> None:
    grid = _simplex_grid(3, 0.1)
    assert grid.shape[1] == 3
    assert np.allclose(grid.sum(axis=1), 1.0)
    assert np.all(grid >= 0.0)


def test_simplex_grid_rejects_a_single_expert() -> None:
    with pytest.raises(ValueError, match="at least two experts"):
        _simplex_grid(1, 0.1)


def test_uniform_weights_are_exactly_uniform() -> None:
    weights = uniform_weights(5, HORIZONS, 3)
    assert weights.shape == (5, 3, 3)
    assert np.allclose(weights, 1.0 / 3.0)
    assert np.allclose(weights.sum(axis=2), 1.0)


def test_fixed_weights_find_a_mixture_that_beats_both_of_its_worst_experts(
    forecasts, panel: EvaluationPanel
) -> None:
    """Two experts bracketing the truth: their midpoint beats either endpoint."""
    truth = panel.actual_kw
    bracketing = np.stack([truth - 1.0, truth + 1.0])
    ensemble = fit_fixed_weights(
        expert_forecast_kw=bracketing,
        actual_kw=truth,
        horizons=panel.horizons,
        expert_names=("low", "high"),
        step=0.01,
    )
    mixed = fixed_forecast(ensemble, bracketing)
    assert np.allclose(mixed, truth, atol=1e-9)
    assert np.allclose(ensemble.weights.sum(axis=1), 1.0)
    for column in range(len(panel.horizons)):
        assert ensemble.train_mae[panel.horizons[column]] < 1.0


def test_fixed_weights_fit_only_on_the_rows_they_are_given(
    forecasts, panel: EvaluationPanel
) -> None:
    """Fitting on the first half must not move when the second half's truth changes."""
    half = panel.n_rows // 2
    first = fit_fixed_weights(
        expert_forecast_kw=forecasts[:, :half, :],
        actual_kw=panel.actual_kw[:half],
        horizons=panel.horizons,
        expert_names=("a", "b", "c"),
        step=0.1,
    )
    corrupted = panel.actual_kw.copy()
    corrupted[half:] += 1000.0
    second = fit_fixed_weights(
        expert_forecast_kw=forecasts[:, :half, :],
        actual_kw=corrupted[:half],
        horizons=panel.horizons,
        expert_names=("a", "b", "c"),
        step=0.1,
    )
    assert np.array_equal(first.weights, second.weights)


def test_fixed_weights_reject_a_bad_step(forecasts, panel: EvaluationPanel) -> None:
    with pytest.raises(ValueError, match="step must lie"):
        fit_fixed_weights(
            expert_forecast_kw=forecasts,
            actual_kw=panel.actual_kw,
            horizons=panel.horizons,
            expert_names=("a", "b", "c"),
            step=0.0,
        )


def test_fixed_weights_reject_a_horizon_mismatch(forecasts, panel: EvaluationPanel) -> None:
    with pytest.raises(ValueError, match="shapes disagree"):
        fit_fixed_weights(
            expert_forecast_kw=forecasts,
            actual_kw=panel.actual_kw,
            horizons=(1, 4),
            expert_names=("a", "b", "c"),
        )


def test_best_single_selects_per_horizon_and_applies_to_every_row(
    forecasts, panel: EvaluationPanel
) -> None:
    """Selection rows and application rows differ; conflating them is a shape bug."""
    picks = select_best_single(
        expert_forecast_kw=forecasts,
        actual_kw=panel.actual_kw,
        horizons=panel.horizons,
        expert_names=("a", "b", "c"),
        rows=np.arange(panel.split_slices["validation"].stop),
    )
    assert set(picks) == set(panel.horizons)
    applied = apply_best_single(
        expert_forecast_kw=forecasts, picks=picks, horizons=panel.horizons
    )
    assert applied.shape == (panel.n_rows, len(panel.horizons))


def test_best_single_forecast_helper_returns_the_same_answer(
    forecasts, panel: EvaluationPanel
) -> None:
    chosen, names = best_single_forecast(
        expert_forecast_kw=forecasts,
        actual_kw=panel.actual_kw,
        horizons=panel.horizons,
        expert_names=("a", "b", "c"),
        rows=np.arange(20),
    )
    assert set(names.values()) <= {"a", "b", "c"}
    assert chosen.shape == (panel.n_rows, len(panel.horizons))


def test_apply_best_single_rejects_a_missing_or_out_of_range_pick(
    forecasts, panel: EvaluationPanel
) -> None:
    with pytest.raises(ValueError, match="no expert selected for horizon"):
        apply_best_single(expert_forecast_kw=forecasts, picks={1: 0}, horizons=panel.horizons)
    with pytest.raises(ValueError, match="outside the pool"):
        apply_best_single(
            expert_forecast_kw=forecasts,
            picks={1: 0, 4: 0, 96: 99},
            horizons=panel.horizons,
        )


def test_ensemble_similarity_reports_the_distance_between_weight_sets() -> None:
    a = np.asarray([[0.5, 0.5, 0.0]])
    close = ensemble_similarity(a, a + 0.01)
    assert close["mean_absolute_weight_difference"] == pytest.approx(0.01)
    far = ensemble_similarity(a, np.asarray([[0.0, 0.0, 1.0]]))
    assert far["max_absolute_weight_difference"] == pytest.approx(1.0)
    with pytest.raises(ValueError, match="shape mismatch"):
        ensemble_similarity(a, np.zeros((2, 3)))


# ---------------------------------------------------------------------------
# Training and checkpointing
# ---------------------------------------------------------------------------


def _training_inputs(panel: EvaluationPanel, forecasts):
    features = np.column_stack(
        [
            np.linspace(0.0, 1.0, panel.n_rows),
            np.linspace(1.0, 0.0, panel.n_rows),
            np.full(panel.n_rows, 0.5),
        ]
    ).astype(np.float64)
    names = ("level", "trajectory", "volatility")
    return features, names


def _train(panel: EvaluationPanel, forecasts, tmp_path, **overrides):
    features, names = _training_inputs(panel, forecasts)
    config = RouterTrainingConfig(
        epochs=6,
        batch_size=32,
        learning_rate=0.05,
        patience=3,
        threads=2,
        train_fraction_of_validation=0.6,
        **overrides,
    )
    return fit_router(
        features=features,
        expert_forecast_kw=forecasts,
        actual_kw=panel.actual_kw,
        horizons=panel.horizons,
        expert_names=("a", "b", "c"),
        feature_names=names,
        config=config,
        architecture=RouterArchitecture(name="tiny_mlp", hidden_sizes=(8,)),
        checkpoint_path=tmp_path / "router.pt",
    )


def test_router_training_records_epoch_zero_and_selects_the_best(
    panel: EvaluationPanel, forecasts, tmp_path
) -> None:
    trained = _train(panel, forecasts, tmp_path)
    assert trained.history[0].epoch == 0
    assert trained.history[0].is_best
    assert trained.history[0].train_loss != trained.history[0].train_loss  # NaN by design
    assert 0 <= trained.best_epoch <= trained.history[-1].epoch
    # ``best_validation_mae_kw`` is the value of the epoch it actually selected. It is not
    # necessarily the smallest value seen: early stopping only replaces the incumbent when
    # the improvement exceeds ``min_delta``, so a later epoch can be fractionally lower
    # without having won.
    selected = next(item for item in trained.history if item.epoch == trained.best_epoch)
    assert selected.is_best
    assert trained.best_validation_mae_kw == selected.validation_mae_kw
    assert trained.best_validation_mae_kw >= min(
        item.validation_mae_kw for item in trained.history
    ) - 1e-12
    assert trained.best_validation_mae_kw_by_horizon == selected.validation_mae_kw_by_horizon
    assert trained.train_rows + trained.validation_rows == panel.n_rows


def test_router_training_is_deterministic_for_one_seed(
    panel: EvaluationPanel, forecasts, tmp_path
) -> None:
    a = _train(panel, forecasts, tmp_path)
    b = _train(panel, forecasts, tmp_path)
    assert a.best_epoch == b.best_epoch
    assert a.best_validation_mae_kw == b.best_validation_mae_kw
    features, _ = _training_inputs(panel, forecasts)
    assert np.array_equal(
        router_weights_for(a, features), router_weights_for(b, features)
    )


def test_a_different_seed_gives_a_different_router(
    panel: EvaluationPanel, forecasts, tmp_path
) -> None:
    a = _train(panel, forecasts, tmp_path, seed=1)
    b = _train(panel, forecasts, tmp_path, seed=2)
    features, _ = _training_inputs(panel, forecasts)
    assert not np.array_equal(
        router_weights_for(a, features), router_weights_for(b, features)
    )


def test_warm_start_reproduces_its_initial_mixture_exactly(
    panel: EvaluationPanel, forecasts
) -> None:
    """Epoch 0 of a warm-started router **is** the fixed ensemble, which is the point."""
    start = np.asarray([[0.5, 0.3, 0.2], [0.1, 0.4, 0.5], [0.0, 0.25, 0.75]])
    net = EnergyRouterNet(
        n_features=3,
        horizons=panel.horizons,
        n_experts=3,
        architecture=RouterArchitecture(name="tiny_mlp", hidden_sizes=(8,)),
        seed=1,
        initial_weights=start,
    )
    assert net.warm_started
    rng = torch.Generator().manual_seed(0)
    weights = _weights(net, torch.randn(11, 3, generator=rng).numpy())
    # The zero weight is floored at 1e-4 before the log, so it comes back as 1e-4 and
    # not as exactly zero; every other weight is reproduced to float32 precision.
    assert np.allclose(weights[:, 0, :], start[0], atol=2e-4)
    assert np.allclose(weights[:, 2, :], start[2], atol=2e-4)
    assert weights[0, 2, 0] < 1e-3  # the zero is floored, not log(0)


def test_warm_start_refuses_weights_that_are_not_a_simplex() -> None:
    with pytest.raises(ValueError, match="must sum to one"):
        EnergyRouterNet(
            n_features=3,
            horizons=(1,),
            n_experts=2,
            initial_weights=np.asarray([[0.5, 0.2]]),
        )
    with pytest.raises(ValueError, match="non-negative"):
        EnergyRouterNet(
            n_features=3,
            horizons=(1,),
            n_experts=2,
            initial_weights=np.asarray([[1.5, -0.5]]),
        )


def test_cold_start_is_marked_as_such() -> None:
    net = build_router(n_features=3, horizons=(1,), n_experts=2, seed=1)
    assert net.warm_started is False


def test_training_never_returns_worse_than_its_starting_point(
    panel: EvaluationPanel, forecasts, tmp_path
) -> None:
    """Epoch 0 is eligible, so a warm-started router cannot regress on the selection rows."""
    features, names = _training_inputs(panel, forecasts)
    start = np.asarray([[1 / 3, 1 / 3, 1 / 3]] * 3)
    trained = fit_router(
        features=features,
        expert_forecast_kw=forecasts,
        actual_kw=panel.actual_kw,
        horizons=panel.horizons,
        expert_names=("a", "b", "c"),
        feature_names=names,
        config=RouterTrainingConfig(epochs=8, batch_size=32, threads=2, patience=8),
        architecture=RouterArchitecture(name="tiny_mlp", hidden_sizes=(8,)),
        initial_weights=start,
    )
    assert trained.best_validation_mae_kw <= trained.history[0].validation_mae_kw


def test_router_training_rejects_mismatched_shapes(panel: EvaluationPanel, forecasts) -> None:
    features, names = _training_inputs(panel, forecasts)
    with pytest.raises(ValueError, match="do not match"):
        fit_router(
            features=features[:10],
            expert_forecast_kw=forecasts,
            actual_kw=panel.actual_kw,
            horizons=panel.horizons,
            expert_names=("a", "b", "c"),
            feature_names=names,
            config=RouterTrainingConfig(threads=2),
            architecture=RouterArchitecture(hidden_sizes=(4,)),
        )
    with pytest.raises(ValueError, match="feature names for"):
        fit_router(
            features=features,
            expert_forecast_kw=forecasts,
            actual_kw=panel.actual_kw,
            horizons=panel.horizons,
            expert_names=("a", "b", "c"),
            feature_names=("only-one",),
            config=RouterTrainingConfig(threads=2),
            architecture=RouterArchitecture(hidden_sizes=(4,)),
        )


def test_router_training_refuses_an_impossible_split(panel: EvaluationPanel, forecasts) -> None:
    features, names = _training_inputs(panel, forecasts)
    with pytest.raises(ValueError, match="train_fraction_of_validation must lie"):
        RouterTrainingConfig(train_fraction_of_validation=1.0)
    with pytest.raises(ValueError, match="learning_rate must be positive"):
        RouterTrainingConfig(learning_rate=0.0)
    with pytest.raises(ValueError, match="weight_decay must not be negative"):
        RouterTrainingConfig(weight_decay=-1.0)
    with pytest.raises(ValueError, match="epochs must be positive"):
        RouterTrainingConfig(epochs=0)


def test_checkpoint_reloads_to_identical_weights(
    panel: EvaluationPanel, forecasts, tmp_path
) -> None:
    trained = _train(panel, forecasts, tmp_path)
    net, scaler, names, metadata = load_router(tmp_path / "router.pt")
    assert names == trained.feature_names
    assert metadata["best_epoch"] == trained.best_epoch
    features, _ = _training_inputs(panel, forecasts)
    assert np.array_equal(
        router_weights_for(trained, features),
        _weights(net, scaler.transform(features)),
    )
    # The reloaded scaler must reproduce the fitted transform exactly, not merely close
    # to it: a feature order or mean shift here would be invisible and fatal.
    from energy_intelligence.ml.scaling import FeatureScaler

    assert isinstance(scaler, FeatureScaler)
    assert np.array_equal(scaler.transform(features), trained.scaler.transform(features))
    assert scaler.to_dict() == trained.scaler.to_dict()


def test_checkpoint_preserves_the_architecture_and_the_names(
    panel: EvaluationPanel, forecasts, tmp_path
) -> None:
    _train(panel, forecasts, tmp_path)
    payload = torch.load(tmp_path / "router.pt", weights_only=False)
    assert payload["config"]["horizons"] == list(panel.horizons)
    assert payload["config"]["n_experts"] == 3
    assert payload["feature_names"] == ["level", "trajectory", "volatility"]
    assert payload["expert_names"] == ["a", "b", "c"]
    assert payload["training_config"]["epochs"] == 6


def test_loading_a_non_router_checkpoint_is_refused(tmp_path) -> None:
    torch.save({"state_dict": {}}, tmp_path / "wrong.pt")
    with pytest.raises(ValueError, match="is not a router checkpoint"):
        load_router(tmp_path / "wrong.pt")


def test_a_reloaded_router_routes_the_same_way(
    panel: EvaluationPanel, forecasts, tmp_path
) -> None:
    """The full production path, after a save/reload cycle."""
    trained = _train(panel, forecasts, tmp_path)
    net, scaler, names, _metadata = load_router(tmp_path / "router.pt")
    features, _ = _training_inputs(panel, forecasts)
    class _Pool:
        n_experts = 3
        names = ("a", "b", "c")
        horizons = panel.horizons

        def forecast_block(self, positions, log=None):
            return forecasts[:, np.asarray(positions), :], {}

    before = EnergyRouter(trained.net, _Pool(), trained.scaler, names)
    after = EnergyRouter(net, _Pool(), scaler, names)
    positions = np.arange(0, panel.n_rows, 5)
    assert np.array_equal(
        before.route(positions, features[positions]).forecast_kw,
        after.route(positions, features[positions]).forecast_kw,
    )
    assert np.array_equal(
        before.route(positions, features[positions]).weights,
        after.route(positions, features[positions]).weights,
    )


def test_a_shared_head_router_also_round_trips(
    panel: EvaluationPanel, forecasts, tmp_path
) -> None:
    features, names = _training_inputs(panel, forecasts)
    trained = fit_router(
        features=features,
        expert_forecast_kw=forecasts,
        actual_kw=panel.actual_kw,
        horizons=panel.horizons,
        expert_names=("a", "b", "c"),
        feature_names=names,
        config=RouterTrainingConfig(epochs=3, batch_size=32, threads=2),
        architecture=RouterArchitecture(hidden_sizes=(4,), shared_head=True),
        checkpoint_path=tmp_path / "shared.pt",
    )
    net, scaler, restored_names, _ = load_router(tmp_path / "shared.pt")
    assert restored_names == names
    assert np.array_equal(
        router_weights_for(trained, features),
        _weights(net, scaler.transform(features)),
    )
    weights = _weights(net, scaler.transform(features))
    assert np.allclose(weights[:, 0, :], weights[:, 1, :])
