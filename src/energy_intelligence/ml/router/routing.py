"""Deployable routing: turn router features plus expert forecasts into one forecast.

This module is the production path. It is deliberately the **only** place that mixes
expert outputs, and it cannot see a realised target, so a routing decision cannot be
computed from one. The offline oracle that *can* - which expert would have been best in
hindsight - lives in ``router.offline.oracle`` and is not importable from here. A test
asserts that.

The output is not just a number. :class:`RoutingDecision` carries the weights, the
selected expert, the per-expert forecasts and the mixed forecast for the same rows, so
a routing decision can be inspected after the fact rather than taken on trust. Writing
that for every row of the test split is 179,520 x 3 x 3 weights - 6.5 MB as float32 -
which is affordable, so no sampling is needed and nothing is hidden by sampling.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from .experts import ExpertPool
from .model import EnergyRouterNet

__all__ = [
    "ROUTING_STRATEGIES",
    "RoutingDecision",
    "EnergyRouter",
    "mix",
    "hard_selection",
]


class ROUTING_STRATEGIES:
    SOFT = "soft"
    HARD = "hard"


ROUTING_STRATEGY_NAMES: tuple[str, ...] = (ROUTING_STRATEGIES.SOFT, ROUTING_STRATEGIES.HARD)


@dataclass(frozen=True, slots=True)
class RoutingDecision:
    """A routing decision and the arithmetic that produced the forecast.

    Attributes:
        positions: Panel positions these rows correspond to.
        horizons: Horizon steps.
        expert_names: Expert names, in pool order.
        weights: ``[N, horizons, experts]`` routing weights, rows summing to one.
        selected: ``[N, horizons]`` index of the highest-weight expert per row.
        strategy: ``soft`` or ``hard``; ``hard`` reports one-hot weights.
        forecast_kw: ``[N, horizons]`` the final forecast in kW.
        expert_forecast_kw: ``[n_experts, N, horizons]`` each expert's own forecast.
        router_feature_names: The columns the weights were computed from.
        seconds: Wall time for the router forward pass and the mix.
    """

    positions: np.ndarray
    horizons: tuple[int, ...]
    expert_names: tuple[str, ...]
    weights: np.ndarray
    selected: np.ndarray
    strategy: str
    forecast_kw: np.ndarray
    expert_forecast_kw: np.ndarray
    router_feature_names: tuple[str, ...]
    seconds: float = 0.0

    @property
    def n_rows(self) -> int:
        return int(self.positions.size)

    def weight_totals(self) -> np.ndarray:
        """Row sums of the weights, which must be one everywhere."""
        return self.weights.sum(axis=2)

    def selected_names(self) -> np.ndarray:
        """``[N, horizons]`` expert names, as strings."""
        table = np.asarray(self.expert_names, dtype=object)
        return table[self.selected]

    def capacity(self) -> dict[str, dict[str, float]]:
        """Share of rows each expert is selected for, per horizon.

        An MoE whose experts each handle a few percent of rows has added parameters
        without buying coverage, so this is reported rather than assumed.
        """
        out: dict[str, dict[str, float]] = {}
        for index, horizon in enumerate(self.horizons):
            picks = self.selected[:, index]
            total = max(1, int(picks.size))
            out[str(horizon)] = {
                name: float(np.count_nonzero(picks == i)) / total
                for i, name in enumerate(self.expert_names)
            }
        return out

    def mean_weights(self) -> dict[str, dict[str, float]]:
        """Mean weight per expert, per horizon."""
        return {
            str(horizon): {
                name: float(self.weights[:, index, i].mean())
                for i, name in enumerate(self.expert_names)
            }
            for index, horizon in enumerate(self.horizons)
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_rows": self.n_rows,
            "horizons": list(self.horizons),
            "experts": list(self.expert_names),
            "strategy": self.strategy,
            "mean_weights": self.mean_weights(),
            "capacity": self.capacity(),
            "router_features": list(self.router_feature_names),
            "seconds": round(self.seconds, 6),
            "max_weight_sum_error": float(np.max(np.abs(self.weight_totals() - 1.0)))
            if self.n_rows
            else 0.0,
        }


def hard_selection(weights: np.ndarray) -> np.ndarray:
    """One-hot the largest weight per row and horizon.

    Args:
        weights: ``[N, horizons, experts]``.

    Returns:
        The same shape, one-hot. Ties resolve to the lowest expert index, which is
        deterministic rather than arbitrary.
    """
    weights = np.asarray(weights, dtype=np.float64)
    picks = np.argmax(weights, axis=2)
    one_hot = np.zeros_like(weights)
    rows = np.arange(weights.shape[0])[:, None]
    horizon_columns = np.arange(weights.shape[1])[None, :]
    one_hot[rows, horizon_columns, picks] = 1.0
    return one_hot


def mix(weights: np.ndarray, expert_forecast_kw: np.ndarray) -> np.ndarray:
    """Combine expert forecasts with routing weights.

    Args:
        weights: ``[N, horizons, experts]`` summing to one per row.
        expert_forecast_kw: ``[n_experts, N, horizons]`` in kW.

    Returns:
        ``[N, horizons]`` in kW.

    Raises:
        ValueError: If the weights do not sum to one, or a weight is negative. Both
            would make the "forecast" something other than a convex combination, which
            is the property that makes this a router rather than a regressor.
    """
    weights = np.asarray(weights, dtype=np.float64)
    expert_forecast_kw = np.asarray(expert_forecast_kw, dtype=np.float64)
    if weights.ndim != 3:
        raise ValueError(f"weights must be [rows, horizons, experts], got {weights.shape}")
    if expert_forecast_kw.ndim != 3:
        raise ValueError(
            f"expert forecasts must be [experts, rows, horizons], got {expert_forecast_kw.shape}"
        )
    if weights.shape[0] != expert_forecast_kw.shape[1] or weights.shape[1] != expert_forecast_kw.shape[2]:
        raise ValueError(
            f"weights {weights.shape} do not match expert forecasts "
            f"{expert_forecast_kw.shape}"
        )
    if weights.shape[2] != expert_forecast_kw.shape[0]:
        raise ValueError(
            f"{weights.shape[2]} weights for {expert_forecast_kw.shape[0]} experts"
        )
    if np.any(weights < -1e-9):
        raise ValueError("routing weights must be non-negative; the softmax guarantees it")
    totals = weights.sum(axis=2)
    if not np.allclose(totals, 1.0, atol=1e-6):
        worst = float(np.max(np.abs(totals - 1.0)))
        raise ValueError(f"routing weights must sum to one; worst deviation {worst:.3e}")
    return np.einsum("nhe,enh->nh", weights, expert_forecast_kw)


class EnergyRouter:
    """A fitted router plus the pool it routes over.

    The object is the deployable unit: ``route`` takes panel positions, asks the pool
    for forecasts, asks the net for weights, and returns both the forecast and the
    decision. It never receives the realised target, which is what makes it safe to
    call in a live loop.
    """

    def __init__(
        self,
        net: EnergyRouterNet,
        pool: ExpertPool,
        feature_scaler: Any,
        feature_names: tuple[str, ...],
    ) -> None:
        self.net = net
        self.pool = pool
        self.feature_scaler = feature_scaler
        self.feature_names = tuple(feature_names)
        if net.n_experts != pool.n_experts:
            raise ValueError(
                f"the router has {net.n_experts} output columns but the pool has "
                f"{pool.n_experts} experts"
            )
        if net.n_features != len(self.feature_names):
            raise ValueError(
                f"the router expects {net.n_features} features but {len(self.feature_names)} "
                f"names were given"
            )
        net.eval()

    @torch.no_grad()
    def weights(self, features: np.ndarray) -> np.ndarray:
        """``[N, horizons, experts]`` weights for a feature matrix.

        Args:
            features: ``[N, n_features]`` origin-observable features, unscaled.

        Returns:
            Weights per row, in the net's own order.
        """
        scaled = self.feature_scaler.transform(np.asarray(features, dtype=np.float64))
        tensor = torch.from_numpy(np.ascontiguousarray(scaled)).to(torch.float32)
        return self.net(tensor).numpy().astype(np.float64)

    def route(
        self,
        positions: np.ndarray,
        features: np.ndarray,
        *,
        strategy: str = ROUTING_STRATEGIES.SOFT,
        forecasts: np.ndarray | None = None,
        expert_seconds: float | None = None,
        log: Any = None,
    ) -> RoutingDecision:
        """Route a set of panel rows.

        Args:
            positions: Panel positions to route.
            features: ``[len(positions), n_features]`` features in the same order.
            strategy: ``soft`` mixes by weight; ``hard`` routes every row to its single
                highest-weight expert.
            forecasts: ``[n_experts, len(positions), horizons]`` in kW, when the caller
                already has them. Supplied to avoid asking a 74k-parameter network for
                forecasts the ensemble baselines also need.
            expert_seconds: Wall time already spent on ``forecasts``, for the cost table.
            log: Optional progress callable.

        Returns:
            The decision.

        Raises:
            ValueError: If ``strategy`` is unknown, or the feature count is wrong.
        """
        if strategy not in ROUTING_STRATEGY_NAMES:
            raise ValueError(
                f"routing strategy must be one of {list(ROUTING_STRATEGY_NAMES)}, "
                f"got {strategy!r}"
            )
        positions = np.asarray(positions, dtype=np.int64)
        features = np.asarray(features, dtype=np.float64)
        if features.shape[0] != positions.size:
            raise ValueError(
                f"got {features.shape[0]} feature rows for {positions.size} positions"
            )
        if features.shape[1] != len(self.feature_names):
            raise ValueError(
                f"expected {len(self.feature_names)} features, got {features.shape[1]}"
            )

        started = time.perf_counter()
        if forecasts is None:
            forecasts, _timings = self.pool.forecast_block(positions, log=log)
        forecasts = np.asarray(forecasts, dtype=np.float64)
        if forecasts.shape != (self.pool.n_experts, positions.size, len(self.pool.horizons)):
            raise ValueError(
                f"expert forecasts must be "
                f"{(self.pool.n_experts, positions.size, len(self.pool.horizons))}, "
                f"got {forecasts.shape}"
            )
        weights = self.weights(features)
        if strategy == ROUTING_STRATEGIES.HARD:
            weights = hard_selection(weights)
        forecast_kw = mix(weights, forecasts)
        seconds = time.perf_counter() - started + (expert_seconds or 0.0)
        return RoutingDecision(
            positions=positions,
            horizons=self.pool.horizons,
            expert_names=self.pool.names,
            weights=weights,
            selected=np.argmax(weights, axis=2),
            strategy=strategy,
            forecast_kw=forecast_kw,
            expert_forecast_kw=forecasts,
            router_feature_names=self.feature_names,
            seconds=seconds,
        )