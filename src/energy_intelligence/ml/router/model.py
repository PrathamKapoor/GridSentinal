"""The learned router itself: origin-observable features in, per-horizon expert weights out.

One small MLP, one output head per horizon, softmax over experts inside each head. The
softmax is what makes the output a routing decision rather than a prediction: every
weight is non-negative, they sum to one, and dropping to a hard selection is an
``argmax`` rather than a change of architecture.

Two design choices are worth defending, because both were available and both were
rejected on the evidence rather than on taste.

**Horizon-specific heads, not the horizon as an input feature.** Phase 5 measured that
the best expert changes with horizon - persistence is untouchable at 15 minutes and
loses badly at 24 hours - and Phase 5's own ``single_horizon`` ablation showed that
*sharing* the trunk across horizons helps by 38.5%. Separate heads give each horizon
its own decision while keeping one shared trunk, which is the same trade-off Phase 5
measured and won.

**The router does not see the experts' forecasts.** It could: the experts have already
been run, so their spread is available and is genuinely informative about uncertainty.
It does not, because then this stops being a router and becomes a blender that can
regress a new forecast rather than a weight over experts that already exist. The
ablation ``no_past_error`` measures how much the lagged expert *outcomes* are worth
without letting the router re-predict anything.

The module is deliberately import-light: torch only, no sklearn, no Phase 5 imports.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

__all__ = [
    "ROUTER_ARCHITECTURES",
    "RouterArchitecture",
    "EnergyRouterNet",
    "build_router",
    "simplex_weights",
]


class ROUTER_ARCHITECTURES:
    TINY_MLP = "tiny_mlp"
    LINEAR = "linear"


ARCHITECTURE_NAMES: tuple[str, ...] = (ROUTER_ARCHITECTURES.TINY_MLP, ROUTER_ARCHITECTURES.LINEAR)


@dataclass(frozen=True, slots=True)
class RouterArchitecture:
    """Router shape.

    Attributes:
        name: ``tiny_mlp`` or ``linear``.
        hidden_sizes: Hidden widths; empty for ``linear``.
        dropout: Dropout applied after each hidden layer.
        temperature: Softmax temperature. ``1.0`` is a plain softmax; above one the
            weights flatten toward uniform, below one they sharpen. It is a single
            scalar over all heads, so it cannot express "be sharp at 15 minutes and
            hedged at 24 hours" - that is the heads' business, not the temperature's.
        shared_head: One weight vector for every horizon instead of one head each. The
            ablation that asks whether expert preference really varies with horizon, and
            which changes *only* the output head so the answer is not confounded with a
            change of inputs.
    """

    name: str = ROUTER_ARCHITECTURES.TINY_MLP
    hidden_sizes: tuple[int, ...] = (32,)
    dropout: float = 0.0
    temperature: float = 1.0
    shared_head: bool = False

    def __post_init__(self) -> None:
        if self.name not in ARCHITECTURE_NAMES:
            raise ValueError(
                f"router architecture must be one of {list(ARCHITECTURE_NAMES)}, "
                f"got {self.name!r}"
            )
        if self.name == ROUTER_ARCHITECTURES.LINEAR and self.hidden_sizes:
            raise ValueError("the linear router has no hidden layers")
        if any(w <= 0 for w in self.hidden_sizes):
            raise ValueError("router hidden sizes must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("router dropout must lie in [0, 1)")
        if self.temperature <= 0.0:
            raise ValueError("router temperature must be positive")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "hidden_sizes": list(self.hidden_sizes),
            "dropout": self.dropout,
            "temperature": self.temperature,
            "shared_head": self.shared_head,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RouterArchitecture":
        return cls(
            name=str(payload.get("name", ROUTER_ARCHITECTURES.TINY_MLP)),
            hidden_sizes=tuple(int(w) for w in payload.get("hidden_sizes", (32,))),
            dropout=float(payload.get("dropout", 0.0)),
            temperature=float(payload.get("temperature", 1.0)),
            shared_head=bool(payload.get("shared_head", False)),
        )


def simplex_weights(logits: torch.Tensor, *, temperature: float = 1.0) -> torch.Tensor:
    """Softmax over the last axis, so the output is a set of expert weights.

    Args:
        logits: ``[..., n_experts]`` scores.
        temperature: Dividing the logits before the softmax.

    Returns:
        ``[..., n_experts]`` non-negative weights summing to one.
    """
    return torch.softmax(logits / temperature, dim=-1)


#: Smallest weight a warm start may hold. A fixed ensemble can legitimately put exactly
#: zero on an expert, and ``log(0)`` is not a number; the floor keeps the bias finite
#: while leaving the weight far below anything the router would choose on purpose.
_MIN_WEIGHT = 1e-4


def _log_weights(weights: np.ndarray) -> torch.Tensor:
    """Log of a simplex weight matrix, floored so it stays finite.

    Args:
        weights: ``[horizons, experts]`` or ``[experts]``, rows summing to one.

    Returns:
        A float32 tensor of the same shape.

    Raises:
        ValueError: If a row does not sum to one, or the shape is not one of the two
            accepted forms.
    """
    array = np.asarray(weights, dtype=np.float64)
    if array.ndim not in (1, 2):
        raise ValueError(f"initial weights must be [experts] or [horizons, experts], got {array.shape}")
    if np.any(array < -1e-12):
        raise ValueError("initial weights must be non-negative")
    if not np.allclose(array.sum(axis=-1), 1.0, atol=1e-6):
        raise ValueError("initial weights must sum to one on every row")
    return torch.from_numpy(np.log(np.clip(array, _MIN_WEIGHT, 1.0)).astype(np.float32))


class EnergyRouterNet(nn.Module):
    """Features in, ``[batch, horizons, experts]`` weights out.

    **The router is trained as a residual on the state-independent optimum.** The
    cold-start failure this avoids is specific and was measured: initialised near
    uniform, an L1 objective through a softmax drives the logits apart and the router
    settles on a single expert per horizon within a few epochs. At 15 minutes that is
    harmless; at 1 hour and 24 hours it cost more than a fixed weighted ensemble
    achieved, because a hedge that reduces absolute error is exactly the solution a
    vertex cannot express and whose gradient vanishes as it is approached. Passing the
    validation-fitted fixed-ensemble weights as ``initial_weights`` starts the router at
    that hedge, so everything it subsequently learns is the part that genuinely depends
    on the state. Those weights come from the router's own training rows, so nothing
    about the test split enters.

    Attributes:
        architecture: The shape this net was built with.
        n_features: Router feature count.
        horizons: Horizon steps, one head each.
        n_experts: Experts in the pool.
        warm_started: Whether ``initial_weights`` was supplied.
    """

    def __init__(
        self,
        *,
        n_features: int,
        horizons: tuple[int, ...],
        n_experts: int,
        architecture: RouterArchitecture | None = None,
        seed: int = 20260101,
        initial_weights: np.ndarray | None = None,
    ) -> None:
        super().__init__()
        self.architecture = architecture or RouterArchitecture()
        if n_features <= 0:
            raise ValueError("the router needs at least one feature")
        if not horizons:
            raise ValueError("the router needs at least one horizon")
        if n_experts < 2:
            raise ValueError(
                "a router over fewer than two experts is a constant; the pool is the "
                "point"
            )
        self.n_features = int(n_features)
        self.horizons = tuple(int(h) for h in horizons)
        self.n_experts = int(n_experts)

        # Seeded explicitly rather than left to global RNG state, so a router rebuilt
        # from its configuration starts from the same initialisation. The global RNG
        # state is saved and restored around construction because ``nn.Linear`` draws
        # from it, which would otherwise make every other torch consumer in the process
        # depend on how many routers had been built.
        rng_state = torch.random.get_rng_state()
        try:
            torch.manual_seed(int(seed))
            layers: list[nn.Module] = []
            width = self.n_features
            for hidden in self.architecture.hidden_sizes:
                layers.append(nn.Linear(width, int(hidden)))
                layers.append(nn.ReLU())
                if self.architecture.dropout:
                    layers.append(nn.Dropout(float(self.architecture.dropout)))
                width = int(hidden)
            head_width = (
                self.n_experts
                if self.architecture.shared_head
                else self.n_experts * len(self.horizons)
            )
            self.trunk = nn.Sequential(*layers)
            self.heads = nn.Linear(width, head_width)
        finally:
            torch.random.set_rng_state(rng_state)

        self.warm_started = initial_weights is not None
        with torch.no_grad():
            if self.warm_started:
                bias = _log_weights(np.asarray(initial_weights, dtype=np.float64))
                self.heads.weight.zero_()
                self.heads.bias.copy_(
                    bias.reshape(-1)
                    if not self.architecture.shared_head
                    else bias.reshape(-1)[: self.n_experts]
                )
            else:
                # A small non-zero head bias breaks the symmetry between experts without
                # giving any of them a head start: zeros would make every row's initial
                # weights exactly uniform, and the first gradient step would be the only
                # thing distinguishing the experts. Deterministic by construction.
                self.heads.bias.copy_(torch.linspace(-0.05, 0.05, head_width))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Route a batch of rows.

        Args:
            features: ``[batch, n_features]`` standardised router features.

        Returns:
            ``[batch, horizons, n_experts]`` weights, each row summing to one.
        """
        return simplex_weights(self.logits(features), temperature=self.architecture.temperature)

    def logits(self, features: torch.Tensor) -> torch.Tensor:
        """Pre-softmax scores, for temperature and stability diagnostics.

        With ``shared_head`` the single vector is broadcast to every horizon, so the
        only thing that changes across horizons is the input state, not the mapping.
        """
        if features.ndim != 2 or features.shape[1] != self.n_features:
            raise ValueError(
                f"expected [batch, {self.n_features}] features, got {tuple(features.shape)}"
            )
        scores = self.heads(self.trunk(features))
        if self.architecture.shared_head:
            return scores[:, None, :].expand(-1, len(self.horizons), -1)
        return scores.reshape(-1, len(self.horizons), self.n_experts)

    def parameter_count(self) -> int:
        return int(sum(p.numel() for p in self.parameters()))

    def trainable_parameter_count(self) -> int:
        return int(sum(p.numel() for p in self.parameters() if p.requires_grad))

    def config_dict(self) -> dict[str, Any]:
        return {
            "n_features": self.n_features,
            "horizons": list(self.horizons),
            "n_experts": self.n_experts,
            "architecture": self.architecture.to_dict(),
            "warm_started": self.warm_started,
        }


def build_router(
    *,
    n_features: int,
    horizons: tuple[int, ...],
    n_experts: int,
    architecture: RouterArchitecture | None = None,
    seed: int = 20260101,
    initial_weights: np.ndarray | None = None,
) -> EnergyRouterNet:
    """Construct a router with a seeded initialisation.

    Args:
        n_features: Router feature count.
        horizons: Horizon steps.
        n_experts: Experts in the pool.
        architecture: Router shape.
        seed: Initialisation seed.
        initial_weights: ``[horizons, experts]`` on the simplex to start from. When given,
            the head weight matrix starts at zero and the bias at ``log(w)``, so the very
            first forward pass reproduces that mixture exactly and training can only move
            away from it if that lowers the loss. See :class:`EnergyRouterNet`.
    """
    return EnergyRouterNet(
        n_features=n_features,
        horizons=horizons,
        n_experts=n_experts,
        architecture=architecture,
        seed=seed,
        initial_weights=initial_weights,
    )