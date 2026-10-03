"""Temporal architectures for the Energy **Demand** Dynamics Model.

Scope discipline
----------------
This is a demand forecasting model. It is **not** an Energy World Model, not a
digital twin, not action-conditioned, and it is not named as one: SMART-DS supplies
no action, no transition and no system response, so any such claim would be
unsupported. See ``docs/world_model_requirements.md``.

Two architectures, chosen for two different questions.

``tcn`` - **primary.** A dilated causal temporal convolutional encoder with a shared
representation and one linear head per horizon.

Why this one, from Phase 4 evidence rather than fashion:

* Phase 4's error analysis put the error in the *morning and evening ramps* (hours
  5, 6, 7, 16, 17) and in November-December, and in a small set of high-volatility
  series. That is a problem about the local shape of the window, which is exactly
  what a convolution reads.
* Phase 4's GRU cost 1232 s and lost to gradient boosting at every horizon. On CPU,
  recurrence is the wrong default here; dilated convolution is fully parallel and
  reaches the same receptive field.
* A daily- and weekly-periodic signal is translation-equivariant, which is the
  inductive bias a convolution already has and an MLP does not.
* It is cheap enough to run a controlled set of ablations on a CPU budget.

``transformer`` - **secondary, and the point is the comparison.** A compact
patch-based encoder: the sequence is cut into patches, embedded, given positional
information, and passed through a small number of transformer blocks.

It exists to answer one question for Phase 6: **does attention buy anything on this
data?** The eventual architecture is a 1.7B-parameter decoder-only foundation model,
so "we never tested attention on the real data" is not an acceptable answer. This is
not the recommendation - it is the measurement.

Both are multi-horizon: one shared encoder, one head per horizon, no horizon dropped.
"""

from __future__ import annotations

import warnings
from typing import Any

import torch
from torch import nn

__all__ = [
    "ARCHITECTURES",
    "model_defaults",
    "TemporalDemandModel",
    "build_model",
]


class ARCHITECTURES:
    """The two architectures under test."""

    TCN = "tcn"
    TRANSFORMER = "transformer"


#: Reported in this order.
ARCHITECTURE_NAMES: tuple[str, ...] = (ARCHITECTURES.TCN, ARCHITECTURES.TRANSFORMER)


def model_defaults(architecture: str) -> dict[str, Any]:
    """Hyperparameters for an architecture.

    Deliberately small. Phase 5 is testing whether temporal inductive bias helps, not
    whether a bigger network helps; a parameter sweep would answer the second
    question and confound the first.
    """
    if architecture == ARCHITECTURES.TCN:
        return {
            "channels": 48,
            "kernel_size": 3,
            "dilations": [1, 2, 4, 8, 16, 32, 64],
            "dropout": 0.05,
            "pool": "last_plus_mean",
            "hidden_size": 64,
            "series_embedding_dim": 8,
            "note": (
                "receptive field 509 tokens covers the whole 168-token window; "
                "GroupNorm rather than BatchNorm so behaviour does not depend on "
                "batch composition"
            ),
        }
    if architecture == ARCHITECTURES.TRANSFORMER:
        return {
            "d_model": 64,
            "num_heads": 4,
            "num_layers": 2,
            "patch_tokens": 8,
            "dropout": 0.05,
            "pool": "last_plus_mean",
            "hidden_size": 64,
            "series_embedding_dim": 8,
            "note": (
                "patches of 8 tokens over the same 168-token window, so it sees the "
                "same information as the TCN with a different mixing mechanism"
            ),
        }
    raise KeyError(
        f"unknown architecture {architecture!r}; available: {list(ARCHITECTURE_NAMES)}"
    )


class _ResidualBlock(nn.Module):
    """Dilated causal convolution block, residual, with normalisation."""

    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.padding = padding
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, 1)
        # GroupNorm over the whole window, not per position. This mixes across
        # positions INSIDE the window, which is deliberate and not leakage: the
        # window ends at the forecast origin, so no post-origin value is present
        # for the norm to see (proved by ml.sequence.assert_sequences_are_causal).
        # It was chosen over BatchNorm because its behaviour does not depend on
        # batch composition, and over per-position LayerNorm because a fixed window
        # is normalised as a whole - the model never sees a partial window, so
        # there is nothing to stream-conserve.
        self.norm1 = nn.GroupNorm(1, channels)
        self.norm2 = nn.GroupNorm(1, channels)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Left-pad so every output position sees only past and present inputs."""
        h = self.conv1(nn.functional.pad(x, (self.padding, 0)))
        h = self.activation(self.norm1(h))
        h = self.dropout(self.conv2(h))
        return self.activation(self.norm2(x + h))


class _TCNEncoder(nn.Module):
    """Dilated causal encoder producing a per-position representation."""

    def __init__(self, in_channels: int, params: dict[str, Any]) -> None:
        super().__init__()
        self.input_projection = nn.Conv1d(in_channels, params["channels"], 1)
        self.blocks = nn.ModuleList(
            _ResidualBlock(
                params["channels"], params["kernel_size"], dilation, params["dropout"]
            )
            for dilation in params["dilations"]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_projection(x)
        for block in self.blocks:
            h = block(h)
        return h


class _TransformerEncoder(nn.Module):
    """Compact patch transformer producing one representation per patch."""

    def __init__(self, in_channels: int, params: dict[str, Any]) -> None:
        super().__init__()
        patch_tokens = params["patch_tokens"]
        self.patch_tokens = patch_tokens
        self.patch_projection = nn.Linear(in_channels * patch_tokens, params["d_model"])
        self.position = nn.Parameter(torch.zeros(1, 512, params["d_model"]))
        nn.init.normal_(self.position, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=params["d_model"],
            nhead=params["num_heads"],
            dim_feedforward=params["d_model"] * 2,
            dropout=params["dropout"],
            batch_first=True,
            norm_first=True,
        )
        # torch warns that its nested-tensor fast path is disabled when
        # ``norm_first=True``. ``norm_first`` is deliberate here (pre-norm trains
        # more stably on a small model), and the warning is a performance hint
        # rather than a correctness problem. The project treats warnings as errors,
        # so it is silenced at the one place it is expected.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=".*enable_nested_tensor is True.*", category=UserWarning
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=params["num_layers"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return ``(batch, channels, tokens)`` - the layout the TCN also uses.

        The transformer naturally works as ``(batch, tokens, channels)``, so the
        result is transposed back. Sharing one layout is what lets a single pooling
        and head implementation serve both architectures; without it the transformer
        path silently pools over the wrong axis.
        """
        # x: (batch, channels, tokens) -> patch embedding
        batch, channels, tokens = x.shape
        usable = (tokens // self.patch_tokens) * self.patch_tokens
        patches = x[:, :, tokens - usable :].reshape(
            batch, channels * self.patch_tokens, usable // self.patch_tokens
        )
        patches = patches.transpose(1, 2)  # (batch, num_patches, channels*patch)
        h = self.patch_projection(patches)
        h = h + self.position[:, : h.shape[1]]
        return self.encoder(h).transpose(1, 2)


class TemporalDemandModel(nn.Module):
    """A shared temporal encoder with one linear head per forecast horizon.

    A real ``nn.Module``, not a wrapper: the optimiser, ``state_dict`` and gradient
    clipping all need to reach the parameters directly, and a dataclass holding
    modules hides them.

    Attributes:
        architecture: ``"tcn"`` or ``"transformer"``.
        encoder: The temporal encoder.
        trunk: Shared head trunk.
        heads: One linear head per horizon, in the dataset's horizon order.
        series_embedding: Optional per-series embedding, because a pooled model must
            still be able to tell a 1.4 kW household from a 388 kW shop (D-066).
        params: The hyperparameters used.
        horizon_count: Number of heads.
    """

    def __init__(
        self,
        architecture: str,
        encoder: nn.Module,
        trunk: nn.Module,
        heads: nn.ModuleList,
        series_embedding: nn.Embedding | None,
        params: dict[str, Any],
    ) -> None:
        super().__init__()
        self.architecture = architecture
        self.encoder = encoder
        self.trunk = trunk
        self.heads = heads
        self.series_embedding = series_embedding
        self.params = dict(params)
        self.horizon_count = len(heads)

    def forward(
        self, x: torch.Tensor, series_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Predict one value per horizon.

        Args:
            x: ``(batch, channels, tokens)``, last token at the forecast origin.
            series_ids: Optional ``(batch,)`` series indices.

        Returns:
            ``(batch, horizons)`` in the dataset's stored units.
        """
        h = self.encoder(x)
        last = h[:, :, -1]
        pooled = h.mean(dim=2)
        representation = torch.cat([last, pooled], dim=1)
        if self.series_embedding is not None:
            # A missing identity is a ZERO embedding, not a shape error: an unknown
            # series must be representable rather than crashing.
            if series_ids is None:
                identity = self.series_embedding.weight.new_zeros(
                    representation.shape[0], self.series_embedding.embedding_dim
                )
            else:
                identity = self.series_embedding(series_ids)
            representation = torch.cat([representation, identity], dim=1)
        outputs = [head(self.trunk(representation)) for head in self.heads]
        return torch.cat(outputs, dim=1)

    def parameter_count(self) -> int:
        return int(sum(p.numel() for p in self.parameters()))

    def trainable_parameter_count(self) -> int:
        return int(sum(p.numel() for p in self.parameters() if p.requires_grad))

    def config_dict(self) -> dict[str, Any]:
        return {
            "architecture": self.architecture,
            "horizon_count": self.horizon_count,
            "series_embedding_dim": (
                self.series_embedding.embedding_dim if self.series_embedding is not None else 0
            ),
            **self.params,
        }

    def __repr__(self) -> str:  # pragma: no cover - developer convenience
        return (
            f"TemporalDemandModel({self.architecture}, horizons={self.horizon_count}, "
            f"params={self.parameter_count()})"
        )


def build_model(
    architecture: str,
    *,
    in_channels: int,
    horizons: tuple[int, ...],
    series_count: int | None = None,
    params: dict[str, Any] | None = None,
) -> TemporalDemandModel:
    """Construct a temporal model.

    Args:
        architecture: One of :data:`ARCHITECTURE_NAMES`.
        in_channels: Channel count the encoder receives.
        horizons: Forecast horizons, one head each.
        series_count: Number of distinct series, for the per-series embedding.
        params: Hyperparameters; defaults from :func:`model_defaults`.

    Returns:
        The assembled model.
    """
    # Partial overrides are merged over the defaults, so a config may change one
    # hyperparameter without restating the architecture.
    resolved = {**model_defaults(architecture), **(params or {})}

    if architecture == ARCHITECTURES.TCN:
        encoder = _TCNEncoder(in_channels, resolved)
    elif architecture == ARCHITECTURES.TRANSFORMER:
        encoder = _TransformerEncoder(in_channels, resolved)
    else:
        raise KeyError(
            f"unknown architecture {architecture!r}; available: {list(ARCHITECTURE_NAMES)}"
        )

    pooled_width = (
        resolved["channels"] * 2
        if architecture == ARCHITECTURES.TCN
        else resolved["d_model"] * 2
    )
    embedding_dim = int(resolved.get("series_embedding_dim", 0))
    series_embedding = (
        nn.Embedding(series_count, embedding_dim) if embedding_dim and series_count else None
    )
    head_input = pooled_width + (embedding_dim if series_embedding is not None else 0)
    hidden = int(resolved.get("hidden_size", 64))

    # One shared trunk, then one linear head per horizon. Sharing the trunk is the
    # whole point of a multi-horizon head: h=1, h=4 and h=96 read the same latent
    # demand representation rather than learning three unrelated mappings.
    trunk = nn.Sequential(nn.Linear(head_input, hidden), nn.GELU())
    heads = nn.ModuleList(nn.Linear(hidden, 1) for _ in horizons)

    return TemporalDemandModel(
        architecture=architecture,
        encoder=encoder,
        trunk=trunk,
        heads=heads,
        series_embedding=series_embedding,
        params=resolved,
    )
