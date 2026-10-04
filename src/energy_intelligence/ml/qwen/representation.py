"""The bridge from SMART-DS numbers into Qwen's embedding space.

The problem is that Qwen is a language model and this task is numerical. The bridge
has to be explicit, because every shortcut here is a silent failure mode: a float
routed through the vocabulary embedding learns nothing about magnitude, and a
promised-textual encoding of a time series invents structure the data does not have.

The design, in four documented steps:

    raw kW per customer
      -> per-unit by that series' own rated kW       (same scaling as Phase 5)
      -> ordered channels, exactly Phase 5's         (demand, dy/dt, sin/cos hour,
                                                       sin/cos day-of-year)
      -> patched and linearly projected               (P sampled positions x C
                                                       channels -> one 2048 token)
      -> Qwen residual stream                        (Qwen's RoPE supplies relative
                                                       position; the cyclic channels
                                                       supply absolute time)

The vocabulary embedding table and the language-model head are **not used**. Both are
documented consequences rather than omissions: 151,936 embedding rows describe token
frequencies in text, which say nothing about kilowatts, and producing logits over
151,936 rows to emit one regression output would be a 311 M-parameter multiply per
position for no information.

On the front end being a *random* projection: with a frozen backbone there is no
gradient path to the projector without backpropagating through 1.7 B parameters, so
the projector is fixed and randomly initialised from a recorded seed. That is the
standard random-features regime, and its consequence is stated wherever results are
reported: a good probe result shows the backbone's activations are linearly
informative, and cannot be attributed to the projector's ~100 k parameters.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch
from torch import nn

__all__ = [
    "EnergyRepresentation",
    "EnergyPatchProjector",
    "representation_version",
    "build_patches",
    "extract_backbone_features",
]


@dataclass(frozen=True, slots=True)
class EnergyRepresentation:
    """How a demand window becomes tokens for Qwen.

    Attributes:
        patch_size: Sampled positions folded into one token. ``8`` at Phase 5's token
            stride of 4 means each token spans 32 native minutes, so a week becomes
            ``168 / 8 = 21`` tokens.
        hidden_size: Must equal Qwen's ``hidden_size``, because the projector writes
            straight into the residual stream.
        layer_norm_input: Whether the flattened patch is LayerNormed before the
            projection. LayerNorm is per-token, so it cannot move information between
            rows or between positions.
        aggregate: How a token sequence is reduced to one feature vector. ``last``
            takes the final position, which under causal attention has attended to the
            whole window; ``mean`` averages every position.
        projector_seed: Seed for the fixed random front-end projection.
    """

    patch_size: int = 8
    hidden_size: int = 2048
    layer_norm_input: bool = True
    aggregate: str = "last"
    projector_seed: int = 20260101

    def token_count(self, sampled_positions: int) -> int:
        """Tokens produced for a window of ``sampled_positions`` sampled points."""
        if sampled_positions % self.patch_size:
            raise ValueError(
                f"patch_size {self.patch_size} does not divide {sampled_positions} "
                "sampled positions; the last partial patch would be silently dropped"
            )
        return sampled_positions // self.patch_size

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__slots__}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EnergyRepresentation":
        return cls(**{name: payload[name] for name in cls.__slots__})


def representation_version(representation: EnergyRepresentation, *, in_channels: int) -> str:
    """A content hash of the representation, recorded in every artifact.

    Two feature caches are only comparable if the same mapping produced them, so the
    mapping gets a version the way the dataset and the sequence index already do.
    """
    payload = json.dumps(
        {"representation": representation.to_dict(), "in_channels": in_channels},
        sort_keys=True,
    )
    return "er-" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:10]


class EnergyPatchProjector(nn.Module):
    """Projects flattened energy patches into Qwen's residual stream.

    Attributes:
        in_features: ``patch_size * in_channels``.
        hidden_size: Qwen's residual width.
        norm: Optional LayerNorm over the flattened patch.
        projection: The fixed linear map into the residual stream.

    Initialisation is deterministic given ``seed``: a generator with that seed builds
    the projection. Determinism matters more here than a good initialisation, because
    the projector is frozen during feature extraction and a different draw per run
    would make two feature caches incomparable for reasons that have nothing to do
    with the backbone.
    """

    def __init__(
        self,
        *,
        patch_size: int,
        in_channels: int,
        hidden_size: int,
        layer_norm_input: bool = True,
        seed: int = 20260101,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.patch_size = int(patch_size)
        self.in_channels = int(in_channels)
        self.hidden_size = int(hidden_size)
        self.in_features = self.patch_size * self.in_channels
        self.seed = int(seed)
        self.norm = nn.LayerNorm(self.in_features) if layer_norm_input else nn.Identity()
        projection = nn.Linear(self.in_features, self.hidden_size, bias=True, dtype=dtype)
        generator = torch.Generator().manual_seed(self.seed)
        with torch.no_grad():
            bound = 1.0 / max(1.0, self.in_features**0.5)
            projection.weight.uniform_(-bound, bound, generator=generator)
            projection.bias.zero_()
        self.projection = projection

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        """Map ``[B, tokens, patch_size * channels]`` to ``[B, tokens, hidden_size]``."""
        if patches.dim() != 3:
            raise ValueError(
                f"expected [batch, tokens, features], got {tuple(patches.shape)}"
            )
        if patches.shape[-1] != self.in_features:
            raise ValueError(
                f"expected last dimension {self.in_features}, got {patches.shape[-1]}"
            )
        return self.projection(self.norm(patches))

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_patches(
    windows: np.ndarray,
    *,
    patch_size: int,
) -> tuple[np.ndarray, int]:
    """Fold ``[B, positions, channels]`` windows into ``[B, tokens, patch*channels]``.

    Args:
        windows: Per-unit channel windows, shape ``[B, positions, channels]``.
        patch_size: Sampled positions per token.

    Returns:
        ``(patches, token_count)``.

    Raises:
        ValueError: ``patch_size`` does not divide the position count. Silently
            dropping a ragged tail would shorten the context without saying so, which
            is precisely the quiet change that invalidates a comparison.
    """
    if windows.ndim != 3:
        raise ValueError(f"expected [batch, positions, channels], got {windows.shape}")
    batch, positions, channels = windows.shape
    if positions % patch_size:
        raise ValueError(
            f"patch_size {patch_size} does not divide {positions} positions; "
            "a ragged tail would be dropped without notice"
        )
    tokens = positions // patch_size
    grouped = windows.reshape(batch, tokens, patch_size * channels)
    return np.ascontiguousarray(grouped), tokens


@torch.no_grad()
def extract_backbone_features(
    backbone: nn.Module,
    patches: np.ndarray,
    *,
    projector: EnergyPatchProjector,
    layers: tuple[int, ...],
    aggregate: str,
    batch_size: int = 64,
    dtype: torch.dtype = torch.float32,
    progress: Callable[[int, int], None] | None = None,
) -> dict[int, np.ndarray]:
    """Run windows through the frozen backbone and return per-layer features.

    The backbone is frozen, so this is a pure function of its input and the mapping -
    which is why the output is cached. Caching is not a convenience: without it every
    head experiment would re-pay a 1.7 B-parameter forward pass.

    Args:
        backbone: A ``Qwen3Model`` trunk with ``requires_grad=False``.
        patches: ``[N, tokens, patch*channels]``, already per-unit.
        projector: The fixed front-end projection. **One instance for the whole
            extraction**: a fresh projector per batch would inject batch-dependent
            noise into the features.
        layers: Hidden-state indices to keep. ``0`` is the embedding output, ``28``
            the final layer. Keeping several makes layer choice an ablation without
            re-running the backbone.
        aggregate: ``"last"`` or ``"mean"``; see :class:`EnergyRepresentation`.
        batch_size: Forward batch size. Larger amortises weight streaming; measured
            best at 64 on this machine.
        dtype: Compute dtype for the backbone.
        progress: Optional callable taking ``(done, total)``.

    Returns:
        ``{layer_index: features}``, each shaped ``[N, hidden_size]`` in float32.
    """
    backbone.eval()
    collected: dict[int, list[np.ndarray]] = {layer: [] for layer in layers}
    total = int(patches.shape[0])

    for start in range(0, total, batch_size):
        block = torch.from_numpy(
            np.ascontiguousarray(patches[start : start + batch_size])
        ).to(dtype)
        outputs = backbone(
            inputs_embeds=projector(block),
            output_hidden_states=True,
            use_cache=False,
        )
        for layer in layers:
            hidden = outputs.hidden_states[layer]
            if aggregate == "last":
                pooled = hidden[:, -1, :]
            elif aggregate == "mean":
                pooled = hidden.mean(dim=1)
            else:
                raise ValueError(
                    f"unknown aggregate {aggregate!r}; expected 'last' or 'mean'"
                )
            collected[layer].append(pooled.to(torch.float32).numpy())
        if progress is not None:
            progress(min(start + batch_size, total), total)

    return {
        layer: np.concatenate(blocks, axis=0) for layer, blocks in collected.items()
    }
