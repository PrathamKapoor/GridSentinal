"""Frozen-backbone feature extraction for the Qwen energy probe.

Two jobs live here.

**Choosing the rows.** A 1.7 B-parameter forward pass costs a measured
0.236 s/sample on this machine, so the full Phase 5 population is out of reach
(15.6 h for training alone). The subsample is therefore chosen *deterministically and
per series*, evenly spaced in time inside each split, so that every arm of the
experiment - pretrained, random, no-cyclic, and every baseline - is scored on exactly
the same rows. That identity is what makes the paired comparison valid.

**Extracting once.** The backbone is frozen, so features are a pure function of the
input and can be cached to disk. Without the cache every head experiment would re-pay
the forward pass, which is the difference between a phase that runs and a phase that
does not.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..sequence import SequenceIndex, gather_sequences
from .checkpoint import Qwen3Checkpoint
from .representation import (
    EnergyPatchProjector,
    EnergyRepresentation,
    build_patches,
    extract_backbone_features,
    representation_version,
)

__all__ = [
    "Subsample",
    "select_subsample",
    "FeatureCache",
    "extract_and_cache",
]


@dataclass(frozen=True, slots=True)
class Subsample:
    """The rows every arm of Phase 6 is scored on.

    Attributes:
        train: Row indices from the training split.
        validation: Row indices from the validation split.
        test: Row indices from the test split.
    """

    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray

    def split(self, name: str) -> np.ndarray:
        return getattr(self, name)

    def version(self) -> str:
        payload = json.dumps(
            {
                "train": hash_rows(self.train),
                "validation": hash_rows(self.validation),
                "test": hash_rows(self.test),
            },
            sort_keys=True,
        )
        return "ss-" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:10]

    def summary(self, index: SequenceIndex) -> dict[str, Any]:
        return {
            "version": self.version(),
            "rows": {
                "train": int(self.train.size),
                "validation": int(self.validation.size),
                "test": int(self.test.size),
            },
            "series_per_split": {
                "train": int(np.unique(index.series[self.train]).size),
                "validation": int(np.unique(index.series[self.validation]).size),
                "test": int(np.unique(index.series[self.test]).size),
            },
            "test_fraction_of_phase5_test": float(
                self.test.size / max(1, index.rows(2).size)
            ),
        }


def hash_rows(rows: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(rows, dtype=np.int64).tobytes()).hexdigest()[:16]


def select_subsample(
    index: SequenceIndex,
    *,
    train_rows: int,
    validation_rows: int,
    test_rows: int,
) -> Subsample:
    """Choose an evenly spaced, per-series balanced subsample from each split.

    Even spacing in time matters for two reasons. It keeps the subsample
    representative across the chronological split rather than clustered at its start,
    and it is deterministic, so a rerun selects the same rows without recording a
    seed somewhere else.

    Args:
        index: The Phase 5 sequence index, reused so row identities match.
        train_rows: Target row count for training.
        validation_rows: Target row count for validation.
        test_rows: Target row count for test.

    Returns:
        The chosen rows.
    """
    def per_series(split_code: int, wanted: int) -> np.ndarray:
        rows = index.rows(split_code)
        series = index.series[rows]
        order = np.argsort(series, kind="stable")
        rows, series = rows[order], series[order]
        unique = np.unique(series)
        if unique.size == 0:
            return rows[:0]
        per = max(1, wanted // unique.size)
        chosen: list[np.ndarray] = []
        for value in unique:
            pool = rows[series == value]
            if pool.size <= per:
                chosen.append(pool)
                continue
            # Evenly spaced positions, endpoints included.
            picks = np.linspace(0, pool.size - 1, per).round().astype(np.int64)
            chosen.append(pool[np.unique(picks)])
        return np.sort(np.concatenate(chosen))

    return Subsample(
        train=per_series(0, train_rows),
        validation=per_series(1, validation_rows),
        test=per_series(2, test_rows),
    )


@dataclass(frozen=True, slots=True)
class FeatureCache:
    """Features for one arm, keyed by everything that could change them.

    Attributes:
        version: Content hash over the row set, mapping, backbone kind and layers.
        layers: The cached hidden-state indices.
        features: ``{layer: array}``, each ``[N, hidden_size]``.
        targets: ``[N, horizons]`` target levels, per-unit.
        persistence: ``[N, horizons]`` persistence levels ``y(t)``, per-unit.
        origins: ``[N]`` native origin indices, for regime and timestamp analysis.
        series: ``[N]`` series indices.
        row_scale: ``[N]`` the series' rated kW, to return to physical units.
    """

    version: str
    layers: tuple[int, ...]
    features: dict[int, np.ndarray]
    targets: np.ndarray
    persistence: np.ndarray
    origins: np.ndarray
    series: np.ndarray
    row_scale: np.ndarray

    def layer(self, index: int) -> np.ndarray:
        if index not in self.features:
            raise KeyError(
                f"layer {index} was not cached; cached layers are {sorted(self.features)}"
            )
        return self.features[index]


def _cache_key(
    *,
    subsample_version: str,
    representation: EnergyRepresentation,
    in_channels: int,
    backbone: str,
    layers: tuple[int, ...],
    aggregate: str,
) -> str:
    payload = json.dumps(
        {
            "subsample": subsample_version,
            "representation": representation.to_dict(),
            "in_channels": in_channels,
            "aggregate": aggregate,
            "backbone": backbone,
            "layers": list(layers),
        },
        sort_keys=True,
    )
    return "fc-" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def extract_and_cache(
    *,
    checkpoint: Qwen3Checkpoint,
    index: SequenceIndex,
    values: np.ndarray,
    scale_vector: np.ndarray,
    subsample: Subsample,
    representation: EnergyRepresentation,
    layers: tuple[int, ...],
    aggregate: str,
    backbone_kind: str,
    out_path: Path,
    batch_size: int = 64,
    dtype: torch.dtype = torch.float32,
    random_seed: int = 20260102,
    torch_threads: int = 16,
    rebuild: bool = False,
    log: Any = None,
) -> FeatureCache:
    """Extract frozen features for every split of the subsample, with disk caching.

    Args:
        checkpoint: The verified base checkpoint.
        index: The Phase 5 sequence index.
        values: ``[n_series, n_steps]`` per-unit values.
        scale_vector: ``[n_series]`` rated kW per series.
        subsample: The rows to extract.
        representation: The mapping in force.
        layers: Hidden-state indices to keep.
        aggregate: ``"last"`` or ``"mean"``.
        backbone_kind: ``"pretrained"`` or ``"random"``. The random arm has the same
            architecture and is initialised from ``random_seed``; it is the control
            that decides whether the pretrained weights contribute anything.
        out_path: Where the cache is written.
        batch_size: Backbone forward batch size.
        dtype: Compute dtype.
        random_seed: Seed for the control backbone.
        torch_threads: Torch CPU threads.
        rebuild: Recompute even if a cache exists.
        log: Optional callable for progress lines.

    Returns:
        The populated cache.
    """
    channels = index.config.channel_count
    version = _cache_key(
        subsample_version=subsample.version(),
        representation=representation,
        in_channels=channels,
        backbone=backbone_kind,
        layers=layers,
        aggregate=aggregate,
    )

    if out_path.is_file() and not rebuild:
        payload = np.load(out_path, allow_pickle=False)
        cached = FeatureCache(
            version=str(payload["version"]),
            layers=tuple(int(v) for v in payload["layers"]),
            features={
                int(layer): payload[f"layer_{int(layer)}"]
                for layer in payload["layers"]
            },
            targets=payload["targets"],
            persistence=payload["persistence"],
            origins=payload["origins"],
            series=payload["series"],
            row_scale=payload["row_scale"],
        )
        if log is not None:
            log(f"feature cache hit {version} <- {out_path}")
        return cached

    torch.set_num_threads(torch_threads)
    backbone = _load_backbone(
        checkpoint, backbone_kind=backbone_kind, random_seed=random_seed, dtype=dtype
    )
    projector = EnergyPatchProjector(
        patch_size=representation.patch_size,
        in_channels=channels,
        hidden_size=representation.hidden_size,
        layer_norm_input=representation.layer_norm_input,
        seed=representation.projector_seed,
        dtype=dtype,
    )

    all_rows = np.concatenate(
        [subsample.train, subsample.validation, subsample.test]
    )
    features: dict[int, np.ndarray] = {layer: [] for layer in layers}
    targets: list[np.ndarray] = []
    persistence: list[np.ndarray] = []
    origins: list[np.ndarray] = []
    series: list[np.ndarray] = []

    # Extracted in blocks so memory stays flat: the cache is far larger than RAM.
    block = max(batch_size, 512)
    for start in range(0, all_rows.size, block):
        rows = all_rows[start : start + block]
        x, y, baseline = gather_sequences(values, index, rows)
        # gather_sequences returns [B, channels, tokens]; patches need
        # [B, tokens, channels * patch].
        windows = np.ascontiguousarray(np.transpose(x, (0, 2, 1)))
        patches, token_count = build_patches(windows, patch_size=representation.patch_size)
        expected = representation.token_count(index.config.token_count)
        if token_count != expected:
            raise ValueError(
                f"patching produced {token_count} tokens, expected {expected}"
            )
        extracted = extract_backbone_features(
            backbone,
            patches,
            projector=projector,
            layers=layers,
            aggregate=aggregate,
            batch_size=batch_size,
            dtype=dtype,
        )
        for layer in layers:
            features[layer].append(extracted[layer])
        targets.append(y)
        persistence.append(baseline)
        origins.append(index.origins[rows])
        series.append(index.series[rows])
        if log is not None:
            done = start + rows.size
            log(
                f"  [{backbone_kind}] {done}/{all_rows.size} rows "
                f"({done * 0.236 / 60:.1f} min at the measured rate)"
            )

    merged = {
        layer: np.concatenate(blocks, axis=0) for layer, blocks in features.items()
    }
    rows_scale = scale_vector[index.series[all_rows]]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "version": np.array(version),
        "layers": np.array(layers, dtype=np.int64),
        "targets": np.concatenate(targets, axis=0),
        "persistence": np.concatenate(persistence, axis=0),
        "origins": np.concatenate(origins, axis=0),
        "series": np.concatenate(series, axis=0),
        "row_scale": rows_scale,
    }
    for layer, block_values in merged.items():
        payload[f"layer_{layer}"] = block_values
    np.savez(out_path, **payload)

    del backbone
    return FeatureCache(
        version=version,
        layers=tuple(layers),
        features=merged,
        targets=payload["targets"],
        persistence=payload["persistence"],
        origins=payload["origins"],
        series=payload["series"],
        row_scale=rows_scale,
    )


def _load_backbone(
    checkpoint: Qwen3Checkpoint,
    *,
    backbone_kind: str,
    random_seed: int,
    dtype: torch.dtype,
):
    """Load the pretrained trunk, or build a randomly initialised control.

    The control is built with the same configuration and therefore the same shapes and
    the same parameter count, then initialised from ``random_seed``. It is not a
    truncated or smaller model: a size difference would confound "random versus
    pretrained" with "small versus large".
    """
    if backbone_kind == "pretrained":
        return checkpoint.load_backbone(dtype="float32" if dtype == torch.float32 else "bfloat16")
    if backbone_kind != "random":
        raise ValueError(f"unknown backbone kind {backbone_kind!r}")

    from transformers import AutoConfig, AutoModel

    config = AutoConfig.from_pretrained(checkpoint.root)
    torch.manual_seed(random_seed)
    model = AutoModel.from_config(config)
    model = model.to(dtype)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model
