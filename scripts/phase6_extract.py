"""Validate the whole Phase 6 extraction path on a tiny slice, then extract for real.

Run as:  uv run python scripts/phase6_extract.py [tiny|full] [arm ...]

Arms: pretrained, random, nocyclic
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

from energy_intelligence.config.qwen import load_qwen_config
from energy_intelligence.data.smartds import SmartDsLayout
from energy_intelligence.ml.config_ml_bridge import dataset_config_from
from energy_intelligence.ml.phase5 import load_series_for_experiment
from energy_intelligence.ml.qwen.checkpoint import Qwen3Checkpoint
from energy_intelligence.ml.qwen.features import extract_and_cache, select_subsample
from energy_intelligence.ml.qwen.representation import EnergyRepresentation
from energy_intelligence.ml.sequence import build_sequence_index

LOG = Path("artifacts/phase6/extraction.log")


def log(message: str) -> None:
    stamp = time.strftime("%H:%M:%S")
    line = f"[{stamp}] {message}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else "tiny"
    arms = sys.argv[2:] or ["pretrained"]

    cfg = load_qwen_config()
    data = dataset_config_from()
    layout = SmartDsLayout(
        data.raw_root, data.version, data.year, data.region,
        data.subregion, data.scenario, data.substation, data.feeder,
    )
    series = load_series_for_experiment(layout, cfg)
    values = series.normalised().astype(np.float32)

    if mode == "tiny":
        train_rows, validation_rows, test_rows = 128, 128, 128
    else:
        train_rows = cfg.train_rows
        validation_rows = cfg.validation_rows
        test_rows = cfg.test_rows

    checkpoint = Qwen3Checkpoint(Path(cfg.checkpoint_root))
    checkpoint.verify()
    log(f"checkpoint verified: {checkpoint.facts.model_id}@{checkpoint.facts.revision[:12]}")

    arm_specs = {
        "pretrained": ("pretrained", cfg.cyclic_channels, cfg.patch_size, cfg.aggregate, "probe"),
        "random": ("random", cfg.cyclic_channels, cfg.patch_size, cfg.aggregate, "random"),
        "nocyclic": ("pretrained", False, cfg.patch_size, cfg.aggregate, "nocyclic"),
    }

    for arm in arms:
        kind, cyclic, patch_size, aggregate, tag = arm_specs[arm]
        sequence_config = cfg.sequence_config()
        object.__setattr__(sequence_config, "cyclic_channels", cyclic)
        index = build_sequence_index(
            series_count=len(series.series_ids),
            total_steps=values.shape[1],
            config=sequence_config,
            target_id=series.spec.target_id,
            value_origin=series.value_origin,
            series_ids=series.series_ids,
            source=series.provenance.source.locator,
        )
        subsample = select_subsample(
            index,
            train_rows=train_rows,
            validation_rows=validation_rows,
            test_rows=test_rows,
        )
        representation = EnergyRepresentation(
            patch_size=patch_size, hidden_size=2048, aggregate=aggregate
        )
        out = Path(f"artifacts/phase6/features/{tag}.npz")
        started = time.perf_counter()
        log(f"=== arm {arm}: backbone={kind} cyclic={cyclic} patch={patch_size} "
            f"tokens={representation.token_count(sequence_config.token_count)} "
            f"rows={subsample.train.size + subsample.validation.size + subsample.test.size} ===")
        cache = extract_and_cache(
            checkpoint=checkpoint,
            index=index,
            values=values,
            scale_vector=series.scale_vector,
            subsample=subsample,
            representation=representation,
            layers=tuple(cfg.feature_layers),
            aggregate=aggregate,
            backbone_kind=kind,
            out_path=out,
            batch_size=cfg.forward_batch_size,
            dtype=torch.float32,
            random_seed=cfg.random_backbone_seed,
            torch_threads=cfg.torch_threads,
            log=log,
        )
        elapsed = time.perf_counter() - started
        total = subsample.train.size + subsample.validation.size + subsample.test.size
        log(f"arm {arm} done in {elapsed:.0f}s ({elapsed / total:.3f}s/sample); "
            f"cache {cache.version} layers {cache.layers} "
            f"feature shape {cache.features[cfg.feature_layer].shape}")
        for layer in cache.layers:
            block = cache.features[layer]
            log(f"    layer {layer:>2}: shape {block.shape} "
                f"mean {block.mean():+.4f} std {block.std():.4f} "
                f"nan {int(np.isnan(block).sum())} inf {int(np.isinf(block).sum())}")
        log(f"    written {out} ({out.stat().st_size / 2**20:.0f} MiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
