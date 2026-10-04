"""Build the Phase 7 expert pool and cache full-split predictions.

Usage:  uv run python scripts/phase7_experts.py

Runs every expert over the full validation and test splits - 179,520 rows each, the
same populations Phase 4 and Phase 5 reported - and caches the result so the oracle,
diversity and router experiments all consume one identical set of forecasts.

Router features are deliberately **not** built here. Half of them (the lagged realised
expert errors) are functions of these forecasts, and building them after this script
means one expensive expert pass feeds every downstream analysis.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from energy_intelligence.config.temporal import load_temporal_config
from energy_intelligence.data.smartds import SmartDsLayout
from energy_intelligence.ml.config_ml_bridge import dataset_config_from
from energy_intelligence.ml.phase5 import load_series_for_experiment
from energy_intelligence.ml.router.experts import build_experts, build_panel
from energy_intelligence.ml.sequence import build_sequence_index
from energy_intelligence.paths import ProjectPaths

LOG = Path("artifacts/phase7/experts.log")
OUT = Path("artifacts/phase7/experts.npz")


def log(message: str) -> None:
    stamp = time.strftime("%H:%M:%S")
    line = f"[{stamp}] {message}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def main() -> int:
    cfg = load_temporal_config()
    paths = ProjectPaths.from_root()
    torch.set_num_threads(cfg.torch_threads)

    data = dataset_config_from()
    layout = SmartDsLayout(
        data.raw_root, data.version, data.year, data.region,
        data.subregion, data.scenario, data.substation, data.feeder,
    )
    series = load_series_for_experiment(layout, cfg)
    values = series.normalised().astype(np.float32)

    def make_index(stride: int):
        sequence_config = cfg.sequence_config(train_origin_stride=stride)
        return build_sequence_index(
            series_count=len(series.series_ids),
            total_steps=values.shape[1],
            config=sequence_config,
            target_id=series.spec.target_id,
            value_origin=series.value_origin,
            series_ids=series.series_ids,
            source=series.provenance.source.locator,
        )

    index = make_index(cfg.train_origin_stride)
    index_stride1 = make_index(1)
    log(f"index {index.version}: panel candidates "
        f"validation {index.rows(1).size}, test {index.rows(2).size}")

    panel = build_panel(index=index, values=values, scale_vector=series.scale_vector)
    log(f"panel: {json.dumps(panel.describe())}")
    assert panel.split_slices["test"].stop - panel.split_slices["test"].start == 179520, (
        "the Phase 7 panel must match Phase 4/5's 179,520 test rows exactly"
    )

    pool, provenance = build_experts(
        panel=panel,
        index=index,
        index_stride1=index_stride1,
        values=values,
        scale_vector=series.scale_vector,
        tcn_checkpoint=paths.artifacts / "phase5" / "tcn-main" / "checkpoint.pt",
        seed=cfg.seed,
        log=log,
    )

    positions = np.arange(panel.n_rows)
    log("forecasting the full panel:")
    block, timings = pool.forecast_block(positions, log=log)
    log(f"expert timings: {timings}")

    for position, name in enumerate(pool.names):
        mae = np.abs(block[position] - panel.actual_kw).mean(axis=0)
        log(f"  {name:22s} MAE " + " ".join(f"h{h}={m:.4f}" for h, m in zip(panel.horizons, mae)))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT,
        forecasts=block,
        actual_kw=panel.actual_kw,
        persistence_kw=panel.persistence_kw,
        row_scale=panel.row_scale,
        origins=panel.origins,
        series=panel.series,
        index_rows=panel.index_rows,
        horizons=np.array(panel.horizons),
        expert_names=np.array(pool.names),
        split_slices=np.array(
            [[panel.split_slices["validation"].start, panel.split_slices["validation"].stop],
             [panel.split_slices["test"].start, panel.split_slices["test"].stop]],
            dtype=np.int64,
        ),
        split_names=np.array(["validation", "test"]),
    )
    log(f"written {OUT} ({OUT.stat().st_size / 2**20:.0f} MiB)")

    meta = {
        "index_version": index.version,
        "panel": panel.describe(),
        "experts": provenance,
        "expert_timings_seconds": timings,
        "expert_mae_kw": {
            name: [float(v) for v in np.abs(block[i] - panel.actual_kw).mean(axis=0)]
            for i, name in enumerate(pool.names)
        },
    }
    (OUT.parent / "experts.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8"
    )
    log(f"written {OUT.parent / 'experts.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
