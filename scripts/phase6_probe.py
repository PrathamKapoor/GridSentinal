"""Run the Phase 6 probe experiments over the cached features.

Usage:  uv run python scripts/phase6_probe.py [arm ...]

Arms: probe (main), random, nocyclic, layer_16, probe_depth

Features are read from artifacts/phase6/features/<arm>.npz, which
scripts/phase6_extract.py produced. Nothing here re-runs the backbone, so the whole
suite costs minutes rather than hours.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

from energy_intelligence.config.qwen import load_qwen_config
from energy_intelligence.data.smartds import SmartDsLayout
from energy_intelligence.ml.config_ml_bridge import dataset_config_from
from energy_intelligence.ml.dataset import build_feature_matrix
from energy_intelligence.ml.phase5 import load_phase4_reference, load_series_for_experiment
from energy_intelligence.ml.qwen.baselines import write_json
from energy_intelligence.ml.qwen.checkpoint import Qwen3Checkpoint
from energy_intelligence.ml.qwen.experiment import run_phase6_experiment
from energy_intelligence.ml.qwen.features import FeatureCache, select_subsample
from energy_intelligence.ml.qwen.probe import ProbeTrainingConfig
from energy_intelligence.ml.qwen.representation import (
    EnergyRepresentation,
    representation_version,
)
from energy_intelligence.ml.sequence import build_sequence_index
from energy_intelligence.paths import ProjectPaths

LOG = Path("artifacts/phase6/probe.log")

ARM_SPECS = {
    # arm -> (feature file tag, cyclic, backbone kind, label, feature_layer, probe_layers)
    "probe": ("probe", True, "pretrained", "qwen-probe", None, None),
    "random": ("random", True, "random", "qwen-random", None, None),
    "nocyclic": ("nocyclic", False, "pretrained", "qwen-nocyclic", None, None),
    "layer_16": ("probe", True, "pretrained", "qwen-layer16", 16, None),
    "probe_depth": ("probe", True, "pretrained", "qwen-depth", None, (0, 128)),
}


def log(message: str) -> None:
    stamp = time.strftime("%H:%M:%S")
    line = f"[{stamp}] {message}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def load_cache(path: Path) -> FeatureCache:
    payload = np.load(path, allow_pickle=False)
    layers = tuple(int(v) for v in payload["layers"])
    return FeatureCache(
        version=str(payload["version"]),
        layers=layers,
        features={layer: payload[f"layer_{layer}"] for layer in layers},
        targets=payload["targets"],
        persistence=payload["persistence"],
        origins=payload["origins"],
        series=payload["series"],
        row_scale=payload["row_scale"],
    )


def main() -> int:
    arms = sys.argv[1:] or ["probe"]
    cfg = load_qwen_config()
    paths = ProjectPaths.from_root()

    data = dataset_config_from()
    layout = SmartDsLayout(
        data.raw_root, data.version, data.year, data.region,
        data.subregion, data.scenario, data.substation, data.feeder,
    )
    series = load_series_for_experiment(layout, cfg)
    values = series.normalised().astype(np.float32)

    phase4_dir = paths.artifacts / "ml" / "load-40" / "reports"
    phase4_reference = load_phase4_reference(phase4_dir / "comparison.json")
    phase5_reference = {
        int(h): v
        for h, v in json.loads(
            (paths.artifacts / "phase5" / "tcn-main" / "result.json").read_text(
                encoding="utf-8"
            )
        )["metrics_by_horizon"].items()
    }
    gbm_reference = {
        horizon: {"mae": phase4_reference["classical_hist_gbm"][horizon]["mae"]}
        for horizon in cfg.horizons
    }

    feature_names = (
        "hour_of_day", "day_of_week", "day_of_year", "is_weekend", "value_at_origin",
        "lag_1", "lag_4", "lag_96", "lag_672", "roll_mean_96", "roll_std_96",
        "ramp_1", "roll_mean_same_hour_7d",
    )

    scaler_holder: dict[str, object] = {}
    results: dict[str, object] = {}

    for arm in arms:
        tag, cyclic, backbone_kind, label, layer_override, probe_layers_override = ARM_SPECS[arm]
        feature_path = paths.artifacts / "phase6" / "features" / f"{tag}.npz"
        if not feature_path.is_file():
            log(f"!! arm {arm}: no feature cache at {feature_path}; skipping")
            continue

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
            train_rows=cfg.train_rows,
            validation_rows=cfg.validation_rows,
            test_rows=cfg.test_rows,
        )
        cache = load_cache(feature_path)
        representation = EnergyRepresentation(
            patch_size=cfg.patch_size, hidden_size=2048, aggregate=cfg.aggregate
        )
        rep_version = representation_version(
            representation, in_channels=index.config.channel_count
        )

        # Phase 4 features for the same rows, so the GBM can be refitted on them.
        train_matrix = build_feature_matrix(
            values,
            index.origins[subsample.train],
            lookback=index.config.lookback_steps,
            feature_names=feature_names,
            series_index=index.series[subsample.train],
        )

        # The Phase 5 window scaler, needed by the TCN baseline. Built once.
        if "scaler" not in scaler_holder:
            from energy_intelligence.ml.phase5 import _window_scaler

            scaler_holder["scaler"] = _window_scaler(values, index, index.rows(0))

        layer = layer_override or cfg.feature_layer
        probe_layers = probe_layers_override or cfg.probe_layers
        training = ProbeTrainingConfig(
            epochs=cfg.epochs,
            batch_size=cfg.batch_size,
            learning_rate=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
            patience=cfg.patience,
            min_delta=cfg.min_delta,
            max_seconds=cfg.max_seconds,
            torch_threads=cfg.torch_threads,
            seed=cfg.seed,
        )

        log(f"=== arm {arm}: tag={tag} backbone={backbone_kind} cyclic={cyclic} "
            f"layer={layer} probe_layers={probe_layers} label={label} ===")
        result = run_phase6_experiment(
            experiment_id=f"phase6-{label}",
            index=index,
            values=values,
            scale_vector=series.scale_vector,
            series_ids=series.series_ids,
            series_origin_iso=series.origin,
            cache=cache,
            subsample=subsample,
            feature_layer=layer,
            representation_version=rep_version,
            horizons=cfg.horizons,
            probe_layers=probe_layers,
            basis=cfg.basis,
            basis_hidden_size=cfg.basis_hidden_size,
            series_embedding_dim=cfg.series_embedding_dim,
            training=training,
            phase4_reference=gbm_reference,
            phase5_reference=phase5_reference,
            phase4_features=(train_matrix, feature_names),
            scaler=scaler_holder["scaler"],
            phase5_checkpoint=paths.artifacts / "phase5" / "tcn-main" / "checkpoint.pt",
            artifact_root=paths.artifacts / "phase6" / label,
            backbone_kind=backbone_kind,
            log=log,
        )
        results[arm] = result.to_dict()
        log(f"arm {arm} verdict: {result.verdict}")

    write_json(paths.artifacts / "phase6" / "experiments.json", results)
    log(f"written artifacts/phase6/experiments.json for arms {sorted(results)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
