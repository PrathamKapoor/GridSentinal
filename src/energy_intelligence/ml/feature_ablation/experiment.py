"""The Phase 10 driver: build data, run the grid, select, confirm, report.

Sequence, and why it is this order:

1. **Load the protocol freeze.** If it is absent, it is written first and the run stops, so
   the freeze always predates the results it governs. If it is present, its hash is recorded
   into every artifact, so a protocol edited after the fact is detectable.
2. **Build the dataset once per target**, at the frozen horizon, from the same series loader
   and split fractions Phases 4, 5, 7 and 8 used.
3. **Build each feature matrix once**, from the catalogue. Feature sets index columns rather
   than recomputing, which is what guarantees that ``A`` is exactly ``C`` minus the lag
   columns.
4. **Fit once per feature set, score all six folds.**
5. **Select on F01-F04 only**, by mean MAE, with the frozen 0.5% simplicity tie-break, then
   write the selection freeze.
6. **Confirm on F05-F06**, which took no part in selection.

A smoke run is explicitly *not* evidence. It proves the pipeline on a small slice and its
output is labelled ``NON_EVIDENCE_SMOKE`` everywhere it appears.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..dataset import DatasetBuildConfig, build_dataset
from ..features import build_feature_matrix
from .folds import Fold, ablation_folds
from .freeze import (
    FREEZE_PATH,
    SELECTION_FREEZE_PATH,
    build_protocol,
    load_freeze,
    protocol_hash,
    write_protocol_freeze,
    write_selection_freeze,
)
from .runner import FoldResult, primary_model_for, run_feature_set
from .selection import confirm, select, incremental_effects
from .sets import SET_ORDER, checksum, members

__all__ = ["AblationOutcome", "run_ablation", "SMOKE_LABEL"]

#: Marker written into every smoke artifact so it can never be mistaken for a result.
SMOKE_LABEL = "NON_EVIDENCE_SMOKE"

#: Targets with usable SMART-DS data, and why the others are excluded.
TARGETS: tuple[str, ...] = ("customer_load", "pv_generation")
UNSUPPORTED: dict[str, str] = {
    "wind_generation": (
        "UNSUPPORTED: SMART-DS v1.0 contains no wind assets; wind speed appears only as a "
        "weather covariate in the solar file, and metrics.csv reports 0 wind capacity"
    )
}

HORIZON_STEPS = 96  # 96 x 15 minutes = 24 hours
SEED = 20260101


@dataclass(frozen=True, slots=True)
class AblationOutcome:
    """Everything one driver invocation produced.

    Attributes:
        mode: ``smoke`` or ``official``.
        label: Run label, which names the artifact directory.
        protocol_hash: The freeze's content hash.
        results: One entry per measured fold cell.
        selection: Per-target selection, ``None`` before selection runs.
        confirmation: Per-target F05-F06 confirmation, ``None`` before it runs.
        incremental: Per-target incremental effects.
        common_samples: Per-target common-row accounting.
        seconds: Wall time.
        artifacts: Written paths.
        notes: Limitations a reader must not miss.
    """

    mode: str
    label: str
    protocol_hash: str
    results: list[dict[str, Any]]
    selection: dict[str, Any]
    confirmation: dict[str, Any]
    incremental: dict[str, Any]
    common_samples: dict[str, Any]
    seconds: float
    artifacts: dict[str, str]
    notes: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "label": self.label,
            "evidence_class": SMOKE_LABEL if self.mode == "smoke" else "OFFICIAL",
            "protocol_freeze": str(FREEZE_PATH),
            "protocol_hash": self.protocol_hash,
            "selection_freeze": str(SELECTION_FREEZE_PATH),
            "results": self.results,
            "selection": self.selection,
            "confirmation": self.confirmation,
            "incremental_effects": self.incremental,
            "common_samples": self.common_samples,
            "seconds": round(self.seconds, 2),
            "artifacts": self.artifacts,
            "notes": list(self.notes),
        }


def _build_target_data(
    root: Path, target: str, *, smoke: bool, log
) -> tuple[Any, np.ndarray, tuple[str, ...]]:
    """Build the dataset and the full feature matrix for one target.

    Args:
        root: Project root, for the raw data location.
        target: Target id.
        smoke: Shrink the series and the fold range.
        log: Progress callable.

    Returns:
        ``(dataset, feature_matrix, feature_names)``.
    """
    from ...config.temporal import TemporalExperimentConfig
    from ..config_ml_bridge import dataset_config_from
    from ..phase5 import load_series_for_experiment
    from ...data.smartds import SmartDsLayout

    data = dataset_config_from()
    layout = SmartDsLayout(
        data.raw_root, data.version, data.year, data.region,
        data.subregion, data.scenario, data.substation, data.feeder,
    )
    customers = 8 if smoke else int(data.customer_count if hasattr(data, "customer_count") else 40)
    temporal = TemporalExperimentConfig(
        target=target,
        label=f"phase10-{'smoke' if smoke else 'main'}",
        customer_count=customers,
        commercial_fraction=0.25,
        seed=SEED,
        torch_threads=8,
        lookback_steps=672,
        token_stride=4,
        train_origin_stride=1,
        train_fraction=0.70,
        validation_fraction=0.15,
        horizons=(HORIZON_STEPS,),
        architecture="tcn",
        cyclic_channels=True,
        delta_target=True,
        model_params={},
        epochs=1,
        batch_size=512,
        learning_rate=0.003,
        weight_decay=0.0,
        loss="l1",
        huber_delta=0.05,
        patience=3,
        min_delta=1e-5,
        max_seconds=3600.0,
        grad_clip=1.0,
        validation_stride=1,
        ablation_budget_epochs=1,
        ablation_budget_stride=4,
        ablation_validation_stride=4,
    )
    log(f"  {target}: loading series ({customers} customers)")
    series = load_series_for_experiment(layout, temporal)
    cfg = DatasetBuildConfig(
        lookback=672,
        horizons=(HORIZON_STEPS,),
        train_fraction=0.70,
        validation_fraction=0.15,
        origin_stride=96 if smoke else 1,
        include_weather=False,
    )
    dataset = build_dataset(series, cfg, verify_no_leakage=not smoke)
    names = tuple(dataset.feature_names)
    log(f"  {target}: {dataset.sample_count} samples, {len(names)} catalogue features")
    # The full matrix is built once and every feature set indexes into it, so a set cannot
    # accidentally get a different value for a shared feature.
    matrix = build_feature_matrix(
        values=np.asarray(series.normalised(), dtype=np.float32),
        origins=np.asarray(dataset.origin_index, dtype=np.int64),
        lookback=672,
        feature_names=names,
    )
    return dataset, matrix, names


def _fold_definitions(dataset, *, smoke: bool) -> tuple[tuple[Fold, ...], int]:
    """Build F01-F06 from the dataset's own validation origin range."""
    origins = np.asarray(dataset.origin_index, dtype=np.int64)
    split = np.asarray(dataset.split)
    validation_origins = np.unique(origins[split == 1])
    if validation_origins.size == 0:
        raise ValueError("the validation split has no origins; the dataset build is wrong")
    start, stop = int(validation_origins.min()), int(validation_origins.max()) + 1
    folds = ablation_folds(start, stop)
    if smoke:
        # One fold only, and only the first: enough to prove the pipeline end to end.
        folds = folds[:1]
    return folds, stop


def run_ablation(
    *,
    root: Path,
    targets: tuple[str, ...] = TARGETS,
    sets: tuple[str, ...] = SET_ORDER,
    folds_filter: tuple[str, ...] | None = None,
    smoke: bool = False,
    label: str | None = None,
    write_freeze: bool = True,
    log=print,
) -> AblationOutcome:
    """Run the ablation grid, select on F01-F04, confirm on F05-F06.

    Args:
        root: Project root.
        targets: Targets to run. Anything in :data:`UNSUPPORTED` is refused with its reason.
        sets: Feature set ids.
        folds_filter: Restrict to these fold ids, for a partial run.
        smoke: Run the non-evidence smoke path.
        label: Artifact directory name.
        write_freeze: Create the protocol freeze if it is absent.
        log: Progress callable.

    Returns:
        The measured outcome.

    Raises:
        FileExistsError: If the freeze exists and would need overwriting.
        KeyError: If a target is not runnable.
    """
    started = time.perf_counter()
    label = label or ("smoke" if smoke else "main")
    artifact_dir = Path(root) / "artifacts" / "phase10" / label
    artifact_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. The protocol, frozen before any measurement -------------------
    protocol_file = Path(root) / FREEZE_PATH
    runnable: list[str] = []
    for target in targets:
        if target in UNSUPPORTED:
            log(f"SKIP {target}: {UNSUPPORTED[target]}")
            continue
        runnable.append(target)
    if not runnable:
        raise ValueError(
            f"none of the requested targets are runnable: "
            f"{ {t: UNSUPPORTED[t] for t in targets} }"
        )
    primary = {t: primary_model_for(t) for t in runnable}
    protocol = build_protocol(
        targets=tuple(runnable),
        unsupported_targets={k: v for k, v in UNSUPPORTED.items()},
        horizon_steps=HORIZON_STEPS,
        primary_model_by_target=primary,
        primary_model_evidence={
            "customer_load": "Phase 5 registry: classical_hist_gbm won load at h=96",
            "pv_generation": "Phase 4/5 registry: classical_ridge won the PV horizon task",
        },
        model_config={"model": primary, "seed": SEED, "horizon_steps": HORIZON_STEPS},
        secondary_model_config={"model": "neural_mlp", "seed": SEED, "frozen": True},
        seed=SEED,
    )
    if protocol_file.exists():
        frozen = load_freeze(protocol_file)
        digest = str(frozen.get("protocol_hash") or "")
        if digest:
            log(f"protocol freeze found: {FREEZE_PATH}")
        phash = digest or protocol_hash(protocol)
    else:
        if not write_freeze:
            raise FileNotFoundError(
                f"{FREEZE_PATH} does not exist and write_freeze is False; Phase 10 requires "
                f"a protocol freeze before official execution"
            )
        path, phash = write_protocol_freeze(protocol, root)
        log(f"protocol freeze written: {path}")
    log(f"protocol hash {phash[:16]}")

    results: list[dict[str, Any]] = []
    selection: dict[str, Any] = {}
    confirmation: dict[str, Any] = {}
    incremental: dict[str, Any] = {}
    common_report: dict[str, Any] = {}
    selection_freeze_payload: dict[str, Any] = {}

    for target in runnable:
        model = primary[target]
        log(f"target {target} with {model}")
        dataset, matrix, names = _build_target_data(root, target, smoke=smoke, log=log)
        folds, validation_stop = _fold_definitions(dataset, smoke=smoke)
        if folds_filter:
            wanted = set(folds_filter)
            folds = tuple(f for f in folds if f.fold_id in wanted)
        column = {name: index for index, name in enumerate(names)}
        for set_id in sets:
            wanted_names = members(set_id)
            missing = [n for n in wanted_names if n not in column]
            if missing:
                raise KeyError(
                    f"feature set {set_id} names {missing}, which the dataset's catalogue "
                    f"does not contain; Phase 10 must not invent features"
                )
            indices = [column[n] for n in wanted_names]
            log(f"  set {set_id} ({len(indices)} features) fitting {model}")
            cells = run_feature_set(
                dataset=dataset,
                feature_matrix=matrix[:, indices],
                feature_names=wanted_names,
                target=target,
                feature_set=set_id,
                folds=folds,
                model=model,
                seed=SEED,
                validation_stop=validation_stop,
                horizon_steps=HORIZON_STEPS,
                keep_timestamps=True,
            )
            results.extend(cell.to_dict(keep_timestamps=True) for cell in cells)
            log(
                "    "
                + "  ".join(
                    f"{c.fold_id} mae={c.metrics['mae']:.5f}" for c in cells
                )
            )
            common_report.setdefault(target, {})[set_id] = {
                "per_fold": {
                    c.fold_id: c.account.to_dict() for c in cells
                }
            }

        target_results = [
            r for r in results if r["target"] == target and r["feature_set"] in sets
        ]
        if not smoke:
            chosen = select(target_results, fold_ids=("F01", "F02", "F03", "F04"))
            chosen = dict(chosen)
            chosen["model"] = model
            chosen["horizon_label"] = protocol.horizon_label
            chosen["horizon_steps"] = protocol.horizon_steps
            selection[target] = chosen
            incremental[target] = incremental_effects(target_results)
            confirmation[target] = confirm(
                target_results, selected_set=chosen["selected_feature_set"]
            )
            selection_freeze_payload[target] = {
                "target": target,
                "model": model,
                "selected_feature_set": chosen["selected_feature_set"],
                "selected_features": list(members(chosen["selected_feature_set"])),
                "feature_set_checksum": checksum(chosen["selected_feature_set"]),
                "fold_mae_f01_f04": chosen["per_fold_mae"],
                "mean_mae_f01_f04": chosen["selected_set_mean_mae"],
                "rationale": chosen["rationale"],
                "tie_break_threshold_relative": chosen["tie_break"]["relative_mae_tolerance"],
                "selection_metric": "mae",
                "selection_folds": ["F01", "F02", "F03", "F04"],
                "confirmation_folds_excluded_from_selection": ["F05", "F06"],
                "final_test_access": "LOCKED - not read by any Phase 10 code path",
                "protocol_hash": phash,
            }

    written: dict[str, str] = {}
    if not smoke and selection_freeze_payload:
        payload = {
            "freeze_kind": "phase10_selected_feature",
            "targets": selection_freeze_payload,
            "note": (
                "written after F01-F04 selection and before F05-F06 confirmation. F05-F06 "
                "took no part in the choice recorded here and are not unseen test data - "
                "they are held-out confirmation folds inside the validation range."
            ),
        }
        path = write_selection_freeze(payload, root, overwrite=True)
        written["selection_freeze"] = str(path)
        log(f"selection freeze written: {path}")

    outcome = AblationOutcome(
        mode="smoke" if smoke else "official",
        label=label,
        protocol_hash=phash,
        results=results,
        selection=selection,
        confirmation=confirmation,
        incremental=incremental,
        common_samples=common_report,
        seconds=time.perf_counter() - started,
        artifacts=written,
        notes=(
            *((SMOKE_LABEL,) if smoke else ()),
            "single seed: a difference smaller than seed-to-seed variation cannot be "
            "separated from it, and Phase 10 runs no formal significance testing",
            "feature effects are incremental validation-performance differences, not causal "
            "effects",
            "the final test split was not read by any run in this phase",
            "wind_generation is excluded: SMART-DS v1.0 contains no wind assets",
        ),
    )
    result_path = artifact_dir / ("smoke.json" if smoke else "result.json")
    result_path.write_text(
        json.dumps(outcome.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
    )
    written["result"] = str(result_path)
    log(f"result written: {result_path}  ({outcome.seconds:.1f}s)")
    return outcome