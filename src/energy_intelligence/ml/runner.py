"""Phase 4 orchestration: from the acquired dataset to measured results.

One call to :func:`run_phase4` produces everything the phase is required to leave
behind:

.. code-block:: text

    artifacts/ml/<target>/dataset/<version>.npz + .json   the ML dataset + manifest
    artifacts/ml/<target>/reports/comparison.md          the model table
    artifacts/ml/<target>/reports/comparison.json        the same, machine-readable
    artifacts/ml/<target>/reports/regimes.json           regime prevalence + error
    artifacts/ml/<target>/reports/error_<model>.json     per-model error analysis
    artifacts/ml/<target>/reports/summary.json           headline numbers
    experiments/registry.jsonl                           one record per experiment

Nothing is committed: the dataset arrays, the reports and the checkpoints are all
regenerable from the configuration, and the registry is the single tracked file.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..paths import ProjectPaths
from .analysis import assign_regimes, error_analysis, regime_report
from .dataset import DatasetBuildConfig, build_dataset, load_dataset, save_dataset
from .experiment import ALL_MODELS, ExperimentResult, compare, record_for, run_experiment
from .metrics import mae
from .registry import REGISTRY_FILENAME
from .splits import SPLIT_TEST, SPLIT_VALIDATION, split_summary
from .targets import TargetSeries

__all__ = ["Phase4Output", "run_phase4", "MODELS_DEFAULT"]

#: Baselines evaluated by default: one from each family plus the seasonal reference.
MODELS_DEFAULT: tuple[str, ...] = ALL_MODELS


@dataclass(frozen=True, slots=True)
class Phase4Output:
    """Everything one Phase 4 run produced."""

    target_id: str
    dataset_version: str
    dataset_dir: Path
    report_dir: Path
    registry_path: Path
    results: tuple[ExperimentResult, ...]
    comparison: str
    summary: dict[str, Any]
    regimes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_id": self.target_id,
            "dataset_version": self.dataset_version,
            "dataset_dir": str(self.dataset_dir),
            "report_dir": str(self.report_dir),
            "registry": str(self.registry_path),
            "summary": self.summary,
            "results": [result.to_dict() for result in self.results],
        }


def run_phase4(
    series: TargetSeries,
    *,
    config: DatasetBuildConfig,
    paths: ProjectPaths,
    target_label: str,
    weather: dict[str, np.ndarray] | None = None,
    models: tuple[str, ...] = MODELS_DEFAULT,
    seed: int = 20260101,
    threads: int = 4,
    reuse_dataset: bool = False,
) -> Phase4Output:
    """Build the dataset, run every baseline, analyse and record.

    Args:
        series: The extracted target series, already scaled-aware.
        config: Dataset shape: lookback, horizons, split fractions.
        paths: Project layout, used to place artifacts.
        target_label: Short label for the artifact directory, e.g. ``"load-40"``.
        weather: Real weather observations, when the target needs them.
        models: Which baselines to evaluate.
        seed: Seed for stochastic models.
        threads: Torch CPU threads.
        reuse_dataset: Load an existing dataset version instead of rebuilding.

    Returns:
        The complete run output.

    Raises:
        FileNotFoundError: If ``reuse_dataset`` is set and the version is absent.
    """
    root = paths.artifacts / "ml" / target_label
    dataset_dir = root / "dataset"
    report_dir = root / "reports"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    registry_path = paths.experiments / REGISTRY_FILENAME

    started = time.perf_counter()
    if reuse_dataset:
        from .dataset import dataset_version

        version = dataset_version(config, series.spec.target_id, series.series_ids)
        dataset = load_dataset(dataset_dir, version)
        dataset_sha256 = ""
    else:
        dataset = build_dataset(series, config, weather=weather)
        npz_path, json_path = save_dataset(dataset, dataset_dir)
        dataset_sha256 = json.loads(json_path.read_text(encoding="utf-8"))["arrays"]["sha256"]
        _ = npz_path
    dataset_seconds = time.perf_counter() - started

    results: list[ExperimentResult] = []
    for model in models:
        result = run_experiment(
            dataset, model=model, seed=seed, series=series.values, threads=threads
        )
        results.append(result)

    comparison = compare(results)
    (report_dir / "comparison.md").write_text(
        _comparison_document(dataset, results, comparison, config), encoding="utf-8"
    )
    (report_dir / "comparison.json").write_text(
        json.dumps(
            {
                "dataset_version": dataset.version,
                "target_id": dataset.target_id,
                "target_unit": dataset.target_unit,
                "models": [result.to_dict() for result in results],
                "seed": seed,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    # ---- Regime and error analysis -------------------------------------
    test_mask = dataset.mask(SPLIT_TEST)
    scale = dataset.series_scale.astype(np.float64)[
        dataset.series_index.astype(np.int64)
    ][test_mask]
    horizon = dataset.horizons[0]
    actual_kw = dataset.target_column(horizon).astype(np.float64)[test_mask] * scale

    absolute_errors: dict[str, np.ndarray] = {}
    previous_kw = None
    if horizon >= 1:
        origin_index = dataset.origin_index.astype(np.int64)[test_mask]
        # The value one step before the target step, for a true ramp measurement.
        previous_origin = origin_index + horizon - 1
        series_index = dataset.series_index.astype(np.int64)[test_mask]
        previous_kw = series.values[series_index, previous_origin] / dataset.series_scale[
            series_index
        ].astype(np.float64) * scale

    # Predictions are recomputed once per model and reused for both the regime and
    # the error analysis, so the two analyses and the reported metrics cannot
    # disagree with each other.
    predictions_by_model: dict[str, np.ndarray] = {}
    for result in results:
        predictions = _predictions_for(
            dataset, model=result.model, seed=seed, series=series.values, threads=threads
        ) * scale
        predictions_by_model[result.model] = predictions
        absolute_errors[result.model] = np.abs(actual_kw - predictions)

    axes = assign_regimes(
        actual_kw,
        target_id=dataset.target_id,
        origin_index=dataset.origin_index.astype(np.int64)[test_mask],
        previous_actual=previous_kw,
    )
    regimes = regime_report(axes, absolute_errors)
    (report_dir / "regimes.json").write_text(
        json.dumps(regimes, indent=2, sort_keys=True), encoding="utf-8"
    )

    for result in results:
        report = error_analysis(
            actual=actual_kw,
            predicted=predictions_by_model[result.model],
            model=result.model,
            origin_index=dataset.origin_index.astype(np.int64)[test_mask],
            series_ids=dataset.series_ids,
            series_index=dataset.series_index.astype(np.int64)[test_mask],
            timestep_minutes=15,
            origin_iso=series.origin,
            quality_flags=sorted(flag.value for flag in series.quality.flags),
        )
        (report_dir / f"error_{result.model}.json").write_text(
            json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
        )

    # ---- Registry ------------------------------------------------------
    split_counts = split_summary(dataset.split)
    for result in results:
        record = record_for(
            result,
            dataset,
            model=result.model,
            seed=None if result.model_family == "naive" else seed,
            dataset_sha256=dataset_sha256,
            split_fractions={
                "train": config.train_fraction,
                "validation": config.validation_fraction,
            },
            split_counts={
                "train": split_counts["train"],
                "validation": split_counts["validation"],
                "test": split_counts["test"],
            },
            artifacts={
                "dataset": str(dataset_dir / f"{dataset.version}.npz"),
                "comparison": str(report_dir / "comparison.md"),
                "regimes": str(report_dir / "regimes.json"),
                "errors": str(report_dir / f"error_{result.model}.json"),
            },
            lookback_steps=config.lookback,
            horizon_steps=horizon,
            hyperparams=result.model_detail.get("params", {}),
        )
        from .registry import append_record

        append_record(registry_path, record)

    summary = {
        "target_id": dataset.target_id,
        "target_unit": dataset.target_unit,
        "dataset_version": dataset.version,
        "dataset_seconds": round(dataset_seconds, 2),
        "total_seconds": round(time.perf_counter() - started, 2),
        "sample_count": dataset.sample_count,
        "feature_count": dataset.feature_count,
        "horizons": list(dataset.horizons),
        "lookback_steps": config.lookback,
        "splits": split_summary(dataset.split),
        "seed": seed,
        "models": [result.model for result in results],
        "headline": _headline(results, horizon),
        "balance_residual_note": (
            "the Phase 3 balance residual of 20.4506 kW is a feeder-level constant "
            "at the published peak; it is not a per-customer quantity and therefore "
            "does not enter any customer-level target here. See "
            "docs/ml_data_gap_report.md."
        ),
    }
    (report_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )

    return Phase4Output(
        target_id=dataset.target_id,
        dataset_version=dataset.version,
        dataset_dir=dataset_dir,
        report_dir=report_dir,
        registry_path=registry_path,
        results=tuple(results),
        comparison=comparison,
        summary=summary,
        regimes=regimes,
    )


def _predictions_for(
    dataset: Any,
    *,
    model: str,
    seed: int,
    series: np.ndarray,
    threads: int,
) -> np.ndarray:
    """Predictions at the first horizon, recomputed for analysis.

    Analysis and evaluation must not disagree, so this re-runs the same fitted model
    rather than reading numbers back out of a metric dict. The cost is one extra fit
    per model, which is acceptable for a phase whose point is auditability.
    """
    from .experiment import _family_of
    from .scaling import FeatureScaler

    family = _family_of(model)
    train_mask = dataset.mask(0)
    test_mask = dataset.mask(SPLIT_TEST)
    horizon = dataset.horizons[0]
    target = dataset.target_column(horizon).astype(np.float64)

    if family == "naive":
        from .experiment import _naive_predictions

        normalised = np.asarray(series, dtype=np.float64) / dataset.series_scale.reshape(-1, 1)
        _, predicted = _naive_predictions(dataset, model, test_mask, normalised)
        return predicted[horizon]

    features = dataset.features.astype(np.float64)
    scaler = FeatureScaler.fit(features[train_mask])
    scaled = scaler.transform(features)
    if model == "neural_gru":
        from .experiment import _lookback_window

        x_train, x_test = _lookback_window(dataset, train_mask, test_mask)
    else:
        x_train, x_test = scaled[train_mask], scaled[test_mask]

    if family == "classical":
        from .baselines.classical import fit_classical

        fitted = fit_classical(
            model, x_train, target[train_mask], feature_names=dataset.feature_names, seed=seed
        )
    else:
        from .baselines.neural import fit_neural

        fitted = fit_neural(model, x_train, target[train_mask], seed=seed, threads=threads)
    return fitted.predict(x_test)


def _headline(results: list[ExperimentResult], horizon: int) -> dict[str, Any]:
    """Best MAE per family at the shortest horizon, and who wins overall."""
    by_family: dict[str, float] = {}
    for result in results:
        value = float(result.metrics_by_horizon[horizon]["mae"])
        current = by_family.get(result.model_family)
        if current is None or value < current:
            by_family[result.model_family] = value
    overall = min(
        ((float(r.metrics_by_horizon[horizon]["mae"]), r.model) for r in results),
        default=(float("nan"), ""),
    )
    return {
        "best_mae_by_family_at_h": {family: value for family, value in sorted(by_family.items())},
        "best_model_at_h": overall[1],
        "best_mae_at_h": overall[0],
    }


def _comparison_document(
    dataset: Any,
    results: list[ExperimentResult],
    table: str,
    config: DatasetBuildConfig,
) -> str:
    """The human-readable comparison document."""
    horizon = dataset.horizons[0]
    lines: list[str] = []
    lines.append(f"# Baseline comparison - target `{dataset.target_id}`")
    lines.append("")
    lines.append(f"- dataset version: `{dataset.version}`")
    lines.append(f"- samples: {dataset.sample_count} over {len(dataset.series_ids)} series")
    lines.append(f"- features: {dataset.feature_count}")
    lines.append(f"- lookback: {config.lookback} steps ({config.lookback * 15 // 60} hours)")
    lines.append(f"- horizons: {list(dataset.horizons)} steps (15 minutes each)")
    lines.append(f"- split: chronological {config.train_fraction:.0%}/"
                 f"{config.validation_fraction:.0%}/"
                 f"{1 - config.train_fraction - config.validation_fraction:.0%}")
    lines.append(f"- storage unit: per unit of each series' scale; metrics reported in "
                 f"{dataset.target_unit}")
    lines.append("")
    lines.append("All metrics are on the **test** split, which is the last "
                 f"{int(100 * (1 - config.train_fraction - config.validation_fraction))}% "
                 "of the year and contains the dataset's own peak timepoint.")
    lines.append("")
    lines.append("```")
    lines.append(table)
    lines.append("```")
    lines.append("")
    lines.append("## Interpretation guards")
    lines.append("")
    lines.append("- No composite score is computed. MAE in the target's own unit is the "
                 "headline because that is what a capacity decision is made against.")
    lines.append("- MAPE is deliberately absent: 52.2% of the PV target is exactly zero, "
                 "so a percentage error would be undefined or explosive at night. sMAPE "
                 "is reported instead, with 0/0 defined as 0.")
    lines.append("- `ramp` needs two consecutive horizons. It reads `n/a` where the "
                 "declared horizons are not consecutive.")
    lines.append("- Models are listed in family order. No ranking is implied by the "
                 "row order.")
    lines.append("")
    return "\n".join(lines)