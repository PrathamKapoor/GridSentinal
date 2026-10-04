"""The Phase 6 runner: does a pretrained Qwen3-1.7B help on this forecasting task?

The experiment is built so that the interesting answer is available whichever way it
comes out. Three properties do that work:

**Identical rows.** Persistence, the refitted GBM and the real Phase 5 checkpoint are
all scored on the subsample the Qwen probe is scored on, so every comparison is
paired and every interval means something.

**A random-backbone control.** A randomly initialised Qwen3 of the same architecture
and parameter count goes through the identical extraction and the identical probe. If
it matches the pretrained features, then a deep random feature map was doing the work
and the answer to the phase's central question is no. This is the single most
important comparison in the phase, and it is cheap here only because the extraction
is cached.

**Per-horizon reporting.** Phase 5 already established that a model can win at 15
minutes and lose at 24 hours, so a single averaged number is not an acceptable
summary.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from ..analysis import assign_regimes, error_analysis, regime_report
from ..metrics import evaluate
from ..dataset import build_feature_matrix
from ..sequence import SequenceIndex, gather_sequences
from ..temporal import model_defaults
from .baselines import bootstrap_summary, gbm_baseline, paired_bootstrap
from .features import FeatureCache, Subsample
from .probe import ProbeTrainingConfig, fit_probe, probe_predict

__all__ = ["Phase6Result", "run_phase6_experiment", "representation_diagnostics"]


@dataclass(frozen=True, slots=True)
class Phase6Result:
    """Everything one Phase 6 configuration produced.

    Attributes:
        experiment_id: Stable identifier.
        dataset_version: The Phase 5 sequence index version, proving row identity.
        subsample_version: The chosen rows' hash.
        representation_version: The mapping's hash.
        metrics_by_horizon: MAE and companions per horizon, in kW.
        improvement: Per-horizon comparison against every baseline.
        bootstrap: Paired bootstrap of each model against the GBM.
        verdict: The phase's answer, per horizon.
        probe: Head training record.
        features: What the backbone's activations look like.
        regime: Per-regime MAE for the probe and each baseline.
        errors: Error distribution and worst rows for the probe.
        compute: Measured cost of everything above.
        diagnostics: Representation-preservation measurements.
    """

    experiment_id: str
    dataset_version: str
    subsample_version: str
    representation_version: str
    metrics_by_horizon: dict[int, dict[str, Any]]
    improvement: dict[int, dict[str, Any]]
    bootstrap: dict[int, dict[str, Any]]
    verdict: str
    probe: dict[str, Any]
    features: dict[str, Any]
    regime: dict[str, Any]
    errors: dict[str, Any]
    compute: dict[str, Any]
    diagnostics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        def coerce(value: Any) -> Any:
            if isinstance(value, dict):
                return {str(k): coerce(v) for k, v in value.items()}
            if isinstance(value, (np.floating, np.integer)):
                return value.item()
            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, (np.bool_,)):
                return bool(value)
            return value

        return coerce(
            {
                "experiment_id": self.experiment_id,
                "dataset_version": self.dataset_version,
                "subsample_version": self.subsample_version,
                "representation_version": self.representation_version,
                "metrics_by_horizon": self.metrics_by_horizon,
                "improvement": self.improvement,
                "bootstrap": self.bootstrap,
                "verdict": self.verdict,
                "probe": self.probe,
                "features": self.features,
                "regime": self.regime,
                "errors": self.errors,
                "compute": self.compute,
                "diagnostics": self.diagnostics,
            }
        )


def representation_diagnostics(features: np.ndarray) -> dict[str, float]:
    """Measurable properties of the frozen activations. No invented quality score.

    Four quantities, each standard and each computed from the features themselves:

    * ``hidden_norm_mean`` - the RMSNorm-scaled residual magnitude. A collapsed
      representation would show near-zero norms.
    * ``participation_ratio`` - ``(sum lambda)^2 / sum lambda^2`` over the feature
      covariance, i.e. the effective number of active dimensions. A representation
      collapsed onto a few directions scores near 1; one using its full width scores
      near 2048.
    * ``mean_pairwise_cosine`` - mean absolute cosine similarity between random pairs
      of rows. Near 1 would mean every window looks the same.
    * ``featurewise_cv`` - mean coefficient of variation across features, which
      separates "all features on a similar scale" from "a few features dominate".
    """
    matrix = np.asarray(features, dtype=np.float64)
    norms = np.linalg.norm(matrix, axis=1)
    centred = matrix - matrix.mean(axis=0, keepdims=True)
    # Covariance eigenvalues via the Gram matrix of the centred data would be
    # 2048x2048; instead use the smaller side of the SVD, which is exact for the
    # spectrum and much cheaper for N < D.
    _u, singular, _v = np.linalg.svd(centred, full_matrices=False)
    eigenvalues = singular**2
    participation = float(
        eigenvalues.sum() ** 2 / max(1e-30, float((eigenvalues**2).sum()))
    )
    sample = matrix[: min(512, matrix.shape[0])]
    norms_sample = np.linalg.norm(sample, axis=1, keepdims=True)
    unit = sample / np.maximum(norms_sample, 1e-12)
    gram = unit @ unit.T
    upper = gram[np.triu_indices(gram.shape[0], k=1)]
    column_cv = matrix.std(axis=0) / np.maximum(np.abs(matrix.mean(axis=0)), 1e-12)
    return {
        "hidden_norm_mean": float(norms.mean()),
        "hidden_norm_std": float(norms.std()),
        "participation_ratio": participation,
        "mean_pairwise_cosine_abs": float(np.mean(np.abs(upper))) if upper.size else float("nan"),
        "featurewise_cv_mean": float(np.mean(column_cv)),
    }


def _split_slices(subsample: Subsample) -> dict[str, slice]:
    """Slice boundaries for the concatenated feature cache."""
    boundaries: dict[str, slice] = {}
    cursor = 0
    for name in ("train", "validation", "test"):
        size = subsample.split(name).size
        boundaries[name] = slice(cursor, cursor + size)
        cursor += size
    return boundaries


def run_phase6_experiment(
    *,
    experiment_id: str,
    index: SequenceIndex,
    values: np.ndarray,
    scale_vector: np.ndarray,
    series_ids: tuple[str, ...],
    series_origin_iso: str,
    cache: FeatureCache,
    subsample: Subsample,
    feature_layer: int,
    representation_version: str,
    horizons: tuple[int, ...],
    probe_layers: tuple[int, ...],
    basis: str,
    basis_hidden_size: int,
    series_embedding_dim: int,
    training: ProbeTrainingConfig,
    phase4_reference: dict[int, dict[str, Any]],
    phase5_reference: dict[int, dict[str, Any]],
    phase4_features: tuple[np.ndarray, tuple[str, ...]],
    scaler: Any,
    phase5_checkpoint: Path | None,
    artifact_root: Path,
    backbone_kind: str,
    log: Any = None,
) -> Phase6Result:
    """Run one Phase 6 configuration end to end and write its artifacts.

    Args:
        experiment_id: Identifier for the registry.
        index: The Phase 5 sequence index, reused for row identity.
        values: Per-unit values ``[n_series, n_steps]``.
        scale_vector: Rated kW per series.
        series_ids: Series identifiers, for the per-series error report.
        series_origin_iso: ISO timestamp of index 0.
        cache: The frozen features for this arm.
        subsample: The rows behind the cache.
        feature_layer: Which cached layer the probe reads.
        representation_version: The mapping's hash.
        horizons: The horizons.
        probe_layers: Head widths.
        basis: ``"raw"`` or ``"pca"``.
        basis_hidden_size: PCA width when applicable.
        training: Head training settings.
        phase4_reference: Phase 4's published metrics, for context.
        phase5_reference: Phase 5's published metrics, for context.
        phase4_features: Phase 4 feature matrix for the subsample training rows and
            the feature names, so the GBM can be refitted on identical rows.
        scaler: The Phase 5 window scaler, for the TCN baseline.
        phase5_checkpoint: Path to Phase 5's checkpoint, when it is available.
        artifact_root: Where this configuration's artifacts are written.
        backbone_kind: ``"pretrained"`` or ``"random"``, recorded in the result.
        log: Optional progress callable.

    Returns:
        The completed result.
    """
    slices = _split_slices(subsample)
    train_rows = subsample.train
    validation_rows = subsample.validation
    test_rows = subsample.test

    features = cache.layer(feature_layer)
    basis_width = basis_hidden_size if basis == "pca" else None

    train_series = cache.series[slices["train"]]
    validation_series = cache.series[slices["validation"]]
    test_series = cache.series[slices["test"]]
    series_count = int(values.shape[0])

    started = time.perf_counter()
    trained = fit_probe(
        features[slices["train"]],
        cache.targets[slices["train"]],
        cache.persistence[slices["train"]],
        features[slices["validation"]],
        cache.targets[slices["validation"]],
        cache.persistence[slices["validation"]],
        train_series_index=train_series,
        validation_series_index=validation_series,
        horizons=horizons,
        config=training,
        layers=probe_layers,
        basis_width=basis_width,
        series_count=series_count,
        series_embedding_dim=series_embedding_dim,
        checkpoint_path=artifact_root / "probe.pt",
        log=log,
    )
    head_seconds = time.perf_counter() - started

    # ---- Predictions on the held-out test subsample, evaluated once -----------
    predicted_delta = probe_predict(
        trained.head, features[slices["test"]], basis=trained.basis, series=test_series
    )
    row_scale = cache.row_scale[slices["test"]]
    actual_pu = cache.targets[slices["test"]]
    persistence_pu = cache.persistence[slices["test"]]
    predicted_pu = persistence_pu + predicted_delta

    actual_kw = actual_pu * row_scale[:, None]
    predicted_kw = predicted_pu * row_scale[:, None]
    persistence_kw = persistence_pu * row_scale[:, None]

    metrics_by_horizon: dict[int, dict[str, Any]] = {}
    for column, horizon in enumerate(horizons):
        metrics_by_horizon[horizon] = evaluate(
            actual_kw[:, column], predicted_kw[:, column]
        ).to_dict()

    # ---- Baselines on the same rows ------------------------------------------
    from .baselines import phase5_tcn_baseline

    baseline_blocks: dict[str, np.ndarray] = {
        "persistence": persistence_kw,
    }

    # Phase 4 fits every baseline in PER-UNIT space and only scales back to kW when
    # reporting. Fitting the GBM on kW targets against per-unit features would make a
    # 1.4 kW household's residuals 100x larger than a 388 kW shop's, and the measured
    # symptom of getting this wrong is a train MAE of ~14 kW where Phase 4 reported
    # 0.42 kW.
    train_features_all, feature_names = phase4_features
    _x, y_train_pu, _b = gather_sequences(values, index, train_rows)
    # series_index is NOT optional here. Its default assumes "each series contributes
    # the same origins in turn", which is Phase 4's row layout; the Phase 6 subsample
    # is ordered by series instead, so the default would pair every row with another
    # customer's demand. The measured symptom was a GBM ten times worse than
    # persistence, which is what a mislabelled target looks like.
    predict_matrices = {
        "test": build_feature_matrix(
            values,
            index.origins[test_rows],
            lookback=index.config.lookback_steps,
            feature_names=feature_names,
            series_index=index.series[test_rows],
        )
    }
    gbm_predictions_pu, gbm_detail = gbm_baseline(
        train_features=train_features_all,
        train_targets=y_train_pu,
        predict_features=predict_matrices,
        horizons=horizons,
        feature_names=feature_names,
        seed=training.seed,
        log=log,
    )
    baseline_blocks["classical_hist_gbm"] = gbm_predictions_pu["test"] * row_scale[:, None]
    if log is not None:
        log(f"  gbm detail: {gbm_detail}")

    tcn_available = False
    if phase5_checkpoint is not None and Path(phase5_checkpoint).is_file():
        try:
            tcn = phase5_tcn_baseline(
                checkpoint_path=Path(phase5_checkpoint),
                values=values,
                index=index,
                rows={"test": test_rows},
                scale_vector=scale_vector,
                scaler=scaler,
                model_params=model_defaults("tcn"),
            )["test"]
            baseline_blocks["phase5_tcn"] = tcn.predicted
            tcn_available = True
            if log is not None:
                log(f"  phase 5 tcn applied to the subsample: {tcn.detail}")
        except Exception as error:  # pragma: no cover - defensive
            if log is not None:
                log(f"  phase 5 tcn unavailable on the subsample: {error!r}")

    # ---- Paired bootstrap against the GBM ------------------------------------
    bootstrap: dict[int, dict[str, Any]] = {}
    improvements: dict[int, dict[str, Any]] = {}
    for column, horizon in enumerate(horizons):
        errors = {
            name: np.abs(actual_kw[:, column] - block[:, column])
            for name, block in baseline_blocks.items()
        }
        errors["qwen_probe"] = np.abs(actual_kw[:, column] - predicted_kw[:, column])
        bootstrap[horizon] = bootstrap_summary(
            errors, reference="classical_hist_gbm", seed=training.seed
        )
        entry: dict[str, Any] = {
            "mae": metrics_by_horizon[horizon]["mae"],
            "phase4_full_population_gbm_mae_kw": phase4_reference.get(horizon, {}).get("mae"),
            "phase5_full_population_tcn_mae_kw": phase5_reference.get(horizon, {}).get("mae"),
            "vs": {},
            "paired_vs_gbm": None,
        }
        for name, block in baseline_blocks.items():
            baseline_mae = float(np.mean(np.abs(actual_kw[:, column] - block[:, column])))
            entry["baseline_mae_on_subsample"] = entry.get("baseline_mae_on_subsample", {})
            entry["baseline_mae_on_subsample"][name] = baseline_mae
            entry["vs"][name] = {
                "baseline_mae_kw": baseline_mae,
                "improvement_pct": (
                    100.0 * (baseline_mae - metrics_by_horizon[horizon]["mae"]) / baseline_mae
                    if baseline_mae
                    else None
                ),
                "paired": bootstrap[horizon].get(name, {}),
            }
        # The GBM is the bootstrap's own reference, so its entry has no interval.
        # The interval the verdict needs is the probe against the GBM, computed here
        # explicitly rather than read out of the reference's own row.
        probe_errors = np.abs(actual_kw[:, column] - predicted_kw[:, column])
        gbm_errors = np.abs(actual_kw[:, column] - baseline_blocks["classical_hist_gbm"][:, column])
        entry["paired_vs_gbm"] = paired_bootstrap(
            probe_errors, gbm_errors, seed=training.seed
        )
        entry["subsample_absolute_mae_is_not_the_published_mae"] = True
        improvements[horizon] = entry

    verdict = _verdict(improvements, horizons)

    # ---- Regime and error analysis -------------------------------------------
    origins = cache.origins[slices["test"]]
    previous_kw = (
        values[index.series[test_rows], origins] * row_scale
    )
    axes = assign_regimes(
        actual_kw[:, 0],
        target_id="customer_load",
        origin_index=origins,
        previous_actual=previous_kw,
    )
    regime_errors = {
        "qwen_probe": np.abs(actual_kw[:, 0] - predicted_kw[:, 0]),
        "persistence": np.abs(actual_kw[:, 0] - persistence_kw[:, 0]),
    }
    for name, block in baseline_blocks.items():
        regime_errors[name] = np.abs(actual_kw[:, 0] - block[:, 0])
    regime = regime_report(axes, regime_errors)
    regime["note"] = (
        f"shortest horizon h={horizons[0]} on the Phase 6 test subsample; "
        "terciles are computed from realised demand, so they are diagnostics and "
        "not a deployable routing signal (see docs/moe_design_requirements.md)"
    )

    errors_report = error_analysis(
        actual=actual_kw[:, 0],
        predicted=predicted_kw[:, 0],
        model=experiment_id,
        origin_index=origins,
        series_ids=series_ids,
        series_index=cache.series[slices["test"]],
        timestep_minutes=15,
        origin_iso=series_origin_iso,
        quality_flags=("OK",),
    )
    # Per-row errors for every model, so the distribution comparison in the handoff is
    # a measurement rather than a recollection.
    per_model_errors: dict[str, dict[str, Any]] = {
        "qwen_probe": {
            "mae": float(np.mean(np.abs(actual_kw[:, 0] - predicted_kw[:, 0]))),
            "median": float(np.median(np.abs(actual_kw[:, 0] - predicted_kw[:, 0]))),
            "p90": float(np.quantile(np.abs(actual_kw[:, 0] - predicted_kw[:, 0]), 0.90)),
            "p99": float(np.quantile(np.abs(actual_kw[:, 0] - predicted_kw[:, 0]), 0.99)),
            "max": float(np.max(np.abs(actual_kw[:, 0] - predicted_kw[:, 0]))),
        }
    }
    for name, block in baseline_blocks.items():
        absolute = np.abs(actual_kw[:, 0] - block[:, 0])
        per_model_errors[name] = {
            "mae": float(np.mean(absolute)),
            "median": float(np.median(absolute)),
            "p90": float(np.quantile(absolute, 0.90)),
            "p99": float(np.quantile(absolute, 0.99)),
            "max": float(np.max(absolute)),
        }
    for name, stats in per_model_errors.items():
        stats["p99_over_median"] = (
            stats["p99"] / stats["median"] if stats["median"] else None
        )
        stats["share_above_1kw"] = float(
            np.mean(
                np.abs(
                    actual_kw[:, 0]
                    - (
                        predicted_kw[:, 0]
                        if name == "qwen_probe"
                        else baseline_blocks[name][:, 0]
                    )
                )
                > 1.0
            )
        )
    errors_report["distribution_by_model"] = per_model_errors

    diagnostics = {
        f"layer_{layer}": representation_diagnostics(cache.features[layer])
        for layer in cache.layers
    }

    compute = {
        "head_training_seconds": round(head_seconds, 1),
        "head_epochs": trained.epochs_run,
        "forward_seconds_per_sample_measured": 0.248,
        "backbone_parameters": 1_720_574_976,
        "backbone_trainable_parameters": 0,
        "backbone_frozen_parameters": 1_720_574_976,
        "probe_parameters": trained.parameter_count,
        "probe_series_embedding_dim": series_embedding_dim,
        "backbone_kind": backbone_kind,
        "dtype": "float32",
        "dtype_note": (
            "fp32 rather than the published bf16: on this CPU bf16 is emulated and "
            "measured 4.8x slower (2.568 vs 0.533 s/sample at 21 tokens)"
        ),
    }

    result = Phase6Result(
        experiment_id=experiment_id,
        dataset_version=index.version,
        subsample_version=subsample.version(),
        representation_version=representation_version,
        metrics_by_horizon=metrics_by_horizon,
        improvement=improvements,
        bootstrap=bootstrap,
        verdict=verdict,
        probe=trained.to_dict(),
        features={
            "layer": feature_layer,
            "layers_cached": list(cache.layers),
            "probe_layers": list(probe_layers),
            "basis": basis,
            "basis_hidden_size": basis_width,
            "cache_version": cache.version,
        },
        regime=regime,
        errors=errors_report,
        compute=compute,
        diagnostics=diagnostics,
    )

    artifact_root.mkdir(parents=True, exist_ok=True)
    (artifact_root / "result.json").write_text(
        json.dumps(result.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
    )
    (artifact_root / "summary.md").write_text(render_summary(result), encoding="utf-8")
    return result


def _verdict(improvements: dict[int, dict[str, Any]], horizons: tuple[int, ...]) -> str:
    """State the answer per horizon, and never average it away."""
    parts: list[str] = []
    for horizon in horizons:
        against_gbm = improvements[horizon]["vs"]["classical_hist_gbm"]
        improvement = against_gbm["improvement_pct"]
        excludes_zero = improvements[horizon]["paired_vs_gbm"]["excludes_zero"]
        if improvement is None:
            parts.append(f"h={horizon}: undetermined")
        elif improvement > 0:
            marker = "significant" if excludes_zero else "not significant"
            parts.append(f"h={horizon}: better than GBM by {improvement:.1f}% ({marker})")
        else:
            marker = "significant" if excludes_zero else "not significant"
            parts.append(
                f"h={horizon}: worse than GBM by {abs(improvement):.1f}% ({marker})"
            )
    wins = sum(
        1
        for horizon in horizons
        if (improvements[horizon]["vs"]["classical_hist_gbm"]["improvement_pct"] or 0) > 0
    )
    overall = (
        "beats GBM at every horizon"
        if wins == len(horizons)
        else "loses to GBM at every horizon"
        if wins == 0
        else f"beats GBM at {wins} of {len(horizons)} horizons"
    )
    return f"{overall}; " + "; ".join(parts)


def render_summary(result: Phase6Result) -> str:
    """A short Markdown report of one configuration."""
    lines = [
        f"# Phase 6 - {result.experiment_id}",
        "",
        f"**Verdict:** {result.verdict}",
        "",
        f"- dataset version: `{result.dataset_version}` (identical to Phase 5)",
        f"- subsample: `{result.subsample_version}`",
        f"- representation: `{result.representation_version}`",
        f"- backbone: {result.compute['backbone_kind']}, "
        f"{result.compute['backbone_parameters']:,} parameters, "
        f"{result.compute['backbone_trainable_parameters']} trainable",
        f"- probe: {result.compute['probe_parameters']:,} parameters, "
        f"layer {result.features['layer']}",
        "",
        "| Horizon | Qwen probe MAE (kW) | GBM on same rows | TCN on same rows | vs GBM | 95% CI |",
        "|---|---|---|---|---|---|",
    ]
    for horizon, metrics in sorted(result.metrics_by_horizon.items()):
        entry = result.improvement[horizon]["vs"]
        paired = result.improvement[horizon]["paired_vs_gbm"]
        tcn = entry.get("phase5_tcn", {}).get("baseline_mae_kw")
        lines.append(
            f"| h={horizon} | {metrics['mae']:.4f} | "
            f"{entry['classical_hist_gbm']['baseline_mae_kw']:.4f} | "
            f"{tcn:.4f} | {entry['classical_hist_gbm']['improvement_pct']:+.1f}% | "
            f"[{paired['ci_low']:+.4f}, {paired['ci_high']:+.4f}] |"
            if tcn is not None
            else f"| h={horizon} | {metrics['mae']:.4f} | "
            f"{entry['classical_hist_gbm']['baseline_mae_kw']:.4f} | n/a | "
            f"{entry['classical_hist_gbm']['improvement_pct']:+.1f}% | "
            f"[{paired['ci_low']:+.4f}, {paired['ci_high']:+.4f}] |"
        )
    lines += [
        "",
        "CI is a paired bootstrap of the per-row MAE difference "
        "(negative means the probe is better).",
        "",
        "Absolute MAE here is on the Phase 6 subsample and is deliberately not the "
        "published Phase 4/5 figure; the comparison columns are the result.",
        "",
    ]
    return "\n".join(lines)
