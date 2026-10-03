"""Phase 5 runner: train, select, evaluate once, analyse, record.

The evaluation discipline
-------------------------
:func:`run_phase5_experiment` selects its checkpoint on **validation** and then
evaluates the test split **once**. The test rows are not visible to
:func:`~energy_intelligence.ml.training.fit_temporal_model` at all, so "do not tune
on test" is enforced by the function signatures rather than by good intentions.

Comparability with Phase 4
--------------------------
The test population here is *identical* to Phase 4's: the same origins, the same 40
customers, the same 179,520 rows, because evaluation origins always use stride 1 and
only training origins are subsampled. The Phase 4 baselines are therefore read from
their own committed artifact rather than retyped, so a stale number cannot creep in
and the comparison is against what was actually measured then.

The Phase 3 balance residual
----------------------------
Orthogonal to this target, exactly as recorded in Phase 4: it is a feeder-level
constant at one published timepoint, not a per-customer or per-step quantity. It is
restated in every summary artifact so the two findings are never conflated.
"""

from __future__ import annotations

import json
import platform
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..config.temporal import TemporalExperimentConfig
from ..data.smartds import SmartDsAdapter, SmartDsLayout
from ..paths import ProjectPaths
from .analysis import assign_regimes, error_analysis, regime_report
from .metrics import evaluate, mae
from .registry import ExperimentRecord, append_record, environment, now_iso
from .scaling import FeatureScaler, assert_train_only_fit
from .sequence import SequenceConfig, SequenceIndex, build_sequence_index, gather_sequences
from .temporal import TemporalDemandModel, build_model
from .targets import TargetSeries, load_target, select_customer_sample
from .training import (
    TrainingConfig,
    TrainedModel,
    evaluate_checkpoint,
    fit_temporal_model,
    predict_series,
)

__all__ = [
    "Phase5Result",
    "analyse_predictions",
    "load_phase4_reference",
    "load_phase4_regimes",
    "load_series_for_experiment",
    "replay_evaluation",
    "run_phase5_experiment",
]


@dataclass(frozen=True, slots=True)
class Phase5Result:
    """One trained temporal model and everything measured about it."""

    experiment_id: str
    dataset_version: str
    index_summary: dict[str, Any]
    trained: TrainedModel
    metrics_by_horizon: dict[int, dict[str, Any]]
    persistence_by_horizon: dict[int, dict[str, Any]]
    phase4_reference: dict[str, dict[int, dict[str, Any]]]
    improvement: dict[int, dict[str, float]]
    error_distribution: dict[str, Any]
    regimes: dict[str, Any]
    errors: dict[str, Any]
    seconds: float
    artifacts: dict[str, str] = field(default_factory=dict)

    def render(self) -> str:
        lines = [
            f"experiment: {self.experiment_id}",
            f"dataset:    {self.dataset_version}",
            f"model:      {self.trained.model.architecture} "
            f"({self.trained.model.parameter_count()} parameters, "
            f"best epoch {self.trained.best_epoch}, "
            f"{'early stop' if self.trained.stopped_early else 'all epochs'})",
            f"train time: {self.trained.train_seconds:.1f}s",
            "",
            f"{'horizon':>8s}{'persistence':>14s}{'phase5':>12s}{'phase4 GBM':>13s}"
            f"{'vs GBM':>10s}{'R2':>8s}",
            "-" * 65,
        ]
        for horizon in sorted(self.metrics_by_horizon):
            mine = self.metrics_by_horizon[horizon]["mae"]
            reference = self.phase4_reference.get("classical_hist_gbm", {}).get(horizon, {})
            gbm = reference.get("mae")
            lines.append(
                f"{horizon:>8d}{self.persistence_by_horizon[horizon]['mae']:>14.4f}"
                f"{mine:>12.4f}"
                f"{f'{gbm:.4f}' if gbm is not None else 'n/a':>13s}"
                f"{self.improvement[horizon]['vs_gbm_pct']:>9.1f}%"
                f"{self.metrics_by_horizon[horizon]['r2']:>8.3f}"
            )
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "dataset_version": self.dataset_version,
            "index": self.index_summary,
            "model": self.trained.model.config_dict(),
            "training": self.trained.to_dict(),
            "history": [h.to_dict() for h in self.trained.history],
            "metrics_by_horizon": {str(k): v for k, v in self.metrics_by_horizon.items()},
            "persistence_by_horizon": {
                str(k): v for k, v in self.persistence_by_horizon.items()
            },
            "phase4_reference": {
                model: {str(h): v for h, v in horizons.items()}
                for model, horizons in self.phase4_reference.items()
            },
            "improvement": {str(h): v for h, v in self.improvement.items()},
            "error_distribution": self.error_distribution,
            "regimes": self.regimes,
            "errors": self.errors,
            "seconds": round(self.seconds, 2),
            "artifacts": self.artifacts,
        }


def load_series_for_experiment(
    layout: SmartDsLayout, config: TemporalExperimentConfig
) -> TargetSeries:
    """Extract the demand series using the Phase 4 sampling, unchanged.

    The customer count, the commercial fraction and the seed all come from the Phase 4
    contract, so the 40 series are the same 40 series and the test population is
    identical. That is what makes the two phases' numbers comparable.
    """
    adapter = SmartDsAdapter(layout)
    names, _report = select_customer_sample(
        adapter,
        count=config.customer_count,
        seed=config.seed,
        commercial_fraction=config.commercial_fraction,
    )
    return load_target(adapter, names)


def load_phase4_reference(path: Path) -> dict[str, dict[int, dict[str, Any]]]:
    """Read the Phase 4 comparison artifact, rather than retyping its numbers.

    Raises:
        FileNotFoundError: If the artifact is absent, so a Phase 5 comparison can
            never be made against remembered values.
    """
    if not path.is_file():
        raise FileNotFoundError(
            f"Phase 4 reference not found at {path}. Run the Phase 4 load experiment "
            "before Phase 5, so the comparison is against measured numbers."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    reference: dict[str, dict[int, dict[str, Any]]] = {}
    for model in payload["models"]:
        reference[model["model"]] = {
            int(horizon): metrics for horizon, metrics in model["metrics_by_horizon"].items()
        }
    return reference


def _load_phase4_regimes(paths: ProjectPaths) -> dict[str, Any] | None:
    """Read the Phase 4 regime report, if it exists, for side-by-side comparison."""
    path = paths.artifacts / "ml" / "load-40" / "reports" / "regimes.json"
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        axis: payload[axis]
        for axis in ("load_level", "load_ramp")
        if axis in payload
    }


def _window_scaler(
    series_values: np.ndarray, index: SequenceIndex, rows: np.ndarray, cap: int = 4096
) -> FeatureScaler:
    """Fit the feature scaler on gathered training windows only.

    A handful of training windows is enough to estimate a mean and a standard
    deviation for six channels, and it avoids materialising 237,600 x 168 x 6 values
    to compute two statistics. Deterministic: the rows are the first ``cap`` training
    rows, not a sample.
    """
    chosen = rows[:cap]
    blocks = []
    for start in range(0, chosen.size, 512):
        raw, _, _ = gather_sequences(series_values, index, chosen[start : start + 512])
        blocks.append(raw.transpose(0, 2, 1).reshape(raw.shape[0], -1).astype(np.float64))
    scaler = FeatureScaler.fit(np.concatenate(blocks, axis=0))
    assert_train_only_fit(scaler, int(sum(block.shape[0] for block in blocks)))
    return scaler


def _to_kw(values: np.ndarray, row_scale: np.ndarray) -> np.ndarray:
    """Convert per-unit values to kW using each row's own scale.

    Handles both a single horizon column (``(n,)``) and a full multi-horizon block
    (``(n, horizons)``). Adding an axis unconditionally would broadcast the row
    scale against itself and allocate an ``(n, n)`` matrix - which is exactly what
    happened before this was fixed.
    """
    values = np.asarray(values, dtype=np.float64)
    scale = np.asarray(row_scale, dtype=np.float64)
    if values.ndim == 1:
        return values * scale
    if values.ndim == 2:
        return values * scale[:, None]
    raise ValueError(f"expected a 1-D or 2-D array, got shape {values.shape}")


def _distribution(actual: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    """Error distribution, not just the mean.

    The question this phase asks is whether a temporal model reduces *large*
    failures, not merely nudges an average, so the upper quantiles and the worst case
    are reported alongside MAE.
    """
    error = np.abs(np.asarray(actual) - np.asarray(predicted))
    return {
        "n": int(error.size),
        "mae": float(error.mean()),
        "median": float(np.median(error)),
        "p75": float(np.quantile(error, 0.75)),
        "p90": float(np.quantile(error, 0.90)),
        "p99": float(np.quantile(error, 0.99)),
        "max": float(error.max()),
        "p99_over_median": (
            float(np.quantile(error, 0.99) / np.median(error)) if np.median(error) > 0 else None
        ),
        "share_above_1kw": float(np.mean(error > 1.0)),
        "share_above_5kw": float(np.mean(error > 5.0)),
    }


def replay_evaluation(
    series: TargetSeries,
    *,
    sequence_config: SequenceConfig,
    paths: ProjectPaths,
    label: str,
    architecture: str,
    experiment_id: str,
    model_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Rebuild a saved checkpoint's test analysis without retraining.

    Used when the *analysis* needs correcting rather than the model. Two things make
    it trustworthy:

    * the checkpoint is the best-validation epoch, restored by ``fit_temporal_model``;
    * the recomputed MAE is compared against the stored ``result.json``, and a
      mismatch raises rather than being written out.

    That makes this both a repair tool and a reproducibility check: a checkpoint
    that no longer reproduces its own recorded metrics is a real problem, not
    something to paper over.

    Returns:
        A payload with the recomputed metrics and the refreshed analysis artifacts.
    """
    from .temporal import build_model
    from .training import evaluate_checkpoint, fit_temporal_model

    root = paths.artifacts / "phase5" / label
    checkpoint_path = root / "checkpoint.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"no checkpoint at {checkpoint_path}; run the experiment first")

    values = series.normalised().astype(np.float32)
    n_series, n_steps = values.shape
    index = build_sequence_index(
        series_count=n_series,
        total_steps=n_steps,
        config=sequence_config,
        target_id=series.spec.target_id,
        value_origin=series.value_origin,
        series_ids=series.series_ids,
        source=series.provenance.source.locator,
    )
    scaler = _window_scaler(values, index, index.rows(0))

    payload = torch.load(checkpoint_path, weights_only=False)
    model = build_model(
        architecture,
        in_channels=sequence_config.channel_count,
        horizons=sequence_config.horizons,
        series_count=n_series,
        params=payload.get("config", model_params),
    )
    model.load_state_dict(payload["state_dict"])

    trained = TrainedModel(
        model=model,
        history=(),
        best_epoch=int(payload.get("best_epoch", 0)),
        best_validation_mae=float(payload.get("best_validation_mae_per_unit", float("nan"))),
        stopped_early=False,
        train_seconds=0.0,
        checkpoint_path=checkpoint_path,
        train_rows=int(index.rows(0).size),
        validation_rows=int(index.rows(1).size),
    )

    actual_pu, predicted_pu, persistence_pu, evaluated_rows = evaluate_checkpoint(
        trained, values, index, split_code=2, scaler=scaler
    )
    row_scale = series.scale_vector[index.series[evaluated_rows]]

    metrics: dict[int, dict[str, Any]] = {}
    improvement: dict[int, dict[str, float]] = {}
    position = {h: i for i, h in enumerate(sequence_config.horizons)}
    phase4_reference = load_phase4_reference(
        paths.artifacts / "ml" / "load-40" / "reports" / "comparison.json"
    )
    for horizon in sequence_config.horizons:
        column = position[horizon]
        actual_kw = _to_kw(actual_pu[:, column], row_scale)
        predicted_kw = _to_kw(predicted_pu[:, column], row_scale)
        persistence_kw = _to_kw(persistence_pu[:, column], row_scale)
        metrics[horizon] = evaluate(actual_kw, predicted_kw).to_dict()
        baseline = evaluate(actual_kw, persistence_kw)
        gbm = phase4_reference.get("classical_hist_gbm", {}).get(horizon, {})
        gbm_mae = gbm.get("mae")
        improvement[horizon] = {
            "mae": metrics[horizon]["mae"],
            "vs_persistence_pct": (
                100.0 * (baseline.mae - metrics[horizon]["mae"]) / baseline.mae
                if baseline.mae
                else None
            ),
            "phase4_gbm_mae_kw": gbm_mae,
            "vs_gbm_pct": (
                100.0 * (gbm_mae - metrics[horizon]["mae"]) / gbm_mae if gbm_mae else None
            ),
        }

    stored_path = root / "result.json"
    if stored_path.is_file():
        stored = json.loads(stored_path.read_text(encoding="utf-8"))
        for horizon in sequence_config.horizons:
            recorded = stored["metrics_by_horizon"][str(horizon)]["mae"]
            replayed = metrics[horizon]["mae"]
            if abs(recorded - replayed) > 1e-9:
                raise AssertionError(
                    f"checkpoint replay does not reproduce the recorded MAE at "
                    f"h={horizon}: {recorded} recorded vs {replayed} replayed"
                )

    regimes_by_horizon, regimes, errors = analyse_predictions(
        series=series,
        index=index,
        values=values,
        evaluated_rows=evaluated_rows,
        actual_pu=actual_pu,
        predicted_pu=predicted_pu,
        persistence_pu=persistence_pu,
        row_scale=row_scale,
        phase4_regimes=_load_phase4_regimes(paths),
        experiment_id=experiment_id,
    )
    (root / "regimes.json").write_text(
        json.dumps(
            {
                "headline_horizon": sequence_config.horizons[0],
                "note": (
                    "Each horizon is analysed against its OWN errors. The Phase 4 "
                    "artifact reported the shortest horizon, so the Phase 4 numbers "
                    "quoted here are comparable only with horizons[0]."
                ),
                "by_horizon": {str(h): v for h, v in regimes_by_horizon.items()},
                "headline": regimes,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (root / "error_analysis.json").write_text(
        json.dumps(errors, indent=2, sort_keys=True), encoding="utf-8"
    )

    return {
        "experiment_id": experiment_id,
        "dataset_version": index.version,
        "metrics_by_horizon": {str(h): v for h, v in metrics.items()},
        "improvement": {str(h): v for h, v in improvement.items()},
        "regimes_headline": regimes,
        "regimes_by_horizon": {str(h): v for h, v in regimes_by_horizon.items()},
        "errors": errors,
    }


def analyse_predictions(
    *,
    series: TargetSeries,
    index: SequenceIndex,
    values: np.ndarray,
    evaluated_rows: np.ndarray,
    actual_pu: np.ndarray,
    predicted_pu: np.ndarray,
    persistence_pu: np.ndarray,
    row_scale: np.ndarray,
    phase4_regimes: dict[str, Any] | None = None,
    experiment_id: str = "phase5",
) -> tuple[dict[str, Any], dict[int, Any], dict[str, Any]]:
    """Regime and error analysis for one model's test predictions.

    Factored out of the runner so that (a) there is exactly one analysis path, and
    (b) a checkpoint can be replayed to regenerate these artifacts without retraining.

    The regime axes are always built from the **shortest horizon's** actual values.
    An earlier version rebuilt the error arrays inside the per-horizon metrics loop,
    so the regime table ended up labelling h=1 regimes with h=96 errors - which
    produced per-regime MAEs four times the overall figure. The horizon is now
    explicit and recorded, and every horizon gets its own regime table.

    Returns:
        ``(regimes_by_horizon, headline_regimes, error_analysis)``.
    """
    position = {h: i for i, h in enumerate(index.config.horizons)}
    origin_index = index.origins[evaluated_rows]

    regimes_by_horizon: dict[int, Any] = {}
    for horizon in index.config.horizons:
        column = position[horizon]
        actual_kw = _to_kw(actual_pu[:, column], row_scale)
        previous = values[
            index.series[evaluated_rows], origin_index + horizon - 1
        ]
        axes = assign_regimes(
            actual_kw,
            target_id=series.spec.target_id,
            origin_index=origin_index,
            previous_actual=_to_kw(previous, row_scale),
        )
        regimes_by_horizon[horizon] = regime_report(
            axes,
            {
                "phase5": np.abs(actual_kw - _to_kw(predicted_pu[:, column], row_scale)),
                "persistence": np.abs(
                    actual_kw - _to_kw(persistence_pu[:, column], row_scale)
                ),
            },
        )

    first = index.config.horizons[0]
    column = position[first]
    actual_first = _to_kw(actual_pu[:, column], row_scale)
    predicted_first = _to_kw(predicted_pu[:, column], row_scale)
    headline = dict(regimes_by_horizon[first])
    headline["horizon"] = first
    if phase4_regimes is not None:
        headline["phase4_reference_regimes"] = phase4_regimes

    errors = error_analysis(
        actual=actual_first,
        predicted=predicted_first,
        model=experiment_id,
        origin_index=origin_index,
        series_ids=series.series_ids,
        series_index=index.series[evaluated_rows],
        timestep_minutes=15,
        origin_iso=series.origin,
        quality_flags=sorted(flag.value for flag in series.quality.flags),
    )
    return regimes_by_horizon, headline, errors


def run_phase5_experiment(
    series: TargetSeries,
    *,
    sequence_config: SequenceConfig,
    training_config: TrainingConfig,
    paths: ProjectPaths,
    architecture: str,
    experiment_id: str,
    phase4_reference: dict[str, dict[int, dict[str, Any]]],
    label: str,
    reference_scale: np.ndarray | None = None,
    validation_stride: int = 1,
    model_params: dict[str, Any] | None = None,
) -> Phase5Result:
    """Train one temporal model and measure it once on the test split.

    Args:
        series: The extracted demand series, already scaled-aware.
        sequence_config: Window, stride, horizons and target formulation.
        training_config: The training protocol.
        paths: Project layout.
        architecture: ``"tcn"`` or ``"transformer"``.
        experiment_id: Stable identifier for the registry record.
        phase4_reference: Phase 4 metrics, read from its artifact.
        label: Artifact sub-directory name.
        reference_scale: Optional per-row kW scale, used by ablations whose series
            differ from the main experiment's.
        model_params: Architecture overrides from the configuration. Passed through
            rather than silently ignored, so a config change actually changes the
            model.

    Returns:
        The complete result, including regime and error analysis.
    """
    started = time.perf_counter()
    values = series.normalised().astype(np.float32)
    n_series, n_steps = values.shape

    index = build_sequence_index(
        series_count=n_series,
        total_steps=n_steps,
        config=sequence_config,
        target_id=series.spec.target_id,
        value_origin=series.value_origin,
        series_ids=series.series_ids,
        source=series.provenance.source.locator,
    )

    train_rows = index.rows(0)
    validation_rows = index.rows(1)
    test_rows = index.rows(2)
    if test_rows.size == 0:
        raise ValueError("test split is empty; there is nothing to evaluate")

    scaler = _window_scaler(values, index, train_rows)

    model = build_model(
        architecture,
        in_channels=sequence_config.channel_count,
        horizons=sequence_config.horizons,
        series_count=n_series,
        params=model_params,
    )

    root = paths.artifacts / "phase5" / label
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "dataset_manifest.json"
    manifest_path.write_text(
        json.dumps(index.to_manifest(), indent=2, sort_keys=True), encoding="utf-8"
    )

    trained = fit_temporal_model(
        model,
        values,
        index,
        config=training_config,
        scaler=scaler,
        checkpoint_path=root / "checkpoint.pt",
        validation_stride=validation_stride,
    )
    training_log = root / "training_log.jsonl"
    training_log.write_text(
        "\n".join(json.dumps(h.to_dict(), sort_keys=True) for h in trained.history) + "\n",
        encoding="utf-8",
    )

    # The single test evaluation.
    actual_pu, predicted_pu, persistence_pu, evaluated_rows = evaluate_checkpoint(
        trained, values, index, split_code=2, scaler=scaler
    )
    # Each row's kW divisor. Rows are stored series-major (every origin repeated once
    # per series), so the series index is exactly which scale applies.
    row_scale = (
        series.scale_vector[index.series[evaluated_rows]]
        if reference_scale is None
        else np.asarray(reference_scale, dtype=np.float64)
    )

    metrics_by_horizon: dict[int, dict[str, Any]] = {}
    persistence_by_horizon: dict[int, dict[str, Any]] = {}
    improvement: dict[int, dict[str, float]] = {}
    position = {h: i for i, h in enumerate(sequence_config.horizons)}
    absolute_errors: dict[str, np.ndarray] = {}

    for horizon in sequence_config.horizons:
        column = position[horizon]
        actual_kw = _to_kw(actual_pu[:, column], row_scale)
        predicted_kw = _to_kw(predicted_pu[:, column], row_scale)
        persistence_kw = _to_kw(persistence_pu[:, column], row_scale)

        ramp_actual = ramp_predicted = None
        if horizon + 1 in sequence_config.horizons:
            ramp_actual = np.stack([actual_pu[:, column], actual_pu[:, column + 1]], axis=1)
            ramp_predicted = np.stack(
                [predicted_pu[:, column], predicted_pu[:, column + 1]], axis=1
            )

        metrics = evaluate(
            actual_kw,
            predicted_kw,
            ramp_actual=None if ramp_actual is None else _to_kw(ramp_actual, row_scale),
            ramp_predicted=(
                None if ramp_predicted is None else _to_kw(ramp_predicted, row_scale)
            ),
        )
        payload = metrics.to_dict()
        payload["unit"] = series.spec.unit
        payload["mae_per_unit"] = mae(actual_pu[:, column], predicted_pu[:, column])
        metrics_by_horizon[horizon] = payload

        baseline = evaluate(actual_kw, persistence_kw)
        persistence_by_horizon[horizon] = baseline.to_dict()
        absolute_errors["phase5"] = np.abs(actual_kw - predicted_kw)
        absolute_errors["persistence"] = np.abs(actual_kw - persistence_kw)

        gbm = phase4_reference.get("classical_hist_gbm", {}).get(horizon, {})
        gbm_mae = gbm.get("mae")
        improvement[horizon] = {
            "mae": metrics.mae,
            "mae_per_unit": payload["mae_per_unit"],
            "vs_persistence_pct": (
                100.0 * (baseline.mae - metrics.mae) / baseline.mae if baseline.mae else None
            ),
            "phase4_gbm_mae_kw": gbm_mae,
            "vs_gbm_pct": (
                100.0 * (gbm_mae - metrics.mae) / gbm_mae if gbm_mae else None
            ),
        }

    distribution = {
        "phase5": _distribution(
            _to_kw(actual_pu[:, 0], row_scale), _to_kw(predicted_pu[:, 0], row_scale)
        ),
        "persistence": _distribution(
            _to_kw(actual_pu[:, 0], row_scale), _to_kw(persistence_pu[:, 0], row_scale)
        ),
        "note": "computed at the shortest horizon, where errors are smallest",
    }
    for model_name, horizons in phase4_reference.items():
        if model_name not in {"classical_hist_gbm", "naive_last_value"}:
            continue
        entry = horizons.get(sequence_config.horizons[0])
        if entry is None:
            continue
        distribution[f"phase4_{model_name}"] = {
            "mae": entry.get("mae"),
            # Phase 4 recorded no quantiles, so these are None rather than guesses.
            "rmse": entry.get("rmse"),
            "peak_error_mae": entry.get("peak_error_mae"),
            "n": entry.get("n"),
            "note": "Phase 4 artifact; it recorded no quantiles, so only MAE, RMSE "
            "and peak error are comparable",
        }

    # ---- Regime and error analysis ---------------------------------
    # One shared analysis path: the same code runs here and when a checkpoint is
    # replayed, so the artifacts cannot drift between a live run and a replay.
    phase4_regimes = _load_phase4_regimes(paths)
    regimes_by_horizon, regimes, errors = analyse_predictions(
        series=series,
        index=index,
        values=values,
        evaluated_rows=evaluated_rows,
        actual_pu=actual_pu,
        predicted_pu=predicted_pu,
        persistence_pu=persistence_pu,
        row_scale=row_scale,
        phase4_regimes=phase4_regimes,
        experiment_id=experiment_id,
    )
    (root / "regimes.json").write_text(
        json.dumps(
            {
                "headline_horizon": sequence_config.horizons[0],
                "note": (
                    "Each horizon is analysed against its OWN errors. The Phase 4 "
                    "artifact reported the shortest horizon, so the Phase 4 numbers "
                    "quoted here are comparable only with horizons[0]."
                ),
                "by_horizon": {str(h): v for h, v in regimes_by_horizon.items()},
                "headline": regimes,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (root / "error_analysis.json").write_text(
        json.dumps(errors, indent=2, sort_keys=True), encoding="utf-8"
    )

    result = Phase5Result(
        experiment_id=experiment_id,
        dataset_version=index.version,
        index_summary=index.summary(),
        trained=trained,
        metrics_by_horizon=metrics_by_horizon,
        persistence_by_horizon=persistence_by_horizon,
        phase4_reference={
            model: {h: horizons[h] for h in sequence_config.horizons if h in horizons}
            for model, horizons in phase4_reference.items()
        },
        improvement=improvement,
        error_distribution=distribution,
        regimes=regimes,
        errors=errors,
        seconds=time.perf_counter() - started,
        artifacts={
            "checkpoint": str(root / "checkpoint.pt"),
            "training_log": str(training_log),
            "dataset_manifest": str(manifest_path),
            "regimes": str(root / "regimes.json"),
            "errors": str(root / "error_analysis.json"),
        },
    )
    (root / "result.json").write_text(
        json.dumps(result.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
    )

    append_record(
        paths.experiments / "registry.jsonl",
        ExperimentRecord(
            experiment_id=experiment_id,
            created_at=now_iso(),
            target_id=index.target_id,
            dataset_version=index.version,
            dataset_sha256=json.loads(manifest_path.read_text(encoding="utf-8")).get(
                "dataset_version", ""
            ),
            feature_names=sequence_config.channel_names,
            lookback_steps=sequence_config.lookback_steps,
            horizon_steps=max(sequence_config.horizons),
            split_fractions={
                "train": sequence_config.train_fraction,
                "validation": sequence_config.validation_fraction,
            },
            split_counts={
                "train": int(train_rows.size),
                "validation": int(validation_rows.size),
                "test": int(test_rows.size),
            },
            model=f"phase5_{architecture}",
            model_family="temporal",
            hyperparameters={
                "sequence": sequence_config.to_dict(),
                "training": training_config.to_dict(),
                "model": trained.model.config_dict(),
                "parameter_count": trained.model.parameter_count(),
                "checkpoint_selection": "validation MAE per unit, mean across horizons",
                "loss_weights": "equal across horizons (D-075)",
            },
            seed=training_config.seed,
            train_seconds=trained.train_seconds,
            predict_seconds=round(result.seconds - trained.train_seconds, 2),
            metrics={
                str(h): {
                    "mae": v["mae"],
                    "rmse": v["rmse"],
                    "r2": v["r2"],
                    "vs_gbm_pct": improvement[h]["vs_gbm_pct"],
                    "vs_persistence_pct": improvement[h]["vs_persistence_pct"],
                }
                for h, v in metrics_by_horizon.items()
            },
            artifacts=result.artifacts,
            environment={**environment(), "cpu": platform.processor() or platform.machine()},
            notes=(
                "Energy Demand Dynamics Model: a forecasting model over demand "
                "dynamics. NOT an action-conditioned world model - SMART-DS supplies "
                "no action, transition or system response (docs/world_model_requirements.md).",
                f"test rows {int(test_rows.size)} are identical to the Phase 4 test "
                "population, so the two phases' numbers are directly comparable",
                "checkpoint selected on validation only; the test split was evaluated once",
            ),
        ),
    )
    return result
