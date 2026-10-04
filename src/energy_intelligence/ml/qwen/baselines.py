"""Baselines re-measured on exactly the rows Phase 6 scores.

This module exists because of one requirement: a subsample comparison is only
meaningful if every model is scored on the same rows. Quoting Phase 4's published
0.4242 kW next to a Qwen number computed on 10,000 rows would compare two different
populations and call it a result.

So persistence, Phase 4's classical gradient boosting and Phase 5's temporal TCN are
all re-run here on the Phase 6 subsample. The published Phase 4/5 numbers stay in the
artifacts as context, and the difference between the two is reported, because a
subsample's absolute MAE is not the full population's MAE and pretending otherwise
would be the easiest way to produce a confident wrong answer.

The uncertainty that follows from subsampling is handled with a **paired** bootstrap:
the same rows are scored by every model, so the difference of per-row errors is what
gets resampled. That cancels the row-to-row variance which dominates the absolute MAE,
and is far tighter than comparing two independent confidence intervals.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "BaselinePredictions",
    "persistence_baseline",
    "gbm_baseline",
    "phase5_tcn_baseline",
    "paired_bootstrap",
    "bootstrap_summary",
]


@dataclass(frozen=True, slots=True)
class BaselinePredictions:
    """Per-horizon predictions in kW for one model on one split.

    Attributes:
        name: Model name as it will be reported.
        predicted: ``[N, horizons]`` in kW.
        actual: ``[N, horizons]`` in kW.
        detail: Model-specific provenance, e.g. fit seconds or checkpoint path.
    """

    name: str
    predicted: np.ndarray
    actual: np.ndarray
    detail: dict[str, Any]


def persistence_baseline(
    actual_kw: np.ndarray, persistence_kw: np.ndarray, *, name: str = "persistence"
) -> BaselinePredictions:
    """Persistence, i.e. ``yhat(t+h) = y(t)``. No fitting, so no leakage surface."""
    return BaselinePredictions(
        name=name, predicted=persistence_kw, actual=actual_kw, detail={"fitted": False}
    )


def gbm_baseline(
    *,
    train_features: np.ndarray,
    train_targets: np.ndarray,
    predict_features: dict[str, np.ndarray],
    horizons: tuple[int, ...],
    feature_names: tuple[str, ...],
    seed: int,
    log: Any = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Phase 4's classical histogram gradient boosting, refitted on the subsample.

    Uses the same estimator family, feature definitions and objective Phase 4 used.
    What changes is only the training population: the subsample's training rows. The
    direction of that effect is recorded in ``detail`` rather than left implicit,
    because a GBM re-trained on 10,000 rows is not the same model Phase 4 shipped.

    Args:
        train_features: Phase 4 feature matrix for the subsample's training rows.
        train_targets: ``[N, horizons]`` kW targets.
        predict_features: Split name to feature matrix, in the order to be returned.
        horizons: Horizon steps; one estimator per horizon.
        feature_names: Phase 4 feature names, in matrix column order.
        seed: Estimator seed.
        log: Optional progress callable.

    Returns:
        ``({split: predictions}, detail)`` with predictions ``[N, horizons]`` in kW.
    """
    from sklearn.ensemble import HistGradientBoostingRegressor

    estimators: list[Any] = []
    total_fit = 0.0
    for column, horizon in enumerate(horizons):
        started = time.perf_counter()
        estimator = HistGradientBoostingRegressor(
            loss="absolute_error",
            max_iter=200,
            learning_rate=0.06,
            max_leaf_nodes=31,
            min_samples_leaf=20,
            l2_regularization=1.0,
            early_stopping=False,
            random_state=seed,
        )
        estimator.fit(train_features, train_targets[:, column])
        fit_seconds = time.perf_counter() - started
        total_fit += fit_seconds
        estimators.append(estimator)
        if log is not None:
            train_mae = float(
                np.mean(np.abs(estimator.predict(train_features) - train_targets[:, column]))
            )
            log(f"  gbm h={horizon}: fit {fit_seconds:.1f}s train MAE {train_mae:.4f} kW")

    predictions = {
        split: np.column_stack([e.predict(matrix) for e, matrix in zip(estimators, [matrix] * len(estimators))])
        for split, matrix in predict_features.items()
    }
    detail = {
        "fitted": True,
        "fit_seconds": round(total_fit, 1),
        "n_features": len(feature_names),
        "feature_names": list(feature_names),
        "trained_on": "Phase 6 subsample training rows",
        "caveat": (
            "refitted on the subsample's training rows, not Phase 4's full training "
            "set; see the handoff for the direction of the expected effect"
        ),
    }
    return predictions, detail


def phase5_tcn_baseline(
    *,
    checkpoint_path: Path,
    values: np.ndarray,
    index: Any,
    rows: dict[str, np.ndarray],
    scale_vector: np.ndarray,
    scaler: Any,
    architecture: str = "tcn",
    model_params: dict[str, Any] | None = None,
    log: Any = None,
) -> dict[str, BaselinePredictions]:
    """Phase 5's actual TCN checkpoint, applied to the Phase 6 rows.

    This is the checkpoint Phase 5 committed its metrics from, loaded and run here -
    not a retrained approximation. Using the real weights matters: the comparison is
    against what Phase 5 delivered, so it must be that model.

    Args:
        checkpoint_path: The Phase 5 ``checkpoint.pt``.
        values: Per-unit values.
        index: The Phase 5 sequence index.
        rows: Split name to rows.
        scaler: The Phase 5 window scaler.
        architecture: Which architecture the checkpoint holds.
        model_params: Recorded model parameters, if the checkpoint stores them.
        log: Optional progress callable.

    Returns:
        ``{split_name: BaselinePredictions}``.
    """
    import torch
    from ..temporal import build_model
    from ..training import predict_series

    payload = torch.load(checkpoint_path, weights_only=False)
    model = build_model(
        architecture,
        in_channels=index.config.channel_count,
        horizons=index.config.horizons,
        series_count=values.shape[0],
        params=payload.get("config", model_params),
    )
    model.load_state_dict(payload["state_dict"])
    model.eval()

    out: dict[str, BaselinePredictions] = {}
    for name, split_rows in rows.items():
        started = time.perf_counter()
        actual_pu, predicted_pu, persistence_pu = predict_series(
            model, values, index, split_rows, scaler=scaler
        )
        # predict_series returns PER-UNIT levels, not kW. Phase 5's runner scales
        # them by the row's own rated kW before reporting, so this must too -
        # comparing per-unit predictions against kW actuals inflates the error to
        # roughly the size of the largest customer, which is what a first pass here
        # actually produced (13.3 kW where Phase 5 reported 0.4178 kW).
        scale = scale_vector[index.series[split_rows]]
        actual = actual_pu * scale[:, None]
        predicted = predicted_pu * scale[:, None]
        persistence = persistence_pu * scale[:, None]
        if log is not None:
            log(
                f"  tcn {name}: {split_rows.size} rows in "
                f"{time.perf_counter() - started:.1f}s, "
                f"MAE {np.abs(actual - predicted).mean():.4f} kW"
            )
        out[name] = BaselinePredictions(
            name="phase5_tcn",
            predicted=predicted,
            actual=actual,
            detail={
                "fitted": False,
                "checkpoint": str(checkpoint_path),
                "best_epoch": payload.get("best_epoch"),
                "best_validation_mae_per_unit": payload.get(
                    "best_validation_mae_per_unit"
                ),
            },
        )
    return out


def paired_bootstrap(
    errors_a: np.ndarray,
    errors_b: np.ndarray,
    *,
    samples: int = 2000,
    seed: int = 20260101,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Bootstrap the paired difference in mean absolute error.

    Args:
        errors_a: Per-row absolute errors for model A.
        errors_b: Per-row absolute errors for model B, row-aligned with A.
        samples: Resamples.
        seed: Resampling seed, so the interval is reproducible.
        alpha: Two-sided interval width.

    Returns:
        Point estimate and interval of ``mean(a) - mean(b)``. A negative value means
        A is better.

    Raises:
        ValueError: The arrays are not row-aligned.
    """
    a = np.asarray(errors_a, dtype=np.float64).ravel()
    b = np.asarray(errors_b, dtype=np.float64).ravel()
    if a.shape != b.shape:
        raise ValueError(f"paired bootstrap needs aligned rows, got {a.shape} vs {b.shape}")
    if a.size < 2:
        raise ValueError("need at least two rows for an interval")
    difference = a - b
    generator = np.random.default_rng(seed)
    n = difference.size
    means = np.empty(samples, dtype=np.float64)
    chunk = max(1, int(4e7 // max(1, n)))
    for start in range(0, samples, chunk):
        size = min(chunk, samples - start)
        picks = generator.integers(0, n, size=(size, n))
        means[start : start + size] = difference[picks].mean(axis=1)
    low = float(np.quantile(means, alpha / 2.0))
    high = float(np.quantile(means, 1.0 - alpha / 2.0))
    return {
        "mean_difference": float(difference.mean()),
        "ci_low": low,
        "ci_high": high,
        "excludes_zero": bool(low > 0.0 or high < 0.0),
        "samples": samples,
        "rows": int(n),
    }


def bootstrap_summary(
    errors: dict[str, np.ndarray], *, reference: str, samples: int = 2000, seed: int = 20260101
) -> dict[str, Any]:
    """Paired bootstrap of every model against one reference.

    Args:
        errors: Model name to per-row absolute errors.
        reference: The model every other model is compared against.
        samples: Resamples per comparison.
        seed: Resampling seed.

    Returns:
        ``{model: statistics}`` with the reference scored against itself as zero.
    """
    if reference not in errors:
        raise KeyError(f"reference {reference!r} not among {sorted(errors)}")
    out: dict[str, Any] = {}
    for name, values in errors.items():
        if name == reference:
            out[name] = {
                "mae": float(np.mean(values)),
                "vs_reference": None,
                "reference": reference,
            }
            continue
        stats = paired_bootstrap(values, errors[reference], samples=samples, seed=seed)
        reference_mae = float(np.mean(errors[reference]))
        stats["mae"] = float(np.mean(values))
        stats["reference"] = reference
        stats["reference_mae"] = reference_mae
        stats["relative_pct"] = (
            100.0 * stats["mean_difference"] / reference_mae if reference_mae else None
        )
        out[name] = stats
    return out


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a JSON artifact, creating parents."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
