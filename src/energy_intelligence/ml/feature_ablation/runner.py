"""One ablation run: fit on train, score on a fold, never touch the final test split.

The runner deliberately does **not** call
:func:`energy_intelligence.ml.experiment.run_experiment`. That function scores on the
repository's own test split, which is exactly the split Phase 10 must not read. Here the model
is fitted on the train rows, and scored only on the fold the caller names, with the fold
builder refusing any range at or beyond validation.

Everything except the feature matrix is held fixed by construction rather than by convention:
the same seed, the same model id, the same rows for training, the same metrics. The only thing
that varies between two rows of the ablation table is ``feature_names``.

Timestamp-level rows are returned, not just aggregates. ``forecast_origin``,
``target_timestamp``, ``actual``, ``prediction``, ``error``, ``absolute_error`` and
``squared_error`` are what a future paired significance analysis would need, and they cannot
be reconstructed after the fact - so they are written now while they are cheap to keep.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..baselines.classical import fit_classical
from ..metrics import mae, rmse
from ..splits import SPLIT_TRAIN, SPLIT_VALIDATION
from .folds import Fold, SampleAccounting, assert_final_test_denied

__all__ = [
    "FoldResult",
    "TimestampRows",
    "run_one",
    "run_feature_set",
    "primary_model_for",
]


@dataclass(frozen=True, slots=True)
class TimestampRows:
    """Timestamp-level evidence for one run, kept for later paired analysis.

    Attributes:
        forecast_origin: Origin step index per row.
        target_timestamp: Target step index per row.
        series: Series index per row.
        actual: Realised value.
        prediction: Model prediction.
    """

    forecast_origin: np.ndarray
    target_timestamp: np.ndarray
    series: np.ndarray
    actual: np.ndarray
    prediction: np.ndarray

    @property
    def error(self) -> np.ndarray:
        return self.actual - self.prediction

    @property
    def absolute_error(self) -> np.ndarray:
        return np.abs(self.error)

    @property
    def squared_error(self) -> np.ndarray:
        return self.error**2

    def to_dict(self, *, limit: int | None = None) -> dict[str, Any]:
        """A JSON-ready view, optionally truncated.

        Args:
            limit: Keep at most this many rows. ``None`` keeps all of them.

        Returns:
            The arrays as lists plus a note saying whether they were truncated.
        """
        size = self.forecast_origin.size if limit is None else min(limit, self.forecast_origin.size)
        return {
            "rows": int(self.forecast_origin.size),
            "kept": int(size),
            "truncated": bool(limit is not None and size < self.forecast_origin.size),
            "fields": [
                "forecast_origin",
                "target_timestamp",
                "series",
                "actual",
                "prediction",
                "error",
                "absolute_error",
                "squared_error",
            ],
            "forecast_origin": self.forecast_origin[:size].tolist(),
            "target_timestamp": self.target_timestamp[:size].tolist(),
            "series": self.series[:size].tolist(),
            "actual": self.actual[:size].tolist(),
            "prediction": self.prediction[:size].tolist(),
            "error": self.error[:size].tolist(),
            "absolute_error": self.absolute_error[:size].tolist(),
            "squared_error": self.squared_error[:size].tolist(),
        }


@dataclass(frozen=True, slots=True)
class FoldResult:
    """One (target, feature set, fold, model) cell of the ablation.

    Attributes:
        target: Target id.
        feature_set: Feature set id.
        fold_id: Fold identifier.
        fold_role: ``ABLATION`` or ``CONFIRMATION``.
        model: Model id actually fitted.
        feature_names: The features handed to the matrix builder.
        metrics: ``mae`` / ``rmse`` / ``n`` over the fold.
        train_rows: Rows used to fit.
        score_rows: Rows scored.
        dropped: Rows dropped for unavailable history.
        seconds: Wall time.
        account: The run's sample accounting.
        timestamp_rows: Per-row evidence.
        model_detail: Anything the model reports about itself.
        notes: Anything a reader would otherwise have to infer.
    """

    target: str
    feature_set: str
    fold_id: str
    fold_role: str
    model: str
    feature_names: tuple[str, ...]
    metrics: dict[str, Any]
    train_rows: int
    score_rows: int
    dropped: int
    seconds: float
    account: SampleAccounting
    timestamp_rows: TimestampRows | None = field(default=None, repr=False)
    model_detail: dict[str, Any] = field(default_factory=dict)
    notes: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self, *, keep_timestamps: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "target": self.target,
            "feature_set": self.feature_set,
            "fold": self.fold_id,
            "fold_role": self.fold_role,
            "model": self.model,
            "feature_names": list(self.feature_names),
            "feature_count": len(self.feature_names),
            "metrics": self.metrics,
            "train_rows": self.train_rows,
            "score_rows": self.score_rows,
            "dropped_rows": self.dropped,
            "seconds": round(self.seconds, 3),
            "samples": self.account.to_dict(),
            "model_detail": self.model_detail,
            "notes": list(self.notes),
        }
        if keep_timestamps and self.timestamp_rows is not None:
            payload["timestamps"] = self.timestamp_rows.to_dict(limit=2000)
        return payload


def primary_model_for(target: str) -> str:
    """The model the phase protocol froze for a target.

    Read from measured evidence rather than assumed uniform: Phase 5's registry recorded
    ``classical_hist_gbm`` as the winner for load and ``classical_ridge`` for PV, and a
    forced-uniform choice would misrepresent what the repository found.

    Args:
        target: Target id.

    Returns:
        A model id from
        :data:`~energy_intelligence.ml.baselines.classical.CLASSICAL_MODELS`.

    Raises:
        KeyError: If the target is not one Phase 10 can run.
    """
    mapping = {
        "customer_load": "classical_hist_gbm",
        "pv_generation": "classical_ridge",
    }
    if target not in mapping:
        raise KeyError(
            f"no frozen primary model for {target!r}; Phase 10 froze one for "
            f"{sorted(mapping)}"
        )
    return mapping[target]


def run_one(
    *,
    dataset,
    feature_matrix: np.ndarray,
    feature_names: tuple[str, ...],
    target: str,
    feature_set: str,
    fold: Fold,
    model: str,
    seed: int,
    validation_stop: int,
    horizon_steps: int,
    keep_timestamps: bool = True,
) -> FoldResult:
    """Fit on the train split and score one fold.

    Args:
        dataset: The Phase 4 ``MlDataset``. Only its **train** rows are read for fitting.
        feature_matrix: ``[n_samples, n_features]`` aligned with ``dataset``.
        feature_names: The column names of ``feature_matrix``.
        target: Target id, for labelling only.
        feature_set: Feature set id, for labelling only.
        fold: The fold to score.
        model: Model id to fit.
        seed: The single frozen seed.
        validation_stop: One past the last validation origin; the guard's boundary.
        horizon_steps: The horizon this run measures.
        keep_timestamps: Retain per-row evidence.

    Returns:
        The measured cell.

    Raises:
        FinalTestAccessError: If the fold would reach past validation.
        ValueError: If the fold has no scorable rows.
    """
    assert_final_test_denied(validation_stop, fold.stop)

    started = time.perf_counter()
    features = np.asarray(feature_matrix, dtype=np.float64)
    targets = np.asarray(dataset.targets[:, 0], dtype=np.float64)
    origins = np.asarray(dataset.origin_index, dtype=np.int64)
    series = np.asarray(dataset.series_index, dtype=np.int64)

    if features.shape[0] != origins.shape[0]:
        raise ValueError(
            f"feature matrix has {features.shape[0]} rows but the dataset has "
            f"{origins.shape[0]} samples; they must be aligned"
        )

    train_mask = dataset.mask(SPLIT_TRAIN)
    validation_mask = dataset.mask(SPLIT_VALIDATION)
    train_rows = int(np.count_nonzero(train_mask))
    if train_rows == 0:
        raise ValueError("the training split is empty; nothing to fit")

    finite_train = np.isfinite(features[train_mask]).all(axis=1)
    usable_train = train_mask.copy()
    usable_train[train_mask] = finite_train
    dropped = int(np.count_nonzero(train_mask) - np.count_nonzero(usable_train))

    score_mask = validation_mask & (origins >= fold.start) & (origins < fold.stop)
    score_rows = int(np.count_nonzero(score_mask))
    if score_rows == 0:
        raise ValueError(
            f"fold {fold.fold_id} [{fold.start}, {fold.stop}) contains no validation rows; "
            f"the fold definition and the dataset's origin range disagree"
        )

    model_object = fit_classical(
        model,
        features[usable_train],
        targets[usable_train],
        feature_names=feature_names,
        seed=int(seed),
    )
    prediction = np.asarray(
        model_object.predict(features[score_mask]), dtype=np.float64
    )
    actual = targets[score_mask]

    account = SampleAccounting(
        target=target,
        fold_id=fold.fold_id,
        feature_set=feature_set,
        origins=origins[score_mask],
        series=series[score_mask],
        dropped=dropped,
    )
    rows = None
    if keep_timestamps:
        rows = TimestampRows(
            forecast_origin=origins[score_mask].copy(),
            target_timestamp=origins[score_mask] + int(horizon_steps),
            series=series[score_mask].copy(),
            actual=actual.copy(),
            prediction=prediction.copy(),
        )

    return FoldResult(
        target=target,
        feature_set=feature_set,
        fold_id=fold.fold_id,
        fold_role=fold.role,
        model=model,
        feature_names=tuple(feature_names),
        metrics={
            "mae": float(mae(actual, prediction)),
            "rmse": float(rmse(actual, prediction)),
            "n": score_rows,
            "unit": str(dataset.target_unit),
        },
        train_rows=train_rows,
        score_rows=score_rows,
        dropped=dropped,
        seconds=time.perf_counter() - started,
        account=account,
        timestamp_rows=rows,
        model_detail=dict(getattr(model_object, "detail", {}) or {}),
        notes=(
            "fitted on the train split only; scored on one validation fold",
            "the final test split was not read by this run",
        ),
    )


def run_feature_set(
    *,
    dataset,
    feature_matrix: np.ndarray,
    feature_names: tuple[str, ...],
    target: str,
    feature_set: str,
    folds: tuple[Fold, ...],
    model: str,
    seed: int,
    validation_stop: int,
    horizon_steps: int,
    keep_timestamps: bool = True,
) -> list[FoldResult]:
    """Fit once on the train split, then score every fold.

    Fitting per fold would multiply the cost of the study by six for no gain: the model does
    not depend on the fold, only the scoring rows do. Measured on this dataset one
    ``classical_hist_gbm`` fit is about 254s against a ridge fit of 0.4s, so fitting five
    feature sets once each (21 minutes) rather than once per fold (two hours) is the
    difference between the classical ablation shipping and not.

    Every other quantity is still identical across folds, because they all come from the same
    fitted object and the same dataset.

    Args:
        dataset: The Phase 4 ``MlDataset``.
        feature_matrix: ``[n_samples, n_features]`` aligned with ``dataset``.
        feature_names: Column names of ``feature_matrix``.
        target: Target id, for labelling.
        feature_set: Feature set id, for labelling.
        folds: The folds to score, in order.
        model: Model id to fit.
        seed: The single frozen seed.
        validation_stop: One past the last validation origin.
        horizon_steps: The horizon this run measures.
        keep_timestamps: Retain per-row evidence.

    Returns:
        One :class:`FoldResult` per fold, in order.

    Raises:
        FinalTestAccessError: If any fold would reach past validation.
        ValueError: If the train split is empty.
    """
    for fold in folds:
        assert_final_test_denied(validation_stop, fold.stop)

    started = time.perf_counter()
    features = np.asarray(feature_matrix, dtype=np.float64)
    targets = np.asarray(dataset.targets[:, 0], dtype=np.float64)
    origins = np.asarray(dataset.origin_index, dtype=np.int64)
    series = np.asarray(dataset.series_index, dtype=np.int64)
    if features.shape[0] != origins.shape[0]:
        raise ValueError(
            f"feature matrix has {features.shape[0]} rows but the dataset has "
            f"{origins.shape[0]} samples; they must be aligned"
        )

    train_mask = dataset.mask(SPLIT_TRAIN)
    validation_mask = dataset.mask(SPLIT_VALIDATION)
    train_rows = int(np.count_nonzero(train_mask))
    if train_rows == 0:
        raise ValueError("the training split is empty; nothing to fit")

    finite_train = np.isfinite(features[train_mask]).all(axis=1)
    usable_train = train_mask.copy()
    usable_train[train_mask] = finite_train
    dropped = int(np.count_nonzero(train_mask) - np.count_nonzero(usable_train))

    model_object = fit_classical(
        model,
        features[usable_train],
        targets[usable_train],
        feature_names=feature_names,
        seed=int(seed),
    )
    fit_seconds = time.perf_counter() - started
    detail = dict(getattr(model_object, "detail", {}) or {})

    out: list[FoldResult] = []
    for fold in folds:
        mask = validation_mask & (origins >= fold.start) & (origins < fold.stop)
        n_scored = int(np.count_nonzero(mask))
        if n_scored == 0:
            raise ValueError(
                f"fold {fold.fold_id} [{fold.start}, {fold.stop}) contains no validation rows; "
                f"the fold definition and the dataset's origin range disagree"
            )
        prediction = np.asarray(model_object.predict(features[mask]), dtype=np.float64)
        actual = targets[mask]
        account = SampleAccounting(
            target=target,
            fold_id=fold.fold_id,
            feature_set=feature_set,
            origins=origins[mask],
            series=series[mask],
            dropped=dropped,
        )
        rows = None
        if keep_timestamps:
            rows = TimestampRows(
                forecast_origin=origins[mask].copy(),
                target_timestamp=origins[mask] + int(horizon_steps),
                series=series[mask].copy(),
                actual=actual.copy(),
                prediction=prediction.copy(),
            )
        out.append(
            FoldResult(
                target=target,
                feature_set=feature_set,
                fold_id=fold.fold_id,
                fold_role=fold.role,
                model=model,
                feature_names=tuple(feature_names),
                metrics={
                    "mae": float(mae(actual, prediction)),
                    "rmse": float(rmse(actual, prediction)),
                    "n": n_scored,
                    "unit": str(dataset.target_unit),
                },
                train_rows=train_rows,
                score_rows=n_scored,
                dropped=dropped,
                seconds=fit_seconds,
                account=account,
                timestamp_rows=rows,
                model_detail=detail,
                notes=(
                    "one fit per feature set, scored on every fold; the fold never changes "
                    "the model",
                    "fitted on the train split only",
                    "the final test split was not read by this run",
                ),
            )
        )
    return out
