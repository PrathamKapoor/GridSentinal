"""Run one baseline experiment end to end, reproducibly.

Order of operations, and why each step is where it is:

1. **Load the dataset** from its versioned artifact, or build it if asked. Nothing
   downstream re-derives features.
2. **Split** by the dataset's own chronological codes. No shuffling anywhere.
3. **Fit the scaler on training rows only**, then transform all three splits.
   :func:`~energy_intelligence.ml.scaling.assert_train_only_fit` proves it.
4. **Fit or evaluate the model.** Naive rules are not fitted: they read the series
   itself, which is why they are given the real history rather than approximated
   from feature columns. An approximation would be wrong by up to ``h`` steps and
   quietly flatter or embarrass a baseline, so the runner refuses to guess.
5. **Evaluate** every declared horizon on the test split. Validation metrics are
   produced for reporting; model selection never happens on test.
6. **Record** the experiment, append it to the registry, and write artifacts.

Determinism
-----------
Seeds are set for numpy and torch. The dataset version, feature list and split
counts are recorded in every record, so two records with the same
``experiment_id`` must have identical metrics; :func:`verify_reproducible` checks
that directly rather than assuming it.

The ramp metric
---------------
A ramp is a change between *consecutive* time steps, so it is only computable when
the dataset declares two consecutive horizons. When the horizon set is e.g.
``(1, 4, 96)`` there is no consecutive pair and the ramp metric is reported as
``n/a`` rather than approximated from a 3-step gap.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .baselines.classical import fit_classical
from .baselines.naive import naive_forecast
from .baselines.neural import fit_neural
from .dataset import MlDataset
from .metrics import evaluate, mae
from .registry import (
    ExperimentRecord,
    append_record,
    environment,
    experiment_id,
    now_iso,
)
from .scaling import FeatureScaler, assert_train_only_fit
from .splits import SPLIT_TEST, SPLIT_TRAIN, SPLIT_VALIDATION

__all__ = [
    "ExperimentResult",
    "run_experiment",
    "compare",
    "verify_reproducible",
    "record_for",
    "ALL_MODELS",
]

#: Every model this phase evaluates, in comparison-table order.
ALL_MODELS: tuple[str, ...] = (
    "naive_last_value",
    "naive_seasonal_day",
    "naive_seasonal_week",
    "naive_drift",
    "classical_ridge",
    "classical_hist_gbm",
    "neural_mlp",
    "neural_gru",
)


@dataclass(frozen=True, slots=True)
class ExperimentResult:
    """One model's measured result on one dataset at every declared horizon."""

    experiment_id: str
    model: str
    model_family: str
    metrics_by_horizon: dict[int, dict[str, Any]]
    train_seconds: float
    predict_seconds: float
    model_detail: dict[str, Any]
    notes: tuple[str, ...] = ()

    def render(self) -> str:
        """Lines for the human comparison table."""
        lines = [f"{self.model} ({self.model_family})"]
        for horizon in sorted(self.metrics_by_horizon):
            payload = self.metrics_by_horizon[horizon]
            lines.append(
                f"  h={horizon:<4d} n={payload['n']:<8d} MAE={payload['mae']:>10.3f} "
                f"RMSE={payload['rmse']:>10.3f} sMAPE={payload['smape']:>6.2f}% "
                f"R2={_fmt(payload['r2']):>7s} peak={_fmt(payload['peak_error_mae']):>9s} "
                f"ramp={_fmt(payload['ramp_error_mae']):>9s}"
            )
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "model": self.model,
            "model_family": self.model_family,
            "metrics_by_horizon": {str(k): v for k, v in self.metrics_by_horizon.items()},
            "train_seconds": round(self.train_seconds, 3),
            "predict_seconds": round(self.predict_seconds, 3),
            "model_detail": self.model_detail,
            "notes": list(self.notes),
        }


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float) and value != value:
        return "n/a"
    return f"{float(value):.3f}"


def _family_of(model: str) -> str:
    from .baselines.classical import CLASSICAL_MODELS
    from .baselines.naive import NAIVE_MODELS
    from .baselines.neural import NEURAL_MODELS

    if model in NAIVE_MODELS or model == "naive_drift":
        return "naive"
    if model in CLASSICAL_MODELS:
        return "classical"
    if model in NEURAL_MODELS:
        return "neural"
    raise KeyError(
        f"unknown model {model!r}; naive={list(NAIVE_MODELS)}, "
        f"classical={list(CLASSICAL_MODELS)}, neural={list(NEURAL_MODELS)}"
    )


def run_experiment(
    dataset: MlDataset,
    *,
    model: str,
    seed: int,
    series: np.ndarray | None = None,
    threads: int = 4,
) -> ExperimentResult:
    """Evaluate one model on one dataset.

    Args:
        dataset: The windowed dataset.
        model: One of :data:`ALL_MODELS`.
        seed: Seed for stochastic models; recorded even when unused.
        series: The underlying ``(n_series, n_steps)`` history. **Required for naive
            models**, which read it directly so their seasonal offsets are exact.
        threads: Torch CPU threads, recorded because it changes runtime.

    Returns:
        The measured result for every horizon in the dataset.

    Raises:
        KeyError: If the model name is unknown.
        ValueError: If the training split is empty, or a naive model was requested
            without the underlying series.
    """
    family = _family_of(model)
    train_mask = dataset.mask(SPLIT_TRAIN)
    test_mask = dataset.mask(SPLIT_TEST)
    if not np.any(train_mask):
        raise ValueError("training split is empty; cannot fit or evaluate a baseline")
    if family == "naive" and series is None:
        raise ValueError(
            f"{model} is a naive rule and must be given the real series history; "
            "approximating it from feature columns would misalign it by up to h "
            "steps and change the result"
        )

    features = dataset.features.astype(np.float64)
    row_scale = dataset.series_scale.astype(np.float64)[
        dataset.series_index.astype(np.int64)
    ]
    notes: list[str] = []
    train_seconds = 0.0
    predict_seconds = 0.0
    model_detail: dict[str, Any] = {}
    actual_by_horizon: dict[int, np.ndarray] = {}
    predicted_by_horizon: dict[int, np.ndarray] = {}

    if family == "naive":
        notes.append(
            "unfitted reference rule evaluated on the test split; predictions come "
            "from the real series, so no training or scaling is involved"
        )
        model_detail = {"name": model, "fitted": False}
        started = time.perf_counter()
        # ``series`` is (n_series, n_steps); ``dataset.series_scale`` is per series.
        normalised_series = np.asarray(series, dtype=np.float64) / dataset.series_scale.reshape(-1, 1)
        actual_by_horizon, predicted_by_horizon = _naive_predictions(
            dataset, model, test_mask, normalised_series
        )
        predict_seconds += time.perf_counter() - started
        notes.append(
            "the series was divided by each row's scale before the rule was applied, "
            "matching how the dataset stores it"
        )

    else:
        scaler = FeatureScaler.fit(features[train_mask])
        assert_train_only_fit(scaler, int(np.count_nonzero(train_mask)))
        scaled = scaler.transform(features)
        notes.append("scaler fitted on training rows only; validation and test transformed unchanged")

        # Per-series context. A pooled model sees one row per (customer, origin) and
        # has no way to tell a 1.4 kW household from a 388 kW shop, so the calendar
        # terms end up absorbing cross-customer differences. These columns restore
        # that identity using the series' own identity plus two statistics computed
        # **from training rows only**.
        #
        # They are created here rather than stored in the dataset, because they
        # depend on the split: a dataset artifact that embedded split-dependent
        # columns could not be reused under a different split.
        context, context_names = _series_context(dataset, train_mask, features)
        scaled = np.concatenate([scaled, context], axis=1)
        feature_names = dataset.feature_names + context_names
        notes.append(
            "per-series context columns added at fit time from training rows only: "
            f"{list(context_names)}"
        )

        if family == "neural" and model == "neural_gru":
            x_train, x_test = _lookback_window(dataset, train_mask, test_mask)
        else:
            x_train, x_test = scaled[train_mask], scaled[test_mask]

        for horizon in dataset.horizons:
            target = dataset.target_column(horizon).astype(np.float64)
            if family == "classical":
                fitted = fit_classical(
                    model,
                    x_train,
                    target[train_mask],
                    feature_names=feature_names,
                    seed=seed,
                )
            else:
                fitted = fit_neural(
                    model, x_train, target[train_mask], seed=seed, threads=threads
                )
            started = time.perf_counter()
            predicted = fitted.predict(x_test)
            predict_seconds += time.perf_counter() - started
            train_seconds += fitted.fit_seconds
            actual_by_horizon[horizon] = target[test_mask]
            predicted_by_horizon[horizon] = predicted
            if horizon == dataset.horizons[0]:
                model_detail = fitted.to_dict()
            if "scaler" not in model_detail:
                model_detail["scaler"] = scaler.to_dict()
            model_detail["feature_names"] = list(feature_names)

    # The dataset stores per-unit values; every reported number is converted back to
    # kW by multiplying by the row's own scale, so the table is readable in the unit
    # an energy engineer would use.
    scale = row_scale[test_mask]
    metrics_by_horizon: dict[int, dict[str, Any]] = {}
    for horizon in dataset.horizons:
        actual_kw = actual_by_horizon[horizon] * scale
        predicted_kw = predicted_by_horizon[horizon] * scale
        ramp_actual = None
        ramp_predicted = None
        if horizon + 1 in dataset.horizons:
            ramp_actual = np.stack(
                [actual_by_horizon[horizon], actual_by_horizon[horizon + 1]], axis=1
            )
            ramp_predicted = np.stack(
                [predicted_by_horizon[horizon], predicted_by_horizon[horizon + 1]], axis=1
            )
        metrics = evaluate(
            actual_kw,
            predicted_kw,
            ramp_actual=None if ramp_actual is None else ramp_actual * scale[:, None],
            ramp_predicted=None if ramp_predicted is None else ramp_predicted * scale[:, None],
        )
        payload = metrics.to_dict()
        payload["unit"] = dataset.target_unit
        payload["mae_per_unit"] = mae(
            actual_by_horizon[horizon], predicted_by_horizon[horizon]
        )
        payload["storage"] = "per unit of the series scale"
        metrics_by_horizon[horizon] = payload

    identifier = experiment_id(
        target_id=dataset.target_id,
        dataset_version=dataset.version,
        model=model,
        horizon=0,
    )
    return ExperimentResult(
        experiment_id=identifier,
        model=model,
        model_family=family,
        metrics_by_horizon=metrics_by_horizon,
        train_seconds=train_seconds,
        predict_seconds=predict_seconds,
        model_detail=model_detail,
        notes=tuple(notes),
    )


def _series_context(
    dataset: MlDataset, train_mask: np.ndarray, features: np.ndarray
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Per-series identity columns, with two statistics fitted on training rows.

    Columns, in order:

    ``series_index``
        Which series the row belongs to. Known at prediction time and legitimate as
        a categorical, though a tree can only split it ordinally - which is why the
        two statistics below matter more.
    ``series_level``
        Mean of that series' per-unit target over **training rows only**.
    ``series_volatility``
        Standard deviation of the same, over training rows only.

    The last two are fitted statistics, so they are computed once from the training
    mask and then applied unchanged to every split. A test row's value never
    contributes to its own feature.

    Returns:
        The column matrix and its names, in order.
    """
    series = dataset.series_index.astype(np.int64)
    n_series = len(dataset.series_ids)
    target = dataset.target_column(dataset.horizons[0]).astype(np.float64)

    level = np.zeros(n_series, dtype=np.float64)
    volatility = np.zeros(n_series, dtype=np.float64)
    for index in range(n_series):
        rows = train_mask & (series == index)
        if not np.any(rows):
            # A series with no training rows cannot have a fitted statistic; zeros
            # are used and the omission is visible because the model still sees the
            # identity column. This cannot happen with the shipped splits, which are
            # chronological and every series spans all of them.
            continue
        values = target[rows]
        level[index] = float(values.mean())
        volatility[index] = float(values.std())

    columns = (
        series.astype(np.float64),
        level[series],
        volatility[series],
    )
    return (
        np.stack(columns, axis=1),
        ("series_index", "series_level", "series_volatility"),
    )


def _naive_predictions(
    dataset: MlDataset,
    model: str,
    test_mask: np.ndarray,
    series: np.ndarray,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Exact naive forecasts for every test row, using the real history.

    Predictions are computed per series so the seasonal offsets are exact: for
    horizon ``h`` the seasonal-naive rule reads ``t + h - 96`` (daily) or
    ``t + h - 672`` (weekly).
    """
    series_values = np.asarray(series, dtype=np.float64)
    origins = dataset.origin_index.astype(np.int64)[test_mask]
    rows = dataset.series_index.astype(np.int64)[test_mask]

    predicted = np.empty((origins.size, len(dataset.horizons)), dtype=np.float64)
    for row in np.unique(rows):
        mask = rows == row
        predicted[mask] = naive_forecast(
            series_values[row], origins[mask], dataset.horizons, model
        )

    actual_by_horizon: dict[int, np.ndarray] = {}
    predicted_by_horizon: dict[int, np.ndarray] = {}
    for position, horizon in enumerate(dataset.horizons):
        actual_by_horizon[horizon] = dataset.target_column(horizon).astype(np.float64)[test_mask]
        predicted_by_horizon[horizon] = predicted[:, position]
    return actual_by_horizon, predicted_by_horizon


def _lookback_window(
    dataset: MlDataset, train_mask: np.ndarray, test_mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """The raw lag window the GRU baseline consumes.

    The dataset is tabular, so the window is assembled from the pure lag columns it
    already contains. Only genuine lags of the target are used - no rolling means,
    no calendar columns - so the sequence model sees the same underlying history a
    sequence model would normally be given, at a width the feature set can support.
    """
    lag_columns = ("lag_1", "lag_4", "lag_96", "lag_672")
    missing = [name for name in lag_columns if name not in dataset.feature_names]
    if missing:
        raise ValueError(
            f"the GRU baseline needs the lag columns {list(lag_columns)}, but this "
            f"dataset was built without {missing}"
        )
    matrix = np.stack(
        [
            dataset.features[:, dataset.feature_names.index(name)].astype(np.float32)
            for name in lag_columns
        ],
        axis=1,
    )
    return matrix[train_mask], matrix[test_mask]


def compare(results: list[ExperimentResult]) -> str:
    """Render a model comparison table.

    Models appear in the order supplied and metrics in a fixed column order. No
    ranking or composite score is produced: one number that silently weights MAE
    against peak error would be an invented criterion.
    """
    header = (
        f"{'model':<24s} {'h':>4s} {'n':>8s} {'MAE':>11s} {'RMSE':>11s} "
        f"{'sMAPE%':>8s} {'R2':>8s} {'peak':>10s} {'ramp':>10s} {'train_s':>8s}"
    )
    lines = [header, "-" * len(header)]
    for result in results:
        for horizon in sorted(result.metrics_by_horizon):
            payload = result.metrics_by_horizon[horizon]
            lines.append(
                f"{result.model:<24s} {horizon:>4d} {payload['n']:>8d} "
                f"{payload['mae']:>11.4f} {payload['rmse']:>11.4f} "
                f"{payload['smape']:>8.2f} {_fmt(payload['r2']):>8s} "
                f"{_fmt(payload['peak_error_mae']):>10s} "
                f"{_fmt(payload['ramp_error_mae']):>10s} {result.train_seconds:>8.2f}"
            )
    return "\n".join(lines)


def verify_reproducible(
    dataset: MlDataset,
    *,
    model: str,
    seed: int,
    series: np.ndarray | None = None,
    tolerance: float = 1e-9,
) -> tuple[bool, str]:
    """Re-run one model and confirm the metrics are identical.

    Returns:
        ``(ok, detail)``. A determinism failure is reported rather than raised,
        because it is a finding about the experiment, not a crash.
    """
    first = run_experiment(dataset, model=model, seed=seed, series=series)
    second = run_experiment(dataset, model=model, seed=seed, series=series)
    for horizon in first.metrics_by_horizon:
        for key, value in first.metrics_by_horizon[horizon].items():
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                continue
            other = second.metrics_by_horizon[horizon][key]
            if value != other and abs(value - other) > tolerance:
                return False, f"{model} h={horizon} {key}: {value} then {other}"
    return True, f"{model} reproduced exactly across two runs (seed {seed})"


def record_for(
    result: ExperimentResult,
    dataset: MlDataset,
    *,
    model: str,
    seed: int | None,
    dataset_sha256: str,
    split_fractions: dict[str, float],
    split_counts: dict[str, int],
    artifacts: dict[str, str],
    lookback_steps: int,
    horizon_steps: int,
    hyperparams: dict[str, Any],
) -> ExperimentRecord:
    """Build the registry record for one experiment."""
    return ExperimentRecord(
        experiment_id=f"{dataset.target_id}-{dataset.version}-{model}-h{horizon_steps:03d}",
        created_at=now_iso(),
        target_id=dataset.target_id,
        dataset_version=dataset.version,
        dataset_sha256=dataset_sha256,
        feature_names=dataset.feature_names,
        lookback_steps=lookback_steps,
        horizon_steps=horizon_steps,
        split_fractions=split_fractions,
        split_counts=split_counts,
        model=model,
        model_family=result.model_family,
        hyperparameters=hyperparams,
        seed=seed,
        train_seconds=result.train_seconds,
        predict_seconds=result.predict_seconds,
        metrics=result.metrics_by_horizon,
        artifacts=artifacts,
        environment=environment(),
        notes=result.notes,
    )


def persist_record(
    record: ExperimentRecord, registry_path: Path
) -> Path:
    """Append a record to the registry, replacing any earlier one with the same id."""
    return append_record(registry_path, record)
