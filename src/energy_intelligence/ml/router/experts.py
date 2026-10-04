"""The heterogeneous expert pool: persistence, classical GBM, temporal TCN.

Three predictors that already exist, verified, and are **deliberately not made
similar**. Their diversity is the entire point: persistence carries almost no
inductive assumption, the GBM is a nonlinear feature map, and the TCN is a learned
temporal convolution. Three copies of one architecture would be three models, not a
mixture of experts.

Every expert exposes the same interface and returns kilowatts::

    expert.predict(positions) -> [N, horizons] in kW

where ``positions`` index an :class:`EvaluationPanel`. Two rules this module enforces,
because getting either wrong produces a confident and wrong experiment:

**Units.** Every expert returns kW. The TCN's ``predict_series`` returns per-unit
levels, so the conversion happens here once rather than at each call site. Phase 6
shipped a bug where a per-unit prediction was compared against a kW target, giving a
13.3 kW error where the truth was 0.4178 kW.

**Fitting discipline.** Both learned experts are fitted on **training-split origins
only**, so their validation and test predictions are out-of-sample. That is what makes
validation rows legitimate router-training data: an expert's training-set predictions
are in-sample and unrealistically clean, and a router trained on them would learn from
information it will not have at inference time.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

__all__ = [
    "EvaluationPanel",
    "ExpertForecast",
    "ForecastExpert",
    "PersistenceExpert",
    "GbmExpert",
    "TcnExpert",
    "ExpertPool",
    "PHASE4_FEATURE_NAMES",
    "PHASE7_EXPERTS",
]

#: Phase 4's feature set, in Phase 4's column order. Reproduced verbatim so the GBM
#: expert is the model Phase 4 measured, not a lookalike.
PHASE4_FEATURE_NAMES: tuple[str, ...] = (
    "hour_of_day",
    "day_of_week",
    "day_of_year",
    "is_weekend",
    "value_at_origin",
    "lag_1",
    "lag_4",
    "lag_96",
    "lag_672",
    "roll_mean_96",
    "roll_std_96",
    "ramp_1",
    "roll_mean_same_hour_7d",
)

#: The pool, in reporting order.
PHASE7_EXPERTS: tuple[str, ...] = ("persistence", "classical_hist_gbm", "phase5_tcn")


@dataclass(frozen=True, slots=True)
class EvaluationPanel:
    """The rows every expert and the router are evaluated on, in one canonical order.

    A panel concatenates the validation and test origins and numbers them
    ``0 .. N-1``. Everything downstream - expert forecasts, router features, targets,
    oracle labels - uses those positions, so nothing has to agree on how to slice two
    different row arrays.

    Attributes:
        index_rows: ``[N]`` Phase 5 index row for each position.
        split: ``[N]`` which split each position belongs to.
        split_slices: Split name to a ``slice`` into the position axis.
        origins: ``[N]`` native origin index.
        series: ``[N]`` series index.
        row_scale: ``[N]`` the series' rated kW.
        actual_kw: ``[N, horizons]`` realised targets in kW.
        persistence_kw: ``[N, horizons]`` ``y(t)`` in kW.
        horizons: Horizon steps.
    """

    index_rows: np.ndarray
    split: np.ndarray
    split_slices: dict[str, slice]
    origins: np.ndarray
    series: np.ndarray
    row_scale: np.ndarray
    actual_kw: np.ndarray
    persistence_kw: np.ndarray
    horizons: tuple[int, ...]

    @property
    def n_rows(self) -> int:
        return int(self.index_rows.size)

    def split_mask(self, name: str) -> np.ndarray:
        return self.split == name

    def describe(self) -> dict[str, Any]:
        return {
            "n_rows": self.n_rows,
            "series": int(np.unique(self.series).size),
            "splits": {
                name: int(self.split_slices[name].stop - self.split_slices[name].start)
                for name in self.split_slices
            },
            "horizons": list(self.horizons),
        }


@dataclass(frozen=True, slots=True)
class ExpertForecast:
    """One expert's output for a set of positions.

    Attributes:
        name: Expert name.
        forecast_kw: ``[N, horizons]`` in kW.
        horizons: The horizon steps, in column order.
        provenance: Where the model came from and what it was fitted on.
    """

    name: str
    forecast_kw: np.ndarray
    horizons: tuple[int, ...]
    provenance: dict[str, Any] = field(default_factory=dict)

    @property
    def n_rows(self) -> int:
        return int(self.forecast_kw.shape[0])

    def __post_init__(self) -> None:
        if self.forecast_kw.ndim != 2:
            raise ValueError(
                f"expert {self.name!r} must return [rows, horizons], got "
                f"{self.forecast_kw.shape}"
            )
        if self.forecast_kw.shape[1] != len(self.horizons):
            raise ValueError(
                f"expert {self.name!r} returned {self.forecast_kw.shape[1]} columns for "
                f"{len(self.horizons)} horizons"
            )


@runtime_checkable
class ForecastExpert(Protocol):
    """The routing interface every expert satisfies."""

    name: str

    def predict(self, positions: np.ndarray) -> ExpertForecast:
        """Forecast for panel ``positions``, in kW."""
        ...


class PersistenceExpert:
    """Expert 1: persistence, ``yhat(t+h) = y(t)``.

    Phase 4's definition, unchanged. It holds no fitted state, so it cannot leak, and
    it is the strongest expert at short horizons by a wide margin - which is precisely
    why a router wants it in the pool.
    """

    name = "persistence"

    def __init__(self, panel: EvaluationPanel) -> None:
        self._panel = panel

    def predict(self, positions: np.ndarray) -> ExpertForecast:
        positions = np.asarray(positions, dtype=np.int64)
        return ExpertForecast(
            name=self.name,
            forecast_kw=self._panel.persistence_kw[positions],
            horizons=self._panel.horizons,
            provenance={"fitted": False, "definition": "yhat(t+h) = y(t)"},
        )


class GbmExpert:
    """Expert 2: Phase 4's classical histogram gradient boosting.

    Takes already-fitted estimators so it cannot accidentally refit on data it should
    not see. One estimator per horizon, exactly as Phase 4 ran them. Features are
    supplied already scaled and context-augmented in panel order, so ``predict`` is a
    row gather.
    """

    name = "classical_hist_gbm"

    def __init__(
        self,
        estimators: dict[int, Any],
        panel_features: np.ndarray,
        horizons: tuple[int, ...],
        row_scale: np.ndarray,
        *,
        provenance: dict[str, Any] | None = None,
    ) -> None:
        missing = [h for h in horizons if h not in estimators]
        if missing:
            raise ValueError(f"the GBM expert has no estimator for horizon(s) {missing}")
        self._estimators = dict(estimators)
        self._features = np.asarray(panel_features)
        self._horizons = tuple(horizons)
        self._row_scale = np.asarray(row_scale, dtype=np.float64)
        self._provenance = dict(provenance or {})

    def predict(self, positions: np.ndarray) -> ExpertForecast:
        positions = np.asarray(positions, dtype=np.int64)
        if positions.size and int(positions.max()) >= self._features.shape[0]:
            raise IndexError(
                f"panel positions reach {int(positions.max())} but the GBM feature "
                f"matrix has {self._features.shape[0]} rows"
            )
        matrix = self._features[positions]
        columns = [
            self._estimators[horizon].predict(matrix)[:, None]
            for horizon in self._horizons
        ]
        # Phase 4 fitted on per-unit targets, so the estimate is per-unit too.
        return ExpertForecast(
            name=self.name,
            forecast_kw=np.concatenate(columns, axis=1) * self._row_scale[positions, None],
            horizons=self._horizons,
            provenance=self._provenance,
        )


class TcnExpert:
    """Expert 3: Phase 5's temporal convolutional network, from its real checkpoint.

    The checkpoint Phase 5 committed its metrics from, loaded and applied here - not a
    retrained approximation. The question is which expert is best, and it is only
    answered by the experts that actually exist.

    ``predict_series`` returns **per-unit** levels; conversion to kW happens here so
    every expert in the pool speaks the same units.
    """

    name = "phase5_tcn"

    def __init__(
        self,
        model: Any,
        panel: EvaluationPanel,
        values: np.ndarray,
        index: Any,
        scaler: Any,
        *,
        batch_size: int = 8192,
        provenance: dict[str, Any] | None = None,
    ) -> None:
        self._model = model
        self._panel = panel
        self._values = values
        self._index = index
        self._scaler = scaler
        self._batch_size = int(batch_size)
        self._provenance = dict(provenance or {})

    def predict(self, positions: np.ndarray) -> ExpertForecast:
        from ..training import predict_series

        positions = np.asarray(positions, dtype=np.int64)
        index_rows = self._panel.index_rows[positions]
        _actual_pu, predicted_pu, _persistence_pu = predict_series(
            self._model, self._values, self._index, index_rows,
            scaler=self._scaler, batch_size=self._batch_size,
        )
        scale = self._panel.row_scale[positions]
        return ExpertForecast(
            name=self.name,
            forecast_kw=predicted_pu * scale[:, None],
            horizons=self._panel.horizons,
            provenance=self._provenance,
        )


class ExpertPool:
    """The three experts, asked once and stacked.

    A single ``[n_experts, N, horizons]`` block means the diversity analysis, the oracle
    and the router all consume the *same* predictions, rather than three subtly
    different recomputations of them.
    """

    def __init__(self, experts: list[ForecastExpert], horizons: tuple[int, ...]) -> None:
        if not experts:
            raise ValueError("an expert pool needs at least one expert")
        names = [e.name for e in experts]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate expert names: {names}")
        self._experts = list(experts)
        self._horizons = tuple(horizons)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(e.name for e in self._experts)

    @property
    def horizons(self) -> tuple[int, ...]:
        """The horizon steps every expert in this pool is built for."""
        return self._horizons

    @property
    def n_experts(self) -> int:
        return len(self._experts)

    def forecast_block(
        self, positions: np.ndarray, *, log: Any = None
    ) -> tuple[np.ndarray, dict[str, float]]:
        """``[n_experts, N, horizons]`` in kW, plus each expert's wall time."""
        positions = np.asarray(positions, dtype=np.int64)
        blocks: list[np.ndarray] = []
        timings: dict[str, float] = {}
        for expert in self._experts:
            started = time.perf_counter()
            forecast = expert.predict(positions)
            elapsed = time.perf_counter() - started
            timings[expert.name] = round(elapsed, 3)
            if forecast.horizons != self._horizons:
                raise ValueError(
                    f"expert {expert.name!r} was built for horizons {forecast.horizons}, "
                    f"pool expects {self._horizons}"
                )
            if forecast.n_rows != positions.size:
                raise ValueError(
                    f"expert {expert.name!r} returned {forecast.n_rows} rows for "
                    f"{positions.size} requested"
                )
            blocks.append(forecast.forecast_kw)
            if log is not None:
                log(f"    {expert.name}: {elapsed:.1f}s for {positions.size} rows")
        stacked = np.stack(blocks, axis=0)
        if not np.isfinite(stacked).all():
            offenders = [
                self.names[i] for i in np.unique(np.where(~np.isfinite(stacked))[0])
            ]
            raise ValueError(f"expert(s) produced non-finite forecasts: {offenders}")
        return stacked, timings

    def provenance(self) -> dict[str, Any]:
        return {e.name: getattr(e, "_provenance", {}) for e in self._experts}


def build_panel(
    *,
    index: Any,
    values: np.ndarray,
    scale_vector: np.ndarray,
    splits: tuple[str, ...] = ("validation", "test"),
) -> EvaluationPanel:
    """Assemble the canonical evaluation panel from the Phase 5 index.

    Args:
        index: The Phase 5 sequence index.
        values: Per-unit values.
        scale_vector: Rated kW per series.
        splits: Which splits to include, in order.

    Returns:
        The panel, with validation first and test second.
    """
    from ..sequence import gather_sequences

    codes = {"train": 0, "validation": 1, "test": 2}
    blocks = [index.rows(codes[name]) for name in splits]
    index_rows = np.concatenate(blocks)
    _x, y_pu, persistence_pu = gather_sequences(values, index, index_rows)
    row_scale = scale_vector[index.series[index_rows]]
    split_names = np.concatenate([np.full(b.size, name) for name, b in zip(splits, blocks)])
    slices: dict[str, slice] = {}
    cursor = 0
    for name, block in zip(splits, blocks):
        slices[name] = slice(cursor, cursor + block.size)
        cursor += block.size
    return EvaluationPanel(
        index_rows=index_rows,
        split=split_names,
        split_slices=slices,
        origins=index.origins[index_rows],
        series=index.series[index_rows],
        row_scale=row_scale,
        actual_kw=y_pu * row_scale[:, None],
        persistence_kw=persistence_pu * row_scale[:, None],
        horizons=tuple(index.config.horizons),
    )


def _tree_size(estimator: Any) -> int:
    """Total nodes across every tree in a fitted gradient-boosted ensemble.

    Reported for the cost table because a boosted tree stores no weight matrix, and its
    honest size measure is node count. sklearn exposes the per-iteration predictors, so
    this sums them; a model shape that does not expose predictors falls back to zero and
    says so through the absence of a number rather than by guessing.
    """
    predictors = getattr(estimator, "_predictors", None)
    if not predictors:
        return 0
    total = 0
    for iteration in predictors:
        # Each iteration holds (per_feature, predictor) pairs for multi-output models.
        for item in iteration:
            predictor = item[1] if isinstance(item, tuple) else item
            nodes = getattr(predictor, "nodes", None)
            if nodes is not None:
                # ``nodes`` is a structured array with one entry per tree node, so its
                # length is the node count. ``n_nodes`` does not exist on this type.
                total += len(nodes)
    return total


def _phase4_series_context(
    series: np.ndarray,
    train_target_pu: np.ndarray,
    *,
    series_count: int,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Phase 4's three per-series context columns, rebuilt from training rows.

    ``series_index``, ``series_level`` and ``series_volatility``, in that order. This is
    ``experiment._series_context`` expressed over a bare array instead of an
    ``MlDataset``, because Phase 7 indexes the Phase 5 sequence grid rather than a
    Phase 4 dataset artifact. The two fitted statistics come from the supplied
    **training** rows only, so a test row never contributes to its own feature.

    Args:
        series: ``[n_train_rows]`` series index of each training row.
        train_target_pu: ``[n_train_rows]`` per-unit target at the first horizon.
        series_count: Total series in the panel.

    Returns:
        ``(context_matrix, context_names)``.
    """
    series = np.asarray(series, dtype=np.int64)
    level = np.zeros(series_count, dtype=np.float64)
    volatility = np.zeros(series_count, dtype=np.float64)
    for index in range(series_count):
        rows = series == index
        if not np.any(rows):
            continue
        values = np.asarray(train_target_pu, dtype=np.float64)[rows]
        level[index] = float(values.mean())
        volatility[index] = float(values.std())
    columns = (
        series.astype(np.float64),
        level[series],
        volatility[series],
    )
    return np.stack(columns, axis=1), ("series_index", "series_level", "series_volatility")


def build_experts(
    *,
    panel: EvaluationPanel,
    index: Any,
    index_stride1: Any,
    values: np.ndarray,
    scale_vector: np.ndarray,
    tcn_checkpoint: Path,
    seed: int,
    horizons: tuple[int, ...] | None = None,
    log: Any = None,
) -> tuple[ExpertPool, dict[str, Any]]:
    """Fit the learned experts on training origins and assemble the pool.

    Args:
        panel: The evaluation panel; its ``horizons`` are used unless overridden.
        index: The Phase 5 index used for prediction.
        index_stride1: An equivalent index with ``train_origin_stride=1``, whose training
            rows are Phase 4's full 950,400-row training set.
        values: Per-unit values.
        scale_vector: Rated kW per series.
        tcn_checkpoint: Phase 5's ``checkpoint.pt``.
        seed: Estimator seed.
        horizons: Override the panel's horizons.
        log: Optional progress callable.

    Returns:
        ``(pool, provenance)``.
    """
    import torch
    from ..baselines.classical import fit_classical
    from ..dataset import build_feature_matrix
    from ..phase5 import _window_scaler
    from ..scaling import FeatureScaler, assert_train_only_fit
    from ..sequence import gather_sequences
    from ..temporal import build_model, model_defaults

    horizons = tuple(horizons or panel.horizons)
    provenance: dict[str, Any] = {"horizons": list(horizons)}

    # ---- GBM, Phase 4's model reproduced exactly ---------------------------
    # Phase 4 fitted its HistGBM on **per-unit** targets and **scaled** features,
    # with three per-series context columns whose two statistics come from training
    # rows only. Reproducing the model means reproducing all of that: fitting on kW
    # targets, or on unscaled features, produces a model that is not Phase 4's, and
    # the first attempt at this did exactly that and scored 10.1 kW where Phase 4
    # scored 0.42 kW.
    train_rows = index_stride1.rows(0)
    started = time.perf_counter()
    lookback = index_stride1.config.lookback_steps
    train_raw = build_feature_matrix(
        values,
        index_stride1.origins[train_rows],
        lookback=lookback,
        feature_names=PHASE4_FEATURE_NAMES,
        series_index=index_stride1.series[train_rows],
    ).astype(np.float64)
    scaler = FeatureScaler.fit(train_raw)
    assert_train_only_fit(scaler, int(train_rows.size))
    train_features = scaler.transform(train_raw)
    del train_raw

    _xt, y_train_pu, _bt = gather_sequences(values, index_stride1, train_rows)
    context, context_names = _phase4_series_context(
        index_stride1.series[train_rows],
        y_train_pu[:, horizons.index(horizons[0])],
        series_count=values.shape[0],
    )
    train_features = np.concatenate([train_features, context], axis=1)
    feature_names = PHASE4_FEATURE_NAMES + context_names

    estimators: dict[int, Any] = {}
    fit_seconds = 0.0
    for column, horizon in enumerate(horizons):
        fitted = fit_classical(
            "classical_hist_gbm",
            train_features,
            y_train_pu[:, column],
            feature_names=feature_names,
            seed=seed,
        )
        estimators[horizon] = fitted.estimator
        fit_seconds += fitted.fit_seconds
        if log is not None:
            log(
                f"  gbm h={horizon}: fit {fitted.fit_seconds:.1f}s "
                f"on {train_rows.size} training rows"
            )
    sizes = [_tree_size(estimator) for estimator in estimators.values()]
    del train_features

    panel_raw = build_feature_matrix(
        values,
        panel.origins,
        lookback=index.config.lookback_steps,
        feature_names=PHASE4_FEATURE_NAMES,
        series_index=panel.series,
    ).astype(np.float64)
    panel_features = np.concatenate(
        [scaler.transform(panel_raw), context[panel.series]], axis=1
    )
    del panel_raw
    provenance["classical_hist_gbm"] = {
        "fit_seconds": round(fit_seconds, 1),
        "total_seconds": round(time.perf_counter() - started, 1),
        "train_rows": int(train_rows.size),
        "n_features": len(feature_names),
        "feature_names": list(feature_names),
        "fitted_on": "Phase 4's full training split, train_origin_stride=1",
        "target_units": "per unit of the series scale, as Phase 4 stored them",
        "params": "Phase 4 defaults: max_iter=200, lr=0.1, 31 leaves, min_samples_leaf=20",
        # A boosted tree has no learned weight vector, so "parameters" is reported as
        # node counts and labelled as such rather than compared to the TCN's tensor
        # count as if the two numbers meant the same thing.
        "size_measure": "total tree nodes across all boosting iterations and all horizons",
        "tree_nodes": int(sum(sizes)),
        "trees_per_horizon": int(next(iter(estimators.values())).n_iter_),
    }
    if log is not None:
        log(f"  gbm ready in {time.perf_counter() - started:.1f}s")

    gbm = GbmExpert(
        estimators,
        panel_features,
        horizons,
        panel.row_scale,
        provenance=provenance["classical_hist_gbm"],
    )

    # ---- TCN, from the real Phase 5 checkpoint -----------------------------
    started = time.perf_counter()
    payload = torch.load(tcn_checkpoint, weights_only=False)
    model = build_model(
        "tcn",
        in_channels=index.config.channel_count,
        horizons=index.config.horizons,
        series_count=values.shape[0],
        params=payload.get("config", model_defaults("tcn")),
    )
    model.load_state_dict(payload["state_dict"])
    model.eval()
    scaler = _window_scaler(values, index, index.rows(0))
    provenance["phase5_tcn"] = {
        "checkpoint": str(tcn_checkpoint),
        "best_epoch": payload.get("best_epoch"),
        "best_validation_mae_per_unit": payload.get("best_validation_mae_per_unit"),
        "parameters": int(sum(p.numel() for p in model.parameters())),
        "load_and_scaler_seconds": round(time.perf_counter() - started, 1),
    }
    if log is not None:
        log(f"  tcn ready in {time.perf_counter() - started:.1f}s")

    tcn = TcnExpert(
        model, panel, values, index, scaler, provenance=provenance["phase5_tcn"]
    )
    pool = ExpertPool(
        [PersistenceExpert(panel), gbm, tcn],
        panel.horizons,
    )
    provenance["panel"] = panel.describe()
    provenance["index_version"] = index.version
    return pool, provenance
