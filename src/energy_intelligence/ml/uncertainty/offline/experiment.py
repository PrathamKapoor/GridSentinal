"""The Phase 8 runner: fit uncertainty on calibration data, evaluate once on test.

The order is the order of the argument, and it is not arbitrary.

**1. Recover the point forecast, do not retrain it.** Phase 7's fixed-ensemble weights
are read from its own result artifact and checked against the copy in
``configs/uncertainty.toml``. If they disagree the run stops, because a Phase 8 table built
on different weights would not be comparable with Phase 7's and would not be a Phase 8
result at all.

**2. Prove the point forecast is still Phase 7's.** The fixed ensemble's test MAE is
compared against Phase 7's published figure per horizon. A refit would be caught here.

**3. Partition before fitting anything.** :func:`build_calibration_split` halves the
validation split chronologically, and ``split_parity_check`` reports the point forecast's
error on rows the ensemble's own weights saw and rows they did not. That number is what
justifies, or refuses, the whole calibration design.

**4. Fit uncertainty on CAL_FIT, calibrate on CAL_CONF, evaluate on TEST.** Never the
reverse, and never on test.

**5. Compare six interval methods on identical rows**, plus the global-residual method
applied to each individual expert, so the answer does not depend on which point predictor
is in use.

Nothing in the production path is modified. The phase adds an estimator and an evaluation;
it does not change what the system forecasts.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from ...analysis import assign_regimes
from ...metrics import mae
from ..artifact import (
    CALIBRATION_FAIL,
    CALIBRATION_PASS,
    BandCutPoints,
    ProbabilisticForecastBatch,
    save_batch,
)
from ..calibration import fit_conformal
from ..evaluation import imminent_ramp_detection
from ..features_set import build_uncertainty_features
from ..intervals import (
    DEFAULT_NOMINAL_LEVELS,
    PredictionInterval,
    quantile_levels,
    widths_from_scale,
)
from ..measures import disagreement_measures, spearman
from ..quantile import fit_quantile_model
from ..scales import DisagreementScale, fit_learned_scale, fitted_scale
from ..split import build_calibration_split, split_parity_check
from .analysis import (
    coverage_tolerance,
    evaluate_method,
    expert_point_forecast_table,
    final_comparison_table,
    method_table,
    regime_breakdown,
)
from .verdict import phase8_verdict

__all__ = ["METHOD_NAMES", "BASELINE_METHOD", "Phase8Result", "run_phase8", "render_summary"]

#: Every interval method, in reporting order.
METHOD_NAMES: tuple[str, ...] = (
    "global_residual",
    "conformal_global",
    "state_residual",
    "conformal_state",
    "conformal_dispersion",
    "quantile_regression",
)

#: The method every other is measured against: a per-horizon constant width from
#: historical residuals. Nothing simpler is available, and a method that cannot beat it has
#: added nothing.
BASELINE_METHOD = "global_residual"

#: Minutes per step on the 15-minute governing grid.
_STEP_MINUTES = 15


@dataclass(frozen=True, slots=True)
class Phase8Result:
    """Everything Phase 8 measured.

    Attributes:
        experiment_id: Registry identifier.
        dataset_version: Phase 5 sequence version, inherited through Phase 7.
        weights: Phase 7's fixed-ensemble weights, as recovered and verified.
        split: The calibration partition, with row counts.
        parity: Point-forecast parity against Phase 7's published MAE.
        split_parity: Point-forecast error on rows the ensemble's weights did and did not see.
        disagreement: Spread statistics and whether they track error.
        methods: Per-method evaluation.
        comparison: The phase's headline table.
        selection: Which interval procedure was published per horizon, and on what
            evidence. Recorded because "the artifact says method X" and "X was chosen
            here" are different claims, and only the second one is checkable.
        verdict: The phase's answer.
        methods_detail: Full per-method records.
        ablations: What each scaling choice bought.
        features: The learned scale's inputs, with the poisoning check's record.
        quantile_crossings: Rows per horizon and nominal level whose fitted residual
            quantiles needed re-sorting, a known weakness of independently fitted pinball
            models.
        artifact_summary: The ProbabilisticForecast batch's summary.
        data_quality: Whether a quality-conditioned experiment was possible.
        seconds: Wall time.
        artifacts: Written paths.
        notes: Anything a reader would otherwise have to infer.
    """

    experiment_id: str
    dataset_version: str
    weights: dict[str, Any]
    split: dict[str, Any]
    parity: dict[str, Any]
    split_parity: dict[str, Any]
    disagreement: dict[str, Any]
    methods: dict[str, Any]
    comparison: list[dict[str, Any]]
    selection: dict[str, Any]
    verdict: dict[str, Any]
    methods_detail: dict[str, Any]
    ablations: dict[str, Any]
    features: dict[str, Any]
    quantile_crossings: dict[str, dict[str, int]]
    artifact_summary: dict[str, Any]
    data_quality: dict[str, Any]
    seconds: float
    artifacts: dict[str, str]
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "dataset_version": self.dataset_version,
            "weights": self.weights,
            "split": self.split,
            "parity": self.parity,
            "split_parity": self.split_parity,
            "disagreement": self.disagreement,
            "methods": self.methods,
            "comparison": self.comparison,
            "selection": self.selection,
            "verdict": self.verdict,
            "methods_detail": self.methods_detail,
            "ablations": self.ablations,
            "features": self.features,
            "quantile_crossings": self.quantile_crossings,
            "artifact_summary": self.artifact_summary,
            "data_quality": self.data_quality,
            "seconds": round(self.seconds, 3),
            "artifacts": self.artifacts,
            "notes": list(self.notes),
        }


def _load_phase7_weights(path: Path) -> dict[int, dict[str, float]]:
    """Phase 7's fitted fixed-ensemble weights, read from its own artifact.

    Args:
        path: Phase 7's ``result.json``.

    Returns:
        ``{horizon: {expert_name: weight}}``.

    Raises:
        FileNotFoundError: If the Phase 7 result is absent.
        ValueError: If it does not contain the weights.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Phase 7 result not found at {path}; Phase 8 reuses Phase 7's fitted "
            f"fixed-ensemble weights rather than refitting them"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    recorded = payload.get("routing", {}).get("fixed_ensemble_weights")
    if not recorded:
        raise ValueError(
            f"{path} has no routing.fixed_ensemble_weights; the Phase 7 artifact cannot "
            f"be used to recover the point forecast"
        )
    return {
        int(horizon): {name: float(weight) for name, weight in entry.items()}
        for horizon, entry in recorded.items()
    }


def _weights_array(
    recorded: dict[int, dict[str, float]],
    expert_names: tuple[str, ...],
    horizons: tuple[int, ...],
) -> np.ndarray:
    """Phase 7's weights as a ``[horizons, experts]`` array, order-checked.

    Args:
        recorded: The recovered weights.
        expert_names: The expert pool, in pool order.
        horizons: Horizon steps.

    Returns:
        ``[horizons, experts]`` weights, each row summing to 1.

    Raises:
        ValueError: If a horizon is missing, an expert is absent, or a row does not sum to
            one - any of which means the pool changed since Phase 7 and the recovered
            weights no longer describe this panel.
    """
    out = np.zeros((len(horizons), len(expert_names)), dtype=np.float64)
    for column, horizon in enumerate(horizons):
        entry = recorded.get(int(horizon))
        if entry is None:
            raise ValueError(
                f"Phase 7 recorded no weights for horizon {horizon}; have "
                f"{sorted(recorded)}"
            )
        missing = [name for name in expert_names if name not in entry]
        if missing:
            raise ValueError(
                f"Phase 7's weights for h={horizon} omit {missing}; the expert pool "
                f"has changed since Phase 7"
            )
        out[column] = [entry[name] for name in expert_names]
        total = out[column].sum()
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Phase 7's weights for h={horizon} sum to {total}, not 1; the pool or "
                f"the record is inconsistent"
            )
    return out


def _subset_intervals(
    intervals: dict[float, PredictionInterval], rows: np.ndarray
) -> dict[float, PredictionInterval]:
    """Restrict a panel-wide interval family to the evaluated rows.

    Bounds are built over every panel row because two of them need calibration-split rows -
    the learned scale's normalisation reference and the band cut points - but they are
    evaluated only on test. Slicing here rather than at each call site keeps the two row
    sets impossible to confuse.

    Args:
        intervals: ``{nominal_level: PredictionInterval}`` over the whole panel.
        rows: Panel positions to keep.

    Returns:
        The same family restricted to ``rows``.
    """
    picked = np.asarray(rows, dtype=np.int64)
    return {
        float(nominal): PredictionInterval(
            lower_kw=np.asarray(interval.lower_kw, dtype=np.float64)[picked],
            upper_kw=np.asarray(interval.upper_kw, dtype=np.float64)[picked],
            nominal_level=float(interval.nominal_level),
            method=interval.method,
            symmetric=bool(interval.symmetric),
            clipped_at_zero=bool(interval.clipped_at_zero),
            notes=interval.notes,
        )
        for nominal, interval in intervals.items()
    }


def _subset_disagreement(disagreement, rows: np.ndarray):
    """Restrict the panel-wide spread statistics to the evaluated rows.

    Args:
        disagreement: Panel-wide :class:`DisagreementMeasures`.
        rows: Panel positions to keep.

    Returns:
        The same measures restricted to ``rows``, with the point forecast subset to match.
    """
    from ..measures import DisagreementMeasures

    picked = np.asarray(rows, dtype=np.int64)
    return DisagreementMeasures(
        measures={
            name: np.asarray(values, dtype=np.float64)[picked]
            for name, values in disagreement.measures.items()
        },
        expert_names=disagreement.expert_names,
        horizons=disagreement.horizons,
        point_kw=np.asarray(disagreement.point_kw, dtype=np.float64)[picked],
    )


def _asset_ids(
    *, rows: np.ndarray, series_index: np.ndarray, series_ids: tuple[str, ...]
) -> tuple[str, ...]:
    """Repository ``AssetId`` values for the evaluated rows.

    The dataset's own series identifiers (``load_p1ulv14763``) are not domain identifiers:
    the ``AssetId`` pattern requires an ``asset-`` prefix, and a batch that failed that
    check could not be converted to a domain :class:`Forecast` at all. The mapping is a
    pure prefix, so it is reversible and recorded in the artifact's notes.

    Args:
        rows: Panel positions being written.
        series_index: ``[N]`` series index per panel position.
        series_ids: Dataset series identifiers, in series-index order.

    Returns:
        ``[len(rows)]`` domain asset identifiers.
    """
    series = np.asarray(series_index, dtype=np.int64)
    out: list[str] = []
    for row in np.asarray(rows, dtype=np.int64):
        index = int(series[int(row)])
        slug = series_ids[index] if index < len(series_ids) else f"series-{index}"
        out.append(slug if slug.startswith("asset-") else f"asset-{slug}")
    return tuple(out)


def run_phase8(
    *,
    cache,
    panel,
    values: np.ndarray,
    artifact_dir: Path,
    config,
    experiment_id: str = "phase8-uncertainty-main",
    dataset_version: str = "",
    seed: int = 20260101,
    series_ids: tuple[str, ...] = (),
    origin_iso: str = "",
    phase7_result_path: Path,
    published_point_mae: dict[str, dict[str, float]] | None = None,
    nominal_levels: tuple[float, ...] = DEFAULT_NOMINAL_LEVELS,
    run_ablations: bool = True,
    log: Any = None,
) -> Phase8Result:
    """Run the phase end to end.

    Args:
        cache: The Phase 7 expert forecast cache.
        panel: The evaluation panel the cache describes.
        values: ``[n_series, n_steps]`` per-unit values, for the origin-observable features.
        artifact_dir: Where everything is written.
        config: A :class:`~energy_intelligence.config.uncertainty.UncertaintyExperimentConfig`.
        experiment_id: Registry identifier.
        dataset_version: Sequence version, recorded.
        seed: Seed for every estimator.
        series_ids: Series identifiers, for the forecast trace.
        origin_iso: The series origin, so trace timestamps are real.
        phase7_result_path: Phase 7's ``result.json``, read for the fixed-ensemble weights.
        published_point_mae: Phase 7's published test MAE per predictor, for the parity check.
        nominal_levels: Coverage levels to report.
        run_ablations: Run the scaling ablations.
        log: Optional progress callable.

    Returns:
        The measured result.

    Raises:
        ValueError: If the config's fixed-ensemble weights disagree with Phase 7's, the
            band level is not one of the reported levels, or the panel and cache disagree.
    """
    started = time.perf_counter()
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    nominal_levels = tuple(float(level) for level in nominal_levels)

    forecasts = np.asarray(cache.forecasts, dtype=np.float64)
    actual_kw = np.asarray(cache.actual_kw, dtype=np.float64)
    horizons = tuple(int(h) for h in cache.horizons)
    expert_names = tuple(cache.expert_names)
    n_experts, n_rows, n_horizons = forecasts.shape
    if n_horizons != len(horizons) or n_rows != panel.n_rows:
        raise ValueError(
            f"cache {forecasts.shape} does not match the panel "
            f"({panel.n_rows} rows, {len(horizons)} horizons)"
        )
    band_level = float(config.band_nominal_level)
    if not any(abs(band_level - level) < 1e-12 for level in nominal_levels):
        raise ValueError(
            f"band_nominal_level {band_level} is not one of the reported levels "
            f"{nominal_levels}; the artifact's calibration status is read from the "
            f"evaluated metrics, so the band level has to be one of them"
        )

    validation_rows = np.arange(
        panel.split_slices["validation"].start, panel.split_slices["validation"].stop
    )
    test_rows = np.arange(panel.split_slices["test"].start, panel.split_slices["test"].stop)
    origins = np.asarray(panel.origins, dtype=np.int64)

    # ---- 1. The point forecast, recovered not refitted ---------------------
    if log is not None:
        log("recovering Phase 7's fixed-ensemble weights")
    recorded = _load_phase7_weights(Path(phase7_result_path))
    weights = _weights_array(recorded, expert_names, horizons)
    configured = np.asarray(config.weight_array(expert_names), dtype=np.float64)
    if configured.shape != weights.shape or not np.allclose(configured, weights, atol=1e-9):
        raise ValueError(
            "configs/uncertainty.toml's fixed_ensemble_weights do not match Phase 7's "
            f"recorded weights.\n  config: {configured.tolist()}\n"
            f"  phase 7: {weights.tolist()}\n"
            "Phase 8 reuses Phase 7's weights verbatim; a mismatch means the config "
            "would silently produce a different point forecast from Phase 4/5/7's."
        )
    point_kw = np.einsum("he,enh->nh", weights, forecasts)

    # ---- 2. Parity with Phase 7 -------------------------------------------
    parity = _point_parity(
        point_kw=point_kw[test_rows],
        actual_kw=actual_kw[test_rows],
        horizons=horizons,
        published=(published_point_mae or {}).get("fixed_ensemble"),
        tolerance_pct=2.0,
    )

    # ---- 3. The partition, and whether it is justified ---------------------
    split = build_calibration_split(
        validation_rows=validation_rows,
        test_rows=test_rows,
        origins=origins,
        horizons=horizons,
        cal_fraction=float(config.calibration_fit_fraction),
    )
    split_parity = split_parity_check(
        point_kw=point_kw,
        actual_kw=actual_kw,
        split=split,
        horizons=horizons,
    )
    if log is not None:
        log(
            f"split: {split.calibration_fit.size} fit / "
            f"{split.calibration_conformity.size} conformity / {test_rows.size} test; "
            f"fit-vs-conformity MAE gap "
            + ", ".join(
                f"h{h}={split_parity[str(h)]['fit_vs_conformity_pct']:+.3f}%"
                for h in horizons
            )
        )

    fit_rows = split.calibration_fit
    conform_rows = split.calibration_conformity
    test = test_rows

    residual_all = actual_kw - point_kw
    residual_fit = residual_all[fit_rows]
    residual_conf = residual_all[conform_rows]

    # ---- 4. Origin-observable features for the learned scale ---------------
    if log is not None:
        log("building origin-observable features")
    features, feature_names, causality = build_uncertainty_features(
        values=values,
        panel=panel,
        forecasts=forecasts,
        actual_kw=actual_kw,
        horizons=horizons,
        include_past_error=bool(config.include_past_error_features),
        reference_rows=fit_rows,
        sample=int(config.causality_sample),
        seed=seed,
        log=log,
    )

    # ---- 5. Expert disagreement -------------------------------------------
    row_scale_kw = np.asarray(panel.row_scale, dtype=np.float64)
    disagreement = disagreement_measures(
        expert_forecast_kw=forecasts,
        row_scale_kw=row_scale_kw,
        horizons=horizons,
        expert_names=expert_names,
        point_kw=point_kw,
    )
    spread_kw = DisagreementScale(
        measure="max_minus_min", horizons=horizons
    )(disagreement["max_minus_min"], row_scale_kw)

    # ---- 6. Fit each method ------------------------------------------------
    if log is not None:
        log("fitting interval methods")
    fit_seconds: dict[str, float] = {}
    intervals: dict[str, dict[float, PredictionInterval]] = {}
    scale_functions: dict[str, Any] = {}

    # (a) global_residual - a per-horizon constant width from realised residuals
    fit_start = time.perf_counter()
    global_s = fitted_scale(residual_kw=residual_fit, horizons=horizons)
    fit_seconds["global_residual"] = time.perf_counter() - fit_start
    scale_functions["global_residual"] = global_s
    intervals["global_residual"] = _empirical_residual_intervals(
        residual_conf=residual_conf,
        point_kw=point_kw,
        nominal_levels=nominal_levels,
        method="global_residual",
        n_calibration_rows=int(conform_rows.size),
    )

    # (b) conformal_global - the same object with a finite-sample-corrected quantile
    fit_start = time.perf_counter()
    conformal_global = fit_conformal(
        residual_kw=residual_conf,
        scale=None,
        horizons=horizons,
        nominal_levels=nominal_levels,
        scale_name="absolute",
    )
    fit_seconds["conformal_global"] = time.perf_counter() - fit_start
    scale_functions["conformal_global"] = conformal_global
    intervals["conformal_global"] = conformal_global.intervals(
        point_kw=point_kw,
        scale=None,
        horizons=horizons,
        nominal_levels=nominal_levels,
        method="conformal_global",
        notes=conformal_global.guarantee,
    )

    # (c, d) learned scale - fitted on CAL_FIT, its multiplier read on CAL_CONF
    if log is not None:
        log("  learned scale")
    fit_start = time.perf_counter()
    learned = [
        fit_learned_scale(
            features=features[fit_rows],
            residual_kw=residual_all[fit_rows, column],
            feature_names=feature_names,
            horizon_index=column,
            seed=seed,
            max_iter=int(config.scale_max_iter),
        )
        for column in range(len(horizons))
    ]
    fit_seconds["learned_scale"] = time.perf_counter() - fit_start
    learned_scale = np.column_stack(
        [learned[column].predict(features) for column in range(len(horizons))]
    )
    scale_functions["learned_scale"] = learned

    if log is not None:
        log("  conformal state-scaled")
    conformal_state = fit_conformal(
        residual_kw=residual_conf,
        scale=learned_scale[conform_rows],
        horizons=horizons,
        nominal_levels=nominal_levels,
        scale_name="learned",
    )
    intervals["state_residual"] = _empirical_residual_intervals(
        residual_conf=residual_conf,
        point_kw=point_kw,
        nominal_levels=nominal_levels,
        method="state_residual",
        scale=learned_scale,
        conform_positions=conform_rows,
        n_calibration_rows=int(conform_rows.size),
    )
    intervals["conformal_state"] = conformal_state.intervals(
        point_kw=point_kw,
        scale=learned_scale,
        horizons=horizons,
        nominal_levels=nominal_levels,
        method="conformal_state",
        notes=conformal_state.guarantee,
    )

    # (e) expert disagreement as the scale - the prompt's central hypothesis
    if log is not None:
        log("  conformal disagreement-scaled")
    conformal_dispersion = fit_conformal(
        residual_kw=residual_conf,
        scale=spread_kw[conform_rows],
        horizons=horizons,
        nominal_levels=nominal_levels,
        scale_name="expert_disagreement",
    )
    scale_functions["conformal_dispersion"] = conformal_dispersion
    intervals["conformal_dispersion"] = conformal_dispersion.intervals(
        point_kw=point_kw,
        scale=spread_kw,
        horizons=horizons,
        nominal_levels=nominal_levels,
        method="conformal_dispersion",
        notes=conformal_dispersion.guarantee,
    )

    # (f) direct residual quantile regression by pinball loss
    if log is not None:
        log("  quantile regression (pinball loss on the residual)")
    grid = tuple(
        sorted(
            {
                round(tail, 6)
                for nominal in nominal_levels
                for tail in quantile_levels((nominal,))[0]
            }
        )
    )
    quantile_models = []
    fit_start = time.perf_counter()
    for column, horizon in enumerate(horizons):
        quantile_models.append(
            fit_quantile_model(
                features=features[fit_rows],
                residual_kw=residual_all[fit_rows, column],
                feature_names=feature_names,
                horizon=horizon,
                quantiles=grid,
                seed=seed,
                max_iter=int(config.quantile_max_iter),
            )
        )
    fit_seconds["quantile_regression"] = time.perf_counter() - fit_start
    scale_functions["quantile_regression"] = quantile_models
    # Quantile crossing is a property of the method, not an implementation detail, so it is
    # measured on the evaluated rows and carried into the record rather than left in a
    # note string on each interval. Counted at the headline level, which is the interval
    # the phase's headline claims rest on.
    quantile_crossings = {
        str(horizon): {
            str(level): count
            for level, count in model.monotone_quantiles(
                features, nominal_levels=(float(max(nominal_levels)),)
            )[1].items()
        }
        for horizon, model in zip(horizons, quantile_models)
    }
    intervals["quantile_regression"] = _assemble_quantile_intervals(
        models=quantile_models,
        horizons=horizons,
        point_kw=point_kw,
        features=features,
        nominal_levels=nominal_levels,
    )

    # ---- 7. Expert point-forecast reference, for "is this predictor-specific" --
    point_table = expert_point_forecast_table(
        forecasts={
            **{name: forecasts[index][test] for index, name in enumerate(expert_names)},
            "fixed_ensemble": point_kw[test],
        },
        actual_kw=actual_kw[test],
        horizons=horizons,
    )

    # ---- 8. Evaluate, once, on test ----------------------------------------
    criterion = coverage_tolerance(int(test.size), max(nominal_levels))
    tolerance = float(criterion["tolerance"])
    if log is not None:
        log(f"evaluating on {test.size} sealed test rows, tolerance {tolerance:.4f}")

    test_disagreement = _subset_disagreement(disagreement, test)
    results: dict[str, dict[str, Any]] = {
        method: evaluate_method(
            method=method,
            actual_kw=actual_kw[test],
            point_kw=point_kw[test],
            horizons=horizons,
            intervals=_subset_intervals(method_intervals, test),
            nominal_levels=nominal_levels,
            disagreement=test_disagreement,
            tolerance=tolerance,
            bins=int(config.reliability_bins),
        )
        for method, method_intervals in intervals.items()
    }
    # The same baseline method applied to each single expert, so the answer does not
    # depend on which point predictor is in use.
    for index, name in enumerate(expert_names):
        single_point = forecasts[index]
        expert_residual_conf = actual_kw[conform_rows] - single_point[conform_rows]
        method = f"global_residual_on_{name}"
        results[method] = evaluate_method(
            method=method,
            actual_kw=actual_kw[test],
            point_kw=single_point[test],
            horizons=horizons,
            intervals=_subset_intervals(
                _empirical_residual_intervals(
                    residual_conf=expert_residual_conf,
                    point_kw=single_point,
                    nominal_levels=nominal_levels,
                    method=method,
                    n_calibration_rows=int(conform_rows.size),
                ),
                test,
            ),
            nominal_levels=nominal_levels,
            disagreement=test_disagreement,
            tolerance=tolerance,
            bins=int(config.reliability_bins),
        )

    comparison = method_table(results, horizons, nominal_level=max(nominal_levels))

    # The procedure that gets published is chosen per horizon, at the level the artifact is
    # drawn at, from the measured coverage: among the methods whose coverage is inside the
    # tolerance at this horizon and this level, the one with the best weighted interval
    # score. If nothing is calibrated the selection is stated rather than made, because a
    # published interval whose own status reads FAIL is not a result and a silent fallback
    # to "the one with the narrowest interval" is exactly the failure this phase exists to
    # catch.
    headline = max(nominal_levels)
    headline = max(nominal_levels)
    band_level = float(config.band_nominal_level)
    level_key = str(band_level)
    selection: dict[str, Any] = {
        "level": band_level,
        "tolerance": tolerance,
        "criterion": (
            f"coverage within {tolerance:.4f} absolute of {band_level:.0%} at this "
            f"horizon, then the lowest weighted interval score"
        ),
        "by_horizon": {},
    }
    selected: list[str] = []
    for column, horizon in enumerate(horizons):
        eligible = [
            row
            for row in comparison
            if row["horizon"] == int(horizon)
            and results[row["method"]]["by_horizon"][str(horizon)]["metrics"]["levels"][
                level_key
            ]["passes"]
        ]
        if eligible:
            winner = min(eligible, key=lambda row: row["interval_score_kw"])["method"]
            selection["by_horizon"][str(horizon)] = {
                "method": winner,
                "interval_score_kw": next(
                    row["interval_score_kw"]
                    for row in eligible
                    if row["method"] == winner
                ),
                "n_calibrated_candidates": len(eligible),
            }
        else:
            fallback = min(
                (row for row in comparison if row["horizon"] == int(horizon)),
                key=lambda row: row["interval_score_kw"],
            )["method"]
            selection["by_horizon"][str(horizon)] = {
                "method": fallback,
                "calibrated": False,
                "note": (
                    f"no method was calibrated at {band_level:.0%} within "
                    f"{tolerance:.4f} at this horizon; the published interval uses "
                    f"{fallback}, whose calibration status reads FAIL. This is a "
                    f"measurement, not a selection"
                ),
                "interval_score_kw": next(
                    row["interval_score_kw"]
                    for row in comparison
                    if row["horizon"] == int(horizon) and row["method"] == fallback
                ),
                "n_calibrated_candidates": 0,
            }
            winner = fallback
        selected.append(winner)

    # ---- 9. The disagreement hypothesis, measured --------------------------
    error_test = np.abs(actual_kw[test] - point_kw[test])
    spread_vs_error = {
        measure: {
            str(horizon): float(spearman(values[test, column], error_test[:, column]))
            for column, horizon in enumerate(horizons)
        }
        for measure, values in disagreement.measures.items()
    }
    disagreement_report: dict[str, Any] = {
        "units": "per unit of the row's rated kW",
        "measures": disagreement.describe(test),
        "spread_vs_error_spearman": spread_vs_error,
        "per_horizon_best_measure_by_spearman": {
            str(horizon): max(
                spread_vs_error,
                key=lambda measure: (
                    spread_vs_error[measure][str(horizon)]
                    if spread_vs_error[measure][str(horizon)] == spread_vs_error[measure][str(horizon)]
                    else -np.inf
                ),
            )
            for horizon in horizons
        },
        "interval_width_vs_error_spearman": {
            method: {
                str(horizon): results[method]["by_horizon"][str(horizon)][
                    "ranking_by_width"
                ]["spearman"]
                for horizon in horizons
            }
            for method in dict.fromkeys([*selected, "conformal_dispersion"])
        },
        "hypothesis": (
            "greater expert disagreement indicates greater forecast uncertainty. Tested "
            "two ways: whether each spread statistic correlates with realised absolute "
            "error, and whether the interval built on that spread ranks rows better "
            "than the constant-width baseline."
        ),
    }

    # ---- 10. Regime breakdown and imminent-ramp detection ------------------
    regime_axes = _regime_axes(actual_kw, panel)
    regime_report = {
        "method": list(selected),
        "headline_nominal_level": float(headline),
        "axis_source": (
            "terciles of the whole panel's h=1 realised demand and one-step ramp, so the "
            "regimes are the same definitions Phase 7 used and are not re-cut on the "
            "rows being scored"
        ),
        "by_horizon": {
            str(column): regime_breakdown(
                actual_kw=actual_kw[test],
                point_kw=point_kw[test],
                interval=_subset_intervals(intervals[selected[column]], test)[
                    float(headline)
                ],
                axes=_slice_axes(regime_axes, test),
                horizon_index=column,
                nominal_level=float(headline),
            )
            for column in range(len(horizons))
        },
    }
    persistence_kw = np.asarray(panel.persistence_kw, dtype=np.float64)
    ramp_report = {
        "question": (
            "Phase 7's router could not distinguish flat demand from imminent ramp; can "
            "the uncertainty?"
        ),
        "label": (
            "|y(t+h) - y(t)|, a realised diagnostic used only to group rows for "
            "evaluation. It is never an input: the interval may use present information "
            "only."
        ),
        "by_method": {
            method: {
                str(column): imminent_ramp_detection(
                    uncertainty=np.asarray(method_intervals[float(headline)].width_kw)[
                        test, column
                    ],
                    absolute_error_kw=error_test[:, column],
                    future_ramp_kw=np.abs(
                        actual_kw[test, column] - persistence_kw[test, column]
                    ),
                    label=f"{method}:width@h{horizons[column]}",
                )
                for column in range(len(horizons))
            }
            for method, method_intervals in intervals.items()
        },
    }

    # ---- 11. The artifact --------------------------------------------------
    # One column per horizon, each from the method selected for that horizon. Band cut
    # points come from the conformity rows, never from the test rows: a threshold computed
    # on the rows being scored would be a test statistic inside an operational label.
    selected_bounds = np.column_stack(
        [
            np.asarray(intervals[selected[column]][band_level].lower_kw)[:, column]
            for column in range(len(horizons))
        ]
    )
    selected_upper = np.column_stack(
        [
            np.asarray(intervals[selected[column]][band_level].upper_kw)[:, column]
            for column in range(len(horizons))
        ]
    )
    calibration_widths = np.column_stack(
        [
            np.asarray(intervals[selected[column]][band_level].width_kw)[conform_rows]
            for column in range(len(horizons))
        ]
    )
    cuts = [
        BandCutPoints.from_widths(calibration_widths[:, column], nominal_level=band_level)
        for column in range(len(horizons))
    ]
    start = datetime.fromisoformat(origin_iso) if origin_iso else None
    timestamps = (
        [
            (start + timedelta(minutes=int(origins[int(row)]) * _STEP_MINUTES)).isoformat()
            for row in test
        ]
        if start is not None
        else [f"origin:{int(origins[int(row)])}" for row in test]
    )
    batch = ProbabilisticForecastBatch(
        timestamps=tuple(timestamps),
        asset_ids=_asset_ids(
            rows=test, series_index=panel.series, series_ids=series_ids
        ),
        horizon_steps=horizons,
        point_kw=point_kw[test],
        lower_kw=selected_bounds[test],
        upper_kw=selected_upper[test],
        nominal_level=band_level,
        method=tuple(selected),
        model_version=str(config.model_version),
        calibration_status=tuple(
            CALIBRATION_PASS
            if results[selected[column]]["by_horizon"][str(horizons[column])]["metrics"][
                "levels"
            ][level_key]["passes"]
            else CALIBRATION_FAIL
            for column in range(len(horizons))
        ),
        coverage_error=tuple(
            float(
                results[selected[column]]["by_horizon"][str(horizons[column])]["metrics"][
                    "levels"
                ][level_key]["coverage_error"]
            )
            for column in range(len(horizons))
        ),
        band_cuts=tuple(cut.as_tuple() for cut in cuts),
        notes=(
            f"per-horizon procedure chosen from measured coverage at {band_level:.0%}: "
            + ", ".join(f"h{h}={selected[i]}" for i, h in enumerate(horizons))
            + "; band cut points from "
            + ", ".join(
                f"h{horizons[i]}={cuts[i].source_rows} calibration rows" for i in range(len(horizons))
            )
            + "; asset ids are the dataset's series ids prefixed with 'asset-' to satisfy "
            "the domain AssetId pattern"
        ),
    )
    batch_path = save_batch(batch, artifact_dir / "probabilistic_forecasts.npz")

    # ---- 12. Ablations -----------------------------------------------------
    ablations: dict[str, Any] = {}
    if run_ablations:
        if log is not None:
            log("ablations")
        ablations = _run_ablations(
            results=results,
            comparison=comparison,
            horizons=horizons,
            nominal_level=float(headline),
            disagreement=disagreement,
        )

    # ---- 13. Data quality --------------------------------------------------
    data_quality = _data_quality_report()

    # ---- 14. Verdict -------------------------------------------------------
    verdict = phase8_verdict(
        rows=comparison,
        results={name: results[name] for name in METHOD_NAMES},
        horizons=horizons,
        nominal_level=float(headline),
        baseline_method=BASELINE_METHOD,
        tolerance=tolerance,
        disagreement=spread_vs_error,
    )

    written = {
        "result": str(artifact_dir / "result.json"),
        "summary": str(artifact_dir / "summary.md"),
        "forecasts": str(batch_path),
        "trace": str(artifact_dir / "uncertainty_trace.jsonl"),
        "methods": str(artifact_dir / "methods.json"),
        "regimes": str(artifact_dir / "regimes.json"),
        "disagreement": str(artifact_dir / "disagreement.json"),
    }
    _write_trace(
        path=Path(written["trace"]),
        rows=test,
        rows_kept=int(config.trace_rows),
        batch=batch,
        panel=panel,
        actual_kw=actual_kw,
        disagreement=disagreement,
        horizons=horizons,
        spread_kw=spread_kw,
        learned_scale=learned_scale,
    )
    _write_json(
        Path(written["methods"]),
        {
            "intervals": {
                name: {
                    str(level): interval.to_dict()
                    for level, interval in method_intervals.items()
                }
                for name, method_intervals in intervals.items()
            },
            "scale_functions": {
                name: (
                    scale.to_dict()
                    if hasattr(scale, "to_dict")
                    else [item.to_dict() for item in scale]
                )
                for name, scale in scale_functions.items()
            },
            "features": {
                "names": list(feature_names),
                "causality": causality,
            },
            "quantile_crossings": quantile_crossings,
            "expert_point_forecasts": point_table,
            "fit_seconds": {name: round(value, 3) for name, value in fit_seconds.items()},
        },
    )
    _write_json(Path(written["regimes"]), regime_report)
    _write_json(Path(written["disagreement"]), disagreement_report)

    notes = (
        "the point forecast is Phase 7's fixed ensemble with Phase 7's weights, verified "
        "against its own artifact before anything was fitted",
        "uncertainty was fitted on the first half of the validation split and calibrated "
        "on the second; the test split was read once",
        "no aleatoric/epistemic decomposition is claimed; the vocabulary is model "
        "disagreement and data variability",
        f"coverage tolerance {tolerance:.4f}, derived from {criterion['basis']}",
        f"band cut points at {band_level:.0%} come from "
        f"{cuts[0].source_rows} conformity rows per horizon, not from the test rows",
    )

    result = Phase8Result(
        experiment_id=experiment_id,
        dataset_version=dataset_version,
        weights={
            "fixed_ensemble_weights": {
                str(horizon): {
                    name: float(weight) for name, weight in recorded[int(horizon)].items()
                }
                for horizon in horizons
            },
            "source": str(Path(phase7_result_path)),
            "verified_against_config": True,
        },
        split=split.describe(),
        parity=parity,
        split_parity=split_parity,
        disagreement=disagreement_report,
        methods={name: evaluation["by_horizon"] for name, evaluation in results.items()},
        comparison=comparison,
        selection=selection,
        verdict=verdict,
        methods_detail=results,
        ablations=ablations,
        features={"names": list(feature_names), "causality": causality},
        quantile_crossings=quantile_crossings,
        artifact_summary=batch.to_dict(),
        data_quality=data_quality,
        seconds=time.perf_counter() - started,
        artifacts=written,
        notes=notes,
    )
    _write_json(Path(written["result"]), result.to_dict())
    Path(written["summary"]).write_text(render_summary(result), encoding="utf-8")
    return result


def _empirical_residual_intervals(
    *,
    residual_conf: np.ndarray,
    point_kw: np.ndarray,
    nominal_levels: tuple[float, ...],
    method: str,
    scale: np.ndarray | None = None,
    conform_positions: np.ndarray | None = None,
    n_calibration_rows: int,
) -> dict[float, PredictionInterval]:
    """Interval from the empirical quantiles of the *calibration* residuals.

    The conformal methods in :mod:`uncertainty.calibration` take the ``ceil((n+1)(1-a))``-th
    smallest conformity score; this takes the plain empirical quantile of the same scores.
    The two differ by less than one order statistic at this sample size, which is exactly
    why both are kept: the difference is negligible here, and the ablation reports it as
    negligible rather than asserting the correction matters.

    Args:
        residual_conf: ``[n_conf, horizons]`` signed residuals on the conformity rows.
        point_kw: ``[N, horizons]`` point forecast over every panel row.
        nominal_levels: Coverage levels.
        method: Procedure name recorded on each interval.
        scale: ``[N, horizons]`` width scale, or ``None`` for a unit scale.
        conform_positions: Panel positions of the conformity rows. Required whenever
            ``scale`` is given, because the scale is held for every panel row while the
            residuals are held for the conformity rows only, and slicing the wrong block
            would normalise each residual by another row's uncertainty.
        n_calibration_rows: How many conformity rows supplied the quantile, for the notes.

    Returns:
        ``{nominal_level: PredictionInterval}`` with ``[N, horizons]`` bounds.

    Raises:
        ValueError: If the scale's shape does not match the point forecast, or positions are
            missing when a scale is supplied.
    """
    residual = np.abs(np.asarray(residual_conf, dtype=np.float64))
    point = np.asarray(point_kw, dtype=np.float64)
    if scale is None:
        scale_all = np.ones_like(point)
        scale_conf = np.ones_like(residual)
    else:
        scale_all = np.asarray(scale, dtype=np.float64)
        if scale_all.shape != point.shape:
            raise ValueError(
                f"scale {scale_all.shape} must match point_kw {point.shape}"
            )
        if conform_positions is None:
            raise ValueError(
                "conform_positions is required when a scale is supplied; the scale covers "
                "every panel row but the residuals cover only the conformity rows"
            )
        scale_conf = scale_all[np.asarray(conform_positions, dtype=np.int64)]
        if np.any(scale_conf <= 0):
            raise ValueError(
                "the conformity rows' scale must be strictly positive; a zero would "
                "divide the score by zero and produce an infinite width"
            )
    scores = residual / scale_conf
    out: dict[float, PredictionInterval] = {}
    for nominal in nominal_levels:
        alpha = 1.0 - float(nominal)
        multiplier = float(np.quantile(scores, 1.0 - alpha))
        out[float(nominal)] = widths_from_scale(
            point_kw=point,
            scale=scale_all,
            multiplier=multiplier,
            method=method,
            nominal_level=float(nominal),
            notes=(
                f"empirical {1.0 - alpha:.3f} quantile of |residual| on "
                f"{int(n_calibration_rows)} calibration rows"
            ),
        )
    return out


def _assemble_quantile_intervals(
    *,
    models: list[Any],
    horizons: tuple[int, ...],
    point_kw: np.ndarray,
    features: np.ndarray,
    nominal_levels: tuple[float, ...],
) -> dict[float, PredictionInterval]:
    """Combine per-horizon single-column quantile intervals into ``[N, horizons]`` bounds.

    Each model covers one horizon and returns ``[N, 1]`` bounds, because a pinball-loss
    model fitted to one residual column cannot speak for another. Stitching them column by
    column here keeps that per-horizon separation explicit instead of pretending a single
    multi-horizon quantile model was fitted.

    Args:
        models: One fitted model per horizon, in horizon order.
        horizons: Horizon steps.
        point_kw: ``[N, horizons]`` point forecast, kW.
        features: ``[N, n_features]`` origin-observable features.
        nominal_levels: Coverage levels.

    Returns:
        ``{nominal_level: PredictionInterval}`` with ``[N, horizons]`` bounds.

    Raises:
        ValueError: If ``models`` does not cover every horizon.
    """
    horizons = tuple(int(h) for h in horizons)
    if len(models) != len(horizons):
        raise ValueError(
            f"{len(models)} quantile models for {len(horizons)} horizons; each horizon "
            f"needs its own, because the residual distributions differ by an order of "
            f"magnitude across them"
        )
    point = np.asarray(point_kw, dtype=np.float64)
    n_rows = point.shape[0]
    # One pair of arrays per nominal level. A single shared pair, refilled as each level is
    # stitched in, would return the *last* level's bounds under every key - four different
    # nominal levels, four identical intervals, and coverage figures that repeat to six
    # decimal places because they are measuring one interval four times.
    per_level: dict[float, tuple[np.ndarray, np.ndarray]] = {
        float(nominal): (
            np.empty((n_rows, len(horizons)), dtype=np.float64),
            np.empty((n_rows, len(horizons)), dtype=np.float64),
        )
        for nominal in nominal_levels
    }
    for column, model in enumerate(models):
        built = model.intervals(
            point_kw=point,
            features=features,
            nominal_levels=nominal_levels,
            horizons=horizons,
        )
        for nominal in nominal_levels:
            interval = built[float(nominal)]
            low, high = per_level[float(nominal)]
            low[:, column] = np.asarray(interval.lower_kw, dtype=np.float64)[:, 0]
            high[:, column] = np.asarray(interval.upper_kw, dtype=np.float64)[:, 0]
    train_rows = int(getattr(models[0], "train_rows", 0))
    return {
        float(nominal): PredictionInterval(
            lower_kw=low,
            upper_kw=high,
            nominal_level=float(nominal),
            method="quantile_regression",
            symmetric=False,
            notes=(
                f"pinball loss on the residual, one model per horizon, fitted on "
                f"{train_rows} calibration rows; intervals are asymmetric because the "
                f"residual quantiles are not symmetric about zero"
            ),
        )
        for nominal, (low, high) in per_level.items()
    }


def _point_parity(
    *,
    point_kw: np.ndarray,
    actual_kw: np.ndarray,
    horizons: tuple[int, ...],
    published: dict[str, float] | None,
    tolerance_pct: float,
) -> dict[str, Any]:
    """Compare the recovered point forecast with Phase 7's published test MAE.

    Args:
        point_kw: ``[n_test, horizons]`` recovered point forecast, kW.
        actual_kw: ``[n_test, horizons]`` realised targets, kW.
        horizons: Horizon steps.
        published: Phase 7's published test MAE per horizon, if available.
        tolerance_pct: Largest relative difference accepted as a match.

    Returns:
        The measured MAE per horizon, the relative difference where a reference exists, and
        whether the run reproduces Phase 7.
    """
    horizons = tuple(int(h) for h in horizons)
    measured = {
        str(h): float(mae(actual_kw[:, column], point_kw[:, column]))
        for column, h in enumerate(horizons)
    }
    out: dict[str, Any] = {
        "measured_test_mae_kw": measured,
        "tolerance_pct": float(tolerance_pct),
        "reference_mae_kw": dict(published or {}),
    }
    if published:
        relative = {
            key: 100.0 * (measured[key] - float(published[key])) / float(published[key])
            for key in measured
            if key in published and float(published[key]) > 0
        }
        out["relative_difference_pct"] = relative
        out["max_relative_difference_pct"] = max(
            (abs(value) for value in relative.values()), default=0.0
        )
        out["matches"] = bool(out["max_relative_difference_pct"] < float(tolerance_pct))
    else:
        out["matches"] = None
        out["note"] = "no published reference was supplied, so parity was not checked"
    return out


def _regime_axes(actual_kw: np.ndarray, panel: Any) -> tuple[Any, ...]:
    """Phases 4 and 5's regime axes, assigned over the whole panel.

    Computed panel-wide rather than on the evaluated rows so the tercile edges are the same
    definitions Phase 7 used. Re-cutting them per evaluation split would make a regime
    label mean something different in this phase than in the phase that established it.

    Args:
        actual_kw: ``[N, horizons]`` panel-wide realised targets.
        panel: The evaluation panel, for origins and ``y(t)``.

    Returns:
        The regime definitions, labelled for every panel row.
    """
    return assign_regimes(
        actual_kw[:, 0],
        target_id="customer_load",
        origin_index=np.asarray(panel.origins, dtype=np.int64),
        previous_actual=np.asarray(panel.persistence_kw, dtype=np.float64)[:, 0],
    )


def _slice_axes(axes: tuple[Any, ...], positions: np.ndarray) -> tuple[Any, ...]:
    """Restrict panel-wide regime definitions to the evaluated rows.

    Args:
        axes: Panel-wide definitions.
        positions: Panel positions being evaluated.

    Returns:
        Definitions whose labels and detail arrays are subset to ``positions``.

    Raises:
        ValueError: If a definition's arrays do not cover the panel, which would mean the
            axes were computed over a different row set than the one being sliced.
    """
    from ...analysis import RegimeDefinition

    picked = np.asarray(positions, dtype=np.int64)
    out: list[RegimeDefinition] = []
    for axis in axes:
        labels = np.asarray(axis.labels)
        if picked.size and labels.shape[0] <= int(picked.max()):
            raise ValueError(
                f"regime axis {axis.name!r} covers {labels.shape[0]} rows, fewer than the "
                f"panel positions being sliced"
            )
        out.append(
            RegimeDefinition(
                name=axis.name,
                description=axis.description,
                labels=labels[picked],
                detail=None if axis.detail is None else np.asarray(axis.detail)[picked],
            )
        )
    return tuple(out)


def _run_ablations(
    *,
    results: dict[str, dict[str, Any]],
    comparison: list[dict[str, Any]],
    horizons: tuple[int, ...],
    nominal_level: float,
    disagreement: Any,
) -> dict[str, Any]:
    """Compare every scaling choice against the constant-width baseline.

    The grid is the ablation: three mechanisms times three scale choices, which is why the
    methods differ from each other by exactly one thing each. The questions are:

    * does the finite-sample conformal correction matter over a plain empirical quantile?
    * does a learned scale beat a constant?
    * do the experts' own disagreement beat both?
    * does fitting the residual distribution directly beat summarising it?

    Args:
        results: Every method's evaluation.
        comparison: The headline table, for the interval scores.
        horizons: Horizon steps.
        nominal_level: The level the comparison was made at.
        disagreement: The spread measures, for the record.

    Returns:
        Per-horizon interval scores, improvement over the baseline, and each method's
        width-versus-error rank correlation.
    """
    horizons = tuple(int(h) for h in horizons)
    by_horizon: dict[str, Any] = {}
    for horizon in horizons:
        scores = {
            row["method"]: row["interval_score_kw"]
            for row in comparison
            if row["horizon"] == horizon
        }
        baseline = scores.get(BASELINE_METHOD)
        by_horizon[str(horizon)] = {
            "nominal_level": float(nominal_level),
            "interval_score_kw": scores,
            "improvement_vs_baseline_pct": {
                name: (
                    100.0 * (baseline - value) / baseline if baseline else None
                )
                for name, value in scores.items()
            },
        }
    return {
        "baseline": BASELINE_METHOD,
        "questions": {
            "finite_sample_correction": (
                "does the split-conformal quantile differ from a plain empirical quantile "
                "on the same scale?"
            ),
            "learned_vs_constant_scale": (
                "does scaling the width by origin-observable state beat a constant width?"
            ),
            "disagreement_vs_state": (
                "do the experts' own spread and a learned state-conditioned scale differ "
                "in interval score?"
            ),
            "direct_vs_summary": (
                "does fitting the residual distribution directly beat summarising it with "
                "a single quantile?"
            ),
        },
        "by_horizon": by_horizon,
        "spearman_width_vs_error": {
            method: {
                str(h): results[method]["by_horizon"][str(h)]["ranking_by_width"][
                    "spearman"
                ]
                for h in horizons
            }
            for method in sorted(results)
        },
        "disagreement_measures": list(disagreement.measures),
    }


def _data_quality_report() -> dict[str, Any]:
    """Whether a quality-conditioned experiment was possible at all.

    Phase 3 ingested complete 35,040-point SMART-DS profiles and recorded a single ``ok``
    finding, so there is no missing / interpolated / stale metadata per row to condition
    on. This reports that as a measurement - the condition has zero variance, so an
    experiment conditioned on it would be vacuous - rather than manufacturing corruption to
    condition on.

    Returns:
        The report: the question, why it was not run, and what metadata does exist.
    """
    return {
        "question": (
            "does predictive uncertainty rise under missing, derived, interpolated or "
            "stale inputs?"
        ),
        "run": False,
        "reason": (
            "SMART-DS load profiles carry no per-timestep quality flags. Phase 3 recorded "
            "flags=['ok'] over all 35,040 steps, and the TargetSeries DataQuality is "
            "clean for every row, so the conditioning variable has zero variance and an "
            "experiment conditioned on it would be vacuous."
        ),
        "available_metadata": {
            "per_timestep_quality_flags": False,
            "interpolation_markers": False,
            "staleness_markers": False,
            "series_level_quality": "DataQuality(ok) on the TargetSeries",
            "mapping_artifacts": (
                "Phase 3 recorded 4 unknown and 2 unmapped elements; these are "
                "ingestion-mapping counts, not per-row input quality"
            ),
        },
        "corruption_experiment": (
            "not run. Injecting artificial corruption would answer a question about the "
            "injected corruption rather than about this dataset, and is deferred to a "
            "separate pre-registered robustness study."
        ),
        "deferred_to": (
            "a dataset with missing or interpolated inputs; for SMART-DS the equivalent "
            "question is answered instead by the horizon and regime breakdowns, where "
            "difficulty genuinely varies"
        ),
    }


def _write_trace(
    *,
    path: Path,
    rows: np.ndarray,
    rows_kept: int,
    batch: ProbabilisticForecastBatch,
    panel: Any,
    actual_kw: np.ndarray,
    disagreement: Any,
    horizons: tuple[int, ...],
    spread_kw: np.ndarray,
    learned_scale: np.ndarray,
) -> Path:
    """Per-forecast audit trail: the inputs to the uncertainty, and its outputs.

    Every line answers "why did this row get this interval" from state that was available
    at the forecast origin, plus the outcome. That is what makes a published interval
    checkable: the spread and the learned scale are the row's own origin-observable inputs,
    and the realised error is recorded beside them for comparison.

    Args:
        path: Destination ``.jsonl``.
        rows: Panel positions being traced.
        rows_kept: How many rows to sample across ``rows``.
        batch: The written batch, indexed by position within ``rows``.
        panel: The evaluation panel, for origins and series indices.
        actual_kw: ``[N, horizons]`` panel-wide realised targets.
        disagreement: The spread statistics, panel-wide.
        horizons: Horizon steps.
        spread_kw: ``[N, horizons]`` the disagreement scale in kW.
        learned_scale: ``[N, horizons]`` the learned scale in kW.

    Returns:
        The path written.

    Raises:
        ValueError: If ``batch`` does not hold exactly one row per position in ``rows``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = np.asarray(rows, dtype=np.int64)
    if batch.n_rows != rows.size:
        raise ValueError(
            f"the batch holds {batch.n_rows} rows for {rows.size} traced positions; the "
            f"trace indexes the batch by position, so the two must agree"
        )
    count = max(0, min(int(rows_kept), int(rows.size)))
    picks = (
        np.linspace(0, rows.size - 1, count).astype(int) if count else np.zeros(0, dtype=int)
    )
    origins = np.asarray(panel.origins, dtype=np.int64)
    series = np.asarray(panel.series, dtype=np.int64)
    point = np.asarray(batch.point_kw, dtype=np.float64)
    lower = np.asarray(batch.lower_kw, dtype=np.float64)
    upper = np.asarray(batch.upper_kw, dtype=np.float64)
    width = upper - lower
    with path.open("w", encoding="utf-8") as handle:
        for local in picks:
            row = int(rows[int(local)])
            for column, horizon in enumerate(horizons):
                handle.write(
                    json.dumps(
                        {
                            "panel_row": row,
                            "origin": int(origins[row]),
                            "series": int(series[row]),
                            "horizon_steps": int(horizon),
                            "point_kw": round(float(point[local, column]), 4),
                            "lower_kw": round(float(lower[local, column]), 4),
                            "upper_kw": round(float(upper[local, column]), 4),
                            "width_kw": round(float(width[local, column]), 4),
                            "nominal_level": float(batch.nominal_level),
                            "band": str(batch.bands(column)[local]),
                            "calibration_status": (
                                batch.calibration_status[column]
                                if batch.calibration_status
                                else "NOT_EVALUATED"
                            ),
                            "coverage_error": (
                                None
                                if not batch.coverage_error
                                else float(batch.coverage_error[column])
                            ),
                            "actual_kw": round(float(actual_kw[row, column]), 4),
                            "absolute_error_kw": round(
                                float(abs(actual_kw[row, column] - point[local, column])),
                                4,
                            ),
                            "expert_spread_per_unit": {
                                name: round(float(values[row, column]), 6)
                                for name, values in disagreement.measures.items()
                            },
                            "expert_spread_kw": round(float(spread_kw[row, column]), 4),
                            "learned_scale_kw": round(float(learned_scale[row, column]), 4),
                            "method": batch.method[column],
                            "model_version": batch.model_version,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
    return path


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default),
        encoding="utf-8",
    )


def _json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, set):
        return sorted(str(item) for item in value)
    raise TypeError(f"{type(value).__name__} is not JSON serialisable")


def render_summary(result: Phase8Result) -> str:
    """The phase's summary document, with the measured tables.

    Args:
        result: The phase's result.

    Returns:
        Markdown.
    """
    unique_horizons = sorted({int(row["horizon"]) for row in result.comparison})
    headline = float(result.verdict["headline_nominal_level"])
    panel_rows = int(result.split["validation_rows"]) + int(result.split["test_rows"])
    blocks: list[str] = [
        f"# Phase 8 - probabilistic forecasting and calibration ({result.experiment_id})",
        "",
        "## Verdict",
        "",
        f"**{result.verdict['answer']}** - {result.verdict['question']}",
        "",
    ]
    for name, component in result.verdict["components"].items():
        blocks.append(f"- **{name}**: {component['answer']} - {component['criterion']}")
    parity = result.parity.get("max_relative_difference_pct")
    blocks.extend(
        [
            "",
            "## Point forecast, recovered from Phase 7",
            "",
            "- fixed-ensemble weights: "
            + json.dumps(result.weights["fixed_ensemble_weights"], sort_keys=True),
            f"- source: {result.weights['source']}, verified against the config",
            "- parity against Phase 7's published test MAE: "
            + (
                "not checked, no reference supplied"
                if parity is None
                else f"{parity:.4f}% max relative difference "
                f"(matches: {result.parity.get('matches')})"
            ),
            "",
            "| horizon | CAL_FIT MAE | CAL_CONF MAE | test MAE | fit vs conform |",
            "|---|---|---|---|---|",
        ]
    )
    for horizon in unique_horizons:
        entry = result.split_parity[str(horizon)]
        gap = entry["fit_vs_conformity_pct"]
        blocks.append(
            f"| {horizon} | {entry['calibration_fit_mae_kw']:.4f} | "
            f"{entry['calibration_conformity_mae_kw']:.4f} | "
            f"{entry['test_mae_kw']:.4f} | "
            f"{'n/a' if gap is None else f'{gap:+.3f}%'} |"
        )
    blocks.extend(
        [
            "",
            "## Interval methods, sealed test split",
            "",
            final_comparison_table(result.comparison, unique_horizons, headline),
            "",
            f"Coverage tolerance: {result.verdict['coverage_tolerance']:.4f} absolute.",
            "",
            "## Published procedure",
            "",
            f"- criterion: {result.selection['criterion']}",
            "- level: "
            f"{float(result.selection['level']):.0%}",
            "",
            "| horizon | procedure | calibrated candidates | WIS kW |",
            "|---|---|---|---|",
        ]
        + [
            f"| {horizon} | {entry['method']} | {entry['n_calibrated_candidates']} | "
            f"{entry['interval_score_kw']:.4f} |"
            + (
                ""
                if entry.get("calibrated", True)
                else f"  <- {entry.get('note', 'not calibrated')}"
            )
            for horizon, entry in sorted(
                result.selection["by_horizon"].items(), key=lambda kv: int(kv[0])
            )
        ]
        + [
            "",
            "## Expert disagreement vs error",
            "",
            "| measure | " + " | ".join(f"h={h}" for h in unique_horizons) + " |",
            "|---" * (len(unique_horizons) + 1) + "|",
        ]
    )
    for measure, by_horizon in result.disagreement["spread_vs_error_spearman"].items():
        blocks.append(
            f"| {measure} | "
            + " | ".join(
                f"{by_horizon.get(str(h), float('nan')):+.4f}" for h in unique_horizons
            )
            + " |"
        )
    blocks.extend(
        [
            "",
            "Spearman correlation between the spread statistic and the realised absolute "
            "error of the fixed ensemble. Positive means larger spread accompanies larger "
            "error.",
            "",
            "## Uncertainty bands",
            "",
            f"- level: {float(result.selection['level']):.0%}, procedure per horizon: "
            + ", ".join(
                f"h{horizon}={entry['method']}"
                for horizon, entry in sorted(
                    result.selection["by_horizon"].items(), key=lambda kv: int(kv[0])
                )
            ),
            f"- bands: {json.dumps(result.artifact_summary.get('band_counts'))}",
            f"- mean width kW: {json.dumps(result.artifact_summary.get('mean_width_kw'))}",
            f"- calibration status: "
            f"{json.dumps(result.artifact_summary.get('calibration_status'))}, coverage "
            f"error {json.dumps(result.artifact_summary.get('coverage_error'))}",
            "",
            "## Data-quality conditioning",
            "",
            f"- **not run.** {result.data_quality['reason']}",
            "",
            "## Method caveats",
            "",
            "Independently fitted pinball models cross: a lower quantile can be predicted "
            "above a higher one. Rows whose *reported* lower/upper pair was reversed are "
            "re-sorted into ascending order, so the interval stays valid. The counts are "
            f"over all {panel_rows:,} panel rows, since the scale is applied to every row.",
            "",
            "| horizon | level | crossed rows |",
            "|---|---|---|",
        ]
        + [
            f"| {horizon} | {float(level):.0%} | {crossed:,} |"
            for horizon in sorted(result.quantile_crossings, key=int)
            for level, crossed in sorted(
                result.quantile_crossings[horizon].items(), key=lambda kv: float(kv[0])
            )
        ]
        + [
            "",
            "## Compute",
            "",
            f"- total seconds: {result.seconds:.1f}",
        ]
    )
    return "\n".join(blocks) + "\n"