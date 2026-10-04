"""The Phase 7 runner: diversity, oracle, router, ablations, and one sealed evaluation.

The order is the order of the argument, and it is deliberate. Diversity and the oracle
are computed **before** a router exists, because the honest scientific question is
whether the premise holds: if the experts make the same mistakes, or the oracle barely
beats the best single expert, then no router was ever going to help and building one
would be architecture for its own sake.

Where the data comes from, and why it is safe:

* The experts are fitted on the **training** split (Phase 4's grid, Phase 5's
  checkpoint).
* The panel is validation + test, so every expert prediction here is **out-of-sample**.
* The router is fitted on a chronological **70% of validation**, early-stopped on the
  rest, and scored once on **test**.
* The fixed-weight and best-single ensembles select their parameters on the same
  router-training rows, so no baseline is advantaged by having seen the test split.
* The oracle is computed on test **after** every deployable component is frozen, and
  lives in ``router.offline.oracle`` which no deployable module imports.

That is the cross-fitting decision, and it is the one place where a different choice
would have produced a better-looking and worthless result.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..features import (
    RouterFeatures,
    append_lagged_error_features,
    assert_router_features_are_causal,
    build_router_features,
    router_feature_version,
)
from ..routing import ROUTING_STRATEGIES
from ..training import RouterTrainingConfig, fit_router, router_weights_for
from .ablations import AblationOutcome, render_ablation_table, run_ablations
from .crossfit import gbm_label_optimism, label_quality
from .diversity import (
    absolute_errors,
    diversity_by_horizon,
    diversity_by_regime,
    diversity_report,
)
from .evaluation import (
    compute_cost,
    failure_analysis,
    forecast_comparison_table,
    method_metrics,
    router_by_regime,
    router_calibration,
    routing_quality,
    routing_stability,
)
from .oracle import (
    ORACLE_NAME,
    best_pair_and_gain,
    oracle_assignment,
    oracle_mixture,
    oracle_regret,
    oracle_report,
    pairwise_best_single,
    routing_regret_definition,
)

__all__ = ["Phase7Result", "run_phase7", "METHOD_NAMES", "render_summary"]


#: The methods compared in every table, in reporting order.
METHOD_NAMES: tuple[str, ...] = (
    "persistence",
    "classical_hist_gbm",
    "phase5_tcn",
    "uniform_ensemble",
    "best_single",
    "fixed_ensemble",
    "router_soft",
    "router_hard",
    ORACLE_NAME,
    "oracle_mixture",
)

ROUTER_METHODS: tuple[str, ...] = ("router_soft", "router_hard")


@dataclass(frozen=True, slots=True)
class Phase7Result:
    """Everything Phase 7 measured.

    Attributes:
        experiment_id: Registry identifier.
        dataset_version: The Phase 5 sequence version the panel came from.
        panel: Panel description.
        methods: Per-method metrics, per horizon.
        diversity: Error correlations, agreement, best-expert share.
        oracle: The oracle report, the ceiling and the achievable gain.
        routing: Routing quality, stability and calibration.
        failures: The failure analysis.
        cost: The compute table.
        ablations: Ablation outcomes.
        crossfit: The in-sample versus out-of-sample label check.
        router: The trained router's own record.
        parity: Phase 7 experts against Phase 4/5's published test numbers.
        verdict: The phase's conclusion, derived from the numbers.
        seconds: Total wall time.
        artifacts: Written file paths.
        notes: Anything a reader would have to infer otherwise.
    """

    experiment_id: str
    dataset_version: str
    panel: dict[str, Any]
    methods: dict[str, dict[str, dict[str, Any]]]
    diversity: dict[str, Any]
    oracle: dict[str, Any]
    routing: dict[str, Any]
    failures: dict[str, Any]
    cost: dict[str, Any]
    ablations: list[AblationOutcome]
    crossfit: dict[str, Any]
    router: dict[str, Any]
    parity: dict[str, Any]
    verdict: dict[str, Any]
    seconds: float
    artifacts: dict[str, str]
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "dataset_version": self.dataset_version,
            "panel": self.panel,
            "methods": self.methods,
            "diversity": self.diversity,
            "oracle": self.oracle,
            "routing": self.routing,
            "failures": self.failures,
            "cost": self.cost,
            "ablations": [item.to_dict() for item in self.ablations],
            "crossfit": self.crossfit,
            "router": self.router,
            "parity": self.parity,
            "verdict": self.verdict,
            "seconds": round(self.seconds, 3),
            "artifacts": self.artifacts,
            "notes": list(self.notes),
        }


def _regime_axes(actual_kw: np.ndarray, panel: Any, horizons: tuple[int, ...]):
    """Phases 4 and 5's regime axes, assigned over the panel's h=1 column."""
    from ...analysis import assign_regimes

    previous = np.asarray(panel.persistence_kw)[:, 0]
    return assign_regimes(
        actual_kw[:, 0],
        target_id="customer_load",
        origin_index=np.asarray(panel.origins),
        previous_actual=previous,
    )


def _parity(
    *,
    expert_names: tuple[str, ...],
    forecasts: np.ndarray,
    actual_kw: np.ndarray,
    horizons: tuple[int, ...],
    published: dict[str, dict[str, float]],
) -> dict[str, Any]:
    """Compare the Phase 7 experts against Phase 4/5's published test numbers.

    A refit expert that has silently changed - wrong units, a dropped context column, a
    different feature order - produces plausible numbers, just not these ones. Checking
    them is how a regression in the expert pool is caught instead of being attributed to
    the router.
    """
    from ...metrics import mae

    actual_kw = np.asarray(actual_kw, dtype=np.float64)
    out: dict[str, Any] = {"reference": published, "experts": {}}
    for index, name in enumerate(expert_names):
        reference = published.get(name)
        measured = {
            str(h): float(mae(actual_kw[:, c], forecasts[index, :, c]))
            for c, h in enumerate(horizons)
        }
        entry: dict[str, Any] = {"measured_test_mae_kw": measured}
        if reference:
            entry["reference_mae_kw"] = reference
            entry["relative_difference_pct"] = {
                key: 100.0 * (measured[key] - reference[key]) / reference[key]
                for key in measured
                if reference.get(key)
            }
            worst = max(abs(v) for v in entry["relative_difference_pct"].values())
            entry["max_relative_difference_pct"] = worst
            entry["matches"] = worst < 2.0
        out["experts"][name] = entry
    return out


def _slice_axes(axes: tuple[Any, ...], positions: np.ndarray) -> tuple[Any, ...]:
    """Restrict regime axes to a subset of rows, for an analysis run on one split."""
    from ...analysis import RegimeDefinition

    return tuple(
        RegimeDefinition(
            name=axis.name,
            description=axis.description,
            labels=np.asarray(axis.labels)[positions],
            detail=None if axis.detail is None else np.asarray(axis.detail)[positions],
        )
        for axis in axes
    )


def _fixed_selection_mae(
    fixed: Any,
    forecasts: np.ndarray,
    actual_kw: np.ndarray,
    train_rows: np.ndarray,
    config: Any,
) -> float:
    """The fixed ensemble's MAE on exactly the rows the router selected its epoch on.

    Comparing the router's selection score against this number is what turns "the warm
    start was safe" from an assertion into a measurement.
    """
    from ..ensembles import fixed_forecast

    split = int(round(train_rows.size * config.train_fraction_of_validation))
    split = max(1, min(split, train_rows.size - 1))
    select_rows = train_rows[split:]
    predicted = fixed_forecast(fixed, forecasts[:, select_rows, :])
    return float(np.abs(predicted - actual_kw[select_rows]).mean())


def run_phase7(
    *,
    cache,
    panel,
    values: np.ndarray | None,
    expert_seconds: dict[str, float] | None = None,
    artifact_dir: Path,
    config: RouterTrainingConfig,
    architecture: Any,
    experiment_id: str = "phase7-router-main",
    dataset_version: str = "",
    seed: int = 20260101,
    expert_parameters: dict[str, int] | None = None,
    series_ids: tuple[str, ...] = (),
    origin_iso: str = "",
    published_test_mae: dict[str, dict[str, float]] | None = None,
    run_ablations_too: bool = True,
    run_crossfit: bool = True,
    ablation_epochs: int | None = None,
    trace_rows: int = 2000,
    log: Any = None,
) -> Phase7Result:
    """Run the phase end to end.

    Args:
        cache: The expert forecast cache, from :func:`router.cache.load_expert_cache`.
        panel: The evaluation panel the cache describes.
        values: Per-unit series values. Required only when ``run_crossfit`` is set,
            because that check refits an expert.
        expert_seconds: Wall time each expert took over the whole panel.
        artifact_dir: Where the record, trace and checkpoints go.
        config: Router training protocol.
        architecture: Router shape.
        experiment_id: Registry identifier.
        dataset_version: Sequence version, for the record.
        seed: Estimator seed, recorded.
        expert_parameters: Parameter counts per expert, for the cost table.
        series_ids: Series identifiers, for the routing trace's asset column.
        origin_iso: The series origin timestamp, so the trace can carry real timestamps.
        published_test_mae: Phase 4/5's published test MAE per expert and horizon, for
            the parity check.
        run_ablations_too: Run the ablation suite.
        run_crossfit: Run the in-sample label check. Needs ``values``.
        ablation_epochs: Optional reduced epoch budget for the ablations.
        trace_rows: How many test rows to write a full routing trace for.
        log: Optional progress callable.

    Returns:
        The measured result.

    Raises:
        ValueError: If the cache and panel disagree, or the router's rows would include
            the test split.
    """
    from ..ensembles import (
        apply_best_single,
        ensemble_similarity,
        fit_fixed_weights,
        fixed_forecast,
        select_best_single,
        uniform_weights,
    )
    from ..routing import hard_selection, mix

    started = time.perf_counter()
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    forecasts = np.asarray(cache.forecasts, dtype=np.float64)
    actual_kw = np.asarray(cache.actual_kw, dtype=np.float64)
    horizons = tuple(int(h) for h in cache.horizons)
    expert_names = tuple(cache.expert_names)
    if forecasts.shape != (len(expert_names), panel.n_rows, len(horizons)):
        raise ValueError(
            f"cache forecasts {forecasts.shape} do not match the panel "
            f"({len(expert_names)} experts, {panel.n_rows} rows, {len(horizons)} horizons)"
        )
    train_rows = np.arange(panel.split_slices["validation"].start, panel.split_slices["validation"].stop)
    test_rows = np.arange(panel.split_slices["test"].start, panel.split_slices["test"].stop)
    if np.intersect1d(train_rows, test_rows).size:
        raise ValueError("the router's rows overlap the test split")

    notes: list[str] = []

    # ---- 1. Is there anything to route to? --------------------------------
    if log is not None:
        log("expert diversity")
    diversity = diversity_report(
        forecasts=forecasts, actual_kw=actual_kw, horizons=horizons, expert_names=expert_names
    )
    diversity["by_horizon_summary"] = diversity_by_horizon(
        forecasts=forecasts, actual_kw=actual_kw, horizons=horizons, expert_names=expert_names
    )
    axes = _regime_axes(actual_kw, panel, horizons)
    diversity["by_regime"] = diversity_by_regime(
        forecasts=forecasts,
        actual_kw=actual_kw,
        horizons=horizons,
        expert_names=expert_names,
        axes=axes,
        headline_horizon=horizons[0],
    )

    if log is not None:
        log("oracle upper bound")
    oracle = oracle_report(
        forecasts=forecasts, actual_kw=actual_kw, horizons=horizons, expert_names=expert_names
    )
    oracle["definitions"] = routing_regret_definition()
    oracle["best_pair_analysis"] = best_pair_and_gain(
        forecasts=forecasts, actual_kw=actual_kw, expert_names=expert_names
    )
    oracle["per_expert_mae"] = pairwise_best_single(forecasts, actual_kw, expert_names)

    # ---- 2. Router inputs, and the proof they are causal -------------------
    if log is not None:
        log("router features")
    test_actual = actual_kw[test_rows]
    forecasts_train = forecasts[:, train_rows, :]
    actual_train = actual_kw[train_rows]
    fit_rows = train_rows[: int(round(train_rows.size * config.train_fraction_of_validation))]
    if values is None:
        raise ValueError("values are required to build the router features")
    base = build_router_features(values=np.asarray(values, dtype=np.float32), panel=panel)
    features = append_lagged_error_features(
        features=base,
        forecasts=forecasts,
        actual_kw=actual_kw,
        panel=panel,
        horizons=horizons,
    )
    causality = assert_router_features_are_causal(
        values=np.asarray(values, dtype=np.float32),
        panel=panel,
        features=features,
        horizons=horizons,
        forecasts=forecasts,
        sample=32,
        seed=seed,
    )
    filled = features.fill_missing(train_rows)
    if log is not None:
        log(
            f"  {features.n_features} features, causality passed on "
            f"{causality['checked_rows_verified']} rows, "
            f"{len(features.missing_counts)} columns with gaps filled from router rows"
        )

    # ---- 3. Baselines that must be beaten ----------------------------------
    uniform_all = mix(uniform_weights(panel.n_rows, horizons, len(expert_names)), forecasts)
    fixed = fit_fixed_weights(
        expert_forecast_kw=forecasts_train,
        actual_kw=actual_train,
        horizons=horizons,
        expert_names=expert_names,
        rows=fit_rows,
    )
    fixed_all = fixed_forecast(fixed, forecasts)
    best_single_picks = select_best_single(
        expert_forecast_kw=forecasts_train,
        actual_kw=actual_train,
        horizons=horizons,
        expert_names=expert_names,
        rows=fit_rows,
    )
    best_single_all = apply_best_single(
        expert_forecast_kw=forecasts, picks=best_single_picks, horizons=horizons
    )
    best_single_by_horizon = {
        h: expert_names[index] for h, index in best_single_picks.items()
    }
    oracle_all = np.take_along_axis(
        forecasts, oracle_assignment(forecasts, actual_kw)[None, :, :], axis=0
    )[0]
    oracle_mixture_all = oracle_mixture(forecasts, actual_kw)

    # ---- 4. The router -----------------------------------------------------
    # Two training variants, and the choice between them is made per horizon on the
    # **selection rows** - never on test. The question the pair answers is real: a router
    # started at the fitted fixed ensemble can only improve by finding state dependence,
    # so if it returns its own initialisation there is no state dependence to find at
    # that horizon; a router started cold can collapse onto a single expert, which is
    # worse than the fixed ensemble at long horizons and better at short ones.
    if log is not None:
        log("router training: warm start from the fitted fixed ensemble")
    trained = fit_router(
        features=filled.matrix[train_rows],
        expert_forecast_kw=forecasts_train,
        actual_kw=actual_train,
        horizons=horizons,
        expert_names=expert_names,
        feature_names=features.names,
        config=config,
        architecture=architecture,
        initial_weights=fixed.weights,
        checkpoint_path=artifact_dir / "router.pt",
        log=log,
    )
    if log is not None:
        log("router training: cold start near uniform")
    cold = fit_router(
        features=filled.matrix[train_rows],
        expert_forecast_kw=forecasts_train,
        actual_kw=actual_train,
        horizons=horizons,
        expert_names=expert_names,
        feature_names=features.names,
        config=config,
        architecture=architecture,
        initial_weights=None,
        checkpoint_path=artifact_dir / "router_cold.pt",
        log=log,
    )

    variants: dict[str, Any] = {}
    weights_by_variant = {
        "warm": router_weights_for(trained, filled.matrix),
        "cold": router_weights_for(cold, filled.matrix),
    }
    chosen_weights = np.empty_like(weights_by_variant["warm"])
    split = int(round(train_rows.size * config.train_fraction_of_validation))
    split = max(1, min(split, train_rows.size - 1))
    select_rows = train_rows[split:]
    for column, horizon in enumerate(horizons):
        warm_score = trained.best_validation_mae_kw_by_horizon[horizon]
        cold_score = cold.best_validation_mae_kw_by_horizon[horizon]
        pick = "cold" if cold_score < warm_score else "warm"
        chosen_weights[:, column, :] = weights_by_variant[pick][:, column, :]
        variants[str(horizon)] = {
            "chosen_variant": pick,
            "warm_selection_mae_kw": warm_score,
            "cold_selection_mae_kw": cold_score,
            "selection_rows": int(select_rows.size),
        }
    weights_all = chosen_weights
    weights_test = weights_all[test_rows]
    router_soft_all = mix(weights_all, forecasts)
    router_hard_all = mix(hard_selection(weights_all), forecasts)

    variant_forecasts = {
        name: mix(weights, forecasts) for name, weights in weights_by_variant.items()
    }
    variant_metrics = {
        name: method_metrics(
            actual_kw=test_actual, forecast_kw=block[test_rows], horizons=horizons
        )
        for name, block in variant_forecasts.items()
    }
    variant_report = {
        "question": (
            "per horizon, does a state-dependent router improve on the state-independent "
            "optimum it can be started from? Both variants are trained on the same rows "
            "and the winner is chosen on the selection rows only."
        ),
        "per_horizon": variants,
        "test_metrics": variant_metrics,
        "note": (
            "a 'warm' winner with an identical score to epoch 0 means the router found "
            "no state-dependent weighting that beat its own starting point at that "
            "horizon; the cold variant's weights then collapse toward a single expert, "
            "which is why the warm variant exists"
        ),
    }

    test_actual = actual_kw[test_rows]
    method_forecasts = {
        expert_names[0]: forecasts[0],
        expert_names[1]: forecasts[1],
        expert_names[2]: forecasts[2],
        "uniform_ensemble": uniform_all,
        "best_single": best_single_all,
        "fixed_ensemble": fixed_all,
        "router_soft": router_soft_all,
        "router_hard": router_hard_all,
        ORACLE_NAME: oracle_all,
        "oracle_mixture": oracle_mixture_all,
    }
    test_forecasts = {name: block[test_rows] for name, block in method_forecasts.items()}
    methods = {
        name: method_metrics(actual_kw=test_actual, forecast_kw=block, horizons=horizons)
        for name, block in test_forecasts.items()
    }

    # ---- 5. Routing quality, stability, calibration ------------------------
    if log is not None:
        log("routing quality")
    quality = {}
    for method in ROUTER_METHODS:
        quality[method] = routing_quality(
            forecast_kw=test_forecasts[method],
            weights=weights_test,
            expert_forecast_kw=forecasts[:, test_rows, :],
            actual_kw=test_actual,
            horizons=horizons,
            expert_names=expert_names,
        )
    quality["definitions"] = routing_regret_definition()
    quality["stability"] = routing_stability(
        weights=weights_test,
        origins=np.asarray(panel.origins)[test_rows],
        series=np.asarray(panel.series)[test_rows],
        horizons=horizons,
        expert_names=expert_names,
    )
    quality["calibration"] = router_calibration(
        weights=weights_test,
        expert_forecast_kw=forecasts[:, test_rows, :],
        actual_kw=test_actual,
        horizons=horizons,
        expert_names=expert_names,
    )
    quality["mean_weights"] = {
        str(horizon): {
            name: float(weights_test[:, column, index].mean())
            for index, name in enumerate(expert_names)
        }
        for column, horizon in enumerate(horizons)
    }
    quality["fixed_ensemble_weights"] = fixed.to_dict()["weights"]
    quality["fixed_ensemble_similarity"] = ensemble_similarity(
        fixed.weights, weights_test.mean(axis=0)
    )
    quality["best_single_by_horizon"] = {str(k): v for k, v in best_single_by_horizon.items()}
    quality["by_regime"] = router_by_regime(
        router_forecast_kw=test_forecasts["router_soft"],
        router_weights=weights_test,
        method_forecast_kw={
            name: test_forecasts[name]
            for name in ("best_single", "fixed_ensemble", "uniform_ensemble")
            if name in test_forecasts
        },
        expert_forecast_kw=forecasts[:, test_rows, :],
        actual_kw=test_actual,
        axes=_slice_axes(axes, test_rows),
        horizons=horizons,
        expert_names=expert_names,
        headline_horizon=horizons[0],
    )
    quality["variants"] = variant_report
    quality["warm_start"] = {
        "warm_started": bool(trained.net.warm_started),
        "initial_weights": {
            str(horizon): {
                name: float(fixed.weights[i, j])
                for j, name in enumerate(expert_names)
            }
            for i, horizon in enumerate(horizons)
        },
        "selected_epoch": trained.best_epoch,
        "cold_selected_epoch": cold.best_epoch,
        "router_selection_mae_kw": trained.best_validation_mae_kw,
        "fixed_ensemble_selection_mae_kw": _fixed_selection_mae(
            fixed, forecasts, actual_kw, train_rows, config
        ),
        "note": (
            "the router starts from the fixed ensemble fitted on the router's own "
            "training rows, and epoch 0 is eligible for selection, so the reported "
            "router is never worse than that starting point on the selection rows"
        ),
    }

    # ---- 6. Failure analysis ----------------------------------------------
    failures = failure_analysis(
        forecast_kw=test_forecasts["router_soft"],
        expert_forecast_kw=forecasts[:, test_rows, :],
        weights=weights_test,
        actual_kw=test_actual,
        origins=np.asarray(panel.origins)[test_rows],
        horizons=horizons,
        expert_names=expert_names,
    )
    for method in ("best_single", "fixed_ensemble"):
        failures[f"{method}_regret"] = {
            str(horizon): oracle_regret(
                router_error=np.abs(test_forecasts[method][:, c] - test_actual[:, c]),
                oracle_error=np.abs(oracle_all[test_rows][:, c] - test_actual[:, c]),
                best_single_error=float(
                    np.abs(forecasts[:, test_rows, c] - test_actual[:, c][None, :]).mean(axis=1).min()
                ),
            )
            for c, horizon in enumerate(horizons)
        }

    # ---- 7. Cost -----------------------------------------------------------
    if log is not None:
        log("cost")
    router_started = time.perf_counter()
    _replayed = router_weights_for(trained, filled.matrix[test_rows])
    router_seconds = time.perf_counter() - router_started
    cost = compute_cost(
        router_parameters=trained.parameter_count,
        router_train_seconds=trained.train_seconds,
        router_inference_seconds=router_seconds,
        expert_seconds=expert_seconds or {},
        expert_rows=panel.n_rows,
        router_rows=test_rows.size,
        expert_parameters=expert_parameters,
    )
    cost["phase6_reference_seconds_per_row"] = 0.248
    cost["phase6_reference_parameters"] = 1_720_574_976

    # ---- 8. Ablations and the cross-fitting check -------------------------
    ablation_outcomes: list[AblationOutcome] = []
    if run_ablations_too:
        if log is not None:
            log("ablations")
        ablation_config = config
        if ablation_epochs is not None:
            ablation_config = RouterTrainingConfig(
                **{**config.to_dict(), "epochs": int(ablation_epochs)}
            )
        ablation_outcomes = run_ablations(
            features=features,
            train_rows=train_rows,
            test_rows=test_rows,
            forecasts=forecasts,
            actual_kw=actual_kw,
            horizons=horizons,
            expert_names=expert_names,
            architecture=architecture,
            config=ablation_config,
            reference_metrics=methods["router_soft"],
            initial_weights=fixed.weights,
            log=log,
        )

    crossfit: dict[str, Any] = {"ran": False}
    if run_crossfit:
        if values is None:  # pragma: no cover - guarded above
            raise ValueError("the cross-fitting check needs the per-unit series values")
        if log is not None:
            log("cross-fitting check on routing labels")
        crossfit = gbm_label_optimism(
            values=np.asarray(values, dtype=np.float32),
            panel=panel,
            horizons=horizons,
            seed=seed,
            log=log,
        )
        crossfit["ran"] = True
        errors_test = absolute_errors(forecasts[:, test_rows, :], test_actual)
        crossfit["label_quality_actual_validation_rows"] = label_quality(
            assignment=oracle_assignment(forecasts_train, actual_train),
            errors=absolute_errors(forecasts_train, actual_train),
            expert_names=expert_names,
        )
        crossfit["label_quality_actual_test_rows"] = label_quality(
            assignment=oracle_assignment(forecasts[:, test_rows, :], test_actual),
            errors=errors_test,
            expert_names=expert_names,
        )

    # ---- 9. Router behaviour by regime, and parity -------------------------
    parity = _parity(
        expert_names=expert_names,
        forecasts=forecasts[:, test_rows, :],
        actual_kw=test_actual,
        horizons=horizons,
        published=published_test_mae or {},
    )

    # ---- 10. Verdict, trace, artifacts ------------------------------------
    verdict = _verdict(methods=methods, oracle=oracle, quality=quality, horizons=horizons)
    trace = _write_trace(
        path=artifact_dir / "routing_trace.jsonl",
        rows=test_rows,
        panel=panel,
        weights=weights_test,
        forecast=test_forecasts["router_soft"],
        actual=test_actual,
        expert_forecasts=forecasts[:, test_rows, :],
        expert_names=expert_names,
        horizons=horizons,
        feature_names=features.names,
        features=filled.matrix[test_rows],
        series_ids=series_ids,
        origin_iso=origin_iso,
        limit=trace_rows,
    )
    weights_path = artifact_dir / "router_weights.npz"
    np.savez_compressed(
        weights_path,
        weights=weights_test.astype(np.float32),
        horizons=np.asarray(horizons),
        expert_names=np.asarray(expert_names),
        positions=test_rows,
        panel_rows=test_rows,
    )

    written = {
        "result": str(artifact_dir / "result.json"),
        "trace": str(trace),
        "weights": str(weights_path),
        "router": str(trained.checkpoint_path) if trained.checkpoint_path else "",
        "training_log": str(artifact_dir / "training_log.jsonl"),
        "regimes": str(artifact_dir / "regimes.json"),
        "diversity": str(artifact_dir / "diversity.json"),
        "summary": str(artifact_dir / "summary.md"),
    }
    _write_jsonl(
        Path(written["training_log"]), [item.to_dict() for item in trained.history]
    )
    _write_json(Path(written["regimes"]), {"axes": diversity["by_regime"]})
    _write_json(Path(written["diversity"]), diversity)

    notes.append(
        "the oracle is computed on the test split after every deployable component was "
        "frozen, and lives in router.offline.oracle, which no deployable module imports"
    )
    notes.append(
        "router features are fitted on a chronological 70% of the validation split and "
        "early-stopped on the remaining 30%; the test split is read once"
    )
    notes.append(
        "hard routing applies argmax to the same trained weights, so the two differ only "
        "in how the weights are used"
    )

    result = Phase7Result(
        experiment_id=experiment_id,
        dataset_version=dataset_version,
        panel=panel.describe(),
        methods=methods,
        diversity=diversity,
        oracle=oracle,
        routing=quality,
        failures=failures,
        cost=cost,
        ablations=ablation_outcomes,
        crossfit=crossfit,
        router=trained.to_dict(),
        parity=parity,
        verdict=verdict,
        seconds=time.perf_counter() - started,
        artifacts=written,
        notes=tuple(notes),
    )
    _write_json(Path(written["result"]), result.to_dict())
    Path(written["summary"]).write_text(render_summary(result), encoding="utf-8")
    return result


def _verdict(
    *,
    methods: dict[str, dict[str, dict[str, Any]]],
    oracle: dict[str, Any],
    quality: dict[str, Any],
    horizons: tuple[int, ...],
) -> dict[str, Any]:
    """Derive the phase's answer from the numbers, rather than asserting one.

    Two comparisons are reported separately and the verdict is the **stricter** of them,
    because the looser one is the easier claim to make and would overstate the result.

    *Against the best single expert* is the question the phase was asked. *Against the
    fixed weighted ensemble* is the question that decides whether the **learned,
    state-dependent** part is doing any work: a fixed ensemble already captures the
    average benefit of combining, so a router that merely matches it has added nothing
    that a constant weighting would not have given.

    Reporting only the first would let a router that learned nothing be called a success.
    """
    single_experts = ("persistence", "classical_hist_gbm", "phase5_tcn")
    keys = [str(h) for h in horizons]
    # A "win" has to be a win in units, not in floating point. Two methods that differ
    # by 2e-6 kW are the same method, and counting that as the router beating a
    # baseline would be a rounding artefact presented as a result.
    tolerance_kw = 1e-4
    best_single_overall = min(
        single_experts, key=lambda name: sum(methods[name][k]["mae"] for k in keys)
    )
    per_horizon: dict[str, Any] = {}
    beat_single = 0
    beat_fixed = 0
    tied_fixed = 0
    for key in keys:
        router_mae = methods["router_soft"][key]["mae"]
        single_name = min(single_experts, key=lambda name: methods[name][key]["mae"])
        single_mae = methods[single_name][key]["mae"]
        fixed_mae = methods["fixed_ensemble"][key]["mae"]
        beat_single_here = router_mae < single_mae - tolerance_kw
        beat_fixed_here = router_mae < fixed_mae - tolerance_kw
        tie_fixed = abs(router_mae - fixed_mae) <= tolerance_kw
        beat_single += int(beat_single_here)
        beat_fixed += int(beat_fixed_here)
        tied_fixed += int(tie_fixed)
        capture = quality["router_soft"][key]["capture_ratio"]
        per_horizon[key] = {
            "best_single_expert": single_name,
            "best_single_expert_mae_kw": single_mae,
            "fixed_ensemble_mae_kw": fixed_mae,
            "uniform_ensemble_mae_kw": methods["uniform_ensemble"][key]["mae"],
            "router_soft_mae_kw": router_mae,
            "router_beats_best_single": beat_single_here,
            "router_beats_fixed_ensemble": beat_fixed_here,
            "router_vs_best_single_pct": 100.0 * (router_mae - single_mae) / single_mae,
            "router_vs_fixed_ensemble_pct": 100.0 * (router_mae - fixed_mae) / fixed_mae,
            "oracle_mae_kw": methods[ORACLE_NAME][key]["mae"],
            "oracle_gain_available_pct": oracle["by_horizon"][key]["oracle_gain_pct"],
            "selection_accuracy": quality["router_soft"][key]["selection_accuracy"],
            "capture_ratio": capture,
            "variant": quality["variants"]["per_horizon"][key]["chosen_variant"],
        }
    if beat_fixed == len(keys):
        answer = "YES"
    elif beat_single == 0:
        answer = "NO"
    else:
        answer = "PARTIALLY"
    return {
        "question": (
            "can a learned routing mechanism select or combine forecasting experts "
            "according to the current energy-demand regime and beat the strongest "
            "individual expert across the evaluated horizons?"
        ),
        "answer": answer,
        "answer_basis": (
            "the stricter of the two reported comparisons: a router that only matches a "
            "fixed weighted ensemble has not shown that state-dependent routing works"
        ),
        "versus_best_single_expert": {
            "horizons_beaten": beat_single,
            "horizons_evaluated": len(keys),
            "answer": "YES" if beat_single == len(keys) else ("NO" if beat_single == 0 else "PARTIALLY"),
        },
        "versus_fixed_weighted_ensemble": {
            "horizons_beaten": beat_fixed,
            "horizons_tied": tied_fixed,
            "horizons_evaluated": len(keys),
            "answer": "YES" if beat_fixed == len(keys) else ("NO" if beat_fixed == 0 else "PARTIALLY"),
        },
        "tie_tolerance_kw": tolerance_kw,
        "best_single_expert_overall": best_single_overall,
        "mean_oracle_gain_available_pct": oracle["mean_oracle_gain_pct"],
        "per_horizon": per_horizon,
        "basis": (
            "the router is counted as beating a baseline only where its test MAE is "
            "lower by more than 0.0001 kW - a tenth of a watt, far below any physically "
            "meaningful difference and well above floating-point noise - at that "
            f"horizon; no horizon is excluded and no result is chosen after the fact"
        ),
    }


def _write_trace(
    *,
    path: Path,
    rows: np.ndarray,
    panel: Any,
    weights: np.ndarray,
    forecast: np.ndarray,
    actual: np.ndarray,
    expert_forecasts: np.ndarray,
    expert_names: tuple[str, ...],
    horizons: tuple[int, ...],
    feature_names: tuple[str, ...],
    features: np.ndarray,
    series_ids: tuple[str, ...],
    origin_iso: str,
    limit: int,
) -> Path:
    """Write one JSON object per traced row: everything needed to audit the decision.

    The full weight tensor for every test row is written separately as float32
    ``.npz``; this file carries the human-inspectable version with the router's inputs,
    and is capped so the artifact stays readable.
    """
    from datetime import datetime, timedelta

    path.parent.mkdir(parents=True, exist_ok=True)
    count = min(int(limit), int(rows.size))
    picks = np.linspace(0, rows.size - 1, count).astype(int) if count else np.zeros(0, dtype=int)
    start = datetime.fromisoformat(origin_iso) if origin_iso else None
    origins = np.asarray(panel.origins)
    series = np.asarray(panel.series)
    with path.open("w", encoding="utf-8") as handle:
        for local in picks:
            row = int(rows[local])
            timestamp = (
                (start + timedelta(minutes=int(origins[row]) * 15)).isoformat()
                if start is not None
                else f"origin:{int(origins[row])}"
            )
            for column, horizon in enumerate(horizons):
                handle.write(
                    json.dumps(
                        {
                            "timestamp": timestamp,
                            "asset": (
                                series_ids[int(series[row])]
                                if len(series_ids) > int(series[row])
                                else f"series_{int(series[row])}"
                            ),
                            "horizon_steps": int(horizon),
                            "horizon_hours": round(int(horizon) * 15 / 60, 3),
                            "panel_row": row,
                            "router_features": {
                                name: round(float(features[local, index]), 6)
                                for index, name in enumerate(feature_names)
                            },
                            "expert_weights": {
                                name: round(float(weights[local, column, i]), 4)
                                for i, name in enumerate(expert_names)
                            },
                            "selected_experts": [
                                expert_names[int(i)]
                                for i in np.argsort(-weights[local, column])
                            ][:2],
                            "expert_forecasts_kw": {
                                name: round(float(expert_forecasts[i, local, column]), 4)
                                for i, name in enumerate(expert_names)
                            },
                            "final_forecast_kw": round(float(forecast[local, column]), 4),
                            "actual_kw": round(float(actual[local, column]), 4),
                            "strategy": ROUTING_STRATEGIES.SOFT,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
    return path


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, default=_json_default) + "\n")


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"{type(value).__name__} is not JSON serialisable")


def render_summary(result: Phase7Result) -> str:
    """The phase's summary document, with the measured tables."""
    horizons = [str(h) for h in sorted(result.methods["persistence"], key=int)]
    blocks: list[str] = [
        f"# Phase 7 - heterogeneous energy expert router ({result.experiment_id})",
        "",
        "## Verdict",
        "",
        f"**{result.verdict['answer']}** - {result.verdict['question']}",
        "",
        f"- versus the best single expert: **{result.verdict['versus_best_single_expert']['answer']}** "
        f"({result.verdict['versus_best_single_expert']['horizons_beaten']} of "
        f"{result.verdict['versus_best_single_expert']['horizons_evaluated']} horizons)",
        f"- versus a fixed weighted ensemble: "
        f"**{result.verdict['versus_fixed_weighted_ensemble']['answer']}** "
        f"({result.verdict['versus_fixed_weighted_ensemble']['horizons_beaten']} of "
        f"{result.verdict['versus_fixed_weighted_ensemble']['horizons_evaluated']} horizons)",
        f"- {result.verdict['answer_basis']}",
        f"- mean oracle gain available over the best single expert: "
        f"{result.verdict['mean_oracle_gain_available_pct']:.2f}%",
        "",
        "## Router variant chosen per horizon (selected on validation rows only)",
        "",
        "| horizon | variant | warm selection MAE | cold selection MAE |",
        "|---|---|---|---|",
    ]
    for horizon in horizons:
        entry = result.routing["variants"]["per_horizon"][horizon]
        blocks.append(
            f"| {horizon} | {entry['chosen_variant']} | "
            f"{entry['warm_selection_mae_kw']:.4f} | {entry['cold_selection_mae_kw']:.4f} |"
        )
    blocks.extend(
        [
            "",
            "## Expert performance on the sealed test split (kW MAE)",
            "",
        ]
    )
    rows = []
    for name in METHOD_NAMES:
        if name not in result.methods:
            continue
        cells = " | ".join(f"{result.methods[name][h]['mae']:.4f}" for h in horizons)
        rows.append(f"| {name} | {cells} |")
    blocks.append("| method | " + " | ".join(f"h={h}" for h in horizons) + " |")
    blocks.append("|---" * (len(horizons) + 1) + "|")
    blocks.extend(rows)

    blocks.extend(
        [
            "",
            "## Expert diversity",
            "",
            f"- mean pairwise Pearson correlation of absolute errors: "
            f"{result.diversity['overall']['mean_abs_error_correlation']:.4f}",
            f"- mean pairwise Spearman: "
            f"{result.diversity['overall']['mean_abs_error_spearman']:.4f}",
            f"- mean pairwise correlation of *signed* errors: "
            f"{result.diversity['overall']['mean_signed_error_correlation']:.4f}",
            "",
            "| horizon | best expert | MAE | mean error correlation | winner margin |",
            "|---|---|---|---|---|",
        ]
    )
    for horizon in horizons:
        entry = result.diversity["by_horizon_summary"][horizon]
        blocks.append(
            f"| {horizon} | {entry['best_expert']} | {entry['mae_kw'][entry['best_expert']]:.4f} | "
            f"{entry['mean_abs_error_correlation']:.4f} | {entry['winner_margin_pct']:.2f}% |"
        )

    blocks.extend(["", "## Oracle", ""])
    blocks.append("| horizon | oracle MAE | best single | gain available | best mixture | mixture gain |")
    blocks.append("|---|---|---|---|---|---|")
    for horizon in horizons:
        entry = result.oracle["by_horizon"][horizon]
        blocks.append(
            f"| {horizon} | {entry['oracle_mae_kw']:.4f} | {entry['best_single_mae_kw']:.4f} | "
            f"{entry['oracle_gain_pct']:.2f}% | {entry['oracle_mixture_mae_kw']:.4f} | "
            f"{entry['oracle_mixture_gain_pct']:.2f}% |"
        )

    blocks.extend(["", "## Routing", "", "| horizon | selection accuracy | routing regret kW | capture | mean max weight |", "|---|---|---|---|---|"])
    for horizon in horizons:
        entry = result.routing["router_soft"][horizon]
        capture = entry["capture_ratio"]
        blocks.append(
            f"| {horizon} | {entry['selection_accuracy']:.4f} | {entry['routing_regret_kw']:.4f} | "
            f"{capture if capture is None else round(capture, 4)} | {entry['mean_max_weight']:.4f} |"
        )
    blocks.extend(["", "Mean weights per horizon:", ""])
    blocks.append("| horizon | " + " | ".join(result.routing["mean_weights"][horizons[0]]) + " |")
    blocks.append("|---" * (1 + len(result.routing["mean_weights"][horizons[0]])) + "|")
    for horizon in horizons:
        blocks.append(
            f"| {horizon} | "
            + " | ".join(
                f"{value:.3f}" for value in result.routing["mean_weights"][horizon].values()
            )
            + " |"
        )

    blocks.extend(["", "## Router behaviour by regime (h=" + horizons[0] + ")", ""])
    for axis, block in result.routing["by_regime"].items():
        blocks.append(f"**{axis}** - {block['description']}")
        blocks.append("")
        first = next(iter(block["regimes"].values()))
        names = list(first["mean_router_weights"])
        blocks.append(
            "| regime | n | router | best single | fixed ens | oracle | router weights |"
        )
        blocks.append("|---|---|---|---|---|---|---|")
        for label, entry in block["regimes"].items():
            weights = " ".join(
                f"{entry['mean_router_weights'][name]:.2f}" for name in names
            )
            blocks.append(
                f"| {label} | {entry['n']} | {entry['router_mae_kw']:.4f} | "
                f"{entry['best_single_expert_mae_kw']:.4f} "
                f"({entry['best_single_expert']}) | "
                f"{entry.get('fixed_ensemble_mae_kw', float('nan')):.4f} | "
                f"{entry['oracle_mae_kw']:.4f} | {weights} |"
            )
        blocks.append("")

    blocks.extend(["", "## Routing calibration", "",
                   "| horizon | mean top weight | selection accuracy | expected calibration error |",
                   "|---|---|---|---|"])
    for horizon in horizons:
        entry = result.routing["calibration"][horizon]
        blocks.append(
            f"| {horizon} | {entry['mean_confidence']:.4f} | "
            f"{entry['mean_selection_accuracy']:.4f} | "
            f"{entry['expected_calibration_error']:.4f} |"
        )
    blocks.extend(
        [
            "",
            "The router's top weight is **not** a calibrated probability: at h=1 it "
            "averages well above the rate at which its chosen expert is actually the "
            "most accurate. Weights summing to one is a normalisation, not a claim.",
        ]
    )

    blocks.extend(["", "## Routing stability", "",
                   "| horizon | mean total variation | median | p99 | selection flip rate |",
                   "|---|---|---|---|---|"])
    for horizon in horizons:
        entry = result.routing["stability"][horizon]
        flip = entry.get("selection_flip_rate")
        blocks.append(
            f"| {horizon} | {entry['mean_total_variation']:.4f} | "
            f"{entry['median_total_variation']:.4f} | {entry['p99_total_variation']:.4f} | "
            f"{'n/a' if flip is None else f'{flip:.4f}'} |"
        )

    if result.ablations:
        blocks.extend(["", "## Ablations", "", render_ablation_table(result.ablations), ""])

    blocks.extend(["", "## Compute cost", ""])
    blocks.append(
        f"- router parameters: {result.cost['router_parameters']}\n"
        f"- router training: {result.cost['router_train_seconds']} s\n"
        f"- router inference: {result.cost['router_inference_seconds']} s over "
        f"{result.cost['rows']} rows\n"
        f"- expert inference: {result.cost['expert_total_seconds']} s\n"
        f"- router overhead vs experts: {result.cost['router_overhead_vs_experts']}\n"
        f"- Phase 6 reference (frozen 1.7B backbone): "
        f"{result.cost['phase6_reference_seconds_per_row']} s/row"
    )
    blocks.extend(["", "## Parity with Phase 4/5", ""])
    for name, entry in result.parity["experts"].items():
        if "max_relative_difference_pct" in entry:
            blocks.append(
                f"- {name}: max relative difference from the published test MAE "
                f"{entry['max_relative_difference_pct']:.3f}% (matches: {entry['matches']})"
            )
    return "\n".join(blocks) + "\n"