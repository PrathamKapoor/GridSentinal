"""Ablations, each of which answers one question and no more.

Four, and the reason each exists is recorded with it. An ablation whose question cannot
be stated is architecture for its own sake.

``no_past_error``
    Does the router need to know how the experts have been *doing*? This is the only
    feature group that is not a property of the demand, so if the router does not need
    it, the system can run without any feedback from the experts at all.

``no_trajectory``
    Is the recent demand path doing the work, or is the router only reading the current
    level? Phase 6 found the absolute-time channels mildly harmful at short range; the
    trajectory is the router's version of that question.

``no_scale``
    Does the router need to know how expensive a mistake is in kW? A pooled router over
    40 customers sees a 1.4 kW household and a 388 kW shop as the same row unless told
    otherwise. Phase 4 needed the same columns for the same reason.

``shared_horizon_head``
    One weight vector for all horizons instead of one per horizon, with the **inputs
    unchanged**. That last point is what makes it an answer rather than a confound: a
    shared head still sees every feature, including the per-horizon lagged errors, so
    the only thing being tested is whether one mapping can serve all three horizons.

Hard versus soft routing needs no ablation run. It is the same trained weights applied
as an ``argmax``, so the main experiment produces both and reports them side by side.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..features import FEATURE_GROUPS, RouterFeatureSpec
from ..training import RouterTrainingConfig, fit_router, router_weights_for

__all__ = ["ABLATIONS", "AblationSpec", "AblationOutcome", "run_ablations", "render_ablation_table"]


@dataclass(frozen=True, slots=True)
class AblationSpec:
    """One ablation's feature contract.

    Attributes:
        name: Identifier, used in the artifact record.
        question: What it is trying to find out. Required.
        drop_groups: Feature groups removed.
        include_past_error: Keep or drop the lagged expert errors.
        include_level_kw: Keep or drop the raw kW level.
        shared_head: Use a single weight vector across horizons.
        warm_start: Start from the fitted fixed ensemble rather than near uniform.
    """

    name: str
    question: str
    drop_groups: tuple[str, ...] = ()
    include_past_error: bool = True
    include_level_kw: bool = True
    shared_head: bool = False
    warm_start: bool = True

    def to_feature_spec(self) -> RouterFeatureSpec:
        """The feature groups this ablation may see."""
        groups = tuple(g for g in FEATURE_GROUPS if g not in self.drop_groups)
        return RouterFeatureSpec(
            groups=groups,
            include_level_kw=self.include_level_kw,
            include_past_error=self.include_past_error,
        )


#: The ablations the phase runs, with their reasons.
ABLATIONS: tuple[AblationSpec, ...] = (
    AblationSpec(
        name="no_past_error",
        question=(
            "does the router need to know how the experts have recently been performing, "
            "or is the demand state enough?"
        ),
        include_past_error=False,
    ),
    AblationSpec(
        name="no_trajectory",
        question=(
            "is the recent demand path carrying the routing decision, or only the "
            "current level?"
        ),
        drop_groups=("trajectory",),
    ),
    AblationSpec(
        name="no_scale",
        question=(
            "does the router need the customer's rated kW to know how expensive a "
            "mistake would be?"
        ),
        drop_groups=("scale",),
    ),
    AblationSpec(
        name="shared_horizon_head",
        question=(
            "do the experts' relative strengths really change with horizon, or does one "
            "weight vector serve all three?"
        ),
        shared_head=True,
    ),
)


@dataclass(frozen=True, slots=True)
class AblationOutcome:
    """One ablation's measured result, next to the main router's on identical rows.

    Attributes:
        name: Ablation name.
        question: What it asked.
        metrics_by_horizon: MAE and RMSE per horizon on the test split.
        reference_metrics_by_horizon: The main router's, on the same rows.
        delta_vs_reference: MAE change per horizon. Negative means the ablation is
            *better*, which is a finding rather than a failure.
        best_epoch: Selected epoch.
        best_validation_mae_kw: Early-stopping score.
        train_seconds: Wall time.
        parameters: Router parameter count.
        epochs_run: Epochs run before stopping.
        n_features: Columns the ablation's router saw.
        mean_weights: The ablation's mean weight per horizon.
        notes: Anything a reader would otherwise have to infer.
    """

    name: str
    question: str
    metrics_by_horizon: dict[str, dict[str, Any]]
    reference_metrics_by_horizon: dict[str, dict[str, Any]]
    delta_vs_reference: dict[str, float | None]
    best_epoch: int
    best_validation_mae_kw: float
    train_seconds: float
    parameters: int
    epochs_run: int
    n_features: int
    mean_weights: dict[str, dict[str, float]] = field(default_factory=dict)
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "question": self.question,
            "metrics_by_horizon": self.metrics_by_horizon,
            "reference_metrics_by_horizon": self.reference_metrics_by_horizon,
            "delta_vs_reference_kw": self.delta_vs_reference,
            "best_epoch": self.best_epoch,
            "best_validation_mae_kw": self.best_validation_mae_kw,
            "train_seconds": round(self.train_seconds, 3),
            "parameters": self.parameters,
            "epochs_run": self.epochs_run,
            "n_features": self.n_features,
            "mean_weights": self.mean_weights,
            "notes": list(self.notes),
        }


def run_ablations(
    *,
    features,
    train_rows: np.ndarray,
    test_rows: np.ndarray,
    forecasts: np.ndarray,
    actual_kw: np.ndarray,
    horizons: tuple[int, ...],
    expert_names: tuple[str, ...],
    architecture: Any,
    config: RouterTrainingConfig,
    reference_metrics: dict[str, dict[str, Any]],
    initial_weights: np.ndarray | None = None,
    log: Any = None,
) -> list[AblationOutcome]:
    """Run every ablation on identical rows.

    Each ablation selects its columns from the full panel-wide feature matrix, fills the
    resulting gaps from the router's own training rows, fits a router on the router's
    training rows and scores the sealed test rows. Nothing else varies.

    Args:
        features: Panel-wide :class:`~router.features.RouterFeatures`.
        train_rows: Panel positions the router is allowed to learn from.
        test_rows: Panel positions it is scored on.
        forecasts: ``[n_experts, N, horizons]`` in kW, panel-wide.
        actual_kw: ``[N, horizons]`` in kW, panel-wide.
        horizons: Horizon steps.
        expert_names: Expert names, in order.
        architecture: The main router's :class:`~router.model.RouterArchitecture`.
        config: Training protocol, identical to the main router's.
        reference_metrics: The main router's test metrics.
        initial_weights: The main router's warm start, so only ``cold_start`` differs.
        log: Optional progress callable.

    Returns:
        One outcome per ablation, in :data:`ABLATIONS` order.
    """
    from ..model import RouterArchitecture
    from ..routing import mix
    from .evaluation import method_metrics

    train_rows = np.asarray(train_rows, dtype=np.int64)
    test_rows = np.asarray(test_rows, dtype=np.int64)
    outcomes: list[AblationOutcome] = []

    for ablation in ABLATIONS:
        selected = features.select(ablation.to_feature_spec())
        filled = selected.fill_missing(train_rows)
        shape = RouterArchitecture(
            name=architecture.name,
            hidden_sizes=architecture.hidden_sizes,
            dropout=architecture.dropout,
            temperature=architecture.temperature,
            shared_head=ablation.shared_head,
        )
        if log is not None:
            log(f"  ablation {ablation.name}: {selected.n_features} features, {shape.to_dict()}")
        trained = fit_router(
            features=filled.matrix[train_rows],
            expert_forecast_kw=forecasts[:, train_rows, :],
            actual_kw=actual_kw[train_rows],
            horizons=horizons,
            expert_names=expert_names,
            feature_names=selected.names,
            config=config,
            architecture=shape,
            initial_weights=initial_weights if ablation.warm_start else None,
            log=log,
        )
        weights = router_weights_for(trained, filled.matrix[test_rows])
        forecast = mix(weights, forecasts[:, test_rows, :])
        metrics = method_metrics(
            actual_kw=actual_kw[test_rows], forecast_kw=forecast, horizons=horizons
        )
        deltas = {
            key: float(value["mae"] - reference_metrics[key]["mae"])
            for key, value in metrics.items()
        }
        outcomes.append(
            AblationOutcome(
                name=ablation.name,
                question=ablation.question,
                metrics_by_horizon=metrics,
                reference_metrics_by_horizon=reference_metrics,
                delta_vs_reference=deltas,
                best_epoch=trained.best_epoch,
                best_validation_mae_kw=trained.best_validation_mae_kw,
                train_seconds=trained.train_seconds,
                parameters=trained.parameter_count,
                epochs_run=len(trained.history),
                n_features=selected.n_features,
                mean_weights={
                    str(horizon): {
                        name: float(weights[:, column, index].mean())
                        for index, name in enumerate(expert_names)
                    }
                    for column, horizon in enumerate(horizons)
                },
                notes=(
                    "soft routing, identical rows and protocol to the main router; the "
                    "only difference is the feature contract and, for "
                    "shared_horizon_head, the output head",
                ),
            )
        )
        if log is not None:
            log(
                f"    {ablation.name}: "
                + " ".join(f"h{k}={v['mae']:.4f}" for k, v in metrics.items())
            )
    return outcomes


def render_ablation_table(outcomes: list[AblationOutcome]) -> str:
    """A markdown table of each ablation against the main router.

    The verdict column reads the sign of the delta, where a positive delta means removing
    the feature group made the forecast *worse* - i.e. the group was doing work. Reading
    it the other way round would have this table praise the ablations that hurt.
    """
    lines = [
        "| ablation | question | h=1 MAE (delta) | h=4 MAE (delta) | h=96 MAE (delta) | verdict |",
        "|---|---|---|---|---|---|",
    ]
    for outcome in outcomes:
        keys = sorted(outcome.metrics_by_horizon, key=int)
        cells = " | ".join(
            f"{outcome.metrics_by_horizon[k]['mae']:.4f} "
            f"({outcome.delta_vs_reference[k]:+.4f})"
            for k in keys
        )
        deltas = [outcome.delta_vs_reference[k] for k in keys]
        if all(d > 0 for d in deltas):
            verdict = "group matters at every horizon"
        elif all(d < 0 for d in deltas):
            verdict = "group costs accuracy; removing it helps"
        elif any(d > 0 for d in deltas):
            verdict = "group matters at some horizons only"
        else:
            verdict = "no measurable effect"
        lines.append(f"| {outcome.name} | {outcome.question} | {cells} | {verdict} |")
    return "\n".join(lines)