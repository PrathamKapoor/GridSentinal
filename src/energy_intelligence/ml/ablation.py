"""Run the Phase 5 ablations and write the comparison table.

Each ablation answers one stated question and runs at the **same** reduced budget, so
comparisons are valid within the set. The delta-versus-level question is also
repeated at the main budget by the caller, because it is the one ablation where a
reduced budget could mislead.

What is deliberately absent: a hyperparameter grid. Phase 5 is testing whether
temporal inductive bias helps, and a sweep over capacity, learning rate and context
would confound that with a search over compute.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..paths import ProjectPaths
from .phase5 import load_series_for_experiment, run_phase5_experiment
from .targets import TargetSeries

__all__ = ["AblationOutcome", "run_ablations", "render_ablation_table"]


@dataclass(frozen=True, slots=True)
class AblationOutcome:
    """One ablation's measured outcome."""

    name: str
    question: str
    label: str
    metrics: dict[int, dict[str, Any]]
    best_epoch: int
    best_validation_mae: float
    train_seconds: float
    parameters: int
    epochs_run: int
    reference_name: str
    reference_metrics: dict[int, dict[str, Any]]
    reference_label: str
    delta_vs_reference: dict[int, float | None]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "question": self.question,
            "label": self.label,
            "metrics": {str(h): v for h, v in self.metrics.items()},
            "best_epoch": self.best_epoch,
            "best_validation_mae_per_unit": self.best_validation_mae,
            "train_seconds": round(self.train_seconds, 2),
            "parameters": self.parameters,
            "epochs_run": self.epochs_run,
            "reference": self.reference_label,
            "reference_metrics": {str(h): v for h, v in self.reference_metrics.items()},
            "delta_vs_reference_pct": {str(h): v for h, v in self.delta_vs_reference.items()},
        }


def _mae_at(result, horizons: tuple[int, ...]) -> dict[int, dict[str, Any]]:
    return {h: result.metrics_by_horizon[h] for h in horizons if h in result.metrics_by_horizon}


def run_ablations(
    series: TargetSeries,
    *,
    config,
    paths: ProjectPaths,
    phase4_reference: dict[str, dict[int, dict[str, Any]]],
    baseline_result=None,
) -> list[AblationOutcome]:
    """Run every configured ablation at the reduced budget.

    Args:
        series: The demand series, shared by every ablation so the sample and the
            test population do not move.
        config: The temporal experiment configuration.
        paths: Project layout.
        phase4_reference: Phase 4 metrics.
        baseline_result: The main experiment, used as the within-budget reference when
            it was itself run at the reduced budget.

    Returns:
        One outcome per ablation.
    """
    outcomes: list[AblationOutcome] = []
    budget_epochs = config.ablation_budget_epochs
    budget_stride = config.ablation_budget_stride

    # The within-budget reference: the main configuration, but at the ablation budget,
    # so every ablation has a like-for-like control.
    reference_label = "ablation_reference"
    reference = run_phase5_experiment(
        series,
        sequence_config=config.sequence_config(train_origin_stride=budget_stride),
        training_config=config.training_config(epochs=budget_epochs),
        paths=paths,
        architecture=config.architecture,
        experiment_id="phase5-ablation-reference",
        phase4_reference=phase4_reference,
        label=reference_label,
        validation_stride=config.ablation_validation_stride,
        model_params=dict(config.model_params),
    )

    for ablation in config.ablations:
        overrides = dict(ablation["overrides"])  # type: ignore[arg-type]
        architecture = str(overrides.pop("architecture", config.architecture))
        label = f"ablation-{ablation['name']}"
        print(f"[ablation] {ablation['name']}: {ablation['question']}", flush=True)

        result = run_phase5_experiment(
            series,
            sequence_config=config.sequence_config(
                train_origin_stride=budget_stride, **overrides
            ),
            training_config=config.training_config(epochs=budget_epochs),
            paths=paths,
            architecture=architecture,
            experiment_id=f"phase5-ablation-{ablation['name']}",
            phase4_reference=phase4_reference,
            label=label,
            validation_stride=config.ablation_validation_stride,
            model_params=dict(
                overrides.get("model_params", config.model_params)
                if isinstance(overrides.get("model_params"), dict)
                else config.model_params
            ),
        )
        horizons = tuple(result.metrics_by_horizon)
        deltas: dict[int, float | None] = {}
        for horizon in horizons:
            mine = result.metrics_by_horizon[horizon]["mae"]
            reference_mae = (
                reference.metrics_by_horizon[horizon]["mae"]
                if horizon in reference.metrics_by_horizon
                else None
            )
            deltas[horizon] = (
                100.0 * (mine - reference_mae) / reference_mae if reference_mae else None
            )
        outcomes.append(
            AblationOutcome(
                name=str(ablation["name"]),
                question=str(ablation["question"]),
                label=label,
                metrics=result.metrics_by_horizon,
                best_epoch=result.trained.best_epoch,
                best_validation_mae=result.trained.best_validation_mae,
                train_seconds=result.trained.train_seconds,
                parameters=result.trained.model.parameter_count(),
                epochs_run=len(result.trained.history),
                reference_name="ablation_reference",
                reference_metrics=reference.metrics_by_horizon,
                reference_label=reference_label,
                delta_vs_reference=deltas,
            )
        )

    table = render_ablation_table(outcomes)
    report_root = paths.artifacts / "phase5"
    (report_root / "ablations.md").write_text(table, encoding="utf-8")
    (report_root / "ablations.json").write_text(
        json.dumps(
            {
                "budget_epochs": budget_epochs,
                "budget_train_stride": budget_stride,
                "note": (
                    "Every row was trained at the same reduced budget, so the "
                    "comparison is valid within this table and not against the "
                    "main run. The delta-vs-level question is repeated at the main "
                    "budget separately."
                ),
                "reference": {
                    "label": reference_label,
                    "metrics": {
                        str(h): v for h, v in reference.metrics_by_horizon.items()
                    },
                    "best_validation_mae_per_unit": reference.trained.best_validation_mae,
                },
                "ablations": [outcome.to_dict() for outcome in outcomes],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return outcomes


def render_ablation_table(outcomes: list[AblationOutcome]) -> str:
    """Markdown table of ablations, with the question each one asked."""
    lines = [
        "# Phase 5 ablations",
        "",
        "Every row was trained at the same reduced budget "
        "(see `ablations.json`), so comparisons are valid **within** this table.",
        "",
        "| Ablation | Question | h | MAE (kW) | vs reference | params | train s |",
        "|---|---|---|---|---|---|---|",
    ]
    for outcome in outcomes:
        first = True
        for horizon in sorted(outcome.metrics):
            metrics = outcome.metrics[horizon]
            delta = outcome.delta_vs_reference.get(horizon)
            delta_text = "n/a" if delta is None else f"{delta:+.1f}%"
            lines.append(
                f"| {outcome.name if first else ''} "
                f"| {outcome.question if first else ''} "
                f"| {horizon} "
                f"| {metrics['mae']:.4f} "
                f"| {delta_text} "
                f"| {outcome.parameters if first else ''} "
                f"| {outcome.train_seconds:.0f}{'' if first else ''} |"
            )
            first = False
    lines.append("")
    lines.append(
        "`vs reference` is the percentage change in MAE against the same "
        "configuration at the same budget. Negative means the ablation is better "
        "than the reference, which for a *removal* ablation means the removed "
        "component was not earning its place."
    )
    return "\n".join(lines)
