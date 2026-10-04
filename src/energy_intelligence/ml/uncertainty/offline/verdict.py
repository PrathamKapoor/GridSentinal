"""The phase's answer, derived from the measurements rather than asserted.

The question has three parts and they can come out differently, so they are answered
separately and then combined:

1. **Can uncertainty be estimated reliably?** - measured by whether any method's coverage
   lands inside the pre-stated tolerance at the reported levels.
2. **Does it say anything useful?** - measured by the ranking correlation and, more
   stringently, by whether the top uncertainty decile really does carry more error.
3. **Is it worth the machinery?** - measured by interval score against the simplest
   possible method, a global constant width.

The third is the one that usually decides it. A sophisticated method that ties with a
constant-width residual quantile on interval score has added nothing, and saying so is more
useful than reporting the tie.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["PHASE8_ANSWERS", "phase8_verdict", "best_method_per_horizon"]

#: The three answers, in the vocabulary the phase is allowed to use.
PHASE8_ANSWERS: tuple[str, ...] = ("YES", "PARTIALLY", "NO", "INSUFFICIENT_DATA")


def best_method_per_horizon(
    rows: list[dict[str, Any]],
    horizons: tuple[int, ...],
    *,
    calibrated_only: bool = False,
) -> dict[int, str]:
    """The narrowest-interval-score method per horizon at the headline level.

    Interval score is the selection criterion because it is the only one of the reported
    measures that scores coverage and sharpness together with published weights. Choosing
    on width alone would reward a method that covers nothing, and choosing on coverage
    alone would reward an unbounded interval.

    Args:
        rows: The comparison rows.
        horizons: Horizon steps.
        calibrated_only: Restrict the candidates to methods that met the coverage
            tolerance at this horizon. Interval score is a proper scoring rule, so a
            method that under-covers is already penalised for it - but only linearly in the
            overshoot, and on this data the penalty was not large enough to stop the
            narrowest, under-covering method from winning. Selecting on it alone would
            therefore reward a method that is confidently wrong, which is the one failure
            mode a prediction interval exists to prevent.

    Returns:
        ``{horizon: method}``, falling back to the unconstrained winner if nothing at a
        horizon is calibrated - in which case the verdict says so rather than inventing a
        winner.
    """
    out: dict[int, str] = {}
    for horizon in horizons:
        candidates = [row for row in rows if row["horizon"] == int(horizon)]
        if not candidates:
            continue
        eligible = [row for row in candidates if row["passes"]] if calibrated_only else []
        pool = eligible or candidates
        out[int(horizon)] = min(pool, key=lambda row: row["interval_score_kw"])["method"]
        out[int(horizon)] = str(out[int(horizon)])
    return out


def phase8_verdict(
    *,
    rows: list[dict[str, Any]],
    results: dict[str, dict[str, Any]],
    horizons: tuple[int, ...],
    nominal_level: float,
    baseline_method: str,
    tolerance: float,
    disagreement: dict[str, Any],
) -> dict[str, Any]:
    """Assemble the phase's verdict from its measurements.

    Args:
        rows: The comparison table from ``analysis.method_table``.
        results: ``{method: evaluation}``.
        horizons: Horizon steps.
        nominal_level: The headline level.
        baseline_method: The method every other is measured against.
        tolerance: Coverage tolerance.
        disagreement: ``{measure_name: {horizon: spearman}}`` from the evaluation.

    Returns:
        The verdict record, with the three answers and the evidence for each.
    """
    horizons = tuple(int(h) for h in horizons)
    level_key = str(float(nominal_level))

    calibrated = {
        method: all(
            evaluation["by_horizon"][str(h)]["metrics"]["levels"][level_key]["passes"]
            for h in horizons
        )
        for method, evaluation in results.items()
    }
    passing = sorted(name for name, ok in calibrated.items() if ok)

    informative: dict[str, bool] = {}
    ranking: dict[str, Any] = {}
    for method, evaluation in results.items():
        rhos = [
            evaluation["by_horizon"][str(h)]["ranking_by_width"]["spearman"]
            for h in horizons
        ]
        finite = [r for r in rhos if r == r]
        mean_rho = float(np.mean(finite)) if finite else float("nan")
        multiples = [
            evaluation["by_horizon"][str(h)]["ranking_by_width"][
                "top_decile_error_multiple"
            ]
            for h in horizons
        ]
        useful = [m for m in multiples if m is not None]
        # The bar is deliberately weak on the correlation and strict on the practical
        # one. A positive correlation is easy to obtain from a heavy-tailed error; the
        # top decile carrying at least 20% more error than average is what would actually
        # change a decision.
        ranking[method] = {
            "mean_spearman_width_vs_error": mean_rho,
            "top_decile_error_multiple_by_horizon": {
                str(h): evaluation["by_horizon"][str(h)]["ranking_by_width"][
                    "top_decile_error_multiple"
                ]
                for h in horizons
            },
        }
        informative[method] = bool(
            mean_rho == mean_rho
            and mean_rho > 0
            and useful
            and min(useful) >= 1.2
        )

    per_horizon_best = best_method_per_horizon(rows, horizons, calibrated_only=True)
    unconstrained_best = best_method_per_horizon(rows, horizons)
    beaten: dict[str, Any] = {}
    for horizon in horizons:
        baseline = next(
            (r for r in rows if r["horizon"] == horizon and r["method"] == baseline_method),
            None,
        )
        winner = per_horizon_best.get(horizon)
        if baseline is None or winner is None:
            continue
        winner_row = next(r for r in rows if r["horizon"] == horizon and r["method"] == winner)
        beaten[str(horizon)] = {
            "baseline": baseline_method,
            "baseline_interval_score_kw": baseline["interval_score_kw"],
            "baseline_calibrated": bool(baseline["passes"]),
            "best_calibrated_method": winner,
            "best_interval_score_kw": winner_row["interval_score_kw"],
            "unconstrained_best_method": unconstrained_best.get(horizon),
            "unconstrained_best_is_calibrated": bool(
                unconstrained_best.get(horizon) == winner
            ),
            "improvement_pct": (
                100.0
                * (baseline["interval_score_kw"] - winner_row["interval_score_kw"])
                / baseline["interval_score_kw"]
                if baseline["interval_score_kw"] > 0
                else None
            ),
        }

    calibration_answer = "YES" if passing else "NO"
    usefulness_answer = "YES" if any(informative.values()) else "NO"
    improvements = [
        entry["improvement_pct"]
        for entry in beaten.values()
        if entry["improvement_pct"] is not None
    ]
    meaningful_improvement = [value for value in improvements if value > 1.0]
    machinery_answer = (
        "YES" if len(meaningful_improvement) == len(horizons) and beaten
        else ("PARTIALLY" if meaningful_improvement else "NO")
    )

    if calibration_answer == "NO":
        answer = "NO"
    elif usefulness_answer == "NO":
        answer = "PARTIALLY"
    elif machinery_answer == "NO":
        answer = "PARTIALLY"
    else:
        answer = "YES"

    return {
        "question": (
            "can calibrated, informative uncertainty be produced around the energy "
            "demand forecast, and does it identify when the forecast cannot be trusted?"
        ),
        "answer": answer,
        "coverage_tolerance": float(tolerance),
        "headline_nominal_level": float(nominal_level),
        "components": {
            "can_be_estimated": {
                "answer": calibration_answer,
                "calibrated_methods": passing,
                "methods_evaluated": sorted(results),
                "criterion": (
                    f"coverage within {tolerance:.4f} absolute of nominal at "
                    f"{nominal_level:.0%}, at every horizon"
                ),
            },
            "is_informative": {
                "answer": usefulness_answer,
                "criterion": (
                    "mean Spearman correlation between interval width and absolute error "
                    "strictly positive across horizons, AND the top uncertainty decile "
                    "carrying at least 1.20x the mean absolute error at every horizon"
                ),
                "methods": {name: bool(value) for name, value in informative.items()},
                "ranking": ranking,
            },
            "is_worth_the_machinery": {
                "answer": machinery_answer,
                "baseline": baseline_method,
                "criterion": (
                    "the best *calibrated* method's weighted interval score beats the "
                    "global constant-width baseline by more than 1% at every horizon. The "
                    "comparison is restricted to calibrated methods because the narrowest "
                    "method on this data was not the calibrated one, and an interval that "
                    "is confidently too narrow is the failure this phase exists to prevent"
                ),
                "per_horizon": beaten,
            },
        },
        "disagreement_is_predictive": {
            measure: {
                str(horizon): disagreement.get(measure, {}).get(str(horizon))
                for horizon in horizons
            }
            for measure in sorted(disagreement)
        },
        "best_method_by_horizon": {str(k): v for k, v in per_horizon_best.items()},
        "best_method_by_horizon_unconstrained": {
            str(k): v for k, v in unconstrained_best.items()
        },
        "basis": (
            "every component answer is computed from the sealed test split and reported "
            "separately; the phase verdict is the weakest component"
        ),
    }