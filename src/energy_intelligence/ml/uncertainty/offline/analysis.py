"""Per-method evaluation, assembled into the tables the phase's questions ask for.

Kept separate from the runner so the tables can be built and asserted on without running
the experiment. Every function here takes realised targets and returns only measurements;
none of them fits anything, and none of them is reachable from a forecasting path.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..evaluation import (
    imminent_ramp_detection,
    interval_audit,
    overconfidence_analysis,
    reliability_table,
    uncertainty_ranking,
)
from ..intervals import PredictionInterval
from ..measures import DisagreementMeasures
from ..metrics import weighted_interval_score

__all__ = [
    "coverage_tolerance",
    "evaluate_method",
    "method_table",
    "regime_breakdown",
    "expert_point_forecast_table",
    "final_comparison_table",
]


def coverage_tolerance(n_rows: int, nominal_level: float, *, floor: float = 0.01) -> dict[str, Any]:
    """The bar a method must clear to be called calibrated, and why it is that bar.

    The criterion is stated before the numbers are seen and is derived rather than chosen.
    For an empirical coverage measured on ``n`` rows, the binomial standard error is
    ``sqrt(p(1-p)/n)``. Three standard errors is a conventional bar for "this is not the
    nominal level, this is noise".

    On 179,520 test rows that works out at 0.0021 at 90%, so the binding number is the
    **1 percentage-point floor**, not three sigma. The floor exists because at this sample
    size a difference below it is not operationally meaningful for an energy dispatch
    decision, and because a pure significance test would declare a 0.3pp miss at n=179,520
    to be a failure and imply that something should be done about it.

    Both the tolerance and the standard error are returned so a reader can apply a
    different criterion without re-running anything.

    Args:
        n_rows: Rows the coverage will be measured on.
        nominal_level: Stated coverage.
        floor: Minimum tolerance in absolute coverage.

    Returns:
        ``{"tolerance", "nominal_level", "standard_error", "basis", "three_sigma"}``.
    """
    if int(n_rows) <= 0:
        raise ValueError(f"n_rows must be positive, got {n_rows}")
    p = float(nominal_level)
    standard_error = float(np.sqrt(p * (1.0 - p) / int(n_rows)))
    tolerance = max(float(floor), 3.0 * standard_error)
    return {
        "nominal_level": p,
        "n_rows": int(n_rows),
        "standard_error": standard_error,
        "three_sigma": 3.0 * standard_error,
        "tolerance": tolerance,
        "basis": (
            f"max({floor} absolute coverage, 3 binomial standard errors = "
            f"{3.0 * standard_error:.6f}). The floor binds at this sample size."
            if 3.0 * standard_error < float(floor)
            else (
                f"3 binomial standard errors = {3.0 * standard_error:.6f}, which binds "
                f"over the {floor} floor at this sample size."
            )
        ),
    }


def evaluate_method(
    *,
    method: str,
    actual_kw: np.ndarray,
    point_kw: np.ndarray,
    horizons: tuple[int, ...],
    intervals: dict[float, PredictionInterval],
    nominal_levels: tuple[float, ...],
    disagreement: DisagreementMeasures,
    tolerance: float,
    bins: int = 10,
) -> dict[str, Any]:
    """Every measurement for one interval method, on one split.

    Args:
        method: Procedure name.
        actual_kw: ``[N, horizons]`` realised values.
        point_kw: ``[N, horizons]`` point forecast.
        horizons: Horizon steps.
        intervals: ``{nominal_level: PredictionInterval}`` with ``[N, horizons]`` bounds.
        nominal_levels: Levels to evaluate.
        disagreement: The expert spread statistics, for the ranking table.
        tolerance: Coverage tolerance.
        bins: Reliability bins.

    Returns:
        ``{horizon: {...}}`` with per-level metrics, ranking, reliability and
        overconfidence, plus a headline row for the phase's comparison table.
    """
    actual = np.asarray(actual_kw, dtype=np.float64)
    point = np.asarray(point_kw, dtype=np.float64)
    horizons = tuple(int(h) for h in horizons)
    error = np.abs(actual - point)
    audit = interval_audit(
        method=method,
        actual_kw=actual,
        point_kw=point,
        bounds=intervals,
        nominal_levels=nominal_levels,
        horizons=horizons,
        disagreement={name: values for name, values in disagreement.measures.items()},
        tolerance=tolerance,
    )
    headline_level = max(nominal_levels)

    by_horizon: dict[str, Any] = {}
    for column, horizon in enumerate(horizons):
        interval = intervals[float(headline_level)]
        lower = np.asarray(interval.lower_kw, dtype=np.float64)[:, column]
        upper = np.asarray(interval.upper_kw, dtype=np.float64)[:, column]
        width = upper - lower
        column_error = error[:, column]
        by_horizon[str(horizon)] = {
            "horizon": horizon,
            "metrics": audit[str(horizon)],
            "ranking_by_width": uncertainty_ranking(
                uncertainty=width,
                absolute_error_kw=column_error,
                label=f"{method}:interval_width",
                deciles=bins,
            ),
            "reliability_at_headline": reliability_table(
                actual_kw=actual[:, column],
                lower_kw=lower,
                upper_kw=upper,
                nominal_level=float(headline_level),
                uncertainty=width,
                bins=bins,
            ),
            "overconfidence": overconfidence_analysis(
                actual_kw=actual[:, column],
                point_kw=point[:, column],
                interval=PredictionInterval(
                    lower_kw=lower[:, None],
                    upper_kw=upper[:, None],
                    nominal_level=float(headline_level),
                    method=method,
                ),
                band=_band_labels(width),
                nominal_level=float(headline_level),
            ),
            "weighted_interval_score_kw": weighted_interval_score(
                actual[:, column],
                {
                    float(level): (
                        np.asarray(intervals[float(level)].lower_kw, dtype=np.float64)[
                            :, column
                        ],
                        np.asarray(intervals[float(level)].upper_kw, dtype=np.float64)[
                            :, column
                        ],
                    )
                    for level in nominal_levels
                },
            ),
        }
    return {
        "method": method,
        "horizons": list(horizons),
        "headline_nominal_level": float(headline_level),
        "by_horizon": by_horizon,
        "mean_point_mae_kw": float(error.mean()),
        "mean_point_rmse_kw": float(np.sqrt(np.square(actual - point).mean())),
        "coverage_tolerance": tolerance,
    }


def _band_labels(width: np.ndarray) -> np.ndarray:
    """Terciles of the width distribution *of this evaluation's rows*.

    Used only for the overconfidence split, which compares confident against
    unconfident rows **within one method and one split**. The operational
    LOW/MEDIUM/HIGH bands on the artifact come from the calibration split instead, so a
    published band never depends on the rows being scored.
    """
    values = np.asarray(width, dtype=np.float64)
    low, high = (float(v) for v in np.quantile(values, [1.0 / 3.0, 2.0 / 3.0]))
    labels = np.full(values.shape, "medium", dtype=object)
    labels[values <= low] = "low"
    labels[values > high] = "high"
    return labels


def method_table(
    results: dict[str, dict[str, Any]],
    horizons: tuple[int, ...],
    nominal_level: float = 0.90,
) -> list[dict[str, Any]]:
    """The phase's headline comparison: one row per method and horizon.

    Args:
        results: ``{method: evaluation}`` from :func:`evaluate_method`.
        horizons: Horizon steps.
        nominal_level: Which level to report.

    Returns:
        Sorted rows, best (narrowest interval score at equal coverage) first.
    """
    rows: list[dict[str, Any]] = []
    for method, evaluation in results.items():
        for horizon in horizons:
            entry = evaluation["by_horizon"][str(horizon)]
            levels = entry["metrics"]["levels"]
            chosen = levels.get(str(nominal_level))
            if chosen is None:
                continue
            rows.append(
                {
                    "method": method,
                    "horizon": int(horizon),
                    "nominal_level": float(nominal_level),
                    "coverage": chosen["coverage"],
                    "coverage_error": chosen["coverage_error"],
                    "passes": chosen["passes"],
                    "mean_width_kw": chosen["mean_width_kw"],
                    "sharpness_ratio": chosen["sharpness_ratio"],
                    "interval_score_kw": entry["weighted_interval_score_kw"],
                    "point_mae_kw": chosen["mae_kw"],
                    "spearman_width_vs_error": entry["ranking_by_width"]["spearman"],
                }
            )
    rows.sort(key=lambda row: (row["horizon"], row["interval_score_kw"]))
    return rows


def final_comparison_table(
    rows: list[dict[str, Any]],
    horizons: list[int] | tuple[int, ...],
    nominal_level: float = 0.90,
) -> str:
    """Render :func:`method_table` as markdown, with coverage and width beside each MAE.

    Args:
        rows: The comparison rows.
        horizons: Horizon steps, in report order.
        nominal_level: The level the coverage column is measured at. Passed in rather than
            written into the header as a literal, because a hardcoded ``@90%`` above a
            column of 95% coverages is a mislabel that reads as a calibration failure.
    """
    horizons = tuple(int(h) for h in horizons)
    lines = [
        f"| Method | h | coverage @{nominal_level:.0%} | coverage error | mean width kW | "
        "width/MAE | WIS kW | Spearman |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for horizon in horizons:
        for row in [r for r in rows if r["horizon"] == horizon]:
            sharpness = row["sharpness_ratio"]
            lines.append(
                f"| {row['method']} | {row['horizon']} | {row['coverage']:.4f} | "
                f"{row['coverage_error']:+.4f} | {row['mean_width_kw']:.4f} | "
                f"{'n/a' if sharpness is None else f'{sharpness:.3f}'} | "
                f"{row['interval_score_kw']:.4f} | "
                f"{row['spearman_width_vs_error']:+.4f} |"
            )
    return "\n".join(lines)


def regime_breakdown(
    *,
    actual_kw: np.ndarray,
    point_kw: np.ndarray,
    interval: PredictionInterval,
    axes: tuple[Any, ...],
    horizon_index: int,
    nominal_level: float,
) -> dict[str, Any]:
    """Error, width and coverage per regime.

    Phase 7 asked whether a router shifts toward the expert that handles the current
    demand behaviour. This asks the uncertainty version of the same question: do the
    difficult regimes naturally produce wider intervals and larger errors, which is what
    would make an uncertainty-aware system useful downstream rather than decorative.

    Args:
        actual_kw: ``[N, horizons]`` realised values.
        point_kw: ``[N, horizons]`` point forecast.
        interval: The interval whose widths are summarised, ``[N, horizons]``.
        axes: Regime definitions over the same rows.
        horizon_index: Which horizon column to break down.
        nominal_level: Stated coverage of ``interval``.

    Returns:
        ``{axis: {description, horizon, regimes: {label: {...}}}}``.
    """
    actual = np.asarray(actual_kw, dtype=np.float64)[:, horizon_index]
    point = np.asarray(point_kw, dtype=np.float64)[:, horizon_index]
    lower = np.asarray(interval.lower_kw, dtype=np.float64)[:, horizon_index]
    upper = np.asarray(interval.upper_kw, dtype=np.float64)[:, horizon_index]
    width = upper - lower
    error = np.abs(actual - point)
    inside = (actual >= lower) & (actual <= upper)
    out: dict[str, Any] = {}
    for axis in axes:
        labels = np.asarray(axis.labels)
        if labels.shape[0] != actual.shape[0]:
            raise ValueError(
                f"regime axis {axis.name!r} has {labels.shape[0]} labels for "
                f"{actual.shape[0]} rows"
            )
        per_label: dict[str, Any] = {}
        for label in axis.labels_present:
            mask = labels == label
            if not np.any(mask):
                continue
            per_label[str(label)] = {
                "n": int(np.count_nonzero(mask)),
                "share_of_rows": float(np.mean(mask)),
                "mae_kw": float(error[mask].mean()),
                "mean_width_kw": float(width[mask].mean()),
                "median_width_kw": float(np.median(width[mask])),
                "coverage": float(np.mean(inside[mask])),
                "coverage_error": float(np.mean(inside[mask]) - float(nominal_level)),
                "p99_error_kw": float(np.quantile(error[mask], 0.99)),
            }
        widths = [entry["mean_width_kw"] for entry in per_label.values()]
        errors = [entry["mae_kw"] for entry in per_label.values()]
        out[axis.name] = {
            "description": axis.description,
            "regimes": per_label,
            "width_ratio_worst_to_easiest": (
                float(max(widths) / min(widths)) if widths and min(widths) > 0 else None
            ),
            "error_ratio_worst_to_easiest": (
                float(max(errors) / min(errors)) if errors and min(errors) > 0 else None
            ),
            "width_tracks_error": bool(
                widths and errors and _spearman(widths, errors) is not None
                and _spearman(widths, errors) > 0
            ),
        }
    return out


def _spearman(a: list[float], b: list[float]) -> float | None:
    from ..measures import spearman

    if len(a) < 2:
        return None
    value = spearman(np.asarray(a), np.asarray(b))
    return None if value != value else float(value)


def expert_point_forecast_table(
    *,
    forecasts: dict[str, np.ndarray],
    actual_kw: np.ndarray,
    horizons: tuple[int, ...],
) -> dict[str, Any]:
    """Point-forecast MAE/RMSE for each expert and for the fixed ensemble.

    Recorded so the phase's uncertainty results can be read against the point-forecast
    table they modify, and so the fixed ensemble's numbers are checked against Phase 7's
    published ones rather than assumed.
    """
    actual = np.asarray(actual_kw, dtype=np.float64)
    horizons = tuple(int(h) for h in horizons)
    out: dict[str, Any] = {}
    for name, forecast in forecasts.items():
        values = np.asarray(forecast, dtype=np.float64)
        if values.shape != actual.shape:
            raise ValueError(
                f"{name} forecast {values.shape} does not match actual {actual.shape}"
            )
        out[name] = {
            "mae_kw": {
                str(h): float(np.abs(values[:, c] - actual[:, c]).mean())
                for c, h in enumerate(horizons)
            },
            "rmse_kw": {
                str(h): float(np.sqrt(np.square(values[:, c] - actual[:, c]).mean()))
                for c, h in enumerate(horizons)
            },
            "peak_error_mae_kw": {
                str(h): float(
                    np.abs(values[:, c] - actual[:, c])[
                        actual[:, c] >= np.quantile(actual[:, c], 0.90)
                    ].mean()
                )
                for c, h in enumerate(horizons)
            },
        }
    return out