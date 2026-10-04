"""Does the uncertainty say anything true?

Coverage and width answer "is the interval honest and how big is it". They do **not**
answer the operational question: *does the uncertainty tell us which forecasts to
distrust?* This module answers that, and it is the part that decides whether the phase
produced something usable or a well-calibrated decoration.

Three diagnostics, deliberately kept apart because they fail independently.

**Ranking.** Spearman correlation between predicted uncertainty and realised absolute
error. Positive means larger predicted uncertainty accompanies larger error, i.e. the
uncertainty orders forecasts by difficulty. Reported with the decile table, because a
correlation can be carried by one tail of the distribution while the bulk of the
uncertainty is uninformative - and the decile table shows whether it is.

**Reliability.** Coverage per predicted-uncertainty bin. A method can be calibrated on
average and badly miscalibrated on the rows that matter, so this is reported per bin and
never as one number.

**Overconfidence.** The dangerous class is a *confident* forecast that is badly wrong.
Counted and inspected, because "calibrated at 90% overall" and "predicts a 12 kW error
with a 2 kW interval on a row it was sure about" are compatible, and only the second one
kills a trust-based decision system.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .intervals import PredictionInterval
from .measures import spearman

__all__ = [
    "uncertainty_ranking",
    "reliability_table",
    "overconfidence_analysis",
    "interval_audit",
    "imminent_ramp_detection",
]


def uncertainty_ranking(
    *,
    uncertainty: np.ndarray,
    absolute_error_kw: np.ndarray,
    label: str,
    deciles: int = 10,
) -> dict[str, Any]:
    """How well does the uncertainty order forecasts by difficulty?

    Args:
        uncertainty: ``[N]`` predicted uncertainty scalar. Interval width is the usual
            choice; a scale function's own output is also valid and is passed where the
            two differ.
        absolute_error_kw: ``[N]`` realised absolute error.
        label: What the uncertainty column is, for the record.
        deciles: Number of equal-count groups.

    Returns:
        Spearman correlation, the decile table, and two ratios that are easier to act on
        than a correlation: the top-decile error multiple, and the bottom-decile one.
    """
    u = np.asarray(uncertainty, dtype=np.float64).ravel()
    e = np.asarray(absolute_error_kw, dtype=np.float64).ravel()
    if u.shape != e.shape:
        raise ValueError(
            f"uncertainty and error must be aligned, got {u.shape} and {e.shape}"
        )
    if u.size == 0:
        return {
            "label": label,
            "n": 0,
            "spearman": float("nan"),
            "deciles": [],
            "top_decile_error_multiple": None,
            "bottom_decile_error_multiple": None,
        }
    rho = spearman(u, e)
    edges = np.quantile(u, np.linspace(0.0, 1.0, int(deciles) + 1))
    edges[-1] = min(float(edges[-1]), float(u.max()))
    rows: list[dict[str, Any]] = []
    overall = float(e.mean())
    for index in range(int(deciles)):
        low = float(edges[index])
        high = float(edges[index + 1])
        if index == int(deciles) - 1:
            mask = (u >= low) & (u <= high)
        else:
            mask = (u >= low) & (u < high)
        if not np.any(mask):
            continue
        rows.append(
            {
                "decile": index + 1,
                "uncertainty_low": round(low, 6),
                "uncertainty_high": round(high, 6),
                "rows": int(np.count_nonzero(mask)),
                "mean_absolute_error_kw": round(float(e[mask].mean()), 6),
                "error_multiple": round(float(e[mask].mean() / overall), 4)
                if overall > 0
                else None,
                "mean_uncertainty": round(float(u[mask].mean()), 6),
            }
        )
    top = rows[-1]["error_multiple"] if rows else None
    bottom = rows[0]["error_multiple"] if rows else None
    return {
        "label": label,
        "n": int(u.size),
        "spearman": float(rho),
        "mean_absolute_error_kw": overall,
        "deciles": rows,
        "top_decile_error_multiple": top,
        "bottom_decile_error_multiple": bottom,
        "monotone_deciles": bool(
            all(
                rows[i]["mean_absolute_error_kw"] <= rows[i + 1]["mean_absolute_error_kw"]
                for i in range(len(rows) - 1)
            )
        ),
    }


def reliability_table(
    *,
    actual_kw: np.ndarray,
    lower_kw: np.ndarray,
    upper_kw: np.ndarray,
    nominal_level: float,
    uncertainty: np.ndarray,
    bins: int = 10,
) -> dict[str, Any]:
    """Coverage and error within each predicted-uncertainty bin.

    Where :func:`uncertainty.metrics.expected_calibration_error` answers "is the interval
    calibrated", this answers "is it calibrated *where the uncertainty says it matters*".
    A method can pass the first and fail this one, and the second is what a downstream
    consumer will actually experience.
    """
    from .metrics import expected_calibration_error

    return expected_calibration_error(
        actual_kw,
        lower_kw,
        upper_kw,
        nominal_level,
        scale=uncertainty,
        bins=bins,
    )


def overconfidence_analysis(
    *,
    actual_kw: np.ndarray,
    point_kw: np.ndarray,
    interval: PredictionInterval,
    band: np.ndarray,
    nominal_level: float,
    severe_error_quantile: float = 0.90,
    top_rows: int = 20,
) -> dict[str, Any]:
    """Count and inspect high-confidence, large-error forecasts.

    **The dangerous class.** A forecast the system was confident about, which was badly
    wrong. Its frequency is what a future decision-assurance layer needs, and its
    magnitude is what would make the failure unrecoverable.

    Definitions, fixed so the count cannot be tuned:

    * a row is **CONFIDENT** when its interval width is in the lowest third of the
      distribution passed in as ``band`` being ``"low"``;
    * a row is **SEVERE** when its absolute error exceeds the ``severe_error_quantile``
      quantile of the absolute error over the evaluated rows - by default the worst 10%.

    A confident-and-severe row is the count this reports.

    Args:
        actual_kw: Realised values.
        point_kw: The point forecast.
        interval: The interval the system issued.
        band: ``[N]`` band labels, e.g. from
            :func:`uncertainty.artifact.ProbabilisticForecastBatch.bands`.
        nominal_level: Stated coverage.
        severe_error_quantile: Quantile defining a severe error.
        top_rows: How many offenders to list.

    Returns:
        Counts per band, the confident-and-severe count, and the worst offenders with
        their timestamps' error, width and actual value.
    """
    actual = np.asarray(actual_kw, dtype=np.float64).ravel()
    point = np.asarray(point_kw, dtype=np.float64).ravel()
    labels = np.asarray(band, dtype=object)
    error = np.abs(actual - point)
    width = np.asarray(interval.width_kw, dtype=np.float64).ravel()
    severe_threshold = float(np.quantile(error, float(severe_error_quantile)))
    severe = error >= severe_threshold

    counts: dict[str, Any] = {}
    for name in sorted({str(value) for value in labels}):
        mask = labels == name
        counts[name] = {
            "rows": int(np.count_nonzero(mask)),
            "share_of_rows": float(np.mean(mask)),
            "mean_width_kw": float(width[mask].mean()) if np.any(mask) else None,
            "mae_kw": float(error[mask].mean()) if np.any(mask) else None,
            "coverage": float(
                np.mean(
                    (
                        (actual[mask] >= np.asarray(interval.lower_kw).ravel()[mask])
                        & (actual[mask] <= np.asarray(interval.upper_kw).ravel()[mask])
                    )
                )
            )
            if np.any(mask)
            else None,
            "severe_rows": int(np.count_nonzero(mask & severe)),
            "severe_rate": float(np.mean(severe[mask])) if np.any(mask) else None,
        }
    confident = labels == "low"
    offenders = np.flatnonzero(confident & severe)
    order = offenders[np.argsort(-error[offenders])][: int(top_rows)] if offenders.size else offenders
    return {
        "nominal_level": float(nominal_level),
        "severe_error_threshold_kw": severe_threshold,
        "severe_error_quantile": float(severe_error_quantile),
        "by_band": counts,
        "confident_and_severe_rows": int(offenders.size),
        "confident_and_severe_share": float(np.mean(confident & severe)),
        "worst_confident_errors": [
            {
                "row": int(index),
                "actual_kw": round(float(actual[index]), 4),
                "point_kw": round(float(point[index]), 4),
                "absolute_error_kw": round(float(error[index]), 4),
                "interval_width_kw": round(float(width[index]), 4),
                "band": str(labels[index]),
            }
            for index in order
        ],
        "definition": (
            "CONFIDENT means band == 'low' (narrowest third of the calibrated width "
            f"distribution); SEVERE means absolute error at or above the "
            f"{severe_error_quantile:.0%} error quantile. The product is the count."
        ),
    }


def interval_audit(
    *,
    method: str,
    actual_kw: np.ndarray,
    point_kw: np.ndarray,
    bounds: dict[float, PredictionInterval],
    nominal_levels: tuple[float, ...],
    horizons: tuple[int, ...],
    disagreement: dict[str, np.ndarray],
    tolerance: float,
) -> dict[str, Any]:
    """One method's complete record, per horizon and level, in a single comparable shape.

    Args:
        method: Procedure name.
        actual_kw: ``[N, horizons]`` realised values.
        point_kw: ``[N, horizons]`` point forecast.
        bounds: ``{nominal_level: PredictionInterval}`` with ``[N, horizons]`` bounds.
        nominal_levels: Levels to report.
        horizons: Horizon steps.
        disagreement: ``{name: [N, horizons]}`` spread statistics, reported alongside
            so the ranking of every measure is visible in one place.
        tolerance: Absolute coverage tolerance used for the PASS/FAIL verdict.

    Returns:
        ``{horizon: {level: {...metrics...}, "ranking": {...}}}``.
    """
    actual = np.asarray(actual_kw, dtype=np.float64)
    point = np.asarray(point_kw, dtype=np.float64)
    horizons = tuple(int(h) for h in horizons)
    out: dict[str, Any] = {}
    for column, horizon in enumerate(horizons):
        per_level: dict[str, Any] = {}
        error = np.abs(actual[:, column] - point[:, column])
        for nominal in nominal_levels:
            interval = bounds[float(nominal)]
            lower = np.asarray(interval.lower_kw, dtype=np.float64)[:, column]
            upper = np.asarray(interval.upper_kw, dtype=np.float64)[:, column]
            inside = (actual[:, column] >= lower) & (actual[:, column] <= upper)
            observed = float(np.mean(inside))
            per_level[str(nominal)] = {
                "nominal_level": float(nominal),
                "n": int(actual.shape[0]),
                "coverage": observed,
                "coverage_error": observed - float(nominal),
                "passes": abs(observed - float(nominal)) <= float(tolerance),
                "mean_width_kw": float(np.mean(upper - lower)),
                "median_width_kw": float(np.median(upper - lower)),
                "sharpness_ratio": float(np.mean(upper - lower) / error.mean())
                if error.mean() > 0
                else None,
                "mae_kw": float(error.mean()),
                "clipped_share": float(np.mean(lower <= 0.0)),
            }
        ranking: dict[str, Any] = {}
        for name, values in disagreement.items():
            column_values = np.asarray(values, dtype=np.float64)[:, column]
            ranking[name] = {
                "spearman_vs_absolute_error": float(spearman(column_values, error)),
                "pearson_vs_absolute_error": float(
                    np.corrcoef(column_values, error)[0, 1]
                )
                if column_values.std() > 0 and error.std() > 0
                else None,
                "mean": float(column_values.mean()),
                "p90": float(np.quantile(column_values, 0.90)),
            }
        out[str(horizon)] = {
            "method": method,
            "levels": per_level,
            "point_mae_kw": float(error.mean()),
            "uncertainty_vs_error": ranking,
            "tolerance": float(tolerance),
        }
    return out


def imminent_ramp_detection(
    *,
    uncertainty: np.ndarray,
    absolute_error_kw: np.ndarray,
    future_ramp_kw: np.ndarray,
    label: str,
) -> dict[str, Any]:
    """Can the uncertainty flag rows that are about to change sharply?

    Phase 7's diagnosed router failure was that its features could not distinguish flat
    demand from demand about to ramp at 15-minute resolution. This asks whether
    *uncertainty* can.

    **The label is a diagnostic, not an input.** ``future_ramp_kw`` is computed from the
    realised target, exactly as Phase 7's regime terciles were, and is used only to group
    rows for evaluation. Any attempt to route on it would be reading the target, which is
    the leak this phase's split design exists to prevent. The asymmetry is deliberate and
    is why the question is answerable at all: the uncertainty may use only present
    information, while the evaluator may use the future to check whether the present
    information was informative.

    Args:
        uncertainty: ``[N]`` predicted uncertainty.
        absolute_error_kw: ``[N]`` realised absolute error.
        future_ramp_kw: ``[N]`` realised change over the coming hour, kW. A diagnostic.
        label: Name of the uncertainty column.

    Returns:
        Whether the top-third-uncertainty rows are enriched in severe future ramps, by how
        much, and what the error looks like in each future-ramp tercile.
    """
    u = np.asarray(uncertainty, dtype=np.float64).ravel()
    e = np.asarray(absolute_error_kw, dtype=np.float64).ravel()
    ramp = np.abs(np.asarray(future_ramp_kw, dtype=np.float64).ravel())
    if not (u.shape == e.shape == ramp.shape):
        raise ValueError(
            f"inputs must be aligned, got {u.shape}, {e.shape}, {ramp.shape}"
        )
    if u.size == 0:
        return {"label": label, "n": 0}
    high = u >= np.quantile(u, 2.0 / 3.0)
    low = u <= np.quantile(u, 1.0 / 3.0)
    severe_ramp = ramp >= np.quantile(ramp, 2.0 / 3.0)
    base = float(np.mean(severe_ramp))
    per_tercile: dict[str, Any] = {}
    for name, edges in (("low", 1.0 / 3.0), ("typical", 2.0 / 3.0)):
        cutoff = np.quantile(ramp, edges)
        mask = ramp < cutoff
        if not np.any(mask):
            continue
        per_tercile[name] = {
            "rows": int(np.count_nonzero(mask)),
            "mae_kw": float(e[mask].mean()),
            "mean_uncertainty": float(u[mask].mean()),
        }
    severe_mask = ramp >= np.quantile(ramp, 2.0 / 3.0)
    per_tercile["high"] = {
        "rows": int(np.count_nonzero(severe_mask)),
        "mae_kw": float(e[severe_mask].mean()) if np.any(severe_mask) else None,
        "mean_uncertainty": float(u[severe_mask].mean()) if np.any(severe_mask) else None,
    }
    return {
        "label": label,
        "n": int(u.size),
        "spearman_uncertainty_vs_future_ramp": float(spearman(u, ramp)),
        "severe_ramp_base_rate": base,
        "severe_ramp_rate_in_high_uncertainty": float(np.mean(severe_ramp[high])),
        "severe_ramp_rate_in_low_uncertainty": float(np.mean(severe_ramp[low])),
        "enrichment_high_over_base": float(np.mean(severe_ramp[high]) / base)
        if base > 0
        else None,
        "enrichment_low_over_base": float(np.mean(severe_ramp[low]) / base)
        if base > 0
        else None,
        "by_future_ramp_tercile": per_tercile,
        "note": (
            "future_ramp_kw is a diagnostic label computed from the realised target; "
            "the uncertainty column contains only present information"
        ),
    }