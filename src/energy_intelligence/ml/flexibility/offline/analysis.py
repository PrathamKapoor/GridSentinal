"""Per-method evaluation tables and the Phase 9 verdict.

What Phase 9 has to answer, and how each part is answered
--------------------------------------------------------

**Can a flexibility envelope be estimated at all from observational data?**
    Fitted on ``CAL_FIT``, applied to rows it has never seen. If it cannot hold its stated
    coverage on unseen data, it is not an estimate of anything.

**Is it sharp, or is it a very wide band that trivially covers?**
    Width is reported absolutely *and* relative to the mean absolute deviation it is
    supposed to bound, because a wide envelope on a volatile load is a different problem
    from a wide envelope on a quiet one.

**Can it tell the two directions apart?**
    Reported as the ratio of mean upward to mean downward magnitude, and as per-direction
    conditional coverage. An envelope that is calibrated overall while missing every
    downward excursion is useless to a system that wants to shed load.

**Is it stable?**
    A claimed envelope that jitters row to row cannot be consumed by a planner even at
    perfect coverage, so the lag-1 autocorrelation of its own width is reported next to how
    well its changes track changes in reality.

**Is uncertainty informative about flexibility here?**
    The reliance coupling is tested, not assumed. If Phase 8's width does not predict
    realised deviation magnitude, the phase says so and refuses to present the discount as
    evidence-based.

**Does aggregating customers create resource?**
    Reported as the aggregate envelope beside the naive sum of individual envelopes, with
    the measured pairwise correlation that explains the gap, and with an explicit statement
    that diversification is historical correlation rather than a control capability.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..evaluation import directional_balance, evaluate_band, stability_report

__all__ = [
    "coverage_tolerance",
    "method_table",
    "regime_breakdown",
    "final_comparison_table",
]


def coverage_tolerance(n_rows: int, nominal_level: float, *, floor: float = 0.01) -> dict[str, Any]:
    """The bar a flexibility envelope must clear to be called calibrated.

    Derived rather than chosen, and identical in form to Phase 8's coverage tolerance: for
    an empirical coverage measured on ``n`` rows the binomial standard error is
    ``sqrt(p(1-p)/n)``, and three of those is the conventional "this is noise" bar.

    The **1 percentage-point floor** is what binds at this sample size, for the same reason
    it binds in Phase 8: below it a difference is statistically detectable and operationally
    irrelevant, and a pure significance test would declare a 0.3pp miss a failure.

    Args:
        n_rows: Rows evaluated.
        nominal_level: The level being checked.
        floor: Minimum tolerance.

    Returns:
        The tolerance, the standard error it was derived from, and the binding term.
    """
    p = float(nominal_level)
    standard_error = float(np.sqrt(p * (1.0 - p) / max(int(n_rows), 1)))
    three_sigma = 3.0 * standard_error
    tolerance = max(float(floor), three_sigma)
    return {
        "tolerance": tolerance,
        "standard_error": standard_error,
        "three_sigma": three_sigma,
        "floor": float(floor),
        "binding_term": "floor" if float(floor) >= three_sigma else "three_sigma",
        "basis": (
            f"max({floor}, 3 * binomial standard error = {three_sigma:.6f}); the "
            f"{'floor' if float(floor) >= three_sigma else 'three-sigma term'} binds at "
            f"n={int(n_rows)}"
        ),
    }


def evaluate_configuration(
    *,
    label: str,
    expected_kw: np.ndarray,
    observed_kw: np.ndarray,
    upward_kw: np.ndarray,
    downward_kw: np.ndarray,
    reliability: np.ndarray | None,
    nominal_level: float,
    tolerance: float,
    series: np.ndarray,
    bins: int = 10,
    spearman=None,
) -> dict[str, Any]:
    """One baseline x granularity x reliance configuration, fully scored.

    Args:
        label: Name for the configuration row.
        expected_kw: ``[rows]`` baseline value, kW.
        observed_kw: ``[rows]`` realised demand, kW.
        upward_kw: ``[rows]`` upward magnitudes before any reliance discount, kW.
        downward_kw: ``[rows]`` downward magnitudes before any reliance discount, kW.
        reliability: ``[rows]`` reliance factors, or ``None`` for the undiscounted envelope.
        nominal_level: The level claimed.
        tolerance: Coverage tolerance for the PASS/FAIL flag.
        series: ``[rows]`` series index, for the within-series stability statistics.
        bins: Bins for the reliability table.
        spearman: Rank-correlation callable.

    Returns:
        The full record, including the raw (undiscounted) scores beside the applied ones so
        the cost of the discount is visible rather than implied.
    """
    up_raw = np.asarray(upward_kw, dtype=np.float64).ravel()
    down_raw = np.asarray(downward_kw, dtype=np.float64).ravel()
    if reliability is None:
        up = up_raw
        down = down_raw
        factors = np.ones(up_raw.size, dtype=np.float64)
    else:
        factors = np.asarray(reliability, dtype=np.float64).ravel()
        up = up_raw * factors
        down = down_raw * factors

    applied = evaluate_band(
        expected_kw=expected_kw,
        observed_kw=observed_kw,
        upward_kw=up,
        downward_kw=down,
        nominal_level=float(nominal_level),
        bins=int(bins),
    )
    raw = evaluate_band(
        expected_kw=expected_kw,
        observed_kw=observed_kw,
        upward_kw=up_raw,
        downward_kw=down_raw,
        nominal_level=float(nominal_level),
        bins=int(bins),
    )
    record: dict[str, Any] = {
        "configuration": label,
        "nominal_level": float(nominal_level),
        "tolerance": float(tolerance),
        "coverage": applied["coverage"],
        "coverage_error": applied["coverage_error"],
        "passes": bool(abs(applied["coverage_error"]) <= float(tolerance)),
        "mean_width_kw": applied["mean_width_kw"],
        "width_over_mean_absolute_deviation": applied["width_over_mean_absolute_deviation"],
        "winkler_interval_score_kw": applied["winkler_interval_score_kw"],
        "weighted_interval_score_kw": applied["weighted_interval_score_kw"],
        "upward_conditional_coverage": applied["upward"]["conditional_coverage"],
        "downward_conditional_coverage": applied["downward"]["conditional_coverage"],
        "mean_absolute_deviation_kw": applied["mean_absolute_deviation_kw"],
        "directional": directional_balance(upward_kw=up, downward_kw=down),
        "reliability_factor": {
            "applied": bool(reliability is not None),
            "mean": float(factors.mean()),
            "share_at_full": float(np.mean(factors >= 1.0 - 1e-12)),
        },
        "undiscounted": {
            "coverage": raw["coverage"],
            "coverage_error": raw["coverage_error"],
            "mean_width_kw": raw["mean_width_kw"],
            "winkler_interval_score_kw": raw["winkler_interval_score_kw"],
        },
        "width_cost_of_discount_pct": (
            100.0 * (raw["mean_width_kw"] - applied["mean_width_kw"]) / raw["mean_width_kw"]
            if raw["mean_width_kw"] > 0 and reliability is not None
            else 0.0
        ),
    }
    if spearman is not None:
        record["stability"] = stability_report(
            width_kw=up + down,
            observed_kw=np.asarray(observed_kw, dtype=np.float64).ravel(),
            series=np.asarray(series, dtype=np.int64).ravel(),
            spearman=spearman,
        )
    return record


def regime_breakdown(
    *,
    expected_kw: np.ndarray,
    observed_kw: np.ndarray,
    upward_kw: np.ndarray,
    downward_kw: np.ndarray,
    axes: tuple[Any, ...],
    nominal_level: float,
    tolerance: float,
) -> dict[str, Any]:
    """Coverage and width per Phase 4/5 regime tercile.

    The regime axes are computed from the target being predicted, so they are an
    **evaluation** grouping and never a feature. That is the same discipline Phase 7 used
    and it is why a regime-conditioned envelope is reported here rather than fitted.

    Args:
        expected_kw: ``[rows]`` baseline value, kW.
        observed_kw: ``[rows]`` realised demand, kW.
        upward_kw: ``[rows]`` upward magnitudes, kW.
        downward_kw: ``[rows]`` downward magnitudes, kW.
        axes: Regime definitions over the same rows.
        nominal_level: The level claimed.
        tolerance: Coverage tolerance.

    Returns:
        ``{axis: {description, regimes: {...}}}``.

    Raises:
        ValueError: If an axis covers a different number of rows.
    """
    expected = np.asarray(expected_kw, dtype=np.float64).ravel()
    observed = np.asarray(observed_kw, dtype=np.float64).ravel()
    up = np.asarray(upward_kw, dtype=np.float64).ravel()
    down = np.asarray(downward_kw, dtype=np.float64).ravel()
    lower = expected - down
    upper = expected + up
    inside = (observed >= lower) & (observed <= upper)
    deviation = np.abs(observed - expected)

    out: dict[str, Any] = {}
    for axis in axes:
        labels = np.asarray(axis.labels)
        if labels.shape[0] != expected.size:
            raise ValueError(
                f"regime axis {axis.name!r} has {labels.shape[0]} labels for "
                f"{expected.size} rows"
            )
        per_label: dict[str, Any] = {}
        for label in axis.labels_present:
            mask = labels == label
            if not np.any(mask):
                continue
            coverage = float(np.mean(inside[mask]))
            per_label[str(label)] = {
                "n": int(np.count_nonzero(mask)),
                "share_of_rows": float(np.mean(mask)),
                "coverage": coverage,
                "coverage_error": coverage - float(nominal_level),
                "passes": bool(abs(coverage - float(nominal_level)) <= float(tolerance)),
                "mae_kw": float(deviation[mask].mean()),
                "mean_width_kw": float((up + down)[mask].mean()),
                "width_over_mae": (
                    float((up + down)[mask].mean() / deviation[mask].mean())
                    if deviation[mask].mean() > 0
                    else None
                ),
            }
        widths = [entry["mean_width_kw"] for entry in per_label.values()]
        out[axis.name] = {
            "description": axis.description,
            "regimes": per_label,
            "width_ratio_worst_to_easiest": (
                float(max(widths) / min(widths)) if widths and min(widths) > 0 else None
            ),
            "all_regimes_calibrated": all(entry["passes"] for entry in per_label.values()),
        }
    return out


def method_table(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Order configurations for selection: calibrated first, then best interval score.

    Selection criterion is the **weighted interval score**, the same reasoning Phase 8 used:
    it is the only reported measure that scores coverage and sharpness together with
    published weights. Choosing on width alone would reward an envelope that covers nothing.

    Args:
        rows: Records from :func:`evaluate_configuration`.

    Returns:
        A new list, sorted by horizon, then calibration, then score.
    """
    ordered = sorted(
        rows,
        key=lambda row: (
            not row["passes"],
            row["weighted_interval_score_kw"],
        ),
    )
    for position, row in enumerate(ordered, start=1):
        row = row
        ordered[position - 1]["rank"] = position
    return ordered


def final_comparison_table(rows: list[dict[str, Any]]) -> str:
    """Render the configuration comparison as markdown.

    Args:
        rows: Records from :func:`evaluate_configuration`.

    Returns:
        Markdown.
    """
    lines = [
        "| configuration | coverage | error | mean width kW | width/MAE | WIS kW | up cov | down cov | rank |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        up_cov = row["upward_conditional_coverage"]
        down_cov = row["downward_conditional_coverage"]
        lines.append(
            f"| {row['configuration']} | {row['coverage']:.4f} | "
            f"{row['coverage_error']:+.4f} | {row['mean_width_kw']:.4f} | "
            f"{'n/a' if row['width_over_mean_absolute_deviation'] is None else format(row['width_over_mean_absolute_deviation'], '.3f')} | "
            f"{row['weighted_interval_score_kw']:.4f} | "
            f"{'n/a' if up_cov is None else format(up_cov, '.4f')} | "
            f"{'n/a' if down_cov is None else format(down_cov, '.4f')} | "
            f"{row['rank']} |"
        )
    return "\n".join(lines)