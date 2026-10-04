"""Scoring the router: forecast error, routing quality, stability, calibration, cost.

Four separate questions, deliberately kept apart because a single number cannot answer
them and because conflating them is how a routing system comes to look successful when
it is not:

1. **Is the forecast better?** MAE and RMSE in kW against every baseline, per horizon,
   from the same test rows, using ``ml.metrics``.
2. **Is the routing right?** How often the highest-weight expert equals the oracle's,
   and how much of the oracle's achievable gain was captured.
3. **Is the routing sensible?** Do the weights move smoothly in time, and do they mean
   what they say - a weight of 0.9 on an expert should coincide with that expert being
   the best one far more often than 9% of the time, or the confidence is decorative.
4. **What does it cost?** Parameters, seconds and the expert time it does not replace.
   Routing only pays if it buys accuracy for a cost that is worth naming.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from ...metrics import evaluate
from .oracle import oracle_assignment, oracle_regret

__all__ = [
    "method_metrics",
    "routing_quality",
    "routing_stability",
    "router_calibration",
    "router_by_regime",
    "failure_analysis",
    "compute_cost",
    "forecast_comparison_table",
]


def method_metrics(
    *,
    actual_kw: np.ndarray,
    forecast_kw: np.ndarray,
    horizons: tuple[int, ...],
) -> dict[str, dict[str, Any]]:
    """Project metrics for one method, one number per horizon column.

    Args:
        actual_kw: ``[N, horizons]`` in kW.
        forecast_kw: ``[N, horizons]`` in kW.
        horizons: Horizon steps.

    Returns:
        ``{"1": {...}, "4": {...}}`` with the same keys Phase 4 and 5 recorded.
    """
    actual_kw = np.asarray(actual_kw, dtype=np.float64)
    forecast_kw = np.asarray(forecast_kw, dtype=np.float64)
    horizons = tuple(int(h) for h in horizons)
    if actual_kw.shape != forecast_kw.shape or actual_kw.shape[1] != len(horizons):
        raise ValueError(
            f"actual {actual_kw.shape} and forecast {forecast_kw.shape} must both be "
            f"[rows, {len(horizons)}]"
        )
    out: dict[str, dict[str, Any]] = {}
    for column, horizon in enumerate(horizons):
        payload = evaluate(actual_kw[:, column], forecast_kw[:, column]).to_dict()
        payload["unit"] = "kW"
        payload["horizon_steps"] = horizon
        out[str(horizon)] = payload
    return out


def routing_quality(
    *,
    forecast_kw: np.ndarray,
    weights: np.ndarray,
    expert_forecast_kw: np.ndarray,
    actual_kw: np.ndarray,
    horizons: tuple[int, ...],
    expert_names: tuple[str, ...],
) -> dict[str, Any]:
    """Selection accuracy and the routing-regret family, per horizon.

    Args:
        forecast_kw: ``[N, horizons]`` the router's output.
        weights: ``[N, horizons, experts]`` routing weights.
        expert_forecast_kw: ``[n_experts, N, horizons]`` in kW.
        actual_kw: ``[N, horizons]`` in kW.
        horizons: Horizon steps.
        expert_names: Expert names, in order.

    Returns:
        Per-horizon selection accuracy, regret, capture ratio, weight concentration and
        the weighted share each expert receives.
    """
    forecast_kw = np.asarray(forecast_kw, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    expert_forecast_kw = np.asarray(expert_forecast_kw, dtype=np.float64)
    actual_kw = np.asarray(actual_kw, dtype=np.float64)
    horizons = tuple(int(h) for h in horizons)
    picks = oracle_assignment(expert_forecast_kw, actual_kw)
    oracle_errors = np.abs(
        np.take_along_axis(expert_forecast_kw, picks[None, :, :], axis=0)[0] - actual_kw
    )
    per_expert = np.abs(expert_forecast_kw - actual_kw[None, :, :]).mean(axis=1)
    router_picks = np.argmax(weights, axis=2)

    out: dict[str, Any] = {}
    for column, horizon in enumerate(horizons):
        best_single = float(per_expert[:, column].min())
        regret = oracle_regret(
            router_error=np.abs(forecast_kw[:, column] - actual_kw[:, column]),
            oracle_error=oracle_errors[:, column],
            best_single_error=best_single,
        )
        selected = router_picks[:, column]
        oracle_choice = picks[:, column]
        weight_column = weights[:, column, :]
        out[str(horizon)] = {
            **regret,
            "selection_accuracy": float(np.mean(selected == oracle_choice)),
            "top2_oracle_coverage": float(
                np.mean(
                    (selected == oracle_choice)
                    | (
                        np.argsort(np.abs(expert_forecast_kw[:, :, column] - actual_kw[None, :, column]), axis=0)[1, :]
                        == selected
                    )
                )
            ),
            "mean_max_weight": float(weight_column.max(axis=1).mean()),
            "mean_entropy": float(
                _entropy(weight_column).mean()
            ),
            "hard_share": float(np.mean(weight_column.max(axis=1) >= 0.999)),
            "weighted_share": {
                name: float(weight_column[:, index].mean())
                for index, name in enumerate(expert_names)
            },
            "selected_share": {
                name: float(np.count_nonzero(selected == index)) / max(1, selected.size)
                for index, name in enumerate(expert_names)
            },
            "oracle_selected_share": {
                name: float(np.count_nonzero(oracle_choice == index))
                / max(1, oracle_choice.size)
                for index, name in enumerate(expert_names)
            },
        }
    return out


def _entropy(weights: np.ndarray) -> np.ndarray:
    """Shannon entropy of each row's weight vector, in nats."""
    weights = np.clip(np.asarray(weights, dtype=np.float64), 1e-12, 1.0)
    return -np.sum(weights * np.log(weights), axis=1)


def routing_stability(
    *,
    weights: np.ndarray,
    origins: np.ndarray,
    series: np.ndarray,
    horizons: tuple[int, ...],
    expert_names: tuple[str, ...],
) -> dict[str, Any]:
    """How violently the weights move between consecutive forecast origins.

    Measured **within a series**, across consecutive origins of the same split, because
    consecutive rows in panel order are consecutive origins for the same series only if
    the ordering is origin-major. A router that flips from 0.80 to 0.05 on the GBM
    between two neighbouring 15-minute steps is not adapting to a regime, it is
    oscillating, and the difference is visible in this number.

    Reported without a smoothing penalty applied, as the phase requires: there is no
    evidence yet that oscillation costs accuracy, so imposing a penalty would be an
    unmeasured assumption.

    Args:
        weights: ``[N, horizons, experts]``.
        origins: ``[N]`` forecast origin index per row.
        series: ``[N]`` series index per row.
        horizons: Horizon steps.
        expert_names: Expert names, in order.

    Returns:
        Per-horizon total variation, the share of steps that flip the selected expert,
        and the largest single-step move.
    """
    weights = np.asarray(weights, dtype=np.float64)
    origins = np.asarray(origins, dtype=np.int64)
    series = np.asarray(series, dtype=np.int64)
    horizons = tuple(int(h) for h in horizons)
    out: dict[str, Any] = {}
    for column, horizon in enumerate(horizons):
        differences: list[np.ndarray] = []
        flips = 0
        pairs = 0
        for index in np.unique(series):
            rows = np.flatnonzero(series == index)
            rows = rows[np.argsort(origins[rows], kind="mergesort")]
            if rows.size < 2:
                continue
            current = weights[rows, column, :]
            step = np.abs(np.diff(current, axis=0))
            differences.append(step)
            changed = np.argmax(current[:-1], axis=1) != np.argmax(current[1:], axis=1)
            flips += int(np.count_nonzero(changed))
            pairs += int(changed.size)
        if not differences:
            out[str(horizon)] = {"pairs": 0}
            continue
        stacked = np.concatenate(differences, axis=0)
        total_variation = stacked.sum(axis=1)
        largest = {
            name: float(stacked[:, index].max()) for index, name in enumerate(expert_names)
        }
        out[str(horizon)] = {
            "pairs": int(pairs),
            "mean_total_variation": float(total_variation.mean()),
            "median_total_variation": float(np.median(total_variation)),
            "p99_total_variation": float(np.quantile(total_variation, 0.99)),
            "selection_flip_rate": float(flips / pairs) if pairs else None,
            "max_step_change_per_expert": largest,
        }
    return out


def router_calibration(
    *,
    weights: np.ndarray,
    expert_forecast_kw: np.ndarray,
    actual_kw: np.ndarray,
    horizons: tuple[int, ...],
    expert_names: tuple[str, ...],
    bins: int = 10,
) -> dict[str, Any]:
    """Whether the top weight means anything.

    For each row the router's claim is "expert *e*, with confidence *c*". The check bins
    rows by *c* and asks how often *e* was in fact the most accurate. A router whose
    0.9-confidence rows are right 90% of the time is calibrated in the useful sense. One
    whose 0.9-confidence rows are right 60% of the time is producing a number that looks
    like confidence and is not.

    Weights summing to one is not calibration, and the gap between the two is the point
    of measuring this.
    """
    weights = np.asarray(weights, dtype=np.float64)
    expert_forecast_kw = np.asarray(expert_forecast_kw, dtype=np.float64)
    actual_kw = np.asarray(actual_kw, dtype=np.float64)
    horizons = tuple(int(h) for h in horizons)
    picks = oracle_assignment(expert_forecast_kw, actual_kw)
    chosen = np.argmax(weights, axis=2)
    confidence = np.max(weights, axis=2)
    correct = (chosen == picks).astype(np.float64)

    out: dict[str, Any] = {}
    for column, horizon in enumerate(horizons):
        conf = confidence[:, column]
        hit = correct[:, column]
        edges = np.quantile(conf, np.linspace(0.0, 1.0, int(bins) + 1))
        edges[-1] = min(1.0, edges[-1] + 1e-9)
        table: list[dict[str, Any]] = []
        predicted = 0.0
        observed = 0.0
        for b in range(int(bins)):
            low, high = float(edges[b]), float(edges[b + 1])
            if b == int(bins) - 1:
                mask = (conf >= low) & (conf <= high)
            else:
                mask = (conf >= low) & (conf < high)
            if not np.any(mask):
                continue
            entry = {
                "confidence_low": round(low, 4),
                "confidence_high": round(high, 4),
                "mean_confidence": round(float(conf[mask].mean()), 4),
                "observed_accuracy": round(float(hit[mask].mean()), 4),
                "rows": int(np.count_nonzero(mask)),
            }
            table.append(entry)
            predicted += float(conf[mask].mean()) * np.count_nonzero(mask)
            observed += float(hit[mask].mean()) * np.count_nonzero(mask)
        total = max(1, conf.size)
        brier = float(np.mean((conf - hit) ** 2))
        out[str(horizon)] = {
            "bins": table,
            "expected_calibration_error": float(
                sum(
                    entry["rows"] / total
                    * abs(entry["mean_confidence"] - entry["observed_accuracy"])
                    for entry in table
                )
            ),
            "mean_confidence": predicted / total,
            "mean_selection_accuracy": observed / total,
            "confidence_minus_accuracy": predicted / total - observed / total,
            "brier_score": brier,
            "base_rate": float(hit.mean()),
        }
    return out


def failure_analysis(
    *,
    forecast_kw: np.ndarray,
    expert_forecast_kw: np.ndarray,
    weights: np.ndarray,
    actual_kw: np.ndarray,
    origins: np.ndarray,
    horizons: tuple[int, ...],
    expert_names: tuple[str, ...],
    top_rows: int = 20,
) -> dict[str, Any]:
    """Where the router loses, sorted by how much it could have won.

    Two lists. The **regret list** is rows where the oracle would have done much better
    than the router did - those are routing mistakes, and they are the ones worth
    understanding. The **harm list** is rows where the router is worse than the best
    single expert by more than a threshold - those are the rows a naive reading of "the
    router beat the baseline on average" would hide.

    Args:
        forecast_kw: ``[N, horizons]`` router output.
        expert_forecast_kw: ``[n_experts, N, horizons]`` in kW.
        weights: ``[N, horizons, experts]``.
        actual_kw: ``[N, horizons]`` in kW.
        origins: ``[N]`` forecast origin index, for the reported timestamps.
        horizons: Horizon steps.
        expert_names: Expert names, in order.
        top_rows: How many rows to list per horizon.

    Returns:
        Per horizon, the two lists plus summary counts.
    """
    forecast_kw = np.asarray(forecast_kw, dtype=np.float64)
    expert_forecast_kw = np.asarray(expert_forecast_kw, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    actual_kw = np.asarray(actual_kw, dtype=np.float64)
    horizons = tuple(int(h) for h in horizons)
    picks = oracle_assignment(expert_forecast_kw, actual_kw)
    oracle_forecast = np.take_along_axis(expert_forecast_kw, picks[None, :, :], axis=0)[0]
    errors = np.abs(expert_forecast_kw - actual_kw[None, :, :])
    per_expert_mae = errors.mean(axis=1)

    out: dict[str, Any] = {}
    for column, horizon in enumerate(horizons):
        router_error = np.abs(forecast_kw[:, column] - actual_kw[:, column])
        oracle_error = np.abs(oracle_forecast[:, column] - actual_kw[:, column])
        best_single = int(np.argmin(per_expert_mae[:, column]))
        # Same sign convention as ``routing_regret_kw``: positive means the router gave
        # up achievable gain. Two different conventions in one artifact would make the
        # failure list look like a success list.
        regret = router_error - oracle_error
        harm = router_error - errors[best_single, :, column]
        regret_order = np.argsort(-regret)[: int(top_rows)]
        harm_order = np.argsort(-harm)[: int(top_rows)]

        def describe(rows: np.ndarray, values: np.ndarray) -> list[dict[str, Any]]:
            out_rows = []
            for row in rows:
                weights_row = weights[row, column, :]
                out_rows.append(
                    {
                        "panel_row": int(row),
                        "origin": int(origins[row]),
                        "actual_kw": float(actual_kw[row, column]),
                        "router_kw": float(forecast_kw[row, column]),
                        "router_error_kw": float(router_error[row]),
                        "oracle_expert": expert_names[int(picks[row, column])],
                        "oracle_error_kw": float(oracle_error[row]),
                        "router_weights": {
                            name: round(float(weights_row[i]), 4)
                            for i, name in enumerate(expert_names)
                        },
                        "value": round(float(values[row]), 4),
                    }
                )
            return out_rows

        out[str(horizon)] = {
            "worst_routing_regret": describe(regret_order, regret),
            "worst_harm_vs_best_single": describe(harm_order, harm),
            "rows_worse_than_best_single": int(np.count_nonzero(harm > 0)),
            "share_worse_than_best_single": float(np.mean(harm > 0)),
            "share_worse_than_best_single_by_more_than_1kw": float(np.mean(harm > 1.0)),
            "p99_harm_kw": float(np.quantile(harm, 0.99)),
            "regret_by_selection": _regret_by_selection(
                np.argmax(weights[:, column, :], axis=1),
                picks[:, column],
                regret,
                expert_names,
            ),
        }
    return out


def _regret_by_selection(
    chosen: np.ndarray,
    oracle: np.ndarray,
    regret: np.ndarray,
    expert_names: tuple[str, ...],
) -> dict[str, Any]:
    """Mean regret broken down by which expert the router picked."""
    out: dict[str, Any] = {}
    for index, name in enumerate(expert_names):
        mask = chosen == index
        if not np.any(mask):
            continue
        out[name] = {
            "rows": int(np.count_nonzero(mask)),
            "mean_regret_kw": float(regret[mask].mean()),
            "hit_rate": float(np.mean(oracle[mask] == index)),
            "mean_regret_when_correct_kw": (
                float(regret[mask & (oracle == index)].mean())
                if np.any(mask & (oracle == index))
                else None
            ),
            "mean_regret_when_wrong_kw": (
                float(regret[mask & (oracle != index)].mean())
                if np.any(mask & (oracle != index))
                else None
            ),
        }
    return out


def router_by_regime(
    *,
    router_forecast_kw: np.ndarray,
    router_weights: np.ndarray,
    method_forecast_kw: dict[str, np.ndarray],
    expert_forecast_kw: np.ndarray,
    actual_kw: np.ndarray,
    axes: tuple[Any, ...],
    horizons: tuple[int, ...],
    expert_names: tuple[str, ...],
    headline_horizon: int | None = None,
) -> dict[str, Any]:
    """Router behaviour inside each regime, next to the baselines it has to beat.

    This is the table that answers the phase's regime question. Expert preference alone
    is not enough: it says which expert wins on average in a regime, not whether the
    router shifts toward that expert when it sees the regime. So each regime reports the
    router's MAE, its mean weights, the oracle's MAE, the best single expert, the fixed
    ensemble and the uniform ensemble. A regime where the router beats both ensembles
    while loading the expert that wins there is a regime where routing worked.

    Args:
        router_forecast_kw: ``[N, horizons]`` the router's output.
        router_weights: ``[N, horizons, experts]`` its weights.
        method_forecast_kw: ``{name: [N, horizons]}`` baseline forecasts to compare with.
        expert_forecast_kw: ``[n_experts, N, horizons]`` in kW.
        actual_kw: ``[N, horizons]`` in kW.
        axes: Regime definitions, labelled over the same rows.
        horizons: Horizon steps.
        expert_names: Expert names, in order.
        headline_horizon: Horizon to break down. Defaults to the first.

    Returns:
        One block per axis, one entry per label.
    """
    router_forecast_kw = np.asarray(router_forecast_kw, dtype=np.float64)
    router_weights = np.asarray(router_weights, dtype=np.float64)
    expert_forecast_kw = np.asarray(expert_forecast_kw, dtype=np.float64)
    actual_kw = np.asarray(actual_kw, dtype=np.float64)
    horizons = tuple(int(h) for h in horizons)
    column = 0 if headline_horizon is None else horizons.index(int(headline_horizon))

    router_error = np.abs(router_forecast_kw[:, column] - actual_kw[:, column])
    expert_errors = np.abs(
        expert_forecast_kw[:, :, column] - actual_kw[:, column][None, :]
    )
    oracle_error = expert_errors.min(axis=0)
    best_single_index = int(np.argmin(expert_errors.mean(axis=1)))
    best_single_error = expert_errors[best_single_index]

    out: dict[str, Any] = {}
    for axis in axes:
        labels = np.asarray(axis.labels)
        if labels.shape[0] != router_error.shape[0]:
            raise ValueError(
                f"regime axis {axis.name!r} has {labels.shape[0]} labels for "
                f"{router_error.shape[0]} rows"
            )
        per_label: dict[str, Any] = {}
        for label in axis.labels_present:
            mask = labels == label
            if not np.any(mask):
                continue
            entry: dict[str, Any] = {
                "n": int(np.count_nonzero(mask)),
                "share_of_rows": float(np.mean(mask)),
                "router_mae_kw": float(router_error[mask].mean()),
                "best_single_expert": expert_names[best_single_index],
                "best_single_expert_mae_kw": float(best_single_error[mask].mean()),
                "oracle_mae_kw": float(oracle_error[mask].mean()),
                "mean_router_weights": {
                    name: float(router_weights[mask, column, index].mean())
                    for index, name in enumerate(expert_names)
                },
                "selected_expert_share": {
                    name: float(
                        np.count_nonzero(np.argmax(router_weights[mask, column, :], axis=1) == index)
                    )
                    / max(1, int(np.count_nonzero(mask)))
                    for index, name in enumerate(expert_names)
                },
                "oracle_expert_share": {
                    name: float(
                        np.count_nonzero(np.argmin(expert_errors[:, mask], axis=0) == index)
                    )
                    / max(1, int(np.count_nonzero(mask)))
                    for index, name in enumerate(expert_names)
                },
            }
            for name, forecast in method_forecast_kw.items():
                errors = np.abs(
                    np.asarray(forecast, dtype=np.float64)[:, column] - actual_kw[:, column]
                )
                entry[f"{name}_mae_kw"] = float(errors[mask].mean())
            per_label[str(label)] = entry
        out[axis.name] = {
            "description": axis.description,
            "horizon": horizons[column],
            "regimes": per_label,
        }
    return out


def compute_cost(
    *,
    router_parameters: int,
    router_train_seconds: float,
    router_inference_seconds: float,
    expert_seconds: dict[str, float],
    expert_rows: int,
    router_rows: int,
    expert_parameters: dict[str, int] | None = None,
) -> dict[str, Any]:
    """The cost table, in the units a reader can compare against Phase 6's.

    Phase 6 measured 0.248 s per sample for a frozen 1.7B-parameter forward pass. The
    comparison that matters is therefore not absolute seconds but the ratio: what does
    routing cost on top of running the experts it routes over, and is the router a
    rounding error next to them or the dominant term.
    """
    expert_total = float(sum(expert_seconds.values()))
    if router_rows > 0:
        per_row_router = router_inference_seconds / router_rows
        per_row_expert = expert_total / expert_rows if expert_rows else 0.0
        overhead = (per_row_router / per_row_expert) if per_row_expert > 0 else None
    else:  # pragma: no cover - guarded by the caller
        per_row_router = per_row_expert = 0.0
        overhead = None
    return {
        "router_parameters": int(router_parameters),
        "expert_parameters": dict(expert_parameters or {}),
        "router_train_seconds": round(float(router_train_seconds), 3),
        "router_inference_seconds": round(float(router_inference_seconds), 3),
        "expert_seconds": {k: round(float(v), 3) for k, v in expert_seconds.items()},
        "expert_total_seconds": round(expert_total, 3),
        "rows": int(router_rows),
        "router_seconds_per_row": per_row_router,
        "expert_seconds_per_row": per_row_expert,
        "router_overhead_vs_experts": overhead,
        "note": (
            "routing does not avoid running the experts it routes over, so its cost is "
            "additive unless an expert can be skipped; hard routing only pays for that "
            "if expert inference can be gated, which this implementation does not do. "
            "Expert sizes are not comparable with each other or with the router's: the "
            "TCN figure counts weights, the GBM figure counts tree nodes, and "
            "persistence has none"
        ),
    }


def forecast_comparison_table(
    *,
    methods: dict[str, np.ndarray],
    actual_kw: np.ndarray,
    horizons: tuple[int, ...],
    headline: int | None = None,
) -> str:
    """The phase's headline table, as text.

    Args:
        methods: ``{method_name: [N, horizons] forecast}``.
        actual_kw: ``[N, horizons]`` in kW.
        horizons: Horizon steps.
        headline: Horizon to sort by. Defaults to the first.

    Returns:
        A markdown table of MAE and RMSE in kW, best per horizon marked.
    """
    horizons = tuple(int(h) for h in horizons)
    metrics = {
        name: method_metrics(actual_kw=actual_kw, forecast_kw=forecast, horizons=horizons)
        for name, forecast in methods.items()
    }
    column = 0 if headline is None else horizons.index(int(headline))
    headline_key = str(horizons[column])
    order = sorted(methods, key=lambda name: metrics[name][headline_key]["mae"])
    # Bold the **best** value in each column, not the value in the headline column.
    # Highlighting the column instead of the winner would bold every row and say nothing.
    best = {
        key: min(methods, key=lambda name: metrics[name][key]["mae"])
        for key in (str(h) for h in horizons)
    }

    def cell(name: str, key: str, value: float) -> str:
        return f"**{value:.4f}**" if best[key] == name else f"{value:.4f}"

    lines = [
        "| method | " + " | ".join(f"MAE h={h}" for h in horizons)
        + " | " + " | ".join(f"RMSE h={h}" for h in horizons) + " |",
        "|---" * (1 + 2 * len(horizons)) + "|",
    ]
    for name in order:
        mae = [metrics[name][str(h)]["mae"] for h in horizons]
        rmse = [metrics[name][str(h)]["rmse"] for h in horizons]
        lines.append(
            f"| {name} | "
            + " | ".join(cell(name, str(h), v) for h, v in zip(horizons, mae))
            + " | "
            + " | ".join(f"{v:.4f}" for v in rmse)
            + " |"
        )
    lines.append("")
    lines.append(f"Best at h={horizons[column]}: **{best[headline_key]}**.")
    return "\n".join(lines)


def timing_block(fn, *args, **kwargs) -> tuple[Any, float]:
    """Run ``fn`` and return its result and wall time.

    Small helper so every timing in the phase is measured the same way, with
    ``perf_counter``, rather than each call site choosing its own clock.
    """
    started = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, time.perf_counter() - started