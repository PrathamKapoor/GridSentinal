"""Oracle expert assignment: an upper bound, computed with hindsight, never deployed.

If the realised target were known, which expert would have been best for each row? That
is the question this module answers, and the answer is the ceiling every deployable
router is measured against. Three uses, none of them operational:

1. **Is routing worth building at all?** The oracle's MAE is the best any selection of
   these three experts can achieve. If it barely beats the best single expert, there is
   almost nothing for a router to find and the honest conclusion is that heterogeneous
   specialists do not decompose this problem.

2. **Routing regret.** For any router's realised selection, the gap between its error and
   the oracle's is the part of the achievable gain the router failed to capture.

3. **Headroom for the fixed ensemble.** The oracle is a hard selection, so a soft
   mixture can in principle beat it. It is not automatically an upper bound on a
   mixture, and this module reports the oracle's own optimal convex combination too, so
   "how much is available at all" is separated from "how much is available by picking".

Everything here reads ``actual_kw``. Every function is therefore offline.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np

__all__ = [
    "ORACLE_NAME",
    "oracle_assignment",
    "oracle_forecast",
    "oracle_report",
    "oracle_mixture",
    "oracle_regret",
    "routing_regret_definition",
]

#: Name used for the oracle in every table, so it can never be mistaken for a model.
ORACLE_NAME = "oracle_expert_assignment"


def oracle_assignment(forecasts: np.ndarray, actual_kw: np.ndarray) -> np.ndarray:
    """Which expert was most accurate, per row and horizon.

    Args:
        forecasts: ``[n_experts, N, horizons]`` in kW.
        actual_kw: ``[N, horizons]`` in kW.

    Returns:
        ``[N, horizons]`` expert index with the smallest absolute error. Ties resolve to
        the lowest expert index, so the assignment is deterministic.
    """
    forecasts = np.asarray(forecasts, dtype=np.float64)
    actual_kw = np.asarray(actual_kw, dtype=np.float64)
    if forecasts.shape[1:] != actual_kw.shape:
        raise ValueError(
            f"forecasts {forecasts.shape} and actual {actual_kw.shape} disagree on "
            f"[rows, horizons]"
        )
    errors = np.abs(forecasts - actual_kw[None, :, :])
    return np.argmin(errors, axis=0)


def oracle_forecast(forecasts: np.ndarray, actual_kw: np.ndarray) -> np.ndarray:
    """The oracle's forecast: each row's most accurate expert.

    This uses the realised target, so it is an upper bound on **hard** routing, not a
    forecast. Do not persist it as one.
    """
    picks = oracle_assignment(forecasts, actual_kw)
    forecasts = np.asarray(forecasts, dtype=np.float64)
    return np.take_along_axis(forecasts, picks[None, :, :], axis=0)[0]


def oracle_mixture(forecasts: np.ndarray, actual_kw: np.ndarray, *, step: float = 0.01) -> np.ndarray:
    """The best *fixed* convex combination, chosen per horizon with hindsight.

    **This is not an upper bound on the router, and the phase measured that it is
    not.** It is a bound on the *fixed* mixtures: one weight vector per horizon, chosen
    without knowing which row it will be used on. A router is a *state-dependent*
    mixture, and Phase 7's router beat this quantity at h=1 (0.3871 against 0.3956 kW) -
    legitimately, because a rule that adapts can beat the best rule that does not, and
    using the test rows to pick the fixed weights is hindsight the router never had.

    So this column answers a different question from the hard oracle's. The hard oracle
    is a per-row selection and *is* an upper bound on hard routing. This is the ceiling
    for any constant weighting, and it is the correct baseline for the honest
    validation-fitted fixed ensemble rather than for the router.

    Args:
        forecasts: ``[n_experts, N, horizons]`` in kW.
        actual_kw: ``[N, horizons]`` in kW.
        step: Simplex grid step.

    Returns:
        ``[N, horizons]`` in kW.
    """
    from ..ensembles import fit_fixed_weights

    forecasts = np.asarray(forecasts, dtype=np.float64)
    n_experts, n_rows, n_horizons = forecasts.shape
    ensemble = fit_fixed_weights(
        expert_forecast_kw=forecasts,
        actual_kw=actual_kw,
        horizons=tuple(range(n_horizons)),
        expert_names=tuple(range(n_experts)),
        step=step,
    )
    from ..ensembles import fixed_forecast

    return fixed_forecast(ensemble, forecasts)


def oracle_report(
    *,
    forecasts: np.ndarray,
    actual_kw: np.ndarray,
    horizons: tuple[int, ...],
    expert_names: tuple[str, ...],
) -> dict[str, Any]:
    """Oracle MAE per horizon, its gain over the best single expert, and the mixture ceiling.

    Attributes measured here:

    ``oracle_mae_kw``
        Hard oracle selection, per horizon.
    ``best_single_mae_kw``
        The best single expert per horizon, chosen on the same rows.
    ``oracle_gain_kw`` / ``oracle_gain_pct``
        How much a perfect selector could win, and as a percentage of the best single
        expert. **This is the number that decides whether the MoE premise holds.** A
        few percent means there is little to route for.
    ``oracle_mixture_mae_kw`` / ``oracle_mixture_gain_pct``
        The best *fixed* convex combination, in hindsight. This bounds fixed weighting,
        not the router - a state-dependent mixture can and here does beat it.
    ``expert_share``
        How often each expert would have been selected. A pool where one expert wins
        almost every row is a pool of one.
    """
    forecasts = np.asarray(forecasts, dtype=np.float64)
    actual_kw = np.asarray(actual_kw, dtype=np.float64)
    horizons = tuple(int(h) for h in horizons)
    picks = oracle_assignment(forecasts, actual_kw)
    errors = np.abs(forecasts - actual_kw[None, :, :])
    oracle_errors = np.take_along_axis(errors, picks[None, :, :], axis=0)[0]
    per_expert = errors.mean(axis=1)
    best_single = per_expert.min(axis=0)
    best_single_name = [expert_names[int(np.argmin(per_expert[:, c]))] for c in range(len(horizons))]

    mixture = oracle_mixture(forecasts, actual_kw)
    mixture_errors = np.abs(mixture - actual_kw)

    by_horizon: dict[str, Any] = {}
    for column, horizon in enumerate(horizons):
        single = float(best_single[column])
        hard = float(oracle_errors[:, column].mean())
        soft = float(mixture_errors[:, column].mean())
        counts = np.bincount(picks[:, column], minlength=len(expert_names))
        by_horizon[str(horizon)] = {
            "oracle_mae_kw": hard,
            "oracle_mixture_mae_kw": soft,
            "best_single_mae_kw": single,
            "best_single_expert": best_single_name[column],
            "oracle_gain_kw": single - hard,
            "oracle_gain_pct": 100.0 * (single - hard) / single if single > 0 else None,
            "oracle_mixture_gain_kw": single - soft,
            "oracle_mixture_gain_pct": 100.0 * (single - soft) / single if single > 0 else None,
            "expert_share": {
                name: float(counts[i]) / max(1, counts.sum())
                for i, name in enumerate(expert_names)
            },
        }
    gains = [
        by_horizon[str(h)]["oracle_gain_pct"] for h in horizons if by_horizon[str(h)]["oracle_gain_pct"] is not None
    ]
    mixture_gains = [
        by_horizon[str(h)]["oracle_mixture_gain_pct"]
        for h in horizons
        if by_horizon[str(h)]["oracle_mixture_gain_pct"] is not None
    ]
    return {
        "name": ORACLE_NAME,
        "is_deployable": False,
        "definition": (
            "per row and horizon, the expert with the smallest realised absolute error; "
            "uses the target, so it is an upper bound and never a forecast"
        ),
        "by_horizon": by_horizon,
        "mean_oracle_gain_pct": float(np.mean(gains)) if gains else None,
        "mean_oracle_mixture_gain_pct": float(np.mean(mixture_gains)) if mixture_gains else None,
    }


def routing_regret_definition() -> dict[str, str]:
    """The exact definitions this phase uses for the routing-regret family.

    Returned as data and embedded in every artifact, because "regret" is used loosely
    enough that two reports can disagree while both sound right.
    """
    return {
        "oracle_error_kw": (
            "mean over rows of |actual - oracle_forecast|, the hard oracle's absolute "
            "error in kW"
        ),
        "routing_regret_kw": (
            "mean over rows of |actual - router_forecast| minus oracle_error_kw. Zero "
            "means the router matched the oracle's selection exactly; positive means "
            "the router gave up part of the achievable gain. Negative is impossible for "
            "a hard router against the hard oracle, and possible only for a soft mixture "
            "against it, because a mixture is not a selection."
        ),
        "routing_regret_pct": (
            "routing_regret_kw divided by the best single expert's MAE at the same "
            "horizon, as a percentage. Normalised so horizons with different error "
            "scales can be compared."
        ),
        "selection_accuracy": (
            "share of rows where the router's highest-weight expert equals the oracle's "
            "expert, per horizon"
        ),
        "capture_ratio": (
            "one minus routing_regret_kw divided by the oracle's own gain over the best "
            "single expert. Zero means the router captured none of the achievable "
            "improvement; one means it matched the oracle; above one means it beat the "
            "hard oracle, which only a soft mixture can do"
        ),
    }


def oracle_regret(
    *,
    router_error: np.ndarray,
    oracle_error: np.ndarray,
    best_single_error: float,
) -> dict[str, float | None]:
    """Routing regret for one method against the oracle.

    Args:
        router_error: ``[N]`` per-row absolute error of the method, one horizon.
        oracle_error: ``[N]`` per-row absolute error of the oracle, same rows.
        best_single_error: The best single expert's MAE at this horizon.

    Returns:
        The regret measures, with the definition strings attached.
    """
    router_error = np.asarray(router_error, dtype=np.float64)
    oracle_error = np.asarray(oracle_error, dtype=np.float64)
    if router_error.shape != oracle_error.shape:
        raise ValueError(
            f"router error {router_error.shape} and oracle error {oracle_error.shape} "
            f"must cover the same rows"
        )
    router_mae = float(router_error.mean())
    oracle_mae = float(oracle_error.mean())
    regret = router_mae - oracle_mae
    headroom = best_single_error - oracle_mae
    return {
        "router_mae_kw": router_mae,
        "oracle_mae_kw": oracle_mae,
        "routing_regret_kw": regret,
        "routing_regret_pct": 100.0 * regret / best_single_error if best_single_error > 0 else None,
        "capture_ratio": (1.0 - regret / headroom) if headroom > 1e-12 else None,
    }


def pairwise_best_single(
    forecasts: np.ndarray, actual_kw: np.ndarray, expert_names: tuple[str, ...]
) -> dict[str, dict[str, float]]:
    """Per-expert MAE, for reference in any table that does not fix the order.

    Args:
        forecasts: ``[n_experts, N, horizons]`` in kW.
        actual_kw: ``[N, horizons]`` in kW.
        expert_names: Expert names, in order.

    Returns:
        ``{expert: {horizon_position: mae_kw}}``.
    """
    forecasts = np.asarray(forecasts, dtype=np.float64)
    errors = np.abs(forecasts - np.asarray(actual_kw, dtype=np.float64)[None, :, :]).mean(axis=1)
    return {
        name: {str(column): float(value) for column, value in enumerate(errors[index])}
        for index, name in enumerate(expert_names)
    }


def best_pair_and_gain(
    forecasts: np.ndarray,
    actual_kw: np.ndarray,
    expert_names: tuple[str, ...],
    *,
    step: float = 0.01,
) -> list[dict[str, Any]]:
    """Best fixed mixture for every two-expert subset, per horizon.

    The oracle gain tells us whether *some* combination helps. This tells us how much of
    it survives when the pool is cut down, which is the practical version of "which
    experts are worth deploying": a pair that matches the full mixture's MAE means the
    third expert is dead weight at inference.

    Args:
        forecasts: ``[n_experts, N, horizons]`` in kW.
        actual_kw: ``[N, horizons]`` in kW.
        expert_names: Expert names, in order.
        step: Simplex grid step for the two-weight search.

    Returns:
        One record per expert pair, sorted by horizon and then best-first.
    """
    from ..ensembles import fit_fixed_weights

    forecasts = np.asarray(forecasts, dtype=np.float64)
    actual_kw = np.asarray(actual_kw, dtype=np.float64)
    n_experts, _n_rows, n_horizons = forecasts.shape
    out: list[dict[str, Any]] = []
    full = oracle_report(
        forecasts=forecasts,
        actual_kw=actual_kw,
        horizons=tuple(range(n_horizons)),
        expert_names=expert_names,
    )
    for left, right in combinations(range(n_experts), 2):
        subset = forecasts[[left, right]]
        pair = fit_fixed_weights(
            expert_forecast_kw=subset,
            actual_kw=actual_kw,
            horizons=tuple(range(n_horizons)),
            expert_names=(expert_names[left], expert_names[right]),
            step=step,
        )
        from ..ensembles import fixed_forecast

        forecast = fixed_forecast(pair, subset)
        errors = np.abs(forecast - actual_kw)
        for column, horizon in enumerate(range(n_horizons)):
            reference = float(full["by_horizon"][str(horizon)]["oracle_mixture_mae_kw"])
            value = float(errors[:, column].mean())
            out.append(
                {
                    "experts": [expert_names[left], expert_names[right]],
                    "horizon_position": column,
                    "pair_mixture_mae_kw": value,
                    "full_mixture_mae_kw": reference,
                    # Positive means dropping the third expert made the mixture worse,
                    # i.e. the third expert earned its place.
                    "gain_from_third_expert_kw": value - reference,
                }
            )
    return out