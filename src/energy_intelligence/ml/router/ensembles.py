"""The ensembles the router has to beat.

Three baselines, all deployable, all built from the same expert forecasts the router
mixes, so no comparison can be won or lost on a different set of predictions.

``uniform``
    Equal weights. Free, and the null hypothesis for "combining helps at all".

``best_single``
    The one expert with the lowest MAE, chosen per horizon on the router's **training**
    rows. This is the bar that matters: a router that cannot beat the best single expert
    has not earned its place, however elegant it is.

``fixed_weight``
    One weight vector per horizon, fitted on the same rows. This is the harder bar,
    because a fixed weighted ensemble already captures the average benefit of
    combining, and a router only adds value if the weights need to *depend on the
    state*. That is the whole hypothesis in one comparison.

The fixed weights come from an exhaustive simplex search at a fixed step, not from a
gradient method. With three experts a 0.01-step grid is 5,151 points per horizon and the
whole thing is deterministic, reproducible and finished in a second; there is nothing a
gradient descent would add except a seed to get wrong.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any

import numpy as np

__all__ = [
    "FixedEnsemble",
    "uniform_weights",
    "fit_fixed_weights",
    "fixed_forecast",
    "best_single_forecast",
    "ensemble_similarity",
]


def uniform_weights(n_rows: int, horizons: tuple[int, ...], n_experts: int) -> np.ndarray:
    """``[n_rows, horizons, experts]`` with every weight ``1 / n_experts``."""
    return np.full((int(n_rows), len(horizons), int(n_experts)), 1.0 / n_experts)


@dataclass(frozen=True, slots=True)
class FixedEnsemble:
    """One weight vector per horizon, fitted rather than learned from the router.

    Attributes:
        weights: ``[horizons, experts]`` rows summing to one.
        horizons: Horizon steps, in order.
        expert_names: Expert names, in order.
        search_step: Grid step used for the search.
        train_mae: MAE of the fitted weights on the rows they were fitted on. Reported
            so the fit's own optimism is visible.
        selected_by: Per horizon, the expert the uniform ensemble would also pick.
    """

    weights: np.ndarray
    horizons: tuple[int, ...]
    expert_names: tuple[str, ...]
    search_step: float
    train_mae: dict[int, float]
    selected_by: dict[int, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "experts": list(self.expert_names),
            "horizons": [str(h) for h in self.horizons],
            "weights": {
                str(h): {
                    name: float(self.weights[i, j])
                    for j, name in enumerate(self.expert_names)
                }
                for i, h in enumerate(self.horizons)
            },
            "search_step": self.search_step,
            "train_mae_kw": {str(h): v for h, v in self.train_mae.items()},
            "uniform_argmax_expert": {str(h): v for h, v in self.selected_by.items()},
        }


def fit_fixed_weights(
    *,
    expert_forecast_kw: np.ndarray,
    actual_kw: np.ndarray,
    horizons: tuple[int, ...],
    expert_names: tuple[str, ...],
    step: float = 0.01,
    rows: np.ndarray | None = None,
) -> FixedEnsemble:
    """Exhaustive simplex search minimising MAE per horizon.

    Args:
        expert_forecast_kw: ``[n_experts, N, horizons]`` in kW.
        actual_kw: ``[N, horizons]`` in kW.
        horizons: Horizon steps.
        expert_names: Expert names, in order.
        step: Grid step over each weight; the simplex remainder goes to the last expert
            so every grid point sums to exactly one.
        rows: Positions to fit on. Defaults to all. Must **not** include test rows.

    Returns:
        The fitted ensemble.

    Raises:
        ValueError: If ``step`` is not a positive fraction, or the shapes disagree.
    """
    expert_forecast_kw = np.asarray(expert_forecast_kw, dtype=np.float64)
    actual_kw = np.asarray(actual_kw, dtype=np.float64)
    horizons = tuple(int(h) for h in horizons)
    n_experts, n_rows, n_horizons = expert_forecast_kw.shape
    if actual_kw.shape != (n_rows, n_horizons) or n_horizons != len(horizons):
        raise ValueError(
            f"shapes disagree: forecasts {expert_forecast_kw.shape}, "
            f"actual {actual_kw.shape}, horizons {horizons}"
        )
    if not 0.0 < step <= 1.0:
        raise ValueError(f"step must lie in (0, 1], got {step}")
    if rows is not None:
        rows = np.asarray(rows, dtype=np.int64)
        expert_forecast_kw = expert_forecast_kw[:, rows, :]
        actual_kw = actual_kw[rows, :]
    if expert_forecast_kw.shape[1] == 0:
        raise ValueError("cannot fit fixed weights on zero rows")

    grid = _simplex_grid(n_experts, step)
    weights = np.zeros((len(horizons), n_experts))
    train_mae: dict[int, float] = {}
    selected_by: dict[int, str] = {}
    n_fit = int(expert_forecast_kw.shape[1])
    for column, horizon in enumerate(horizons):
        scores = _simplex_mae(
            grid,
            expert_forecast_kw[:, :, column],
            actual_kw[:, column],
            chunk_rows=_chunk_rows(grid.shape[0]),
        )
        best = int(np.argmin(scores))
        weights[column] = grid[best]
        train_mae[horizon] = float(scores[best])
        selected_by[horizon] = expert_names[int(np.argmax(grid[best]))]
    assert n_fit >= 1
    return FixedEnsemble(
        weights=weights,
        horizons=horizons,
        expert_names=tuple(expert_names),
        search_step=float(step),
        train_mae=train_mae,
        selected_by=selected_by,
    )


def _chunk_rows(n_grid: int, *, budget_elements: int = 8_000_000) -> int:
    """How many rows to score at once without materialising the whole grid.

    The naive form of this search is a ``[5151, 359040]`` array - 14.8 GB in float64,
    which is not a search, it is an out-of-memory error. Chunking trades a few more
    matmuls for a bounded allocation and the identical answer.
    """
    return max(1, int(budget_elements // max(1, n_grid)))


def _simplex_mae(
    grid: np.ndarray, forecasts: np.ndarray, actual: np.ndarray, *, chunk_rows: int
) -> np.ndarray:
    """Mean absolute error of every grid point, accumulated in row chunks.

    Args:
        grid: ``[n_grid, n_experts]``, rows summing to one.
        forecasts: ``[n_experts, n_rows]`` in kW.
        actual: ``[n_rows]`` in kW.
        chunk_rows: Rows scored per pass.

    Returns:
        ``[n_grid]`` mean absolute errors.
    """
    n_grid = int(grid.shape[0])
    n_rows = int(forecasts.shape[1])
    totals = np.zeros(n_grid, dtype=np.float64)
    for start in range(0, n_rows, int(chunk_rows)):
        stop = min(n_rows, start + int(chunk_rows))
        predicted = grid @ forecasts[:, start:stop]
        totals += np.abs(predicted - actual[None, start:stop]).sum(axis=1)
    return totals / max(1, n_rows)


def _simplex_grid(n_experts: int, step: float) -> np.ndarray:
    """All weight vectors on a ``step``-grid simplex that sum to exactly one.

    The first ``n_experts - 1`` weights sweep the grid and the last is the remainder.
    That keeps every point exactly on the simplex instead of near it, which matters
    because :func:`routing.mix` refuses weights that do not sum to one.
    """
    if n_experts < 2:
        raise ValueError(f"a fixed ensemble needs at least two experts, got {n_experts}")
    steps = int(round(1.0 / step))
    partial = np.array(
        [
            np.array(point, dtype=np.float64) / steps
            for point in product(range(steps + 1), repeat=n_experts - 1)
            if sum(point) <= steps
        ]
    )
    return np.column_stack([partial, 1.0 - partial.sum(axis=1)])


def fixed_forecast(
    ensemble: FixedEnsemble, expert_forecast_kw: np.ndarray
) -> np.ndarray:
    """Apply fitted per-horizon weights.

    Args:
        ensemble: The fitted weights.
        expert_forecast_kw: ``[n_experts, N, horizons]`` in kW.

    Returns:
        ``[N, horizons]`` in kW.
    """
    expert_forecast_kw = np.asarray(expert_forecast_kw, dtype=np.float64)
    weights = np.asarray(ensemble.weights, dtype=np.float64)
    if expert_forecast_kw.shape[0] != weights.shape[1]:
        raise ValueError(
            f"the ensemble has {weights.shape[1]} experts but got "
            f"{expert_forecast_kw.shape[0]} forecasts"
        )
    if expert_forecast_kw.shape[2] != weights.shape[0]:
        raise ValueError(
            f"the ensemble has {weights.shape[0]} horizons but got "
            f"{expert_forecast_kw.shape[2]}"
        )
    return np.einsum("he,enh->nh", weights, expert_forecast_kw)


def select_best_single(
    *,
    expert_forecast_kw: np.ndarray,
    actual_kw: np.ndarray,
    horizons: tuple[int, ...],
    expert_names: tuple[str, ...],
    rows: np.ndarray | None = None,
) -> dict[int, int]:
    """Which single expert is most accurate per horizon, on the given rows.

    Selection and application are separate because the rows that *choose* the expert and
    the rows it is then *applied to* are not the same set: the choice is made on the
    router's training rows and applied to every panel row. Conflating them silently
    produces a forecast array with the wrong number of rows.

    Args:
        expert_forecast_kw: ``[n_experts, N, horizons]`` in kW.
        actual_kw: ``[N, horizons]`` in kW.
        horizons: Horizon steps.
        expert_names: Expert names, in order.
        rows: Positions to select on. Defaults to all. Must not include test rows.

    Returns:
        ``{horizon: expert index}``.

    Raises:
        ValueError: If no row is available to select on.
    """
    expert_forecast_kw = np.asarray(expert_forecast_kw, dtype=np.float64)
    actual_kw = np.asarray(actual_kw, dtype=np.float64)
    horizons = tuple(int(h) for h in horizons)
    selection = expert_forecast_kw if rows is None else expert_forecast_kw[:, np.asarray(rows), :]
    truth = actual_kw if rows is None else actual_kw[np.asarray(rows), :]
    if selection.shape[1] == 0:
        raise ValueError("cannot select a best single expert on zero rows")
    mae = np.abs(selection - truth[None, :, :]).mean(axis=1)
    return {h: int(np.argmin(mae[:, column])) for column, h in enumerate(horizons)}


def apply_best_single(
    *, expert_forecast_kw: np.ndarray, picks: dict[int, int], horizons: tuple[int, ...]
) -> np.ndarray:
    """Apply a per-horizon expert selection to every row.

    Args:
        expert_forecast_kw: ``[n_experts, N, horizons]`` in kW.
        picks: ``{horizon: expert index}``.
        horizons: Horizon steps, in column order.

    Returns:
        ``[N, horizons]`` in kW.

    Raises:
        ValueError: If a horizon has no selection, or a selection is out of range.
    """
    expert_forecast_kw = np.asarray(expert_forecast_kw, dtype=np.float64)
    horizons = tuple(int(h) for h in horizons)
    missing = [h for h in horizons if h not in picks]
    if missing:
        raise ValueError(f"no expert selected for horizon(s) {missing}")
    columns = []
    for horizon in horizons:
        index = picks[horizon]
        if not 0 <= index < expert_forecast_kw.shape[0]:
            raise ValueError(
                f"expert index {index} for horizon {horizon} is outside the pool"
            )
        columns.append(expert_forecast_kw[index, :, horizons.index(horizon)])
    return np.stack(columns, axis=1)


def best_single_forecast(
    *,
    expert_forecast_kw: np.ndarray,
    actual_kw: np.ndarray,
    horizons: tuple[int, ...],
    expert_names: tuple[str, ...],
    rows: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[int, str]]:
    """Select the best single expert per horizon and apply it to every row.

    Args:
        expert_forecast_kw: ``[n_experts, N, horizons]`` in kW.
        actual_kw: ``[N, horizons]`` in kW.
        horizons: Horizon steps.
        expert_names: Expert names, in order.
        rows: Positions to select on. Defaults to all. Must not include test rows.

    Returns:
        ``(forecast_kw, chosen_by_horizon)``.
    """
    picks = select_best_single(
        expert_forecast_kw=expert_forecast_kw,
        actual_kw=actual_kw,
        horizons=horizons,
        expert_names=expert_names,
        rows=rows,
    )
    forecast = apply_best_single(
        expert_forecast_kw=expert_forecast_kw, picks=picks, horizons=tuple(horizons)
    )
    return forecast, {h: expert_names[index] for h, index in picks.items()}


def ensemble_similarity(
    weights_a: np.ndarray, weights_b: np.ndarray
) -> dict[str, float]:
    """How far apart two weight matrices are.

    Reported so "the router learned something" can be checked rather than asserted:
    if its mean weights are within noise of the fixed ensemble's, it has rediscovered a
    constant, and the horizon-specific weights should say so.

    Args:
        weights_a: ``[horizons, experts]`` or ``[rows, horizons, experts]``.
        weights_b: The same shape.

    Returns:
        Mean absolute weight difference, and the largest per-expert difference.
    """
    a = np.asarray(weights_a, dtype=np.float64)
    b = np.asarray(weights_b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    difference = np.abs(a - b)
    return {
        "mean_absolute_weight_difference": float(difference.mean()),
        "max_absolute_weight_difference": float(difference.max()),
    }