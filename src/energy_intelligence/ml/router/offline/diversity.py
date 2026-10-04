"""Do these three experts actually make different mistakes?

The purpose of a mixture is complementarity, not headcount. Three models that agree to
three decimal places give a router nothing to decide, and an ensemble built from them is
one model written three times. So diversity is measured **before** any router is
trained, and the oracle gain is read alongside it.

Four quantities, each answering a question that would otherwise be answered by
assertion:

**Error correlation.** Pearson and Spearman correlation of the experts' absolute errors.
High correlation means when one expert is wrong the others are wrong too, by a similar
amount, which is what kills conditional mixing. Note this is the correlation of
*magnitudes*; the signed errors are also reported, because two experts can be equally
bad in magnitude and opposite in sign, which is precisely the case where a 50/50 blend
beats both.

**Agreement rate.** Share of rows where two forecasts are within a tolerance in kW. A
tolerance is needed because demand forecasts never match to the digit, and the tolerance
has to be physically stated rather than tuned - one percent of the row's own demand is
used, with a floor, so it means "the same forecast" in the units an engineer would use.

**Disagreement.** Mean absolute difference between two forecasts, and the distribution's
tail. Where the tail is heavy, a router has real decisions to make on a minority of rows.

**Best-expert share.** How often each expert is the most accurate, per horizon and per
regime. If one expert wins 95% of rows the pool is effectively one model plus noise.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np

__all__ = [
    "signed_errors",
    "absolute_errors",
    "agreement_rate",
    "diversity_report",
    "diversity_by_regime",
    "diversity_by_horizon",
    "correlation_matrix",
]


def signed_errors(forecasts: np.ndarray, actual_kw: np.ndarray) -> np.ndarray:
    """``[n_experts, N, horizons]`` signed residuals, prediction minus truth."""
    forecasts = np.asarray(forecasts, dtype=np.float64)
    actual_kw = np.asarray(actual_kw, dtype=np.float64)
    if forecasts.shape[1:] != actual_kw.shape:
        raise ValueError(
            f"forecasts {forecasts.shape} and actual {actual_kw.shape} disagree"
        )
    return forecasts - actual_kw[None, :, :]


def absolute_errors(forecasts: np.ndarray, actual_kw: np.ndarray) -> np.ndarray:
    """``[n_experts, N, horizons]`` absolute residuals, in kW."""
    return np.abs(signed_errors(forecasts, actual_kw))


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.size < 2:
        return float("nan")
    a = a - a.mean()
    b = b - b.mean()
    denominator = float(np.sqrt(np.sum(a * a) * np.sum(b * b)))
    if denominator <= 1e-15:
        return float("nan")
    return float(np.sum(a * b) / denominator)


def _rank(values: np.ndarray) -> np.ndarray:
    """Average ranks, so ties do not invent an ordering."""
    flat = np.asarray(values, dtype=np.float64).ravel()
    order = np.argsort(flat, kind="mergesort")
    ranks = np.empty(flat.size, dtype=np.float64)
    ranks[order] = np.arange(flat.size, dtype=np.float64)
    # Average the ranks within tied groups.
    unique, inverse, counts = np.unique(flat, return_inverse=True, return_counts=True)
    sums = np.zeros(unique.size, dtype=np.float64)
    np.add.at(sums, inverse, ranks)
    return (sums / counts)[inverse]


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    return _pearson(_rank(a), _rank(b))


def correlation_matrix(
    errors: np.ndarray, *, method: str = "pearson"
) -> np.ndarray:
    """Expert-by-expert correlation of the supplied error arrays.

    Args:
        errors: ``[n_experts, ...]``; every trailing axis is flattened.
        method: ``pearson`` or ``spearman``.

    Returns:
        ``[n_experts, n_experts]``, with ``1.0`` on the diagonal.
    """
    errors = np.asarray(errors, dtype=np.float64)
    if errors.ndim < 2:
        raise ValueError("errors must have at least an expert axis and a sample axis")
    if method not in {"pearson", "spearman"}:
        raise ValueError(f"method must be pearson or spearman, got {method!r}")
    statistic = _pearson if method == "pearson" else _spearman
    count = errors.shape[0]
    matrix = np.eye(count)
    for left, right in combinations(range(count), 2):
        value = statistic(errors[left], errors[right])
        matrix[left, right] = value
        matrix[right, left] = value
    return matrix


def agreement_rate(
    forecasts: np.ndarray,
    actual_kw: np.ndarray,
    *,
    relative_tolerance: float = 0.01,
    absolute_floor_kw: float = 0.05,
) -> dict[str, Any]:
    """How often two experts make "the same" forecast.

    The tolerance is one percent of the row's own demand, with a 0.05 kW floor so a row
    near zero does not require exact agreement to count. It is a physical statement, not
    a tuned parameter, and the floor is recorded.

    Args:
        forecasts: ``[n_experts, N, horizons]`` in kW.
        actual_kw: ``[N, horizons]`` in kW.
        relative_tolerance: Share of the row's demand allowed as disagreement.
        absolute_floor_kw: Lower bound on that allowance.

    Returns:
        ``{"pairs": {...}, "definition": str}``.
    """
    forecasts = np.asarray(forecasts, dtype=np.float64)
    actual = np.asarray(actual_kw, dtype=np.float64)
    allowance = np.maximum(
        np.abs(actual) * float(relative_tolerance), float(absolute_floor_kw)
    )
    names = [f"{left}-{right}" for left, right in combinations(range(forecasts.shape[0]), 2)]
    pairs: dict[str, Any] = {}
    for (left, right), name in zip(combinations(range(forecasts.shape[0]), 2), names):
        difference = np.abs(forecasts[left] - forecasts[right])
        pairs[name] = {
            "agreement_rate": float(np.mean(difference <= allowance)),
            "mean_absolute_difference_kw": float(difference.mean()),
            "p99_absolute_difference_kw": float(np.quantile(difference, 0.99)),
            "share_disagreeing": float(np.mean(difference > allowance)),
        }
    return {
        "definition": (
            f"two experts agree when they differ by at most {relative_tolerance:.0%} of "
            f"the row's own demand, with a {absolute_floor_kw} kW floor"
        ),
        "relative_tolerance": float(relative_tolerance),
        "absolute_floor_kw": float(absolute_floor_kw),
        "pairs": pairs,
    }


def diversity_report(
    *,
    forecasts: np.ndarray,
    actual_kw: np.ndarray,
    horizons: tuple[int, ...],
    expert_names: tuple[str, ...],
) -> dict[str, Any]:
    """Correlations, agreement and best-expert share, per horizon and overall.

    Returns:
        A report with ``overall``, ``by_horizon``, ``agreement`` and ``best_expert_share``.
        ``overall`` pools all horizons, which is what a reader wants first and what a
        per-horizon table alone can hide.
    """
    forecasts = np.asarray(forecasts, dtype=np.float64)
    errors = absolute_errors(forecasts, actual_kw)
    signed = signed_errors(forecasts, actual_kw)
    off = ~np.eye(len(expert_names), dtype=bool)

    def summarise(absolute: np.ndarray, residual: np.ndarray) -> dict[str, Any]:
        pearson = correlation_matrix(absolute, method="pearson")
        spearman = correlation_matrix(absolute, method="spearman")
        signed_pearson = correlation_matrix(residual, method="pearson")
        return {
            "samples": int(absolute.shape[1] * absolute.shape[2]) if absolute.ndim == 3 else int(absolute.shape[1]),
            "error_pearson": pearson.round(4).tolist(),
            "error_spearman": spearman.round(4).tolist(),
            "signed_error_pearson": signed_pearson.round(4).tolist(),
            "mean_abs_error_correlation": float(np.mean(pearson[off])),
            "mean_abs_error_spearman": float(np.mean(spearman[off])),
            "mean_signed_error_correlation": float(np.mean(signed_pearson[off])),
            "mae_kw": {
                name: float(absolute[index].mean()) for index, name in enumerate(expert_names)
            },
        }

    by_horizon: dict[str, Any] = {}
    for column, horizon in enumerate(horizons):
        by_horizon[str(horizon)] = summarise(errors[:, :, column], signed[:, :, column])
    picks = np.argmin(errors, axis=0)
    best_share = {
        str(horizon): {
            name: float(np.count_nonzero(picks[:, column] == index))
            / max(1, picks.shape[0])
            for index, name in enumerate(expert_names)
        }
        for column, horizon in enumerate(horizons)
    }
    return {
        "experts": list(expert_names),
        "horizons": [int(h) for h in horizons],
        "overall": summarise(errors, signed),
        "by_horizon": by_horizon,
        "agreement": agreement_rate(forecasts, actual_kw),
        "best_expert_share": best_share,
    }


def diversity_by_horizon(
    *,
    forecasts: np.ndarray,
    actual_kw: np.ndarray,
    horizons: tuple[int, ...],
    expert_names: tuple[str, ...],
) -> dict[str, dict[str, Any]]:
    """Per-horizon expert MAE and pairwise absolute-error correlation.

    Split out from :func:`diversity_report` because it is the table the phase's
    headline claim rests on: does the best expert change with horizon?
    """
    forecasts = np.asarray(forecasts, dtype=np.float64)
    errors = absolute_errors(forecasts, actual_kw)
    out: dict[str, dict[str, Any]] = {}
    for column, horizon in enumerate(horizons):
        per_expert = errors[:, :, column].mean(axis=1)
        correlation = correlation_matrix(errors[:, :, column], method="pearson")
        off = ~np.eye(len(expert_names), dtype=bool)
        picks = np.argmin(errors[:, :, column], axis=0)
        counts = np.bincount(picks, minlength=len(expert_names))
        out[str(horizon)] = {
            "mae_kw": {
                name: float(per_expert[index]) for index, name in enumerate(expert_names)
            },
            "best_expert": expert_names[int(np.argmin(per_expert))],
            "mean_abs_error_correlation": float(np.mean(correlation[off])),
            "best_expert_share": {
                name: float(counts[index]) / max(1, counts.sum())
                for index, name in enumerate(expert_names)
            },
            "winner_margin_pct": (
                100.0
                * float(np.sort(per_expert)[1] - np.sort(per_expert)[0])
                / float(np.sort(per_expert)[0])
                if per_expert.min() > 0
                else None
            ),
        }
    return out


def diversity_by_regime(
    *,
    forecasts: np.ndarray,
    actual_kw: np.ndarray,
    horizons: tuple[int, ...],
    expert_names: tuple[str, ...],
    axes: tuple[Any, ...],
    headline_horizon: int | None = None,
) -> dict[str, Any]:
    """Best expert and best-expert share, per regime label, per axis.

    Uses the regime axes Phases 4 and 5 already established, assigned from the realised
    demand. Those labels are diagnostics and are explicitly **not** router inputs - see
    ``router.features``.

    Args:
        forecasts: ``[n_experts, N, horizons]`` in kW.
        actual_kw: ``[N, horizons]`` in kW.
        horizons: Horizon steps.
        expert_names: Expert names, in order.
        axes: Regime definitions from :func:`ml.analysis.assign_regimes`, labelled over
            the same rows as the forecasts.
        headline_horizon: Horizon to break down. Defaults to the first.

    Returns:
        One block per axis, with one entry per label.
    """
    forecasts = np.asarray(forecasts, dtype=np.float64)
    horizons = tuple(int(h) for h in horizons)
    column = 0 if headline_horizon is None else horizons.index(int(headline_horizon))
    errors = np.abs(forecasts[:, :, column] - np.asarray(actual_kw, dtype=np.float64)[:, column])
    picks = np.argmin(errors, axis=0)
    out: dict[str, Any] = {}
    for axis in axes:
        labels = np.asarray(axis.labels)
        if labels.shape[0] != errors.shape[1]:
            raise ValueError(
                f"regime axis {axis.name!r} has {labels.shape[0]} labels for "
                f"{errors.shape[1]} rows"
            )
        per_label: dict[str, Any] = {}
        for label in axis.labels_present:
            mask = labels == label
            if not np.any(mask):
                continue
            counts = np.bincount(picks[mask], minlength=len(expert_names))
            per_expert = errors[:, mask].mean(axis=1)
            per_label[str(label)] = {
                "n": int(np.count_nonzero(mask)),
                "share_of_rows": float(np.mean(mask)),
                "mae_kw": {
                    name: float(per_expert[index]) for index, name in enumerate(expert_names)
                },
                "best_expert": expert_names[int(np.argmin(per_expert))],
                "best_expert_share": {
                    name: float(counts[index]) / max(1, counts.sum())
                    for index, name in enumerate(expert_names)
                },
                "oracle_mae_kw": float(errors[:, mask].min(axis=0).mean()),
                "gain_to_best_single_kw": float(
                    per_expert.min() - errors[:, mask].min(axis=0).mean()
                ),
            }
        out[axis.name] = {
            "description": axis.description,
            "horizon": horizons[column],
            "regimes": per_label,
        }
    return out