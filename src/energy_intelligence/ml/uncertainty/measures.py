"""Expert disagreement, and whether it knows anything about future error.

Phase 7 measured that the three experts *disagree*: on 59-75% of rows their forecasts
differ by more than 1% of the row's demand, and their error correlation falls from 0.90 at
15 minutes to 0.69 at 24 hours. That establishes diversity. It does **not** establish
that the diversity is informative. Those are separate claims, and only the second one
makes disagreement useful as an uncertainty signal.

The distinction matters because the intuitive form of the hypothesis -

    the experts disagree  =>  the forecast is uncertain

- is easy to accept and is not guaranteed. Three models can disagree a lot and all be
wrong in the same direction, in which case spread carries no information about error at
all. So this module computes the spread statistics *and* their relationship to realised
error, and the phase's verdict rests on the second, not the first.

Two deliberate restraints on terminology.

**No aleatoric/epistemic split.** It is tempting to call expert spread "epistemic
uncertainty" and residual scatter "aleatoric uncertainty". That decomposition requires
assumptions about the error-generating process which this data cannot support: nothing
here separates variation in the demand process from variation in what the models have
learned. The conservative vocabulary used throughout is **model disagreement** and
**data variability**, and neither is claimed to be irreducible or reducible.

**Spread is computed in per-unit terms.** Three customers rated 1.4 kW and 388 kW produce
kW spreads differing by two orders of magnitude, so a raw-kW spread would rank a large
customer as the uncertain one purely because it is large. Dividing by the row's own rated
kW makes the statistic a property of the *forecast problem* rather than of the asset's size.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = [
    "DisagreementMeasures",
    "disagreement_measures",
    "spearman",
    "PAIRWISE_MEASURES",
]

#: Which spread statistics are computed, and what each is for. A named set rather than an
#: open-ended pile, so the comparison table cannot grow a column that answers nothing.
PAIRWISE_MEASURES: tuple[str, ...] = (
    "max_minus_min",
    "std",
    "mean_pairwise_range",
    "ensemble_minus_each",
)


def _rank(values: np.ndarray) -> np.ndarray:
    """Average ranks, so ties do not invent an ordering."""
    flat = np.asarray(values, dtype=np.float64).ravel()
    order = np.argsort(flat, kind="mergesort")
    ranks = np.empty(flat.size, dtype=np.float64)
    ranks[order] = np.arange(flat.size, dtype=np.float64)
    unique, inverse, counts = np.unique(flat, return_inverse=True, return_counts=True)
    sums = np.zeros(unique.size, dtype=np.float64)
    np.add.at(sums, inverse, ranks)
    return (sums / counts)[inverse]


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation, as a Pearson correlation of average ranks.

    Used rather than Pearson because the relationship of interest - does larger predicted
    uncertainty come with larger realised error - is monotone rather than linear, and
    because both quantities have heavy right tails where Pearson is dominated by a handful
    of rows.

    Args:
        a: First sample.
        b: Second sample.

    Returns:
        The correlation, or ``nan`` when either sample is constant.

    Raises:
        ValueError: If the samples have different lengths.
    """
    left = np.asarray(a, dtype=np.float64).ravel()
    right = np.asarray(b, dtype=np.float64).ravel()
    if left.size != right.size:
        raise ValueError(
            f"spearman needs aligned samples, got {left.size} and {right.size}"
        )
    if left.size < 2:
        return float("nan")
    lr = _rank(left)
    rr = _rank(right)
    lr = lr - lr.mean()
    rr = rr - rr.mean()
    denominator = float(np.sqrt(np.sum(lr * lr) * np.sum(rr * rr)))
    if denominator <= 1e-15:
        return float("nan")
    return float(np.sum(lr * rr) / denominator)


@dataclass(frozen=True, slots=True)
class DisagreementMeasures:
    """Spread statistics for one method's expert pool, per row and horizon.

    All values are **per-unit** (divided by the row's rated kW) so they are comparable
    across customers of different sizes. ``__getitem__`` returns ``[rows, horizons]``.

    Attributes:
        measures: Statistic name to ``[rows, horizons]`` array.
        expert_names: Expert names, in pool order.
        horizons: Horizon steps.
        point_kw: The ensemble point forecast, for the ``ensemble_minus_each`` measure.
    """

    measures: dict[str, np.ndarray]
    expert_names: tuple[str, ...]
    horizons: tuple[int, ...]
    point_kw: np.ndarray

    def __getitem__(self, name: str) -> np.ndarray:
        if name not in self.measures:
            raise KeyError(
                f"unknown disagreement measure {name!r}; have "
                f"{sorted(self.measures)}"
            )
        return self.measures[name]

    def __contains__(self, name: str) -> bool:
        return name in self.measures

    @property
    def primary(self) -> np.ndarray:
        """The statistic used as an uncertainty scalar unless stated otherwise.

        ``max_minus_min`` - the full range - because it needs no distributional assumption
        and needs no choice of pair to average over. ``std`` is reported alongside because
        a range is driven by one outlier expert, which is a real weakness of the measure
        and should be visible rather than hidden.
        """
        return self.measures["max_minus_min"]

    def describe(self, rows: np.ndarray | None = None) -> dict[str, Any]:
        """Per-measure, per-horizon mean and quantile summary.

        Args:
            rows: Row positions to summarise. Defaults to all.

        Returns:
            ``{measure: {str(horizon): {...}}}``.
        """
        out: dict[str, Any] = {}
        for name, values in self.measures.items():
            per_horizon: dict[str, Any] = {}
            selected = values if rows is None else values[rows]
            for column, horizon in enumerate(self.horizons):
                column_values = selected[:, column]
                per_horizon[str(horizon)] = {
                    "mean": float(column_values.mean()),
                    "median": float(np.median(column_values)),
                    "p90": float(np.quantile(column_values, 0.90)),
                    "p99": float(np.quantile(column_values, 0.99)),
                    "max": float(column_values.max()),
                    "share_zero": float(np.mean(column_values <= 1e-9)),
                }
            out[name] = per_horizon
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "experts": list(self.expert_names),
            "horizons": [int(h) for h in self.horizons],
            "measures": list(self.measures),
            "units": "per unit of the row's rated kW",
            "primary": "max_minus_min",
        }


def disagreement_measures(
    *,
    expert_forecast_kw: np.ndarray,
    row_scale_kw: np.ndarray,
    horizons: tuple[int, ...],
    expert_names: tuple[str, ...],
    point_kw: np.ndarray,
) -> DisagreementMeasures:
    """Compute every spread statistic for an expert pool.

    Args:
        expert_forecast_kw: ``[n_experts, N, horizons]`` in kW.
        row_scale_kw: ``[N]`` rated kW, used to make the statistics per-unit.
        horizons: Horizon steps.
        expert_names: Expert names, in pool order.
        point_kw: ``[N, horizons]`` the ensemble's point forecast.

    Returns:
        The measures.

    Raises:
        ValueError: If shapes disagree, fewer than two experts are supplied, or any rated
            kW is non-positive - the last because the per-unit normalisation would divide
            by zero and silently produce an infinite spread.
    """
    forecasts = np.asarray(expert_forecast_kw, dtype=np.float64)
    scale = np.asarray(row_scale_kw, dtype=np.float64)
    horizons = tuple(int(h) for h in horizons)
    point = np.asarray(point_kw, dtype=np.float64)
    n_experts, n_rows, n_horizons = forecasts.shape
    if n_experts < 2:
        raise ValueError(
            f"disagreement needs at least two experts, got {n_experts}; a single expert "
            f"cannot disagree with anything"
        )
    if scale.shape != (n_rows,):
        raise ValueError(
            f"row_scale_kw must be [rows], got {scale.shape} for {n_rows} rows"
        )
    if point.shape != (n_rows, n_horizons):
        raise ValueError(
            f"point_kw must be [rows, horizons] = {(n_rows, n_horizons)}, got {point.shape}"
        )
    if n_horizons != len(horizons):
        raise ValueError(
            f"{n_horizons} horizon columns for {len(horizons)} horizons"
        )
    if np.any(scale <= 0):
        raise ValueError(
            f"rated kW must be positive; got a minimum of {float(scale.min())}"
        )
    if not np.isfinite(forecasts).all():
        raise ValueError("expert forecasts must be finite to measure disagreement")

    per_unit = forecasts / scale[None, :, None]
    point_per_unit = point / scale[:, None]

    measures: dict[str, np.ndarray] = {
        "max_minus_min": (per_unit.max(axis=0) - per_unit.min(axis=0)),
        "std": per_unit.std(axis=0, ddof=1),
    }
    if n_experts == 2:
        # With two experts the mean pairwise range *is* the full range; recording it
        # separately would be a duplicate column that reads as independent evidence.
        measures["mean_pairwise_range"] = measures["max_minus_min"]
    else:
        ranges = []
        for left in range(n_experts):
            for right in range(left + 1, n_experts):
                ranges.append(np.abs(per_unit[left] - per_unit[right]))
        measures["mean_pairwise_range"] = np.mean(np.stack(ranges, axis=0), axis=0)
    measures["ensemble_minus_each"] = np.abs(per_unit - point_per_unit[None, :, :]).mean(
        axis=0
    )
    measures["disagreement_to_error_scale"] = _disagreement_to_error_scale(
        point_per_unit, per_unit
    )

    for name, values in measures.items():
        if values.shape != (n_rows, n_horizons):
            raise AssertionError(
                f"measure {name!r} produced {values.shape}, expected "
                f"{(n_rows, n_horizons)}"
            )
    return DisagreementMeasures(
        measures=measures,
        expert_names=tuple(expert_names),
        horizons=horizons,
        point_kw=point,
    )


def _disagreement_to_error_scale(
    point_per_unit: np.ndarray, per_unit: np.ndarray
) -> np.ndarray:
    """Spread divided by the ensemble's own per-unit magnitude.

    A row's demand level varies by two orders of magnitude across the 40 customers, so a
    spread of 0.02 per-unit means something quite different at 1 kW and at 100 kW. This is
    the scale-free form, and it is the one a router would actually want: "uncertainty
    relative to the size of what is being forecast".
    """
    magnitude = np.abs(point_per_unit)
    floor = 1e-6
    return (per_unit.max(axis=0) - per_unit.min(axis=0)) / np.maximum(magnitude, floor)