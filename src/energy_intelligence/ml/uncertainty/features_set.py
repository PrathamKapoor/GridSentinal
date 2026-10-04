"""Origin-observable features for the learned uncertainty scale.

Phase 8 needs the same inputs Phase 7's router was given, and for the same reasons: a
scale function that reads the future is not an uncertainty estimator, it is a description
of the answer. So this is a thin wrapper over Phase 7's feature builders rather than a
second implementation, and it inherits their causality proof unchanged:

* :func:`~energy_intelligence.ml.router.features.build_router_features` - level,
  trajectory, volatility, calendar, scale;
* :func:`~energy_intelligence.ml.router.features.append_lagged_error_features` - each
  expert's lagged realised error, delayed by ``h + 1`` steps;
* :func:`~energy_intelligence.ml.router.features.assert_router_features_are_causal` - the
  poisoning test, run here on every experiment rather than only on Phase 7's router.

The wrapper exists for three Phase 8-specific reasons.

**The lagged-error group is optional.** It is the strongest single group for Phase 7's
router, and Phase 8 needs to know how much of its own result comes from it, so
``include_past_error=False`` must be able to drop it.

**The imputation reference changes.** Phase 7 filled gaps from its router-training rows.
Phase 8 fills from ``CAL_FIT``, which is a different and strictly earlier set. Filling
from the whole panel would let a test row's error rate choose the value a test row is
scored against.

**The causality check runs on every experiment.** Phase 7 ran it once as part of building
its feature matrix. Phase 8 runs it again, because the feature set is reused rather than
rebuilt, and a reused component's invariants are not automatically preserved.

The regime labels of Phases 4 and 5 are **not** features. They are computed from the demand
being predicted, so using them as inputs would be reading the target. They appear in this
phase's evaluation and nowhere else.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..router.features import (
    RouterFeatureSpec,
    append_lagged_error_features,
    assert_router_features_are_causal,
    build_router_features,
)

__all__ = ["build_uncertainty_features"]


def build_uncertainty_features(
    *,
    values: np.ndarray,
    panel: Any,
    forecasts: np.ndarray,
    actual_kw: np.ndarray,
    horizons: tuple[int, ...],
    include_past_error: bool = True,
    reference_rows: np.ndarray,
    sample: int = 32,
    seed: int = 0,
    log: Any = None,
) -> tuple[np.ndarray, tuple[str, ...], dict[str, Any]]:
    """Build, check and impute the uncertainty scale's inputs.

    Args:
        values: ``[n_series, n_steps]`` per-unit series.
        panel: The evaluation panel.
        forecasts: ``[n_experts, N, horizons]`` in kW, for the lagged-error group.
        actual_kw: ``[N, horizons]`` in kW, used only by the causality check's
            poisoned-target recomputation.
        horizons: Horizon steps.
        include_past_error: Whether to include the lagged expert errors.
        reference_rows: Positions whose observed values define the imputation, and which
            the leakage guard re-verifies. Must be the calibration-fitting rows.
        sample: How many rows the poisoning test checks.
        seed: Seed for the deterministic test sample.
        log: Optional progress callable.

    Returns:
        ``(matrix, names, causality_record)``.

    Raises:
        ValueError: If the causality check fails, or a column cannot be imputed.
    """
    base = build_router_features(values=np.asarray(values, dtype=np.float32), panel=panel)
    if include_past_error:
        built = append_lagged_error_features(
            features=base,
            forecasts=np.asarray(forecasts),
            actual_kw=np.asarray(actual_kw),
            panel=panel,
            horizons=tuple(horizons),
        )
    else:
        built = base
    spec = RouterFeatureSpec(include_past_error=bool(include_past_error))
    selected = built.select(spec)
    missing = selected.missing_counts

    # The guard runs on the *selected* matrix, because that is the matrix that will be
    # fitted. Checking the full set and then dropping columns would leave the dropped ones
    # unverified for no benefit.
    record = assert_router_features_are_causal(
        values=np.asarray(values, dtype=np.float32),
        panel=panel,
        features=selected,
        horizons=tuple(horizons) if include_past_error else None,
        forecasts=np.asarray(forecasts) if include_past_error else None,
        sample=int(sample),
        seed=int(seed),
    )
    filled = selected.fill_missing(np.asarray(reference_rows, dtype=np.int64))
    if not np.isfinite(filled.matrix).all():
        raise ValueError("uncertainty features contain non-finite values after imputation")
    if log is not None:
        log(
            f"  uncertainty features: {filled.n_features} columns, causality passed on "
            f"{record['checked_rows_verified']} rows, {len(missing)} imputed"
        )
    return filled.matrix, filled.names, record