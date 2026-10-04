"""Is the router's training signal real, or an artefact of in-sample expert accuracy?

The concern is specific and it is serious. A routing label says "expert *e* was the most
accurate for this row". If the predictions behind that label came from a model that had
already fitted the row's own target, that expert looks far better than it will ever be
at inference, the label is noise, and a router trained on those labels learns a world
that does not exist. Nothing in the loss would look wrong.

Phase 7 sidesteps the trap **structurally** rather than statistically: the learned
experts are fitted on the training split, and the router is trained on validation rows
whose expert predictions are out-of-sample. That is the arrangement in force.

This module measures how much the trap would have mattered, because "we avoided it" is
a claim and the interesting number is how big the avoided effect is. It splits the
validation block in half **in time**, fits the GBM on the first half only, and then
computes the oracle gain over persistence twice on the same model:

``in_sample``
    Rows the GBM was fitted on.
``out_of_sample``
    Rows it never saw.

If the in-sample oracle gain is much larger, that excess is the illusion a router would
have chased had the labels come from in-sample expert predictions.

Two deliberate simplifications, both stated in the returned record. Persistence is the
second expert because it needs no fitting and is exactly reproducible, so the whole
measurement turns on the GBM's honesty rather than on two models drifting together. And
the GBM is fitted on the thirteen Phase 4 features without the per-series context
columns, which does not affect the in-sample/out-of-sample contrast - the context
columns are constant statistics of the training split and cannot make a model
better on its own fitting rows.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["gbm_label_optimism", "label_quality"]


def label_quality(
    *,
    assignment: np.ndarray,
    errors: np.ndarray,
    expert_names: tuple[str, ...],
) -> dict[str, Any]:
    """How informative a set of routing labels is, independent of its accuracy.

    A label set where one expert wins 95% of rows carries almost no decision, however
    correct the labels are. The margin between the winner and the runner-up says how
    close the calls were, which is what decides whether a router can learn them at all.

    Args:
        assignment: ``[N, horizons]`` oracle expert index.
        errors: ``[n_experts, N, horizons]`` absolute errors in kW.
        expert_names: Expert names, in order.

    Returns:
        Per-horizon label share, decision margin, and the chosen expert's MAE.
    """
    assignment = np.asarray(assignment, dtype=np.int64)
    errors = np.asarray(errors, dtype=np.float64)
    out: dict[str, Any] = {}
    for column in range(assignment.shape[1]):
        counts = np.bincount(assignment[:, column], minlength=len(expert_names))
        share = counts / max(1, counts.sum())
        per_row = np.sort(errors[:, :, column], axis=0)
        margin = (per_row[1] - per_row[0]) / np.maximum(per_row[0], 1e-12)
        out[str(column)] = {
            "label_share": {
                name: float(share[index]) for index, name in enumerate(expert_names)
            },
            "dominant_label_share": float(share.max()),
            "mean_margin_pct": float(100.0 * margin.mean()),
            "share_margin_below_5pct": float(np.mean(margin < 0.05)),
            "chosen_expert_mae_kw": float(per_row[0].mean()),
        }
    return out


def _gain(block: np.ndarray, actual: np.ndarray, horizons: tuple[int, ...]) -> dict[str, Any]:
    """Oracle gain over the best single expert for a ``[2, rows, horizons]`` block."""
    from .oracle import oracle_assignment

    errors = np.abs(block - actual[None, :, :])
    picks = oracle_assignment(block, actual)
    oracle_mae = np.take_along_axis(errors, picks[None, :, :], axis=0)[0].mean(axis=0)
    per_expert = errors.mean(axis=1)
    best = per_expert.min(axis=0)
    return {
        "per_expert_mae_kw": per_expert.tolist(),
        "best_single_mae_kw": best.tolist(),
        "oracle_mae_kw": oracle_mae.tolist(),
        "oracle_gain_pct": [
            float(100.0 * (best[c] - oracle_mae[c]) / best[c]) if best[c] > 0 else None
            for c in range(len(horizons))
        ],
        "oracle_choice_share": [
            (np.bincount(picks[:, c], minlength=2) / max(1, picks.shape[0])).tolist()
            for c in range(len(horizons))
        ],
    }


def gbm_label_optimism(
    *,
    values: np.ndarray,
    panel: Any,
    horizons: tuple[int, ...],
    seed: int,
    split: str = "validation",
    fit_fraction: float = 0.5,
    log: Any = None,
) -> dict[str, Any]:
    """Measure how much in-sample expert accuracy inflates the oracle gain.

    Args:
        values: ``[n_series, n_steps]`` per-unit values.
        panel: The evaluation panel.
        horizons: Horizon steps.
        seed: Estimator seed, the same one the real GBM used.
        split: Which panel split to halve. Must be a split the experts never fitted on.
        fit_fraction: Share of that split, chronologically, used to fit the diagnostic
            GBM.
        log: Optional progress callable.

    Returns:
        A record with the in-sample and out-of-sample oracle gains per horizon and the
        ratio between them.

    Raises:
        ValueError: If the split is too small to halve.
    """
    from ...baselines.classical import fit_classical
    from ...dataset import build_feature_matrix
    from ...scaling import FeatureScaler
    from ..experts import PHASE4_FEATURE_NAMES

    origins = np.asarray(panel.origins, dtype=np.int64)
    series = np.asarray(panel.series, dtype=np.int64)
    actual = np.asarray(panel.actual_kw, dtype=np.float64)
    persistence = np.asarray(panel.persistence_kw, dtype=np.float64)
    scale = np.asarray(panel.row_scale, dtype=np.float64)
    horizons = tuple(int(h) for h in horizons)

    mask = np.asarray(panel.split) == split
    positions = np.flatnonzero(mask)
    order = np.argsort(origins[positions], kind="mergesort")
    positions = positions[order]
    if positions.size < 4:
        raise ValueError(f"split {split!r} has only {positions.size} rows to halve")
    cut = int(round(positions.size * float(fit_fraction)))
    fit_rows = positions[:cut]
    held_rows = positions[cut:]

    lookback = int(origins[positions].min()) - 96
    lookback = max(lookback, 1)
    features_fit = build_feature_matrix(
        values,
        origins[fit_rows],
        lookback=lookback,
        feature_names=PHASE4_FEATURE_NAMES,
        series_index=series[fit_rows],
    ).astype(np.float64)
    features_held = build_feature_matrix(
        values,
        origins[held_rows],
        lookback=lookback,
        feature_names=PHASE4_FEATURE_NAMES,
        series_index=series[held_rows],
    ).astype(np.float64)
    scaler = FeatureScaler.fit(features_fit)
    features_fit = scaler.transform(features_fit)
    features_held = scaler.transform(features_held)

    target_fit = (actual[fit_rows] / scale[fit_rows, None]).astype(np.float64)
    target_held = (actual[held_rows] / scale[held_rows, None]).astype(np.float64)

    gbm_fit = np.empty_like(actual[fit_rows])
    gbm_held = np.empty_like(actual[held_rows])
    for column, horizon in enumerate(horizons):
        fitted = fit_classical(
            "classical_hist_gbm",
            features_fit,
            target_fit[:, column],
            feature_names=PHASE4_FEATURE_NAMES,
            seed=seed,
        )
        gbm_fit[:, column] = fitted.predict(features_fit) * scale[fit_rows]
        gbm_held[:, column] = fitted.predict(features_held) * scale[held_rows]
        if log is not None:
            log(f"    diagnostic gbm h={horizon} fitted in {fitted.fit_seconds:.1f}s")

    in_sample = _gain(
        np.stack([persistence[fit_rows], gbm_fit]), actual[fit_rows], horizons
    )
    out_of_sample = _gain(
        np.stack([persistence[held_rows], gbm_held]), actual[held_rows], horizons
    )
    errors_fit = np.abs(np.stack([persistence[fit_rows], gbm_fit]) - actual[fit_rows][None, :, :])
    errors_held = np.abs(np.stack([persistence[held_rows], gbm_held]) - actual[held_rows][None, :, :])
    labels_fit = label_quality(
        assignment=np.argmin(errors_fit, axis=0),
        errors=errors_fit,
        expert_names=("persistence", "classical_hist_gbm"),
    )
    labels_held = label_quality(
        assignment=np.argmin(errors_held, axis=0),
        errors=errors_held,
        expert_names=("persistence", "classical_hist_gbm"),
    )

    ratios = []
    for column in range(len(horizons)):
        inside = in_sample["oracle_gain_pct"][column]
        outside = out_of_sample["oracle_gain_pct"][column]
        ratios.append(
            float(inside / outside) if inside is not None and outside not in (None, 0.0) else None
        )
    return {
        "question": (
            "would router labels built from in-sample expert predictions have been "
            "worth more than the ones actually used?"
        ),
        "method": (
            f"the validation block was halved in time; one GBM was fitted on the first "
            f"{fit_fraction:.0%} and scored on both halves, against persistence"
        ),
        "rows_fitted_on": int(fit_rows.size),
        "rows_never_seen": int(held_rows.size),
        "note": (
            "uses the thirteen Phase 4 features without the per-series context columns, "
            "which cannot change the in-sample versus out-of-sample contrast"
        ),
        "in_sample": in_sample,
        "out_of_sample": out_of_sample,
        "gain_ratio_in_sample_over_out_of_sample": ratios,
        "label_quality_in_sample": labels_fit,
        "label_quality_out_of_sample": labels_held,
        "conclusion": (
            "an in-sample oracle gain far above the out-of-sample one means router "
            "labels from in-sample expert predictions would overstate the available "
            "headroom; the phase's labels come from out-of-sample predictions"
        ),
    }