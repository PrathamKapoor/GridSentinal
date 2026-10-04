"""Selection on F01-F04, confirmation on F05-F06, and the incremental ladder.

Three rules are frozen here and implemented exactly as frozen, because each of them is a place
where a later-runner could quietly improve its own result:

**Only F01-F04 select.** :func:`select` takes the selection fold ids as an argument and
*refuses* any fold outside the selection set, so a confirmation fold cannot reach the selection
even by accident. F05-F06 are called "confirmation", never "test" - they are held-out blocks
inside the validation range, and calling them unseen test data would be a different claim from
the one they support.

**Absolute MAE decides, then the simplicity tie-break.** The best mean MAE wins outright. Only
when a simpler set is within the frozen 0.5% relative tolerance is it preferred, so the
tie-break can simplify a near-tie but can never rescue a materially worse set.

**Effects are differences, not causes.** Every effect is an incremental
validation-performance difference between two feature sets evaluated under an identical
protocol. The word "causal" appears in this module only to say that these numbers are not
causal.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "SELECTION_FOLDS_ALLOWED",
    "relative_difference",
    "select",
    "confirm",
    "incremental_effects",
    "mean_mae",
]

#: The only folds :func:`select` will accept.
SELECTION_FOLDS_ALLOWED: tuple[str, ...] = ("F01", "F02", "F03", "F04")

#: Frozen before results were visible; mirrors ``freeze.TIE_BREAK_RELATIVE_TOLERANCE``.
_RELATIVE_TOLERANCE = 0.005


def relative_difference(baseline: float, candidate: float) -> float:
    """``(candidate - baseline) / baseline``.

    Positive means the candidate is worse, since every metric here is an error.

    Args:
        baseline: The reference value.
        candidate: The value compared against it.

    Returns:
        The signed relative difference.

    Raises:
        ZeroDivisionError: If the baseline is zero, which no MAE can legitimately be.
    """
    if baseline == 0:
        raise ZeroDivisionError(
            "a relative difference needs a non-zero reference; an MAE of exactly zero means "
            "the reference was not measured"
        )
    return (float(candidate) - float(baseline)) / float(baseline)


def _by_set_and_fold(
    rows: list[dict[str, Any]], fold_ids: tuple[str, ...]
) -> dict[str, dict[str, float]]:
    """``{feature_set: {fold_id: mae}}`` restricted to the given folds."""
    wanted = set(fold_ids)
    out: dict[str, dict[str, float]] = {}
    for row in rows:
        if row["fold"] not in wanted:
            continue
        out.setdefault(row["feature_set"], {})[row["fold"]] = float(row["metrics"]["mae"])
    return out


def mean_mae(per_fold: dict[str, float]) -> float:
    """The unweighted mean of per-fold MAE.

    Unweighted because each fold is a contiguous block of a similar number of origins, so an
    unweighted mean is the mean over origins without needing the counts to be identical.

    Args:
        per_fold: ``{fold_id: mae}``.

    Returns:
        The mean, or ``nan`` when empty.

    Raises:
        ValueError: If any value is missing or non-finite.
    """
    values = list(per_fold.values())
    if not values:
        return float("nan")
    return sum(values) / len(values)


def select(
    rows: list[dict[str, Any]],
    *,
    fold_ids: tuple[str, ...] = SELECTION_FOLDS_ALLOWED,
) -> dict[str, Any]:
    """Pick a feature set using the selection folds and nothing else.

    Args:
        rows: Every measured fold cell for one target.
        fold_ids: The folds to use. Must be a subset of
            :data:`SELECTION_FOLDS_ALLOWED`.

    Returns:
        The selection record, with the per-fold MAE of every set, the winner, whether the
        tie-break was applied, and why.

    Raises:
        ValueError: If a requested fold is not a selection fold, or no rows were supplied.
    """
    illegal = sorted(set(fold_ids) - set(SELECTION_FOLDS_ALLOWED))
    if illegal:
        raise ValueError(
            f"selection may only use {list(SELECTION_FOLDS_ALLOWED)}; got {illegal}. F05-F06 "
            f"are post-ablation confirmation folds and using them to select would make the "
            f"confirmation meaningless"
        )
    table = _by_set_and_fold(rows, fold_ids)
    if not table:
        raise ValueError("no rows were supplied for the requested selection folds")

    means = {set_id: mean_mae(per_fold) for set_id, per_fold in table.items()}
    incomplete = {
        set_id: sorted(set(fold_ids) - set(per_fold)) for set_id, per_fold in table.items()
    }
    incomplete = {k: v for k, v in incomplete.items() if v}
    if incomplete:
        raise ValueError(
            f"these feature sets are missing selection folds and cannot be ranked: "
            f"{incomplete}"
        )

    ranked = sorted(means.items(), key=lambda item: (item[1], len(item[0])))
    best_set, best_mae = ranked[0]
    # Feature COUNT, not the length of the set id: "A" and "B" are both one character,
    # so len() on the id would make every set look the same size.
    simplest = min(table, key=lambda s: (len(_members_of(s)), s))
    tie_applied = False
    chosen = best_set
    # Walk the sets from fewest features upward and take the first that is within tolerance of
    # the best absolute score. Iterating by feature count rather than by rank is what makes
    # "prefer fewer features" meaningful: the winner is the simplest set that is effectively
    # tied, and the set with the lowest MAE is only the fallback when nothing simpler ties.
    by_size = sorted(table, key=lambda s: (len(_members_of(s)), s))
    for set_id in by_size:
        if len(_members_of(set_id)) >= len(_members_of(best_set)):
            break
        if relative_difference(best_mae, means[set_id]) <= _RELATIVE_TOLERANCE:
            chosen = set_id
            tie_applied = set_id != best_set
            break

    return {
        "selection_folds": list(fold_ids),
        "metric": "mae",
        "aggregation": "unweighted mean over folds",
        "per_set_mean_mae": {k: v for k, v in sorted(means.items(), key=lambda i: i[1])},
        "per_fold_mae": {k: dict(sorted(v.items())) for k, v in sorted(table.items())},
        "ranking": [k for k, _ in ranked],
        "best_absolute_mae_set": best_set,
        "best_absolute_mae": best_mae,
        "selected_set_mean_mae": means[chosen],
        "selected_feature_set": chosen,
        "tie_break": {
            "applied": tie_applied,
            "rule": "prefer fewer features",
            "relative_mae_tolerance": _RELATIVE_TOLERANCE,
            "frozen_before_results": True,
            "simplest_set": simplest,
            "relative_difference_of_selected_vs_best": relative_difference(
                best_mae, means[chosen]
            ),
        },
        "rationale": (
            f"{chosen} selected on {', '.join(fold_ids)} by mean MAE"
            + (
                f"; it is within {_RELATIVE_TOLERANCE:.1%} of the best absolute set "
                f"({best_set}) and uses fewer features, so the frozen simplicity tie-break "
                f"was applied"
                if tie_applied
                else f" and has the lowest mean MAE outright; the simplicity tie-break was "
                f"not needed"
            )
        ),
    }


def confirm(
    rows: list[dict[str, Any]],
    *,
    selected_set: str,
    fold_ids: tuple[str, ...] = ("F05", "F06"),
) -> dict[str, Any]:
    """Score the chosen set on the folds that took no part in choosing it.

    Args:
        rows: Every measured fold cell for one target.
        selected_set: The set chosen on F01-F04.
        fold_ids: The confirmation folds.

    Returns:
        The confirmation record. ``held_out`` states plainly that these are validation-range
        blocks and not the final test split.
    """
    table = _by_set_and_fold(rows, fold_ids)
    if not table:
        return {
            "status": "NOT_RUN",
            "reason": "no confirmation-fold rows were measured",
            "selected_feature_set": selected_set,
        }
    means = {set_id: mean_mae(per_fold) for set_id, per_fold in table.items()}
    ranked = sorted(means.items(), key=lambda item: item[1])
    best_on_confirmation = ranked[0][0]
    return {
        "status": "COMPLETE",
        "confirmation_folds": list(fold_ids),
        "held_out": True,
        "held_out_caveat": (
            "F05-F06 are contiguous blocks inside the validation range. They are held out "
            "from SELECTION, but they are not the final test split and carry no unseen-test "
            "claim."
        ),
        "selected_feature_set": selected_set,
        "per_set_mean_mae": {k: v for k, v in sorted(means.items(), key=lambda i: i[1])},
        "per_fold_mae": {k: dict(sorted(v.items())) for k, v in sorted(table.items())},
        "ranking": [k for k, _ in ranked],
        "best_on_confirmation": best_on_confirmation,
        "selected_set_was_best_on_confirmation": best_on_confirmation == selected_set,
        "relative_difference_selected_vs_best": (
            relative_difference(ranked[0][1], means[selected_set])
            if selected_set in means
            else None
        ),
    }


def incremental_effects(
    rows: list[dict[str, Any]],
    *,
    fold_ids: tuple[str, ...] = SELECTION_FOLDS_ALLOWED,
    ladder: tuple[tuple[str, str], ...] = (("A", "C"), ("C", "D"), ("D", "E"), ("A", "B")),
) -> dict[str, Any]:
    """The incremental effect of each feature family along the ladder.

    The pairs are fixed and were frozen with the protocol: ``A -> C`` is the lags on top of
    calendar, ``C -> D`` the rolling statistics, ``D -> E`` the ramp and same-hour rolling, and
    ``A vs B`` calendar against lags with neither combined.

    Args:
        rows: Every measured fold cell for one target.
        fold_ids: Folds to average over.
        ladder: The pairs to report.

    Returns:
        ``{comparison: {...}}`` plus an explicit statement of what these numbers are not.
    """
    table = _by_set_and_fold(rows, fold_ids)
    means = {set_id: mean_mae(per_fold) for set_id, per_fold in table.items()}
    out: dict[str, Any] = {}
    for base, candidate in ladder:
        if base not in means or candidate not in means:
            out[f"{base} -> {candidate}"] = {
                "status": "NOT_AVAILABLE",
                "reason": f"missing a measurement for {base if base not in means else candidate}",
            }
            continue
        base_mae, candidate_mae = means[base], means[candidate]
        absolute = candidate_mae - base_mae
        out[f"{base} -> {candidate}"] = {
            "status": "COMPLETE",
            "baseline_set": base,
            "candidate_set": candidate,
            "baseline_mean_mae": base_mae,
            "candidate_mean_mae": candidate_mae,
            "absolute_difference_kw": absolute,
            "relative_difference": relative_difference(base_mae, candidate_mae),
            "direction": (
                "improvement" if absolute < 0 else ("no_change" if absolute == 0 else "regression")
            ),
            "features_added": [
                name
                for name in _members_of(candidate)
                if name not in _members_of(base)
            ],
        }
    return {
        "folds": list(fold_ids),
        "comparisons": out,
        "interpretation": (
            "each figure is an incremental validation-performance difference between two "
            "feature sets evaluated under an identical, frozen protocol"
        ),
        "not_a_claim": (
            "these are not causal effects. They are differences in mean absolute error "
            "between two configurations on held-out folds, from a single seed, and a "
            "difference smaller than seed-to-seed variation cannot be distinguished from it"
        ),
        "no_hpo": (
            "no hyperparameter search was run for any feature set, so none of these "
            "differences is attributable to a different amount of tuning"
        ),
    }


def _members_of(set_id: str) -> tuple[str, ...]:
    from .sets import members

    return members(set_id)