"""The phase's answer, derived from the measurements rather than asserted.

The question has five parts, and on SMART-DS they come out very differently, so they are
answered separately and then combined:

1. **Is any flexibility physically supported by this data?** - audited per dimension from the
   dataset's own metadata. On SMART-DS the answer is no for every dimension, and that is a
   *result*, not a failure of the phase.
2. **Can a behavioural envelope be estimated at all?** - measured by whether it holds its
   stated coverage on rows it has never seen.
3. **Does it distinguish the two directions?** - measured by per-direction conditional
   coverage. A band that covers overall while missing every downward excursion is useless to
   anyone trying to shed load.
4. **Is it stable enough to consume?** - measured by the lag-1 autocorrelation of its own
   width.
5. **Is predictive uncertainty informative about flexibility?** - measured, not assumed.

The phase verdict is the **weakest** component, which on this dataset means a behavioural
envelope is reportable while no physical capability is claimed. Reporting that honestly is
the correct outcome; upgrading it by inferring capability from historical variation would be
the failure this module exists to prevent.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["PHASE9_ANSWERS", "phase9_verdict"]

#: The answers, in the vocabulary the phase is allowed to use.
PHASE9_ANSWERS: tuple[str, ...] = ("YES", "PARTIALLY", "NO", "INSUFFICIENT_DATA")


def _worst(*answers: str) -> str:
    """The weakest of the component answers."""
    for candidate in PHASE9_ANSWERS[::-1]:
        if candidate in answers:
            return candidate
    return "INSUFFICIENT_DATA"


def phase9_verdict(
    *,
    capability: dict[str, Any],
    per_horizon: dict[str, Any],
    selection: dict[str, Any],
    reliance: dict[str, Any],
    aggregation: dict[str, Any],
    nominal_level: float,
    tolerance: float,
) -> dict[str, Any]:
    """Assemble the phase's verdict from its measurements.

    Args:
        capability: The capability audit's ``to_dict()``.
        per_horizon: ``{horizon: test evaluation}``.
        selection: The configuration chosen on the conformity split.
        reliance: The reliance-coupling result.
        aggregation: The aggregation result.
        nominal_level: The headline level.
        tolerance: Coverage tolerance.

    Returns:
        The verdict record, with each component's answer and the criterion it was judged by.
    """
    # Every record is keyed by the string form of the horizon, matching the rest of the
    # phase's JSON. Normalised once here so the loops below can iterate on integers without
    # each one repeating the conversion.
    by_horizon = {str(key): value for key, value in per_horizon.items()}
    horizons = tuple(sorted(int(h) for h in by_horizon))
    n_rows = max((int(by_horizon[str(h)]["n"]) for h in horizons), default=0)

    # ---- 1. Physical support, straight from the audit ----------------------
    supported = [r for r in capability["records"] if r["basis"] != "unknown"]
    physical_answer = "YES" if capability["physically_supported"] else "NO"

    # ---- 2. Can it be estimated, and does it hold on unseen data? -----------
    calibrated = {
        str(h): bool(by_horizon[str(h)]["passes"]) for h in horizons
    }
    all_calibrated = bool(calibrated) and all(calibrated.values())
    estimate_answer = "YES" if all_calibrated else "PARTIALLY"

    # ---- 3. Directionality --------------------------------------------------
    directional: dict[str, Any] = {}
    directional_ok = bool(horizons)
    for h in horizons:
        entry = by_horizon[str(h)]
        up = entry["upward_conditional_coverage"]
        down = entry["downward_conditional_coverage"]
        balance = entry["directional"]
        directional[str(h)] = {
            "upward_conditional_coverage": up,
            "downward_conditional_coverage": down,
            "upward_excursion_share": balance.get("upward_excursion_share"),
            "downward_excursion_share": balance.get("downward_excursion_share"),
            "mean_upward_to_downward_ratio": balance.get("mean_upward_to_downward_ratio"),
        }
        # Each direction must clear the tolerance on its own, otherwise the envelope is
        # calibrated only because one direction is absorbing the loss.
        if up is None or down is None:
            directional_ok = False
        elif (
            abs(up - float(nominal_level)) > float(tolerance)
            or abs(down - float(nominal_level)) > float(tolerance)
        ):
            directional_ok = False
    direction_answer = "YES" if directional_ok else "PARTIALLY"

    # ---- 4. Stability -------------------------------------------------------
    stability: dict[str, Any] = {}
    smooth = bool(horizons)
    for h in horizons:
        report = by_horizon[str(h)].get("stability") or {}
        lag1 = report.get("width_lag1_autocorrelation")
        stability[str(h)] = {
            "width_lag1_autocorrelation": lag1,
            "width_is_constant_within_series": report.get("width_is_constant_within_series"),
            "width_change_vs_error_change_spearman": report.get(
                "width_change_vs_error_change_spearman"
            ),
            "share_width_changed": report.get("share_width_changed"),
            "mean_absolute_width_change_kw": report.get("mean_absolute_width_change_kw"),
        }
        # A width that jumps every step is not consumable downstream. The bar is a lag-1
        # autocorrelation above 0.5: the envelope should be smoother than the demand it
        # bounds, which is a weak requirement that still rules out per-step jitter.
        if lag1 is None or lag1 <= 0.5:
            smooth = False
    stability_answer = "YES" if smooth else "PARTIALLY"

    # ---- 5. Is uncertainty informative here? --------------------------------
    judgements = reliance.get("justification", {})
    any_supported = bool(
        any(bool(j.get("coupling_supported")) for j in judgements.values())
    )
    rhos = [
        j.get("spearman_width_vs_absolute_deviation")
        for j in judgements.values()
        if j.get("spearman_width_vs_absolute_deviation") is not None
    ]
    mean_rho = float(np.mean(rhos)) if rhos else None
    coupling_answer = (
        "YES"
        if any_supported
        else ("NO" if judgements else "INSUFFICIENT_DATA")
    )

    aggregation_answer = "NO"

    components = {
        "physical_support": {
            "answer": physical_answer,
            "criterion": (
                "at least one flexibility dimension carries evidence of a physical limit "
                "and a control interface in the dataset's own metadata"
            ),
            "dimensions_audited": capability["dimensions_audited"],
            "physically_supported": capability["physically_supported"],
            "unknown": capability["unknown"],
            "supported_dimensions": [r["dimension"] for r in supported],
            "consequence": (
                "no physical flexibility may be published, so every estimate is "
                "STATISTICAL_PROXY with NOT_CONTROLLABLE authority"
            ),
        },
        "can_be_estimated": {
            "answer": estimate_answer,
            "criterion": (
                f"coverage within {tolerance:.4f} absolute of {nominal_level:.0%} at every "
                f"horizon on the sealed test split, having been fitted on CAL_FIT only"
            ),
            "per_horizon_pass": calibrated,
            "all_horizons_calibrated": all_calibrated,
            "rows_evaluated_per_horizon": n_rows,
            "configuration": {
                "baseline": selection["baseline"],
                "granularity": selection["granularity"],
                "selected_on": "the conformity split, never the test split",
                "criterion": selection["criterion"],
            },
        },
        "is_directional": {
            "answer": direction_answer,
            "criterion": (
                "upward-only and downward-only conditional coverage each within the "
                "tolerance; a band calibrated only by one direction absorbing the other's "
                "loss is not directional"
            ),
            "per_horizon": directional,
        },
        "is_stable": {
            "answer": stability_answer,
            "criterion": (
                "lag-1 autocorrelation of the envelope's own width above 0.5 at every "
                "horizon, so a downstream planner sees a smooth claim rather than per-step "
                "jitter"
            ),
            "per_horizon": stability,
        },
        "uncertainty_is_informative": {
            "answer": coupling_answer,
            "criterion": (
                "Spearman correlation between Phase 8's normalised interval width and "
                "realised absolute deviation strictly positive at a horizon, on the sealed "
                "test split"
            ),
            "mean_spearman": mean_rho,
            "applied": bool(reliance.get("applied")),
            "per_horizon": {
                str(h): {
                    "spearman_width_vs_absolute_deviation": j.get(
                        "spearman_width_vs_absolute_deviation"
                    ),
                    "coupling_supported": j.get("coupling_supported"),
                    "n_rows": j.get("n_rows"),
                    "threshold": j.get("threshold"),
                }
                for h, j in judgements.items()
            },
            "consequence": (
                "the coupling was measured and then applied, or withheld"
                if any_supported
                else "the coupling was measured and withheld; the published envelope is the "
                "raw behavioural one"
            ),
        },
        "aggregation_creates_capability": {
            "answer": aggregation_answer,
            "criterion": (
                "not a measurement: aggregating historical variation cannot create the "
                "ability to command anything. The measured diversification ratio quantifies "
                "how much of the naive sum survives aggregation, and the correlation that "
                "explains it, but the answer is NO by construction"
            ),
            "measured_diversification_upward": {
                str(h): aggregation["by_horizon"][str(h)]["diversification_upward"]
                for h in horizons
            },
            "mean_pairwise_correlation": {
                str(h): aggregation["by_horizon"][str(h)]["correlation"]["mean"]
                for h in horizons
            },
        },
    }

    component_answers = [c["answer"] for c in components.values()]
    answer = _worst(*component_answers)

    return {
        "question": (
            "can a flexibility envelope be estimated from this data, is it usable and "
            "directional, and is any flexibility physically supported?"
        ),
        "answer": answer,
        "coverage_tolerance": float(tolerance),
        "headline_nominal_level": float(nominal_level),
        "components": components,
        "component_answers": dict(zip(components, component_answers)),
        "reading": (
            "a behavioural envelope was estimated, evaluated once on held-out data, and "
            "published as STATISTICAL_PROXY / NOT_CONTROLLABLE. No physical flexibility is "
            "supported by this dataset, so no dispatchable capability is claimed and Phase 10 "
            "must obtain physical limits from an external source before it can plan against "
            "anything."
            if answer in {"PARTIALLY", "NO"}
            else "every component passed"
        ),
        "basis": (
            "each component is answered from its own measurement and the phase verdict is "
            "the weakest component, so no strong claim is carried by a weak one"
        ),
    }