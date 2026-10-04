"""Phase 9: uncertainty-aware flexibility estimation.

The layer between Phase 8's calibrated forecast and a future optimiser. It answers one
question:

    How much response capability can reasonably be associated with the current energy
    state, over what horizon, in which direction, and on what evidence?

What is deliberately absent
    No optimizer, no MILP, no MPC, no dispatch, no reinforcement learning, and no import of
    :mod:`energy_intelligence.domain.actions`. Phase 10 consumes what this produces; this
    package must not know a consumer exists, or the two phases will drift into each other.

The distinction everything else serves
    ----------------------------------
    * **Physical** flexibility needs authoritative capability metadata *and* a control
      authority the system actually holds. On SMART-DS, :mod:`capability` establishes that
      no dimension has either, so every physical figure is ``UNKNOWN``.
    * **Statistical proxy** flexibility (:mod:`envelope`) says how far demand has
      *historically moved* from its own expected profile. It carries
      ``AuthorityLevel.NOT_CONTROLLABLE`` by construction, and the domain object refuses to
      let one claim otherwise.
    * **Assumed** flexibility (:mod:`scenarios`) is declared for demonstration, tagged, and
      structurally barred from a real-data result.

Methodology, inherited rather than reinvented
    ``CAL_FIT`` fits the baseline and the envelope, ``CAL_CONF`` selects between baseline
    and granularity, and the sealed test split is read once. That is Phase 8's partition and
    Phase 7's fit/select split, so the three phases are comparable.
"""

from __future__ import annotations

from .aggregation import AggregationResult, aggregate_deviations, pairwise_correlation
from .baselines import BASELINE_METHODS, BaselineFit, evaluate_baseline, fit_baseline
from .capability import CAPABILITY_DIMENSIONS, CapabilityAudit, audit_capabilities
from .demand import (
    SLOT_GRANULARITIES,
    DemandField,
    demand_series_kw,
    verify_demand_reconstruction,
)
from .envelope import EnvelopeFit, evaluate_envelope, fit_envelope
from .evaluation import directional_balance, evaluate_band, stability_report
from .reliance import RelianceFactor, apply_reliance, justification, reliance_from_widths
from .scenarios import (
    REAL_DATA_MODE,
    SCENARIO_MODE,
    ScenarioAssumptions,
    ScenarioViolation,
    assumed_flexibility,
    assert_real_data_mode,
    load_scenario,
)

__all__ = [
    "AggregationResult",
    "BASELINE_METHODS",
    "BaselineFit",
    "CAPABILITY_DIMENSIONS",
    "CapabilityAudit",
    "DemandField",
    "EnvelopeFit",
    "REAL_DATA_MODE",
    "RelianceFactor",
    "SCENARIO_MODE",
    "SLOT_GRANULARITIES",
    "ScenarioAssumptions",
    "ScenarioViolation",
    "aggregate_deviations",
    "apply_reliance",
    "assumed_flexibility",
    "assert_real_data_mode",
    "audit_capabilities",
    "demand_series_kw",
    "directional_balance",
    "evaluate_band",
    "evaluate_baseline",
    "evaluate_envelope",
    "fit_baseline",
    "fit_envelope",
    "justification",
    "load_scenario",
    "pairwise_correlation",
    "reliance_from_widths",
    "stability_report",
    "verify_demand_reconstruction",
]