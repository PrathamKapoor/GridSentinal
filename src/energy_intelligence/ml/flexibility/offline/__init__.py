"""Offline experiment machinery for Phase 9.

Kept separate from :mod:`energy_intelligence.ml.flexibility` because it is the only part of
the phase that reads artifacts, writes results and holds an opinion about what the numbers
mean. The estimators themselves stay free of it, so they remain usable from a notebook or a
scheduled job without dragging in the runner.
"""

from __future__ import annotations

from .analysis import (
    coverage_tolerance,
    evaluate_configuration,
    final_comparison_table,
    method_table,
    regime_breakdown,
)
from .experiment import Phase9Result, render_summary, run_phase9
from .verdict import PHASE9_ANSWERS, phase9_verdict

__all__ = [
    "PHASE9_ANSWERS",
    "Phase9Result",
    "coverage_tolerance",
    "evaluate_configuration",
    "final_comparison_table",
    "method_table",
    "phase9_verdict",
    "regime_breakdown",
    "render_summary",
    "run_phase9",
]