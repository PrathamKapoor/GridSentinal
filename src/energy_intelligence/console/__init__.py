"""The operator and research console.

A read-only layer over what the phases have already measured. It never recomputes a result and
never presents a missing measurement as a zero; where a value does not exist it says UNKNOWN
and says why.
"""

from __future__ import annotations

from .render import (
    PHASE_STATUS,
    Console,
    PhaseRow,
    format_data_status,
    format_flexibility,
    format_health,
    next_command,
)

__all__ = [
    "PHASE_STATUS",
    "Console",
    "PhaseRow",
    "format_data_status",
    "format_flexibility",
    "format_health",
    "next_command",
]
