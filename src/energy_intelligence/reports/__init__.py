"""Research tables and phase completion reports, derived from recorded artifacts.

Every table is generated from a result file. A table with no data is not written, because an
empty table that looks complete is worse than no table at all.
"""

from __future__ import annotations

from .phase10 import write_report, write_tables

__all__ = ["write_report", "write_tables"]
