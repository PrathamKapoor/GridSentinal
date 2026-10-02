"""Energy Intelligence - adaptive, self-verifying energy management.

Phase 1: foundation and project scaffolding only.

This package deliberately contains **no** AI, mixture-of-experts, optimizer,
digital twin, red-team engine or MLOps agent. Those belong to later phases and
are tracked in ``handoff.md`` so that nothing here implies capability that does
not exist yet.

What exists in Phase 1:

* :mod:`energy_intelligence.config` - validated, immutable configuration
* :mod:`energy_intelligence.logging_setup` - structured JSON Lines logging
* :mod:`energy_intelligence.paths` - canonical filesystem layout
* :mod:`energy_intelligence.health` - environment health check
* :mod:`energy_intelligence.cli` - command line entry point

Example:
    >>> import energy_intelligence
    >>> energy_intelligence.__version__
    '0.1.0'
"""

from __future__ import annotations

from .config import AppConfig, ConfigError, ConfigValidationError, load_config
from .health import HealthReport, check_health, render_report
from .logging_setup import get_logger, initialize_logging
from .paths import ProjectPaths

__version__ = "0.1.0"

__all__ = [
    "AppConfig",
    "ConfigError",
    "ConfigValidationError",
    "HealthReport",
    "ProjectPaths",
    "__version__",
    "check_health",
    "get_logger",
    "initialize_logging",
    "load_config",
    "render_report",
]
