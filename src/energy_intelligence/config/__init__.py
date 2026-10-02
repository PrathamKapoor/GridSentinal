"""Configuration subsystem.

Public surface for Phase 1:

    load_config(...)  ->  AppConfig  (validated, frozen, absolute paths)
    ConfigError / ConfigValidationError  ->  typed failure modes

Example:
    >>> from energy_intelligence.config import load_config
    >>> config = load_config()
    >>> config.experiment_name
    'phase1-baseline'
"""

from .loader import (
    CONFIG_DIR,
    CONFIG_DIR_ENV_VAR,
    DEFAULT_CONFIG_FILENAME,
    PROJECT_ROOT,
    default_config_path,
    find_project_root,
    load_config,
    load_raw_toml,
)
from .schema import (
    CONFIG_FIELDS,
    EXPERIMENT_NAME_PATTERN,
    MAX_SEED,
    VALID_DEVICES,
    VALID_LOG_LEVELS,
    AppConfig,
    ConfigError,
    ConfigValidationError,
    Device,
    LogLevel,
    coerce_app_config,
    validate_app_config,
)

__all__ = [
    "AppConfig",
    "CONFIG_DIR",
    "CONFIG_DIR_ENV_VAR",
    "CONFIG_FIELDS",
    "ConfigError",
    "ConfigValidationError",
    "DEFAULT_CONFIG_FILENAME",
    "Device",
    "EXPERIMENT_NAME_PATTERN",
    "LogLevel",
    "MAX_SEED",
    "PROJECT_ROOT",
    "VALID_DEVICES",
    "VALID_LOG_LEVELS",
    "coerce_app_config",
    "default_config_path",
    "find_project_root",
    "load_config",
    "load_raw_toml",
    "validate_app_config",
]
