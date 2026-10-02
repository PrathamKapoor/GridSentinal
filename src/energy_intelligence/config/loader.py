"""Configuration loading.

Phase 1 deliberately keeps loading simple and explicit:

    configs/*.toml  ->  tomllib  ->  dict  ->  coerce_app_config  ->  AppConfig

There is no inheritance chain, no environment-variable interpolation and no
remote configuration source. Those features are only worth their complexity once
a second consumer (the Jenkins pipeline in Phase 15) actually needs them. See
``decisions.md`` (D-005).
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any

from .schema import (
    CONFIG_FIELDS,
    AppConfig,
    ConfigError,
    ConfigValidationError,
    coerce_app_config,
)

__all__ = [
    "PROJECT_ROOT",
    "DEFAULT_CONFIG_FILENAME",
    "CONFIG_DIR_ENV_VAR",
    "find_project_root",
    "default_config_path",
    "load_config",
    "load_raw_toml",
]

#: Environment variable that points at a specific config file. Lets CI or the
#: Jenkins pipeline select a config without rewriting the command.
CONFIG_DIR_ENV_VAR = "ENERGY_INTEL_CONFIG"

DEFAULT_CONFIG_FILENAME = "default.toml"


def find_project_root(start: Path | None = None) -> Path:
    """Locate the repository root.

    Walks upward looking for a directory containing both ``pyproject.toml`` and
    ``src/energy_intelligence``. When the package is installed rather than run
    from a checkout, falls back to the current working directory.

    Args:
        start: Directory to begin the search from. Defaults to this file's
            location, which is ``src/energy_intelligence/config/loader.py``.

    Returns:
        Absolute path to the project root.

    Raises:
        ConfigError: If the package is installed outside a project checkout and
            no root marker can be found. Failing loudly beats silently
            resolving relative paths against an arbitrary directory.
    """
    origin = Path(start) if start is not None else Path(__file__).resolve()
    current = origin if origin.is_dir() else origin.parent

    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").is_file() and (
            candidate / "src" / "energy_intelligence"
        ).is_dir():
            return candidate

    raise ConfigError(
        "Could not locate the project root: no ancestor directory contains both "
        "'pyproject.toml' and 'src/energy_intelligence/'. Run commands from a "
        "repository checkout, or set "
        f"{CONFIG_DIR_ENV_VAR} to an absolute config path."
    )


def project_root() -> Path:
    """Return the project root, cached for the process lifetime."""
    return find_project_root()


PROJECT_ROOT: Path = project_root()
CONFIG_DIR: Path = PROJECT_ROOT / "configs"


def default_config_path() -> Path:
    """Return the path of the default config file."""
    return CONFIG_DIR / DEFAULT_CONFIG_FILENAME


def load_raw_toml(path: Path) -> dict[str, Any]:
    """Parse a TOML file into a plain dict.

    Args:
        path: Config file to read.

    Returns:
        The parsed mapping.

    Raises:
        ConfigError: If the file is missing, unreadable, not valid TOML, or does
            not contain a top-level table.
    """
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"Config file not found: {path}")

    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"Config file {path} is not valid TOML: {exc}") from exc
    except OSError as exc:
        raise ConfigError(f"Could not read config file {path}: {exc}") from exc

    if not isinstance(raw, dict):  # pragma: no cover - tomllib always returns a dict
        raise ConfigError(f"Config file {path} must contain a top-level table")

    return raw


def load_config(
    path: Path | str | None = None,
    *,
    overrides: dict[str, Any] | None = None,
) -> AppConfig:
    """Load, merge and validate configuration.

    Resolution order, lowest priority first:

    1. ``configs/default.toml`` in the project root.
    2. The TOML file given by ``path``.
    3. The ``path`` from the ``ENERGY_INTEL_CONFIG`` environment variable, when
       ``path`` is not given.
    4. The ``overrides`` mapping (e.g. parsed CLI flags).

    Args:
        path: Optional config file to load on top of the defaults.
        overrides: Optional key/value overrides applied last. Keys outside
            :data:`~energy_intelligence.config.schema.CONFIG_FIELDS` are
            rejected.

    Returns:
        A validated, frozen :class:`AppConfig`.

    Raises:
        ConfigError: If a file cannot be read or does not exist.
        ConfigValidationError: If the merged configuration is invalid.

    Example:
        >>> from energy_intelligence.config import load_config
        >>> config = load_config(overrides={"seed": 7})  # doctest: +SKIP
    """
    root = PROJECT_ROOT
    merged: dict[str, Any] = {}

    default_path = default_config_path()
    if default_path.is_file():
        merged.update(load_raw_toml(default_path))
    elif path is None:
        raise ConfigError(
            f"Default config file is missing: {default_path}. Create it or pass an "
            "explicit config path."
        )

    chosen = path
    if chosen is None:
        env_path = os.environ.get(CONFIG_DIR_ENV_VAR)
        if env_path:
            chosen = env_path

    if chosen is not None:
        chosen_path = Path(chosen).expanduser()
        if not chosen_path.is_absolute():
            chosen_path = root / chosen_path
        source_config = load_raw_toml(chosen_path)
        unknown = set(source_config) - set(CONFIG_FIELDS)
        if unknown:
            raise ConfigValidationError(
                [
                    f"{chosen_path}: unknown configuration key(s) {sorted(unknown)}; "
                    f"valid keys: {sorted(CONFIG_FIELDS)}"
                ]
            )
        merged.update(source_config)

    if overrides:
        unknown_overrides = set(overrides) - set(CONFIG_FIELDS)
        if unknown_overrides:
            raise ConfigValidationError(
                [
                    f"Overrides: unknown configuration key(s) {sorted(unknown_overrides)}; "
                    f"valid keys: {sorted(CONFIG_FIELDS)}"
                ]
            )
        merged.update({key: value for key, value in overrides.items() if value is not None})

    return coerce_app_config(merged, project_root=root, source=str(chosen or default_path))
