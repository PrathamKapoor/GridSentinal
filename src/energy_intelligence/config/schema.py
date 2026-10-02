"""Configuration schema and validation.

Design constraints for Phase 1:

* Zero runtime dependencies. The schema is built from :mod:`dataclasses` and
  validated by explicit, hand-written checks so that every validation rule is
  visible, testable and auditable.
* Configuration is immutable. Instances are frozen dataclasses so that a config
  handed to a component cannot be mutated mid-run, which would silently break
  experiment reproducibility.
* No energy-domain constants are hard-coded here. Values such as the number of
  MoE experts, the optimizer, or the target energy system are deliberately
  ``TBD`` until their phases define them. See ``decisions.md`` (D-006).
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

__all__ = [
    "ConfigError",
    "ConfigValidationError",
    "AppConfig",
    "Device",
    "LogLevel",
    "VALID_DEVICES",
    "VALID_LOG_LEVELS",
    "CONFIG_FIELDS",
    "MAX_SEED",
    "EXPERIMENT_NAME_PATTERN",
    "validate_app_config",
    "coerce_app_config",
]

MAX_SEED = 2**32 - 1

# Experiment names are used to build log file names and artifact directory
# names. Restricting the character set prevents path traversal and filesystem
# breakage from a typo in a config file.
EXPERIMENT_NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")


class ConfigError(Exception):
    """Base class for configuration failures."""


class ConfigValidationError(ConfigError):
    """Raised when a configuration value is missing, malformed or out of range.

    Carries every problem found, not just the first, so a user editing a TOML
    file can fix all errors in one pass.
    """

    def __init__(self, errors: list[str]) -> None:
        self.errors = tuple(errors)
        detail = "\n".join(f"  - {error}" for error in self.errors)
        super().__init__(f"Invalid configuration ({len(self.errors)} error(s)):\n{detail}")


class Device(StrEnum):
    """Compute devices the project is allowed to target.

    Phase 1 performs no computation and imports no ML framework, so this enum is
    configuration vocabulary only. It does not assert that a device is present.
    """

    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"


class LogLevel(StrEnum):
    """Accepted logging levels."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


VALID_DEVICES: frozenset[str] = frozenset(device.value for device in Device)
VALID_LOG_LEVELS: frozenset[str] = frozenset(level.value for level in LogLevel)


@dataclass(frozen=True, slots=True)
class AppConfig:
    """Root application configuration.

    The seven fields below are the complete Phase 1 surface: they are exactly
    the knobs named in the Phase 1 brief (seed, device, model path, dataset
    path, experiment name, logging level, output directory) plus
    ``artifact_dir``, which experiment runners need in order to write outputs
    somewhere other than ``data/``.

    ``project_root`` is injected by the loader, not authored by hand, so that
    relative paths can be resolved against a single, known anchor.
    """

    seed: int
    device: str
    model_path: Path
    dataset_path: Path
    experiment_name: str
    log_level: str
    output_dir: Path
    artifact_dir: Path
    log_dir: Path
    project_root: Path = field(default_factory=Path.cwd)

    @property
    def is_cpu_only(self) -> bool:
        return self.device == Device.CPU.value

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable view, used for logging and test assertions."""
        return {
            "seed": self.seed,
            "device": self.device,
            "model_path": str(self.model_path),
            "dataset_path": str(self.dataset_path),
            "experiment_name": self.experiment_name,
            "log_level": self.log_level,
            "output_dir": str(self.output_dir),
            "artifact_dir": str(self.artifact_dir),
            "log_dir": str(self.log_dir),
            "project_root": str(self.project_root),
        }

    def fingerprint(self) -> str:
        """Return a short stable hash of the configuration.

        Recorded in logs so a run can later be tied back to the exact settings
        that produced it.

        Note: the ``project_root`` key is excluded, but every path has already
        been resolved to an absolute path that *contains* that root. The
        fingerprint is therefore stable for repeated runs in one checkout and
        deliberately differs between two different checkouts.
        """
        material = "|".join(
            str(value)
            for key, value in sorted(self.as_dict().items())
            if key != "project_root"
        )
        return hashlib.sha256(material.encode("utf-8")).hexdigest()[:12]


#: Authored configuration keys, i.e. everything except the loader-injected
#: ``project_root``. Unknown keys in a TOML file are rejected so typos such as
#: ``experimant_name`` cannot be silently ignored.
CONFIG_FIELDS: tuple[str, ...] = (
    "seed",
    "device",
    "model_path",
    "dataset_path",
    "experiment_name",
    "log_level",
    "output_dir",
    "artifact_dir",
    "log_dir",
)

_PATH_FIELDS: tuple[str, ...] = (
    "model_path",
    "dataset_path",
    "output_dir",
    "artifact_dir",
    "log_dir",
)


def _coerce_seed(raw: object, errors: list[str]) -> int:
    # bool is a subclass of int; True must not silently become seed 1.
    if isinstance(raw, bool) or not isinstance(raw, int):
        errors.append(f"'seed' must be an integer, got {type(raw).__name__}: {raw!r}")
        return 0
    if not 0 <= raw <= MAX_SEED:
        errors.append(f"'seed' must be between 0 and {MAX_SEED}, got {raw}")
    return raw


def _coerce_choice(raw: object, field_name: str, allowed: frozenset[str], errors: list[str]) -> str:
    if not isinstance(raw, str):
        errors.append(f"'{field_name}' must be a string, got {type(raw).__name__}: {raw!r}")
        return ""
    normalised = raw.strip().upper() if field_name == "log_level" else raw.strip().lower()
    if normalised not in allowed:
        errors.append(
            f"'{field_name}' must be one of {sorted(allowed)}, got {raw!r}"
        )
    return normalised


def _coerce_experiment_name(raw: object, errors: list[str]) -> str:
    if not isinstance(raw, str):
        errors.append(
            f"'experiment_name' must be a string, got {type(raw).__name__}: {raw!r}"
        )
        return ""
    if not EXPERIMENT_NAME_PATTERN.match(raw):
        errors.append(
            "'experiment_name' must match "
            f"{EXPERIMENT_NAME_PATTERN.pattern} (lowercase, starts alphanumeric, "
            f"max 64 chars); got {raw!r}"
        )
    return raw


def _coerce_path(raw: object, field_name: str, project_root: Path, errors: list[str]) -> Path:
    if not isinstance(raw, str) or not raw.strip():
        errors.append(f"'{field_name}' must be a non-empty string path, got {raw!r}")
        return project_root
    candidate = Path(raw.strip()).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (project_root / candidate).resolve()


def coerce_app_config(
    raw: object,
    *,
    project_root: Path,
    source: str = "<mapping>",
) -> AppConfig:
    """Build and fully validate an :class:`AppConfig` from a raw mapping.

    Args:
        raw: Mapping of configuration keys to values, typically a TOML table.
        project_root: Anchor used to resolve relative paths.
        source: Human-readable origin used in error messages.

    Returns:
        A validated, frozen :class:`AppConfig`.

    Raises:
        ConfigValidationError: If any value is missing, of the wrong type, or
            out of range. All problems are reported together.
    """
    errors: list[str] = []
    root = Path(project_root).resolve()

    if not isinstance(raw, dict):
        raise ConfigValidationError(
            [f"{source}: expected a table of settings, got {type(raw).__name__}"]
        )

    known = set(CONFIG_FIELDS)
    for key in sorted(set(raw) - known):
        errors.append(f"{source}: unknown configuration key {key!r} (valid keys: {sorted(known)})")

    for key in CONFIG_FIELDS:
        if key not in raw:
            errors.append(f"{source}: missing required configuration key {key!r}")

    seed = _coerce_seed(raw.get("seed"), errors) if "seed" in raw else 0
    device = _coerce_choice(raw.get("device"), "device", VALID_DEVICES, errors) if "device" in raw else ""
    log_level = (
        _coerce_choice(raw.get("log_level"), "log_level", VALID_LOG_LEVELS, errors)
        if "log_level" in raw
        else ""
    )
    experiment_name = (
        _coerce_experiment_name(raw.get("experiment_name"), errors)
        if "experiment_name" in raw
        else ""
    )

    resolved: dict[str, Path] = {}
    for key in _PATH_FIELDS:
        if key in raw:
            resolved[key] = _coerce_path(raw[key], key, root, errors)
        else:
            resolved[key] = root

    if errors:
        raise ConfigValidationError(errors)

    return AppConfig(
        seed=seed,
        device=device,
        model_path=resolved["model_path"],
        dataset_path=resolved["dataset_path"],
        experiment_name=experiment_name,
        log_level=log_level,
        output_dir=resolved["output_dir"],
        artifact_dir=resolved["artifact_dir"],
        log_dir=resolved["log_dir"],
        project_root=root,
    )


def validate_app_config(config: AppConfig) -> None:
    """Re-validate an already-constructed config.

    Useful as a defensive check at trust boundaries (e.g. a config built by
    hand in a test or passed across a process boundary). Raises
    :class:`ConfigValidationError` on the first failing field.
    """
    errors: list[str] = []

    if isinstance(config.seed, bool) or not isinstance(config.seed, int):
        errors.append(f"'seed' must be an integer, got {type(config.seed).__name__}")
    elif not 0 <= config.seed <= MAX_SEED:
        errors.append(f"'seed' must be between 0 and {MAX_SEED}, got {config.seed}")

    if config.device not in VALID_DEVICES:
        errors.append(f"'device' must be one of {sorted(VALID_DEVICES)}, got {config.device!r}")

    if config.log_level not in VALID_LOG_LEVELS:
        errors.append(
            f"'log_level' must be one of {sorted(VALID_LOG_LEVELS)}, got {config.log_level!r}"
        )

    if not EXPERIMENT_NAME_PATTERN.match(config.experiment_name):
        errors.append(
            f"'experiment_name' must match {EXPERIMENT_NAME_PATTERN.pattern}, "
            f"got {config.experiment_name!r}"
        )

    for key in _PATH_FIELDS:
        value = getattr(config, key)
        if not isinstance(value, Path):
            errors.append(f"{key!r} must be a pathlib.Path, got {type(value).__name__}")
        elif not value.is_absolute():
            errors.append(f"{key!r} must be an absolute path, got {value!r}")

    if errors:
        raise ConfigValidationError(errors)
