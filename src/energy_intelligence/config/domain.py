"""Domain configuration.

Phase 1 established *how* the project is organised; this module adds *what
energy system* it models, in the same spirit as D-006: nested frozen dataclasses
validated explicitly, read from TOML through the standard library, with unknown
keys rejected.

Kept deliberately separate from :class:`~energy_intelligence.config.schema.AppConfig`
rather than widening it. ``AppConfig`` describes a *run*; this describes the
*environment under study*. Keeping them apart means the 160 Phase 1 tests remain
meaningful, and Phase 3 can load a dataset config without a run config and vice
versa. See ``decisions.md`` (D-037).

Only fields that Phase 2 actually uses are present. No dataset paths, no model
paths, no simulation settings: those belong to their phases.
"""

from __future__ import annotations

import hashlib
import tomllib
from dataclasses import dataclass
from pathlib import Path

from .loader import CONFIG_DIR, PROJECT_ROOT
from .schema import ConfigError, ConfigValidationError

__all__ = [
    "TimeConfig",
    "EnergySystemConfig",
    "DataAnchor",
    "DomainConfig",
    "DOMAIN_CONFIG_FILENAME",
    "default_domain_config_path",
    "load_domain_config",
]

DOMAIN_CONFIG_FILENAME = "domain.toml"

VALID_TIMESTEPS_MINUTES: frozenset[int] = frozenset({1, 5, 15, 30, 60})


class DataAnchor:
    """Dataset families the domain model is known to represent.

    Not a choice of dataset -- Phase 3 selects and downloads. This records which
    families the Phase 2 model has been *checked* to be able to represent, so a
    later claim of representability is traceable rather than assumed (D-038).

    Only ``SMART_DS`` is recorded. RTS-GMLC, which the source forecasting
    repository used, cannot represent this system's action space: it has no
    batteries, EVs, flexible loads or network topology.
    """

    SMART_DS = "smart-ds"
    NONE = "none"


@dataclass(frozen=True, slots=True)
class TimeConfig:
    """Temporal grid configuration.

    Attributes:
        timestep_minutes: Native resolution. Constrained to a set of values for
            which integer aggregation to coarser grids is well defined.
        horizon_steps: Number of steps planned/forecast ahead.
        origin: ISO-8601 datetime the grid is anchored to.
    """

    timestep_minutes: int
    horizon_steps: int
    origin: str

    def __post_init__(self) -> None:
        errors: list[str] = []
        if isinstance(self.timestep_minutes, bool) or not isinstance(self.timestep_minutes, int):
            errors.append(
                f"time.timestep_minutes must be an int, got {type(self.timestep_minutes).__name__}"
            )
        elif self.timestep_minutes not in VALID_TIMESTEPS_MINUTES:
            errors.append(
                f"time.timestep_minutes must be one of {sorted(VALID_TIMESTEPS_MINUTES)}, "
                f"got {self.timestep_minutes}"
            )
        if isinstance(self.horizon_steps, bool) or not isinstance(self.horizon_steps, int):
            errors.append(
                f"time.horizon_steps must be an int, got {type(self.horizon_steps).__name__}"
            )
        elif self.horizon_steps <= 0:
            errors.append(f"time.horizon_steps must be positive, got {self.horizon_steps}")
        if not isinstance(self.origin, str) or not self.origin.strip():
            errors.append(f"time.origin must be a non-empty ISO-8601 string, got {self.origin!r}")
        if errors:
            raise ConfigValidationError(errors)

    @property
    def horizon_hours(self) -> float:
        return self.horizon_steps * self.timestep_minutes / 60.0


@dataclass(frozen=True, slots=True)
class EnergySystemConfig:
    """Identity and boundary of the environment under study.

    Attributes:
        system_id: Identifier used to construct
            :class:`~energy_intelligence.domain.identifiers.SystemId`.
        name: Human-readable name.
        description: What the boundary includes and excludes.
        topology_kind: ``"aggregate"`` (single node) or ``"feeder"``. Declared
            rather than inferred, because a conclusion drawn on an aggregate
            system cannot support per-location claims and the distinction must
            survive into every report.
    """

    system_id: str
    name: str
    description: str
    topology_kind: str

    def __post_init__(self) -> None:
        from ..domain.identifiers import SystemId
        from ..domain.topology import NetworkTopology  # noqa: F401 - import guard

        errors: list[str] = []
        try:
            SystemId(self.system_id)
        except Exception as exc:  # DomainValidationError
            errors.append(f"energy_system.system_id is invalid: {exc}")
        for field_name in ("name", "description"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                errors.append(f"energy_system.{field_name} must be a non-empty string")
        if self.topology_kind not in ("aggregate", "feeder"):
            errors.append(
                f"energy_system.topology_kind must be 'aggregate' or 'feeder', "
                f"got {self.topology_kind!r}"
            )
        if errors:
            raise ConfigValidationError(errors)


@dataclass(frozen=True, slots=True)
class DomainConfig:
    """Complete Phase 2 domain configuration.

    Attributes:
        time: Temporal grid.
        energy_system: Identity and boundary of the environment.
        data_anchor: Dataset family the model has been checked against.
    """

    time: TimeConfig
    energy_system: EnergySystemConfig
    data_anchor: str = DataAnchor.NONE

    def __post_init__(self) -> None:
        errors: list[str] = []
        if not isinstance(self.time, TimeConfig):
            errors.append(f"time must be a TimeConfig, got {type(self.time).__name__}")
        if not isinstance(self.energy_system, EnergySystemConfig):
            errors.append(
                f"energy_system must be an EnergySystemConfig, "
                f"got {type(self.energy_system).__name__}"
            )
        known = {DataAnchor.SMART_DS, DataAnchor.NONE}
        if self.data_anchor not in known:
            errors.append(
                f"data_anchor must be one of {sorted(known)}, got {self.data_anchor!r}"
            )
        if errors:
            raise ConfigValidationError(errors)

    def as_dict(self) -> dict[str, object]:
        return {
            "time": {
                "timestep_minutes": self.time.timestep_minutes,
                "horizon_steps": self.time.horizon_steps,
                "horizon_hours": self.time.horizon_hours,
                "origin": self.time.origin,
            },
            "energy_system": {
                "system_id": self.energy_system.system_id,
                "name": self.energy_system.name,
                "description": self.energy_system.description,
                "topology_kind": self.energy_system.topology_kind,
            },
            "data_anchor": self.data_anchor,
        }

    def fingerprint(self) -> str:
        """Short stable hash, recorded in logs to tie a run to its domain config."""
        material = "|".join(f"{key}={value}" for key, value in sorted(self._flat().items()))
        return hashlib.sha256(material.encode("utf-8")).hexdigest()[:12]

    def _flat(self) -> dict[str, object]:
        return {
            "time.timestep_minutes": self.time.timestep_minutes,
            "time.horizon_steps": self.time.horizon_steps,
            "time.origin": self.time.origin,
            "energy_system.system_id": self.energy_system.system_id,
            "energy_system.name": self.energy_system.name,
            "energy_system.description": self.energy_system.description,
            "energy_system.topology_kind": self.energy_system.topology_kind,
            "data_anchor": self.data_anchor,
        }


def default_domain_config_path() -> Path:
    return CONFIG_DIR / DOMAIN_CONFIG_FILENAME


def load_domain_config(path: Path | str | None = None) -> DomainConfig:
    """Load and validate the domain configuration.

    Args:
        path: Config file to read. Defaults to ``configs/domain.toml``.

    Returns:
        A validated :class:`DomainConfig`.

    Raises:
        ConfigError: If the file is missing, unreadable or invalid TOML.
        ConfigValidationError: If the content is not a valid domain config.
    """
    target = Path(path) if path is not None else default_domain_config_path()
    if not target.is_absolute():
        target = PROJECT_ROOT / target

    if not target.is_file():
        raise ConfigError(
            f"Domain config file not found: {target}. Create {DOMAIN_CONFIG_FILENAME} "
            "or pass an explicit path."
        )

    try:
        raw = tomllib.loads(target.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"Domain config {target} is not valid TOML: {exc}") from exc
    except OSError as exc:
        raise ConfigError(f"Could not read domain config {target}: {exc}") from exc

    errors: list[str] = []
    known_top = {"time", "energy_system", "data_anchor"}
    for key in sorted(set(raw) - known_top):
        errors.append(f"{target}: unknown domain config key {key!r} (valid: {sorted(known_top)})")

    for section in ("time", "energy_system"):
        if section not in raw:
            errors.append(f"{target}: missing required section [{section}]")
        # No isinstance check needed: tomllib always yields a dict for a table
        # header, so raw[section] is a mapping whenever the key is present.

    if "time" in raw and isinstance(raw["time"], dict):
        known_time = {"timestep_minutes", "horizon_steps", "origin"}
        for key in sorted(set(raw["time"]) - known_time):
            errors.append(f"{target}: unknown key {key!r} in [time] (valid: {sorted(known_time)})")

    if "energy_system" in raw and isinstance(raw["energy_system"], dict):
        known_sys = {"system_id", "name", "description", "topology_kind"}
        for key in sorted(set(raw["energy_system"]) - known_sys):
            errors.append(
                f"{target}: unknown key {key!r} in [energy_system] (valid: {sorted(known_sys)})"
            )

    if errors:
        raise ConfigValidationError(errors)

    anchor = raw.get("data_anchor", DataAnchor.NONE)
    if not isinstance(anchor, str):
        errors.append(f"{target}: data_anchor must be a string, got {type(anchor).__name__}")

    if errors:
        raise ConfigValidationError(errors)

    try:
        time_cfg = TimeConfig(
            timestep_minutes=raw["time"]["timestep_minutes"],
            horizon_steps=raw["time"]["horizon_steps"],
            origin=raw["time"]["origin"],
        )
        system_cfg = EnergySystemConfig(
            system_id=raw["energy_system"]["system_id"],
            name=raw["energy_system"]["name"],
            description=raw["energy_system"]["description"],
            topology_kind=raw["energy_system"]["topology_kind"],
        )
    except KeyError as exc:
        raise ConfigValidationError([f"{target}: missing required field {exc}"]) from exc
    except ConfigValidationError as exc:
        raise ConfigValidationError([f"{target}: {error}" for error in exc.errors]) from exc

    return DomainConfig(time=time_cfg, energy_system=system_cfg, data_anchor=anchor)
