"""Dataset selection configuration.

Kept separate from both the run config (Phase 1, D-006) and the domain config
(Phase 2, D-037), because dataset selection is a third concern: *what external
data is being ingested*. Following the same pattern as D-037 keeps all three
loaders independent, so a machine can select a dataset without inheriting a run
configuration it does not use.

Unknown keys are rejected everywhere, for the reason in D-007.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

from .loader import CONFIG_DIR, PROJECT_ROOT
from .schema import ConfigError, ConfigValidationError

__all__ = [
    "DataConfig",
    "DATA_CONFIG_FILENAME",
    "default_data_config_path",
    "load_data_config",
]

DATA_CONFIG_FILENAME = "data.toml"

VALID_DEVICES_UNITS = ("degrees",)


@dataclass(frozen=True, slots=True)
class DataConfig:
    """Which SMART-DS subset to ingest.

    Attributes:
        dataset: Dataset family. Only ``"SMART-DS"`` is supported.
        version: Dataset version directory, e.g. ``"v1.0"``.
        year: Year directory.
        region: Region code.
        subregion: Sub-region code.
        scenario: Scenario directory name.
        substation: Sub-station folder.
        feeder: Feeder folder.
        raw_root: Local root the dataset was downloaded into. Relative paths
            resolve against the project root.
        peak_only: Materialise only the published peak timepoint rather than
            whole-year series.
    """

    dataset: str
    version: str
    year: str
    region: str
    subregion: str
    scenario: str
    substation: str
    feeder: str
    raw_root: Path
    peak_only: bool = True

    def __post_init__(self) -> None:
        errors: list[str] = []
        if self.dataset != "SMART-DS":
            errors.append(
                f"dataset must be 'SMART-DS' (the only adapter implemented), "
                f"got {self.dataset!r}"
            )
        for field_name in ("version", "year", "region", "subregion", "scenario",
                           "substation", "feeder"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                errors.append(f"{field_name} must be a non-empty string")
        if not isinstance(self.raw_root, Path):
            errors.append("raw_root must be a Path")
        if not isinstance(self.peak_only, bool):
            errors.append("peak_only must be a boolean")


        if errors:
            raise ConfigValidationError(errors)

    def as_dict(self) -> dict[str, object]:
        return {
            "dataset": self.dataset,
            "version": self.version,
            "year": self.year,
            "region": self.region,
            "subregion": self.subregion,
            "scenario": self.scenario,
            "substation": self.substation,
            "feeder": self.feeder,
            "raw_root": str(self.raw_root),
            "peak_only": self.peak_only,
        }


def default_data_config_path() -> Path:
    return CONFIG_DIR / DATA_CONFIG_FILENAME


def load_data_config(path: Path | str | None = None) -> DataConfig:
    """Load and validate the dataset selection configuration.

    Raises:
        ConfigError: If the file is missing or unreadable.
        ConfigValidationError: If the content is not a valid data config.
    """
    target = Path(path) if path is not None else default_data_config_path()
    if not target.is_absolute():
        target = PROJECT_ROOT / target

    if not target.is_file():
        raise ConfigError(
            f"Data config file not found: {target}. Create {DATA_CONFIG_FILENAME} "
            "or pass an explicit path."
        )

    try:
        raw = tomllib.loads(target.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"Data config {target} is not valid TOML: {exc}") from exc
    except OSError as exc:
        raise ConfigError(f"Could not read data config {target}: {exc}") from exc

    known = {
        "dataset", "version", "year", "region", "subregion", "scenario",
        "substation", "feeder", "raw_root", "peak_only",
    }
    errors = [
        f"{target}: unknown data config key {key!r} (valid: {sorted(known)})"
        for key in sorted(set(raw) - known)
    ]
    missing = sorted(known - set(raw) - {"peak_only"})
    errors.extend(f"{target}: missing required key {key!r}" for key in missing)
    if errors:
        raise ConfigValidationError(errors)

    raw_root = raw["raw_root"]
    if not isinstance(raw_root, str) or not raw_root.strip():
        raise ConfigValidationError([f"{target}: raw_root must be a non-empty string"])
    root = Path(raw_root).expanduser()
    if not root.is_absolute():
        root = PROJECT_ROOT / root

    peak_only = raw.get("peak_only", True)
    if not isinstance(peak_only, bool):
        raise ConfigValidationError([f"{target}: peak_only must be a boolean"])

    try:
        return DataConfig(
            dataset=raw["dataset"],
            version=raw["version"],
            year=str(raw["year"]),
            region=raw["region"],
            subregion=raw["subregion"],
            scenario=raw["scenario"],
            substation=raw["substation"],
            feeder=raw["feeder"],
            raw_root=root,
            peak_only=peak_only,
        )
    except ConfigValidationError as exc:
        raise ConfigValidationError([f"{target}: {e}" for e in exc.errors]) from exc
