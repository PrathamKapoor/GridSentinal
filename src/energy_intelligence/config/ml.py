"""Phase 4 ML configuration: a fourth file, for a fourth concern.

``configs/default.toml`` describes **a run**. ``configs/domain.toml`` describes
**the environment**. ``configs/data.toml`` describes **the dataset**. This file
describes **the experiment shape**: which target, which lookback, which horizons,
which split, which baselines.

It is separate from ``data.toml`` on purpose. The dataset selection is a fact
about what was downloaded; the lookback and horizon are choices about a question.
Changing the experiment shape must not invalidate the dataset selection, and
changing the dataset must not silently change the experiment.

Unknown keys are rejected, exactly as in the other three, so a typo fails loudly
instead of being ignored.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

from .loader import CONFIG_DIR

__all__ = ["MlConfig", "load_ml_config", "DEFAULT_ML_CONFIG"]

DEFAULT_ML_CONFIG = "ml.toml"


@dataclass(frozen=True, slots=True)
class MlConfig:
    """The experiment shape, resolved and validated.

    Attributes:
        target: Which declared target to forecast.
        label: Short artifact-directory label.
        lookback_steps: Steps of history per sample.
        horizons: Forecast horizons in 15-minute steps, ascending.
        train_fraction: Share of the grid used for training.
        validation_fraction: Share used for validation.
        origin_stride: Keep every Nth origin.
        customer_count: How many customers to sample for the load task.
        commercial_fraction: Share of the sample that must be commercial.
        include_weather: Whether to use real weather observations.
        seed: Seed for stochastic models.
        torch_threads: Torch CPU threads.
        models: Baselines to evaluate.
    """

    target: str
    label: str
    lookback_steps: int
    horizons: tuple[int, ...]
    train_fraction: float
    validation_fraction: float
    origin_stride: int
    customer_count: int
    commercial_fraction: float
    include_weather: bool
    seed: int
    torch_threads: int
    models: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "target": self.target,
            "label": self.label,
            "lookback_steps": self.lookback_steps,
            "horizons": list(self.horizons),
            "train_fraction": self.train_fraction,
            "validation_fraction": self.validation_fraction,
            "origin_stride": self.origin_stride,
            "customer_count": self.customer_count,
            "commercial_fraction": self.commercial_fraction,
            "include_weather": self.include_weather,
            "seed": self.seed,
            "torch_threads": self.torch_threads,
            "models": list(self.models),
        }


_ALLOWED_KEYS = frozenset(MlConfig.__dataclass_fields__)

_MODEL_CHOICES = frozenset(
    {
        "naive_last_value",
        "naive_seasonal_day",
        "naive_seasonal_week",
        "naive_drift",
        "classical_ridge",
        "classical_hist_gbm",
        "neural_mlp",
        "neural_gru",
    }
)


def load_ml_config(path: Path | None = None) -> MlConfig:
    """Load and validate ``configs/ml.toml``.

    Args:
        path: Explicit config path. Relative paths resolve against the project root.

    Returns:
        The resolved configuration.

    Raises:
        ConfigError: If the file is missing, unreadable, or fails validation.
    """
    from .schema import ConfigError, ConfigValidationError

    resolved = Path(path) if path is not None else CONFIG_DIR / DEFAULT_ML_CONFIG
    if not resolved.is_absolute():
        from .loader import PROJECT_ROOT

        resolved = (PROJECT_ROOT / resolved).resolve()
    if not resolved.is_file():
        raise ConfigError(f"ML config not found: {resolved}")

    try:
        payload = tomllib.loads(resolved.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{resolved}: invalid TOML: {exc}") from exc

    unknown = sorted(set(payload) - _ALLOWED_KEYS)
    errors: list[str] = []
    if unknown:
        errors.append(f"unknown keys: {unknown}")
    if errors:
        raise ConfigValidationError(errors)

    try:
        config = MlConfig(
            target=str(payload["target"]),
            label=str(payload["label"]),
            lookback_steps=int(payload["lookback_steps"]),
            horizons=tuple(int(h) for h in payload["horizons"]),
            train_fraction=float(payload["train_fraction"]),
            validation_fraction=float(payload["validation_fraction"]),
            origin_stride=int(payload["origin_stride"]),
            customer_count=int(payload["customer_count"]),
            commercial_fraction=float(payload["commercial_fraction"]),
            include_weather=bool(payload["include_weather"]),
            seed=int(payload["seed"]),
            torch_threads=int(payload["torch_threads"]),
            models=tuple(str(m) for m in payload["models"]),
        )
    except KeyError as exc:
        raise ConfigValidationError(
            [f"missing required key: {exc.args[0]!r} in {resolved}"]
        ) from exc

    errors = _validate(config)
    if errors:
        raise ConfigValidationError(errors)
    return config


def _validate(config: MlConfig) -> list[str]:
    """Check the experiment shape is coherent before any data is touched."""
    errors: list[str] = []
    if config.lookback_steps <= 0:
        errors.append("lookback_steps must be positive")
    if not config.horizons:
        errors.append("at least one horizon is required")
    elif any(h <= 0 for h in config.horizons):
        errors.append(f"horizons must be positive steps, got {list(config.horizons)}")
    elif tuple(sorted(config.horizons)) != config.horizons:
        errors.append("horizons must be sorted and unique")
    if not 0.0 < config.train_fraction < 1.0:
        errors.append("train_fraction must lie in (0, 1)")
    if not 0.0 < config.validation_fraction < 1.0:
        errors.append("validation_fraction must lie in (0, 1)")
    if config.train_fraction + config.validation_fraction >= 1.0:
        errors.append("train_fraction + validation_fraction must leave a test range")
    if config.origin_stride <= 0:
        errors.append("origin_stride must be positive")
    if config.customer_count <= 0:
        errors.append("customer_count must be positive")
    if not 0.0 <= config.commercial_fraction <= 1.0:
        errors.append("commercial_fraction must lie in [0, 1]")
    if config.torch_threads <= 0:
        errors.append("torch_threads must be positive")
    unknown_models = [m for m in config.models if m not in _MODEL_CHOICES]
    if unknown_models:
        errors.append(f"unknown models: {unknown_models}; allowed: {sorted(_MODEL_CHOICES)}")
    if not config.models:
        errors.append("at least one model is required")
    return errors