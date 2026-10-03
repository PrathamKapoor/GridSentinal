"""Phase 5 configuration: the temporal experiment shape.

A third experiment-shaped file would be duplication, so this lives inside the same
concern as Phase 4's ``configs/ml.toml`` but is kept separate because Phase 5 asks a
different question and must not silently inherit Phase 4's answers: Phase 4 asked
"can this data be forecast at all", Phase 5 asks "does a temporal neural architecture
add anything".

Inheriting the Phase 4 task definition - same target, same 40 customers, same seed,
same horizons, same split fractions - is deliberate and is what makes the two phases'
test numbers comparable. Everything *architectural* is decided here.

Unknown keys are rejected, as in every other config file.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

from .loader import CONFIG_DIR

__all__ = ["TemporalExperimentConfig", "load_temporal_config", "DEFAULT_TEMPORAL_CONFIG"]

DEFAULT_TEMPORAL_CONFIG = "temporal.toml"

_ALLOWED = frozenset(
    {
        "target",
        "label",
        "customer_count",
        "commercial_fraction",
        "seed",
        "torch_threads",
        "lookback_steps",
        "token_stride",
        "train_origin_stride",
        "train_fraction",
        "validation_fraction",
        "horizons",
        "architecture",
        "cyclic_channels",
        "delta_target",
        "model_params",
        "epochs",
        "batch_size",
        "learning_rate",
        "weight_decay",
        "loss",
        "huber_delta",
        "patience",
        "min_delta",
        "max_seconds",
        "grad_clip",
        "validation_stride",
        "ablation_budget_epochs",
        "ablation_budget_stride",
        "ablation_validation_stride",
        "ablations",
    }
)


@dataclass(frozen=True, slots=True)
class TemporalExperimentConfig:
    """The full Phase 5 experiment definition.

    Attributes:
        target: Which declared target. Demand only, in this phase.
        label: Artifact directory under ``artifacts/phase5``.
        customer_count: Customers sampled, matching Phase 4 so the test population
            is identical.
        commercial_fraction: Minority-class share in the sample.
        seed: Sampling and training seed.
        torch_threads: Torch CPU threads.
        lookback_steps: History window in native 15-minute steps.
        token_stride: Read every Nth step of the window.
        train_origin_stride: Origin stride for training rows; evaluation always 1.
        train_fraction: Training share of the grid.
        validation_fraction: Validation share of the grid.
        horizons: Forecast horizons in native steps.
        architecture: ``tcn`` or ``transformer``.
        cyclic_channels: Whether the four cyclic channels are fed.
        delta_target: Whether the model predicts the change from y(t).
        model_params: Architecture overrides.
        epochs: Maximum epochs.
        batch_size: Samples per step.
        learning_rate: Adam learning rate.
        weight_decay: Adam weight decay.
        loss: ``l1`` or ``huber``.
        huber_delta: Delta for the Huber loss.
        patience: Validation patience.
        min_delta: Minimum validation improvement.
        max_seconds: Wall-clock cap per training run.
        grad_clip: Gradient-norm clip, 0 to disable.
        validation_stride: Validation rows skipped for per-epoch checkpoint
            selection. The final evaluation always uses every test row.
        ablation_budget_epochs: Epochs for reduced-budget ablations.
        ablation_budget_stride: Training stride for reduced-budget ablations.
        ablation_validation_stride: Validation stride for reduced-budget ablations.
        ablations: Named ablation definitions.
    """

    target: str
    label: str
    customer_count: int
    commercial_fraction: float
    seed: int
    torch_threads: int
    lookback_steps: int
    token_stride: int
    train_origin_stride: int
    train_fraction: float
    validation_fraction: float
    horizons: tuple[int, ...]
    architecture: str
    cyclic_channels: bool
    delta_target: bool
    model_params: dict[str, object]
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    loss: str
    huber_delta: float
    patience: int
    min_delta: float
    max_seconds: float
    grad_clip: float
    validation_stride: int
    ablation_budget_epochs: int
    ablation_budget_stride: int
    ablation_validation_stride: int
    ablations: tuple[dict[str, object], ...] = ()

    def sequence_config(self, **overrides: object):
        """Build the Phase 5 :class:`SequenceConfig` for this experiment.

        Overrides are merged rather than passed as extra keyword arguments, so an
        override of a key that is also a base field replaces it instead of colliding.
        """
        from ..ml.sequence import SequenceConfig

        payload: dict[str, object] = {
            "lookback_steps": self.lookback_steps,
            "token_stride": self.token_stride,
            "horizons": self.horizons,
            "train_origin_stride": self.train_origin_stride,
            "train_fraction": self.train_fraction,
            "validation_fraction": self.validation_fraction,
            "cyclic_channels": self.cyclic_channels,
            "delta_target": self.delta_target,
        }
        payload.update(overrides)
        return SequenceConfig(**payload)  # type: ignore[arg-type]

    def training_config(self, *, epochs: int | None = None, seed: int | None = None):
        """Build the training protocol, optionally at a reduced budget."""
        from ..ml.training import TrainingConfig

        return TrainingConfig(
            epochs=epochs or self.epochs,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            loss=self.loss,
            huber_delta=self.huber_delta,
            patience=self.patience,
            min_delta=self.min_delta,
            max_seconds=self.max_seconds,
            seed=seed or self.seed,
            threads=self.torch_threads,
            grad_clip=self.grad_clip,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "target": self.target,
            "label": self.label,
            "customer_count": self.customer_count,
            "commercial_fraction": self.commercial_fraction,
            "seed": self.seed,
            "torch_threads": self.torch_threads,
            "lookback_steps": self.lookback_steps,
            "token_stride": self.token_stride,
            "train_origin_stride": self.train_origin_stride,
            "train_fraction": self.train_fraction,
            "validation_fraction": self.validation_fraction,
            "horizons": list(self.horizons),
            "architecture": self.architecture,
            "cyclic_channels": self.cyclic_channels,
            "delta_target": self.delta_target,
            "model_params": dict(self.model_params),
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "loss": self.loss,
            "huber_delta": self.huber_delta,
            "patience": self.patience,
            "min_delta": self.min_delta,
            "max_seconds": self.max_seconds,
            "grad_clip": self.grad_clip,
            "validation_stride": self.validation_stride,
            "ablation_budget_epochs": self.ablation_budget_epochs,
            "ablation_budget_stride": self.ablation_budget_stride,
            "ablation_validation_stride": self.ablation_validation_stride,
            "ablations": [dict(item) for item in self.ablations],
        }


def _ablations_from(payload: object) -> tuple[dict[str, object], ...]:
    """Validate the ablation table.

    Each entry names the question it asks and the overrides that answer it, so an
    ablation without a stated reason is refused.
    """
    if not isinstance(payload, list):
        return ()
    allowed_overrides = {
        "lookback_steps",
        "token_stride",
        "cyclic_channels",
        "delta_target",
        "horizons",
        "architecture",
        "train_origin_stride",
        "learning_rate",
        "model_params",
    }
    out: list[dict[str, object]] = []
    for entry in payload:
        if not isinstance(entry, dict):
            raise ValueError(f"each ablation must be a table, got {type(entry).__name__}")
        name = entry.get("name")
        question = entry.get("question")
        if not name or not question:
            raise ValueError(f"ablation {entry} must declare both 'name' and 'question'")
        overrides = entry.get("overrides", {})
        unknown = set(overrides) - allowed_overrides
        if unknown:
            raise ValueError(f"ablation {name!r} has unknown overrides: {sorted(unknown)}")
        out.append({"name": name, "question": question, "overrides": dict(overrides)})
    return tuple(out)


def load_temporal_config(path: Path | None = None) -> TemporalExperimentConfig:
    """Load and validate ``configs/temporal.toml``.

    Raises:
        ConfigError: If the file is missing or unreadable.
        ConfigValidationError: If any field is missing, unknown or out of range.
    """
    from .schema import ConfigError, ConfigValidationError

    resolved = Path(path) if path is not None else CONFIG_DIR / DEFAULT_TEMPORAL_CONFIG
    if not resolved.is_absolute():
        from .loader import PROJECT_ROOT

        resolved = (PROJECT_ROOT / resolved).resolve()
    if not resolved.is_file():
        raise ConfigError(f"temporal config not found: {resolved}")
    try:
        payload = tomllib.loads(resolved.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{resolved}: invalid TOML: {exc}") from exc

    unknown = sorted(set(payload) - _ALLOWED)
    if unknown:
        raise ConfigValidationError([f"unknown keys: {unknown}"])

    required = {
        "target", "label", "customer_count", "commercial_fraction", "seed",
        "torch_threads", "lookback_steps", "token_stride", "train_origin_stride",
        "train_fraction", "validation_fraction", "horizons", "architecture",
        "epochs", "batch_size", "learning_rate",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ConfigValidationError([f"missing required key: {name}" for name in missing])

    config = TemporalExperimentConfig(
        target=str(payload["target"]),
        label=str(payload["label"]),
        customer_count=int(payload["customer_count"]),
        commercial_fraction=float(payload["commercial_fraction"]),
        seed=int(payload["seed"]),
        torch_threads=int(payload["torch_threads"]),
        lookback_steps=int(payload["lookback_steps"]),
        token_stride=int(payload["token_stride"]),
        train_origin_stride=int(payload["train_origin_stride"]),
        train_fraction=float(payload["train_fraction"]),
        validation_fraction=float(payload["validation_fraction"]),
        horizons=tuple(int(h) for h in payload["horizons"]),
        architecture=str(payload["architecture"]),
        cyclic_channels=bool(payload.get("cyclic_channels", True)),
        delta_target=bool(payload.get("delta_target", True)),
        model_params=dict(payload.get("model_params", {})),
        epochs=int(payload["epochs"]),
        batch_size=int(payload["batch_size"]),
        learning_rate=float(payload["learning_rate"]),
        weight_decay=float(payload.get("weight_decay", 0.0)),
        loss=str(payload.get("loss", "l1")),
        huber_delta=float(payload.get("huber_delta", 0.05)),
        patience=int(payload.get("patience", 3)),
        min_delta=float(payload.get("min_delta", 1e-5)),
        max_seconds=float(payload.get("max_seconds", 3600.0)),
        grad_clip=float(payload.get("grad_clip", 1.0)),
        validation_stride=int(payload.get("validation_stride", 1)),
        ablation_budget_epochs=int(payload.get("ablation_budget_epochs", 3)),
        ablation_budget_stride=int(payload.get("ablation_budget_stride", 16)),
        ablation_validation_stride=int(payload.get("ablation_validation_stride", 8)),
        ablations=_ablations_from(payload.get("ablations", [])),
    )

    errors = _validate(config)
    if errors:
        raise ConfigValidationError(errors)
    return config


def _validate(config: TemporalExperimentConfig) -> list[str]:
    """Reject any configuration that cannot produce a valid experiment."""
    from ..ml.temporal import ARCHITECTURE_NAMES

    errors: list[str] = []
    if config.architecture not in ARCHITECTURE_NAMES:
        errors.append(
            f"architecture must be one of {list(ARCHITECTURE_NAMES)}, got "
            f"{config.architecture!r}"
        )
    if config.target != "customer_load":
        errors.append(
            "Phase 5 targets demand only: SMART-DS provides no battery, EV, wind or "
            "net-load series, so no other target can be modelled without inventing data"
        )
    if config.lookback_steps <= 0:
        errors.append("lookback_steps must be positive")
    if config.token_stride <= 0:
        errors.append("token_stride must be positive")
    if config.token_stride >= config.lookback_steps:
        errors.append("token_stride must be shorter than lookback_steps")
    if config.train_origin_stride <= 0:
        errors.append("train_origin_stride must be positive")
    if not 0.0 < config.train_fraction < 1.0:
        errors.append("train_fraction must lie in (0, 1)")
    if not 0.0 < config.validation_fraction < 1.0:
        errors.append("validation_fraction must lie in (0, 1)")
    if config.train_fraction + config.validation_fraction >= 1.0:
        errors.append("train_fraction + validation_fraction must leave a test range")
    if not config.horizons:
        errors.append("at least one horizon is required")
    elif any(h <= 0 for h in config.horizons):
        errors.append("horizons must be positive native steps")
    elif tuple(sorted(config.horizons)) != config.horizons:
        errors.append("horizons must be sorted and unique")
    if config.epochs <= 0:
        errors.append("epochs must be positive")
    if config.batch_size <= 0:
        errors.append("batch_size must be positive")
    if config.learning_rate <= 0:
        errors.append("learning_rate must be positive")
    if config.customer_count <= 0:
        errors.append("customer_count must be positive")
    if config.torch_threads <= 0:
        errors.append("torch_threads must be positive")
    if config.validation_stride <= 0:
        errors.append("validation_stride must be positive")
    if config.ablation_budget_epochs <= 0:
        errors.append("ablation_budget_epochs must be positive")
    if config.ablation_budget_stride <= 0:
        errors.append("ablation_budget_stride must be positive")
    if config.ablation_validation_stride <= 0:
        errors.append("ablation_validation_stride must be positive")
    return errors
