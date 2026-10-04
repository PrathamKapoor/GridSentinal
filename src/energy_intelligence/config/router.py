"""Phase 7 configuration: the router experiment shape.

A fifth experiment file, for the same reason Phase 5 and Phase 6 each got one: the phase
asks a different question and must not silently inherit another's answers. What is
inherited deliberately is the **task** - same target, same 40 customers, same seed, same
horizons, same chronological split fractions - because that is what makes Phase 7's test
numbers comparable with Phase 4's and Phase 5's. Everything about the router is decided
here.

The task fields are validated, not trusted: a config that moved the horizons or the split
would produce a table that looks like Phase 4's and is not comparable with it, and the
loader rejects that rather than letting it reach a published number.

Unknown keys are rejected, as in every other config file.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

from .loader import CONFIG_DIR

__all__ = [
    "RouterExperimentConfig",
    "load_router_config",
    "DEFAULT_ROUTER_CONFIG",
    "PHASE7_TASK_REFERENCE",
]

DEFAULT_ROUTER_CONFIG = "router.toml"

#: The task Phase 7 inherits from Phase 5, restated so a mismatch is caught at load time.
PHASE7_TASK_REFERENCE: dict[str, object] = {
    "target": "customer_load",
    "customer_count": 40,
    "horizons": [1, 4, 96],
    "train_fraction": 0.70,
    "validation_fraction": 0.15,
    "lookback_steps": 672,
}

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
        "train_fraction",
        "validation_fraction",
        "horizons",
        "architecture",
        "hidden_sizes",
        "dropout",
        "temperature",
        "epochs",
        "batch_size",
        "learning_rate",
        "weight_decay",
        "patience",
        "min_delta",
        "max_seconds",
        "router_train_fraction_of_validation",
        "ablation_epochs",
        "run_ablations",
        "run_crossfit",
        "trace_rows",
        "expert_cache",
        "published_test_mae",
    }
)


@dataclass(frozen=True, slots=True)
class RouterExperimentConfig:
    """The full Phase 7 experiment definition.

    Attributes:
        target: Which declared target. Demand only, as in Phases 4 and 5.
        label: Artifact directory under ``artifacts/phase7``.
        customer_count: Customers sampled, matching Phase 5.
        commercial_fraction: Minority-class share in the sample.
        seed: Sampling, training and estimator seed.
        torch_threads: Torch CPU threads.
        lookback_steps: History window; must match Phase 5 so the panel is identical.
        token_stride: Read-every-Nth step of the window; must match Phase 5.
        train_fraction: Training share of the grid.
        validation_fraction: Validation share of the grid.
        horizons: Forecast horizons in native steps.
        architecture: ``tiny_mlp`` or ``linear``.
        hidden_sizes: Hidden widths.
        dropout: Dropout inside the router trunk.
        temperature: Softmax temperature.
        epochs: Maximum router epochs.
        batch_size: Rows per step.
        learning_rate: Adam learning rate.
        weight_decay: Adam weight decay.
        patience: Early-stopping patience in epochs.
        min_delta: Minimum early-stopping improvement.
        max_seconds: Wall-clock cap on training.
        router_train_fraction_of_validation: Chronological share of validation that
            fits the router; the rest selects the epoch.
        ablation_epochs: Optional reduced epoch budget for ablations.
        run_ablations: Whether to run the ablation suite.
        run_crossfit: Whether to run the in-sample label check.
        trace_rows: How many test rows get a full routing trace.
        expert_cache: Path to the expert forecast cache, relative to the project root.
        published_test_mae: Phase 4/5's published test MAE per expert and horizon, for
            the parity check.
    """

    target: str
    label: str
    customer_count: int
    commercial_fraction: float
    seed: int
    torch_threads: int
    lookback_steps: int
    token_stride: int
    train_fraction: float
    validation_fraction: float
    horizons: tuple[int, ...]
    architecture: str
    hidden_sizes: tuple[int, ...]
    dropout: float
    temperature: float
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    patience: int
    min_delta: float
    max_seconds: float
    router_train_fraction_of_validation: float = 0.7
    ablation_epochs: int | None = None
    run_ablations: bool = True
    run_crossfit: bool = True
    trace_rows: int = 2000
    expert_cache: str = "artifacts/phase7/experts.npz"
    published_test_mae: dict[str, dict[str, float]] | None = None

    def router_architecture(self):
        """Build the :class:`~ml.router.model.RouterArchitecture` this config describes."""
        from ..ml.router.model import RouterArchitecture

        return RouterArchitecture(
            name=self.architecture,
            hidden_sizes=tuple(self.hidden_sizes),
            dropout=self.dropout,
            temperature=self.temperature,
        )

    def training_config(self, *, epochs: int | None = None):
        """Build the router's :class:`~ml.router.training.RouterTrainingConfig`."""
        from ..ml.router.training import RouterTrainingConfig

        return RouterTrainingConfig(
            epochs=epochs or self.epochs,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            temperature=self.temperature,
            seed=self.seed,
            threads=self.torch_threads,
            patience=self.patience,
            min_delta=self.min_delta,
            max_seconds=self.max_seconds,
            train_fraction_of_validation=self.router_train_fraction_of_validation,
        )

    def temporal_config(self):
        """The Phase 5 sequence configuration this phase must match, row for row."""
        from ..ml.sequence import SequenceConfig

        return SequenceConfig(
            lookback_steps=self.lookback_steps,
            token_stride=self.token_stride,
            horizons=self.horizons,
            train_origin_stride=1,
            train_fraction=self.train_fraction,
            validation_fraction=self.validation_fraction,
            cyclic_channels=True,
            delta_target=True,
        )

    def temporal_experiment_config(self):
        """A Phase 5 config carrying the same task, for reusing its series loader.

        Phase 7 does not train a temporal model, but it must read the *same* 40 series
        from the *same* sample as Phase 5, and ``load_series_for_experiment`` is the
        function that guarantees that. Building the Phase 5 config here rather than
        reusing one from disk keeps the coupling explicit and typed.
        """
        from .temporal import TemporalExperimentConfig

        return TemporalExperimentConfig(
            target=self.target,
            label=self.label,
            customer_count=self.customer_count,
            commercial_fraction=self.commercial_fraction,
            seed=self.seed,
            torch_threads=self.torch_threads,
            lookback_steps=self.lookback_steps,
            token_stride=self.token_stride,
            train_origin_stride=1,
            train_fraction=self.train_fraction,
            validation_fraction=self.validation_fraction,
            horizons=self.horizons,
            architecture="tcn",
            cyclic_channels=True,
            delta_target=True,
            model_params={},
            epochs=1,
            batch_size=512,
            learning_rate=0.003,
            weight_decay=0.0,
            loss="l1",
            huber_delta=0.05,
            patience=3,
            min_delta=1e-5,
            max_seconds=3600.0,
            grad_clip=1.0,
            validation_stride=1,
            ablation_budget_epochs=1,
            ablation_budget_stride=4,
            ablation_validation_stride=4,
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
            "train_fraction": self.train_fraction,
            "validation_fraction": self.validation_fraction,
            "horizons": list(self.horizons),
            "architecture": self.architecture,
            "hidden_sizes": list(self.hidden_sizes),
            "dropout": self.dropout,
            "temperature": self.temperature,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "patience": self.patience,
            "min_delta": self.min_delta,
            "max_seconds": self.max_seconds,
            "router_train_fraction_of_validation": self.router_train_fraction_of_validation,
            "ablation_epochs": self.ablation_epochs,
            "run_ablations": self.run_ablations,
            "run_crossfit": self.run_crossfit,
            "trace_rows": self.trace_rows,
            "expert_cache": self.expert_cache,
            "published_test_mae": dict(self.published_test_mae or {}),
        }


def load_router_config(path: Path | None = None) -> RouterExperimentConfig:
    """Load and validate ``configs/router.toml``.

    Raises:
        ConfigError: If the file is missing or unreadable.
        ConfigValidationError: If any field is missing, unknown, out of range, or
            redefines the inherited task.
    """
    from .schema import ConfigError, ConfigValidationError

    resolved = Path(path) if path is not None else CONFIG_DIR / DEFAULT_ROUTER_CONFIG
    if not resolved.is_absolute():
        from .loader import PROJECT_ROOT

        resolved = (PROJECT_ROOT / resolved).resolve()
    if not resolved.is_file():
        raise ConfigError(f"router config not found: {resolved}")
    try:
        payload = tomllib.loads(resolved.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{resolved}: invalid TOML: {exc}") from exc

    unknown = sorted(set(payload) - _ALLOWED)
    if unknown:
        raise ConfigValidationError([f"unknown keys: {unknown}"])

    required = {
        "target", "label", "customer_count", "commercial_fraction", "seed",
        "torch_threads", "lookback_steps", "token_stride", "train_fraction",
        "validation_fraction", "horizons", "architecture", "epochs",
        "batch_size", "learning_rate",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ConfigValidationError([f"missing required key: {name}" for name in missing])

    published = payload.get("published_test_mae", {})
    if not isinstance(published, dict):
        raise ConfigValidationError(["published_test_mae must be a table of expert -> horizon -> MAE"])

    ablation_epochs = payload.get("ablation_epochs")
    config = RouterExperimentConfig(
        target=str(payload["target"]),
        label=str(payload["label"]),
        customer_count=int(payload["customer_count"]),
        commercial_fraction=float(payload["commercial_fraction"]),
        seed=int(payload["seed"]),
        torch_threads=int(payload["torch_threads"]),
        lookback_steps=int(payload["lookback_steps"]),
        token_stride=int(payload["token_stride"]),
        train_fraction=float(payload["train_fraction"]),
        validation_fraction=float(payload["validation_fraction"]),
        horizons=tuple(int(h) for h in payload["horizons"]),
        architecture=str(payload["architecture"]),
        hidden_sizes=tuple(int(w) for w in payload.get("hidden_sizes", (32,))),
        dropout=float(payload.get("dropout", 0.0)),
        temperature=float(payload.get("temperature", 1.0)),
        epochs=int(payload["epochs"]),
        batch_size=int(payload["batch_size"]),
        learning_rate=float(payload["learning_rate"]),
        weight_decay=float(payload.get("weight_decay", 0.0)),
        patience=int(payload.get("patience", 5)),
        min_delta=float(payload.get("min_delta", 1e-6)),
        max_seconds=float(payload.get("max_seconds", 900.0)),
        router_train_fraction_of_validation=float(
            payload.get("router_train_fraction_of_validation", 0.7)
        ),
        ablation_epochs=int(ablation_epochs) if ablation_epochs is not None else None,
        run_ablations=bool(payload.get("run_ablations", True)),
        run_crossfit=bool(payload.get("run_crossfit", True)),
        trace_rows=int(payload.get("trace_rows", 2000)),
        expert_cache=str(payload.get("expert_cache", "artifacts/phase7/experts.npz")),
        published_test_mae={
            str(name): {str(horizon): float(value) for horizon, value in entry.items()}
            for name, entry in published.items()
        },
    )

    errors = _validate(config)
    if errors:
        raise ConfigValidationError(errors)
    return config


def _validate(config: RouterExperimentConfig) -> list[str]:
    """Reject any configuration that cannot produce a comparable experiment."""
    from ..ml.router.model import ARCHITECTURE_NAMES

    errors: list[str] = []
    if config.architecture not in ARCHITECTURE_NAMES:
        errors.append(
            f"architecture must be one of {list(ARCHITECTURE_NAMES)}, got "
            f"{config.architecture!r}"
        )
    # The task is inherited, not decided here. These four are what make Phase 7's test
    # numbers comparable with Phase 4's and Phase 5's.
    if config.target != PHASE7_TASK_REFERENCE["target"]:
        errors.append(
            "Phase 7 inherits Phase 5's task: target must be "
            f"{PHASE7_TASK_REFERENCE['target']!r}"
        )
    if config.customer_count != PHASE7_TASK_REFERENCE["customer_count"]:
        errors.append(
            "Phase 7 inherits Phase 5's 40-customer sample; customer_count must be "
            f"{PHASE7_TASK_REFERENCE['customer_count']}"
        )
    if list(config.horizons) != list(PHASE7_TASK_REFERENCE["horizons"]):
        errors.append(
            "Phase 7 inherits Phase 5's horizons "
            f"{list(PHASE7_TASK_REFERENCE['horizons'])}; other horizons would not be "
            "comparable with the Phase 4 and Phase 5 tables"
        )
    if config.lookback_steps != PHASE7_TASK_REFERENCE["lookback_steps"]:
        errors.append(
            "Phase 7 inherits Phase 5's 672-step window so the panel is the same rows"
        )
    if abs(config.train_fraction - float(PHASE7_TASK_REFERENCE["train_fraction"])) > 1e-12:
        errors.append("Phase 7 inherits Phase 5's train_fraction of 0.70")
    if abs(config.validation_fraction - float(PHASE7_TASK_REFERENCE["validation_fraction"])) > 1e-12:
        errors.append("Phase 7 inherits Phase 5's validation_fraction of 0.15")
    if not 0.0 < config.train_fraction < 1.0:
        errors.append("train_fraction must lie in (0, 1)")
    if not 0.0 < config.validation_fraction < 1.0:
        errors.append("validation_fraction must lie in (0, 1)")
    if config.train_fraction + config.validation_fraction >= 1.0:
        errors.append("train_fraction + validation_fraction must leave a test range")
    if config.token_stride <= 0 or config.token_stride >= config.lookback_steps:
        errors.append("token_stride must be positive and shorter than lookback_steps")
    if any(w <= 0 for w in config.hidden_sizes):
        errors.append("hidden_sizes must be positive")
    if not 0.0 <= config.dropout < 1.0:
        errors.append("dropout must lie in [0, 1)")
    if config.temperature <= 0.0:
        errors.append("temperature must be positive")
    if config.epochs <= 0:
        errors.append("epochs must be positive")
    if config.batch_size <= 0:
        errors.append("batch_size must be positive")
    if config.learning_rate <= 0.0:
        errors.append("learning_rate must be positive")
    if config.weight_decay < 0.0:
        errors.append("weight_decay must not be negative")
    if config.patience <= 0:
        errors.append("patience must be positive")
    if config.min_delta < 0.0:
        errors.append("min_delta must not be negative")
    if config.max_seconds <= 0.0:
        errors.append("max_seconds must be positive")
    if not 0.0 < config.router_train_fraction_of_validation < 1.0:
        errors.append("router_train_fraction_of_validation must lie in (0, 1)")
    if config.ablation_epochs is not None and config.ablation_epochs <= 0:
        errors.append("ablation_epochs must be positive when set")
    if config.trace_rows < 0:
        errors.append("trace_rows must not be negative")
    if config.torch_threads <= 0:
        errors.append("torch_threads must be positive")
    return errors