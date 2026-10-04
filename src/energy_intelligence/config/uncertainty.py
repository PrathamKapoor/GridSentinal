"""Phase 8 configuration: the uncertainty experiment shape.

Phase 8 changes the **output** of the system, not the task, so what is inherited from
Phase 7 is everything that decides which rows are evaluated and what is being forecast:
same target, same 40 customers, same seed, same horizons, same window, same split
fractions. The loader validates those rather than trusting them, for the same reason
Phase 7's loader does: a config that moved them would produce a table that looks like a
continuation of Phase 7's and is not comparable with it.

What is decided here, and why each is a Phase 8 choice rather than an inherited one:

``fixed_ensemble_weights``
    Phase 7's fitted weights, copied verbatim. The runner re-reads them from Phase 7's own
    artifact and **stops** if this table disagrees, so the two cannot drift apart quietly.
    This is the mechanism that keeps the point forecast fixed while the uncertainty around
    it is what varies.

``calibration_fit_fraction``
    How much of the validation split fits the scale functions. The rest supplies the
    conformity scores. 0.5 is a choice, and :mod:`uncertainty.split` records the point
    forecast's error on both halves so the choice can be judged from a measurement rather
    than trusted.

``nominal_levels`` / ``band_nominal_level``
    Which coverage levels are reported and which one the operational band is drawn at.

``include_past_error_features``
    Whether the learned scale may use the experts' lagged realised errors. It is the
    strongest single feature group Phase 7 found, so it is switchable and the ablation
    reports what it bought.

Unknown keys are rejected, as in every other config file.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

from .loader import CONFIG_DIR

__all__ = [
    "UncertaintyExperimentConfig",
    "load_uncertainty_config",
    "DEFAULT_UNCERTAINTY_CONFIG",
    "PHASE8_TASK_REFERENCE",
]

DEFAULT_UNCERTAINTY_CONFIG = "uncertainty.toml"

#: The task Phase 8 inherits from Phase 7, restated so a mismatch is caught at load time.
PHASE8_TASK_REFERENCE: dict[str, object] = {
    "target": "customer_load",
    "customer_count": 40,
    "horizons": [1, 4, 96],
    "train_fraction": 0.70,
    "validation_fraction": 0.15,
    "lookback_steps": 672,
}

#: Phase 7's fitted fixed-ensemble weights. Restated here so the config is self-describing
#: and so the runner can prove the two sources agree before it fits anything.
PHASE7_FIXED_ENSEMBLE_WEIGHTS: dict[str, dict[str, float]] = {
    "1": {"persistence": 0.16, "classical_hist_gbm": 0.34, "phase5_tcn": 0.50},
    "4": {"persistence": 0.00, "classical_hist_gbm": 0.34, "phase5_tcn": 0.66},
    "96": {"persistence": 0.00, "classical_hist_gbm": 0.26, "phase5_tcn": 0.74},
}

_ALLOWED = frozenset(
    {
        "target",
        "label",
        "customer_count",
        "seed",
        "torch_threads",
        "lookback_steps",
        "token_stride",
        "train_fraction",
        "validation_fraction",
        "horizons",
        "fixed_ensemble_weights",
        "calibration_fit_fraction",
        "include_past_error_features",
        "nominal_levels",
        "band_nominal_level",
        "scale_max_iter",
        "quantile_max_iter",
        "causality_sample",
        "reliability_bins",
        "trace_rows",
        "model_version",
        "run_ablations",
        "expert_cache",
        "phase7_result",
        "published_test_mae",
    }
)


@dataclass(frozen=True, slots=True)
class UncertaintyExperimentConfig:
    """The full Phase 8 experiment definition.

    Attributes:
        target: Which declared target. Demand only, as in Phases 4, 5 and 7.
        label: Artifact directory under ``artifacts/phase8``.
        customer_count: Customers sampled, matching Phase 5 and Phase 7.
        seed: Estimator seed.
        torch_threads: Torch CPU threads, inherited for a like-for-like budget.
        lookback_steps: History window; must match Phase 5 so the panel is identical.
        token_stride: Read-every-Nth step of the window; must match Phase 5.
        train_fraction: Training share of the grid.
        validation_fraction: Validation share of the grid.
        horizons: Forecast horizons in native steps.
        fixed_ensemble_weights: ``{horizon: {expert: weight}}``, Phase 7's fitted values.
        calibration_fit_fraction: Chronological share of validation that fits the scales.
        include_past_error_features: Whether the learned scale may use lagged expert errors.
        nominal_levels: Coverage levels to report and calibrate.
        band_nominal_level: Level the operational band and calibration status use.
        scale_max_iter: Boosting iterations for the learned scale.
        quantile_max_iter: Boosting iterations per fitted residual quantile.
        causality_sample: Rows the origin-poisoning check verifies.
        reliability_bins: Deciles in the reliability and ranking tables.
        trace_rows: How many test rows get a full uncertainty trace.
        model_version: Identifier recorded on every published forecast.
        run_ablations: Whether to run the scaling ablations.
        expert_cache: Path to the Phase 7 expert forecast cache.
        phase7_result: Path to Phase 7's ``result.json``, read for the weights.
        published_test_mae: Phase 4/5/7's published test MAE per expert and horizon.
    """

    target: str
    label: str
    customer_count: int
    seed: int
    torch_threads: int
    lookback_steps: int
    token_stride: int
    train_fraction: float
    validation_fraction: float
    horizons: tuple[int, ...]
    fixed_ensemble_weights: dict[int, dict[str, float]]
    calibration_fit_fraction: float = 0.5
    include_past_error_features: bool = True
    nominal_levels: tuple[float, ...] = (0.50, 0.80, 0.90, 0.95)
    band_nominal_level: float = 0.90
    scale_max_iter: int = 200
    quantile_max_iter: int = 200
    causality_sample: int = 32
    reliability_bins: int = 10
    trace_rows: int = 2000
    model_version: str = "phase8-uncertainty-v1"
    run_ablations: bool = True
    expert_cache: str = "artifacts/phase7/experts.npz"
    phase7_result: str = "artifacts/phase7/router-main/result.json"
    published_test_mae: dict[str, dict[str, float]] | None = None

    def weight_array(self, expert_names: tuple[str, ...]) -> list[list[float]]:
        """The weights as ``[horizon][expert]``, in the pool's order.

        Args:
            expert_names: The expert pool, in pool order.

        Returns:
            One weight row per horizon.

        Raises:
            ValueError: If an expert is missing from a horizon's weights.
        """
        rows: list[list[float]] = []
        for horizon in self.horizons:
            entry = self.fixed_ensemble_weights[int(horizon)]
            missing = [name for name in expert_names if name not in entry]
            if missing:
                raise ValueError(
                    f"fixed_ensemble_weights for h={horizon} omit {missing}; the expert "
                    f"pool in the cache is authoritative and must match"
                )
            rows.append([float(entry[name]) for name in expert_names])
        return rows

    def temporal_experiment_config(self):
        """A Phase 5 config carrying the same task, for reusing its series loader.

        Phase 8 fits no temporal model, but it must read the *same* 40 series from the
        *same* sample as Phase 7, and ``load_series_for_experiment`` is the function that
        guarantees that. Built here rather than read from disk so the coupling stays typed.
        """
        from .temporal import TemporalExperimentConfig

        return TemporalExperimentConfig(
            target=self.target,
            label=self.label,
            customer_count=self.customer_count,
            commercial_fraction=0.25,
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
            "seed": self.seed,
            "torch_threads": self.torch_threads,
            "lookback_steps": self.lookback_steps,
            "token_stride": self.token_stride,
            "train_fraction": self.train_fraction,
            "validation_fraction": self.validation_fraction,
            "horizons": list(self.horizons),
            "fixed_ensemble_weights": {
                str(horizon): dict(entry)
                for horizon, entry in self.fixed_ensemble_weights.items()
            },
            "calibration_fit_fraction": self.calibration_fit_fraction,
            "include_past_error_features": self.include_past_error_features,
            "nominal_levels": list(self.nominal_levels),
            "band_nominal_level": self.band_nominal_level,
            "scale_max_iter": self.scale_max_iter,
            "quantile_max_iter": self.quantile_max_iter,
            "causality_sample": self.causality_sample,
            "reliability_bins": self.reliability_bins,
            "trace_rows": self.trace_rows,
            "model_version": self.model_version,
            "run_ablations": self.run_ablations,
            "expert_cache": self.expert_cache,
            "phase7_result": self.phase7_result,
            "published_test_mae": dict(self.published_test_mae or {}),
        }


def load_uncertainty_config(path: Path | None = None) -> UncertaintyExperimentConfig:
    """Load and validate ``configs/uncertainty.toml``.

    Args:
        path: Override for the config location. A relative path is resolved against the
            project root.

    Returns:
        The validated configuration.

    Raises:
        ConfigError: If the file is missing or unreadable.
        ConfigValidationError: If any field is missing, unknown, out of range, or redefines
            the inherited task or the fixed-ensemble weights.
    """
    from .schema import ConfigError, ConfigValidationError

    resolved = Path(path) if path is not None else CONFIG_DIR / DEFAULT_UNCERTAINTY_CONFIG
    if not resolved.is_absolute():
        from .loader import PROJECT_ROOT

        resolved = (PROJECT_ROOT / resolved).resolve()
    if not resolved.is_file():
        raise ConfigError(f"uncertainty config not found: {resolved}")
    try:
        payload = tomllib.loads(resolved.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{resolved}: invalid TOML: {exc}") from exc

    unknown = sorted(set(payload) - _ALLOWED)
    if unknown:
        raise ConfigValidationError([f"unknown keys: {unknown}"])

    required = {
        "target", "label", "customer_count", "seed", "torch_threads",
        "lookback_steps", "token_stride", "train_fraction", "validation_fraction",
        "horizons", "fixed_ensemble_weights",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ConfigValidationError([f"missing required key: {name}" for name in missing])

    raw_weights = payload["fixed_ensemble_weights"]
    if not isinstance(raw_weights, dict) or not raw_weights:
        raise ConfigValidationError(
            ["fixed_ensemble_weights must be a non-empty table of horizon -> expert -> weight"]
        )
    weights: dict[int, dict[str, float]] = {}
    for horizon, entry in raw_weights.items():
        if not isinstance(entry, dict):
            raise ConfigValidationError(
                [f"fixed_ensemble_weights[{horizon!r}] must be a table of expert -> weight"]
            )
        weights[int(horizon)] = {
            str(name): float(weight) for name, weight in entry.items()
        }

    published = payload.get("published_test_mae", {})
    if not isinstance(published, dict):
        raise ConfigValidationError(
            ["published_test_mae must be a table of predictor -> horizon -> MAE"]
        )

    config = UncertaintyExperimentConfig(
        target=str(payload["target"]),
        label=str(payload["label"]),
        customer_count=int(payload["customer_count"]),
        seed=int(payload["seed"]),
        torch_threads=int(payload["torch_threads"]),
        lookback_steps=int(payload["lookback_steps"]),
        token_stride=int(payload["token_stride"]),
        train_fraction=float(payload["train_fraction"]),
        validation_fraction=float(payload["validation_fraction"]),
        horizons=tuple(int(h) for h in payload["horizons"]),
        fixed_ensemble_weights=weights,
        calibration_fit_fraction=float(payload.get("calibration_fit_fraction", 0.5)),
        include_past_error_features=bool(payload.get("include_past_error_features", True)),
        nominal_levels=tuple(
            float(level) for level in payload.get("nominal_levels", (0.50, 0.80, 0.90, 0.95))
        ),
        band_nominal_level=float(payload.get("band_nominal_level", 0.90)),
        scale_max_iter=int(payload.get("scale_max_iter", 200)),
        quantile_max_iter=int(payload.get("quantile_max_iter", 200)),
        causality_sample=int(payload.get("causality_sample", 32)),
        reliability_bins=int(payload.get("reliability_bins", 10)),
        trace_rows=int(payload.get("trace_rows", 2000)),
        model_version=str(payload.get("model_version", "phase8-uncertainty-v1")),
        run_ablations=bool(payload.get("run_ablations", True)),
        expert_cache=str(payload.get("expert_cache", "artifacts/phase7/experts.npz")),
        phase7_result=str(
            payload.get("phase7_result", "artifacts/phase7/router-main/result.json")
        ),
        published_test_mae={
            str(name): {str(horizon): float(value) for horizon, value in entry.items()}
            for name, entry in published.items()
        },
    )

    errors = _validate(config)
    if errors:
        raise ConfigValidationError(errors)
    return config


def _validate(config: UncertaintyExperimentConfig) -> list[str]:
    """Reject any configuration that cannot produce a comparable experiment.

    Returns:
        One message per problem. An empty list means the config is usable.
    """
    errors: list[str] = []

    # The task is inherited, not decided here. These are what make Phase 8's rows the same
    # rows Phase 7 evaluated.
    if config.target != PHASE8_TASK_REFERENCE["target"]:
        errors.append(
            "Phase 8 inherits Phase 7's task: target must be "
            f"{PHASE8_TASK_REFERENCE['target']!r}"
        )
    if config.customer_count != PHASE8_TASK_REFERENCE["customer_count"]:
        errors.append(
            "Phase 8 inherits Phase 5's 40-customer sample; customer_count must be "
            f"{PHASE8_TASK_REFERENCE['customer_count']}"
        )
    if list(config.horizons) != list(PHASE8_TASK_REFERENCE["horizons"]):
        errors.append(
            "Phase 8 inherits Phase 5's horizons "
            f"{list(PHASE8_TASK_REFERENCE['horizons'])}; other horizons would not be "
            "comparable with the Phase 4, 5 and 7 tables"
        )
    if config.lookback_steps != PHASE8_TASK_REFERENCE["lookback_steps"]:
        errors.append(
            "Phase 8 inherits Phase 5's 672-step window so the panel is the same rows"
        )
    if abs(config.train_fraction - float(PHASE8_TASK_REFERENCE["train_fraction"])) > 1e-12:
        errors.append("Phase 8 inherits Phase 5's train_fraction of 0.70")
    if abs(
        config.validation_fraction - float(PHASE8_TASK_REFERENCE["validation_fraction"])
    ) > 1e-12:
        errors.append("Phase 8 inherits Phase 5's validation_fraction of 0.15")

    # The point forecast is fixed, so the weights are checked against Phase 7's recorded
    # values here as well as at run time. Catching it at load time means a typo is a config
    # error rather than a failed experiment three minutes in.
    expected = PHASE7_FIXED_ENSEMBLE_WEIGHTS
    for horizon in config.horizons:
        recorded = config.fixed_ensemble_weights.get(int(horizon))
        reference = expected.get(str(horizon))
        if recorded is None:
            errors.append(f"fixed_ensemble_weights has no entry for horizon {horizon}")
            continue
        if reference is None:
            continue
        if set(recorded) != set(reference):
            errors.append(
                f"fixed_ensemble_weights for h={horizon} names {sorted(recorded)}; Phase 7 "
                f"used {sorted(reference)}. The expert pool has changed and Phase 8's point "
                f"forecast would no longer be Phase 7's"
            )
            continue
        total = sum(recorded.values())
        if abs(total - 1.0) > 1e-6:
            errors.append(
                f"fixed_ensemble_weights for h={horizon} sum to {total}, not 1"
            )
        for name, weight in reference.items():
            if abs(recorded[name] - weight) > 1e-9:
                errors.append(
                    f"fixed_ensemble_weights[{horizon}][{name!r}] is {recorded[name]}, but "
                    f"Phase 7 fitted {weight}. Phase 8 does not refit the point forecast, so "
                    f"a different weight here would silently produce a different forecast"
                )

    if not 0.0 < config.calibration_fit_fraction < 1.0:
        errors.append("calibration_fit_fraction must lie in (0, 1)")
    if not config.nominal_levels:
        errors.append("nominal_levels must list at least one coverage level")
    if any(not 0.0 < level < 1.0 for level in config.nominal_levels):
        errors.append(f"every nominal level must lie in (0, 1); got {config.nominal_levels}")
    if len(set(config.nominal_levels)) != len(config.nominal_levels):
        errors.append(f"nominal_levels contains duplicates: {config.nominal_levels}")
    if not any(
        abs(float(config.band_nominal_level) - level) < 1e-12
        for level in config.nominal_levels
    ):
        errors.append(
            f"band_nominal_level {config.band_nominal_level} must be one of the reported "
            f"nominal_levels {config.nominal_levels}: the artifact's calibration status is "
            f"read from the evaluated metrics at that level"
        )
    if config.scale_max_iter <= 0:
        errors.append("scale_max_iter must be positive")
    if config.quantile_max_iter <= 0:
        errors.append("quantile_max_iter must be positive")
    if config.causality_sample <= 0:
        errors.append("causality_sample must be positive")
    if config.reliability_bins < 2:
        errors.append("reliability_bins must be at least 2")
    if config.trace_rows < 0:
        errors.append("trace_rows must not be negative")
    if config.torch_threads <= 0:
        errors.append("torch_threads must be positive")
    if not 0.0 < config.train_fraction < 1.0:
        errors.append("train_fraction must lie in (0, 1)")
    if not 0.0 < config.validation_fraction < 1.0:
        errors.append("validation_fraction must lie in (0, 1)")
    if config.train_fraction + config.validation_fraction >= 1.0:
        errors.append("train_fraction + validation_fraction must leave a test range")
    if config.token_stride <= 0 or config.token_stride >= config.lookback_steps:
        errors.append("token_stride must be positive and shorter than lookback_steps")
    if not config.model_version.strip():
        errors.append("model_version must be a non-empty string; it is recorded on every forecast")
    return errors