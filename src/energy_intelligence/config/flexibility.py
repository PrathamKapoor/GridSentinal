"""Phase 9 configuration: the flexibility experiment shape.

Phase 9 changes the **output** of the system again, not the task, so everything that
decides which rows are evaluated is inherited from Phase 5 and validated here rather than
trusted: same target, same 40 customers, same horizons, same window, same split fractions.
A config that moved any of them would produce a table that looked like a continuation of
Phase 8's and was not comparable with it.

What is decided here, and why each is a Phase 9 choice rather than an inherited one:

``nominal_level``
    The coverage the behavioural envelope claims. Phase 8 reported a ladder of levels;
    Phase 9 publishes one, because an envelope is a claim a planner would act on and
    publishing four of them invites picking whichever is convenient after the fact.

``granularity``
    The calendar key the baseline is conditioned on - one value for all customers pooled,
    "time of day", or "day type and time of day". This is a default, not a commitment: the
    runner fits every granularity and *selects* on the conformity split, so this field only
    needs to be a legal starting point. It is reported so a reader can tell what a run would
    have chosen before it chose.

``min_samples_per_slot``
    How many calibration observations a slot needs before its median is trusted. Below the
    threshold the run falls back to the per-customer median rather than publishing an
    estimate from three points, and reports what share of rows fell back.

``reliance_*``
    The uncertainty-to-flexibility coupling: how far back the trailing reference is drawn
    from, how many rows it needs, and the floor the discount bottoms out at. These only size
    a counterfactual in Phase 9, because the coupling is measured and then withheld - but
    they are configuration rather than constants so that a later phase which does apply it
    has an explicit, reviewable setting.

``uncertainty_artifact``
    Which Phase 8 intervals to consume. Reading another phase's published artifact rather
    than refitting is what makes Phase 9's numbers describe the forecast Phase 8 actually
    measured.

Unknown keys are rejected, as in every other config file.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

from .loader import CONFIG_DIR

__all__ = [
    "FlexibilityExperimentConfig",
    "load_flexibility_config",
    "DEFAULT_FLEXIBILITY_CONFIG",
    "PHASE9_TASK_REFERENCE",
    "SLOT_GRANULARITIES",
]

DEFAULT_FLEXIBILITY_CONFIG = "flexibility.toml"

#: The calendar keys a baseline may be conditioned on. Mirrors
#: ``energy_intelligence.ml.flexibility.demand.SLOT_GRANULARITIES``; restated here so the
#: config can reject an illegal granularity without importing the ML package.
SLOT_GRANULARITIES: tuple[str, ...] = ("horizon", "time_of_day", "day_type_time_of_day")

#: The task Phase 9 inherits from Phases 5, 7 and 8, restated so a mismatch is caught at
#: load time rather than producing an incomparable table.
PHASE9_TASK_REFERENCE: dict[str, object] = {
    "target": "customer_load",
    "customer_count": 40,
    "horizons": [1, 4, 96],
    "train_fraction": 0.70,
    "validation_fraction": 0.15,
    "lookback_steps": 672,
    "token_stride": 4,
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
        "granularity",
        "nominal_level",
        "calibration_fit_fraction",
        "min_samples_per_slot",
        "reliability_bins",
        "reliance_floor",
        "reliance_window_rows",
        "reliance_min_periods",
        "model_version",
        "model_family",
        "expert_cache",
        "uncertainty_artifact",
        "phase7_result",
        "published_test_mae",
    }
)


@dataclass(frozen=True, slots=True)
class FlexibilityExperimentConfig:
    """The full Phase 9 experiment definition.

    Attributes:
        target: Which declared target. Demand only, as in Phases 4, 5, 7 and 8.
        label: Artifact directory under ``artifacts/phase9``.
        customer_count: Customers sampled, matching Phase 5, 7 and 8.
        seed: Recorded for reproducibility. Phase 9 draws no random numbers - calendar
            medians and empirical quantiles are deterministic - so this is carried for
            provenance and not passed to an estimator.
        torch_threads: Torch CPU threads, inherited so the budget is comparable.
        lookback_steps: History window; must match Phase 5 so the panel is identical.
        token_stride: Read-every-Nth step of the window; must match Phase 5.
        train_fraction: Training share of the grid; inherited, unused by Phase 9's fit.
        validation_fraction: Validation share; inherited, and the only split Phase 9 reads.
        horizons: Forecast horizons, inherited.
        granularity: Default calendar key, a legal value from
            :data:`SLOT_GRANULARITIES`. Selection overrides it.
        nominal_level: The coverage the envelope claims.
        calibration_fit_fraction: Share of the validation split that fits the baseline and
            envelope; the rest selects.
        min_samples_per_slot: Observations a slot needs before its median is trusted.
        reliability_bins: Bins for the width-versus-deviation reliability table.
        reliance_floor: The discount's lower bound.
        reliance_window_rows: Trailing window for the causal reference.
        reliance_min_periods: Rows the trailing window needs before it is used.
        model_version: Recorded on the published envelope.
        model_family: Recorded on the registry record.
        expert_cache: Phase 7's forecast cache, read for panel geometry.
        uncertainty_artifact: Phase 8's probabilistic forecast artifact, read for widths.
        phase7_result: Phase 7's result, read for the published MAE parity check.
        published_test_mae: Phase 7's published ``{model: {horizon: MAE}}``.
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
    granularity: str
    nominal_level: float
    calibration_fit_fraction: float
    min_samples_per_slot: int
    reliability_bins: int
    reliance_floor: float
    reliance_window_rows: int
    reliance_min_periods: int
    model_version: str
    model_family: str
    expert_cache: str
    uncertainty_artifact: str
    phase7_result: str
    published_test_mae: dict[str, dict[str, float]]

    @property
    def artifact_dir(self) -> str:
        return f"artifacts/phase9/{self.label}"

    def to_dict(self) -> dict[str, object]:
        """The config as plain data, for ``flexibility config`` and for the record.

        The reported-only field is included and labelled, so a reader of the printed config
        can see that ``granularity`` is a starting point the runner overrides by
        measurement rather than a setting that reaches the published envelope.
        """
        return {
            "target": self.target,
            "label": self.label,
            "artifact_dir": self.artifact_dir,
            "customer_count": self.customer_count,
            "seed": self.seed,
            "torch_threads": self.torch_threads,
            "lookback_steps": self.lookback_steps,
            "token_stride": self.token_stride,
            "train_fraction": self.train_fraction,
            "validation_fraction": self.validation_fraction,
            "horizons": list(self.horizons),
            "granularity_default": self.granularity,
            "granularity_is_reported_only": True,
            "nominal_level": self.nominal_level,
            "calibration_fit_fraction": self.calibration_fit_fraction,
            "min_samples_per_slot": self.min_samples_per_slot,
            "reliability_bins": self.reliability_bins,
            "reliance_floor": self.reliance_floor,
            "reliance_window_rows": self.reliance_window_rows,
            "reliance_min_periods": self.reliance_min_periods,
            "model_version": self.model_version,
            "model_family": self.model_family,
            "expert_cache": self.expert_cache,
            "uncertainty_artifact": self.uncertainty_artifact,
            "phase7_result": self.phase7_result,
            "published_test_mae": self.published_test_mae,
            "claims": {
                "physical_flexibility": "audited, never assumed",
                "dispatchable_capability": "never claimed from observational data",
                "reliance_coupling": "measured, then withheld from the published envelope",
            },
        }

    def temporal_experiment_config(self):
        """A Phase 5 config carrying the same task, for reusing its series loader.

        Phase 9 fits no temporal model, but it must read the *same* 40 series from the same
        sample as Phases 5, 7 and 8 - the demand reconstruction it verifies against the
        panel is only meaningful on the same rows - and ``load_series_for_experiment`` is
        the function that guarantees that. Built here rather than read from disk so the
        coupling stays typed.

        The field values are deliberately identical to Phase 8's equivalent method.
        ``tests/config/test_flexibility_config.py`` asserts the two produce the same
        temporal config, so a change to one that is not made to the other fails the suite
        rather than silently producing a differently-sampled panel.
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


def _validate(config: FlexibilityExperimentConfig) -> list[str]:
    """Every complaint about the config, collected rather than raised on the first.

    Returns:
        The messages. Empty means valid.
    """
    errors: list[str] = []
    if config.target != PHASE9_TASK_REFERENCE["target"]:
        errors.append(
            f"target is {config.target!r}; Phase 9 inherits "
            f"{PHASE9_TASK_REFERENCE['target']!r} from Phases 4/5/7/8"
        )
    if config.customer_count != PHASE9_TASK_REFERENCE["customer_count"]:
        errors.append(
            f"customer_count is {config.customer_count}; Phase 9 inherits "
            f"{PHASE9_TASK_REFERENCE['customer_count']}"
        )
    if tuple(int(h) for h in config.horizons) != tuple(
        int(h) for h in PHASE9_TASK_REFERENCE["horizons"]
    ):
        errors.append(
            f"horizons are {list(config.horizons)}; Phase 9 inherits "
            f"{list(PHASE9_TASK_REFERENCE['horizons'])}. Phase 8's artifact holds one "
            f"interval column per horizon, so a different set would mis-pair the widths"
        )
    if config.lookback_steps != PHASE9_TASK_REFERENCE["lookback_steps"]:
        errors.append(
            "lookback_steps must match Phase 5's 672 so the evaluated rows are the same rows"
        )
    if config.token_stride != PHASE9_TASK_REFERENCE["token_stride"]:
        errors.append("token_stride must match Phase 5's value")
    if abs(config.train_fraction - float(PHASE9_TASK_REFERENCE["train_fraction"])) > 1e-12:
        errors.append("Phase 9 inherits Phase 5's train_fraction of 0.70")
    if abs(
        config.validation_fraction - float(PHASE9_TASK_REFERENCE["validation_fraction"])
    ) > 1e-12:
        errors.append("Phase 9 inherits Phase 5's validation_fraction of 0.15")
    if config.train_fraction + config.validation_fraction >= 1.0:
        errors.append("train_fraction + validation_fraction must leave a test range")

    if config.granularity not in SLOT_GRANULARITIES:
        errors.append(
            f"granularity is {config.granularity!r}; it must be one of "
            f"{list(SLOT_GRANULARITIES)}"
        )
    if not 0.0 < config.nominal_level < 1.0:
        errors.append(f"nominal_level must lie in (0, 1); got {config.nominal_level}")
    if not 0.0 < config.calibration_fit_fraction < 1.0:
        errors.append("calibration_fit_fraction must lie in (0, 1)")
    if config.min_samples_per_slot <= 0:
        errors.append("min_samples_per_slot must be positive")
    if config.reliability_bins < 2:
        errors.append("reliability_bins must be at least 2")
    if not 0.0 < config.reliance_floor <= 1.0:
        errors.append(
            f"reliance_floor must lie in (0, 1]; got {config.reliance_floor}. A floor of 0 "
            f"would let a wide interval imply zero flexibility, which is a claim this "
            f"phase has no evidence for"
        )
    if config.reliance_window_rows <= 0:
        errors.append("reliance_window_rows must be positive")
    if config.reliance_min_periods <= 0:
        errors.append("reliance_min_periods must be positive")
    if config.reliance_min_periods > config.reliance_window_rows:
        errors.append(
            f"reliance_min_periods ({config.reliance_min_periods}) exceeds "
            f"reliance_window_rows ({config.reliance_window_rows}), so the trailing "
            f"reference would never be populated"
        )
    if config.torch_threads <= 0:
        errors.append("torch_threads must be positive")
    if not config.model_version.strip():
        errors.append("model_version must be a non-empty string; it is recorded on the envelope")
    if not config.label.strip():
        errors.append("label must be non-empty; it names the artifact directory")
    if not config.uncertainty_artifact.strip():
        errors.append(
            "uncertainty_artifact must name Phase 8's artifact; Phase 9 measures its "
            "coupling against intervals it must read, not intervals it refits"
        )
    if not config.published_test_mae:
        errors.append(
            "published_test_mae must carry Phase 7's fixed_ensemble figures; without them "
            "Phase 9 cannot prove the point forecast its envelopes sit around is Phase 7's"
        )
    return errors


def load_flexibility_config(path: Path | None = None) -> FlexibilityExperimentConfig:
    """Load and validate ``configs/flexibility.toml``.

    Args:
        path: Override for the config location. A relative path is resolved against the
            project root.

    Returns:
        The validated configuration.

    Raises:
        ConfigError: If the file is missing or unreadable.
        ConfigValidationError: If any field is missing, unknown, out of range, or redefines
            the inherited task.
    """
    from .schema import ConfigError, ConfigValidationError

    resolved = Path(path) if path is not None else CONFIG_DIR / DEFAULT_FLEXIBILITY_CONFIG
    if not resolved.is_absolute():
        from .loader import PROJECT_ROOT

        resolved = (PROJECT_ROOT / resolved).resolve()
    if not resolved.is_file():
        raise ConfigError(f"flexibility config not found: {resolved}")
    try:
        payload = tomllib.loads(resolved.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{resolved}: invalid TOML: {exc}") from exc

    unknown = sorted(set(payload) - _ALLOWED)
    if unknown:
        raise ConfigValidationError(
            [f"unknown key(s) in {resolved.name}: {unknown}", f"allowed: {sorted(_ALLOWED)}"]
        )
    required = _ALLOWED - {
        "granularity",
        "nominal_level",
        "calibration_fit_fraction",
        "min_samples_per_slot",
        "reliability_bins",
        "reliance_floor",
        "reliance_window_rows",
        "reliance_min_periods",
        "model_version",
        "model_family",
        "expert_cache",
        "uncertainty_artifact",
        "phase7_result",
        "published_test_mae",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ConfigValidationError(
            [f"missing key(s) in {resolved.name}: {missing}"]
        )

    published = payload.get("published_test_mae", {})
    if not isinstance(published, dict):
        raise ConfigValidationError(
            ["published_test_mae must be a table of predictor -> horizon -> MAE"]
        )

    config = FlexibilityExperimentConfig(
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
        granularity=str(payload.get("granularity", "day_type_time_of_day")),
        nominal_level=float(payload.get("nominal_level", 0.90)),
        calibration_fit_fraction=float(payload.get("calibration_fit_fraction", 0.5)),
        min_samples_per_slot=int(payload.get("min_samples_per_slot", 5)),
        reliability_bins=int(payload.get("reliability_bins", 10)),
        reliance_floor=float(payload.get("reliance_floor", 0.25)),
        reliance_window_rows=int(payload.get("reliance_window_rows", 3840)),
        reliance_min_periods=int(payload.get("reliance_min_periods", 96)),
        model_version=str(payload.get("model_version", "phase9-flexibility-v1")),
        model_family=str(payload.get("model_family", "behavioural_response_envelope")),
        expert_cache=str(payload.get("expert_cache", "artifacts/phase7/experts.npz")),
        uncertainty_artifact=str(
            payload.get(
                "uncertainty_artifact",
                "artifacts/phase8/uncertainty-main/probabilistic_forecasts.npz",
            )
        ),
        phase7_result=str(
            payload.get("phase7_result", "artifacts/phase7/router-main/result.json")
        ),
        published_test_mae={
            str(name): {str(horizon): float(value) for horizon, value in entry.items()}
            for name, entry in published.items()
            if isinstance(entry, dict)
        },
    )

    errors = _validate(config)
    if errors:
        raise ConfigValidationError(errors)
    return config