"""Phase 6 configuration: the Qwen energy-specialization experiment shape.

Phase 5's ``config/temporal.py`` already established how an experiment is declared
here: strict keys, explicit ablation questions, no silent inheritance. This module
keeps that shape but is a separate file on purpose. Phase 6 must not be able to
quietly inherit Phase 5's architecture answers, and the honest finding of this phase
may well be that those answers were right.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..ml.sequence import SequenceConfig

__all__ = ["QwenExperimentConfig", "load_qwen_config"]

DEFAULT_QWEN_CONFIG = "configs/qwen.toml"

_ALLOWED_KEYS = {
    "target",
    "label",
    "customer_count",
    "commercial_fraction",
    "seed",
    "horizons",
    "train_fraction",
    "validation_fraction",
    "lookback_steps",
    "token_stride",
    "train_origin_stride",
    "cyclic_channels",
    "delta_target",
    "patch_size",
    "aggregate",
    "projector_seed",
    "feature_layers",
    "feature_layer",
    "probe_layers",
    "series_embedding_dim",
    "random_backbone",
    "random_backbone_seed",
    "train_rows",
    "validation_rows",
    "test_rows",
    "batch_size",
    "epochs",
    "learning_rate",
    "weight_decay",
    "loss",
    "patience",
    "min_delta",
    "max_seconds",
    "torch_threads",
    "forward_batch_size",
    "checkpoint_root",
    "basis",
    "basis_hidden_size",
    "ablations",
}

_REQUIRED = {
    "target",
    "label",
    "seed",
    "lookback_steps",
    "token_stride",
    "horizons",
}


@dataclass(frozen=True, slots=True)
class QwenExperimentConfig:
    """The full Phase 6 experiment definition.

    Attributes:
        target: Only demand targets are in scope; Phase 6 does not change the task.
        label: Artifact directory under ``artifacts/phase6``.
        customer_count: Customers sampled, matching Phase 4/5 so the row identities
            match and the Phase 5 checkpoint can be applied to the same rows.
        commercial_fraction: Minority-class share, as in Phase 5.
        seed: Sampling, projector and training seed.
        horizons: Native 15-minute steps. Inherited from Phase 4/5, not redefined.
        train_fraction: Chronological fraction for training origins.
        validation_fraction: Chronological fraction for validation origins.
        lookback_steps: History window in native steps.
        token_stride: Input-window subsampling; targets stay native.
        train_origin_stride: Training origin stride for compute. Validation and test
            origins are not strided, so row identity with Phase 5 is preserved.
        cyclic_channels: Whether the sin/cos time channels are included.
        delta_target: Predict the change from ``y(t)`` rather than the level, as in
            Phase 5. Measured to be worth 253% / 52% / 20% there.
        patch_size: Sampled positions per Qwen token.
        aggregate: How tokens reduce to a feature vector.
        projector_seed: Seed for the fixed random front-end projection.
        feature_layers: Hidden-state indices cached during extraction, so layer
            choice becomes an ablation over one forward pass.
        feature_layer: The layer the shipped configuration probes.
        probe_layers: Depth of the trainable head on top of the frozen features.
            ``[0]`` is the plain linear probe.
        random_backbone: Whether to also extract features from a randomly initialised
            backbone of the same architecture. This is the control that decides
            whether the *pretrained* weights matter at all.
        random_backbone_seed: Seed for that control's initialisation.
        train_rows: Subsample size for training. A subsample is forced by compute:
            measured 0.236 s/sample means the full 237,600 training origins would take
            15.6 hours of pure forward passes.
        validation_rows: Subsample size for validation.
        test_rows: Subsample size for test.
        batch_size: Head training batch size.
        epochs: Maximum head epochs.
        learning_rate: AdamW step size.
        weight_decay: AdamW weight decay.
        loss: ``"l1"`` by default, matching Phase 5's D-075 and the reported metric.
        patience: Early-stopping patience on validation, in epochs.
        min_delta: Minimum validation improvement that resets the patience counter.
        max_seconds: Compute guard on head training.
        torch_threads: Torch CPU threads.
        forward_batch_size: Backbone forward batch size.
        checkpoint_root: Where the verified base checkpoint lives.
        basis: Hidden-state basis used by the probe. ``"raw"`` keeps all 2048
            dimensions; ``"pca"`` compresses to ``basis_hidden_size`` components
            fitted on training rows only.
        basis_hidden_size: Width when ``basis="pca"``.
        ablations: Each entry declares a name, a question and an override.
    """

    target: str
    label: str
    customer_count: int
    commercial_fraction: float
    seed: int
    horizons: tuple[int, ...]
    train_fraction: float
    validation_fraction: float
    lookback_steps: int
    token_stride: int
    train_origin_stride: int
    cyclic_channels: bool
    delta_target: bool
    patch_size: int
    aggregate: str
    projector_seed: int
    feature_layers: tuple[int, ...]
    feature_layer: int
    probe_layers: tuple[int, ...]
    series_embedding_dim: int
    random_backbone: bool
    random_backbone_seed: int
    train_rows: int
    validation_rows: int
    test_rows: int
    batch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float
    loss: str
    patience: int
    min_delta: float
    max_seconds: float
    torch_threads: int
    forward_batch_size: int
    checkpoint_root: str
    basis: str
    basis_hidden_size: int
    ablations: tuple[dict[str, Any], ...] = field(default_factory=tuple)

    def sequence_config(self, *, train_origin_stride: int | None = None) -> SequenceConfig:
        """The Phase 5 sequence contract, so the rows are the same rows."""
        return SequenceConfig(
            lookback_steps=self.lookback_steps,
            token_stride=self.token_stride,
            horizons=self.horizons,
            train_origin_stride=(
                self.train_origin_stride if train_origin_stride is None else train_origin_stride
            ),
            train_fraction=self.train_fraction,
            validation_fraction=self.validation_fraction,
            cyclic_channels=self.cyclic_channels,
            delta_target=self.delta_target,
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for name in self.__slots__:
            value = getattr(self, name)
            payload[name] = list(value) if isinstance(value, tuple) else value
        return payload


def _coerce_int_tuple(value: Any, field_name: str) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field_name} must be a list of integers, got {type(value).__name__}")
    return tuple(int(item) for item in value)


def load_qwen_config(path: str | Path | None = None) -> QwenExperimentConfig:
    """Load and validate the Phase 6 configuration.

    Args:
        path: TOML file; defaults to ``configs/qwen.toml``.

    Returns:
        The validated configuration.

    Raises:
        ConfigValidationError: A required key is missing, an unknown key is present,
            or a value is out of range. Unknown keys are rejected rather than ignored
            (D-007) so a typo cannot silently disable a setting.
    """
    import tomllib

    from .loader import PROJECT_ROOT
    from .schema import ConfigError, ConfigValidationError

    resolved = Path(path) if path is not None else PROJECT_ROOT / DEFAULT_QWEN_CONFIG
    if not resolved.is_absolute():
        resolved = (PROJECT_ROOT / resolved).resolve()
    if not resolved.is_file():
        raise ConfigError(f"qwen config not found: {resolved}")
    try:
        raw = tomllib.loads(resolved.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{resolved}: invalid TOML: {exc}") from exc
    target = resolved

    unknown = sorted(set(raw) - _ALLOWED_KEYS)
    if unknown:
        raise ConfigValidationError(
            [f"unknown key(s) in {target.name}: {unknown}"]
        )
    missing = sorted(_REQUIRED - set(raw))
    if missing:
        raise ConfigValidationError(
            [f"missing required key(s) in {target.name}: {missing}"]
        )

    config = QwenExperimentConfig(
        target=str(raw["target"]),
        label=str(raw["label"]),
        customer_count=int(raw.get("customer_count", 40)),
        commercial_fraction=float(raw.get("commercial_fraction", 0.25)),
        seed=int(raw["seed"]),
        horizons=_coerce_int_tuple(raw["horizons"], "horizons"),
        train_fraction=float(raw.get("train_fraction", 0.70)),
        validation_fraction=float(raw.get("validation_fraction", 0.15)),
        lookback_steps=int(raw["lookback_steps"]),
        token_stride=int(raw.get("token_stride", 4)),
        train_origin_stride=int(raw.get("train_origin_stride", 4)),
        cyclic_channels=bool(raw.get("cyclic_channels", True)),
        delta_target=bool(raw.get("delta_target", True)),
        patch_size=int(raw.get("patch_size", 8)),
        aggregate=str(raw.get("aggregate", "last")),
        projector_seed=int(raw.get("projector_seed", 20260101)),
        feature_layers=_coerce_int_tuple(raw.get("feature_layers", [8, 16, 24, 28]), "feature_layers"),
        feature_layer=int(raw.get("feature_layer", 28)),
        probe_layers=_coerce_int_tuple(raw.get("probe_layers", [0]), "probe_layers"),
        series_embedding_dim=int(raw.get("series_embedding_dim", 8)),
        random_backbone=bool(raw.get("random_backbone", True)),
        random_backbone_seed=int(raw.get("random_backbone_seed", 20260102)),
        train_rows=int(raw.get("train_rows", 10000)),
        validation_rows=int(raw.get("validation_rows", 5000)),
        test_rows=int(raw.get("test_rows", 10000)),
        batch_size=int(raw.get("batch_size", 256)),
        epochs=int(raw.get("epochs", 40)),
        learning_rate=float(raw.get("learning_rate", 0.003)),
        weight_decay=float(raw.get("weight_decay", 0.0)),
        loss=str(raw.get("loss", "l1")),
        patience=int(raw.get("patience", 8)),
        min_delta=float(raw.get("min_delta", 1e-5)),
        max_seconds=float(raw.get("max_seconds", 3600.0)),
        torch_threads=int(raw.get("torch_threads", 16)),
        forward_batch_size=int(raw.get("forward_batch_size", 64)),
        checkpoint_root=str(raw.get("checkpoint_root", "models/qwen3-1.7b-base")),
        basis=str(raw.get("basis", "pca")),
        basis_hidden_size=int(raw.get("basis_hidden_size", 256)),
        ablations=tuple(dict(entry) for entry in raw.get("ablations", ())),
    )

    _validate(config)
    return config


def _validate(config: QwenExperimentConfig) -> None:
    """Collect every problem, then raise once.

    Matching the project's existing convention (D-007): all errors are reported
    together so a user editing the TOML can fix them in one pass, rather than
    discovering them one run at a time.
    """
    from .schema import ConfigValidationError

    errors: list[str] = []

    if config.target != "customer_load":
        errors.append(
            "target: Phase 6 forecasts the same demand target as Phases 4 and 5; "
            f"changing it here would invalidate every comparison (got {config.target!r})"
        )
    # Compared as an ordered tuple, not a sorted list: [4, 1, 96] sorts to the same
    # set but declares a different task ordering, and a permutation that survives
    # validation here would surface later as an unrelated SequenceConfig error.
    if tuple(config.horizons) != (1, 4, 96):
        errors.append(
            "horizons: must be [1, 4, 96] in order, inherited from Phase 4/5 "
            f"(D-081), got {list(config.horizons)}"
        )
    if config.token_stride <= 0 or config.lookback_steps % config.token_stride:
        errors.append(
            f"token_stride: {config.token_stride} must divide lookback_steps "
            f"{config.lookback_steps}"
        )
    else:
        sampled = config.lookback_steps // config.token_stride
        if sampled % config.patch_size:
            errors.append(
                f"patch_size: {config.patch_size} does not divide the {sampled} "
                "sampled positions; the trailing partial patch would be dropped "
                "without notice"
            )
    if config.feature_layer not in config.feature_layers:
        errors.append(
            f"feature_layer: {config.feature_layer} is not among the cached "
            f"feature_layers {list(config.feature_layers)}"
        )
    if config.aggregate not in {"last", "mean"}:
        errors.append(
            f"aggregate: must be 'last' or 'mean', got {config.aggregate!r}"
        )
    if config.basis not in {"raw", "pca"}:
        errors.append(f"basis: must be 'raw' or 'pca', got {config.basis!r}")
    if config.basis == "pca" and not 0 < config.basis_hidden_size < 2048:
        errors.append(
            "basis_hidden_size: must be between 1 and the backbone hidden size "
            f"(2048), got {config.basis_hidden_size}"
        )
    if config.loss != "l1":
        errors.append(
            "loss: Phase 6 keeps Phase 5's L1 objective so the metric and the loss "
            f"agree (D-083), got {config.loss!r}"
        )
    if config.series_embedding_dim < 0:
        errors.append(
            "series_embedding_dim: must be >= 0, got "
            f"{config.series_embedding_dim}"
        )
    for name in ("train_rows", "validation_rows", "test_rows"):
        if getattr(config, name) <= 0:
            errors.append(f"{name}: must be positive, got {getattr(config, name)}")
    for position, entry in enumerate(config.ablations):
        if "name" not in entry or "question" not in entry:
            errors.append(
                f"ablations[{position}]: must declare both a name and the question "
                f"it answers (D-087), got keys {sorted(entry)}"
            )
            continue
        unknown = sorted(set(entry.get("overrides", {})) - _ALLOWED_KEYS)
        if unknown:
            errors.append(
                f"ablations[{entry['name']}]: unknown override(s) {unknown}"
            )
    if errors:
        raise ConfigValidationError(errors)
