"""The Phase 10 protocol freeze: written before official execution, immutable after.

Every number that could otherwise be chosen once results are visible is written here first:
the feature sets and their checksums, the model per target, the folds, the metrics, the seed
policy, the selection rule, its tie-break threshold, and the final-test restrictions.

The file is hashed, and the hash is printed at the start of every run and written into every
result artifact. If the protocol changes after official execution has begun, the hash changes
with it and the change is visible in the result rather than buried in a diff.

Design choices worth naming:

**The model per target is read from evidence, not assumed.** The brief says not to assume the
same classical model suits every target, and that is also what the repository found: Phase 5's
registry records a different winner for load than for PV. The freeze records what was actually
selected.

**The tie-break threshold is frozen at 0.5%** and is applied *after* the absolute comparison,
so a set that is meaningfully worse is never rescued by being simpler.

**Final-test access is enumerated as five separate flags** rather than one, because they are
different acts: training on it, tuning on it, selecting a model on it, selecting features on it
and reporting its performance are five different ways to destroy the benchmark.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .folds import CONFIRMATION_FOLDS, SELECTION_FOLDS
from .sets import FEATURE_SET_IDS, SET_ORDER, checksum, members

__all__ = [
    "PROTOCOL_VERSION",
    "FinalTestPolicy",
    "Phase10Protocol",
    "build_protocol",
    "write_protocol_freeze",
    "protocol_hash",
    "FREEZE_PATH",
    "SELECTION_FREEZE_PATH",
    "write_selection_freeze",
    "load_freeze",
]

#: Bumped whenever the protocol's *content* changes. Not the file hash - that covers content too.
PROTOCOL_VERSION = "phase10-ablation-v1"

FREEZE_PATH = Path("artifacts/experimental_design/phase_10_ablation_protocol_freeze.yaml")
SELECTION_FREEZE_PATH = Path(
    "artifacts/experimental_design/phase_10_selected_feature_freeze.yaml"
)

#: Relative MAE difference at or below which the simpler feature set wins. Frozen before
#: results are visible, and never revisited.
TIE_BREAK_RELATIVE_TOLERANCE = 0.005

#: The secondary robustness model. Its configuration is frozen and reused verbatim.
SECONDARY_MODEL = "neural_mlp"
SECONDARY_MODEL_PUBLIC_NAME = "PYTORCH_MLP_V1"


@dataclass(frozen=True, slots=True)
class FinalTestPolicy:
    """The five ways Phase 10 could reach the final test split, all refused."""

    training_access: bool = False
    hpo_access: bool = False
    model_selection_access: bool = False
    feature_selection_access: bool = False
    performance_evaluation: bool = False
    integrity_audit_access: str = "historical P9-DEV-002 only"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["state"] = "LOCKED"
        return payload

    def violations(self) -> list[str]:
        """Which access flags are set. Any entry here is a protocol breach.

        The integrity-audit field is a documented allowance rather than a permission, so it is
        reported rather than counted - it is the one field whose value is a string.

        Returns:
            Names of every boolean access flag that is ``True``.
        """
        out: list[str] = []
        for name, value in asdict(self).items():
            if isinstance(value, bool) and value:
                out.append(name)
        return out


@dataclass(frozen=True, slots=True)
class Phase10Protocol:
    """Everything frozen before official execution."""

    protocol_version: str
    generated_for: str
    targets: tuple[str, ...]
    unsupported_targets: dict[str, str]
    horizon_steps: int
    horizon_label: str
    steps_per_day: int
    feature_sets: dict[str, list[str]]
    feature_set_checksums: dict[str, str]
    primary_model_by_target: dict[str, str]
    primary_model_evidence: dict[str, str]
    primary_model_config_checksums: dict[str, str]
    secondary_model: str
    secondary_model_public_name: str
    secondary_model_config_checksum: str
    selection_folds: tuple[str, ...]
    confirmation_folds: tuple[str, ...]
    metrics: tuple[str, ...]
    selection_metric: str
    tie_break: dict[str, Any]
    seed_policy: dict[str, Any]
    common_sample_policy: dict[str, Any]
    fixed_between_feature_sets: tuple[str, ...]
    prohibited: tuple[str, ...]
    final_test: FinalTestPolicy
    notes: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["targets"] = list(self.targets)
        payload["selection_folds"] = list(self.selection_folds)
        payload["confirmation_folds"] = list(self.confirmation_folds)
        payload["metrics"] = list(self.metrics)
        payload["fixed_between_feature_sets"] = list(self.fixed_between_feature_sets)
        payload["prohibited"] = list(self.prohibited)
        payload["notes"] = list(self.notes)
        payload["final_test"] = self.final_test.to_dict()
        return payload


def _dump(payload: dict[str, Any]) -> str:
    """Deterministic YAML for a freeze file.

    ``sort_keys`` and an explicit block width make the bytes a function of the content alone,
    which is what lets the freeze carry a meaningful hash. A stable emitter matters more here
    than pretty output: the hash is quoted in every result artifact.
    """
    import yaml

    return yaml.safe_dump(
        payload,
        sort_keys=True,
        default_flow_style=False,
        width=100,
        allow_unicode=True,
    )


def _checksum_of(payload: Any) -> str:
    """A stable full-length sha256 of any JSON-serialisable payload."""
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def protocol_hash(protocol: Phase10Protocol) -> str:
    """The freeze's content hash, recorded in every result artifact."""
    return _checksum_of(protocol.to_dict())


def build_protocol(
    *,
    targets: tuple[str, ...],
    unsupported_targets: dict[str, str],
    horizon_steps: int,
    primary_model_by_target: dict[str, str],
    primary_model_evidence: dict[str, str],
    model_config: dict[str, Any],
    secondary_model_config: dict[str, Any],
    seed: int,
    steps_per_day: int = 96,
    generated_for: str = "gridsentral-phase-10",
) -> Phase10Protocol:
    """Assemble the frozen protocol.

    Args:
        targets: Targets with usable data.
        unsupported_targets: ``{target: why}`` for those without, so the freeze records the
            exclusions rather than leaving them implicit.
        horizon_steps: Primary horizon in native steps.
        primary_model_by_target: ``{target: model_id}`` chosen from measured evidence.
        primary_model_evidence: Where that choice is recorded.
        model_config: The frozen model configuration, checksummed per target.
        secondary_model_config: The frozen secondary configuration.
        seed: The single seed used everywhere.
        steps_per_day: Native steps per day.
        generated_for: Provenance string.

    Returns:
        The protocol.
    """
    horizon_label = f"H{int(round(horizon_steps * 15 / 60))}"
    return Phase10Protocol(
        protocol_version=PROTOCOL_VERSION,
        generated_for=generated_for,
        targets=tuple(targets),
        unsupported_targets=dict(unsupported_targets),
        horizon_steps=int(horizon_steps),
        horizon_label=horizon_label,
        steps_per_day=int(steps_per_day),
        feature_sets={set_id: list(members(set_id)) for set_id in SET_ORDER},
        feature_set_checksums={set_id: checksum(set_id) for set_id in SET_ORDER},
        primary_model_by_target=dict(primary_model_by_target),
        primary_model_evidence=dict(primary_model_evidence),
        primary_model_config_checksums={
            target: _checksum_of(model_config) for target in targets
        },
        secondary_model=SECONDARY_MODEL,
        secondary_model_public_name=SECONDARY_MODEL_PUBLIC_NAME,
        secondary_model_config_checksum=_checksum_of(secondary_model_config),
        selection_folds=SELECTION_FOLDS,
        confirmation_folds=CONFIRMATION_FOLDS,
        metrics=("mae", "rmse"),
        selection_metric="mae",
        tie_break={
            "rule": "prefer fewer features",
            "relative_mae_tolerance": TIE_BREAK_RELATIVE_TOLERANCE,
            "applied_when": (
                "the relative MAE difference between the best set and a simpler set is at or "
                "below the tolerance"
            ),
            "frozen_before_results": True,
            "order": "compare absolute MAE first, then apply the tolerance to the simpler sets",
        },
        seed_policy={
            "seed": int(seed),
            "same_seed_for_every_feature_set": True,
            "rationale": (
                "one seed, held fixed across feature sets, so the only difference between "
                "two rows of the ablation is the feature set"
            ),
            "replicates": 1,
            "replicate_note": (
                "a single seed is a limitation: a difference smaller than seed-to-seed "
                "variation cannot be distinguished from it, and Phase 10 does not run formal "
                "significance testing"
            ),
        },
        common_sample_policy={
            "enabled": True,
            "method": "intersect usable (origin, series) identity across feature sets",
            "rationale": (
                "larger feature sets read longer histories and so have fewer usable origins; "
                "scoring each set on whatever rows it has would compare models on different "
                "row counts"
            ),
            "record_fields": [
                "target",
                "fold",
                "feature_set",
                "sample_count",
                "timestamp_identity",
            ],
        },
        fixed_between_feature_sets=(
            "model implementation",
            "model hyperparameters",
            "training procedure",
            "feature scaling policy",
            "seed",
            "forecast horizon",
            "temporal folds",
            "metrics",
            "target definition",
            "timestamp policy",
        ),
        prohibited=(
            "Optuna",
            "GridSearch",
            "RandomSearch",
            "feature-specific hyperparameter tuning",
            "feature-specific model changes",
            "new feature families",
            "weather features",
            "final-test access of any kind",
        ),
        final_test=FinalTestPolicy(),
        notes=(
            "Phase 10 is an ablation study, not feature engineering: every feature name is an "
            "existing repository FeatureSpec and no new feature is introduced.",
            "H24 is horizon step 96 at the native 15-minute resolution.",
            "A feature family's effect is an incremental validation-performance difference, "
            "never a causal effect.",
            "F01-F04 select. F05-F06 confirm after selection is frozen. The final test split "
            "is read by no Phase 10 code path.",
            "Single-seed results cannot separate a real feature effect from seed noise; "
            "formal significance testing is explicitly out of scope for this phase.",
        ),
    )


def write_protocol_freeze(
    protocol: Phase10Protocol, root: Path, *, overwrite: bool = False
) -> tuple[Path, str]:
    """Write the freeze YAML and return its path and content hash.

    Args:
        protocol: The frozen protocol.
        root: Project root.
        overwrite: Permit rewriting an existing freeze. False by default, so a re-run cannot
            silently replace the protocol it is supposed to be obeying.

    Returns:
        ``(path, content_hash)``.

    Raises:
        FileExistsError: If the freeze exists and ``overwrite`` is False.
    """
    path = Path(root) / FREEZE_PATH
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"{FREEZE_PATH} already exists. The Phase 10 protocol is frozen before official "
            f"execution and must not change once results are visible; delete it explicitly "
            f"if a re-freeze is genuinely intended."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_dump(protocol.to_dict()), encoding="utf-8")
    return path, protocol_hash(protocol)


def write_selection_freeze(
    payload: dict[str, Any], root: Path, *, overwrite: bool = False
) -> Path:
    """Write the post-selection freeze.

    Args:
        payload: Target, model, selected set, F01-F04 MAE, rationale, checksums.
        root: Project root.
        overwrite: Permit rewriting.

    Returns:
        The path written.

    Raises:
        FileExistsError: If it exists and ``overwrite`` is False.
    """
    path = Path(root) / SELECTION_FREEZE_PATH
    if path.exists() and not overwrite:
        raise FileExistsError(f"{SELECTION_FREEZE_PATH} already exists and is frozen")
    path.parent.mkdir(parents=True, exist_ok=True)
    body = dict(payload)
    body["freeze_hash"] = _checksum_of(payload)
    path.write_text(_dump(body), encoding="utf-8")
    return path


def load_freeze(path: Path) -> dict[str, Any]:
    """Read a freeze file back as a plain dictionary.

    Args:
        path: The freeze file.

    Returns:
        The parsed mapping.

    Raises:
        ValueError: If the file is missing or is not a YAML mapping.
    """
    import yaml

    if not Path(path).is_file():
        raise ValueError(f"freeze not found: {path}")
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} did not parse to a mapping")
    return payload
