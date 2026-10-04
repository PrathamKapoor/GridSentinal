"""Phase 9 configuration tests.

The config is the only place a reader checks what the phase was told to do, so the tests
here are about **refusing** rather than accepting: a config that quietly moved the inherited
task, the artifact it reads, or the claims it is allowed to make has to fail at load time,
not produce an incomparable table three minutes into a run.

The temporal-config parity test is the one that matters most. Phase 9 fits no temporal model
but must read the *same* 40 series from the *same* sample as Phases 5, 7 and 8, and it does
that by borrowing Phase 8's ``TemporalExperimentConfig`` bridge. Two hand-maintained copies
of a 30-field constructor is two things to forget to update, so the test asserts they produce
identical sampling parameters - which is the invariant the duplication exists to serve.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pytest

from energy_intelligence.config.flexibility import (
    DEFAULT_FLEXIBILITY_CONFIG,
    PHASE9_TASK_REFERENCE,
    SLOT_GRANULARITIES,
    FlexibilityExperimentConfig,
    load_flexibility_config,
)
from energy_intelligence.config.uncertainty import load_uncertainty_config
from energy_intelligence.config.schema import ConfigError, ConfigValidationError

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _write(tmp_path: Path, **overrides: object) -> Path:
    """A minimal valid config file with ``overrides`` applied.

    Written from an explicit template rather than by serialising a dataclass: ``repr`` of a
    string yields single quotes, which TOML does not accept, and a test helper that emits
    subtly invalid TOML tests the wrong thing.
    """
    base = load_flexibility_config()
    values = {
        "target": base.target,
        "label": base.label,
        "customer_count": base.customer_count,
        "seed": base.seed,
        "torch_threads": base.torch_threads,
        "lookback_steps": base.lookback_steps,
        "token_stride": base.token_stride,
        "train_fraction": base.train_fraction,
        "validation_fraction": base.validation_fraction,
        "horizons": list(base.horizons),
        "granularity": base.granularity,
        "nominal_level": base.nominal_level,
        "calibration_fit_fraction": base.calibration_fit_fraction,
        "min_samples_per_slot": base.min_samples_per_slot,
        "reliability_bins": base.reliability_bins,
        "reliance_floor": base.reliance_floor,
        "reliance_window_rows": base.reliance_window_rows,
        "reliance_min_periods": base.reliance_min_periods,
        "model_version": base.model_version,
        "model_family": base.model_family,
        "expert_cache": base.expert_cache,
        "uncertainty_artifact": base.uncertainty_artifact,
        "phase7_result": base.phase7_result,
        "published_test_mae": base.published_test_mae["fixed_ensemble"],
    }
    values.update(overrides)
    table = values["published_test_mae"]
    inner = ", ".join(f'"{key}" = {value!r}' for key, value in table.items())
    body = f'''
target = "{values["target"]}"
label = "{values["label"]}"
customer_count = {values["customer_count"]}
seed = {values["seed"]}
torch_threads = {values["torch_threads"]}
lookback_steps = {values["lookback_steps"]}
token_stride = {values["token_stride"]}
train_fraction = {values["train_fraction"]!r}
validation_fraction = {values["validation_fraction"]!r}
horizons = {values["horizons"]}
granularity = "{values["granularity"]}"
nominal_level = {values["nominal_level"]!r}
calibration_fit_fraction = {values["calibration_fit_fraction"]!r}
min_samples_per_slot = {values["min_samples_per_slot"]}
reliability_bins = {values["reliability_bins"]}
reliance_floor = {values["reliance_floor"]!r}
reliance_window_rows = {values["reliance_window_rows"]}
reliance_min_periods = {values["reliance_min_periods"]}
model_version = "{values["model_version"]}"
model_family = "{values["model_family"]}"
expert_cache = "{values["expert_cache"]}"
uncertainty_artifact = "{values["uncertainty_artifact"]}"
phase7_result = "{values["phase7_result"]}"

[published_test_mae]
fixed_ensemble = {{ {inner} }}
'''
    path = tmp_path / "flexibility.toml"
    path.write_text(body.lstrip(), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# The shipped config
# ---------------------------------------------------------------------------


def test_shipped_config_loads_and_validates() -> None:
    config = load_flexibility_config()
    assert config.artifact_dir.startswith("artifacts/phase9/")
    assert config.nominal_level == 0.90, "matches Phase 8's published level, so the two compare"
    assert config.uncertainty_artifact.endswith(".npz")


def test_shipped_config_records_phase7_mae_verbatim() -> None:
    """Phase 9's parity check is worthless if the reference was retyped by hand.

    These are Phase 7's published fixed-ensemble figures, and the run refuses to start if the
    rebuilt ensemble does not reproduce them, so a drift here surfaces as a hard failure
    rather than as a subtly wrong envelope.
    """
    config = load_flexibility_config()
    published = config.published_test_mae["fixed_ensemble"]
    assert published["1"] == pytest.approx(0.3983380494202863)
    assert published["4"] == pytest.approx(0.7917144666626339)
    assert published["96"] == pytest.approx(1.5399040140142268)


def test_shipped_config_makes_no_dispatchable_claim() -> None:
    """The printed config is the first thing a reader checks; it must not imply capability."""
    payload = load_flexibility_config().to_dict()
    assert payload["claims"]["dispatchable_capability"] == (
        "never claimed from observational data"
    )
    assert payload["granularity_is_reported_only"] is True


def test_temporal_config_matches_phase8_sampling_exactly() -> None:
    """Phase 9 must read the same series from the same sample as Phases 5, 7 and 8.

    ``label`` is expected to differ - it names a different artifact directory - and every
    other field must not, because that is what guarantees the panel is the same rows.
    """
    phase9 = asdict(load_flexibility_config().temporal_experiment_config())
    phase8 = asdict(load_uncertainty_config().temporal_experiment_config())
    differences = {
        key: (phase9[key], phase8[key]) for key in phase9 if phase9[key] != phase8[key]
    }
    assert set(differences) == {"label"}, differences


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


def test_unknown_key_is_rejected(tmp_path: Path) -> None:
    path = _write(tmp_path)
    text = path.read_text(encoding="utf-8")
    # Inserted above the table header: a bare key after `[published_test_mae]` would belong
    # to that table and fail as a TOML syntax error rather than as an unknown-key rejection.
    text = text.replace("granularity =", "not_a_real_key = 1\ngranularity =", 1)
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigValidationError, match="unknown key"):
        load_flexibility_config(path)


def test_missing_required_key_is_rejected(tmp_path: Path) -> None:
    path = _write(tmp_path)
    path.write_text(
        "\n".join(
            line
            for line in path.read_text(encoding="utf-8").splitlines()
            if not line.startswith("customer_count")
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigValidationError, match="missing key"):
        load_flexibility_config(path)


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("customer_count", 41, "inherits"),
        ("lookback_steps", 512, "672"),
        ("train_fraction", 0.60, "train_fraction"),
        ("horizons", [1, 4], "horizons"),
    ],
)
def test_moving_the_inherited_task_is_rejected(
    tmp_path: Path, field: str, value: object, expected: str
) -> None:
    """A config that redefines the inherited task would produce an incomparable table."""
    path = _write(tmp_path, **{field: value})
    with pytest.raises(ConfigValidationError, match=expected):
        load_flexibility_config(path)


@pytest.mark.parametrize("level", [0.0, 1.0, -0.1, 1.4])
def test_out_of_range_nominal_level_is_rejected(tmp_path: Path, level: float) -> None:
    path = _write(tmp_path, nominal_level=level)
    with pytest.raises(ConfigValidationError, match="nominal_level"):
        load_flexibility_config(path)


def test_unknown_granularity_is_rejected(tmp_path: Path) -> None:
    path = _write(tmp_path, granularity="fortnight")
    with pytest.raises(ConfigValidationError, match="granularity"):
        load_flexibility_config(path)


@pytest.mark.parametrize("floor", [0.0, -0.1, 1.5])
def test_reliance_floor_outside_range_is_rejected(tmp_path: Path, floor: float) -> None:
    """A floor of zero would let a wide interval imply *zero* flexibility.

    That is a claim about capability rather than about confidence, and the phase has no
    evidence for it, so the floor is bounded away from zero.
    """
    path = _write(tmp_path, reliance_floor=floor)
    with pytest.raises(ConfigValidationError, match="reliance_floor"):
        load_flexibility_config(path)


def test_reliance_window_smaller_than_its_minimum_is_rejected(tmp_path: Path) -> None:
    """Otherwise the trailing reference would never be populated and the factor is undefined."""
    path = _write(tmp_path, reliance_window_rows=50, reliance_min_periods=100)
    with pytest.raises(ConfigValidationError, match="reliance_min_periods"):
        load_flexibility_config(path)


def test_missing_published_mae_is_rejected(tmp_path: Path) -> None:
    """Without Phase 7's figures the run cannot prove its point forecast has not drifted."""
    path = tmp_path / "flexibility.toml"
    path.write_text(
        "\n".join(
            line
            for line in _write(tmp_path).read_text(encoding="utf-8").splitlines()
            if not line.startswith("[published_test_mae]")
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigValidationError):
        load_flexibility_config(path)


def test_missing_file_is_a_config_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        load_flexibility_config(tmp_path / "absent.toml")


def test_invalid_toml_is_a_config_error(tmp_path: Path) -> None:
    path = tmp_path / "flexibility.toml"
    path.write_text("target = \n", encoding="utf-8")
    with pytest.raises(ConfigError, match="invalid TOML"):
        load_flexibility_config(path)


# ---------------------------------------------------------------------------
# The constants themselves
# ---------------------------------------------------------------------------


def test_slot_granularities_match_the_ml_module() -> None:
    """The config restates the granularity list so it can reject a bad one without importing
    the ML package. If the two ever diverge the config would accept something the runner
    cannot fit, so they are asserted equal rather than trusted."""
    from energy_intelligence.ml.flexibility.demand import (
        SLOT_GRANULARITIES as ML_GRANULARITIES,
    )

    assert SLOT_GRANULARITIES == ML_GRANULARITIES


def test_task_reference_matches_the_shipped_config() -> None:
    config = load_flexibility_config()
    assert config.target == PHASE9_TASK_REFERENCE["target"]
    assert config.customer_count == PHASE9_TASK_REFERENCE["customer_count"]
    assert list(config.horizons) == list(PHASE9_TASK_REFERENCE["horizons"])
    assert config.lookback_steps == PHASE9_TASK_REFERENCE["lookback_steps"]


def test_default_config_filename_exists() -> None:
    assert (PROJECT_ROOT / "configs" / DEFAULT_FLEXIBILITY_CONFIG).is_file()


def test_config_is_frozen() -> None:
    """Immutable, like every other config here: a mutated config is a run nobody can repeat."""
    config = load_flexibility_config()
    with pytest.raises(Exception):
        config.nominal_level = 0.5  # type: ignore[misc]


def test_artifact_dir_is_under_phase9() -> None:
    config = FlexibilityExperimentConfig(
        **{
            **{
                field: getattr(load_flexibility_config(), field)
                for field in FlexibilityExperimentConfig.__dataclass_fields__
            }
        }
    )
    assert config.artifact_dir == f"artifacts/phase9/{config.label}"