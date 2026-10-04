"""Phase 7, part 5: the experiment config and the CLI surface.

Two things are being protected here, and both have bitten before.

**The task is inherited, not re-decided.** Phase 7's test numbers are only comparable
with Phase 4's and Phase 5's because they run on the same 40 customers, the same horizons,
the same window and the same split fractions. A config that quietly changed one of those
would produce a table that looks like Phase 5's and is not comparable with it, and nothing
downstream would notice. So the loader rejects a redefined task.

**The CLI wires every subcommand.** A missing subcommand is a typo that reads as a feature
request, and ``router experts`` and ``router run`` have different costs, so one of them
being unreachable matters.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from energy_intelligence.cli import build_parser, main
from energy_intelligence.config.router import (
    PHASE7_TASK_REFERENCE,
    RouterExperimentConfig,
    load_router_config,
)
from energy_intelligence.config.schema import ConfigError, ConfigValidationError
from energy_intelligence.config.loader import CONFIG_DIR


@pytest.fixture
def router_config_path() -> Path:
    return CONFIG_DIR / "router.toml"


def write_config(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "router.toml"
    path.write_text(body, encoding="utf-8")
    return path


BASE = """
target = "customer_load"
label = "router-main"
customer_count = 40
commercial_fraction = 0.25
seed = 20260101
torch_threads = 8
lookback_steps = 672
token_stride = 4
train_fraction = 0.70
validation_fraction = 0.15
horizons = [1, 4, 96]
architecture = "tiny_mlp"
hidden_sizes = [32]
epochs = 40
batch_size = 4096
learning_rate = 0.01
"""


# ---------------------------------------------------------------------------
# The shipped config
# ---------------------------------------------------------------------------


def test_the_shipped_router_config_loads() -> None:
    config = load_router_config()
    assert config.target == "customer_load"
    assert config.horizons == (1, 4, 96)
    assert config.customer_count == 40
    assert config.lookback_steps == 672
    assert config.seed == 20260101


def test_the_shipped_config_publishes_the_phase4_and_5_numbers_for_parity() -> None:
    """A refit expert that has drifted must fail the parity check, not be published."""
    config = load_router_config()
    published = config.published_test_mae or {}
    assert set(published) == {"persistence", "classical_hist_gbm", "phase5_tcn"}
    for expert, by_horizon in published.items():
        assert set(by_horizon) == {"1", "4", "96"}
        for value in by_horizon.values():
            assert 0.0 < value < 100.0


def test_the_shipped_config_builds_the_router_architecture_and_protocol() -> None:
    config = load_router_config()
    architecture = config.router_architecture()
    assert architecture.name == "tiny_mlp"
    assert architecture.hidden_sizes == (32,)
    training = config.training_config()
    assert training.seed == config.seed
    assert training.train_fraction_of_validation == config.router_train_fraction_of_validation
    assert training.threads == config.torch_threads


def test_the_shipped_config_builds_a_phase5_matching_sequence_config() -> None:
    """The panel must be Phase 5's rows, or the TCN checkpoint does not apply."""
    config = load_router_config()
    sequence = config.temporal_config()
    assert sequence.lookback_steps == 672
    assert sequence.token_stride == 4
    assert sequence.horizons == (1, 4, 96)
    assert sequence.train_fraction == 0.70
    assert sequence.validation_fraction == 0.15
    # Training origins at stride 1, because Phase 7's GBM expert is Phase 4's model on
    # Phase 4's full training grid.
    assert sequence.train_origin_stride == 1


def test_the_shipped_config_builds_a_phase5_compatible_loader_config() -> None:
    config = load_router_config()
    temporal = config.temporal_experiment_config()
    assert temporal.customer_count == 40
    assert temporal.commercial_fraction == 0.25
    assert temporal.seed == 20260101
    assert temporal.horizons == (1, 4, 96)


def test_config_round_trips_through_its_dict() -> None:
    config = load_router_config()
    payload = config.to_dict()
    assert payload["horizons"] == [1, 4, 96]
    assert payload["hidden_sizes"] == [32]
    assert payload["expert_cache"].endswith(".npz")


# ---------------------------------------------------------------------------
# Rejecting a redefined task
# ---------------------------------------------------------------------------


def _replace(body: str, line: str) -> str:
    key = line.split("=")[0].strip()
    kept = [entry for entry in body.splitlines() if not entry.startswith(f"{key} =")]
    kept.append(line)
    return "\n".join(kept) + "\n"


@pytest.mark.parametrize(
    ("replacement", "message"),
    [
        ('target = "pv_generation"', "target must be"),
        ("customer_count = 10", "40-customer sample"),
        ("horizons = [1, 4]", "inherits Phase 5's horizons"),
        ("lookback_steps = 96", "672-step window"),
        ("train_fraction = 0.5", "train_fraction of 0.70"),
        ("validation_fraction = 0.2", "validation_fraction of 0.15"),
    ],
)
def test_a_redefined_task_is_refused(tmp_path: Path, replacement: str, message: str) -> None:
    with pytest.raises(ConfigValidationError, match=message):
        load_router_config(write_config(tmp_path, _replace(BASE, replacement)))


def test_the_reference_constants_match_the_shipped_config() -> None:
    config = load_router_config()
    assert config.target == PHASE7_TASK_REFERENCE["target"]
    assert config.customer_count == PHASE7_TASK_REFERENCE["customer_count"]
    assert list(config.horizons) == list(PHASE7_TASK_REFERENCE["horizons"])
    assert config.lookback_steps == PHASE7_TASK_REFERENCE["lookback_steps"]


# ---------------------------------------------------------------------------
# Ordinary validation
# ---------------------------------------------------------------------------


def test_a_missing_file_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="router config not found"):
        load_router_config(tmp_path / "absent.toml")


def test_invalid_toml_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="invalid TOML"):
        load_router_config(write_config(tmp_path, "target = \n"))


def test_unknown_keys_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigValidationError, match="unknown keys"):
        load_router_config(write_config(tmp_path, BASE + '\nmystery = 1\n'))


def test_missing_required_keys_are_all_reported(tmp_path: Path) -> None:
    body = "target = \"customer_load\"\n"
    with pytest.raises(ConfigValidationError, match="missing required key"):
        load_router_config(write_config(tmp_path, body))


@pytest.mark.parametrize(
    ("line", "message"),
    [
        ('architecture = "lstm"', "architecture must be one of"),
        ("hidden_sizes = [0]", "hidden_sizes must be positive"),
        ("dropout = 1.5", "dropout must lie"),
        ("temperature = 0.0", "temperature must be positive"),
        ("epochs = 0", "epochs must be positive"),
        ("batch_size = 0", "batch_size must be positive"),
        ("learning_rate = 0.0", "learning_rate must be positive"),
        ("weight_decay = -1.0", "weight_decay must not be negative"),
        ("patience = 0", "patience must be positive"),
        ("min_delta = -1.0", "min_delta must not be negative"),
        ("max_seconds = 0.0", "max_seconds must be positive"),
        ("router_train_fraction_of_validation = 1.0", "must lie in"),
        ("ablation_epochs = 0", "ablation_epochs must be positive"),
        ("trace_rows = -1", "trace_rows must not be negative"),
        ("torch_threads = 0", "torch_threads must be positive"),
        ("token_stride = 0", "token_stride must be positive"),
        ("token_stride = 700", "token_stride must be positive"),
    ],
)
def test_out_of_range_values_are_refused(tmp_path: Path, line: str, message: str) -> None:
    with pytest.raises(ConfigValidationError, match=message):
        load_router_config(write_config(tmp_path, _replace(BASE, line)))


def test_published_test_mae_must_be_a_table(tmp_path: Path) -> None:
    with pytest.raises(ConfigValidationError, match="published_test_mae must be a table"):
        load_router_config(write_config(tmp_path, BASE + '\npublished_test_mae = 3\n'))


def test_a_relative_path_is_resolved_against_the_project_root(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="router config not found"):
        load_router_config(Path("configs/does-not-exist.toml"))


def test_config_is_hashable_and_frozen() -> None:
    config = load_router_config()
    with pytest.raises(Exception):
        config.seed = 1  # type: ignore[misc]


def test_router_config_dataclass_defaults_are_sane() -> None:
    """A hand-built config, not the shipped file, still produces a valid protocol."""
    config = RouterExperimentConfig(
        target="customer_load",
        label="scratch",
        customer_count=40,
        commercial_fraction=0.25,
        seed=1,
        torch_threads=2,
        lookback_steps=672,
        token_stride=4,
        train_fraction=0.70,
        validation_fraction=0.15,
        horizons=(1, 4, 96),
        architecture="linear",
        hidden_sizes=(),
        dropout=0.0,
        temperature=1.0,
        epochs=2,
        batch_size=64,
        learning_rate=0.01,
        weight_decay=0.0,
        patience=1,
        min_delta=1e-6,
        max_seconds=30.0,
    )
    assert config.to_dict()["architecture"] == "linear"
    assert config.router_architecture().hidden_sizes == ()
    assert config.published_test_mae is None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_the_router_parser_exposes_every_subcommand() -> None:
    parser = build_parser()
    args = parser.parse_args(["router", "config"])
    assert args.command == "router"
    assert args.router_command == "config"
    for subcommand in ("config", "experts", "run", "experiments"):
        parsed = parser.parse_args(["router", subcommand])
        assert parsed.router_command == subcommand


def test_the_router_parser_takes_a_config_override() -> None:
    parser = build_parser()
    args = parser.parse_args(["router", "--router-config", "other.toml", "run"])
    assert args.router_config == "other.toml"


def test_router_requires_a_subcommand() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["router"])


def test_router_config_subcommand_prints_the_resolved_config(capsys) -> None:
    assert main(["router", "config"]) == 0
    printed = capsys.readouterr().out
    assert '"horizons"' in printed
    assert '"label": "router-main"' in printed


def test_router_config_subcommand_reports_a_missing_file(capsys, tmp_path: Path) -> None:
    code = main(["router", "--router-config", str(tmp_path / "absent.toml"), "config"])
    assert code == 1
    assert "router config not found" in capsys.readouterr().err


def test_router_run_refuses_to_start_without_an_expert_cache(capsys, tmp_path: Path) -> None:
    body = _replace(BASE, 'expert_cache = "artifacts/phase7/absent.npz"')
    code = main(
        [
            "router",
            "--router-config",
            str(write_config(tmp_path, body)),
            "run",
        ]
    )
    assert code == 1
    # One read: ``readouterr`` consumes what it returns, so a second call sees nothing.
    errors = capsys.readouterr().err
    assert "expert cache not found" in errors
    assert "energy-intel router experts" in errors


def test_router_experiments_reports_when_nothing_is_recorded(capsys, tmp_path: Path) -> None:
    body = _replace(BASE, 'label = "absent-label"')
    code = main(
        [
            "router",
            "--router-config",
            str(write_config(tmp_path, body)),
            "experiments",
        ]
    )
    assert code == 1
    assert "no Phase 7 experiment recorded" in capsys.readouterr().err