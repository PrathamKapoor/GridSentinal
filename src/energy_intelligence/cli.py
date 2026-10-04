"""Command line entry point.

Commands:

``health``
    Run every Phase 1 health check. Exit code 0 when healthy, 1 otherwise, so it
    can gate CI directly.

``config show``
    Print the fully resolved configuration as JSON.

``config validate``
    Validate configuration only. Exit code 0 when valid, 1 when not.

``init-dirs``
    Create the canonical project directory layout.

``paths``
    Print the canonical directory layout.

Usage::

    python -m energy_intelligence health
    python -m energy_intelligence --config configs/experiment.toml health
    energy-intel config show
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from .config import ConfigError, load_config
from .health import check_health, render_report
from .logging_setup import get_logger, initialize_logging
from .paths import ProjectPaths

__all__ = ["build_parser", "main"]

logger = get_logger(__name__)

_EXIT_OK = 0
_EXIT_FAILED = 1


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for the ``energy-intel`` command."""
    parser = argparse.ArgumentParser(
        prog="energy-intel",
        description=(
            "Energy Intelligence project tooling. "
            "Phase 1 provides foundation utilities only - no AI, optimization, "
            "simulation or MLOps functionality exists yet."
        ),
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "Configuration file to load on top of configs/default.toml. "
            "Defaults to $ENERGY_INTEL_CONFIG when set."
        ),
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default=None,
        help="Override the configured log level.",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        metavar="NAME",
        help="Override the experiment name (lowercase slug).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        metavar="INT",
        help="Override the random seed.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    health = subparsers.add_parser(
        "health", help="Verify the Phase 1 environment."
    )
    health.add_argument(
        "--create-dirs",
        action="store_true",
        help="Create any missing standard directories instead of failing.",
    )
    health.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="Emit the report as JSON instead of text.",
    )

    config_parser = subparsers.add_parser(
        "config", help="Inspect or validate configuration."
    )
    config_subparsers = config_parser.add_subparsers(dest="config_command", required=True)
    config_subparsers.add_parser("show", help="Print resolved configuration as JSON.")
    config_subparsers.add_parser("validate", help="Validate configuration only.")

    domain_parser = subparsers.add_parser(
        "domain", help="Inspect or validate the energy-domain configuration."
    )
    domain_parser.add_argument(
        "--domain-config",
        type=str,
        default=None,
        metavar="PATH",
        help="Domain config file. Defaults to configs/domain.toml.",
    )
    domain_subparsers = domain_parser.add_subparsers(dest="domain_command", required=True)
    domain_subparsers.add_parser(
        "show", help="Print resolved energy-domain configuration as JSON."
    )
    domain_subparsers.add_parser(
        "validate", help="Validate the energy-domain configuration."
    )
    domain_subparsers.add_parser(
        "vocabulary",
        help="Print the energy-domain vocabulary (asset types, roles, constraints, objectives).",
    )

    data_parser = subparsers.add_parser(
        "data", help="Inspect, ingest and validate external energy data."
    )
    data_parser.add_argument(
        "--data-config",
        type=str,
        default=None,
        metavar="PATH",
        help="Dataset config file. Defaults to configs/data.toml.",
    )
    data_parser.add_argument(
        "--out",
        type=str,
        default=None,
        metavar="DIR",
        help="Directory to write JSON artifacts into. Defaults to artifacts/data.",
    )
    data_subparsers = data_parser.add_subparsers(dest="data_command", required=True)
    data_subparsers.add_parser(
        "config", help="Print the resolved dataset selection."
    )
    data_subparsers.add_parser(
        "inspect",
        help="Discover schema and report assets, time and data quality without mapping.",
    )
    data_subparsers.add_parser(
        "ingest",
        help="Run the full pipeline: manifest, schema, quality, mapping, balance.",
    )
    data_subparsers.add_parser(
        "validate",
        help="Run the pipeline and report the energy balance result.",
    )
    data_subparsers.add_parser(
        "mapping",
        help="Print the SMART-DS to domain field mapping.",
    )
    data_subparsers.add_parser(
        "manifest",
        help="Print the dataset manifest including per-file checksums.",
    )

    ml_parser = subparsers.add_parser(
        "ml",
        help="Build the ML dataset and run baseline forecasting experiments.",
    )
    ml_parser.add_argument(
        "--ml-config",
        type=str,
        default=None,
        metavar="PATH",
        help="Experiment config file. Defaults to configs/ml.toml.",
    )
    ml_subparsers = ml_parser.add_subparsers(dest="ml_command", required=True)
    ml_subparsers.add_parser(
        "config", help="Print the resolved experiment configuration."
    )
    ml_subparsers.add_parser(
        "targets",
        help="List every forecasting target with its measured support status.",
    )
    ml_subparsers.add_parser(
        "features",
        help="Print the feature catalogue with availability and leakage policy.",
    )
    ml_subparsers.add_parser(
        "dataset",
        help="Build the ML dataset and write its manifest, without training.",
    )
    ml_subparsers.add_parser(
        "run",
        help="Build the dataset, run every baseline, analyse and record results.",
    )
    ml_subparsers.add_parser(
        "experiments",
        help="Print the experiment registry.",
    )

    temporal_parser = subparsers.add_parser(
        "temporal",
        help="Phase 5: train the Energy Demand Dynamics Model and run ablations.",
    )
    temporal_parser.add_argument(
        "--temporal-config",
        type=str,
        default=None,
        metavar="PATH",
        help="Temporal experiment config. Defaults to configs/temporal.toml.",
    )
    temporal_subparsers = temporal_parser.add_subparsers(
        dest="temporal_command", required=True
    )
    temporal_subparsers.add_parser(
        "config", help="Print the resolved temporal experiment configuration."
    )
    temporal_subparsers.add_parser(
        "run",
        help="Train the main model, select on validation, evaluate test once.",
    )
    temporal_subparsers.add_parser(
        "ablate",
        help="Run the configured ablations at the reduced budget.",
    )
    temporal_subparsers.add_parser(
        "experiments",
        help="Print the Phase 5 registry records.",
    )

    qwen_parser = subparsers.add_parser(
        "qwen",
        help="Phase 6: Qwen3-1.7B-Base energy specialization.",
    )
    qwen_parser.add_argument(
        "--qwen-config",
        type=str,
        default=None,
        metavar="PATH",
        help="Qwen experiment config. Defaults to configs/qwen.toml.",
    )
    qwen_subparsers = qwen_parser.add_subparsers(dest="qwen_command", required=True)
    qwen_subparsers.add_parser(
        "config", help="Print the resolved Qwen experiment configuration."
    )
    qwen_subparsers.add_parser(
        "verify",
        help="Verify the local base checkpoint against the pinned facts.",
    )
    qwen_subparsers.add_parser(
        "inspect",
        help="Report the backbone architecture and the Phase 7 surgery sites.",
    )
    probe_parser = qwen_subparsers.add_parser(
        "probe",
        help="Run the probe experiment(s) over the cached frozen features.",
    )
    probe_parser.add_argument(
        "arm",
        nargs="*",
        help="Arms to run. Default: the main pretrained probe.",
    )
    qwen_subparsers.add_parser(
        "experiments", help="Print the Phase 6 experiment records."
    )

    router_parser = subparsers.add_parser(
        "router",
        help="Phase 7: heterogeneous energy expert router (MoE).",
    )
    router_parser.add_argument(
        "--router-config",
        type=str,
        default=None,
        metavar="PATH",
        help="Router experiment config. Defaults to configs/router.toml.",
    )
    router_subparsers = router_parser.add_subparsers(dest="router_command", required=True)
    router_subparsers.add_parser(
        "config", help="Print the resolved router experiment configuration."
    )
    router_subparsers.add_parser(
        "experts",
        help="Rebuild the expert pool and cache full-split forecasts (the slow step).",
    )
    router_subparsers.add_parser(
        "run",
        help="Diversity, oracle, router training, ablations and one sealed evaluation.",
    )
    router_subparsers.add_parser(
        "experiments", help="Print the Phase 7 registry records."
    )

    uncertainty_parser = subparsers.add_parser(
        "uncertainty",
        help="Phase 8: calibrated predictive uncertainty and prediction intervals.",
    )
    uncertainty_parser.add_argument(
        "--uncertainty-config",
        type=str,
        default=None,
        metavar="PATH",
        help="Uncertainty experiment config. Defaults to configs/uncertainty.toml.",
    )
    uncertainty_subparsers = uncertainty_parser.add_subparsers(
        dest="uncertainty_command", required=True
    )
    uncertainty_subparsers.add_parser(
        "config", help="Print the resolved uncertainty experiment configuration."
    )
    uncertainty_subparsers.add_parser(
        "run",
        help=(
            "Fit six interval methods on the calibration split and evaluate once on the "
            "sealed test split."
        ),
    )
    uncertainty_subparsers.add_parser(
        "experiments", help="Print the Phase 8 registry records."
    )

    subparsers.add_parser("init-dirs", help="Create the standard directory layout.")
    subparsers.add_parser("paths", help="Print the standard directory layout.")

    return parser


def _overrides(args: argparse.Namespace) -> dict[str, object]:
    overrides: dict[str, object] = {}
    if args.log_level is not None:
        overrides["log_level"] = args.log_level
    if args.experiment_name is not None:
        overrides["experiment_name"] = args.experiment_name
    if args.seed is not None:
        overrides["seed"] = args.seed
    return overrides


def _command_health(args: argparse.Namespace) -> int:
    report = check_health(args.config, create_missing_dirs=args.create_dirs)

    if args.as_json:
        payload = {
            "ok": report.ok,
            "checks": [
                {
                    "name": check.name,
                    "status": check.status,
                    "detail": check.detail,
                    "remedy": check.remedy,
                }
                for check in sorted(report.checks, key=lambda item: item.sort_key)
            ],
        }
        print(json.dumps(payload, indent=2))
    else:
        print(render_report(report))

    return report.exit_code


def _command_config(args: argparse.Namespace) -> int:
    try:
        config = load_config(args.config, overrides=_overrides(args))
    except ConfigError as exc:
        print(f"configuration error:\n{exc}", file=sys.stderr)
        return _EXIT_FAILED

    if args.config_command == "show":
        print(json.dumps(config.as_dict(), indent=2))
    else:
        print(f"configuration valid (fingerprint={config.fingerprint()})")

    return _EXIT_OK


def _command_domain(args: argparse.Namespace) -> int:
    from .config.domain import load_domain_config

    if args.domain_command == "vocabulary":
        print(json.dumps(_domain_vocabulary(), indent=2))
        return _EXIT_OK

    try:
        domain_config = load_domain_config(args.domain_config)
    except ConfigError as exc:
        print(f"domain configuration error:\n{exc}", file=sys.stderr)
        return _EXIT_FAILED

    if args.domain_command == "show":
        print(json.dumps(domain_config.as_dict(), indent=2))
    else:
        print(
            f"domain configuration valid "
            f"(fingerprint={domain_config.fingerprint()})"
        )
    return _EXIT_OK


def _domain_vocabulary() -> dict[str, object]:
    """Summarise the domain vocabulary for human inspection.

    Reporting this through the CLI rather than only in documentation means the
    contract that later phases depend on is inspectable at runtime, and a
    documentation drift becomes visible.
    """
    from .domain import (
        CONTROLLABLE_AUTHORITY_LEVELS,
        ActionType,
        AssetType,
        AuthorityLevel,
        ConstraintCategory,
        NetworkElementKind,
        ObjectiveCategory,
        QualityFlag,
        StateOrigin,
        UncertaintyKind,
        VariableRole,
    )
    from .domain.assets import ASSET_TYPES
    from .domain.serialization import SCHEMA_VERSION

    return {
        "schema_version": SCHEMA_VERSION,
        "variable_roles": sorted(role.value for role in VariableRole),
        "asset_types": sorted(asset_type.value for asset_type in AssetType),
        "asset_implementations": {
            asset_type.value: cls.__name__ for asset_type, cls in sorted(
                ASSET_TYPES.items(), key=lambda pair: pair[0].value
            )
        },
        "authority_levels": sorted(level.value for level in AuthorityLevel),
        "controllable_authority_levels": sorted(
            level.value for level in CONTROLLABLE_AUTHORITY_LEVELS
        ),
        "action_types": sorted(action.value for action in ActionType),
        "constraint_categories": sorted(c.value for c in ConstraintCategory),
        "objective_categories": sorted(o.value for o in ObjectiveCategory),
        "network_element_kinds": sorted(k.value for k in NetworkElementKind),
        "quality_flags": sorted(f.value for f in QualityFlag),
        "state_origins": sorted(o.value for o in StateOrigin),
        "uncertainty_kinds": sorted(u.value for u in UncertaintyKind),
    }


def _layout_from_config(data_config_path: str | None):
    """Resolve the dataset selection into a :class:`SmartDsLayout`."""
    from .config.data import load_data_config
    from .data.smartds import SmartDsLayout

    data_config = load_data_config(data_config_path)
    layout = SmartDsLayout(
        root=data_config.raw_root,
        version=data_config.version,
        year=data_config.year,
        region=data_config.region,
        subregion=data_config.subregion,
        scenario=data_config.scenario,
        substation=data_config.substation,
        feeder=data_config.feeder,
    )
    return data_config, layout


def _artifact_dir(out: str | None):
    from .paths import ProjectPaths

    base = Path(out) if out else ProjectPaths.from_root().artifacts / "data"
    if not base.is_absolute():
        base = ProjectPaths.from_root().root / base
    return base


def _write_json(directory: Path, name: str, payload: object) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )
    return path


def _command_data(args: argparse.Namespace) -> int:
    from .data.smartds import SmartDsAdapter, mapping_table, run_ingestion

    if args.data_command == "mapping":
        print(json.dumps(list(mapping_table()), indent=2))
        return _EXIT_OK

    data_config, layout = _layout_from_config(args.data_config)

    if args.data_command == "config":
        print(json.dumps(data_config.as_dict(), indent=2))
        return _EXIT_OK

    if not layout.profiles_dir.is_dir():
        print(
            f"dataset not acquired: {layout.profiles_dir} does not exist. "
            "See docs/smart_ds_acquisition.md for the authoritative URL.",
            file=sys.stderr,
        )
        return _EXIT_FAILED

    if args.data_command == "inspect":
        adapter = SmartDsAdapter(layout)
        schema = adapter.schema()
        payload = {
            "layout": layout.describe(),
            "schema": schema.to_dict(),
            "quality": adapter.quality_report().to_dict(),
            "der_assets": {
                "solar_count": adapter.der_assets().solar_count,
                "battery_count": adapter.der_assets().battery_count,
                "ev_sites": adapter.der_assets().ev_sites,
            },
        }
        print(json.dumps(payload, indent=2, default=str))
        return _EXIT_OK

    result = run_ingestion(layout, peak_only=data_config.peak_only)
    directory = _artifact_dir(args.out)

    if args.data_command == "manifest":
        print(result.manifest.summary_line())
        path = _write_json(directory, "manifest.json", result.manifest.to_dict())
        print(f"written: {path}")
        return _EXIT_OK

    if args.data_command == "validate":
        print(result.balance.render())
        path = _write_json(directory, "balance_report.json", result.balance.to_dict())
        print(f"\nwritten: {path}")
        # A failing balance is a legitimate, reportable outcome, not a crash.
        return _EXIT_OK if result.balance.passed else 1

    # ingest
    from .domain.serialization import encode, to_dict

    mapping_payload = result.mapping.to_dict()
    mapping_payload["observations"] = [
        to_dict(observation) for observation in result.mapping_result.observations
    ]
    written = [
        _write_json(directory, "manifest.json", result.manifest.to_dict()),
        _write_json(directory, "schema_report.json", result.schema),
        _write_json(directory, "quality_report.json", result.quality),
        _write_json(directory, "balance_report.json", result.balance.to_dict()),
        _write_json(directory, "reconstruction.json", result.reconstruction),
        _write_json(directory, "mapping_report.json", mapping_payload),
    ]

    system_path = _write_json(
        directory, "energy_system.json", json.loads(encode(result.mapping_result.system))
    )
    written.append(system_path)

    print(result.manifest.summary_line())
    print()
    print("reconstruction at the evaluated timepoint:")
    for key, value in result.reconstruction.items():
        print(f"  {key}: {value}")
    print()
    print(f"balance: {'PASS' if result.balance.passed else 'FAIL'} "
          f"(max |residual| = {result.balance.max_absolute_residual_kw:.4f} kW, "
          f"tolerance {result.balance.tolerance_kw} kW)")
    print()
    print("artifacts written:")
    for path in written:
        print(f"  {path}")
    return _EXIT_OK


def _command_ml(args: argparse.Namespace) -> int:
    """Phase 4: dataset construction and baseline forecasting.

    Sub-commands
    ------------
    ``config``      resolved experiment configuration
    ``targets``     every forecasting candidate and its support status
    ``features``    the feature catalogue and the leakage policy
    ``dataset``     build and persist the dataset without training
    ``run``         full experiment: dataset, baselines, analysis, registry
    ``experiments`` the recorded registry
    """
    from .config.ml import load_ml_config

    if args.ml_command == "config":
        print(json.dumps(load_ml_config(args.ml_config).to_dict(), indent=2))
        return _EXIT_OK

    if args.ml_command == "targets":
        from .ml.targets import TARGET_REGISTRY

        payload = {
            "targets": [spec.to_dict() for spec in TARGET_REGISTRY],
            "supported": [
                spec.target_id
                for spec in TARGET_REGISTRY
                if spec.status != "UNSUPPORTED"
            ],
        }
        print(json.dumps(payload, indent=2))
        return _EXIT_OK

    if args.ml_command == "features":
        from .ml.features import feature_catalogue

        print(json.dumps([spec.to_dict() for spec in feature_catalogue()], indent=2))
        return _EXIT_OK

    config = load_ml_config(args.ml_config)

    if args.ml_command == "experiments":
        from .ml.registry import read_records

        records = read_records(ProjectPaths.from_root().experiments / REGISTRY_FILENAME)
        print(json.dumps([record.to_dict() for record in records], indent=2))
        return _EXIT_OK

    from .ml.pipeline import run_ml_pipeline

    try:
        output = run_ml_pipeline(config, train=(args.ml_command == "run"))
    except FileNotFoundError as exc:
        print(f"{exc}", file=sys.stderr)
        print(
            "the SMART-DS dataset must be acquired first; see docs/smart_ds_acquisition.md",
            file=sys.stderr,
        )
        return _EXIT_FAILED

    print(output.render())
    return _EXIT_OK


def _command_temporal(args: argparse.Namespace) -> int:
    """Phase 5: the Energy Demand Dynamics Model.

    Sub-commands
    ------------
    ``config``      resolved temporal experiment configuration
    ``run``         train, select on validation, evaluate the test split once
    ``ablate``      the configured ablations at a reduced, matched budget
    ``experiments`` the Phase 5 registry records
    """
    import time

    from .config.temporal import load_temporal_config
    from .data.smartds import SmartDsLayout
    from .ml.config_ml_bridge import dataset_config_from
    from .ml.phase5 import load_phase4_reference, load_series_for_experiment
    from .ml.registry import REGISTRY_FILENAME

    config = load_temporal_config(args.temporal_config)

    if args.temporal_command == "config":
        print(json.dumps(config.to_dict(), indent=2))
        return _EXIT_OK

    if args.temporal_command == "experiments":
        from .ml.registry import read_records

        records = [
            record
            for record in read_records(ProjectPaths.from_root().experiments / REGISTRY_FILENAME)
            if record.model.startswith("phase5")
        ]
        print(json.dumps([record.to_dict() for record in records], indent=2))
        return _EXIT_OK

    data_config = dataset_config_from()
    layout = SmartDsLayout(
        root=data_config.raw_root,
        version=data_config.version,
        year=data_config.year,
        region=data_config.region,
        subregion=data_config.subregion,
        scenario=data_config.scenario,
        substation=data_config.substation,
        feeder=data_config.feeder,
    )
    if not layout.profiles_dir.is_dir():
        print(f"dataset not acquired: {layout.profiles_dir}", file=sys.stderr)
        print("see docs/smart_ds_acquisition.md", file=sys.stderr)
        return _EXIT_FAILED

    series = load_series_for_experiment(layout, config)
    paths = ProjectPaths.from_root()
    reference = load_phase4_reference(
        paths.artifacts / "ml" / "load-40" / "reports" / "comparison.json"
    )

    started = time.perf_counter()
    if args.temporal_command == "ablate":
        from .ml.ablation import run_ablations, render_ablation_table

        outcomes = run_ablations(
            series, config=config, paths=paths, phase4_reference=reference
        )
        print()
        print(render_ablation_table(outcomes))
        print(f"\ntotal {time.perf_counter() - started:.0f}s")
        return _EXIT_OK

    from .ml.phase5 import run_phase5_experiment

    result = run_phase5_experiment(
        series,
        sequence_config=config.sequence_config(),
        training_config=config.training_config(),
        paths=paths,
        architecture=config.architecture,
        experiment_id=f"phase5-{config.architecture}-{config.label}",
        phase4_reference=reference,
        label=config.label,
        validation_stride=config.validation_stride,
        model_params=dict(config.model_params),
    )
    print(result.render())
    print(f"\ntotal {time.perf_counter() - started:.0f}s")
    return _EXIT_OK


def _command_qwen(args: argparse.Namespace) -> int:
    """Phase 6: Qwen3-1.7B-Base energy specialization.

    Sub-commands
    ------------
    ``config``      resolved experiment configuration
    ``verify``      checkpoint verification against the pinned facts
    ``inspect``     architecture report and the Phase 7 surgery sites
    ``probe``       the probe experiments over cached frozen features
    ``experiments`` the recorded Phase 6 results

    ``verify`` and ``inspect`` need the checkpoint but not the dataset, so they work
    before the SMART-DS subset is present. ``probe`` needs both the checkpoint's
    cached features and the dataset, and it deliberately never downloads anything:
    a 3.44 GB fetch should be an explicit, separate step.
    """
    from pathlib import Path as _Path

    from .config.qwen import load_qwen_config
    from .ml.qwen.checkpoint import Qwen3Checkpoint, inspect_architecture

    config = load_qwen_config(args.qwen_config)
    paths = ProjectPaths.from_root()

    if args.qwen_command == "config":
        print(json.dumps(config.to_dict(), indent=2))
        return _EXIT_OK

    checkpoint = Qwen3Checkpoint(_Path(config.checkpoint_root))

    if args.qwen_command == "verify":
        try:
            record = checkpoint.verify()
        except (FileNotFoundError, ValueError) as error:
            print(str(error), file=sys.stderr)
            return _EXIT_FAILED
        print(json.dumps(record, indent=2, sort_keys=True))
        return _EXIT_OK

    if args.qwen_command == "inspect":
        try:
            checkpoint.verify()
            backbone = checkpoint.load_backbone()
        except (FileNotFoundError, ValueError) as error:
            print(str(error), file=sys.stderr)
            return _EXIT_FAILED
        print(json.dumps(inspect_architecture(backbone), indent=2, sort_keys=True))
        return _EXIT_OK

    if args.qwen_command == "experiments":
        summary = paths.artifacts / "phase6" / "experiments.json"
        if not summary.is_file():
            print("no Phase 6 experiments recorded yet", file=sys.stderr)
            return _EXIT_FAILED
        print(summary.read_text(encoding="utf-8"))
        return _EXIT_OK

    # probe
    import subprocess
    import sys as _sys

    script = Path(__file__).resolve().parents[3] / "scripts" / "phase6_probe.py"
    if not script.is_file():
        print(f"probe driver not found at {script}", file=sys.stderr)
        return _EXIT_FAILED
    completed = subprocess.run(
        [_sys.executable, str(script), *(args.arm or [])], cwd=Path.cwd()
    )
    return completed.returncode


def _script_path(name: str) -> Path | None:
    """Locate a driver script under ``scripts/``.

    Two candidate roots, because neither works alone. The package's own location is right
    when running from a source checkout (``src/energy_intelligence/cli.py`` -> repo root) and
    wrong when running from an installed console script inside a virtual environment. The
    working directory is right in the second case and in neither when the process was
    launched from elsewhere. Both are checked, and a missing script is reported rather than
    guessed at.
    """
    candidates = [
        Path(__file__).resolve().parents[3] / "scripts" / name,
        Path.cwd() / "scripts" / name,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _command_router(args: argparse.Namespace) -> int:
    """Phase 7: heterogeneous energy expert routing.

    Sub-commands
    ------------
    ``config``      resolved experiment configuration
    ``experts``    refit persistence/GBM/TCN and cache full-split forecasts
    ``run``        diversity, oracle, router, ablations, one sealed test evaluation
    ``experiments`` the recorded Phase 7 results

    ``experts`` is separated from ``run`` because it is the expensive step - a GBM fit on
    950,400 training rows plus a 74k-parameter temporal model over 359,040 rows - and
    every later question in the phase reads the same cached forecasts. Separating them
    also means the router can be re-trained without re-running the experts, which is
    what makes the ablations affordable.

    ``run`` never downloads anything and never reads the test split until its final
    evaluation.
    """
    import subprocess
    import sys as _sys

    from .config.router import load_router_config

    config = load_router_config(args.router_config)
    paths = ProjectPaths.from_root()

    if args.router_command == "config":
        print(json.dumps(config.to_dict(), indent=2, sort_keys=True))
        return _EXIT_OK

    if args.router_command == "experiments":
        summary = paths.artifacts / "phase7" / config.label / "summary.md"
        if not summary.is_file():
            print("no Phase 7 experiment recorded yet", file=sys.stderr)
            return _EXIT_FAILED
        print(summary.read_text(encoding="utf-8"))
        return _EXIT_OK

    if args.router_command == "experts":
        script = _script_path("phase7_experts.py")
        if script is None:
            print(
                "expert driver scripts/phase7_experts.py not found; run it from the "
                "project root",
                file=sys.stderr,
            )
            return _EXIT_FAILED
        completed = subprocess.run([_sys.executable, str(script)], cwd=Path.cwd())
        return completed.returncode

    return _command_router_run(config, paths)


def _command_router_run(config, paths: ProjectPaths) -> int:
    """Run the Phase 7 experiment over the cached expert forecasts."""
    import torch

    from .data.smartds import SmartDsLayout
    from .ml.config_ml_bridge import dataset_config_from
    from .ml.phase5 import load_series_for_experiment
    from .ml.router.cache import load_expert_cache
    from .ml.router.offline.experiment import run_phase7
    from .ml.registry import ExperimentRecord, append_record, now_iso

    torch.set_num_threads(config.torch_threads)
    root = paths.root
    cache_path = root / config.expert_cache
    if not cache_path.is_file():
        print(
            f"expert cache not found at {cache_path}; run "
            f"`energy-intel router experts` first",
            file=sys.stderr,
        )
        return _EXIT_FAILED

    data = dataset_config_from()
    layout = SmartDsLayout(
        data.raw_root, data.version, data.year, data.region,
        data.subregion, data.scenario, data.substation, data.feeder,
    )
    series = load_series_for_experiment(layout, config.temporal_experiment_config())
    cache = load_expert_cache(cache_path)
    panel = cache.panel()
    artifact_dir = paths.artifacts / "phase7" / config.label
    artifact_dir.mkdir(parents=True, exist_ok=True)

    def log(message: str) -> None:
        print(message, flush=True)
        with (artifact_dir / "run.log").open("a", encoding="utf-8") as handle:
            handle.write(message + "\n")

    log(f"Phase 7: cache {cache.describe()}")
    result = run_phase7(
        cache=cache,
        panel=panel,
        values=series.normalised().astype("float32"),
        expert_seconds=cache.metadata.get("expert_timings_seconds"),
        artifact_dir=artifact_dir,
        config=config.training_config(),
        architecture=config.router_architecture(),
        experiment_id=f"phase7-{config.label}",
        dataset_version=str(cache.metadata.get("index_version", "")),
        seed=config.seed,
        expert_parameters=_expert_parameters(cache),
        series_ids=series.series_ids,
        origin_iso=series.origin,
        published_test_mae=config.published_test_mae,
        run_ablations_too=config.run_ablations,
        run_crossfit=config.run_crossfit,
        ablation_epochs=config.ablation_epochs,
        trace_rows=config.trace_rows,
        log=log,
    )

    record = ExperimentRecord(
        experiment_id=result.experiment_id,
        created_at=now_iso(),
        target_id=config.target,
        dataset_version=result.dataset_version,
        dataset_sha256=str(cache.metadata.get("index_version", "")),
        feature_names=list(result.router.get("router_features", [])),
        lookback_steps=config.lookback_steps,
        horizon_steps=max(config.horizons),
        split_fractions={"train": config.train_fraction, "validation": config.validation_fraction},
        split_counts={
            "validation_router_fit": result.router.get("train_rows"),
            "validation_router_select": result.router.get("validation_rows"),
            "test": int(cache.split_bounds[cache.split_names.index("test")][1])
            - int(cache.split_bounds[cache.split_names.index("test")][0]),
        },
        model="phase7_energy_router",
        model_family="mixture_of_experts",
        hyperparameters={
            "experts": list(cache.expert_names),
            "router": result.router.get("architecture"),
            "router_parameters": result.router.get("parameters"),
            "router_features": result.router.get("router_features"),
            "training": result.router.get("history") and config.training_config().to_dict(),
            "oracle_gain_available_pct": result.verdict.get("mean_oracle_gain_available_pct"),
        },
        seed=config.seed,
        train_seconds=float(result.cost["router_train_seconds"]),
        predict_seconds=float(result.cost["router_inference_seconds"]),
        metrics={
            horizon: {
                "mae": result.methods["router_soft"][horizon]["mae"],
                "rmse": result.methods["router_soft"][horizon]["rmse"],
                "n": result.methods["router_soft"][horizon]["n"],
                "unit": "kW",
                "storage": "kW, the row's rated kW applied to a per-unit forecast",
                "vs_best_single_pct": result.verdict["per_horizon"][horizon][
                    "router_vs_best_single_pct"
                ],
                "selection_accuracy": result.routing["router_soft"][horizon][
                    "selection_accuracy"
                ],
                "routing_regret_kw": result.routing["router_soft"][horizon][
                    "routing_regret_kw"
                ],
            }
            for horizon in result.verdict["per_horizon"]
        },
        artifacts=result.artifacts,
        environment=_environment(torch),
        notes=list(result.notes) + [f"verdict: {result.verdict['answer']}"],
    )
    registry_path = paths.experiments / "registry.jsonl"
    append_record(registry_path, record)
    log(f"verdict: {result.verdict['answer']}")
    log(f"recorded {record.experiment_id} in {registry_path}")
    return _EXIT_OK


def _command_uncertainty(args: argparse.Namespace) -> int:
    """Phase 8: calibrated predictive uncertainty.

    Sub-commands
    ------------
    ``config``      resolved experiment configuration
    ``run``        fit six interval methods, evaluate once on the sealed test split
    ``experiments`` the recorded Phase 8 results

    ``run`` reads Phase 7's cached expert forecasts and reuses Phase 7's fitted
    fixed-ensemble weights. It fits no point-forecast model of its own and downloads
    nothing, so the only thing that varies between its methods is the interval.

    The sealed test split is read once, at the final evaluation, and nothing in the run
    adjusts to what is found there.
    """
    from .config.uncertainty import load_uncertainty_config

    config = load_uncertainty_config(args.uncertainty_config)
    paths = ProjectPaths.from_root()

    if args.uncertainty_command == "config":
        print(json.dumps(config.to_dict(), indent=2, sort_keys=True))
        return _EXIT_OK

    if args.uncertainty_command == "experiments":
        summary = paths.artifacts / "phase8" / config.label / "summary.md"
        if not summary.is_file():
            print("no Phase 8 experiment recorded yet", file=sys.stderr)
            return _EXIT_FAILED
        print(summary.read_text(encoding="utf-8"))
        return _EXIT_OK

    return _command_uncertainty_run(config, paths)


def _command_uncertainty_run(config, paths: ProjectPaths) -> int:
    """Run the Phase 8 experiment over Phase 7's cached expert forecasts.

    Args:
        config: The resolved :class:`UncertaintyExperimentConfig`.
        paths: The project's standard directory layout.

    Returns:
        ``0`` on success, ``1`` when an input is missing.
    """
    import torch

    from .data.smartds import SmartDsLayout
    from .ml.config_ml_bridge import dataset_config_from
    from .ml.phase5 import load_series_for_experiment
    from .ml.registry import ExperimentRecord, append_record, now_iso
    from .ml.router.cache import load_expert_cache
    from .ml.uncertainty.offline.experiment import run_phase8

    torch.set_num_threads(config.torch_threads)
    root = paths.root
    cache_path = root / config.expert_cache
    if not cache_path.is_file():
        print(
            f"expert cache not found at {cache_path}; run "
            f"`energy-intel router experts` first - Phase 8 reuses Phase 7's forecasts "
            f"rather than refitting them",
            file=sys.stderr,
        )
        return _EXIT_FAILED
    phase7_path = root / config.phase7_result
    if not phase7_path.is_file():
        print(
            f"Phase 7 result not found at {phase7_path}; Phase 8 reads the fixed-ensemble "
            f"weights from it so the point forecast cannot drift",
            file=sys.stderr,
        )
        return _EXIT_FAILED

    data = dataset_config_from()
    layout = SmartDsLayout(
        data.raw_root, data.version, data.year, data.region,
        data.subregion, data.scenario, data.substation, data.feeder,
    )
    series = load_series_for_experiment(layout, config.temporal_experiment_config())
    cache = load_expert_cache(cache_path)
    panel = cache.panel()
    artifact_dir = paths.artifacts / "phase8" / config.label
    artifact_dir.mkdir(parents=True, exist_ok=True)

    def log(message: str) -> None:
        print(message, flush=True)
        with (artifact_dir / "run.log").open("a", encoding="utf-8") as handle:
            handle.write(message + "\n")

    published = dict(config.published_test_mae or {})
    if "fixed_ensemble" not in published:
        # Phase 7's own fixed-ensemble MAE, read from its result rather than copied into
        # this config: it is the number the parity check is most likely to catch a refit
        # with, and a second hand-typed copy is a second thing to forget to update.
        import json as _json

        phase7 = _json.loads(phase7_path.read_text(encoding="utf-8"))
        table = (phase7.get("methods") or {}).get("fixed_ensemble") or {}
        published["fixed_ensemble"] = {
            str(horizon): float(entry["mae"]) for horizon, entry in table.items()
        }

    log(f"Phase 8: cache {cache.describe()}")
    result = run_phase8(
        cache=cache,
        panel=panel,
        values=series.normalised().astype("float32"),
        artifact_dir=artifact_dir,
        config=config,
        experiment_id=f"phase8-{config.label}",
        dataset_version=str(cache.metadata.get("index_version", "")),
        seed=config.seed,
        series_ids=series.series_ids,
        origin_iso=series.origin,
        phase7_result_path=phase7_path,
        published_point_mae=published,
        nominal_levels=config.nominal_levels,
        run_ablations=config.run_ablations,
        log=log,
    )

    record = ExperimentRecord(
        experiment_id=result.experiment_id,
        created_at=now_iso(),
        target_id=config.target,
        dataset_version=result.dataset_version,
        dataset_sha256=str(cache.metadata.get("index_version", "")),
        feature_names=list(result.features["names"]),
        lookback_steps=config.lookback_steps,
        horizon_steps=max(config.horizons),
        split_fractions={"train": config.train_fraction, "validation": config.validation_fraction},
        split_counts={
            "calibration_fit": result.split["calibration_fit_rows"],
            "calibration_conformity": result.split["calibration_conformity_rows"],
            "test": int(cache.split_bounds[cache.split_names.index("test")][1])
            - int(cache.split_bounds[cache.split_names.index("test")][0]),
        },
        model="phase8_probabilistic_forecast",
        model_family="prediction_interval",
        hyperparameters={
            "experts": list(cache.expert_names),
            "fixed_ensemble_weights": result.weights["fixed_ensemble_weights"],
            "methods": [name for name in result.methods],
            "nominal_levels": list(config.nominal_levels),
            "band_nominal_level": config.band_nominal_level,
            "calibration_fit_fraction": config.calibration_fit_fraction,
            "include_past_error_features": config.include_past_error_features,
            "coverage_tolerance": result.verdict["coverage_tolerance"],
            "calibration_status": result.artifact_summary["calibration_status"],
            "point_parity_max_pct": result.parity.get("max_relative_difference_pct"),
        },
        seed=config.seed,
        train_seconds=float(result.seconds),
        predict_seconds=0.0,
        metrics={
            str(horizon): {
                "mae": result.split_parity[str(horizon)]["test_mae_kw"],
                "n": result.split_parity[str(horizon)]["test_rows"],
                "unit": "kW",
                "storage": "kW, the row's rated kW applied to a per-unit forecast",
                "coverage_at_headline": {
                    row["method"]: row["coverage"]
                    for row in result.comparison
                    if row["horizon"] == int(horizon)
                },
                "interval_score_kw": {
                    row["method"]: row["interval_score_kw"]
                    for row in result.comparison
                    if row["horizon"] == int(horizon)
                },
                "calibrated_methods": [
                    row["method"]
                    for row in result.comparison
                    if row["horizon"] == int(horizon) and row["passes"]
                ],
            }
            for horizon in config.horizons
        },
        artifacts=result.artifacts,
        environment=_environment(torch),
        notes=list(result.notes)
        + [
            f"verdict: {result.verdict['answer']}",
            "components: "
            + ", ".join(
                f"{name}={component['answer']}"
                for name, component in result.verdict["components"].items()
            ),
        ],
    )
    registry_path = paths.experiments / "registry.jsonl"
    append_record(registry_path, record)
    log(f"verdict: {result.verdict['answer']}")
    log(f"recorded {record.experiment_id} in {registry_path}")
    return _EXIT_OK


def _expert_parameters(cache) -> dict[str, int]:
    """Per-expert size, from the builder's own provenance record.

    The three experts do not have a comparable notion of "parameters": the TCN has a
    weight tensor, the GBM has tree nodes and persistence has none at all. Each number is
    therefore labelled by what it counts, in the cost table's ``note``, rather than summed
    into one figure that would mean nothing.
    """
    provenance = cache.metadata.get("experts", {}) or {}
    out: dict[str, int] = {}
    for name in cache.expert_names:
        entry = provenance.get(name, {}) or {}
        for key in ("parameters", "tree_nodes"):
            value = entry.get(key)
            if value:
                out[name] = int(value)
                break
    return out


def _environment(torch_module) -> dict[str, str]:
    import platform
    import sys

    import numpy
    import sklearn

    return {
        "python": platform.python_version(),
        "platform": sys.platform,
        "machine": platform.machine(),
        "cpu": platform.processor(),
        "numpy": numpy.__version__,
        "sklearn": sklearn.__version__,
        "torch": torch_module.__version__,
    }


def _command_init_dirs() -> int:
    created = ProjectPaths.from_root().ensure()
    if created:
        print(f"created {len(created)} directory(ies):")
        for path in created:
            print(f"  {path}")
    else:
        print("all standard directories already present")
    return _EXIT_OK


def _command_paths() -> int:
    paths = ProjectPaths.from_root()
    payload = {
        "root": str(paths.root),
        # as_posix() so keys are stable across platforms (Windows would
        # otherwise emit 'data\\raw').
        "required": {
            path.relative_to(paths.root).as_posix(): path.is_dir()
            for path in paths.required_directories
        },
    }
    print(json.dumps(payload, indent=2))
    return _EXIT_OK



def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point.

    Args:
        argv: Argument list; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code: 0 on success, 1 on failure.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        match args.command:
            case "health":
                return _command_health(args)
            case "config":
                return _command_config(args)
            case "domain":
                return _command_domain(args)
            case "data":
                return _command_data(args)
            case "ml":
                return _command_ml(args)
            case "temporal":
                return _command_temporal(args)
            case "qwen":
                return _command_qwen(args)
            case "router":
                return _command_router(args)
            case "uncertainty":
                return _command_uncertainty(args)
            case "init-dirs":
                return _command_init_dirs()
            case "paths":
                return _command_paths()
            case _:  # pragma: no cover - argparse rejects unknown commands
                parser.error(f"unknown command: {args.command}")
    except ConfigError as exc:
        print(f"configuration error:\n{exc}", file=sys.stderr)
        return _EXIT_FAILED

    return _EXIT_FAILED  # pragma: no cover - unreachable


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
