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
