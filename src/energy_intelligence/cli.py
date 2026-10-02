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
