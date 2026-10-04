"""Phase 10 execution script: the operator entry point for the controlled feature ablation.

Two modes, and the distinction matters:

``--smoke``
    Proves the pipeline on a small slice - a handful of customers, a coarse origin stride, one
    fold, one feature set. It exists to answer "does this run end to end", nothing else. Every
    artifact it writes is stamped ``NON_EVIDENCE_SMOKE``.

``--all``
    The official grid: five feature sets, six folds, the primary classical model per target,
    selection on F01-F04, confirmation on F05-F06.

There is deliberately **no** option to evaluate the final test split. The omission is the
control: an option that does not exist cannot be reached for by a well-meaning flag, and
``energy_intelligence.ml.feature_ablation.folds.assert_final_test_denied`` refuses the range at
runtime as well.

Examples::

    python scripts/run_feature_ablation.py --smoke
    python scripts/run_feature_ablation.py --all
    python scripts/run_feature_ablation.py --all --target load
    python scripts/run_feature_ablation.py --all --model classical
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from energy_intelligence.ml.feature_ablation.experiment import (  # noqa: E402
    TARGETS,
    UNSUPPORTED,
    run_ablation,
)
from energy_intelligence.ml.feature_ablation.sets import SET_ORDER  # noqa: E402

#: The public target names, mapped to repository target ids.
TARGET_ALIASES = {
    "load": "customer_load",
    "customer_load": "customer_load",
    "wind": "wind_generation",
    "wind_generation": "wind_generation",
    "pv": "pv_generation",
    "pv_generation": "pv_generation",
}


def build_parser() -> argparse.ArgumentParser:
    """The command-line surface."""
    parser = argparse.ArgumentParser(
        prog="run_feature_ablation",
        description=(
            "Phase 10 controlled feature ablation. Feature set is the independent variable; "
            "the model, its configuration, the seed, the folds and the metrics are held fixed."
        ),
        epilog="There is no final-test option, by design. F01-F04 select; F05-F06 confirm.",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--smoke",
        action="store_true",
        help="Run the non-evidence smoke path: proves the pipeline, proves nothing else.",
    )
    mode.add_argument(
        "--all", action="store_true", help="Run the official ablation grid."
    )
    parser.add_argument(
        "--target",
        action="append",
        choices=sorted(TARGET_ALIASES),
        help="Restrict to a target. Repeatable. Default: every runnable target.",
    )
    parser.add_argument(
        "--model",
        choices=("classical", "mlp"),
        help=(
            "classical runs the frozen primary model per target. mlp is the secondary "
            "robustness model and is recorded as NOT RUN when skipped."
        ),
    )
    parser.add_argument(
        "--fold",
        action="append",
        help="Restrict to fold ids, e.g. F01. Confirmation folds are reported, not selected on.",
    )
    parser.add_argument("--label", default=None, help="Artifact directory name under artifacts/phase10.")
    parser.add_argument(
        "--no-write-freeze",
        action="store_true",
        help="Refuse to create the protocol freeze if it is absent.",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the ablation.

    Args:
        argv: Argument list; defaults to ``sys.argv[1:]``.

    Returns:
        ``0`` on success, ``1`` when nothing was runnable.
    """
    args = build_parser().parse_args(argv)
    log = (lambda message: None) if args.quiet else (lambda message: print(message, flush=True))

    targets = TARGETS
    if args.target:
        requested = [TARGET_ALIASES[name] for name in args.target]
        blocked = [t for t in requested if t in UNSUPPORTED]
        if blocked:
            for target in blocked:
                print(f"REFUSED {target}: {UNSUPPORTED[target]}", file=sys.stderr)
            usable = [t for t in requested if t not in UNSUPPORTED]
            if not usable:
                return 1
            targets = tuple(usable)

    if args.model == "mlp":
        print(
            "NOTE: the secondary PYTORCH_MLP_V1 robustness arm is not executed by this "
            "script's official path. It is recorded as NOT RUN rather than approximated, "
            "because training it properly costs more than the classical grid and a "
            "half-trained MLP ablation would be worse than none.",
            file=sys.stderr,
        )

    sets = SET_ORDER
    if args.smoke:
        sets = ("A",)
        log("SMOKE: non-evidence pipeline check. 1 feature set, 1 fold, coarse stride.")

    try:
        outcome = run_ablation(
            root=ROOT,
            targets=targets,
            sets=sets,
            folds_filter=tuple(args.fold) if args.fold else None,
            smoke=args.smoke,
            label=args.label,
            write_freeze=not args.no_write_freeze,
            log=log,
        )
    except (ValueError, KeyError, FileExistsError, FileNotFoundError) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    log("")
    log(f"mode {outcome.mode}  evidence {outcome.to_dict()['evidence_class']}")
    for target, chosen in outcome.selection.items():
        log(
            f"  {target}: selected {chosen['selected_feature_set']} "
            f"(selection-fold mean MAE {chosen['selected_set_mean_mae']:.5f}, "
            f"best absolute {chosen['best_absolute_mae']:.5f} from "
            f"{chosen['best_absolute_mae_set']}, "
            f"tie-break applied={chosen['tie_break']['applied']})"
        )
    for target, conf in outcome.confirmation.items():
        log(
            f"  {target}: F05-F06 confirmation, chosen set best="
            f"{conf.get('selected_set_was_best_on_confirmation')}"
        )
    if not outcome.selection and not args.smoke:
        log("  selection did not run (partial grid)")
    log(json.dumps(outcome.artifacts, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())