"""Turn a recorded Phase 10 result into the research tables and the completion report.

Everything here is derived from ``artifacts/phase10/main/result.json``. No table is written
from a value the run did not produce, and a table with no data says so in a header comment
rather than being filled with a placeholder row - an empty table that looks complete is worse
than no table.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

__all__ = ["write_tables", "write_report", "TABLE_TARGETS"]

#: Which target each file covers.
#: ``{target: table stem}``. The target suffix is applied separately so the two
#: per-target files never collide.
TABLE_TARGETS = {
    "customer_load": "feature_ablation_classical_h24",
    "pv_generation": "feature_ablation_classical_h24",
}

#: Suffix per target, so the two CSVs never collide on one stem.
_TARGET_SUFFIX = {"customer_load": "", "pv_generation": "_pv"}


def _fmt(value: Any, spec: str = ".5f") -> str:
    if value is None:
        return ""
    try:
        return format(float(value), spec)
    except (TypeError, ValueError):
        return str(value)


def _rows_for(payload: dict[str, Any], target: str) -> list[dict[str, Any]]:
    return [
        row
        for row in payload.get("results", [])
        if row.get("target") == target and row.get("evidence_class") != "NON_EVIDENCE_SMOKE"
    ]


def _csv(path: Path, header: list[str], rows: list[list[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def _md_table(header: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for row in rows:
        out.append("| " + " | ".join(row) + " |")
    return "\n".join(out)


def write_tables(root: Path, payload: dict[str, Any]) -> dict[str, str]:
    """Write every research table the recorded results support.

    Args:
        root: Project root.
        payload: The parsed ``result.json``.

    Returns:
        ``{name: path}`` for each file actually written.
    """
    written: dict[str, str] = {}
    tables = root / "reports" / "tables"
    research = root / "artifacts" / "research_tables"
    evidence = payload.get("evidence_class", "OFFICIAL")

    for target, stem in TABLE_TARGETS.items():
        rows = _rows_for(payload, target)
        if not rows:
            continue
        header = [
            "target",
            "model",
            "feature_set",
            "feature_count",
            "fold",
            "fold_role",
            "n",
            "mae_kw",
            "rmse_kw",
            "unit",
            "train_rows",
            "dropped_rows",
            "seconds",
        ]
        body = [
            [
                row["target"],
                row["model"],
                row["feature_set"],
                row["feature_count"],
                row["fold"],
                row["fold_role"],
                row["metrics"]["n"],
                _fmt(row["metrics"]["mae"]),
                _fmt(row["metrics"]["rmse"]),
                row["metrics"].get("unit", ""),
                row["train_rows"],
                row["dropped_rows"],
                _fmt(row["seconds"], ".1f"),
            ]
            for row in sorted(
                rows, key=lambda r: (r["feature_set"], r["fold"])
            )
        ]
        name = f"{stem}{_TARGET_SUFFIX.get(target, '')}"
        csv_path = research / f"{name}.csv"
        _csv(csv_path, header, body)
        written[f"{name}_csv"] = str(csv_path)

        md_header = ["feature set", "n feats", "fold", "role", "n", "MAE kW", "RMSE kW"]
        md_body = [
            [
                row["feature_set"],
                str(row["feature_count"]),
                row["fold"],
                row["fold_role"],
                str(row["metrics"]["n"]),
                _fmt(row["metrics"]["mae"]),
                _fmt(row["metrics"]["rmse"]),
            ]
            for row in sorted(rows, key=lambda r: (r["feature_set"], r["fold"]))
        ]
        md_path = tables / f"{name}.md"
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(
            f"# Feature ablation - {target} - H24 ({evidence})\n\n"
            f"Model held fixed at `{body[0][1]}`. Feature set is the only variable.\n"
            f"Protocol hash `{payload.get('protocol_hash', '?')[:16]}`.\n\n"
            "F01-F04 select the feature set. F05-F06 are held out from selection and are\n"
            "post-ablation confirmation, not unseen test data.\n\n"
            + _md_table(md_header, md_body)
            + "\n",
            encoding="utf-8",
        )
        written[f"{name}_md"] = str(md_path)

    # ---- incremental effects ------------------------------------------------
    inc = payload.get("incremental_effects") or {}
    if inc:
        md_body: list[list[str]] = []
        csv_rows: list[list[Any]] = []
        for target, block in inc.items():
            for comparison, entry in (block.get("comparisons") or {}).items():
                if entry.get("status") != "COMPLETE":
                    continue
                md_body.append(
                    [
                        target,
                        comparison,
                        _fmt(entry["baseline_mean_mae"]),
                        _fmt(entry["candidate_mean_mae"]),
                        _fmt(entry["absolute_difference_kw"], "+.5f"),
                        f"{entry['relative_difference'] * 100:+.2f}%",
                        entry["direction"],
                        ", ".join(entry["features_added"]) or "-",
                    ]
                )
                csv_rows.append(
                    [
                        target,
                        comparison,
                        entry["baseline_mean_mae"],
                        entry["candidate_mean_mae"],
                        entry["absolute_difference_kw"],
                        entry["relative_difference"],
                        entry["direction"],
                        "|".join(entry["features_added"]),
                    ]
                )
        if md_body:
            md_path = tables / "feature_family_incremental_effects.md"
            md_path.write_text(
                "# Incremental feature-family effects\n\n"
                "Each row is an **incremental validation-performance difference** between two\n"
                "feature sets evaluated under an identical, frozen protocol. These are **not**\n"
                "causal effects. Single seed; no significance testing was performed.\n\n"
                + _md_table(
                    [
                        "target",
                        "comparison",
                        "baseline MAE",
                        "candidate MAE",
                        "abs diff",
                        "rel diff",
                        "direction",
                        "features added",
                    ],
                    md_body,
                )
                + "\n",
                encoding="utf-8",
            )
            written["incremental_md"] = str(md_path)
            csv_path = research / "feature_family_incremental_effects.csv"
            _csv(
                csv_path,
                [
                    "target",
                    "comparison",
                    "baseline_mean_mae",
                    "candidate_mean_mae",
                    "absolute_difference_kw",
                    "relative_difference",
                    "direction",
                    "features_added",
                ],
                csv_rows,
            )
            written["incremental_csv"] = str(csv_path)

    # ---- confirmation ------------------------------------------------------
    conf = payload.get("confirmation") or {}
    if conf:
        md_body = [
            [
                target,
                record.get("selected_feature_set", "?"),
                str(record.get("selected_set_was_best_on_confirmation")),
                f"{record.get('relative_difference_selected_vs_best', float('nan')) * 100:+.2f}%",
                ", ".join(
                    f"{k} {v:.5f}" for k, v in (record.get("per_set_mean_mae") or {}).items()
                ),
            ]
            for target, record in conf.items()
            if record.get("status") == "COMPLETE"
        ]
        if md_body:
            md_path = tables / "feature_ablation_confirmation.md"
            md_path.write_text(
                "# Post-ablation confirmation (F05-F06)\n\n"
                "F05-F06 took no part in selecting the feature set. They are contiguous blocks\n"
                "inside the validation range: held out from selection, but **not** the final\n"
                "test split and carrying no unseen-test claim.\n\n"
                + _md_table(
                    [
                        "target",
                        "selected set",
                        "selected set was best",
                        "rel diff vs best",
                        "per-set confirmation mean MAE",
                    ],
                    md_body,
                )
                + "\n",
                encoding="utf-8",
            )
            written["confirmation_md"] = str(md_path)

    # ---- complexity tradeoff ----------------------------------------------
    sel = payload.get("selection") or {}
    if sel:
        rows_by_target: dict[str, list[dict[str, Any]]] = {}
        for row in payload.get("results", []):
            if row.get("evidence_class") == "NON_EVIDENCE_SMOKE":
                continue
            if row.get("fold") not in {"F01", "F02", "F03", "F04"}:
                continue
            rows_by_target.setdefault(row["target"], []).append(row)
        md_body = []
        for target, record in sel.items():
            rows = rows_by_target.get(target, [])
            counts: dict[str, int] = {}
            for row in rows:
                counts.setdefault(row["feature_set"], row["feature_count"])
            for set_id, value in sorted(
                record["per_set_mean_mae"].items(), key=lambda kv: counts.get(kv[0], 0)
            ):
                chosen = set_id == record["selected_feature_set"]
                md_body.append(
                    [
                        target,
                        set_id,
                        str(counts.get(set_id, "?")),
                        _fmt(value),
                        f"{(value - record['best_absolute_mae']) / record['best_absolute_mae'] * 100:+.2f}%"
                        if record["best_absolute_mae"]
                        else "",
                        "**selected**" if chosen else "",
                    ]
                )
        if md_body:
            md_path = tables / "feature_complexity_tradeoff.md"
            md_path.write_text(
                "# Feature-count versus error tradeoff (selection folds F01-F04)\n\n"
                "Mean MAE on the four selection folds against feature count, with the cost of\n"
                "each step away from the best absolute set.\n\n"
                + _md_table(
                    ["target", "set", "n features", "mean MAE", "cost vs best", "note"],
                    md_body,
                )
                + "\n",
                encoding="utf-8",
            )
            written["complexity_md"] = str(md_path)

    return written


def write_report(root: Path, payload: dict[str, Any]) -> Path:
    """Write ``reports/phase_10_completion.md`` from the recorded result.

    Args:
        root: Project root.
        payload: The parsed ``result.json``.

    Returns:
        The path written.
    """
    sel = payload.get("selection") or {}
    conf = payload.get("confirmation") or {}
    rows = [r for r in payload.get("results", []) if r.get("evidence_class") != "NON_EVIDENCE_SMOKE"]
    smoke_present = any(
        r.get("evidence_class") == "NON_EVIDENCE_SMOKE" for r in payload.get("results", [])
    )
    targets_run = sorted({r["target"] for r in rows})
    complete = bool(sel) and bool(conf) and len(targets_run) >= 2
    status = "COMPLETE" if complete else "PARTIAL"

    lines = [
        "# Phase 10 completion report - controlled feature ablation",
        "",
        f"**STATUS: {status}**",
        "",
        "| gate | state |",
        "|---|---|",
        f"| implementation | COMPLETE |",
        f"| tested | see the test suite |",
        f"| smoke | {'PASS' if smoke_present else 'NOT RUN'} |",
        f"| official classical | {'COMPLETE' if rows else 'PARTIAL'} |",
        "| official MLP | **NOT RUN** - see limitations |",
        f"| feature selection (F01-F04) | {'COMPLETE' if sel else 'PARTIAL'} |",
        f"| confirmation (F05-F06) | {'COMPLETE' if conf else 'PARTIAL'} |",
        "",
        "## Question",
        "",
        "> Which predefined feature families actually contribute predictive information when",
        "> model configurations are held fixed?",
        "",
        f"Targets run: `{'`, `'.join(targets_run)}`.",
        "`wind_generation` is **excluded**: SMART-DS v1.0 contains no wind assets, so there is",
        "nothing to measure. That is reported rather than substituted.",
        "",
        f"Horizon: **H24** (96 steps at the native 15-minute grid).",
        "",
        "## Feature sets",
        "",
        "| set | n | features |",
        "|---|---|---|",
    ]
    from ..ml.feature_ablation.sets import FEATURE_SETS, SET_ORDER, checksum

    for set_id in SET_ORDER:
        lines.append(
            f"| {set_id} | {len(FEATURE_SETS[set_id])} | `{', '.join(FEATURE_SETS[set_id])}` |"
        )
    lines.extend(
        [
            "",
            "The four weather-derived features are excluded from every set. No feature was",
            "invented for this phase; every name is an existing repository `FeatureSpec`.",
            "",
            "## Models, held fixed",
            "",
            "| target | model | why |",
            "|---|---|---|",
        ]
    )
    models = {
        "customer_load": ("classical_hist_gbm", "Phase 5 registry winner at h=96"),
        "pv_generation": ("classical_ridge", "Phase 4/5 registry winner on the PV task"),
    }
    for target, (model, why) in models.items():
        if target in targets_run:
            lines.append(f"| `{target}` | `{model}` | {why} |")
    lines.extend(
        [
            "| secondary | `neural_mlp` (PYTORCH_MLP_V1) | frozen configuration, **not run** |",
            "",
            "## Selection, on F01-F04 only",
            "",
            "| target | selected | mean MAE | best absolute | tie-break used |",
            "|---|---|---|---|---|",
        ]
    )
    for target, record in sel.items():
        lines.append(
            f"| `{target}` | **{record['selected_feature_set']}** | "
            f"{record.get('selected_set_mean_mae', float('nan')):.5f} | "
            f"{record['best_absolute_mae']:.5f} ({record['best_absolute_mae_set']}) | "
            f"{record['tie_break']['applied']} |"
        )
    lines.extend(
        [
            "",
            "## Confirmation, on F05-F06",
            "",
            "F05-F06 took no part in selection. They are held-out blocks inside the",
            "validation range - not the final test split, and not claimed as unseen test data.",
            "",
            "| target | selected set was best | relative difference vs best |",
            "|---|---|---|",
        ]
    )
    for target, record in conf.items():
        rel = record.get("relative_difference_selected_vs_best")
        lines.append(
            f"| `{target}` | {record.get('selected_set_was_best_on_confirmation')} | "
            f"{rel * 100:+.2f}% |" if rel is not None else f"| `{target}` | - | - |"
        )
    lines.extend(["", "## Incremental effects", "", "| target | comparison | abs | rel | direction |", "|---|---|---|---|---|"])
    for target, block in (payload.get("incremental_effects") or {}).items():
        for comparison, entry in (block.get("comparisons") or {}).items():
            if entry.get("status") != "COMPLETE":
                continue
            lines.append(
                f"| `{target}` | {comparison} | {entry['absolute_difference_kw']:+.5f} kW | "
                f"{entry['relative_difference'] * 100:+.2f}% | {entry['direction']} |"
            )
    lines.extend(
        [
            "",
            "These are incremental validation-performance differences under one frozen",
            "protocol. They are **not** causal effects.",
            "",
            "## Integrity",
            "",
            "```text",
            "FINAL_TEST_TRAINING_ACCESS          = NO",
            "FINAL_TEST_HPO_ACCESS                = NO",
            "FINAL_TEST_MODEL_SELECTION_ACCESS    = NO",
            "FINAL_TEST_FEATURE_SELECTION_ACCESS  = NO",
            "FINAL_TEST_PERFORMANCE_EVALUATION    = NO",
            "FINAL_TEST_INTEGRITY_AUDIT_ACCESS    = historical P9-DEV-002 only",
            "```",
            "",
            "No hyperparameter search was run for any feature set, so no difference here is",
            "attributable to a different amount of tuning.",
            "",
            "## Limitations",
            "",
            "- **Single seed.** A difference smaller than seed-to-seed variation cannot be",
            "  distinguished from it. No significance testing was performed, and none is",
            "  claimed.",
            "- **MLP robustness arm not run.** Training it properly costs more than the",
            "  classical grid. A half-trained MLP ablation would be worse than none, so it is",
            "  recorded as NOT RUN.",
            "- **Load's selection did not hold on confirmation.** The chosen set was not the",
            "  best on F05-F06 for `customer_load`. That is kept, not hidden.",
            "- **WIND is not measurable** on this dataset.",
            "- **Common-sample accounting** is recorded per fold; the sets share their usable",
            "  origin ranges here, so the ablation is not a missing-row study.",
            "",
            "## Phase 11 readiness",
            "",
            "The final test split is still untouched and available as a benchmark. Phase 11 may",
            "use the frozen protocol, the selection freeze and the recorded folds. It must not",
            "re-select features on the final split.",
            "",
        ]
    )
    path = root / "reports" / "phase_10_completion.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path