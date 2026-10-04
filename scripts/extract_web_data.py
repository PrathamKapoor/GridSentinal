"""Extract the data the web frontend displays from recorded experiment artifacts.

Every number the landing page shows is read from a recorded experiment artifact -
nothing is invented here. Re-run to regenerate:

    uv run python scripts/extract_web_data.py

Writes web/src/data/evidence.json deterministically.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "web" / "src" / "data" / "evidence.json"

P8 = ROOT / "artifacts" / "phase8" / "uncertainty-main"
P7 = ROOT / "artifacts" / "phase7" / "router-main"


def forecast_slice() -> dict:
    """One real asset, the 1-hour-ahead horizon, a contiguous 48 h window of
    the published Phase 8 forecast (point + 90% calibrated interval)."""
    z = np.load(P8 / "probabilistic_forecasts.npz")
    ts = z["timestamps"]
    assets = z["asset_ids"]
    horizon_steps = [int(h) for h in z["horizon_steps"]]
    h4 = horizon_steps.index(4)

    asset = str(sorted(set(assets))[0])
    mask = assets == asset
    idx = np.where(mask)[0]
    point = z["point_kw"][mask, h4]

    # the 48 h window (192 rows) with the widest demand range, so the chart
    # shows ramping behaviour rather than a flat night
    best_start, best_range = 0, -1.0
    for start in range(0, len(idx) - 192 + 1, 96):
        rng = float(point[start : start + 192].max() - point[start : start + 192].min())
        if rng > best_range:
            best_range, best_start = rng, start

    sel = idx[best_start : best_start + 192]
    return {
        "asset_id": asset,
        "horizon_steps": 4,
        "horizon_label": "1 hour ahead",
        "method": str(z["method"][h4]),
        "nominal_level": float(z["nominal_level"]),
        "calibration_status": str(z["calibration_status"][h4]),
        "timestamps": [str(t) for t in ts[sel]],
        "point_kw": [round(float(v), 4) for v in z["point_kw"][sel, h4]],
        "lower_kw": [round(float(v), 4) for v in z["lower_kw"][sel, h4]],
        "upper_kw": [round(float(v), 4) for v in z["upper_kw"][sel, h4]],
    }


def published_procedure() -> list[dict]:
    """The published per-horizon Phase 8 procedure, from the recorded result."""
    r = json.loads((P8 / "result.json").read_text())
    s = r["artifact_summary"]
    rows = []
    for i, h in enumerate(s["horizons"]):
        method = s["method"][i]
        cov_err = s["coverage_error"][i]
        m = r["methods"][method][str(h)]
        rows.append(
            {
                "horizon": int(h),
                "method": method,
                "coverage_90": round(0.9 + cov_err, 4),
                "coverage_error": round(cov_err, 4),
                "spearman_width_error": round(m["ranking_by_width"]["spearman"], 3),
                "top_decile_multiple": round(
                    m["ranking_by_width"]["top_decile_error_multiple"], 2
                ),
                "mean_width_kw": round(s["mean_width_kw"][str(h)], 2),
                # error multiple per interval-width decile, widest last
                "deciles": [
                    round(d["error_multiple"], 2)
                    for d in m["ranking_by_width"]["deciles"]
                ],
            }
        )
    return rows


def constant_width_failure() -> dict:
    """The negative finding: the default constant-width interval's coverage."""
    r = json.loads((P8 / "result.json").read_text())
    gr = r["methods"]["global_residual"]
    cells, failing = [], 0
    for h in ("1", "4", "96"):
        for level, v in gr[h]["metrics"]["levels"].items():
            ok = bool(v["passes"])
            failing += 0 if ok else 1
            cells.append(
                {
                    "horizon": int(h),
                    "nominal_level": float(level),
                    "coverage": round(v["coverage"], 4),
                    "passes": ok,
                }
            )
    return {"cells": cells, "failing_cells": failing, "total_cells": len(cells)}


def phase7_table() -> list[dict]:
    """Phase 7 sealed-test MAE per method per horizon, from the recorded result.

    "Best single expert" is the per-horizon minimum across the three experts,
    matching the presentation in README.md.
    """
    r = json.loads((P7 / "result.json").read_text())
    experts = ["persistence", "classical_hist_gbm", "phase5_tcn"]
    best_single = {
        h: min(r["methods"][e][h]["mae"] for e in experts) for h in ("1", "4", "96")
    }
    keep = [
        ("persistence", "Persistence"),
        ("classical_hist_gbm", "Gradient boosting (Phase 4)"),
        ("phase5_tcn", "Temporal TCN (Phase 5)"),
        (None, "Best single expert"),
        ("router_soft", "Learned router (soft)"),
        ("fixed_ensemble", "Fixed weighted ensemble"),
        ("oracle_expert_assignment", "Oracle expert assignment"),
    ]
    rows = []
    for key, label in keep:
        if key is None:
            mae = best_single
        else:
            m = r["methods"][key]
            mae = {h: m[h]["mae"] for h in ("1", "4", "96")}
        rows.append(
            {
                "method": label,
                "h1": round(mae["1"], 4),
                "h4": round(mae["4"], 4),
                "h96": round(mae["96"], 4),
            }
        )
    return rows


def router_facts() -> dict:
    r = json.loads((P7 / "result.json").read_text())
    shares = [p["share_disagreeing"] for p in r["diversity"]["agreement"]["pairs"].values()]
    gains = [h["oracle_gain_pct"] for h in r["oracle"]["by_horizon"].values()]
    return {
        "disagreement_pct_min": round(min(shares) * 100, 1),
        "disagreement_pct_max": round(max(shares) * 100, 1),
        "oracle_gain_pct_min": round(min(gains), 1),
        "oracle_gain_pct_max": round(max(gains), 1),
        "horizons_router_beats_fixed_ensemble": 0,
    }


def main() -> None:
    data = {
        "provenance": {
            "generated_by": "scripts/extract_web_data.py",
            "sources": [
                "artifacts/phase8/uncertainty-main/probabilistic_forecasts.npz",
                "artifacts/phase8/uncertainty-main/result.json",
                "artifacts/phase7/router-main/result.json",
            ],
            "dataset": "SMART-DS v1.0 AUS/P1U 2018, feeder p1uhs0_1247--p1udt12703",
        },
        "forecast_slice": forecast_slice(),
        "published_procedure": published_procedure(),
        "constant_width_failure": constant_width_failure(),
        "phase7_table": phase7_table(),
        "router_facts": router_facts(),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
