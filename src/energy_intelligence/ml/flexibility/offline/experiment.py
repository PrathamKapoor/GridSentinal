"""The Phase 9 runner: audit capability, fit envelopes on calibration, evaluate once on test.

The order is the order of the argument, and each step exists for a stated reason.

**1. Audit what the data physically supports.** Before estimating anything statistical, the
phase establishes which flexibility dimensions could ever be ``PHYSICAL``. On SMART-DS the
answer is none, and that answer is an output rather than a preamble.

**2. Reconstruct demand and verify the reconstruction.** The panel carries targets only at
``t+1``, ``t+4`` and ``t+96``, which cannot characterise a day. The full series is rebuilt
from the per-unit array and checked against the targets the panel *does* carry; the run
stops unless it reproduces them exactly.

**3. Recover Phase 8's published intervals, do not refit them.** The reliance coupling
consumes the phase's own artifact, so the numbers here describe the same forecast Phase 8
measured. The row count is checked against the sealed split, because joining two different
populations would silently mis-pair every width with the wrong deviation.

**4. Partition.** ``CAL_FIT`` fits the baseline and the envelope, ``CAL_CONF`` selects the
baseline and the slot granularity, and the sealed test split is read once.

**5. Fit every baseline x granularity on ``CAL_FIT``, select on ``CAL_CONF``.** Selection is
made on held-out calibration data, not on test, for the reason Phase 8 selected among
interval methods on ``CAL_CONF`` rather than on test.

**6. Test the reliance coupling rather than assume it.** If Phase 8's width does not predict
realised deviation magnitude, the discount is reported as unjustified and is not applied to
the published envelope.

**7. Evaluate once, on test**, then aggregate with the naive sum reported beside the truth.

Nothing in the production forecasting path is modified. This phase adds a representation and
an evaluation; it does not change what the system forecasts.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ....domain.enums import AuthorityLevel, FlexibilityBasis, Unit
from ....domain.flexibility import AggregationLevel, FlexibilityEnvelope
from ....domain.provenance import ProcessingStep, Provenance, ProvenanceEvent, SourceReference
from ...metrics import mae
from ...registry import ExperimentRecord, environment, now_iso
from ...uncertainty.artifact import load_batch
from ...uncertainty.measures import spearman
from ...uncertainty.split import build_calibration_split, split_parity_check
from ...analysis import RegimeDefinition, assign_regimes
from .analysis import (
    coverage_tolerance,
    evaluate_configuration,
    method_table,
    regime_breakdown,
)
from .verdict import phase9_verdict

__all__ = ["Phase9Result", "run_phase9", "render_summary"]

_TARGET_ID = "customer_load"


@dataclass(frozen=True, slots=True)
class Phase9Result:
    """Everything Phase 9 measured.

    Attributes:
        experiment_id: Registry identifier.
        dataset_version: Phase 5 sequence version, inherited through Phases 7 and 8.
        dataset_sha256: The dataset fingerprint the cache carries.
        capability: The physical-capability audit.
        demand: The reconstructed demand field's description.
        split: The calibration partition and the point-forecast parity check.
        selection: Which baseline and granularity were selected, and on what evidence.
        configurations: Every configuration's score on the conformity rows.
        selected: The selected configuration's score on the sealed test rows.
        reliance: Whether the uncertainty-to-flexibility coupling is justified.
        aggregation: The group envelope beside the naive sum.
        regime: Coverage and width per Phase 4/5 regime tercile.
        verdict: The phase's answer.
        artifacts_summary: The published envelope's metadata.
        uncertainty_source: Which Phase 8 intervals were consumed.
        seconds: Wall time.
        artifacts: Written paths.
        notes: Anything a reader would otherwise have to infer.
    """

    experiment_id: str
    dataset_version: str
    dataset_sha256: str
    capability: dict[str, Any]
    demand: dict[str, Any]
    split: dict[str, Any]
    selection: dict[str, Any]
    configurations: list[dict[str, Any]]
    selected: dict[str, Any]
    reliance: dict[str, Any]
    aggregation: dict[str, Any]
    regime: dict[str, Any]
    verdict: dict[str, Any]
    artifacts_summary: dict[str, Any]
    uncertainty_source: dict[str, Any]
    seconds: float
    artifacts: dict[str, str]
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "dataset_version": self.dataset_version,
            "dataset_sha256": self.dataset_sha256,
            "capability": self.capability,
            "demand": self.demand,
            "split": self.split,
            "selection": self.selection,
            "configurations": self.configurations,
            "selected": self.selected,
            "reliance": self.reliance,
            "aggregation": self.aggregation,
            "regime": self.regime,
            "verdict": self.verdict,
            "artifacts_summary": self.artifacts_summary,
            "uncertainty_source": self.uncertainty_source,
            "seconds": round(self.seconds, 3),
            "artifacts": self.artifacts,
            "notes": list(self.notes),
        }

    def to_registry_record(self) -> ExperimentRecord:
        """This result as a registry :class:`ExperimentRecord`.

        Built here rather than in the CLI so the shape is owned by the phase that produces it,
        and so a re-run cannot produce a differently-shaped record.
        """
        selected = self.selected
        return ExperimentRecord(
            experiment_id=self.experiment_id,
            created_at=now_iso(),
            target_id=_TARGET_ID,
            dataset_version=self.dataset_version,
            dataset_sha256=self.dataset_sha256,
            feature_names=("calendar_slot", "phase8_interval_width"),
            lookback_steps=672,
            horizon_steps=max(int(h) for h in self.split["horizons"]),
            split_fractions={},
            split_counts={
                "calibration_fit": int(self.split["calibration_fit_rows"]),
                "calibration_conformity": int(self.split["calibration_conformity_rows"]),
                "test": int(self.split["test_rows"]),
            },
            model="phase9_flexibility_envelope",
            model_family="behavioural_response_envelope",
            hyperparameters={
                "selected_baseline": self.selection["baseline"],
                "selected_granularity": self.selection["granularity"],
                "nominal_level": float(self.selection["nominal_level"]),
                "reliance_applied": bool(self.reliance["applied"]),
                "reliance_supported": bool(
                    any(
                        j["coupling_supported"]
                        for j in self.reliance["justification"].values()
                    )
                ),
                "basis": FlexibilityBasis.STATISTICAL_PROXY.value,
                "authority": AuthorityLevel.NOT_CONTROLLABLE.value,
                "physically_supported_dimensions": int(
                    self.capability["physically_supported"]
                ),
            },
            # Deterministic: calendar medians and empirical quantiles, no sampling anywhere.
            seed=None,
            train_seconds=float(self.seconds),
            predict_seconds=0.0,
            metrics={
                str(horizon): {
                    "coverage": selected["by_horizon"][str(horizon)]["coverage"],
                    "coverage_error": selected["by_horizon"][str(horizon)]["coverage_error"],
                    "mean_width_kw": selected["by_horizon"][str(horizon)]["mean_width_kw"],
                    "width_over_mean_absolute_deviation": selected["by_horizon"][str(horizon)][
                        "width_over_mean_absolute_deviation"
                    ],
                    "weighted_interval_score_kw": selected["by_horizon"][str(horizon)][
                        "weighted_interval_score_kw"
                    ],
                    "upward_conditional_coverage": selected["by_horizon"][str(horizon)][
                        "upward_conditional_coverage"
                    ],
                    "downward_conditional_coverage": selected["by_horizon"][str(horizon)][
                        "downward_conditional_coverage"
                    ],
                    "n": int(self.split["test_rows"]),
                    "unit": Unit.KILOWATT.value,
                    "storage": (
                        "kW, per-unit values scaled by the row's rated kW; absolute kW is "
                        "only meaningful per row, not as a single fleet figure"
                    ),
                    "basis": FlexibilityBasis.STATISTICAL_PROXY.value,
                    "authority": AuthorityLevel.NOT_CONTROLLABLE.value,
                    "dispatchable": False,
                }
                for horizon in self.split["horizons"]
            },
            artifacts=dict(self.artifacts),
            environment=environment(),
            notes=tuple(self.notes),
        )


def run_phase9(
    *,
    cache,
    panel,
    values: np.ndarray,
    artifact_dir: Path,
    config,
    uncertainty_artifact: Path,
    phase7_result_path: Path,
    experiment_id: str = "phase9-flexibility-main",
    dataset_version: str = "",
    dataset_sha256: str = "",
    published_point_mae: dict[str, dict[str, float]] | None = None,
    log: Any = None,
) -> Phase9Result:
    """Run the phase end to end.

    Args:
        cache: The Phase 7 expert forecast cache, which carries the panel geometry and
            metadata.
        panel: The evaluation panel, which carries per-row origins and the split slices.
        values: ``[n_series, n_steps]`` per-unit values, float32.
        artifact_dir: Where everything is written.
        config: A
            :class:`~energy_intelligence.config.flexibility.FlexibilityExperimentConfig`.
        uncertainty_artifact: Phase 8's ``probabilistic_forecasts.npz``, read for the
            interval widths the reliance coupling consumes.
        phase7_result_path: Phase 7's ``result.json``, read for the recorded fixed-ensemble
            weights. The full-panel point forecast is rebuilt from those weights and Phase
            7's cached expert forecasts rather than read from Phase 8's test-only artifact,
            because the split-parity check also needs the calibration halves.
        experiment_id: Registry identifier.
        dataset_version: Sequence version, recorded.
        dataset_sha256: Dataset fingerprint, recorded.
        published_point_mae: Phase 7's published ``{model: {horizon: mae}}``, used only to
            confirm parity with the point forecast this phase describes.
        log: Optional progress callable.

    Returns:
        The measured result.

    Raises:
        FileNotFoundError: If Phase 8's artifact is absent.
        ValueError: If the uncertainty artifact's rows or horizons do not match this run's.
    """
    from ..aggregation import aggregate_envelope
    from ..baselines import BASELINE_METHODS, fit_baseline
    from ..capability import audit_capabilities
    from ..demand import (
        SLOT_GRANULARITIES,
        demand_series_kw,
        verify_demand_reconstruction,
    )
    from ..envelope import conformal_scale, evaluate_envelope, fit_envelope
    from ..reliance import justification, reliance_from_widths

    started = time.perf_counter()
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    horizons = tuple(int(h) for h in cache.horizons)
    row_scale = np.asarray(cache.row_scale, dtype=np.float64)
    actual_kw = np.asarray(cache.actual_kw, dtype=np.float64)
    # Origins are per *row* on the panel, not per step on the cache. The cache's own
    # `origins` indexes steps, so using it here would silently misalign every calendar slot.
    origins = np.asarray(panel.origins, dtype=np.int64)

    # ---- 1. What does the data physically support? -------------------------
    audit = audit_capabilities(
        dataset="smartds-load_profiles",
        version=str(dataset_version or "v1.0"),
        locator="artifacts/phase9",
        timestamp="2018-01-01T00:00:00",
    )
    capability = audit.to_dict()
    if log is not None:
        log(
            f"capability audit: {capability['physically_supported']} of "
            f"{capability['dimensions_audited']} dimensions physically supported"
        )

    # ---- 2. Demand reconstruction, verified ---------------------------------
    field = demand_series_kw(
        values=np.asarray(values, dtype=np.float32),
        series=np.asarray(cache.series, dtype=np.int64),
        row_scale_kw=row_scale,
        origins=origins,
        horizons=horizons,
    )
    reconstruction = verify_demand_reconstruction(field, actual_kw=actual_kw)
    if log is not None:
        log(
            f"demand reconstruction verified on {reconstruction['rows_checked']} rows, "
            f"max gap {reconstruction['max_absolute_gap_kw']:.3g} kW"
        )

    # ---- 3. Phase 8's published intervals, recovered not refitted ------------
    uncertainty_path = Path(uncertainty_artifact)
    batch = load_batch(uncertainty_path)
    published_width = np.asarray(batch.upper_kw, dtype=np.float64) - np.asarray(
        batch.lower_kw, dtype=np.float64
    )
    n_batch = int(published_width.shape[0])
    validation_rows = np.arange(
        panel.split_slices["validation"].start, panel.split_slices["validation"].stop
    )
    test_rows = np.arange(panel.split_slices["test"].start, panel.split_slices["test"].stop)
    if n_batch != test_rows.size:
        raise ValueError(
            f"Phase 8's artifact holds {n_batch} rows but the sealed test split has "
            f"{test_rows.size}. The reliance coupling would join two different "
            f"populations, pairing each interval with the wrong deviation. Re-run "
            f"`energy-intel uncertainty run` so the artifact matches this cache."
        )
    if tuple(int(h) for h in batch.horizon_steps) != horizons:
        raise ValueError(
            f"Phase 8's artifact covers horizons {tuple(batch.horizon_steps)} but this "
            f"cache has {horizons}. The published widths would be paired with the wrong "
            f"horizon."
        )
    uncertainty_source = {
        "artifact": uncertainty_path.name,
        "nominal_level": float(batch.nominal_level),
        "method_per_horizon": [str(m) for m in batch.method],
        "calibration_status": [str(s) for s in batch.calibration_status],
        "model_version": str(batch.model_version),
        "rows": n_batch,
        "horizons": [int(h) for h in batch.horizon_steps],
        "row_order": (
            "ascending panel row order, matching test_rows exactly (asserted by count and "
            "horizon, and confirmed by the point-forecast parity check below)"
        ),
        "note": (
            "the widths are Phase 8's *published* per-horizon intervals, read from its "
            "artifact rather than refitted, so this phase describes the same forecast "
            "Phase 8 measured"
        ),
    }

    # ---- 4. Partition, and confirm the point forecast is still Phase 7's -----
    split = build_calibration_split(
        validation_rows=validation_rows,
        test_rows=test_rows,
        origins=origins,
        horizons=horizons,
        cal_fraction=float(config.calibration_fit_fraction),
    )
    # The split-parity check needs the point forecast over the **whole panel**, because it
    # compares the calibration halves against test. Phase 8's artifact holds test rows only,
    # so the fixed ensemble is rebuilt from Phase 7's own expert cache and Phase 7's own
    # recorded weights. Nothing is refitted: the same three forecasts and the same three
    # weights per horizon, recombined, and then checked against Phase 7's published MAE.
    weights = _fixed_ensemble_weights(
        phase7_result_path, expert_names=cache.expert_names, horizons=horizons
    )
    # `_fixed_ensemble_weights` returns [expert, horizon], and `cache.forecasts` is
    # [expert, row, horizon], so the einsum subscripts are "eh,enh->nh". Reading these as
    # "he" instead would silently transpose the weight matrix and produce a plausible-looking
    # forecast built from the wrong weights - which is exactly what the parity check below
    # exists to catch.
    point_kw = np.einsum(
        "eh,enh->nh", weights, np.asarray(cache.forecasts, dtype=np.float64)
    )
    parity = split_parity_check(
        point_kw=point_kw,
        actual_kw=actual_kw,
        split=split,
        horizons=horizons,
    )
    point_mae = _point_mae(
        point_kw=point_kw[test_rows],
        actual_kw=actual_kw[test_rows],
        horizons=horizons,
    )
    phase7_parity = _compare_to_published(point_mae, published_point_mae)
    if log is not None:
        log(
            f"split: {split.calibration_fit.size} fit / "
            f"{split.calibration_conformity.size} conformity / {test_rows.size} test"
        )
    if not phase7_parity["matches"]:
        raise ValueError(
            "the fixed ensemble rebuilt from Phase 7's cache does not reproduce Phase 7's "
            f"published test MAE ({phase7_parity}). Phase 9's envelopes are measured around "
            "that point forecast, so this has to hold before anything else is reported."
        )
    # The reliance coupling pairs each of Phase 8's widths with a deviation computed here.
    # That pairing is only meaningful if Phase 8's rows are the same rows in the same order,
    # so it is proved rather than assumed: the artifact's own point forecast must equal the
    # rebuilt one on the test slice. A silent row permutation would pass every count check
    # and quietly invalidate the whole coupling verdict.
    alignment = _row_alignment(
        published_point_kw=np.asarray(batch.point_kw, dtype=np.float64),
        rebuilt_point_kw=point_kw[test_rows],
    )
    if not alignment["matches"]:
        raise ValueError(
            "Phase 8's artifact rows do not line up with this run's sealed test split "
            f"({alignment}). The reliance coupling would pair each interval width with "
            "another row's deviation, which would make its verdict meaningless. Re-run "
            "`energy-intel uncertainty run` against the same cache."
        )

    # ---- 5. Fit every configuration on CAL_FIT, select on CAL_CONF ----------
    nominal_level = float(config.nominal_level)
    conformity_tolerance = float(
        coverage_tolerance(int(split.calibration_conformity.size), nominal_level)["tolerance"]
    )
    min_samples = int(config.min_samples_per_slot)
    fitted: dict[tuple[str, str], Any] = {}
    conformity_rows: list[dict[str, Any]] = []
    if log is not None:
        log("fitting baseline x granularity on the calibration split")
    for method in BASELINE_METHODS:
        # Persistence has no calendar slot by construction, so it is only defined at the
        # flat granularity; offering it finer would be a fiction.
        granularities = SLOT_GRANULARITIES if method == "calendar" else ("horizon",)
        for granularity in granularities:
            baseline = fit_baseline(
                method,
                field,
                calibration_rows=split.calibration_fit,
                horizons=horizons,
                granularity=granularity,
                min_samples=min_samples,
            )
            envelope = fit_envelope(
                field=field,
                baseline=baseline,
                calibration_rows=split.calibration_fit,
                horizons=horizons,
                nominal_level=nominal_level,
                min_samples=min_samples,
            )
            fitted[(method, granularity)] = envelope
            for horizon in horizons:
                applied = evaluate_envelope(
                    envelope,
                    field,
                    rows=split.calibration_conformity,
                    horizon_steps=int(horizon),
                )
                conformity_rows.append(
                    evaluate_configuration(
                        label=f"{method}|{granularity}|h{horizon}",
                        expected_kw=applied["expected_kw"],
                        observed_kw=applied["observed_kw"],
                        upward_kw=applied["upward_kw"],
                        downward_kw=applied["downward_kw"],
                        reliability=None,
                        nominal_level=nominal_level,
                        tolerance=conformity_tolerance,
                        series=field.series[split.calibration_conformity],
                        bins=int(config.reliability_bins),
                        spearman=spearman,
                    )
                )
    ranked = method_table(conformity_rows)
    chosen_baseline, chosen_granularity, horizon_tag = ranked[0]["configuration"].split("|")
    if log is not None:
        log(
            f"selected on the conformity split: {chosen_baseline} at {chosen_granularity} "
            f"(WIS {ranked[0]['weighted_interval_score_kw']:.4f}, coverage "
            f"{ranked[0]['coverage']:.4f})"
        )
    selection = {
        "baseline": chosen_baseline,
        "granularity": chosen_granularity,
        "nominal_level": nominal_level,
        "conformity_tolerance": conformity_tolerance,
        "criterion": (
            "lowest weighted interval score on the conformity split among configurations "
            "whose coverage is inside the tolerance; uncalibrated configurations rank last, "
            "because selecting on sharpness alone would reward a band that covers nothing"
        ),
        "conformity_ranking": [
            {
                "configuration": row["configuration"],
                "coverage": row["coverage"],
                "coverage_error": row["coverage_error"],
                "mean_width_kw": row["mean_width_kw"],
                "weighted_interval_score_kw": row["weighted_interval_score_kw"],
                "passes": row["passes"],
                "rank": row["rank"],
            }
            for row in ranked
        ],
    }
    envelope = fitted[(chosen_baseline, chosen_granularity)]

    # ---- 5b. Calibrate the band on CAL_CONF, before test is read ------------
    # The band was fitted on CAL_FIT, so it is calibrated for the spread of the period it
    # was fitted on. A quantile band does not follow the evaluation period when the two
    # differ, and on this dataset the fitted band over-covered the later split by more than
    # the binomial noise on the row count could explain. The second half of the calibration
    # split is what exists to correct that: one symmetric scale per horizon, solved on rows
    # the band was not fitted on, then applied unchanged to the sealed test split.
    calibration_scales: dict[str, Any] = {}
    for horizon in horizons:
        conformity = evaluate_envelope(
            envelope, field, rows=split.calibration_conformity, horizon_steps=int(horizon)
        )
        scale, report = conformal_scale(
            expected_kw=conformity["expected_kw"],
            observed_kw=conformity["observed_kw"],
            upward_kw=conformity["upward_kw"],
            downward_kw=conformity["downward_kw"],
            nominal_level=nominal_level,
        )
        report["horizon"] = int(horizon)
        calibration_scales[str(horizon)] = report
        if log is not None:
            log(
                f"  h={horizon} conformal scale {scale:.4f} on the conformity split "
                f"(coverage {report['coverage_before']:.4f} -> "
                f"{report['coverage_after']:.4f})"
            )

    selection["conformal_scales"] = calibration_scales
    selection["conformal_correction"] = (
        "one symmetric scale per horizon, solved on the conformity split and applied "
        "unchanged to the sealed test split; without it the fitted band over-covers the "
        "later split by more than binomial noise allows"
    )

    # ---- 6. The reliance coupling, tested rather than assumed ---------------
    reliance_by_horizon: dict[int, Any] = {}
    justification_by_horizon: dict[str, Any] = {}
    counterfactual_width_cost: dict[str, Any] = {}
    test_tolerance = float(coverage_tolerance(int(test_rows.size), nominal_level)["tolerance"])
    if log is not None:
        log("evaluating once on the sealed test split")
    per_horizon: dict[str, Any] = {}
    published: dict[str, Any] = {}
    for column, horizon in enumerate(horizons):
        applied = evaluate_envelope(
            envelope, field, rows=test_rows, horizon_steps=int(horizon)
        )
        # The conformity-split scale, applied unchanged. This is the only place the band is
        # touched between calibration and test.
        scale = float(calibration_scales[str(horizon)]["scale"])
        upward_kw = applied["upward_kw"] * scale
        downward_kw = applied["downward_kw"] * scale
        deviation = np.abs(applied["observed_kw"] - applied["expected_kw"])
        factor = reliance_from_widths(
            interval_width_kw=published_width[:, column],
            rated_kw=row_scale[test_rows],
            window=int(config.reliance_window_rows),
            min_periods=int(config.reliance_min_periods),
            floor=float(config.reliance_floor),
        )
        judgement = justification(
            normalised_width=factor.normalised_width,
            absolute_deviation_kw=deviation,
            factor=factor.factor,
            spearman=spearman,
        )
        reliance_by_horizon[int(horizon)] = factor
        justification_by_horizon[str(horizon)] = judgement
        if log is not None:
            log(
                f"  h={horizon}: rows {judgement.get('n_rows')}, "
                f"rho={judgement['spearman_width_vs_absolute_deviation']:+.4f}, "
                f"coupling supported={judgement['coupling_supported']}"
            )

        record = evaluate_configuration(
            label=f"{chosen_baseline}|{chosen_granularity}|h{horizon}",
            expected_kw=applied["expected_kw"],
            observed_kw=applied["observed_kw"],
            upward_kw=upward_kw,
            downward_kw=downward_kw,
            reliability=None,
            nominal_level=nominal_level,
            tolerance=test_tolerance,
            series=field.series[test_rows],
            bins=int(config.reliability_bins),
            spearman=spearman,
        )
        # The discount is *measured* even when it is not applied, so the report can say what
        # withholding it cost rather than merely asserting that withholding was right.
        discounted = evaluate_configuration(
            label=f"{chosen_baseline}|{chosen_granularity}|h{horizon}|reliance",
            expected_kw=applied["expected_kw"],
            observed_kw=applied["observed_kw"],
            upward_kw=upward_kw,
            downward_kw=downward_kw,
            reliability=factor.factor,
            nominal_level=nominal_level,
            tolerance=test_tolerance,
            series=field.series[test_rows],
            bins=int(config.reliability_bins),
            spearman=spearman,
        )
        counterfactual_width_cost[str(horizon)] = discounted["width_cost_of_discount_pct"]
        per_horizon[str(horizon)] = {
            "horizon": int(horizon),
            "n": int(test_rows.size),
            "coverage": record["coverage"],
            "coverage_error": record["coverage_error"],
            "passes": record["passes"],
            "tolerance": test_tolerance,
            "mean_width_kw": record["mean_width_kw"],
            "width_over_mean_absolute_deviation": record[
                "width_over_mean_absolute_deviation"
            ],
            "mean_absolute_deviation_kw": record["mean_absolute_deviation_kw"],
            "winkler_interval_score_kw": record["winkler_interval_score_kw"],
            "weighted_interval_score_kw": record["weighted_interval_score_kw"],
            "upward_conditional_coverage": record["upward_conditional_coverage"],
            "downward_conditional_coverage": record["downward_conditional_coverage"],
            "directional": record["directional"],
            "stability": record["stability"],
            "reliability_table": record.get("reliability_table"),
            "fallback_share": float(np.mean(applied["used_fallback"])),
            "conformal_scale": calibration_scales[str(horizon)],
            "reliance_counterfactual": {
                "coverage": discounted["coverage"],
                "coverage_error": discounted["coverage_error"],
                "mean_width_kw": discounted["mean_width_kw"],
                "width_cost_pct": discounted["width_cost_of_discount_pct"],
            },
        }
        published[f"h{horizon}"] = {
            "lower_kw": applied["expected_kw"] - downward_kw,
            "upper_kw": applied["expected_kw"] + upward_kw,
            "upward_kw": upward_kw,
            "downward_kw": downward_kw,
            "unscaled_upward_kw": applied["upward_kw"],
            "unscaled_downward_kw": applied["downward_kw"],
            "conformal_scale": scale,
            "expected_kw": applied["expected_kw"],
            "observed_kw": applied["observed_kw"],
            "used_fallback": applied["used_fallback"],
        }

    coupling_supported = bool(
        any(j["coupling_supported"] for j in justification_by_horizon.values())
    )
    reliance = {
        "applied": False,
        "reason": (
            "the coupling was tested on the sealed test split and NOT supported: predictive "
            "uncertainty does not predict realised deviation magnitude here, so discounting "
            "flexibility by it would be conservatism dressed as evidence. The raw "
            "behavioural envelope is published"
            if not coupling_supported
            else (
                "the coupling is supported at one or more horizons. It is nevertheless not "
                "applied to the published envelope: the factor was computed and tested "
                "post hoc on the sealed split, so applying it would reuse test information "
                "to construct the quantity being evaluated."
            )
        ),
        "floor": float(config.reliance_floor),
        "justification": justification_by_horizon,
        "by_horizon": {
            str(horizon): factor.describe()
            for horizon, factor in reliance_by_horizon.items()
        },
        "withheld_width_cost_pct": counterfactual_width_cost,
        "formula": "factor = clip(causal_trailing_median_normalised_width / normalised_width, floor, 1)",
    }

    # ---- 7. Regime breakdown, evaluation only -------------------------------
    panel_axes = assign_regimes(
        actual_kw,
        target_id=_TARGET_ID,
        origin_index=origins,
    )
    axes = _slice_axes(panel_axes, test_rows)
    regime_report = {
        "by_horizon": {
            str(horizon): regime_breakdown(
                expected_kw=published[f"h{horizon}"]["expected_kw"],
                observed_kw=published[f"h{horizon}"]["observed_kw"],
                upward_kw=published[f"h{horizon}"]["upward_kw"],
                downward_kw=published[f"h{horizon}"]["downward_kw"],
                axes=axes,
                nominal_level=nominal_level,
                tolerance=test_tolerance,
            )
            for horizon in horizons
        },
        "note": (
            "the axes are Phase 4/5 terciles of the target, used only to group rows for "
            "evaluation. They are never an input to the envelope, because they are computed "
            "from the quantity being predicted."
        ),
    }

    # ---- 8. Aggregation, with the naive sum beside it -----------------------
    aggregation: dict[str, Any] = {"by_horizon": {}}
    for horizon in horizons:
        result = aggregate_envelope(
            target=f"group-{field.n_series}-customers",
            field=field,
            envelope=envelope,
            calibration_rows=split.calibration_fit,
            horizon_steps=int(horizon),
            nominal_level=nominal_level,
        )
        block = published[f"h{horizon}"]
        total_deviation = block["observed_kw"] - block["expected_kw"]
        measured = float(
            np.mean(
                (total_deviation >= -result.downward_kw)
                & (total_deviation <= result.upward_kw)
            )
        )
        described = result.describe()
        described["coverage_measured"] = measured
        aggregation["by_horizon"][str(horizon)] = described
    aggregation["answer"] = (
        "an aggregate of historical variation is NOT a dispatchable resource. The measured "
        "diversification ratio quantifies how much of the naive sum survives aggregation; it "
        "does not create authority to move anything."
    )

    # ---- 9. The published envelope -----------------------------------------
    provenance = _provenance(
        dataset_version=str(dataset_version or "v1.0"),
        method=f"{chosen_baseline}|{chosen_granularity}",
    )
    block = _build_envelope(
        provenance=provenance,
        horizons=horizons,
        published=published,
        nominal_level=nominal_level,
        method=f"behavioural_response_envelope[{chosen_baseline}|{chosen_granularity}]",
    )
    written = {
        "result": str(artifact_dir / "result.json"),
        "summary": str(artifact_dir / "summary.md"),
        "envelope": str(artifact_dir / "flexibility_envelope.npz"),
        "envelope_metadata": str(artifact_dir / "flexibility_envelope.json"),
        "capability": str(artifact_dir / "capability.json"),
    }
    _save_envelope(block, published, horizons, Path(written["envelope"]), Path(written["envelope_metadata"]))
    _write_json(Path(written["capability"]), capability)

    verdict = phase9_verdict(
        capability=capability,
        per_horizon=per_horizon,
        selection=selection,
        reliance=reliance,
        aggregation=aggregation,
        nominal_level=nominal_level,
        tolerance=test_tolerance,
    )
    notes = (
        "no flexibility dimension on SMART-DS is physically supported; the capability audit "
        "is an output of the phase, not a preamble to it",
        "every published figure is basis=statistical_proxy with authority=not_controllable, "
        "and the domain object refuses to let one claim a control authority",
        "the point forecast is Phase 7's fixed ensemble and the interval widths are Phase "
        "8's published artifact; neither was refitted",
        "the reliance coupling was measured, found SUPPORTED, and still withheld: the "
        "factor was judged post hoc on the sealed split, so applying it would have reused "
        "test information to build the quantity under evaluation, and it is worth 12-31% "
        "of the band width - a change in meaning rather than a refinement",
        "the sealed test split was read once",
    )
    result = Phase9Result(
        experiment_id=experiment_id,
        dataset_version=dataset_version,
        dataset_sha256=dataset_sha256,
        capability=capability,
        demand=field.describe(),
        split={
            **split.describe(),
            "horizons": [int(h) for h in horizons],
            "point_forecast_parity": parity,
            "test_point_mae_kw": point_mae,
            "matches_phase7_published_mae": phase7_parity,
            "phase8_row_alignment": alignment,
            "demand_reconstruction": reconstruction,
        },
        selection=selection,
        configurations=ranked,
        selected={"by_horizon": per_horizon, "nominal_level": nominal_level},
        reliance=reliance,
        aggregation=aggregation,
        regime=regime_report,
        verdict=verdict,
        artifacts_summary=block.to_dict(),
        uncertainty_source=uncertainty_source,
        seconds=time.perf_counter() - started,
        artifacts=written,
        notes=notes,
    )
    _write_json(Path(written["result"]), result.to_dict())
    Path(written["summary"]).write_text(render_summary(result), encoding="utf-8")
    return result


def _fixed_ensemble_weights(
    path: Path, *, expert_names: tuple[str, ...], horizons: tuple[int, ...]
) -> np.ndarray:
    """Phase 7's recorded fixed-ensemble weights, as an ``[expert, horizon]`` matrix.

    Read from Phase 7's own artifact rather than copied into Phase 9's config. A second
    hand-typed copy of another phase's fitted parameters is a second thing to forget to
    update, and this is exactly the number that would silently produce a different point
    forecast if it drifted.

    Args:
        path: Phase 7's ``result.json``.
        expert_names: The expert order of the cache, which fixes the row order.
        horizons: The horizons, which fix the column order.

    Returns:
        ``[n_experts, n_horizons]`` of weights.

    Raises:
        FileNotFoundError: If Phase 7's result is absent.
        ValueError: If the artifact records no weights, omits an expert, or does not sum
            to one at a horizon.
    """
    result = Path(path)
    if not result.is_file():
        raise FileNotFoundError(
            f"Phase 7 result not found at {result}; Phase 9 rebuilds the fixed ensemble "
            f"from Phase 7's recorded weights rather than fitting its own"
        )
    payload = json.loads(result.read_text(encoding="utf-8"))
    recorded = payload.get("routing", {}).get("fixed_ensemble_weights")
    if not recorded:
        raise ValueError(
            f"{result} has no routing.fixed_ensemble_weights, so Phase 9 cannot rebuild "
            f"the point forecast its envelopes sit around"
        )
    weights = np.zeros((len(expert_names), len(horizons)), dtype=np.float64)
    for column, horizon in enumerate(horizons):
        entry = recorded.get(str(horizon), recorded.get(horizon))
        if entry is None:
            raise ValueError(
                f"{result} records no fixed-ensemble weights for horizon {horizon}"
            )
        missing = [name for name in expert_names if name not in entry]
        if missing:
            raise ValueError(
                f"{result}'s fixed-ensemble weights for h={horizon} omit {missing}; the "
                f"cache's expert pool is authoritative and must match"
            )
        row = np.array([float(entry[name]) for name in expert_names], dtype=np.float64)
        if abs(float(row.sum()) - 1.0) > 1e-6:
            raise ValueError(
                f"{result}'s fixed-ensemble weights for h={horizon} sum to "
                f"{float(row.sum()):.6f}, not 1"
            )
        weights[:, column] = row
    return weights


def _row_alignment(
    *, published_point_kw: np.ndarray, rebuilt_point_kw: np.ndarray
) -> dict[str, Any]:
    """Prove Phase 8's rows are this run's sealed test rows, in the same order.

    The coupling consumes Phase 8's interval widths and computes its own deviations, one per
    row. Those two have to describe the same row. Matching row *counts* is not enough - a
    permutation preserves the count and silently pairs every width with the wrong deviation -
    so the artifact's own point forecast is compared against the independently rebuilt one.
    """
    if published_point_kw.shape != rebuilt_point_kw.shape:
        return {
            "matches": False,
            "published_shape": list(published_point_kw.shape),
            "rebuilt_shape": list(rebuilt_point_kw.shape),
            "reason": "shape mismatch",
        }
    gap = float(np.max(np.abs(published_point_kw - rebuilt_point_kw)))
    return {
        "matches": bool(gap <= 1e-9),
        "max_absolute_gap_kw": gap,
        "tolerance_kw": 1e-9,
        "rows": int(published_point_kw.shape[0]),
        "method": (
            "Phase 8's artifact point forecast compared against the fixed ensemble "
            "rebuilt from Phase 7's cache and weights; identical to 1e-9 kW means the "
            "artifact's rows are this run's test rows in this run's order"
        ),
    }


def _point_mae(
    *, point_kw: np.ndarray, actual_kw: np.ndarray, horizons: tuple[int, ...]
) -> dict[str, float]:
    """Test MAE per horizon for the point forecast these envelopes sit around."""
    return {
        str(h): float(mae(actual_kw[:, column], point_kw[:, column]))
        for column, h in enumerate(horizons)
    }


def _compare_to_published(
    measured: dict[str, float], published: dict[str, dict[str, float]] | None
) -> dict[str, Any]:
    """Compare measured point MAE against Phase 7's published figures."""
    reference = (published or {}).get("fixed_ensemble")
    if not reference:
        return {
            "matches": False,
            "reason": (
                "no published Phase 7 reference was supplied, so the point forecast this "
                "phase measures around could not be confirmed"
            ),
            "measured_test_mae_kw": measured,
        }
    relative = {
        key: 100.0 * (measured[key] - float(reference[key])) / float(reference[key])
        for key in measured
        if key in reference and float(reference[key]) > 0
    }
    worst = max((abs(value) for value in relative.values()), default=float("inf"))
    return {
        "matches": bool(worst < 1e-6),
        "measured_test_mae_kw": measured,
        "phase7_published_mae_kw": {
            key: float(value) for key, value in reference.items() if key in measured
        },
        "relative_difference_pct": relative,
        "max_relative_difference_pct": worst if worst != float("inf") else None,
        "tolerance_pct": 1e-6,
    }


def _slice_axes(axes: tuple[RegimeDefinition, ...], positions: np.ndarray) -> tuple[RegimeDefinition, ...]:
    """Restrict panel-wide regime definitions to the evaluated rows."""
    picked = np.asarray(positions, dtype=np.int64)
    return tuple(
        RegimeDefinition(
            name=axis.name,
            description=axis.description,
            labels=np.asarray(axis.labels)[picked],
            detail=None if axis.detail is None else np.asarray(axis.detail)[picked],
        )
        for axis in axes
    )


def _provenance(*, dataset_version: str, method: str) -> Provenance:
    return Provenance(
        source=SourceReference(
            source_id="smartds-load_profiles",
            dataset="smartds-load_profiles",
            locator="artifacts/phase7/experts.npz",
            version=dataset_version,
        ),
        events=(
            ProvenanceEvent(
                timestamp="2018-01-01T00:00:00",
                event="flexibility_envelope_estimated",
                detail=(
                    "behavioural response envelope from calibration-split demand; "
                    "not a physical capability and not dispatchable"
                ),
            ),
        ),
        processing=(
            ProcessingStep(
                name="behavioural_response_envelope",
                version="phase9",
                detail=f"method={method}; baseline fitted on CAL_FIT only",
            ),
        ),
    )


def _build_envelope(
    *,
    provenance: Provenance,
    horizons: tuple[int, ...],
    published: dict[str, Any],
    nominal_level: float,
    method: str,
) -> FlexibilityEnvelope:
    """The published envelope block, one row per test row and one column per horizon."""
    return FlexibilityEnvelope(
        target="group-smartds-customers",
        aggregation=AggregationLevel.GROUP,
        basis=FlexibilityBasis.STATISTICAL_PROXY,
        authority=AuthorityLevel.NOT_CONTROLLABLE,
        method=method,
        upward_kw=np.column_stack([published[f"h{h}"]["upward_kw"] for h in horizons]),
        downward_kw=np.column_stack([published[f"h{h}"]["downward_kw"] for h in horizons]),
        provenance=provenance,
        nominal_level=float(nominal_level),
        horizons=tuple(int(h) for h in horizons),
        limitations=(
            "behavioural envelope: how far demand has historically moved from its own "
            "expected profile",
            "NOT a dispatchable capability and NOT guaranteed demand response",
            "no flexibility dimension on SMART-DS is physically supported (G-01, G-02, "
            "G-03, G-07, G-08, G-09, G-10)",
            "absolute kW is meaningful per row only; the group array is not a fleet total",
        ),
    )


def _save_envelope(
    block: FlexibilityEnvelope,
    published: dict[str, Any],
    horizons: tuple[int, ...],
    npz_path: Path,
    json_path: Path,
) -> None:
    """Write the envelope arrays and their metadata side by side.

    The contract fields go to JSON via the domain object's own ``to_dict`` and the numeric
    arrays go to NPZ, because they are per-row data whose size depends on the panel. The JSON
    names the file and the array keys, so the two halves stay connected.
    """
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        npz_path,
        upward_kw=np.asarray(block.upward_kw),
        downward_kw=np.asarray(block.downward_kw),
        expected_kw=np.column_stack([published[f"h{h}"]["expected_kw"] for h in horizons]),
        observed_kw=np.column_stack([published[f"h{h}"]["observed_kw"] for h in horizons]),
        lower_kw=np.column_stack([published[f"h{h}"]["lower_kw"] for h in horizons]),
        upper_kw=np.column_stack([published[f"h{h}"]["upper_kw"] for h in horizons]),
        fallback_used=np.column_stack([published[f"h{h}"]["used_fallback"] for h in horizons]),
        horizons=np.asarray(horizons, dtype=np.int64),
        nominal_level=np.asarray(float(block.nominal_level or 0.0)),
    )
    _write_json(
        json_path,
        {
            **block.to_dict(),
            "arrays": {
                "file": npz_path.name,
                "columns": "one per horizon, in the order given by `horizons`",
                "row_order": "ascending sealed-test panel row order",
                "keys": [
                    "upward_kw",
                    "downward_kw",
                    "expected_kw",
                    "observed_kw",
                    "lower_kw",
                    "upper_kw",
                    "fallback_used",
                ],
            },
        },
    )


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default),
        encoding="utf-8",
    )


def _json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (set, frozenset)):
        return sorted(str(item) for item in value)
    raise TypeError(f"{type(value).__name__} is not JSON serialisable")


def render_summary(result: Phase9Result) -> str:
    """The phase's summary document, with the measured tables.

    Args:
        result: The phase's result.

    Returns:
        Markdown.
    """
    horizons = result.split["horizons"]

    def number(value: Any, spec: str = ".4f") -> str:
        return "n/a" if value is None else format(value, spec)

    lines: list[str] = [
        f"# Phase 9 - uncertainty-aware flexibility estimation ({result.experiment_id})",
        "",
        "## Verdict",
        "",
        f"**{result.verdict['answer']}** - {result.verdict['question']}",
        "",
        result.verdict["reading"],
        "",
        "| component | answer | criterion |",
        "|---|---|---|",
    ]
    for name, component in result.verdict["components"].items():
        lines.append(
            f"| `{name}` | **{component['answer']}** | {component['criterion']} |"
        )
    lines.extend(
        [
            "",
            "## What the data physically supports",
            "",
            f"- {result.capability['answer']}",
            f"- {result.capability['physically_supported']} of "
            f"{result.capability['dimensions_audited']} dimensions physically supported; "
            f"{result.capability['unknown']} unknown",
            "",
            "| dimension | basis | gap | blocks Phase 10 |",
            "|---|---|---|---|",
        ]
    )
    for record in result.capability["records"]:
        lines.append(
            f"| `{record['dimension']}` | {record['basis'].upper()} | "
            f"{record['gap_reference']} | {record['blocking_for_phase10']} |"
        )
    lines.extend(
        [
            "",
            "## Configuration selected on the conformity split",
            "",
            f"- criterion: {result.selection['criterion']}",
            f"- chosen: baseline `{result.selection['baseline']}`, granularity "
            f"`{result.selection['granularity']}`, nominal level "
            f"{result.selection['nominal_level']:.0%}",
            "",
            "| configuration | coverage | error | mean width kW | WIS kW | calibrated | rank |",
            "|---|---|---|---|---|---|---|",
        ]
    )
    for row in result.selection["conformity_ranking"]:
        lines.append(
            f"| {row['configuration']} | {row['coverage']:.4f} | "
            f"{row['coverage_error']:+.4f} | {row['mean_width_kw']:.4f} | "
            f"{row['weighted_interval_score_kw']:.4f} | {row['passes']} | {row['rank']} |"
        )
    lines.extend(
        [
            "",
            "## Behavioural envelope on the sealed test split",
            "",
            f"- rows per horizon: {result.split['test_rows']} (read once)",
            f"- fitted on {result.split['calibration_fit_rows']} CAL_FIT rows, selected on "
            f"{result.split['calibration_conformity_rows']} CAL_CONF rows",
            f"- coverage tolerance: {result.verdict['coverage_tolerance']:.4f} absolute",
            "",
            f"- {result.selection['conformal_correction']}",
            "",
            "| horizon | conformal scale | conformity coverage before | after |",
            "|---|---|---|---|",
        ]
        + [
            "| "
            f"{h} | {result.selection['conformal_scales'][str(h)]['scale']:.4f} | "
            f"{result.selection['conformal_scales'][str(h)]['coverage_before']:.4f} | "
            f"{result.selection['conformal_scales'][str(h)]['coverage_after']:.4f} |"
            for h in horizons
        ]
        + [
            "",
            "| horizon | coverage | error | mean width kW | width/MAE | WIS kW | up cov | down cov | calibrated |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
    )
    for horizon in horizons:
        entry = result.selected["by_horizon"][str(horizon)]
        lines.append(
            f"| {horizon} | {entry['coverage']:.4f} | {entry['coverage_error']:+.4f} | "
            f"{entry['mean_width_kw']:.4f} | "
            f"{number(entry['width_over_mean_absolute_deviation'], '.3f')} | "
            f"{entry['weighted_interval_score_kw']:.4f} | "
            f"{number(entry['upward_conditional_coverage'])} | "
            f"{number(entry['downward_conditional_coverage'])} | {entry['passes']} |"
        )
    lines.extend(
        [
            "",
            "Every figure above is `basis=STATISTICAL_PROXY` with "
            "`authority=NOT_CONTROLLABLE`. It describes how far demand has historically moved "
            "from its own expected profile; it does not describe what an operator can command.",
            "",
            "## Uncertainty coupling",
            "",
            f"- formula: `{result.reliance['formula']}`",
            f"- applied: **{result.reliance['applied']}**",
            f"- reason: {result.reliance['reason']}",
            "",
            "| horizon | rho(width, abs deviation) | supported | n | width cost if withheld % |",
            "|---|---|---|---|---|",
        ]
    )
    for horizon, judgement in result.reliance["justification"].items():
        lines.append(
            f"| {horizon} | "
            f"{judgement['spearman_width_vs_absolute_deviation']:+.4f} | "
            f"{judgement['coupling_supported']} | {judgement.get('n_rows')} | "
            f"{result.reliance['withheld_width_cost_pct'][horizon]:+.2f} |"
        )
    lines.extend(
        [
            "",
            "## Aggregation",
            "",
            result.aggregation["answer"],
            "",
            "| horizon | aggregate up kW | naive sum kW | diversification | mean pairwise corr | coverage |",
            "|---|---|---|---|---|---|",
        ]
    )
    for horizon in horizons:
        entry = result.aggregation["by_horizon"][str(horizon)]
        lines.append(
            f"| {horizon} | {entry['upward_kw']:.2f} | "
            f"{entry['sum_individual_upward_kw']:.2f} | "
            f"{number(entry['diversification_upward'])} | "
            f"{number(entry['correlation']['mean'], '+.4f')} | "
            f"{number(entry['coverage_measured'])} |"
        )
    lines.extend(
        [
            "",
            "## Regime breakdown",
            "",
            result.regime["note"],
            "",
            "| horizon | axis | regimes | width ratio worst:easiest | all calibrated |",
            "|---|---|---|---|---|",
        ]
    )
    for horizon in horizons:
        for axis, data in result.regime["by_horizon"][str(horizon)].items():
            lines.append(
                f"| {horizon} | {axis} | {len(data['regimes'])} | "
                f"{number(data['width_ratio_worst_to_easiest'], '.2f')}x | "
                f"{data['all_regimes_calibrated']} |"
            )
    lines.extend(
        [
            "",
            "## What is not claimed",
            "",
            "- No physical flexibility is claimed for any asset; the capability audit "
            "establishes that none is supported.",
            "- No dispatchable resource is claimed. Historical variation is not the ability "
            "to be commanded.",
            "- No guaranteed demand response is claimed, and no demand-response programme "
            "exists in this dataset.",
            "- Aggregation does not create capability; it is measured historical correlation.",
            "- The uncertainty coupling is not applied, so no flexibility figure here is "
            "discounted by predictive uncertainty.",
            "",
            "## Compute",
            "",
            f"- total seconds: {result.seconds:.1f}",
        ]
    )
    return "\n".join(lines) + "\n"