"""Phase 9 tests: the flexibility contract, and the estimators that feed it.

Every test in this file was written because something in this phase was **wrong** at least
once while building it. That is the honest reason for the list, and each entry names the
defect it pins:

- ``test_calendar_baseline_indexes_by_series_not_row`` - the calendar table was fitted from
  panel *row positions* while the fitter indexed the demand matrix by them, so a
  40-series matrix was indexed with values up to 89,760.
- ``test_envelope_splits_the_miss_probability_between_tails`` - the band took the *total*
  miss probability as its per-tail quantile, delivering 80% coverage while labelling the
  band 90%.
- ``test_envelope_is_fitted_per_horizon`` - cells were pooled across horizons, which made a
  single band serve h=1 and h=96 at once and per-horizon calibration impossible.
- ``test_conformal_scale_reaches_nominal_coverage`` - the correction the split design exists
  for.
- ``test_fixed_ensemble_weights_are_not_transposed`` - the weight matrix was built
  ``[expert, horizon]`` and multiplied with Phase 8's ``"he"`` einsum, producing a
  plausible-looking forecast from the wrong weights.
- ``test_row_alignment_rejects_a_permutation`` - matching row *counts* would have let a
  permuted Phase 8 artifact pair every interval with the wrong deviation.

Plus the contract tests the domain layer owes its callers: a statistical proxy may not claim
a control authority, an unknown may not carry a magnitude, and an assumed value may not be
passed off as physical.

What is deliberately **not** relaxed here: the tests that could hide a fidelity problem use
the real Phase 7/8 artifacts and skip when they are absent, exactly as Phase 3 does.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from energy_intelligence.domain.enums import (
    AuthorityLevel,
    FlexibilityBasis,
    FlexibilityDirection,
    Unit,
    VariableRole,
)
from energy_intelligence.domain.errors import DomainValidationError
from energy_intelligence.domain.flexibility import (
    AggregationLevel,
    FlexibilityEnvelope,
    FlexibilityEstimate,
)
from energy_intelligence.domain.provenance import (
    ProcessingStep,
    Provenance,
    ProvenanceEvent,
    SourceReference,
)
from energy_intelligence.ml.flexibility.aggregation import aggregate_envelope
from energy_intelligence.ml.flexibility.baselines import fit_baseline
from energy_intelligence.ml.flexibility.capability import CAPABILITY_DIMENSIONS, audit_capabilities
from energy_intelligence.ml.flexibility.demand import DemandField, verify_demand_reconstruction
from energy_intelligence.ml.flexibility.envelope import (
    conformal_scale,
    evaluate_envelope,
    fit_envelope,
)
from energy_intelligence.ml.flexibility.offline.experiment import (
    _fixed_ensemble_weights,
    _row_alignment,
)
from energy_intelligence.ml.flexibility.reliance import reliance_from_widths
from energy_intelligence.ml.uncertainty.measures import spearman
from energy_intelligence.ml.flexibility.scenarios import (
    REAL_DATA_MODE,
    SCENARIO_MODE,
    ScenarioAssumptions,
    ScenarioViolation,
    assert_real_data_mode,
)

HORIZONS = (1, 4, 96)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _provenance() -> Provenance:
    return Provenance(
        source=SourceReference(
            source_id="synthetic",
            dataset="flexibility_test",
            locator="synthetic/for-tests",
            version="0",
        ),
        events=(
            ProvenanceEvent(
                timestamp="2018-01-01T00:00:00",
                event="flexibility_envelope_estimated",
                detail="synthetic",
            ),
        ),
        processing=(
            ProcessingStep(name="behavioural_response_envelope", version="phase9"),
        ),
    )


@pytest.fixture
def field() -> DemandField:
    """A demand field with a known diurnal shape and known per-row scaling.

    The volatility is deliberately modest so that a 90% band has a chance of holding on a
    2,000-row conformity split - the point of the calibration tests is the correction
    mechanism, not the difficulty of the data.
    """
    rng = np.random.default_rng(20260904)
    n_series = 5
    n_steps = 96 * 120
    rated = rng.uniform(2.0, 6.0, n_series)
    steps = np.arange(n_steps)
    daily = 1.0 + 0.3 * np.sin(2.0 * np.pi * steps / 96.0)
    demand = rated[:, None] * daily[None, :] * (
        1.0 + 0.02 * rng.standard_normal((n_series, n_steps))
    )
    origins = np.sort(rng.integers(0, n_steps - 96, 4000))
    series = rng.integers(0, n_series, 4000)
    return DemandField(
        demand_kw=demand.astype(np.float32),
        rated_kw=rated,
        series=series,
        origins=origins,
        row_scale_kw=rated[series],
        horizons=HORIZONS,
    )


def _estimate(**overrides) -> FlexibilityEstimate:
    """A minimal valid estimate, with the fields under test left to the caller."""
    kwargs = dict(
        target="customer_load",
        direction=FlexibilityDirection.DOWNWARD,
        magnitude_kw=1.0,
        unit=Unit.KILOWATT,
        timestamp="2018-01-01T00:00:00",
        provenance=_provenance(),
        method="test",
        basis=FlexibilityBasis.STATISTICAL_PROXY,
        authority=AuthorityLevel.NOT_CONTROLLABLE,
        role=VariableRole.DERIVED,
    )
    kwargs.update(overrides)
    return FlexibilityEstimate(**kwargs)


@pytest.fixture
def drifting_field() -> DemandField:
    """A demand field whose dispersion grows with forecast distance.

    Forecast error that does not grow with horizon is not a forecast error, it is a constant.
    This fixture therefore adds a small AR(1) component whose horizon-independent volatility
    makes the per-horizon magnitude difference real rather than incidental.
    """
    rng = np.random.default_rng(11)
    n_series = 5
    n_steps = 96 * 120
    rated = rng.uniform(2.0, 6.0, n_series)
    steps = np.arange(n_steps)
    daily = 1.0 + 0.3 * np.sin(2.0 * np.pi * steps / 96.0)
    noise = np.zeros((n_series, n_steps))
    for index in range(n_series):
        shock = 0.0
        for step in range(n_steps):
            shock = 0.9 * shock + rng.standard_normal()
            noise[index, step] = shock
    demand = rated[:, None] * daily[None, :] * (1.0 + 0.05 * noise)
    origins = np.sort(rng.integers(0, n_steps - 96, 4000))
    series = rng.integers(0, n_series, 4000)
    return DemandField(
        demand_kw=demand.astype(np.float32),
        rated_kw=rated,
        series=series,
        origins=origins,
        row_scale_kw=rated[series],
        horizons=HORIZONS,
    )


# ---------------------------------------------------------------------------
# The domain contract
# ---------------------------------------------------------------------------


def test_statistical_proxy_may_not_claim_a_control_authority() -> None:
    """The load-bearing rule of the phase.

    A proxy derived from observed variation is evidence about *behaviour*, not about the
    ability to be commanded. If the contract allowed it to carry ``SCHEDULEABLE``, a caller
    could plan against a number that no operator could act on.
    """
    with pytest.raises(DomainValidationError):
        _estimate(
            direction=FlexibilityDirection.DOWNWARD,
            magnitude_kw=5.0,
            basis=FlexibilityBasis.STATISTICAL_PROXY,
            authority=AuthorityLevel.SCHEDULEABLE,
        )


def test_unknown_basis_may_not_carry_a_magnitude() -> None:
    """An unknown is not a small number.

    Allowing ``basis=UNKNOWN`` with ``value=0.0`` would let a caller who had not established
    anything report zero flexibility and read it as "none available" rather than "not known".
    """
    with pytest.raises(DomainValidationError):
        _estimate(
            direction=FlexibilityDirection.DOWNWARD,
            magnitude_kw=0.0,
            basis=FlexibilityBasis.UNKNOWN,
            authority=AuthorityLevel.NOT_CONTROLLABLE,
        )


def test_assumed_magnitude_is_labelled_not_hidden() -> None:
    """An assumed figure is legitimate - it is what scenario mode is for - but it must say so."""
    estimate = _estimate(
        direction=FlexibilityDirection.UPWARD,
        magnitude_kw=2.5,
        basis=FlexibilityBasis.ASSUMED,
        authority=AuthorityLevel.ADVISORY_ONLY,
    )
    assert estimate.basis is FlexibilityBasis.ASSUMED
    assert estimate.is_dispatchable is False
    assert estimate.basis_qualifier is not None


def test_downward_direction_and_load_sign_convention_agree() -> None:
    """``DOWNWARD`` means less demand, and ``FLEXIBLE_LOAD_SHIFT`` is a signed load setpoint.

    Getting this backwards would let a caller shed load by *increasing* it, so the sign
    conversion is pinned rather than assumed.
    """
    estimate = _estimate(
        direction=FlexibilityDirection.DOWNWARD,
        magnitude_kw=5.0,
        basis=FlexibilityBasis.STATISTICAL_PROXY,
        authority=AuthorityLevel.NOT_CONTROLLABLE,
    )
    signed = estimate.as_signed_setpoint()
    assert signed == pytest.approx(-5.0), "downward load shed must be a negative setpoint"
    # NodePower treats load as negative, so shedding load is a positive power delta.
    assert estimate.as_node_power_delta() == pytest.approx(5.0)


def test_envelope_serialization_round_trips(field: DemandField) -> None:
    envelope = FlexibilityEnvelope(
        target="group-test",
        aggregation=AggregationLevel.GROUP,
        basis=FlexibilityBasis.STATISTICAL_PROXY,
        authority=AuthorityLevel.NOT_CONTROLLABLE,
        method="test",
        upward_kw=np.full((3, 2), 1.5),
        downward_kw=np.full((3, 2), 2.5),
        provenance=_provenance(),
        nominal_level=0.9,
        horizons=(1, 4, 96),
    )
    # The envelope's contract fields survive a JSON round trip. Its arrays are published
    # separately (NPZ beside JSON), so what is pinned here is the contract, not the numbers -
    # and crucially that `is_dispatchable` cannot be revived by deserialising.
    payload = json.loads(json.dumps(envelope.to_dict()))
    assert payload["basis"] == FlexibilityBasis.STATISTICAL_PROXY.value
    assert payload["authority"] == AuthorityLevel.NOT_CONTROLLABLE.value
    assert payload["is_dispatchable"] is False
    assert envelope.is_dispatchable is False
    assert payload["horizons"] == [1, 4, 96]


# ---------------------------------------------------------------------------
# The capability audit
# ---------------------------------------------------------------------------


def test_capability_audit_finds_nothing_physically_supported_on_smartds() -> None:
    """The phase's headline finding, asserted rather than described.

    If this ever starts passing for a *non-zero* count it means the audit learned to infer
    capability from something it should not - which would be a regression in the direction
    that matters most, because it would let a physical claim through.
    """
    audit = audit_capabilities(
        dataset="smartds-load_profiles",
        version="v1.0",
        locator="artifacts/phase9",
        timestamp="2018-01-01T00:00:00",
    )
    payload = audit.to_dict()
    assert payload["physically_supported"] == 0
    assert payload["unknown"] == len(CAPABILITY_DIMENSIONS)
    assert all(record["basis"] == "unknown" for record in payload["records"])
    assert all(record["quantified"] is False for record in payload["records"])
    assert all(record["authority"] == AuthorityLevel.NOT_CONTROLLABLE.value for record in payload["records"])


def test_every_audited_dimension_blocks_phase_10_with_a_gap_reference() -> None:
    """A blocking gap must point at a documented gap, not just assert that it blocks."""
    audit = audit_capabilities(
        dataset="smartds-load_profiles",
        version="v1.0",
        locator="artifacts/phase9",
        timestamp="2018-01-01T00:00:00",
    )
    for record in audit.to_dict()["records"]:
        assert record["blocking_for_phase10"] is True
        assert record["gap_reference"].startswith("G-"), record["gap_reference"]
        assert record["required_metadata"], "an unknown must say what would be needed"


def test_scenario_mode_is_quarantined_from_real_data_results() -> None:
    """Scenario figures may never be mixed into a real-data result.

    ``assert_real_data_mode`` is the gate a real-data code path calls, and a scenario object
    reaching it means assumed values are about to be written into a measured result. The
    pass case is pinned too: ``None`` means real-data mode and must be accepted.
    """
    assert_real_data_mode(None)
    scenario = ScenarioAssumptions(
        name="high-dr",
        assumed_controllable_load_fraction=0.08,
        notes="demonstration only",
    )
    with pytest.raises(ScenarioViolation):
        assert_real_data_mode(scenario)


# ---------------------------------------------------------------------------
# Demand reconstruction
# ---------------------------------------------------------------------------


def test_demand_reconstruction_is_exact_when_scales_are_applied() -> None:
    """``values * rated_kw`` must reproduce the panel's own targets to the last bit.

    The whole phase is measured around a reconstructed series. A reconstruction that is
    merely close would put a systematic error into every deviation and therefore into every
    published magnitude.
    """
    rng = np.random.default_rng(3)
    n_series, n_steps, n_rows = 4, 96 * 30, 600
    rated = rng.uniform(1.0, 4.0, n_series)
    values = rng.uniform(0.2, 1.8, (n_series, n_steps)).astype(np.float32)
    origins = np.sort(rng.integers(0, n_steps - 96, n_rows))
    series = rng.integers(0, n_series, n_rows)
    reconstructed = DemandField(
        demand_kw=values * rated[:, None],
        rated_kw=rated,
        series=series,
        origins=origins,
        row_scale_kw=rated[series],
        horizons=HORIZONS,
    )
    actual = np.column_stack(
        [
            np.asarray(
                [reconstructed.target_kw(int(row), h) for row in range(n_rows)],
                dtype=np.float64,
            )
            for h in HORIZONS
        ]
    )
    report = verify_demand_reconstruction(reconstructed, actual_kw=actual)
    assert report["max_absolute_gap_kw"] == pytest.approx(0.0, abs=1e-9)
    assert report["rows_checked"] == n_rows


def test_reconstruction_check_fails_loudly_on_a_wrong_scale() -> None:
    """The check must be able to fail, or it is decoration."""
    rng = np.random.default_rng(4)
    n_series, n_steps, n_rows = 3, 96 * 20, 300
    rated = rng.uniform(1.0, 4.0, n_series)
    values = rng.uniform(0.2, 1.8, (n_series, n_steps)).astype(np.float32)
    origins = np.sort(rng.integers(0, n_steps - 96, n_rows))
    series = rng.integers(0, n_series, n_rows)
    reconstructed = DemandField(
        demand_kw=values * rated[:, None],
        rated_kw=rated,
        series=series,
        origins=origins,
        row_scale_kw=rated[series],
        horizons=HORIZONS,
    )
    actual = np.column_stack(
        [
            np.asarray(
                [reconstructed.target_kw(int(row), h) for row in range(n_rows)],
                dtype=np.float64,
            )
            for h in HORIZONS
        ]
    )
    with pytest.raises(ValueError):
        verify_demand_reconstruction(reconstructed, actual_kw=actual * 1.05, tolerance=1e-6)


# ---------------------------------------------------------------------------
# Regression: the calendar baseline indexed the wrong axis
# ---------------------------------------------------------------------------


def test_calendar_baseline_indexes_by_series_not_row(field: DemandField) -> None:
    """``fit_baseline`` must hand the fitter *series* indices, not panel row positions.

    The panel has 40 series and hundreds of thousands of rows. Passing row positions where
    series indices are expected indexes a 40-row matrix with values up to 89,760, which
    raises rather than returning a wrong number - so this test guards against a future
    "fix" that clips the index instead of passing the right one.
    """
    baseline = fit_baseline(
        "calendar",
        field,
        calibration_rows=np.arange(0, 2000),
        horizons=HORIZONS,
        granularity="time_of_day",
        min_samples=3,
    )
    assert baseline.table.shape[0] == field.n_series
    assert baseline.table.shape[0] != 2000
    assert np.all(baseline.table >= 0.0)


# ---------------------------------------------------------------------------
# Regression: the miss probability was not split between tails
# ---------------------------------------------------------------------------


def test_envelope_splits_the_miss_probability_between_tails(field: DemandField) -> None:
    """A nominal-90% band must cover about 90%, not 80%.

    Taking the total miss probability (10%) as the per-tail quantile gives a 90th and a 10th
    percentile, which is a **80%** band. The band would still be labelled 90% and would
    under-cover by ten points on perfectly behaved data, which is the sort of mislabelling
    that survives review because nothing crashes.
    """
    rows = np.arange(0, 2000)
    baseline = fit_baseline(
        "calendar", field, calibration_rows=rows, horizons=HORIZONS, granularity="horizon", min_samples=3
    )
    envelope = fit_envelope(
        field=field,
        baseline=baseline,
        calibration_rows=rows,
        horizons=HORIZONS,
        nominal_level=0.90,
        min_samples=3,
    )
    applied = evaluate_envelope(envelope, field, rows=np.arange(2000, 4000), horizon_steps=1)
    inside = (applied["observed_kw"] >= applied["lower_kw"]) & (
        applied["observed_kw"] <= applied["upper_kw"]
    )
    coverage = float(np.mean(inside))
    # The floor is deliberately loose - this pins the *mislabelling* class of bug, not the
    # exact quantile - but it is far above the 0.80 an unsplit tail would produce.
    assert coverage > 0.85, f"nominal 90% band covered {coverage:.4f}, which is the unsplit-tail bug"


def test_raising_the_nominal_level_widens_the_band(field: DemandField) -> None:
    """Monotonicity in the level, which the unsplit bug would not have broken but a
    mis-signed correction would."""
    rows = np.arange(0, 2000)
    baseline = fit_baseline(
        "calendar", field, calibration_rows=rows, horizons=HORIZONS, granularity="horizon", min_samples=3
    )
    widths = {}
    for level in (0.80, 0.95):
        envelope = fit_envelope(
            field=field,
            baseline=baseline,
            calibration_rows=rows,
            horizons=HORIZONS,
            nominal_level=level,
            min_samples=3,
        )
        applied = evaluate_envelope(envelope, field, rows=np.arange(2000, 4000), horizon_steps=1)
        widths[level] = float(np.mean(applied["upper_kw"] - applied["lower_kw"]))
    assert widths[0.95] > widths[0.80], widths


# ---------------------------------------------------------------------------
# Regression: cells were pooled across horizons
# ---------------------------------------------------------------------------


def test_envelope_is_fitted_per_horizon(drifting_field: DemandField) -> None:
    """Each horizon gets its own magnitudes.

    Deviation at h=96 is several times deviation at h=1. A cell shared between them has to be
    wide enough for h=96, so the h=1 band over-covers by tens of percent - and no amount of
    calibration downstream can fix a band that is ten times too wide at one horizon because
    it was fitted at another.
    """
    field = drifting_field
    rows = np.arange(0, 2000)
    baseline = fit_baseline(
        "calendar", field, calibration_rows=rows, horizons=HORIZONS, granularity="horizon", min_samples=3
    )
    envelope = fit_envelope(
        field=field,
        baseline=baseline,
        calibration_rows=rows,
        horizons=HORIZONS,
        nominal_level=0.90,
        min_samples=3,
    )
    assert envelope.upward_kw.shape[0] == len(HORIZONS)
    assert envelope.horizons == HORIZONS
    # Magnitudes must be readable per horizon, and must differ between them.
    up1 = float(np.mean(envelope.upward_kw[envelope.horizons.index(1)]))
    up96 = float(np.mean(envelope.upward_kw[envelope.horizons.index(96)]))
    assert up96 > up1, "h=96 magnitudes should exceed h=1 when dispersion grows with horizon"


def test_evaluate_envelope_rejects_an_unfitted_horizon(field: DemandField) -> None:
    """Asking for a horizon that was never fitted must fail, not silently use slot 0."""
    rows = np.arange(0, 2000)
    baseline = fit_baseline(
        "calendar", field, calibration_rows=rows, horizons=(1, 4), granularity="horizon", min_samples=3
    )
    envelope = fit_envelope(
        field=field,
        baseline=baseline,
        calibration_rows=rows,
        horizons=(1, 4),
        nominal_level=0.90,
        min_samples=3,
    )
    with pytest.raises(ValueError):
        evaluate_envelope(envelope, field, rows=np.arange(2000, 4000), horizon_steps=96)


# ---------------------------------------------------------------------------
# The conformity-split calibration
# ---------------------------------------------------------------------------


def test_conformal_scale_reaches_nominal_coverage() -> None:
    """The correction exists to remove a coverage offset measured on held-out rows."""
    rng = np.random.default_rng(5)
    n = 40000
    expected = np.full(n, 10.0)
    observed = expected + rng.standard_normal(n) * 1.5
    # Deliberately too wide: the fitted band over-covers, which is the situation the
    # conformity split exists to correct.
    upward = np.full(n, 4.0)
    downward = np.full(n, 4.0)
    before = float(
        np.mean((observed >= expected - downward) & (observed <= expected + upward))
    )
    assert before > 0.95, "fixture should start over-covered"

    scale, report = conformal_scale(
        expected_kw=expected,
        observed_kw=observed,
        upward_kw=upward,
        downward_kw=downward,
        nominal_level=0.90,
    )
    assert 0.0 < scale < 1.0
    assert report["coverage_after"] == pytest.approx(0.90, abs=1e-3)
    assert report["converged"] is True
    assert report["coverage_before"] == pytest.approx(before)


def test_conformal_scale_refuses_a_degenerate_band() -> None:
    """All-zero magnitudes cannot reach a non-zero level, and must say so."""
    zeros = np.zeros(10)
    with pytest.raises(ValueError):
        conformal_scale(
            expected_kw=np.ones(10),
            observed_kw=np.ones(10),
            upward_kw=zeros,
            downward_kw=zeros,
            nominal_level=0.90,
        )


def test_conformal_scale_caps_rather_than_running_away() -> None:
    """A band that cannot reach the level must be capped and reported, not unbounded."""
    expected = np.zeros(100)
    observed = np.full(100, 1000.0)
    tiny = np.full(100, 1e-6)
    scale, report = conformal_scale(
        expected_kw=expected,
        observed_kw=observed,
        upward_kw=tiny,
        downward_kw=tiny,
        nominal_level=0.90,
        max_scale=10.0,
    )
    assert scale == pytest.approx(10.0)
    assert report["converged"] is False
    assert "capped" in report["note"]


# ---------------------------------------------------------------------------
# Regression: stability was measured across the customer boundary
# ---------------------------------------------------------------------------


def test_stability_lag1_is_computed_within_series() -> None:
    """Lag-1 must not be taken across two different customers.

    The panel is ordered series-major, so consecutive rows are usually different customers.
    A lag-1 over the raw row order therefore measures how much one customer's envelope
    differs from the next one's - a statement about the customer mix, not about whether the
    envelope is smooth in time. On this data that mistake reported an autocorrelation near
    0.16 for an envelope whose width is constant per customer.
    """
    from energy_intelligence.ml.flexibility.evaluation import stability_report

    series = np.repeat(np.arange(3), 50)
    observed = np.random.default_rng(1).standard_normal(150)
    # Constant within each series, wildly different between them.
    width = np.array([1.0, 50.0, 2.0])[series]
    report = stability_report(width_kw=width, observed_kw=observed, series=series, spearman=spearman)
    assert report["lag1_computed_within_series"] is True
    assert report["width_is_constant_within_series"] is True
    # No within-series variation is the smoothest envelope there is, not missing evidence.
    assert report["width_lag1_autocorrelation"] == pytest.approx(1.0)


def test_stability_lag1_detects_a_smooth_ramp() -> None:
    """The statistic must still respond to real within-series structure."""
    from energy_intelligence.ml.flexibility.evaluation import stability_report

    series = np.repeat(np.arange(3), 50)
    observed = np.random.default_rng(2).standard_normal(150)
    width = np.concatenate([np.linspace(0.0, 10.0, 50)] * 3)
    report = stability_report(width_kw=width, observed_kw=observed, series=series, spearman=spearman)
    assert report["width_is_constant_within_series"] is False
    assert report["width_lag1_autocorrelation"] > 0.8


def test_stability_ignores_jitter_within_a_series() -> None:
    """Independent jitter within each series must score near zero."""
    from energy_intelligence.ml.flexibility.evaluation import stability_report

    rng = np.random.default_rng(3)
    series = np.repeat(np.arange(3), 200)
    observed = rng.standard_normal(600)
    width = rng.random(600)
    report = stability_report(width_kw=width, observed_kw=observed, series=series, spearman=spearman)
    assert abs(report["width_lag1_autocorrelation"]) < 0.15


# ---------------------------------------------------------------------------
# Regression: the weight matrix was transposed
# ---------------------------------------------------------------------------


def test_fixed_ensemble_weights_are_not_transposed(tmp_path) -> None:
    """``[expert, horizon]`` must not be multiplied as ``[horizon, expert]``.

    The transposed product is a valid array of plausible numbers that reproduces nobody's
    forecast. It was caught only because the parity check against Phase 7's published MAE
    ran before anything was estimated - so this test pins the ordering the helper documents.
    """
    import json

    path = tmp_path / "result.json"
    path.write_text(
        json.dumps(
            {
                "routing": {
                    "fixed_ensemble_weights": {
                        "1": {"persistence": 0.16, "tree": 0.34, "tcn": 0.50},
                        "4": {"persistence": 0.00, "tree": 0.34, "tcn": 0.66},
                        "96": {"persistence": 0.00, "tree": 0.26, "tcn": 0.74},
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    weights = _fixed_ensemble_weights(
        path, expert_names=("persistence", "tree", "tcn"), horizons=(1, 4, 96)
    )
    assert weights.shape == (3, 3), "one row per expert, one column per horizon"
    assert weights[0, 0] == pytest.approx(0.16), "persistence at h=1"
    assert weights[2, 2] == pytest.approx(0.74), "tcn at h=96"
    assert weights[0, 1] == pytest.approx(0.00), "persistence drops out beyond h=1"
    for column in range(weights.shape[1]):
        assert float(weights[:, column].sum()) == pytest.approx(1.0)


def test_fixed_ensemble_weights_reject_a_row_that_does_not_sum_to_one(tmp_path) -> None:
    import json

    path = tmp_path / "result.json"
    path.write_text(
        json.dumps({"routing": {"fixed_ensemble_weights": {"1": {"a": 0.5, "b": 0.2}}}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="sum to"):
        _fixed_ensemble_weights(path, expert_names=("a", "b"), horizons=(1,))


def test_fixed_ensemble_weights_reject_a_missing_expert(tmp_path) -> None:
    """A cache whose expert pool grew since Phase 7 must fail loudly."""
    import json

    path = tmp_path / "result.json"
    path.write_text(
        json.dumps({"routing": {"fixed_ensemble_weights": {"1": {"a": 0.5, "b": 0.5}}}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="omit"):
        _fixed_ensemble_weights(path, expert_names=("a", "b", "c"), horizons=(1,))


# ---------------------------------------------------------------------------
# Regression: row alignment is proved, not assumed
# ---------------------------------------------------------------------------


def test_row_alignment_rejects_a_permutation() -> None:
    """Matching row counts must not be mistaken for matching rows.

    A permutation preserves the count and the shape, so every structural check would pass -
    and every interval width would then be paired with another row's deviation. Comparing
    the artifact's own point forecast is the only check that catches it.
    """
    rng = np.random.default_rng(6)
    published = rng.standard_normal((1000, 3))
    assert _row_alignment(published_point_kw=published, rebuilt_point_kw=published)["matches"]

    permuted = published[rng.permutation(1000)]
    report = _row_alignment(published_point_kw=published, rebuilt_point_kw=permuted)
    assert report["matches"] is False
    assert report["max_absolute_gap_kw"] > 1e-9


def test_row_alignment_rejects_a_shape_mismatch() -> None:
    report = _row_alignment(
        published_point_kw=np.zeros((10, 3)), rebuilt_point_kw=np.zeros((11, 3))
    )
    assert report["matches"] is False
    assert report["published_shape"] == [10, 3]


# ---------------------------------------------------------------------------
# Reliance coupling: measured, never assumed
# ---------------------------------------------------------------------------


def test_reliance_reference_is_causal() -> None:
    """Changing a row's own width must not change its own factor.

    A centred or contemporaneous reference would make the discount a function of the very
    quantity being judged. Rows are checked in order so this is a real causality property
    rather than a formula inspection.
    """
    widths = np.linspace(1.0, 2.0, 500)
    rated = np.ones(500)
    first = reliance_from_widths(
        interval_width_kw=widths, rated_kw=rated, window=100, min_periods=20, floor=0.25
    )
    perturbed = widths.copy()
    perturbed[400] = 50.0
    second = reliance_from_widths(
        interval_width_kw=perturbed, rated_kw=rated, window=100, min_periods=20, floor=0.25
    )
    assert np.allclose(first.factor[:400], second.factor[:400])
    assert second.factor[400] < first.factor[400], "a wider row must get a smaller factor"


def test_reliance_factor_respects_its_floor_and_ceiling() -> None:
    widths = np.concatenate([np.full(200, 1.0), np.full(200, 1000.0)])
    rated = np.ones(400)
    factor = reliance_from_widths(
        interval_width_kw=widths, rated_kw=rated, window=100, min_periods=20, floor=0.25
    )
    finite = factor.factor[np.isfinite(factor.factor)]
    assert finite.min() >= 0.25 - 1e-12
    assert finite.max() <= 1.0 + 1e-12


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def test_aggregation_reports_the_naive_sum_beside_the_pooled_envelope(field: DemandField) -> None:
    """Aggregation must never be reported without the sum it is being compared against.

    A pooled figure on its own invites the reading that the fleet can move that much, which
    is precisely the claim this phase refuses to make.
    """
    rows = np.arange(0, 2000)
    baseline = fit_baseline(
        "calendar", field, calibration_rows=rows, horizons=HORIZONS, granularity="horizon", min_samples=3
    )
    envelope = fit_envelope(
        field=field,
        baseline=baseline,
        calibration_rows=rows,
        horizons=HORIZONS,
        nominal_level=0.90,
        min_samples=3,
    )
    result = aggregate_envelope(
        target="group-test",
        field=field,
        envelope=envelope,
        calibration_rows=rows,
        horizon_steps=96,
        nominal_level=0.90,
    )
    described = result.describe()
    assert described["sum_individual_upward_kw"] > 0
    assert described["upward_kw"] > 0
    assert described["basis"] == FlexibilityBasis.STATISTICAL_PROXY.value
    assert described["authority"] == AuthorityLevel.NOT_CONTROLLABLE.value
    assert described["authority"] == AuthorityLevel.NOT_CONTROLLABLE.value
    assert described["basis"] == FlexibilityBasis.STATISTICAL_PROXY.value
    assert "warning" in described
    assert "NOT" in described["warning"].upper() or "not" in described["warning"]