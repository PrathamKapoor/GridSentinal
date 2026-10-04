"""Tests for identifiers, quantities, provenance, quality, timebase, uncertainty.

These are the value types everything else is built on, so their invariants are
tested directly and exhaustively rather than only through the composites.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta, timezone

import pytest

from energy_intelligence.domain import (
    ActionId,
    AssetId,
    CLEAN_QUALITY,
    ConstraintId,
    DataQuality,
    DomainValidationError,
    ForecastId,
    Identifier,
    NetworkElementId,
    NodeId,
    ObjectiveId,
    ProcessingStep,
    Provenance,
    ProvenanceEvent,
    Quantity,
    QualityFlag,
    SourceReference,
    SystemId,
    TimeBase,
    TBD_METHOD,
    UncertaintyEstimate,
    UncertaintyKind,
    Unit,
)


# ----------------------------------------------------------------------
# Identifiers
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("cls", "prefix"),
    [
        (AssetId, "asset"),
        (NodeId, "node"),
        (NetworkElementId, "net"),
        (ConstraintId, "con"),
        (ObjectiveId, "obj"),
        (ActionId, "act"),
        (ForecastId, "fc"),
        (SystemId, "sys"),
    ],
)
def test_valid_identifier_constructs_and_is_hashable(cls, prefix: str) -> None:
    identifier = cls(f"{prefix}-example-1")
    assert isinstance(identifier, cls)
    assert hash(identifier) is not None
    assert identifier == cls(identifier.value)
    assert str(identifier) == f"{prefix}-example-1"


def test_identifier_enforces_its_prefix() -> None:
    with pytest.raises(DomainValidationError, match="does not match"):
        AssetId("node-feeder-a")


@pytest.mark.parametrize(
    "bad",
    ["", "asset-", "asset-UPPER", "asset-has space", "asset-bad!", "a" * 100, "asset-a/b"],
)
def test_invalid_asset_identifier_is_rejected(bad: str) -> None:
    with pytest.raises(DomainValidationError):
        AssetId(bad)


def test_identifier_rejects_non_string() -> None:
    with pytest.raises(DomainValidationError, match="must be a string"):
        NodeId(7)  # type: ignore[arg-type]


def test_identifier_is_immutable() -> None:
    identifier = NodeId("node-pcc")
    with pytest.raises(FrozenInstanceError):
        identifier.value = "node-other"  # type: ignore[misc]


def test_identifier_str_returns_value() -> None:
    assert str(NodeId("node-pcc")) == "node-pcc"


def test_identifier_base_class_has_a_working_pattern() -> None:
    assert Identifier("id-generic").value == "id-generic"


def test_identifiers_of_different_kinds_cannot_be_interchanged() -> None:
    """The prefix scheme makes cross-kind references impossible to write."""
    with pytest.raises(DomainValidationError):
        AssetId("node-feeder-a")
    with pytest.raises(DomainValidationError):
        NodeId("asset-battery-1")

    # A well-formed constraint id is not a well-formed objective id.
    assert ConstraintId("con-window").value == "con-window"
    with pytest.raises(DomainValidationError):
        ObjectiveId("con-window")


# ----------------------------------------------------------------------
# Quantity
# ----------------------------------------------------------------------


def test_quantity_holds_value_and_unit() -> None:
    quantity = Quantity(12.5, Unit.KILOWATT)
    assert quantity.value == 12.5
    assert quantity.unit is Unit.KILOWATT
    assert str(quantity) == "12.5 kW"


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_quantity_rejects_non_finite(bad: float) -> None:
    with pytest.raises(DomainValidationError):
        Quantity(bad, Unit.KILOWATT)


@pytest.mark.parametrize("bad", [True, "5", None, [1]])
def test_quantity_rejects_non_numeric(bad: object) -> None:
    with pytest.raises(DomainValidationError, match="real number"):
        Quantity(bad, Unit.KILOWATT)  # type: ignore[arg-type]


def test_quantity_rejects_unknown_unit() -> None:
    with pytest.raises(DomainValidationError, match="must be a Unit member"):
        Quantity(1.0, "kW")  # type: ignore[arg-type]


def test_quantity_allows_negative_for_signed_quantities() -> None:
    assert Quantity(-50.0, Unit.KILOWATT).value == -50.0


def test_quantity_is_zero_and_comparison() -> None:
    assert Quantity(0.0, Unit.KILOWATT).is_zero
    assert not Quantity(0.1, Unit.KILOWATT).is_zero


def test_quantity_unit_comparison_is_explicit() -> None:
    kw = Quantity(1.0, Unit.KILOWATT)
    mw = Quantity(1.0, Unit.MEGAWATT)
    assert not kw.same_unit_as(mw)
    with pytest.raises(DomainValidationError, match="unit mismatch"):
        kw.require_same_unit(mw, "test")


def test_quantity_is_immutable() -> None:
    with pytest.raises(FrozenInstanceError):
        Quantity(1.0, Unit.KILOWATT).value = 2.0  # type: ignore[misc]


# ----------------------------------------------------------------------
# SourceReference / ProvenanceEvent / ProcessingStep
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "field", ["source_id", "dataset", "locator", "version"]
)
def test_source_reference_requires_every_string_field(field: str) -> None:
    kwargs = {
        "source_id": "src",
        "dataset": "load",
        "locator": "path",
        "version": "v1",
    }
    kwargs[field] = ""
    with pytest.raises(DomainValidationError, match=field):
        SourceReference(**kwargs)


def test_source_reference_checksum_is_optional() -> None:
    assert SourceReference("s", "d", "l", "v1").checksum is None


def test_provenance_event_requires_timestamp_and_event() -> None:
    with pytest.raises(DomainValidationError):
        ProvenanceEvent("", "ingested")
    with pytest.raises(DomainValidationError):
        ProvenanceEvent("2026-01-01T00:00:00+00:00", "")


def test_processing_step_requires_name_and_version() -> None:
    with pytest.raises(DomainValidationError):
        ProcessingStep("", "v1")
    with pytest.raises(DomainValidationError):
        ProcessingStep("resample", "")


# ----------------------------------------------------------------------
# Provenance - the "cannot silently disappear" guarantee
# ----------------------------------------------------------------------


def test_provenance_requires_a_source(provenance: Provenance) -> None:
    """Provenance cannot be constructed without naming where data came from."""
    with pytest.raises(DomainValidationError, match="source must be a SourceReference"):
        Provenance(source=None, events=(ProvenanceEvent("2026-01-01T00:00:00+00:00", "x"),))  # type: ignore[arg-type]


def test_provenance_requires_at_least_one_event(provenance: Provenance) -> None:
    """A datum with no history is unrepresentable."""
    with pytest.raises(DomainValidationError, match="at least one provenance event"):
        Provenance(source=provenance.source, events=())


def test_provenance_rejects_wrong_event_type(provenance: Provenance) -> None:
    with pytest.raises(DomainValidationError, match="must be a ProvenanceEvent"):
        Provenance(source=provenance.source, events=("not-an-event",))  # type: ignore[arg-type]


def test_provenance_processing_is_optional(provenance: Provenance) -> None:
    assert Provenance(source=provenance.source, events=provenance.events).processing == ()


def test_provenance_rejects_wrong_processing_type(provenance: Provenance) -> None:
    with pytest.raises(DomainValidationError, match="must be a ProcessingStep"):
        Provenance(
            source=provenance.source,
            events=provenance.events,
            processing=("resample",),  # type: ignore[arg-type]
        )


def test_provenance_is_hashable_and_comparable(processed_provenance: Provenance) -> None:
    clone = Provenance(
        source=processed_provenance.source,
        events=processed_provenance.events,
        processing=processed_provenance.processing,
    )
    assert clone == processed_provenance
    assert hash(clone) == hash(processed_provenance)


# ----------------------------------------------------------------------
# DataQuality
# ----------------------------------------------------------------------


def test_clean_quality_singleton() -> None:
    assert CLEAN_QUALITY.is_clean
    assert not CLEAN_QUALITY.has
    assert CLEAN_QUALITY.is_usable_for_decision


@pytest.mark.parametrize(
    "flag",
    [
        QualityFlag.MISSING,
        QualityFlag.STALE,
        QualityFlag.OUTLIER,
        QualityFlag.INVALID_RANGE,
        QualityFlag.SENSOR_ERROR,
        QualityFlag.ESTIMATED,
        QualityFlag.INTERPOLATED,
    ],
)
def test_each_adverse_flag_disallows_direct_decision_use(flag: QualityFlag) -> None:
    """No degraded reading may back an action by default.

    This is the strict default Phase 13 may relax, tested per flag so a newly
    added flag cannot silently become decision-usable.
    """
    quality = DataQuality(flags=frozenset({flag}))
    assert quality.has
    assert not quality.is_clean
    assert not quality.is_usable_for_decision


def test_flags_may_co_occur() -> None:
    quality = DataQuality(flags=frozenset({QualityFlag.STALE, QualityFlag.INTERPOLATED}))
    assert quality.has_flag(QualityFlag.STALE)
    assert quality.has_flag(QualityFlag.INTERPOLATED)
    assert not quality.is_clean


def test_ok_cannot_be_combined_with_other_flags() -> None:
    """Mixing ok with missing would defeat any naive `OK in flags` check."""
    with pytest.raises(DomainValidationError, match="cannot be combined"):
        DataQuality(flags=frozenset({QualityFlag.OK, QualityFlag.MISSING}))


def test_empty_flags_are_rejected() -> None:
    """Absence of flags means unknown quality, which must be stated."""
    with pytest.raises(DomainValidationError, match="at least one flag"):
        DataQuality(flags=frozenset())


def test_quality_requires_a_frozenset() -> None:
    with pytest.raises(DomainValidationError, match="frozenset"):
        DataQuality(flags={QualityFlag.OK})  # type: ignore[arg-type]


def test_unknown_flag_is_rejected() -> None:
    # Note: QualityFlag is a StrEnum, so "ok" is a *valid* member value and is
    # deliberately not used here. "corrupt" is genuinely unknown.
    with pytest.raises(DomainValidationError, match="unknown QualityFlag"):
        DataQuality(flags=frozenset({"corrupt"}))  # type: ignore[arg-type]


def test_with_flag_drops_ok() -> None:
    degraded = CLEAN_QUALITY.with_flag(QualityFlag.STALE)
    assert degraded.has_flag(QualityFlag.STALE)
    assert not degraded.has_flag(QualityFlag.OK)
    assert degraded.detail == CLEAN_QUALITY.detail


def test_quality_str_lists_flags() -> None:
    assert "missing" in str(DataQuality(flags=frozenset({QualityFlag.MISSING})))


# ----------------------------------------------------------------------
# TimeBase
# ----------------------------------------------------------------------


def test_time_base_reports_grid_geometry(time_base: TimeBase) -> None:
    assert time_base.timestep == timedelta(minutes=15)
    assert time_base.horizon_duration == timedelta(hours=24)
    assert time_base.horizon_hours == 24.0


def test_time_base_step_time_is_relative_to_origin(time_base: TimeBase) -> None:
    assert time_base.step_time(0) == time_base.origin
    assert time_base.step_time(4) == datetime(2026, 1, 1, 1, 0, tzinfo=timezone.utc)
    assert time_base.step_time(-1) == datetime(2025, 12, 31, 23, 45, tzinfo=timezone.utc)


def test_time_base_step_index_round_trips(time_base: TimeBase) -> None:
    for index in (-96, -1, 0, 1, 95):
        assert time_base.step_index(time_base.step_time(index)) == index


def test_time_base_rejects_off_grid_timestamps(time_base: TimeBase) -> None:
    """Silent rounding would shift the whole system by a partial step."""
    off_grid = time_base.origin + timedelta(minutes=7)
    assert not time_base.is_aligned(off_grid)
    with pytest.raises(DomainValidationError, match="not aligned"):
        time_base.step_index(off_grid)


def test_time_base_rejects_naive_origin() -> None:
    """A naive datetime is ambiguous across daylight-saving transitions."""
    with pytest.raises(DomainValidationError, match="timezone-aware"):
        TimeBase(15, 96, datetime(2026, 1, 1))  # type: ignore[arg-type]


def test_time_base_rejects_naive_moment(time_base: TimeBase) -> None:
    with pytest.raises(DomainValidationError, match="timezone-aware"):
        time_base.step_index(datetime(2026, 1, 1, 1, 0))  # type: ignore[arg-type]


@pytest.mark.parametrize("bad", [0, -15, 7.5, True, "15"])
def test_time_base_rejects_invalid_timestep(bad: object) -> None:
    with pytest.raises(DomainValidationError):
        TimeBase(bad, 96, datetime(2026, 1, 1, tzinfo=timezone.utc))  # type: ignore[arg-type]


@pytest.mark.parametrize("bad", [0, -96, 1.5])
def test_time_base_rejects_invalid_horizon(bad: object) -> None:
    with pytest.raises(DomainValidationError):
        TimeBase(15, bad, datetime(2026, 1, 1, tzinfo=timezone.utc))  # type: ignore[arg-type]


def test_hourly_aggregation_is_available_at_15_minutes(time_base: TimeBase) -> None:
    assert time_base.steps_per_hour() == 4
    assert time_base.aggregate_to_hourly()


def test_hourly_aggregation_unavailable_for_indivisible_timestep() -> None:
    base = TimeBase(7, 96, datetime(2026, 1, 1, tzinfo=timezone.utc))
    assert base.steps_per_hour() is None
    assert not base.aggregate_to_hourly()


def test_time_base_step_time_rejects_non_int(time_base: TimeBase) -> None:
    with pytest.raises(DomainValidationError, match="must be an int"):
        time_base.step_time(1.5)  # type: ignore[arg-type]


def test_time_base_str_is_readable(time_base: TimeBase) -> None:
    assert "15min" in str(time_base)
    assert "96 steps" in str(time_base)


# ----------------------------------------------------------------------
# UncertaintyEstimate
# ----------------------------------------------------------------------


def test_uncertainty_can_be_declared_unknown() -> None:
    """A model may honestly say uncertainty is not yet characterised."""
    estimate = UncertaintyEstimate(kind=UncertaintyKind.UNKNOWN, unit=Unit.KILOWATT)
    assert not estimate.is_quantified
    assert not estimate.is_method_selected
    assert "unquantified" in str(estimate)


def test_uncertainty_quantified_requires_a_number() -> None:
    """An uncertainty that asserts nothing is not an estimate."""
    with pytest.raises(DomainValidationError, match="must quantify something"):
        UncertaintyEstimate(kind=UncertaintyKind.MODEL, unit=Unit.KILOWATT)


def test_uncertainty_rejects_mixing_an_interval_with_a_scalar_summary() -> None:
    """A paired interval is one quantity, but not one that carries a standard deviation.

    Phase 2 counted the two bounds as separate summaries and refused a prediction interval
    outright, which made the one representation Phase 8 exists to produce the one the
    container could not hold. D-099 repaired that: a *pair* is a single quantity. Pairing it
    with a scalar is still refused, because the relationship depends on the
    distributional assumption the interval's own method makes.
    """
    with pytest.raises(DomainValidationError, match="not both"):
        UncertaintyEstimate(
            kind=UncertaintyKind.MODEL,
            unit=Unit.KILOWATT,
            lower_bound=1.0,
            upper_bound=2.0,
            standard_deviation=0.5,
        )
    with pytest.raises(DomainValidationError, match="not both"):
        UncertaintyEstimate(
            kind=UncertaintyKind.MODEL,
            unit=Unit.KILOWATT,
            lower_bound=1.0,
            upper_bound=2.0,
            relative_std=0.1,
        )


def test_uncertainty_accepts_a_paired_prediction_interval() -> None:
    """The repair itself, asserted: an interval is one quantity and is accepted."""
    estimate = UncertaintyEstimate(
        kind=UncertaintyKind.COMBINED,
        unit=Unit.KILOWATT,
        lower_bound=1.0,
        upper_bound=2.0,
        method="conformal_global",
    )
    assert estimate.is_quantified
    assert estimate.is_method_selected
    assert estimate.lower_bound == 1.0
    assert estimate.upper_bound == 2.0


@pytest.mark.parametrize("field", ["lower_bound", "upper_bound"])
def test_uncertainty_rejects_half_an_interval(field: str) -> None:
    """One bound alone is not an interval; it is half a claim."""
    with pytest.raises(DomainValidationError, match="needs both bounds"):
        UncertaintyEstimate(
            kind=UncertaintyKind.PARAMETRIC, unit=Unit.KILOWATT, **{field: 1.0}
        )


@pytest.mark.parametrize(
    "field", ["standard_deviation", "relative_std"]
)
def test_uncertainty_each_scalar_summary_is_accepted_alone(field: str) -> None:
    estimate = UncertaintyEstimate(
        kind=UncertaintyKind.PARAMETRIC, unit=Unit.KILOWATT, **{field: 0.5}
    )
    assert estimate.is_quantified
    # The method is still TBD until a method is selected, independent of whether
    # a number has been attached.
    assert not estimate.is_method_selected


def test_uncertainty_rejects_negative_deviation() -> None:
    with pytest.raises(DomainValidationError, match="non-negative"):
        UncertaintyEstimate(
            kind=UncertaintyKind.MODEL, unit=Unit.KILOWATT, standard_deviation=-0.1
        )


def test_uncertainty_rejects_inverted_bounds() -> None:
    with pytest.raises(DomainValidationError, match="exceeds upper_bound"):
        UncertaintyEstimate(
            kind=UncertaintyKind.PARAMETRIC, unit=Unit.KILOWATT, lower_bound=10.0, upper_bound=1.0
        )


def test_uncertainty_rejects_nan() -> None:
    with pytest.raises(DomainValidationError, match="NaN"):
        UncertaintyEstimate(
            kind=UncertaintyKind.MODEL, unit=Unit.KILOWATT, standard_deviation=float("nan")
        )


def test_uncertainty_rejects_empty_method() -> None:
    with pytest.raises(DomainValidationError, match="method must be"):
        UncertaintyEstimate(
            kind=UncertaintyKind.UNKNOWN, unit=Unit.KILOWATT, method="  "
        )


def test_uncertainty_default_method_is_tbd() -> None:
    estimate = UncertaintyEstimate(
        kind=UncertaintyKind.MODEL, unit=Unit.KILOWATT, relative_std=0.1
    )
    assert estimate.method == TBD_METHOD
    assert not estimate.is_method_selected


def test_uncertainty_rejects_unknown_kind() -> None:
    with pytest.raises(DomainValidationError, match="must be an UncertaintyKind"):
        UncertaintyEstimate(kind="aleatoric", unit=Unit.KILOWATT)  # type: ignore[arg-type]


def test_uncertainty_str_reports_method_when_known() -> None:
    estimate = UncertaintyEstimate(
        kind=UncertaintyKind.MODEL, unit=Unit.KILOWATT, method="quantile", relative_std=0.2
    )
    assert "quantile" in str(estimate)
