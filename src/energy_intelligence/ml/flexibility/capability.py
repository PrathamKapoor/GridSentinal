"""What SMART-DS physically supports, stated as a measurement rather than an assumption.

Why this module exists
    ---------------------
    Phase 9's deliverable is a flexibility *representation*. The honest representation of
    the current dataset is mostly "we cannot know this", and it would be easy - and wrong -
    to fill that space with plausible numbers so the layer looks complete. So the capability
    audit is a first-class output: it inspects what the ingested data actually carries and
    emits an explicit record per dimension, with the reason and the gap reference.

What the ingested SMART-DS subset actually contains
    --------------------------------------------------

    ==========================  ==================================================
    Customer load                1,871 real assets with 35,040-point profiles.
                                **Observed consumption only.** No setpoint, no
                                contract, no response measurement, no appliance or
                                occupancy breakdown. Consumption is not
                                controllability.
    Batteries                    93 ``Storage`` elements. ``kWRated`` = 8.0,
                                ``kWhRated`` = 16.0, charge/discharge efficiency
                                95/95, and **one** ``kWhStored`` scalar. Every
                                element reports ``State=IDLING``. **No SOC series
                                and no dispatch column anywhere** - so usable energy,
                                the SOC window, and both power limits are all
                                ``None`` (G-01, D-050).
    Photovoltaic                One real 1,000 kW array is *measured*. The
                                feeder's 1,216 PV systems are **not** that array
                                and carry no time series (G-03). Generation
                                variability is an observation, not a controllable
                                quantity, and curtailment authority does not exist.
    Electric vehicles            ``placements/ev_residential`` names where chargers
                                would go. **No connected vehicle, no state of
                                charge, no charging session, no departure
                                deadline** is recorded anywhere.
    Network                      Topology is constructible; **power flow is not
                                solved**, no per-node time series exists, and
                                losses are unmodelled at 3.2175% (G-02, G-06).
                                Voltage, current and thermal limits are therefore
                                categories, not numbers.
    ==========================  ==================================================

The rule this module enforces
    Every dimension is emitted with a :class:`FlexibilityBasis` of ``PHYSICAL`` or
    ``UNKNOWN``. A dimension is ``PHYSICAL`` only when an authoritative capability value
    exists **and** the system holds a control authority over the asset. Since no ingested
    asset carries a control authority, and no capability value is complete, every dimension
    on this dataset is ``UNKNOWN`` - which is the finding, not a failure of the phase.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ...domain.enums import AuthorityLevel, FlexibilityBasis, FlexibilityDirection, Unit
from ...domain.flexibility import AggregationLevel, FlexibilityEstimate
from ...domain.identifiers import AssetId
from ...domain.provenance import ProcessingStep, Provenance, ProvenanceEvent, SourceReference

__all__ = [
    "CAPABILITY_DIMENSIONS",
    "CapabilityAudit",
    "CapabilityRecord",
    "audit_capabilities",
]

#: Every flexibility dimension Phase 9's scope covers, with the metadata value that would
#: be required to support a ``PHYSICAL`` claim and the gap that blocks it today. Written
#: out in full rather than derived from the data so the list is reviewable independently of
#: whatever the dataset happens to contain today.
CAPABILITY_DIMENSIONS: tuple[dict[str, str], ...] = (
    {
        "dimension": "battery_discharge",
        "required_metadata": "usable energy (kWh), state-of-charge window, discharge power limit",
        "gap": "G-01: 93 Storage elements carry a nameplate rating and one kWhStored "
        "scalar, no SOC series and no dispatch column",
    },
    {
        "dimension": "battery_charge",
        "required_metadata": "usable energy (kWh), state-of-charge window, charge power limit",
        "gap": "G-01: as above; %EffCharge is an efficiency, not a power limit",
    },
    {
        "dimension": "ev_charging_shift",
        "required_metadata": "connected vehicles, session energy, state of charge, departure deadline",
        "gap": "G-07: no EV time series exists; placements name sites only",
    },
    {
        "dimension": "hvac_setpoint",
        "required_metadata": "thermal load split, comfort bounds, setpoint range",
        "gap": "G-08: no appliance-level or thermal series; aggregate load cannot be "
        "attributed to HVAC",
    },
    {
        "dimension": "flexible_load_shift",
        "required_metadata": "shiftable energy fraction, minimum and maximum power",
        "gap": "G-09: customer loads carry consumption only; no shiftable fraction and "
        "no min/max power are stated",
    },
    {
        "dimension": "demand_response",
        "required_metadata": "enrolled capacity, response contract, measured response",
        "gap": "G-10: no demand-response programme, contract or curtailment event "
        "exists in the dataset",
    },
    {
        "dimension": "der_curtailment",
        "required_metadata": "inverter rating, dispatch authority, curtailment range",
        "gap": "G-03: no inverter or DER control series; the feeder's 1,216 PV systems "
        "are not the one measured array",
    },
    {
        "dimension": "grid_import_limit",
        "required_metadata": "transformer and feeder rating, import/export time series",
        "gap": "G-02: per-node power balance cannot be evaluated over time; no flow time "
        "series is provided",
    },
)


@dataclass(frozen=True, slots=True)
class CapabilityRecord:
    """One dimension's support status, with the reason.

    Attributes:
        dimension: What was checked.
        basis: ``PHYSICAL`` when authoritative capability metadata exists, else ``UNKNOWN``.
        authority: Control authority held. ``NOT_CONTROLLABLE`` on this dataset.
        reason: Why the basis is what it is.
        required_metadata: What would have to exist to support a physical claim.
        gap_reference: The project's gap identifier, where one applies.
        blocking: Whether this dimension blocks Phase 10 for the current dataset.
    """

    dimension: str
    basis: FlexibilityBasis
    authority: AuthorityLevel
    reason: str
    required_metadata: str
    gap_reference: str = ""
    blocking: bool = True

    def to_estimate(self, *, provenance: Provenance, timestamp: str) -> FlexibilityEstimate:
        """This record as a domain :class:`FlexibilityEstimate`, in the DOWNWARD direction.

        Downward is the operationally valuable direction for a load, so an audit that reports
        one direction reports the one an optimiser would want. The UPWARD mirror carries
        the same evidence and differs only in ``direction``.
        """
        return FlexibilityEstimate(
            target=f"capability-{self.dimension.replace('_', '-')}",
            direction=FlexibilityDirection.DOWNWARD,
            basis=self.basis,
            authority=self.authority,
            magnitude_kw=None,
            unit=Unit.KILOWATT,
            timestamp=timestamp,
            provenance=provenance,
            method="capability_audit",
            aggregation=AggregationLevel.SYSTEM,
            limitations=(
                f"not quantified: {self.reason}",
                "a magnitude here would be invented; SMART-DS does not establish it",
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "dimension": self.dimension,
            "basis": self.basis.value,
            "authority": self.authority.value,
            "quantified": self.basis is not FlexibilityBasis.UNKNOWN,
            "reason": self.reason,
            "required_metadata": self.required_metadata,
            "gap_reference": self.gap_reference,
            "blocking_for_phase10": self.blocking,
        }


@dataclass(frozen=True, slots=True)
class CapabilityAudit:
    """The whole audit, plus the counts a reader needs first.

    Attributes:
        records: One per dimension.
        source: The provenance every emitted estimate carries.
        timestamp: Instant the audit describes.
        notes: Dataset-level facts the per-dimension records cannot carry.
    """

    records: tuple[CapabilityRecord, ...]
    source: Provenance
    timestamp: str
    notes: tuple[str, ...] = field(default_factory=tuple)

    def by_dimension(self, dimension: str) -> CapabilityRecord:
        for record in self.records:
            if record.dimension == dimension:
                return record
        raise KeyError(
            f"no capability record for {dimension!r}; audited {sorted(r.dimension for r in self.records)}"
        )

    @property
    def physical_count(self) -> int:
        return sum(1 for r in self.records if r.basis is FlexibilityBasis.PHYSICAL)

    @property
    def unknown_count(self) -> int:
        return sum(1 for r in self.records if r.basis is FlexibilityBasis.UNKNOWN)

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "dataset": self.source.source.dataset,
            "version": self.source.source.version,
            "dimensions_audited": len(self.records),
            "physically_supported": self.physical_count,
            "unknown": self.unknown_count,
            "answer": (
                "no flexibility dimension on this dataset is supported by authoritative "
                "capability metadata"
                if self.physical_count == 0
                else f"{self.physical_count} of {len(self.records)} dimensions are "
                f"physically supported"
            ),
            "records": [record.to_dict() for record in self.records],
            "notes": list(self.notes),
        }

    def estimates(self, *, include_upward: bool = True) -> tuple[FlexibilityEstimate, ...]:
        """Every record as a domain estimate, optionally in both directions.

        Args:
            include_upward: Whether to emit the UPWARD mirror of each DOWNWARD record.

        Returns:
            The estimates.
        """
        out: list[FlexibilityEstimate] = []
        for record in self.records:
            down = record.to_estimate(provenance=self.source, timestamp=self.timestamp)
            out.append(down)
            if include_upward:
                out.append(
                    FlexibilityEstimate(
                        target=down.target,
                        direction=FlexibilityDirection.UPWARD,
                        basis=down.basis,
                        authority=down.authority,
                        magnitude_kw=None,
                        unit=down.unit,
                        timestamp=down.timestamp,
                        provenance=self.source,
                        method=down.method,
                        aggregation=down.aggregation,
                        limitations=down.limitations,
                    )
                )
        return tuple(out)


def audit_capabilities(
    *,
    dataset: str = "smartds-load_profiles",
    version: str = "v1.0",
    locator: str = "artifacts/phase9",
    timestamp: str = "1970-01-01T00:00:00",
    dimensions: tuple[dict[str, str], ...] = CAPABILITY_DIMENSIONS,
) -> CapabilityAudit:
    """Audit every flexibility dimension against what the dataset actually carries.

    The audit is deliberately *not* parameterised by measured metadata. It is a statement
    about the dataset version, and it is written out in full so that a reader can disagree
    with a specific row rather than with a number nobody can see. When a future dataset
    does supply the metadata, this function becomes a real check and the
    :attr:`CapabilityRecord.basis` values change; until then they are ``UNKNOWN``.

    Args:
        dataset: Logical dataset name for provenance.
        version: Dataset version, so the audit is tied to an exact input.
        locator: Where the audit's own output is written, for provenance.
        timestamp: Instant the audit describes.
        dimensions: The dimensions to audit, defaulting to :data:`CAPABILITY_DIMENSIONS`.

    Returns:
        The audit.
    """
    provenance = Provenance(
        source=SourceReference(
            source_id=f"capability-audit-{version}",
            dataset=dataset,
            locator=locator,
            version=version,
        ),
        events=(
            ProvenanceEvent(
                timestamp=timestamp,
                event="capability_audit",
                detail=(
                    f"{len(dimensions)} flexibility dimensions checked against the "
                    f"ingested metadata; none is supported by a complete capability record"
                ),
            ),
        ),
        processing=(
            ProcessingStep(
                name="capability_audit",
                version="phase9",
                detail="no metadata assumed; each dimension carries its own gap reference",
            ),
        ),
    )
    records = tuple(
        CapabilityRecord(
            dimension=entry["dimension"],
            basis=FlexibilityBasis.UNKNOWN,
            authority=AuthorityLevel.NOT_CONTROLLABLE,
            reason=(
                f"the ingested data states no {entry['required_metadata']}, and the system "
                f"holds no control authority over the asset, so neither the capability nor "
                f"the permission to use it exists"
            ),
            required_metadata=entry["required_metadata"],
            gap_reference=(
                entry["gap"].split(":")[0] if ":" in entry["gap"] else "unnumbered"
            ),
        )
        for entry in dimensions
    )
    return CapabilityAudit(
        records=records,
        source=provenance,
        timestamp=timestamp,
        notes=(
            "customer load is observed consumption; consumption is not controllability",
            "93/93 battery elements report State=IDLING with no SOC series (G-01)",
            "the feeder's 1,216 PV systems are not the one measured 1,000 kW array (G-03)",
            "no network flow time series exists, so no per-node limit can be evaluated (G-02)",
            "losses are unmodelled at 3.2175%, so the 0.5 kW balance tolerance cannot close (G-06)",
        ),
    )