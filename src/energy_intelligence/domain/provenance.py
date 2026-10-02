"""Provenance: where an energy datum came from.

Carried by every observation, state, forecast, action, constraint and objective.

Motivation is taken directly from SAT-SA, where the principle is that data must
have provenance rather than appearing magically inside the system (see
``decisions.md`` in that repository and D-032 here). SAT-SA enforces this with
hash-chained ledgers and cryptographic signatures; Phase 2 deliberately does
**not** reproduce that machinery. It establishes the *contract* instead: a
provenance record is mandatory on every domain object, cannot be absent, and is
validated. Whether later phases add digests or ledgers is their decision.

The chain modelled here is:

    SourceReference -> ProvenanceEvent -> ProcessingStep -> Provenance
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .errors import DomainValidationError

__all__ = ["SourceReference", "ProvenanceEvent", "ProcessingStep", "Provenance"]


@dataclass(frozen=True, slots=True)
class SourceReference:
    """An immutable pointer to one external data source.

    Attributes:
        source_id: Stable identifier of the source, e.g. ``"smartds-oes-feeder-a"``.
        dataset: Logical dataset name, e.g. ``"load_profile"``.
        locator: Where the data physically lives, e.g. a relative path.
        version: Version of the source. Required so a result can be tied to an
            exact input.
        checksum: Content digest of the source, if one is known. Optional in
            Phase 2 because no dataset has been ingested yet; a later phase
            should make it mandatory.
    """

    source_id: str
    dataset: str
    locator: str
    version: str
    checksum: str | None = None

    def __post_init__(self) -> None:
        errors: list[str] = []
        for field_name in ("source_id", "dataset", "locator", "version"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                errors.append(f"{field_name} must be a non-empty string, got {value!r}")
        if errors:
            raise DomainValidationError(errors, subject="SourceReference")


@dataclass(frozen=True, slots=True)
class ProvenanceEvent:
    """One timestamped point in a datum's history.

    Attributes:
        timestamp: When the event occurred, ISO-8601.
        event: What happened. Free-form because the event vocabulary belongs to
            the pipeline that produces data (Phase 3), not to the data model.
        detail: Optional human-readable elaboration.
    """

    timestamp: str
    event: str
    detail: str = ""

    def __post_init__(self) -> None:
        errors: list[str] = []
        if not isinstance(self.timestamp, str) or not self.timestamp.strip():
            errors.append(f"timestamp must be a non-empty ISO-8601 string, got {self.timestamp!r}")
        if not isinstance(self.event, str) or not self.event.strip():
            errors.append(f"event must be a non-empty string, got {self.event!r}")
        if errors:
            raise DomainValidationError(errors, subject="ProvenanceEvent")


@dataclass(frozen=True, slots=True)
class ProcessingStep:
    """One transformation applied between events.

    Attributes:
        name: Name of the operation, e.g. ``"resample_mean"``.
        version: Version of the implementation, so results remain reproducible
            after the code changes.
        detail: Optional description, including parameters.
    """

    name: str
    version: str
    detail: str = ""

    def __post_init__(self) -> None:
        errors: list[str] = []
        for field_name in ("name", "version"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                errors.append(f"{field_name} must be a non-empty string, got {value!r}")
        if errors:
            raise DomainValidationError(errors, subject="ProcessingStep")


@dataclass(frozen=True, slots=True)
class Provenance:
    """Complete provenance for one domain datum.

    A ``Provenance`` with no source is not constructible: this is the mechanism
    that stops provenance from silently disappearing downstream, which is the
    specific failure mode the SAT-SA project documented and guarded against.

    Attributes:
        source: The originating external source. Mandatory.
        events: Ordered history of what happened to the datum. At minimum one
            event is required, so a datum can never exist with no history.
        processing: Transformations applied. Empty is valid for raw,
            unprocessed data.
    """

    source: SourceReference
    events: tuple[ProvenanceEvent, ...] = field(default_factory=tuple)
    processing: tuple[ProcessingStep, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        errors: list[str] = []
        if not isinstance(self.source, SourceReference):
            errors.append(
                f"source must be a SourceReference, got {type(self.source).__name__}"
            )
        if not isinstance(self.events, tuple) or not self.events:
            errors.append(
                "events must be a non-empty tuple; a datum must have at least "
                "one provenance event"
            )
        elif not all(isinstance(event, ProvenanceEvent) for event in self.events):
            errors.append("every entry in events must be a ProvenanceEvent")
        if not isinstance(self.processing, tuple):
            errors.append(f"processing must be a tuple, got {type(self.processing).__name__}")
        elif not all(isinstance(step, ProcessingStep) for step in self.processing):
            errors.append("every entry in processing must be a ProcessingStep")

        if errors:
            raise DomainValidationError(errors, subject="Provenance")
