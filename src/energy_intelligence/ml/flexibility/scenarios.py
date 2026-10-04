"""Scenario assumptions: the quarantine for numbers SMART-DS does not contain.

The problem this solves
    ---------------------
    A hackathon demonstration eventually needs a battery, or a controllable load fraction,
    or a dispatchable asset that this dataset does not contain. Inventing one silently would
    contaminate every real-data result in the project. Inventing one *explicitly* is
    legitimate and necessary - it is what an assumption is for.

So the two are separated structurally, not by convention:

* :class:`ScenarioAssumptions` is the only place an assumed capability can be declared,
  and every field is named ``assumed_*`` so a reader of a dict key sees it.
* Every value it carries is emitted with :attr:`FlexibilityBasis.ASSUMED` **and** the
  limitations text says it is not evidence.
* :func:`assumed_flexibility` is the only constructor of a scenario-backed estimate, and it
  refuses ``AuthorityLevel.DISPATCHABLE`` unless the scenario *itself* declares a control
  authority. Assuming the capability and assuming the permission are separate acts.
* :func:`assert_real_data_mode` is called by every real-data entry point, so a scenario
  cannot reach a published real-data result by being passed where a config is expected.

What a scenario may **not** do
    It may not alter a real-data result, contribute to a coverage measurement, or appear in
    a registry record as a measurement. Scenario output is written under a different
    artifact name and its registry ``model_family`` says ``scenario_assumption``.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ...domain.enums import (
    CONTROLLABLE_AUTHORITY_LEVELS,
    AuthorityLevel,
    FlexibilityBasis,
    FlexibilityDirection,
    Unit,
)
from ...domain.errors import DomainValidationError
from ...domain.flexibility import AggregationLevel, FlexibilityEstimate
from ...domain.provenance import ProcessingStep, Provenance, ProvenanceEvent, SourceReference

__all__ = [
    "REAL_DATA_MODE",
    "SCENARIO_MODE",
    "ScenarioAssumptions",
    "ScenarioViolation",
    "assumed_flexibility",
    "assert_real_data_mode",
    "load_scenario",
]


class ScenarioViolation(DomainValidationError):
    """Raised when a scenario assumption reaches a real-data code path.

    A distinct type so a caller can catch *this specific mistake* rather than every
    validation failure the domain can raise.
    """


REAL_DATA_MODE = "real_data"
SCENARIO_MODE = "scenario"


@dataclass(frozen=True, slots=True)
class ScenarioAssumptions:
    """Controllability assumptions for demonstration. None of it is measured.

    Attributes:
        name: Identifier for the scenario.
        assumed_controllable_load_fraction: Share of aggregate load the scenario pretends is
            curtailable, in ``[0, 1]``.
        assumed_battery_usable_kwh: Usable battery energy the scenario pretends exists.
            ``None`` means the scenario does not posit a battery.
        assumed_battery_max_kw: Battery power limit the scenario pretends exists.
        assumed_ev_count: Number of vehicles the scenario pretends are connected.
        assumed_ev_max_kw: Aggregate EV charging limit the scenario pretends exists.
        authority: The control authority the scenario *declares*, which is itself an
            assumption. ``NOT_CONTROLLABLE`` by default: a scenario that grants itself
            dispatch authority has to say so explicitly.
        notes: Free-form provenance for the assumptions.
    """

    name: str
    assumed_controllable_load_fraction: float = 0.0
    assumed_battery_usable_kwh: float | None = None
    assumed_battery_max_kw: float | None = None
    assumed_ev_count: int | None = None
    assumed_ev_max_kw: float | None = None
    authority: AuthorityLevel = AuthorityLevel.NOT_CONTROLLABLE
    notes: str = ""

    def __post_init__(self) -> None:
        errors: list[str] = []
        if not isinstance(self.name, str) or not self.name.strip():
            errors.append(f"name must be a non-empty string, got {self.name!r}")
        fraction = self.assumed_controllable_load_fraction
        if isinstance(fraction, bool) or not isinstance(fraction, (int, float)):
            errors.append(
                f"assumed_controllable_load_fraction must be a real number, got {fraction!r}"
            )
        elif not 0.0 <= fraction <= 1.0:
            errors.append(
                f"assumed_controllable_load_fraction must lie in [0, 1], got {fraction}"
            )
        for name in ("assumed_battery_usable_kwh", "assumed_battery_max_kw", "assumed_ev_max_kw"):
            value = getattr(self, name)
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                errors.append(f"{name} must be a real number or None, got {value!r}")
            elif value < 0:
                errors.append(f"{name} must be non-negative, got {value}")
        if self.assumed_ev_count is not None:
            if isinstance(self.assumed_ev_count, bool) or not isinstance(self.assumed_ev_count, int):
                errors.append(
                    f"assumed_ev_count must be an int or None, got {self.assumed_ev_count!r}"
                )
            elif self.assumed_ev_count < 0:
                errors.append(f"assumed_ev_count must be non-negative, got {self.assumed_ev_count}")
        if not isinstance(self.authority, AuthorityLevel):
            errors.append(f"authority must be an AuthorityLevel, got {type(self.authority).__name__}")
        battery = (self.assumed_battery_usable_kwh, self.assumed_battery_max_kw)
        if (battery[0] is None) != (battery[1] is None):
            errors.append(
                "assumed_battery_usable_kwh and assumed_battery_max_kw must be given "
                "together; usable energy without a power limit, or the reverse, is not a "
                "battery"
            )
        ev = (self.assumed_ev_count, self.assumed_ev_max_kw)
        if (ev[0] is None) != (ev[1] is None):
            errors.append(
                "assumed_ev_count and assumed_ev_max_kw must be given together; a vehicle "
                "count without a charging limit is not a charging capability"
            )
        if errors:
            raise DomainValidationError(errors, subject="ScenarioAssumptions")

    @property
    def grants_dispatch_authority(self) -> bool:
        """Whether the scenario declares a controllable authority."""
        return self.authority in CONTROLLABLE_AUTHORITY_LEVELS

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": SCENARIO_MODE,
            "name": self.name,
            "assumed_controllable_load_fraction": float(self.assumed_controllable_load_fraction),
            "assumed_battery_usable_kwh": self.assumed_battery_usable_kwh,
            "assumed_battery_max_kw": self.assumed_battery_max_kw,
            "assumed_ev_count": self.assumed_ev_count,
            "assumed_ev_max_kw": self.assumed_ev_max_kw,
            "authority": self.authority.value,
            "grants_dispatch_authority": self.grants_dispatch_authority,
            "notes": self.notes,
            "warning": (
                "every field is an ASSUMPTION declared for demonstration. None of it is "
                "measured, none of it is evidence about any real asset, and none of it may "
                "contribute to a real-data result."
            ),
        }


def load_scenario(path: Path) -> ScenarioAssumptions:
    """Load a scenario from TOML.

    Args:
        path: The scenario file.

    Returns:
        The assumptions.

    Raises:
        ScenarioViolation: If the file is missing or declares no assumptions.
        ConfigValidationError: If a field is out of range.
    """
    path = Path(path)
    if not path.is_file():
        raise ScenarioViolation([f"scenario file not found: {path}"], subject="ScenarioAssumptions")
    payload = tomllib.loads(path.read_text(encoding="utf-8"))
    assumptions = payload.get("scenario")
    if not isinstance(assumptions, dict):
        raise ScenarioViolation(
            ["a scenario file must contain a [scenario] table"], subject="ScenarioAssumptions"
        )
    allowed = {
        "name",
        "assumed_controllable_load_fraction",
        "assumed_battery_usable_kwh",
        "assumed_battery_max_kw",
        "assumed_ev_count",
        "assumed_ev_max_kw",
        "authority",
        "notes",
    }
    unknown = sorted(set(assumptions) - allowed)
    if unknown:
        raise ScenarioViolation(
            [f"unknown scenario keys: {unknown}"], subject="ScenarioAssumptions"
        )
    if not any(key.startswith("assumed_") for key in assumptions):
        raise ScenarioViolation(
            [
                "a scenario must declare at least one assumed_* value; an empty scenario "
                "would add nothing and its presence would imply capability that is not there"
            ],
            subject="ScenarioAssumptions",
        )
    return ScenarioAssumptions(
        name=str(assumptions.get("name", path.stem)),
        assumed_controllable_load_fraction=float(
            assumptions.get("assumed_controllable_load_fraction", 0.0)
        ),
        assumed_battery_usable_kwh=assumptions.get("assumed_battery_usable_kwh"),
        assumed_battery_max_kw=assumptions.get("assumed_battery_max_kw"),
        assumed_ev_count=assumptions.get("assumed_ev_count"),
        assumed_ev_max_kw=assumptions.get("assumed_ev_max_kw"),
        authority=AuthorityLevel(str(assumptions.get("authority", "not_controllable"))),
        notes=str(assumptions.get("notes", "")),
    )


def assert_real_data_mode(scenario: ScenarioAssumptions | None) -> None:
    """Refuse a scenario inside a real-data code path.

    Args:
        scenario: The scenario in play, or ``None`` for real-data mode.

    Raises:
        ScenarioViolation: If a scenario was supplied.
    """
    if scenario is not None:
        raise ScenarioViolation(
            [
                f"scenario {scenario.name!r} was supplied to a real-data code path. A "
                f"scenario's values are assumptions for demonstration and must not reach a "
                f"measured result. Use the scenario entry point, which tags every output "
                f"ASSUMED."
            ],
            subject="ScenarioAssumptions",
        )


def assumed_flexibility(
    *,
    scenario: ScenarioAssumptions,
    target: str,
    direction: FlexibilityDirection,
    magnitude_kw: float,
    timestamp: str,
    aggregation: AggregationLevel = AggregationLevel.SYSTEM,
    asset_id: Any = None,
    horizon_steps: int | None = None,
    limitations: tuple[str, ...] = (),
) -> FlexibilityEstimate:
    """Build an explicitly-assumed flexibility estimate.

    The authority comes from the **scenario**, never from the data: a scenario that wants
    ``DISPATCHABLE`` has to declare it, and that declaration is visible in the estimate's
    own fields.

    Args:
        scenario: The assumptions in force.
        target: What the claim is about.
        direction: Which way net demand would move.
        magnitude_kw: The assumed magnitude, kW.
        timestamp: Validity instant.
        aggregation: Scope of the claim.
        asset_id: Typed asset reference, when the claim is asset-scoped.
        horizon_steps: Horizon the claim applies to.
        limitations: Extra caveats, appended after the mandatory ones.

    Returns:
        The estimate, tagged ``ASSUMED``.

    Raises:
        ScenarioViolation: If the magnitude is not a positive real number.
    """
    if isinstance(magnitude_kw, bool) or not isinstance(magnitude_kw, (int, float)):
        raise ScenarioViolation(
            [f"magnitude_kw must be a real number, got {magnitude_kw!r}"],
            subject="ScenarioAssumptions",
        )
    if magnitude_kw < 0:
        raise ScenarioViolation(
            [f"magnitude_kw must be non-negative, got {magnitude_kw}"],
            subject="ScenarioAssumptions",
        )
    provenance = Provenance(
        source=SourceReference(
            source_id=f"scenario-{scenario.name}",
            dataset="scenario_assumptions",
            locator=f"scenarios/{scenario.name}.toml",
            version="1",
        ),
        events=(
            ProvenanceEvent(
                timestamp=timestamp,
                event="scenario_declared",
                detail=(
                    f"assumptions declared for demonstration; SMART-DS establishes none of "
                    f"them (G-01, G-02, G-03)"
                ),
            ),
        ),
        processing=(
            ProcessingStep(
                name="scenario_application",
                version="phase9",
                detail=f"scenario={scenario.name}",
            ),
        ),
    )
    mandatory = (
        "ASSUMED, not measured: this figure is declared by a scenario and is not evidence "
        "about any real asset",
        f"the scenario's declared control authority is {scenario.authority.value!r}",
    )
    if not scenario.grants_dispatch_authority:
        mandatory = mandatory + (
            "the scenario grants no control authority, so this is still not dispatchable",
        )
    return FlexibilityEstimate(
        target=target,
        direction=direction,
        basis=FlexibilityBasis.ASSUMED,
        authority=scenario.authority,
        magnitude_kw=float(magnitude_kw),
        unit=Unit.KILOWATT,
        timestamp=timestamp,
        provenance=provenance,
        method=f"scenario[{scenario.name}]",
        horizon_steps=horizon_steps,
        aggregation=aggregation,
        asset_id=asset_id,
        limitations=mandatory + tuple(limitations),
    )