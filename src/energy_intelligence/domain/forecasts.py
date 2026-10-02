"""Forecasts.

A forecast is a *claim about the future with an attached uncertainty*, which is
the shape every downstream phase needs:

* Phase 4/5 consume it to produce predictions;
* Phase 8 attaches a real uncertainty method in place of ``TBD``;
* Phase 13 reads :attr:`Forecast.uncertainty` when deciding whether to trust a
  decision;
* Phase 17 measures calibration, which requires the uncertainty to have been
  recorded at issuance time rather than reconstructed later.

Two fields carry deliberate rigour:

* ``issued_at`` must be strictly earlier than ``target_time``. A forecast stamped
  at or after the moment it describes is not a forecast, and admitting one would
  quietly contaminate any later evaluation with hindsight.
* ``horizon_steps`` is recorded explicitly rather than derived, so a horizon that
  disagrees with the governing :class:`~energy_intelligence.domain.timebase.TimeBase`
  can be detected rather than assumed consistent.

Uncertainty is optional here -- not because it should be, but because Phase 2
cannot yet produce one. :attr:`Forecast.has_uncertainty` reports the absence
explicitly so a consumer can refuse to act on an unquantified forecast.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from .enums import VariableRole
from .errors import DomainValidationError
from .identifiers import ForecastId
from .provenance import Provenance
from .quantities import Quantity
from .uncertainty import UncertaintyEstimate

__all__ = ["Forecast"]


@dataclass(frozen=True, slots=True)
class Forecast:
    """One predicted value for one variable at one future time.

    Attributes:
        forecast_id: Unique identifier.
        variable: Name of the predicted variable.
        role: Always ``OBSERVATION``. A forecast predicts something the system
            observes; it is never itself an action. Enforced so a predicted
            variable cannot be laundered into a decision variable.
        target_time: ISO-8601 instant being predicted.
        issued_at: ISO-8601 instant the forecast was produced.
        horizon_steps: Steps between issuance and target, on the governing grid.
        value: The point prediction.
        uncertainty: Optional quantitative or qualitative uncertainty. Expected
            to become mandatory once Phase 8 selects a method.
        provenance: Which model, data and configuration produced this forecast.
            Mandatory, and the field that makes a forecast auditable.
    """

    forecast_id: ForecastId
    variable: str
    role: VariableRole
    target_time: str
    issued_at: str
    horizon_steps: int
    value: Quantity
    provenance: Provenance
    uncertainty: UncertaintyEstimate | None = None

    def __post_init__(self) -> None:
        errors: list[str] = []

        if not isinstance(self.forecast_id, ForecastId):
            errors.append(f"forecast_id must be a ForecastId, got {type(self.forecast_id).__name__}")
        if not isinstance(self.variable, str) or not self.variable.strip():
            errors.append(f"variable must be a non-empty string, got {self.variable!r}")

        if self.role is not VariableRole.OBSERVATION:
            errors.append(
                f"role must be {VariableRole.OBSERVATION.value!r} on a Forecast; "
                f"got {getattr(self.role, 'value', self.role)!r}. A forecast predicts "
                "an observed variable and is never itself an action."
            )

        if not isinstance(self.value, Quantity):
            errors.append(f"value must be a Quantity, got {type(self.value).__name__}")
        if not isinstance(self.provenance, Provenance):
            errors.append(f"provenance must be a Provenance, got {type(self.provenance).__name__}")

        if isinstance(self.horizon_steps, bool) or not isinstance(self.horizon_steps, int):
            errors.append(f"horizon_steps must be an int, got {type(self.horizon_steps).__name__}")
        elif self.horizon_steps < 0:
            errors.append(f"horizon_steps must be non-negative, got {self.horizon_steps}")

        issued = _parse(issued_at := self.issued_at, "issued_at", errors)
        target = _parse(target_time := self.target_time, "target_time", errors)
        if issued is not None and target is not None:
            if issued >= target:
                errors.append(
                    f"issued_at {issued_at} must be strictly before target_time "
                    f"{target_time}; a forecast cannot be issued at or after the "
                    "moment it describes"
                )

        if self.uncertainty is not None:
            if not isinstance(self.uncertainty, UncertaintyEstimate):
                errors.append(
                    f"uncertainty must be an UncertaintyEstimate or None, "
                    f"got {type(self.uncertainty).__name__}"
                )
            elif self.uncertainty.unit != self.value.unit:
                errors.append(
                    f"uncertainty unit {self.uncertainty.unit.value} must match the "
                    f"forecast value unit {self.value.unit.value}"
                )

        if errors:
            raise DomainValidationError(errors, subject="Forecast")

    @property
    def has_uncertainty(self) -> bool:
        return self.uncertainty is not None

    @property
    def is_quantified_uncertainty(self) -> bool:
        """Whether the uncertainty carries an actual numeric summary."""
        return self.uncertainty is not None and self.uncertainty.is_quantified


def _parse(value: object, label: str, errors: list[str]) -> datetime | None:
    """Parse an ISO-8601 string, recording an error rather than raising."""
    if not isinstance(value, str) or not value.strip():
        errors.append(f"{label} must be a non-empty ISO-8601 string, got {value!r}")
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        errors.append(f"{label} is not valid ISO-8601: {value!r}")
        return None
