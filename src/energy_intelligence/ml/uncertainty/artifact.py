"""``ProbabilisticForecast``: the output a future optimiser or assurance engine consumes.

This is the phase's stable interface, and it is deliberately built on the Phase 2 domain
objects rather than beside them. A Phase 8 forecast is not a new kind of thing; it is a
Phase 2 :class:`~energy_intelligence.domain.forecasts.Forecast` whose optional
``uncertainty`` field is finally populated, which is exactly what
``domain/uncertainty.py`` said Phase 2 was deferring.

One domain defect had to be fixed to make that possible, and it is worth stating because it
was not a Phase 8 choice, it was a Phase 2 gap this phase exposed.

``UncertaintyEstimate`` rejected a two-sided interval. Its rule counted ``lower_bound`` and
``upper_bound`` as two separate quantities and refused more than one, so the one
representation Phase 8 exists to produce - a prediction interval - was the one
representation the container could not hold. The stated reason was "how an interval relates
to a standard deviation depends on a distributional assumption that Phase 8 has not yet
made", so the rule is repaired here rather than worked around: a **paired** interval is one
quantity, and pairing it with a standard deviation is still refused because that
relationship remains method-dependent. Recorded in ``decisions.md`` D-099.

What this object is for downstream, and what it is not
-----------------------------------------------------

It carries enough for a future consumer to decide whether to rely on a forecast: the point
value, an interval at a stated level, a *measured* calibration status rather than an
assumed one, a coarse uncertainty band, the provenance of the model that produced it, and
the data-quality state of the row.

It does **not** carry a decision, a confidence score for an action, or anything derived
from uncertainty and an objective together. Those belong to Phase 13's decision-assurance
layer, and defining them here would fix an assurance formula on the evidence of one
forecasting experiment.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np

from ...domain.enums import Unit, UncertaintyKind, VariableRole
from ...domain.forecasts import Forecast
from ...domain.identifiers import ForecastId
from ...domain.provenance import ProcessingStep, Provenance, ProvenanceEvent, SourceReference
from ...domain.quantities import Quantity
from ...domain.uncertainty import UncertaintyEstimate
from .intervals import PredictionInterval

__all__ = [
    "CALIBRATION_PASS",
    "CALIBRATION_FAIL",
    "CALIBRATION_UNKNOWN",
    "UncertaintyBand",
    "ProbabilisticForecast",
    "ProbabilisticForecastBatch",
    "interval_to_estimate",
    "forecast_to_domain",
]

CALIBRATION_PASS = "PASS"
CALIBRATION_FAIL = "FAIL"
CALIBRATION_UNKNOWN = "NOT_EVALUATED"


class UncertaintyBand(StrEnum):
    """A coarse, three-level uncertainty label.

    Deliberately three levels and not a percentage. A continuous uncertainty number is
    what the interval already is; the band's job is to be readable by an operator at a
    glance, and a band with more levels stops being glanceable. The cut points are
    terciles of the *calibration* width distribution, so the bands mean "narrower than a
    third of calibrated forecasts, typical, wider" - not an absolute kW threshold that
    would drift with the model.
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass(frozen=True, slots=True)
class ProbabilisticForecast:
    """One forecast with an interval, a calibration status and everything to audit it.

    Attributes:
        timestamp: The instant being forecast, ISO-8601.
        asset_id: Which asset, in the repository's typed identifier form.
        horizon_steps: Steps ahead on the governing 15-minute grid.
        point_kw: The point forecast, kW.
        lower_kw: Interval lower bound, kW.
        upper_kw: Interval upper bound, kW.
        nominal_level: Stated coverage level.
        uncertainty_kind: Which source of uncertainty this interval represents. Carried
            through to the domain object so a consumer can refuse an epistemic interval
            where it needed an aleatoric one.
        calibration_status: ``PASS``/``FAIL``/``NOT_EVALUATED``, measured elsewhere in
            this phase rather than asserted here.
        coverage_error: Measured coverage minus nominal. Signed.
        band: Coarse label derived from the calibrated width.
        method: The interval procedure that produced these bounds.
        model_version: Identifier of the point-forecast model.
        data_quality: The row's quality state, as the domain's flag values.
        uncertainty_notes: Caveats, e.g. the calibration sample size.
    """

    timestamp: str
    asset_id: str
    horizon_steps: int
    point_kw: float
    lower_kw: float
    upper_kw: float
    nominal_level: float
    uncertainty_kind: UncertaintyKind = UncertaintyKind.COMBINED
    calibration_status: str = CALIBRATION_UNKNOWN
    coverage_error: float | None = None
    band: UncertaintyBand = UncertaintyBand.MEDIUM
    method: str = ""
    model_version: str = ""
    data_quality: tuple[str, ...] = ("ok",)
    uncertainty_notes: str = ""

    def __post_init__(self) -> None:
        for name in ("timestamp", "asset_id", "method"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string, got {value!r}")
        if not isinstance(self.asset_id, str) or not self.asset_id.startswith("asset-"):
            raise ValueError(
                f"asset_id must be a repository AssetId value such as 'asset-load-1', "
                f"got {self.asset_id!r}"
            )
        for name in ("point_kw", "lower_kw", "upper_kw"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a real number, got {value!r}")
            if value != value:
                raise ValueError(f"{name} must not be NaN")
        if self.lower_kw > self.upper_kw:
            raise ValueError(
                f"lower_kw {self.lower_kw} exceeds upper_kw {self.upper_kw}"
            )
        if not 0.0 < float(self.nominal_level) < 1.0:
            raise ValueError(
                f"nominal_level must lie in (0, 1), got {self.nominal_level}"
            )
        if not isinstance(self.horizon_steps, int) or self.horizon_steps <= 0:
            raise ValueError(
                f"horizon_steps must be a positive int, got {self.horizon_steps!r}"
            )
        if self.calibration_status not in {
            CALIBRATION_PASS,
            CALIBRATION_FAIL,
            CALIBRATION_UNKNOWN,
        }:
            raise ValueError(
                f"calibration_status must be one of PASS/FAIL/NOT_EVALUATED, got "
                f"{self.calibration_status!r}"
            )
        if not isinstance(self.band, UncertaintyBand):
            raise ValueError(
                f"band must be an UncertaintyBand, got {type(self.band).__name__}"
            )
        if not isinstance(self.uncertainty_kind, UncertaintyKind):
            raise ValueError(
                f"uncertainty_kind must be an UncertaintyKind, got "
                f"{type(self.uncertainty_kind).__name__}"
            )

    @property
    def width_kw(self) -> float:
        """Interval width in kW."""
        return self.upper_kw - self.lower_kw

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "asset_id": self.asset_id,
            "horizon_steps": int(self.horizon_steps),
            "point_kw": float(self.point_kw),
            "lower_kw": float(self.lower_kw),
            "upper_kw": float(self.upper_kw),
            "width_kw": float(self.width_kw),
            "nominal_level": float(self.nominal_level),
            "uncertainty_kind": self.uncertainty_kind.value,
            "calibration_status": self.calibration_status,
            "coverage_error": (
                None if self.coverage_error is None else float(self.coverage_error)
            ),
            "band": self.band.value,
            "method": self.method,
            "model_version": self.model_version,
            "data_quality": list(self.data_quality),
            "uncertainty_notes": self.uncertainty_notes,
        }

    def render(self) -> str:
        """The conceptual output of the phase, as text."""
        coverage = (
            "not evaluated"
            if self.coverage_error is None
            else f"error {self.coverage_error:+.3f}"
        )
        return (
            f"Forecast: {self.point_kw:.2f} kW\n"
            f"Prediction interval: [{self.lower_kw:.2f}, {self.upper_kw:.2f}] kW\n"
            f"Nominal coverage: {self.nominal_level:.0%}\n"
            f"Calibration status: {self.calibration_status} ({coverage})\n"
            f"Uncertainty: {self.band.value.upper()}"
        )


def interval_to_estimate(
    *,
    interval: PredictionInterval,
    kind: UncertaintyKind = UncertaintyKind.COMBINED,
    unit: Unit = Unit.KILOWATT,
) -> UncertaintyEstimate:
    """Convert a :class:`PredictionInterval` into the Phase 2 domain container.

    This is the join between the two layers and it is one line of real logic: an interval
    becomes ``lower_bound`` and ``upper_bound``, which the domain object permits only after
    D-099's repair, and ``method`` carries the procedure name so a consumer can tell a
    conformal interval from an empirical-quantile one.

    Args:
        interval: A single-element interval.
        kind: Which source of uncertainty it represents.
        unit: The bound's unit.

    Returns:
        The domain estimate.

    Raises:
        ValueError: If the interval does not hold exactly one entry.
    """
    if interval.n_rows != 1:
        raise ValueError(
            f"interval_to_estimate takes a single forecast's interval, got "
            f"{interval.n_rows} entries"
        )
    # Flattened rather than indexed at ``[0]``: a single-forecast interval is naturally
    # shaped ``[1, 1]``, so ``lower_kw[0]`` is a length-1 array and ``float()`` on it raises
    # rather than converting.
    lower = np.asarray(interval.lower_kw, dtype=np.float64).reshape(-1)
    upper = np.asarray(interval.upper_kw, dtype=np.float64).reshape(-1)
    notes = interval.notes or f"prediction interval at nominal {interval.nominal_level:.0%}"
    return UncertaintyEstimate(
        kind=kind,
        unit=unit,
        method=interval.method,
        lower_bound=float(lower[0]),
        upper_bound=float(upper[0]),
        notes=notes,
    )


def forecast_to_domain(
    forecast: ProbabilisticForecast,
    *,
    variable: str = "customer_load",
    issued_at: str | None = None,
    forecast_id: str | None = None,
    source: SourceReference | None = None,
) -> Forecast:
    """Build the Phase 2 :class:`Forecast` this probabilistic forecast represents.

    A future consumer that already speaks the domain model needs no new type; it needs the
    existing one, populated. Every field is filled from the
    :class:`ProbabilisticForecast` and nothing is invented.

    Args:
        forecast: The probabilistic forecast.
        variable: The domain variable name.
        issued_at: Issuance instant. Defaults to 15 minutes before ``timestamp``, which is
            only correct for the shortest horizon and is therefore rejected rather than
            silently applied when the horizon disagrees - see the check below.
        forecast_id: Identifier. Defaults to one derived from asset, variable and horizon.
        source: Provenance source. A Phase 8 placeholder is used if omitted.

    Returns:
        The domain forecast.

    Raises:
        ValueError: If the issuance instant is not strictly before the target.
    """
    from datetime import datetime, timedelta

    step_minutes = 15
    horizon_minutes = int(forecast.horizon_steps) * step_minutes
    target = datetime.fromisoformat(forecast.timestamp)
    if issued_at is None:
        issued = target - timedelta(minutes=horizon_minutes)
    else:
        issued = datetime.fromisoformat(issued_at)
    if issued >= target:
        raise ValueError(
            f"issued_at {issued.isoformat()} must be strictly before target_time "
            f"{target.isoformat()}; a forecast cannot be issued at or after the moment "
            f"it describes"
        )
    identifier = forecast_id or (
        f"fc-{forecast.asset_id.replace('asset-', '', 1)}-{variable}-h{forecast.horizon_steps}"
    )
    reference = source or SourceReference(
        source_id="phase8-uncertainty",
        dataset=variable,
        locator=forecast.model_version or "unknown",
        version=forecast.method or "unknown",
    )
    provenance = Provenance(
        source=reference,
        events=(
            ProvenanceEvent(
                timestamp=issued.isoformat(),
                event="probabilistic_forecast_issued",
                detail=(
                    f"{forecast.method} at nominal {forecast.nominal_level:.0%}, "
                    f"band {forecast.band.value}"
                ),
            ),
        ),
        processing=(
            ProcessingStep(
                name="interval_construction",
                version="phase8",
                detail=(
                    f"calibration {forecast.calibration_status}; coverage error "
                    f"{forecast.coverage_error}"
                ),
            ),
        ),
    )
    single = PredictionInterval(
        lower_kw=np.asarray([forecast.lower_kw]),
        upper_kw=np.asarray([forecast.upper_kw]),
        nominal_level=forecast.nominal_level,
        method=forecast.method or "phase8",
        symmetric=bool(
            abs((forecast.upper_kw - forecast.point_kw) - (forecast.point_kw - forecast.lower_kw))
            < 1e-9
        ),
    )
    return Forecast(
        forecast_id=ForecastId(identifier),
        variable=variable,
        role=VariableRole.OBSERVATION,
        target_time=forecast.timestamp,
        issued_at=issued.isoformat(),
        horizon_steps=int(forecast.horizon_steps),
        value=Quantity(float(forecast.point_kw), Unit.KILOWATT),
        provenance=provenance,
        uncertainty=interval_to_estimate(
            interval=single, kind=forecast.uncertainty_kind
        ),
    )


def _band_from_width(
    widths: np.ndarray, cuts: tuple[float, float]
) -> np.ndarray:
    """Assign uncertainty bands from two cut points, as :class:`UncertaintyBand` values.

    Args:
        widths: ``[N]`` interval widths.
        cuts: ``(low_high_cut, high_cut)`` in ascending order, from the calibration split.

    Returns:
        ``[N]`` array of :class:`UncertaintyBand`.
    """
    values = np.asarray(widths, dtype=np.float64)
    low_cut, high_cut = (float(cuts[0]), float(cuts[1]))
    if low_cut >= high_cut:
        raise ValueError(
            f"band cut points must ascend, got {low_cut} and {high_cut}"
        )
    out = np.full(values.shape, UncertaintyBand.MEDIUM, dtype=object)
    out[values <= low_cut] = UncertaintyBand.LOW
    out[values > high_cut] = UncertaintyBand.HIGH
    return out


@dataclass(frozen=True, slots=True)
class ProbabilisticForecastBatch:
    """A rectangular block of probabilistic forecasts, one horizon column at a time.

    Arrays rather than a list of objects, because 179,520 rows x 3 horizons is 538,560
    objects and a batch of arrays is both smaller and easier to evaluate. Construction
    validates every invariant a per-row object would, so the invariants are not lost.

    Attributes:
        timestamps: ``[N]`` ISO-8601 target instants.
        asset_ids: ``[N]`` asset identifier values.
        horizon_steps: ``[horizons]``.
        point_kw: ``[N, horizons]``.
        lower_kw: ``[N, horizons]``.
        upper_kw: ``[N, horizons]``.
        nominal_level: Stated coverage, shared by the block.
        method: Interval procedure, one name per horizon. Per-horizon rather than one
            string for the block, because the phase selects the procedure per horizon from
            the measured evidence: publishing one name for all three would credit a horizon
            with a method that was not selected for it.
        model_version: Point-forecast model identifier.
        calibration_status: Per-horizon status, ``[horizons]``.
        coverage_error: Per-horizon signed coverage error.
        band_cuts: ``[horizons]`` pairs of width cut points from the calibration split, one
            pair per horizon. Per-horizon rather than one pair for the block, because mean
            widths differ by 5x across h=1 and h=96: a single pair would call nearly every
            short-horizon forecast "low" and nearly every long-horizon one "high", which
            says nothing about the row in front of an operator.
        data_quality: ``[N]`` tuples of quality flag values.
    """

    timestamps: tuple[str, ...]
    asset_ids: tuple[str, ...]
    horizon_steps: tuple[int, ...]
    point_kw: np.ndarray
    lower_kw: np.ndarray
    upper_kw: np.ndarray
    nominal_level: float
    method: tuple[str, ...]
    model_version: str
    calibration_status: tuple[str, ...] = ()
    coverage_error: tuple[float | None, ...] = ()
    band_cuts: tuple[tuple[float, float], ...] = ()
    data_quality: tuple[tuple[str, ...], ...] = ()
    notes: str = ""

    def __post_init__(self) -> None:
        point = np.asarray(self.point_kw, dtype=np.float64)
        lower = np.asarray(self.lower_kw, dtype=np.float64)
        upper = np.asarray(self.upper_kw, dtype=np.float64)
        n_horizons = len(self.horizon_steps)
        expected = (len(self.timestamps), n_horizons)
        if point.shape != expected or lower.shape != expected or upper.shape != expected:
            raise ValueError(
                f"point/lower/upper must all be [rows, horizons] = {expected}; got "
                f"{point.shape}, {lower.shape}, {upper.shape}"
            )
        if len(self.asset_ids) != len(self.timestamps):
            raise ValueError(
                f"{len(self.asset_ids)} asset ids for {len(self.timestamps)} timestamps"
            )
        if len(self.method) != n_horizons:
            raise ValueError(
                f"method must name one interval procedure per horizon ({n_horizons}), got "
                f"{self.method}; a single name for all horizons would credit a horizon "
                f"with a method that was not selected for it"
            )
        if any(not isinstance(name, str) or not name.strip() for name in self.method):
            raise ValueError(f"every method name must be a non-empty string, got {self.method}")
        if len(self.calibration_status) not in (0, n_horizons):
            raise ValueError(
                f"calibration_status must be empty or have one entry per horizon "
                f"({n_horizons}), got {len(self.calibration_status)}"
            )
        if len(self.coverage_error) not in (0, n_horizons):
            raise ValueError(
                f"coverage_error must be empty or have one entry per horizon "
                f"({n_horizons}), got {len(self.coverage_error)}"
            )
        for name, array in (("lower_kw", lower), ("upper_kw", upper), ("point_kw", point)):
            if not np.isfinite(array).all():
                raise ValueError(f"{name} must be finite")
        offenders = int(np.count_nonzero(lower > upper))
        if offenders:
            raise ValueError(
                f"{offenders} of {lower.size} intervals have lower_kw > upper_kw"
            )
        if not 0.0 < float(self.nominal_level) < 1.0:
            raise ValueError(
                f"nominal_level must lie in (0, 1), got {self.nominal_level}"
            )
        if len(self.band_cuts) != n_horizons:
            raise ValueError(
                f"band_cuts must hold one ascending (low, high) pair per horizon "
                f"({n_horizons}), got {self.band_cuts}"
            )
        for column, cuts in enumerate(self.band_cuts):
            if len(cuts) != 2 or cuts[0] >= cuts[1]:
                raise ValueError(
                    f"band_cuts[{column}] must be two ascending numbers, got {cuts}"
                )

    @property
    def n_rows(self) -> int:
        return len(self.timestamps)

    def width_kw(self, horizon_index: int) -> np.ndarray:
        """Interval width for one horizon column."""
        column = int(horizon_index)
        if not 0 <= column < len(self.horizon_steps):
            raise IndexError(
                f"horizon_index {horizon_index} out of range for "
                f"{len(self.horizon_steps)} horizons"
            )
        return np.asarray(self.upper_kw, dtype=np.float64)[:, column] - np.asarray(
            self.lower_kw, dtype=np.float64
        )[:, column]

    def bands(self, horizon_index: int) -> np.ndarray:
        """Uncertainty bands for one horizon column.

        Args:
            horizon_index: Which horizon column.

        Returns:
            ``[N]`` array of :class:`UncertaintyBand`.

        Raises:
            IndexError: If the column is out of range.
            ValueError: If the block holds no cut point for this horizon.
        """
        column = int(horizon_index)
        if not 0 <= column < len(self.horizon_steps):
            raise IndexError(
                f"horizon_index {horizon_index} out of range for "
                f"{len(self.horizon_steps)} horizons"
            )
        if len(self.band_cuts) != len(self.horizon_steps):
            raise ValueError(
                f"band_cuts holds {len(self.band_cuts)} pairs for "
                f"{len(self.horizon_steps)} horizons; the bands cannot be assigned"
            )
        return _band_from_width(self.width_kw(column), self.band_cuts[column])

    def band_summary(self, horizon_index: int) -> dict[str, int]:
        """Row counts per band, for one horizon."""
        bands = self.bands(horizon_index)
        return {
            band.value: int(np.count_nonzero(bands == band))
            for band in UncertaintyBand
        }

    def row(
        self, row: int, horizon_index: int
    ) -> ProbabilisticForecast:
        """One :class:`ProbabilisticForecast`, for a consumer that wants objects."""
        column = int(horizon_index)
        status = (
            self.calibration_status[column]
            if self.calibration_status
            else CALIBRATION_UNKNOWN
        )
        error = self.coverage_error[column] if self.coverage_error else None
        quality = (
            self.data_quality[row] if self.data_quality else ("ok",)
        )
        bands = self.bands(column)
        return ProbabilisticForecast(
            timestamp=self.timestamps[int(row)],
            asset_id=self.asset_ids[int(row)],
            horizon_steps=int(self.horizon_steps[column]),
            point_kw=float(np.asarray(self.point_kw)[row, column]),
            lower_kw=float(np.asarray(self.lower_kw)[row, column]),
            upper_kw=float(np.asarray(self.upper_kw)[row, column]),
            nominal_level=float(self.nominal_level),
            calibration_status=status,
            coverage_error=error,
            band=UncertaintyBand(bands[row]),
            method=self.method[column],
            model_version=self.model_version,
            data_quality=tuple(quality),
            uncertainty_notes=self.notes,
        )

    def rows(self, row_positions: np.ndarray) -> tuple[ProbabilisticForecast, ...]:
        """A sample of per-row objects, for a readable artifact."""
        return tuple(
            self.row(int(row), column)
            for column in range(len(self.horizon_steps))
            for row in np.asarray(row_positions, dtype=np.int64)
        )

    def to_dict(self) -> dict[str, Any]:
        """A JSON-friendly summary. The full arrays live in the ``.npz`` beside it."""
        return {
            "n_rows": self.n_rows,
            "horizons": [int(h) for h in self.horizon_steps],
            "nominal_level": float(self.nominal_level),
            "method": list(self.method),
            "model_version": self.model_version,
            "calibration_status": list(self.calibration_status),
            "coverage_error": [
                None if value is None else float(value) for value in self.coverage_error
            ],
            "band_cuts_kw": {
                str(h): [float(cuts[0]), float(cuts[1])]
                for h, cuts in zip(self.horizon_steps, self.band_cuts)
            },

            "mean_point_kw": {
                str(h): float(np.asarray(self.point_kw)[:, column].mean())
                for column, h in enumerate(self.horizon_steps)
            },
            "mean_width_kw": {
                str(h): float(self.width_kw(column).mean())
                for column, h in enumerate(self.horizon_steps)
            },
            "band_counts": {
                str(h): self.band_summary(column)
                for column, h in enumerate(self.horizon_steps)
            },
            "notes": self.notes,
        }


def save_batch(batch: ProbabilisticForecastBatch, path: Path) -> Path:
    """Write the batch so it can be read back exactly.

    Every field the object validates is written, not just the bounds. An artifact that
    stores the numbers and drops the calibration status, the band cut points or the method
    name would force a reader to reconstruct the labels from a threshold they do not have,
    and the reconstruction would be theirs rather than the experiment's.

    Args:
        batch: The batch.
        path: Destination ``.npz``. The summary is written beside it as ``.json``.

    Returns:
        The ``.npz`` path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        point_kw=np.asarray(batch.point_kw),
        lower_kw=np.asarray(batch.lower_kw),
        upper_kw=np.asarray(batch.upper_kw),
        timestamps=np.asarray(batch.timestamps),
        asset_ids=np.asarray(batch.asset_ids),
        horizon_steps=np.asarray(batch.horizon_steps),
        method=np.asarray(batch.method),
        model_version=np.asarray(batch.model_version),
        nominal_level=np.asarray(float(batch.nominal_level)),
        calibration_status=np.asarray(batch.calibration_status),
        coverage_error=np.asarray(
            [
                np.nan if value is None else float(value)
                for value in batch.coverage_error
            ]
        ),
        band_cuts=np.asarray(batch.band_cuts, dtype=np.float64),
        data_quality=np.asarray(
            ["|".join(flags) for flags in batch.data_quality] or []
        ),
        notes=np.asarray(batch.notes),
    )
    path.with_suffix(".json").write_text(
        json.dumps(batch.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
    )
    return path


def load_batch(path: Path) -> ProbabilisticForecastBatch:
    """Read a batch written by :func:`save_batch`.

    Args:
        path: The ``.npz`` path.

    Returns:
        The batch, reconstructed.

    Raises:
        FileNotFoundError: If the file is absent.
        ValueError: If the archive is missing a field the object requires.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"probabilistic forecast batch not found: {path}")
    with np.load(path, allow_pickle=False) as archive:
        required = (
            "point_kw", "lower_kw", "upper_kw", "timestamps", "asset_ids",
            "horizon_steps", "method", "nominal_level",
        )
        missing = [name for name in required if name not in archive]
        if missing:
            raise ValueError(
                f"{path} is missing {missing}; it was not written by save_batch"
            )
        coverage = (
            archive["coverage_error"]
            if "coverage_error" in archive
            else np.zeros(0)
        )
        quality = archive["data_quality"] if "data_quality" in archive else np.zeros(0, dtype="U1")
        return ProbabilisticForecastBatch(
            timestamps=tuple(str(value) for value in archive["timestamps"]),
            asset_ids=tuple(str(value) for value in archive["asset_ids"]),
            horizon_steps=tuple(int(h) for h in archive["horizon_steps"]),
            point_kw=archive["point_kw"],
            lower_kw=archive["lower_kw"],
            upper_kw=archive["upper_kw"],
            nominal_level=float(archive["nominal_level"]),
            method=tuple(str(value) for value in archive["method"]),
            model_version=(
                str(archive["model_version"])
                if "model_version" in archive
                else ""
            ),
            calibration_status=(
                tuple(str(value) for value in archive["calibration_status"])
                if "calibration_status" in archive
                else ()
            ),
            coverage_error=(
                tuple(
                    None if not np.isfinite(value) else float(value)
                    for value in coverage
                )
                if coverage.size
                else ()
            ),
            band_cuts=(
                tuple(
                    (float(row[0]), float(row[1])) for row in archive["band_cuts"]
                )
                if "band_cuts" in archive
                else ()
            ),
            data_quality=(
                tuple(tuple(str(value).split("|")) for value in quality)
                if quality.size
                else ()
            ),
            notes=(
                str(archive["notes"]) if "notes" in archive and archive["notes"].size else ""
            ),
        )


@dataclass(frozen=True, slots=True)
class BandCutPoints:
    """The two width thresholds that separate LOW, MEDIUM and HIGH.

    Derived from the **calibration** split's width distribution, not the test split, so the
    bands mean the same thing at evaluation time as they would in deployment. A band
    threshold computed on test rows would be a test-set statistic leaking into an
    operational label.

    Attributes:
        low_high_kw: Width at or below which a forecast is LOW.
        high_kw: Width above which a forecast is HIGH.
        nominal_level: The level the widths were computed at.
        source_rows: How many calibration rows produced the cut points.
    """

    low_high_kw: float
    high_kw: float
    nominal_level: float
    source_rows: int

    @classmethod
    def from_widths(
        cls, widths: np.ndarray, *, nominal_level: float
    ) -> "BandCutPoints":
        """Terciles of a width distribution.

        Args:
            widths: ``[N]`` calibration widths.
            nominal_level: The level they were computed at.

        Returns:
            The cut points.

        Raises:
            ValueError: If fewer than three widths are supplied, or they are not finite.
        """
        values = np.asarray(widths, dtype=np.float64).ravel()
        if values.size < 3:
            raise ValueError(
                f"at least three widths are needed for terciles, got {values.size}"
            )
        if not np.isfinite(values).all():
            raise ValueError("band widths must be finite")
        low, high = (float(v) for v in np.quantile(values, [1.0 / 3.0, 2.0 / 3.0]))
        return cls(
            low_high_kw=low,
            high_kw=high,
            nominal_level=float(nominal_level),
            source_rows=int(values.size),
        )

    def as_tuple(self) -> tuple[float, float]:
        return (float(self.low_high_kw), float(self.high_kw))

    def to_dict(self) -> dict[str, Any]:
        return {
            "low_high_kw": float(self.low_high_kw),
            "high_kw": float(self.high_kw),
            "nominal_level": float(self.nominal_level),
            "source_rows": int(self.source_rows),
            "definition": (
                "terciles of the calibration-split width distribution at this nominal "
                "level; LOW is at or below the first, HIGH above the second"
            ),
        }
