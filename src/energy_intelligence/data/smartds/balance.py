"""Energy balance validation against real data.

The Phase 2 brief set a target of **0.5 kW per node**. This module reports
honestly against whatever the data can actually support, and never adjusts the
tolerance to make a result pass.

What the data supports, and what it does not
---------------------------------------------
Determined by inspecting the real dataset, not assumed:

* **Available:** customer demand (derivable for every grid index), aggregate
  circuit power, aggregate real losses -- but the latter two only at the single
  published peak timepoint.
* **Not available:** grid import/export as a time series, per-node flows,
  per-node losses, battery dispatch, PV output timeseries.

Consequently a per-node balance over time **cannot be evaluated**: there is no
grid flow series to close against. What *can* be evaluated is the aggregate
feeder identity at the published peak timepoint:

    circuit_power ?= customer_power + losses

and the dataset's own internal consistency:

    reconstructed_customer_power ?= published_customer_power

Both are reported, with residuals, and both are allowed to fail.

The residual at the feeder head is expected to be **non-zero and large**,
because a distribution feeder genuinely loses power: SMART-DS itself publishes
501.67 kW of losses at 15,090.72 kW of customer power (3.22%). Phase 2 deferred
network physics (D-023), so those losses are, in domain terms, an
``unaccounted_kw`` residual. That is a physics gap, not a data defect, and it is
reported as such.
"""

from __future__ import annotations

import csv
import statistics
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "TOLERANCE_KW",
    "PublishedSummary",
    "BalanceSample",
    "BalanceReport",
    "read_summary_data",
    "validate_feeder_balance",
]

#: Target per-node tolerance established in Phase 2. NOT adjustable here.
TOLERANCE_KW = 0.5


@dataclass(frozen=True, slots=True)
class PublishedSummary:
    """One row of ``analysis/Summary_data.csv``.

    Attributes:
        losses_kw: Published real losses at the peak timepoint.
        circuit_power_kw: Published total circuit real power.
        losses_percent: Published loss percentage.
        customer_power_kw: Published total customer real power.
        commercial_kw: Published commercial real power.
        residential_kw: Published residential real power.
        peak_day: Peak day of year (1-based).
        peak_hour: Peak hour of day.
        peak_minute: Peak minute of hour.
    """

    losses_kw: float
    circuit_power_kw: float
    losses_percent: float
    customer_power_kw: float
    commercial_kw: float
    residential_kw: float
    peak_day: int
    peak_hour: int
    peak_minute: int

    @property
    def peak_index(self) -> int:
        """Zero-based grid index of the peak timepoint on a 15-minute grid."""
        return (self.peak_day - 1) * 96 + self.peak_hour * 4 + self.peak_minute // 15

    @property
    def published_split_consistent(self) -> bool:
        """Whether commercial + residential equals the published customer total."""
        return abs(self.commercial_kw + self.residential_kw - self.customer_power_kw) < 1e-6

    def to_dict(self) -> dict[str, object]:
        return {
            "losses_kw": self.losses_kw,
            "circuit_power_kw": self.circuit_power_kw,
            "losses_percent": self.losses_percent,
            "customer_power_kw": self.customer_power_kw,
            "commercial_kw": self.commercial_kw,
            "residential_kw": self.residential_kw,
            "peak_day": self.peak_day,
            "peak_hour": self.peak_hour,
            "peak_minute": self.peak_minute,
            "peak_index": self.peak_index,
        }


def read_summary_data(path: Path) -> PublishedSummary:
    """Parse ``analysis/Summary_data.csv``.

    Raises:
        KeyError: If a required column is absent, rather than defaulting it.
    """
    with path.open(newline="", encoding="utf-8-sig") as handle:
        row = next(csv.DictReader(handle))
    return PublishedSummary(
        losses_kw=float(row["Total peak time real losses (kW)"]),
        circuit_power_kw=float(row["Total peak time circuit power (kW)"]),
        losses_percent=float(row["Percentage peak time real losses"]),
        customer_power_kw=float(row["Total peak time real customer power (kW)"]),
        commercial_kw=float(row["Total peak time real commercial power (kW)"]),
        residential_kw=float(row["Total peak time real residential power (kW)"]),
        peak_day=int(row["Peak load day of year"]),
        peak_hour=int(row["Peak load hour of day"]),
        peak_minute=int(row["Peak load minute of hour"]),
    )


@dataclass(frozen=True, slots=True)
class BalanceSample:
    """One evaluated balance identity.

    Attributes:
        label: Identity name.
        computed_kw: Value reconstructed from source data.
        published_kw: Value the dataset itself publishes.
        residual_kw: ``computed - published``.
        scope: What the identity covers.
        note: Why it holds or fails, where determinable.
    """

    label: str
    computed_kw: float
    published_kw: float
    scope: str
    note: str = ""

    @property
    def residual_kw(self) -> float:
        return self.computed_kw - self.published_kw

    @property
    def absolute_residual_kw(self) -> float:
        return abs(self.residual_kw)

    @property
    def within_tolerance(self) -> bool:
        return self.absolute_residual_kw <= TOLERANCE_KW

    def to_dict(self) -> dict[str, object]:
        return {
            "label": self.label,
            "computed_kw": self.computed_kw,
            "published_kw": self.published_kw,
            "residual_kw": self.residual_kw,
            "absolute_residual_kw": self.absolute_residual_kw,
            "within_tolerance": self.within_tolerance,
            "scope": self.scope,
            "note": self.note,
        }


@dataclass(frozen=True, slots=True)
class BalanceReport:
    """Machine-readable balance validation outcome.

    Metrics that could not be computed are ``None``, never zero and never a
    placeholder.
    """

    tolerance_kw: float
    node_count: int
    timestamp_count: int
    nodes_with_balance_data: int
    nodes_without_balance_data: int
    total_observations: int
    samples: tuple[BalanceSample, ...] = ()
    max_absolute_residual_kw: float | None = None
    mean_absolute_residual_kw: float | None = None
    median_absolute_residual_kw: float | None = None
    p95_absolute_residual_kw: float | None = None
    above_tolerance_count: int = 0
    worst_nodes: tuple[str, ...] = ()
    worst_timestamps: tuple[str, ...] = ()
    unavailable: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    @property
    def passed(self) -> bool:
        """True only when every evaluated identity met the tolerance."""
        return bool(self.samples) and all(
            sample.within_tolerance for sample in self.samples
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "tolerance_kw": self.tolerance_kw,
            "passed": self.passed,
            "node_count": self.node_count,
            "timestamp_count": self.timestamp_count,
            "nodes_with_balance_data": self.nodes_with_balance_data,
            "nodes_without_balance_data": self.nodes_without_balance_data,
            "total_observations": self.total_observations,
            "max_absolute_residual_kw": self.max_absolute_residual_kw,
            "mean_absolute_residual_kw": self.mean_absolute_residual_kw,
            "median_absolute_residual_kw": self.median_absolute_residual_kw,
            "p95_absolute_residual_kw": self.p95_absolute_residual_kw,
            "above_tolerance_count": self.above_tolerance_count,
            "above_tolerance_percent": (
                100.0 * self.above_tolerance_count / len(self.samples)
                if self.samples
                else None
            ),
            "worst_nodes": list(self.worst_nodes),
            "worst_timestamps": list(self.worst_timestamps),
            "samples": [sample.to_dict() for sample in self.samples],
            "unavailable": list(self.unavailable),
            "notes": list(self.notes),
        }

    def render(self) -> str:
        """Human-readable summary."""
        lines = [
            "SMART-DS energy balance validation",
            "=" * 62,
            f"tolerance                  : {self.tolerance_kw} kW (fixed, not adjustable)",
            f"nodes in graph             : {self.node_count}",
            f"nodes with balance data    : {self.nodes_with_balance_data}",
            f"nodes without balance data : {self.nodes_without_balance_data}",
            f"timestamps evaluated       : {self.timestamp_count}",
            f"total observations         : {self.total_observations}",
            f"result                     : {'PASS' if self.passed else 'FAIL'}",
            "",
            "residuals:",
            f"  max    : {_fmt(self.max_absolute_residual_kw)}",
            f"  mean   : {_fmt(self.mean_absolute_residual_kw)}",
            f"  median : {_fmt(self.median_absolute_residual_kw)}",
            f"  p95    : {_fmt(self.p95_absolute_residual_kw)}",
            f"  above tolerance: {self.above_tolerance_count} of {len(self.samples)}",
        ]
        if self.samples:
            lines.append("")
            lines.append("identities:")
            for sample in self.samples:
                mark = "PASS" if sample.within_tolerance else "FAIL"
                lines.append(f"  [{mark}] {sample.label}")
                lines.append(
                    f"         computed={sample.computed_kw:.4f} kW  "
                    f"published={sample.published_kw:.4f} kW  "
                    f"residual={sample.residual_kw:+.4f} kW"
                )
                if sample.note:
                    lines.append(f"         {sample.note}")
        if self.unavailable:
            lines.append("")
            lines.append("NOT EVALUATED (data absent, not defaulted):")
            for item in self.unavailable:
                lines.append(f"  - {item}")
        if self.notes:
            lines.append("")
            lines.append("notes:")
            for note in self.notes:
                lines.append(f"  {note}")
        return "\n".join(lines)


def _fmt(value: float | None) -> str:
    return "unknown" if value is None else f"{value:.6f} kW"


def validate_feeder_balance(
    *,
    summary: PublishedSummary,
    reconstructed_customer_kw: float,
    reconstructed_commercial_kw: float | None = None,
    reconstructed_residential_kw: float | None = None,
    node_count: int = 0,
    timestamp_count: int = 1,
) -> BalanceReport:
    """Evaluate the balance identities that real data can support.

    Two distinct identities are evaluated, and they must not be conflated:

    1. **Reconstruction fidelity** -- does our derivation reproduce the value the
       dataset publishes? A failure here means our *ingestion* is wrong.
    2. **Physical closure** -- does circuit power equal customer power plus
       losses? A failure here is expected to be explainable by unmodelled
       distribution losses, not by an ingestion error.

    Args:
        summary: Parsed ``Summary_data.csv`` for the feeder.
        reconstructed_customer_kw: Customer demand we computed from profiles.
        reconstructed_commercial_kw: Our commercial split, if computed.
        reconstructed_residential_kw: Our residential split, if computed.
        node_count: Nodes discovered in the graph.
        timestamp_count: Timepoints actually evaluated.

    Returns:
        A report whose ``unavailable`` list names what could not be evaluated.
    """
    samples: list[BalanceSample] = []

    samples.append(
        BalanceSample(
            label="reconstruction: total customer demand",
            computed_kw=reconstructed_customer_kw,
            published_kw=summary.customer_power_kw,
            scope="feeder aggregate at the published peak timepoint",
            note=(
                "Tests the SMART-DS derivation rule kW(t) = rating x profile_pu(t), "
                "including the centre-tap pair summation."
            ),
        )
    )

    if reconstructed_commercial_kw is not None:
        samples.append(
            BalanceSample(
                label="reconstruction: commercial demand",
                computed_kw=reconstructed_commercial_kw,
                published_kw=summary.commercial_kw,
                scope="feeder aggregate, commercial class only",
            )
        )
    if reconstructed_residential_kw is not None:
        samples.append(
            BalanceSample(
                label="reconstruction: residential demand",
                computed_kw=reconstructed_residential_kw,
                published_kw=summary.residential_kw,
                scope="feeder aggregate, residential class only",
            )
        )

    samples.append(
        BalanceSample(
            label="published internal: losses + customer = circuit",
            computed_kw=summary.losses_kw + summary.customer_power_kw,
            published_kw=summary.circuit_power_kw,
            scope="feeder aggregate, dataset's own figures",
            note=(
                "This checks SMART-DS's internal consistency, not our ingestion."
            ),
        )
    )

    samples.append(
        BalanceSample(
            label="published internal: commercial + residential = customer",
            computed_kw=summary.commercial_kw + summary.residential_kw,
            published_kw=summary.customer_power_kw,
            scope="feeder aggregate, dataset's own figures",
        )
    )

    residuals = [sample.absolute_residual_kw for sample in samples]
    above = sum(1 for sample in samples if not sample.within_tolerance)

    unavailable = (
        "per-node power balance over time: grid import/export is not provided as a "
        "time series, so no per-node closure is computable",
        "PV output time series: PVSystems.dss gives Pmpp and an irradiance loadshape "
        "reference, but the irradiance shape is defined outside the feeder folder, so "
        "no generation series was derived",
        "battery dispatch: no SOC series exists, so no power derivative is possible",
        "EV charging demand: placements give locations only, with no load series",
        "wind generation: the dataset contains no wind assets",
        "per-node losses: only a single feeder-level aggregate is published",
    )

    notes = (
        f"Feeders have real losses: SMART-DS publishes {summary.losses_kw:.3f} kW "
        f"({summary.losses_percent:.4f}%) at {summary.customer_power_kw:.3f} kW of "
        "customer power. Phase 2 deferred network physics (D-023), so in domain "
        "terms this is an unaccounted_kw residual at the feeder head, not an "
        "ingestion error.",
        "The 0.5 kW tolerance was NOT adjusted. Residuals above it are reported as "
        "found.",
    )

    worst = tuple(
        sample.label
        for sample in sorted(samples, key=lambda s: -s.absolute_residual_kw)[:3]
    )

    return BalanceReport(
        tolerance_kw=TOLERANCE_KW,
        node_count=node_count,
        timestamp_count=timestamp_count,
        nodes_with_balance_data=0,
        nodes_without_balance_data=node_count,
        total_observations=node_count * timestamp_count,
        samples=tuple(samples),
        max_absolute_residual_kw=max(residuals),
        mean_absolute_residual_kw=statistics.fmean(residuals),
        median_absolute_residual_kw=statistics.median(residuals),
        p95_absolute_residual_kw=sorted(residuals)[max(0, int(0.95 * len(residuals)) - 1)],
        above_tolerance_count=above,
        worst_nodes=worst,
        worst_timestamps=(),
        unavailable=unavailable,
        notes=notes,
    )
