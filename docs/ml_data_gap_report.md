# ML Data Gap Report

What the eventual system would ideally need from a dataset, what SMART-DS actually
provides, and what that costs.

Every row was established by inspecting the acquired files or by measuring the
baseline experiments, not by inference. Where a gap is *fixable by better
ingestion* it says so; where it needs a different dataset it says that too. "No"
alone is not a useful answer, so every gap carries a consequence.

Phase 3's companion document is `gaps_report.md`, which covers what the *domain*
needs. This one covers what the *machine-learning* layer needs.

---

## Summary

| # | Requirement | Available? | Fixable by ingestion? | Consequence if unresolved |
|---|---|---|---|---|
| M-01 | Per-asset demand history | **Yes** | — | none |
| M-02 | Per-asset demand at 15-minute resolution | **Yes** | — | none |
| M-03 | Feeder-level demand history | **Yes** | — | none |
| M-04 | PV generation for the feeder's own arrays | **No** | **No** - irradiance shapes absent | feeder PV cannot be forecast; net load impossible |
| M-05 | PV generation for one real array | **Yes** | — | renewable forecasting possible, but as a proxy |
| M-06 | Weather **forecast** (not observation) | **No** | **No** | PV beyond 15 minutes is unforecastable |
| M-07 | Battery SOC or dispatch | **No** | **No** | no battery flexibility baseline |
| M-08 | EV charging demand | **No** | **No** | no EV flexibility baseline |
| M-09 | Wind generation | **No** | **No** | no wind forecasting track |
| M-10 | Multi-year history for seasonality | **Partial** - one year | **No** | annual models cannot be validated across years |
| M-11 | Explicit holidays | **No** | **No** | holiday effects leak into the weekly pattern |
| M-12 | Voltage / current / reactive power per node | **No** as time series | **No** | no state estimation, no voltage forecasting |
| M-13 | Asset-level outages or derates | **No** | **No** | availability must be assumed, not learned |
| M-14 | End-use breakdown (HVAC, lighting, motors) | **Not acquired** | **Yes** - `load_data/*.parquet` exists | load models cannot decompose by end use |
| M-15 | Labels of any kind | **No** | **No** | no supervised task other than forecasting |

---

## Demand

### M-01 / M-02 / M-03 - Per-asset, 15-minute, whole-year demand

**Available: yes, completely.** 1,871 customers resolved from 3,690 OpenDSS `Load`
objects, each with a 35,040-point per-unit profile; 341 distinct profile files; the
feeder total by summation. No missing intervals, no NaN, no negative values, and no
profile that fails to resolve.

**Transformation:** `P(t) = Σ kW_rating × profile_pu(t)`, centre-tap members summed
(verified empirically against the dataset's own published peak, D-043).

**Value origin:** `DERIVED`. This matters for reporting: it is not telemetry.

**Impact:** none on forecasting. Demand forecasting is the one task this dataset
supports fully, and it is the strongest result in Phase 4 (MAE 0.4043 kW at 15
minutes, R2 0.998 on a test span containing the dataset's own peak).

**Future requirement:** none for a baseline. For a day-ahead *system* forecast, the
same data at a **wider** granularity - substation or feeder, which exists here -
is the right target, and multi-year history would be needed to fit annual patterns.

---

## Renewables

### M-04 - PV generation for the feeder's own 1,216 arrays: **NOT AVAILABLE**

**Available: no.** Every `PVSystem` in the PV scenarios references an irradiance
loadshape named `AUS_30.4196_-97.8095_<tilt>_<azimuth>` - 20 distinct combinations.
None of those shapes exists in `LoadShapes.dss` (which holds 341 load profiles and no
solar shapes), and no file matching that naming exists anywhere in the acquired
subset. The one file that *is* present, `solar_data/AUS_30.3459_-97.8095_25_180_full.csv`,
is a **different location** (30.3459 N, not 30.4196 N).

**Fixable by ingestion: no.** This is not a parsing or linking mistake; the referenced
data is not in the dataset. This closes G-03 from Phase 3, which had recorded the
link as unresolved. It is now known to be unresolvable from this source.

**Impact:** the feeder's PV output cannot be reconstructed, so `net_load` for this
feeder cannot be formed without fabricating it. That is why `net_load` is recorded as
`UNSUPPORTED` rather than approximated.

### M-05 - PV generation for one real array: **AVAILABLE**

**Available: yes.** `kW Generated (1000 kW Array)`, 35,040 points, 0-834.05 kW, mean
159.73 kW, with 52.2% exact zeros that are physically night-time values.

**Transformation:** read directly; the column already reports a 1000 kW array.

**Value origin:** `OBSERVED` - a published column, not a derivation.

**Impact:** renewable forecasting is possible, but as a **proxy**: one location, one
orientation, not this feeder's fleet. Reported honestly as `PARTIALLY_SUPPORTED`.

**Future requirement:** the 20 missing irradiance files, if the source can supply
them; otherwise a second PV dataset.

### M-06 - Weather **forecast**: **NOT AVAILABLE**

**Available: no.** The dataset ships *actual* observations - DNI, DHI, GHI, PoA
irradiance, temperature, wind speed - for one location. There is no forecast product,
no ensemble, and no second location.

**This is the single most consequential gap in the phase**, and it is measured rather
than asserted. With weather allowed only where it is genuinely knowable (the origin,
plus one step), PV at 15 minutes reaches MAE 8.57 kW on a 1000 kW array - 0.86% of
capacity. Without weather at all, at 24 hours, the best model reaches MAE 74.3 kW
against persistence's 79.6 kW and an R2 of 0.62: **barely better than doing nothing.**

**Impact:** day-ahead PV forecasting on this dataset is not a modelling problem, it
is a missing-input problem. No architecture will fix it. Phase 4 enforces this
structurally: weather features are withdrawn automatically beyond a 15-minute horizon
(D-063), and the withdrawal reason is recorded in every dataset manifest.

**Future requirement:** a weather forecast source, or an explicitly simulated one
marked `SIMULATED`. This is a Phase 5 decision, not a Phase 4 one.

---

## Storage, EVs, wind

### M-07 - Battery SOC or dispatch: **NOT AVAILABLE**

Unchanged from Phase 3's G-01 and now measured against a model: 93 batteries with
ratings and one `kWhStored` scalar, no SOC series, no dispatch, no depth of discharge,
no directional power limit. Nothing was derived.

**Impact:** no battery flexibility baseline exists, so Phase 9 and Phase 10 have no
empirical starting point for storage. **Future requirement:** a second dataset with
observed dispatch, or an OpenDSS storage-control run marked `SIMULATED` (D-046).

### M-08 - EV charging demand: **NOT AVAILABLE**

2,991 adoption sites are named in the placement JSON; no charging profile, power or
session series exists.

**Impact:** no EV flexibility baseline. **Future requirement:** a dataset with EV
sessions. The placement data is still worth keeping - it constrains *where* chargers
could be, even though it says nothing about *when* they load.

### M-09 - Wind generation: **NOT AVAILABLE**

SMART-DS v1.0 contains no wind asset. Wind speed appears only as a weather covariate
in the solar file, with no turbine to convert it.

**Impact:** either drop wind forecasting from scope, or source a second dataset.
**This should be an explicit scope decision before Phase 4 is called complete**; it
is recorded here rather than assumed away.

---

## Temporal and calendar

### M-10 - Multi-year history: **PARTIAL**

**Available: one year** (2018, 365 days, 35,040 steps) for the AUS/P1U sub-region.

**Impact, measured:** the split is chronological within that single year, so the
test span is a different *season* from much of the training span. Results therefore
measure generalisation across time within a year, not across years. The 24-hour
load result is the most affected: a seasonal-climatology model cannot be validated
on a year it has never seen.

**Future requirement:** at least 3-5 years, so annual models can be validated and
weather-conditioned patterns can be separated from calendar artefacts.

### M-11 - Explicit holidays: **NOT AVAILABLE**

No holiday calendar is shipped. Day-of-week features absorb holidays into the weekly
pattern, so a model can learn "Sundays are quiet" but cannot distinguish a public
holiday from an ordinary weekend day.

**Impact:** error will concentrate on holiday periods in the reported test span.
**Future requirement:** a public holiday calendar for the region, as a legitimate
external input (not fabricated). This is a small, cheap, high-value addition.

---

## Network state

### M-12 - Voltage, current, reactive power per node: **NOT AVAILABLE as time series**

`Loads.dss` does carry `kvar` and a `Phases` count, and reactive profiles exist
alongside the active ones, but **no per-node voltage or current time series is
published** for this feeder. Feeder-level metrics exist in `metrics.csv` as static
planning quantities.

**Impact:** no state estimation, no voltage/VAR forecasting, no constraint-aware
dispatch validation. Phase 11 (digital twin) will have to *produce* electrical state
rather than read it, exactly as per-node power balance must be produced rather than
read (G-02).

**Future requirement:** a power-flow solver run on the same feeder, or a dataset with
SCADA measurements.

### M-13 - Outages, derates, availability: **NOT AVAILABLE**

Every asset is modelled as in service; `Storage.State=IDLING` is a static modelling
state, not an availability signal.

**Impact:** `availability = 1.0` is used for every asset and is recorded in the
dataset manifest as a **modelling fact, not a measurement**. No model can learn
degradation or outage behaviour from this data.

---

## What the dataset does provide that the requirements list did not ask for

Two things worth noting, because they are easy to overlook:

1. **End-use detail exists but was not acquired.** `load_data/*.parquet` is described
   as carrying 34 columns of end-use breakdown (heating, cooling, lighting, motors,
   appliances). It is not in the acquired subset. That is the highest-value *cheap*
   acquisition available: it would let a load model decompose demand by end use,
   which is what makes HVAC-driven flexibility tractable later.
2. **Feeder topology and geometry are rich.** 4,555 line elements, 5,196 nodes,
   coordinates for 10 buses per excerpt, plus 60+ static planning metrics per feeder
   in `metrics.csv`. Good enough for graph-based and constraint-aware work; not a
   substitute for measured electrical state.

---

## What this means for the architecture

The honest conclusion from Phase 4 is narrow and specific:

> This dataset can support **demand** forecasting, well. It can support renewable
> forecasting only as a **proxy with actual weather at the origin**, not as
> day-ahead forecasting of this feeder's own assets. It cannot support storage, EV,
> wind, state estimation or flexibility work at all.

So the Energy World Model of Phase 5, if built on this data, will be a **demand**
world model. Anything that claims to model the feeder's PV, its batteries or its
voltage on this dataset alone is claiming something the data cannot support. That
constraint is a finding, and it should shape the next dataset decision rather than be
discovered again in Phase 9.