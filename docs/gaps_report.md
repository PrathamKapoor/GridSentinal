# SMART-DS Gaps Report

What SMART-DS **does not provide** that the project's future stages require.
Everything here was established by inspecting the real dataset, not inferred.

This is the honest negative half of Phase 3. Several items were expected to be
present and are not; one contradicts a previous project document.

---

## G-01 — Battery dispatch does not exist and cannot be derived

**Severity: critical for Phase 9 (flexibility) and Phase 10 (optimization).**

| | |
|---|---|
| **Required** | `storage.charge_power_kw`, `storage.discharge_power_kw`, `state_of_charge` series |
| **Available** | Ratings only: `kWRated` (8.0), `kWhRated` (16.0), `%EffCharge`/`%EffDischarge` (95/95), and **one** `kWhStored` scalar (8.0) |
| **Evidence** | 93/93 `Storage` objects inspected in `solar_high_batteries_low_timeseries`. Every one has `State=IDLING`. No storage loadshape, no SOC column, no dispatch column, no discharge curve anywhere in `Storage.dss` or `LoadShapes.dss` |

**This corrects a Phase 2 handoff statement.** That document said dispatch "may
need to be derived from state of charge". Verification shows **that route is
also unavailable**: deriving power requires SOC(t) and SOC(t+1), and only a
single scalar exists. There is nothing to difference.

**Action taken:** no battery dispatch was derived and no `StorageState` was
constructed. Deriving one would have meant inventing it.

**Also absent, and equally load-bearing:** the *usable* energy capacity (no depth
of discharge), the SOC window, and any directional charge/discharge power limit.
`kWhRated` is a **nameplate** rating. Phase 3 initially mapped it into
`Battery.usable_energy` and one `kWRated` into both `max_charge_power` and
`max_discharge_power`; both were inventions, and both were corrected in D-050.
The domain now carries `nameplate_energy` and leaves the rest explicitly
`None`, so the absence is visible in the data rather than hidden inside a
plausible number.

**Consequence:** the project's headline flexibility story cannot be built on
SMART-DS battery dispatch alone. Options for Phase 3 follow-up:

1. Run OpenDSS/CYME power flow with a storage control strategy — but the
   resulting dispatch is **simulated**, and the strategy would be ours, not the
   dataset's. Must be marked `SIMULATED`, never `OBSERVED`.
2. Use `kWhRated`/`kWRated` only for *capacity* and treat dispatch as a decision
   variable in Phase 10 (which is where it belongs anyway).
3. Add a second dataset that provides battery dispatch.

Recommended: **option 2 plus option 1 as an explicitly simulated augmentation.**

---

## G-02 — Grid import/export is not provided as a time series

**Severity: critical for per-node balance validation.**

| | |
|---|---|
| **Required** | `grid.import_kw` / `grid.export_kw` per node per timepoint |
| **Available** | A single aggregate `Total peak time circuit power (kW)` at the published peak timepoint |
| **Evidence** | The sub-region layout is `profiles/`, `solar_data/`, `load_data/`, `load_curves/`, `scenarios/`, `placements/`. None contains grid flow. `Summary_data.csv` is one row |

**Consequence:** the Phase 2 per-node power balance **cannot be evaluated over
time**, because there is nothing to close against. Phase 11's power-flow solver
would have to *produce* the flows rather than read them, which changes the
validation from a data check into a simulation check. This is reported as
`NOT EVALUATED` in the balance report rather than defaulted.

---

## G-03 — PV output is not directly provided

**Severity: moderate.**

| | |
|---|---|
| **Available** | `Pmpp` per array (1,216 arrays), plus a reference to an irradiance loadshape |
| **Missing** | The irradiance shape is defined in the **sub-region level** `LoadShapes.dss`, not the feeder folder |
| **Also missing** | No per-array or per-feeder generation timeseries |

`solar_data/*.csv` does contain a usable `kW Generated (1000 kW Array)` column
from PVWatts, so generation *is* derivable as `Pmpp × kW_generated/1000`.

**Not implemented in Phase 3** because the loadshape→file linkage could not be
resolved from the acquired subset: the sub-region `LoadShapes.dss` (239,586
bytes at `opendss/` level) was not downloaded.

**Action:** download the sub-region level `LoadShapes.dss` and resolve the
linkage. Marked `PARTIAL`, not `DIRECT`, until then.

---

## G-04 — EV charging demand does not exist

**Severity: high for Phase 9 (flexibility).**

| | |
|---|---|
| **Available** | Placement JSONs listing which customer loads have adopted an EV, per penetration level |
| **Missing** | Any charging power, energy, session or departure data |

The User Guide describes these purely as *charging locations*, and the
sub-region layout has no EV timeseries folder. A count of adopted sites is not a
charging profile.

**Action:** no `EVCharger` asset was created, because a charging profile cannot
be invented. The count is reported so a later phase knows the population exists.

---

## G-05 — No wind generation

**Severity: moderate.** SMART-DS contains no wind assets. No wind folder in the
sub-region layout; `metrics.csv` has PV and Battery columns only.

**Consequence:** `WindAsset` and `RenewableState` have no source in this dataset.
The Phase 2 domain correctly supports wind; this dataset does not exercise it.

---

## G-06 — Distribution losses are real and unmodelled

**Severity: high — this is why the 0.5 kW criterion fails.**

SMART-DS publishes, at the peak timepoint:

| Quantity | Value |
|---|---|
| Customer power | 15,090.7182 kW |
| Real losses | 501.6724 kW (**3.2175 %**) |
| Circuit power | 15,591.9560 kW |

A distribution feeder genuinely loses power. Phase 2 deferred network physics
(D-023), so those losses are, in domain terms, an `unaccounted_kw` residual at
the feeder head.

**The 0.5 kW per-node tolerance was NOT adjusted.** It cannot close at the
feeder head while losses are unmodelled, and that is a physics gap, not an
ingestion error.

**Consequence for Phase 11:** the digital twin will need a loss model before
per-node balance becomes a meaningful validation target.

---

## G-07 — Commercial demand reconstruction is 20.45 kW short

**Severity: moderate — an unexplained residual.**

At the published peak timepoint (day 204, 16:15, grid index 19553):

| Class | Reconstructed | Published | Residual |
|---|---|---|---|
| Residential | 12,380.5126 kW | 12,380.5126 kW | **0.0000 kW** |
| Commercial | 2,689.7550 kW | 2,710.2056 kW | **−20.4506 kW** (−0.75 %) |
| Total | 15,070.2676 kW | 15,090.7182 kW | −20.4506 kW (−0.136 %) |

Investigated and **not explained**:

* All 1,819 centre-tap pairs are complete (no orphaned `_1` or `_2`).
* All loads are `model=1`, `conn=wye` — no ZIP variance.
* The total residual equals the commercial residual exactly, so this is **not** a
  class-reclassification artifact.
* No single load or half-load explains it (closest candidate leaves 21.48 kW).

**Cause: UNKNOWN.** Recorded as a data-quality finding rather than explained
away. Residential matching to eight decimal places proves the derivation rule
is right, so the gap is specific to the commercial subset.

---

## G-08 — Availability is a modelling fact, not a measurement

`availability = 1.0` is used for every asset because SMART-DS models every
equipment item as in service. That is **not** a measurement: there is no
outage, derate or availability timeseries anywhere. `State=IDLING` on batteries
is a static control state, not an availability signal.

Recorded on every affected asset's `metadata.availability_basis`.

---

## G-09 — No timezone

SMART-DS carries no UTC offset, no local time, and no daylight-saving handling.
OpenDSS's yearly mode uses a continuous synthetic clock.

The Phase 2 domain requires timezone-aware timestamps, so the synthetic clock is
anchored to `01-01T00:00:00+00:00` and treated as UTC. This preserves every
**relative** time relationship exactly. It does **not** preserve true local
wall-clock time.

**Consequence:** any future claim about local solar noon, tariff windows or
peak-shifting in *local* time is only valid up to this fixed offset.

---

## G-10 — Reactive power has no domain home

`Loads.dss:kvar` is available for every load, but `GridState` carries no
reactive power field. Recorded as `PARTIAL`/`UNMAPPED` rather than silently
dropped.

**Consequence:** power factor and reactive-power objectives are not representable
in the current domain model.

---

## G-11 — Licence is UNKNOWN

Not stated in the User Guide excerpt retrieved from the authoritative source.
Must be resolved before any redistribution or publication of processed data.

---

## G-12 — Reactive power, ZIP models and per-phase detail exceed the domain

`model`, `kvar`, `Phases` and per-phase bus suffixes are all available in
SMART-DS and understood, but Phase 2 has no ZIP, reactive or per-phase concept.
Recorded as `UNMAPPED`/`PARTIAL` so the information is not lost.

---

## Summary

| ID | Gap | Blocks | Can be fixed by ingestion alone? |
|---|---|---|---|
| G-01 | Battery dispatch absent **and underivable** | Phase 9, 10 | **No** — needs a second dataset or an explicitly simulated augmentation |
| G-02 | No grid flow timeseries | Per-node balance, Phase 11 | **No** — needs a power-flow solve |
| G-03 | PV irradiance shape link unresolved | Phase 4 renewables | **Yes** — download sub-region `LoadShapes.dss` |
| G-04 | No EV charging demand | Phase 9 flexibility | **No** — needs a second dataset |
| G-05 | No wind | Phase 4 renewables | **No** — needs a second dataset |
| G-06 | Losses unmodelled (3.22 %) | 0.5 kW balance criterion | **No** — needs a loss model |
| G-07 | Commercial −20.45 kW unexplained | confidence in ingestion | **Unknown** |
| G-08 | Availability not measured | Phase 9 flexibility | **No** |
| G-09 | No timezone | local-time claims | **No** — structural |
| G-10 | No reactive power in domain | power-factor objectives | Yes — domain change |
| G-11 | Licence UNKNOWN | publication | Yes — resolve with source |
| G-12 | ZIP/reactive/per-phase exceed domain | future objectives | Yes — domain change |

**Four of twelve can be fixed by better ingestion.** The rest are properties of
the dataset or of the Phase 2 domain, and must be resolved by a recorded
decision rather than by working around them.
