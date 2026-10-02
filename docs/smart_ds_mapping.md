# SMART-DS Field Mapping

Semantic mapping from SMART-DS fields to the Phase 2 energy domain.

**Machine-readable source of truth:**
`src/energy_intelligence/data/smartds/mapping.py`, printable with
`energy-intel data mapping`. This document explains it; the code is
authoritative, and a test asserts every row is fully documented.

## Confidence vocabulary

Confidence describes the **mapping** (does this source field mean what the
domain field means), not statistical model confidence.

| Value | Meaning |
|---|---|
| `DIRECT` | The source field already carries the domain quantity, in a compatible unit, with no interpretation |
| `DERIVED` | Computed by an arithmetic relation documented by the authoritative source **and verified against published values** |
| `PARTIAL` | Related but not equivalent; something is missing, aggregated away, or assumption-dependent |
| `UNKNOWN` | Meaning could not be established from the data or documentation |
| `UNMAPPED` | Understood, but the Phase 2 domain has no corresponding concept — recorded so it is not silently dropped |

## Value origin

Origin describes the **value**, on an independent axis. A derived value with a
verified mapping is still `DERIVED`.

| Value | Meaning |
|---|---|
| `OBSERVED` | Directly provided by SMART-DS |
| `DERIVED` | Mathematically computed from SMART-DS values |
| `ESTIMATED` | Requires an estimation method — **none used in Phase 3** |
| `SIMULATED` | Introduced by a simulator — **none in Phase 3** |
| `UNKNOWN` | Not available |

A test enforces that no `DERIVED` mapping claims an `OBSERVED` origin.

---

## Demand

| | |
|---|---|
| **SMART-DS field** | `Loads.dss:kW` |
| **SMART-DS description** | For timeseries scenarios, the maximum power drawn by the load at any time during the year, scaled by the referenced loadshape multiplier at each timepoint |
| **SMART-DS unit** | kW |
| **Domain field** | `load.rated_power_kw` |
| **Domain meaning** | Peak real power the customer can draw over the year |
| **Transformation** | none (stored as a rating, not a timepoint value) |
| **Loss of information** | none |
| **Confidence** | `DIRECT` |
| **Origin** | `OBSERVED` |
| **Evidence** | User Guide v1.0 |

| | |
|---|---|
| **SMART-DS field** | `Loads.dss:kW` × `profiles/<shape>_pu.csv` |
| **SMART-DS description** | Annual maximum kW multiplied by the per-unit load multiplier at the timepoint |
| **SMART-DS unit** | kW (profile is a **dimensionless fraction of the annual maximum**, max exactly 1.0) |
| **Domain field** | `demand.total_demand_kw` (time series) |
| **Domain meaning** | Real power demand at a timepoint |
| **Transformation** | `P(t) = Σ over customer Load objects of (kW_rating × profile_pu(t))` |
| **Loss of information** | None for demand |
| **Confidence** | `DERIVED` |
| **Origin** | `DERIVED` |
| **Evidence** | Derivation rule from the User Guide, then **empirically verified**: residential reconstructed 12380.5126 kW vs 12380.5126 kW published, error **0.0000 kW** |

### Centre-tap loads — the most important resolved ambiguity

| | |
|---|---|
| **SMART-DS field** | `Loads.dss:name` suffix `_1` / `_2` |
| **SMART-DS description** | Centre-tap customers appear as two Load objects, one per active line. Both carry the **same** kW |
| **Domain field** | `load.customer` grouping |
| **Domain meaning** | One Phase 2 load asset per customer |
| **Transformation** | `P_customer(t) = kW₁·profile(t) + kW₂·profile(t)` |
| **Loss of information** | The per-phase split is discarded; Phase 2 has no per-phase node |
| **Confidence** | `DERIVED` |
| **Origin** | `DERIVED` |
| **Evidence** | **Empirically resolved on the real feeder** (see below) |

This was resolved with evidence, not by reading the documentation carefully. The
documentation says centre-tap loads are "two equal loads which are half of the
total customer load", which is easy to misread. Observed: all 1,819 pairs in the
real feeder carry **identical** kW. Both readings were tested against the
dataset's own published peak power:

| Hypothesis | Reconstructed | vs published 15090.72 kW |
|---|---|---|
| De-duplicate the pair | 9,157.14 kW | **−39.3 %** |
| **Sum both members** | **15,070.27 kW** | **−0.14 %** |
| Sum both, residential only | 12,380.5126 kW | **error 0.0000 kW** |

Summing is correct. The User Guide's "multiply by 0.5" guidance applies to
*driving an OpenDSS solve*, not to reconstructing customer total demand.

## Grid

| | |
|---|---|
| **SMART-DS field** | `Summary_data.csv:Total peak time real losses (kW)` |
| **SMART-DS description** | Published real losses at the peak timepoint |
| **SMART-DS unit** | kW |
| **Domain field** | `node_power.unaccounted_kw` |
| **Domain meaning** | Explicit residual when reported node figures do not close. Phase 2 has no network-loss concept (physics deferred, D-023) |
| **Transformation** | `residual = supplied − consumed` |
| **Loss of information** | One aggregate value, no per-element or per-node attribution, no timeseries |
| **Confidence** | `PARTIAL` |
| **Origin** | `OBSERVED` |
| **Evidence** | Observed: 501.672 kW = **3.2175 %** of customer power |

| | |
|---|---|
| **SMART-DS field** | `Loads.dss:kvar` |
| **Domain field** | `grid.reactive_power_var` |
| **Transformation** | none |
| **Loss of information** | `GridState` carries no reactive power field. Recorded as a gap |
| **Confidence** | `PARTIAL` |
| **Origin** | `OBSERVED` |

| | |
|---|---|
| **SMART-DS field** | grid import / export flow (timeseries) |
| **SMART-DS description** | **Not provided.** Only a single aggregate circuit power at the published peak timepoint |
| **Domain field** | `grid.import_kw` / `grid.export_kw` |
| **Transformation** | `UNAVAILABLE` without a power-flow solve |
| **Confidence** | `UNKNOWN` |
| **Origin** | `UNKNOWN` |
| **Evidence** | The sub-region layout contains profiles, solar_data, load_data, load_curves, scenarios and placements. **None** provides grid flow |

## Renewables

| | |
|---|---|
| **SMART-DS field** | `PVSystems.dss:Pmpp` |
| **Domain field** | `solar.capacity_kw` |
| **Transformation** | none |
| **Confidence** | `DIRECT` |
| **Origin** | `OBSERVED` |
| **Evidence** | Observed: 1,216 PVSystem objects in `solar_high_batteries_low_timeseries` |

| | |
|---|---|
| **SMART-DS field** | `PVSystems.dss:yearly` (irradiance loadshape) |
| **Domain field** | `renewable.actual_power` |
| **Transformation** | would be `P(t) = Pmpp × irradiance(t)/1000` — **NOT implemented** |
| **Loss of information** | The irradiance shape is defined in the **sub-region level** `LoadShapes.dss`, not the feeder folder, so the linkage is unresolvable from the feeder folder alone |
| **Confidence** | `PARTIAL` |
| **Origin** | `DERIVED` (would be) |

| | |
|---|---|
| **SMART-DS field** | `solar_data/*.csv:kW Generated (1000 kW Array)` |
| **Domain field** | `solar.generation_derivation` |
| **Transformation** | would be `P(t) = Pmpp × kW_generated_1000(t)/1000` |
| **Loss of information** | Aggregate weather location, not per-array |
| **Confidence** | `PARTIAL` |
| **Origin** | `DERIVED` |
| **Evidence** | Observed: max 834.050 kW per 1000 kW array, PoA irradiance max 1119.0 W/m² |

`solar_data` columns, all `DIRECT` / `OBSERVED`:
`DNI`, `DHI`, `GHI`, `PoA Irradiance (W/m^2)`, `Wind Speed`, `Temperature`.

Note: NSRDB source data is half-hourly, **interpolated to 15 minutes** by
SMART-DS. The finest real information is 30 minutes.

| | |
|---|---|
| **SMART-DS field** | wind generation |
| **Domain field** | `wind.*` |
| **Transformation** | `UNAVAILABLE` |
| **Confidence** | `UNKNOWN` |
| **Origin** | `UNKNOWN` |
| **Evidence** | No wind folder in the sub-region layout; `metrics.csv` has PV and Battery columns only |

## Storage — the decisive Phase 3 finding

| | |
|---|---|
| **SMART-DS field** | `Storage.dss:kWRated` |
| **Domain field** | `battery.rated_power` |
| **Loss of information** | One rating; charge and discharge limits are not distinguished |
| **Confidence** | `DIRECT` |
| **Origin** | `OBSERVED` |
| **Evidence** | Observed: 93 Storage objects, all `kWRated=8.0` |

| | |
|---|---|
| **SMART-DS field** | `Storage.dss:kWhRated` |
| **Domain field** | `battery.nameplate_energy` |
| **Loss of information** | No depth-of-discharge limit published, so **usable** energy is UNKNOWN and is mapped as `None`, not as the nameplate (D-050) |
| **Confidence** | `PARTIAL` |
| **Origin** | `OBSERVED` |
| **Evidence** | Observed: all `kWhRated=16.0` with `kWRated=8.0` (a 2-hour battery) |

| | |
|---|---|
| **SMART-DS field** | `Storage.dss:kWRated` |
| **Domain field** | `battery.max_charge_power` and `battery.max_discharge_power` |
| **Loss of information** | **A single rating, not a directional split.** One `kWRated` cannot state separate charge and discharge limits, so both map to `None`; the nameplate lives in `rated_power` |
| **Confidence** | `PARTIAL` |
| **Origin** | `OBSERVED` |
| **Evidence** | Observed: 93 Storage objects, all `kWRated=8.0` |

| | |
|---|---|
| **SMART-DS field** | `Storage.dss:kWhStored` |
| **Domain field** | `storage.state_of_charge` (**initial only**) |
| **Transformation** | `SOC₀ = kWhStored / kWhRated` |
| **Loss of information** | **One value for the whole year.** No SOC(t) series |
| **Confidence** | `PARTIAL` |
| **Origin** | `OBSERVED` |

| | |
|---|---|
| **SMART-DS field** | `Storage.dss:%EffCharge` / `%EffDischarge` |
| **Domain field** | `battery.round_trip_efficiency` |
| **Transformation** | `η = (EffCharge/100) × (EffDischarge/100)` = 0.9025 |
| **Confidence** | `DERIVED` |
| **Origin** | `DERIVED` |
| **Evidence** | Observed: all 95.0 / 95.0 |

| | |
|---|---|
| **SMART-DS field** | `Master.dss: Circuit bus1` |
| **Domain field** | `topology.grid_connection_node` (`is_grid_connection=True`) |
| **Transformation** | phase suffix stripped; the bus name is kept verbatim in the node name |
| **Loss of information** | **NONE.** Read, not inferred: the feeder *folder* name does not contain the source bus, so deriving it from the folder would be a guess (D-051) |
| **Confidence** | `DIRECT` |
| **Origin** | `OBSERVED` |
| **Evidence** | Observed: `New Circuit.feeder_p1udt12703-p1uhs0_1247x bus1=p1udt12703-p1uhs0_1247x`, and that bus appears as a `Line` terminal |

| | |
|---|---|
| **SMART-DS field** | `Storage.dss:State` |
| **Domain field** | `battery.availability` |
| **Loss of information** | `IDLING` is a static modelling state, **not** an availability signal. No outage or derate data exists |
| **Confidence** | `UNKNOWN` |
| **Origin** | `UNKNOWN` |
| **Evidence** | Observed: 93/93 are `State=IDLING` |

### Battery dispatch — UNAVAILABLE, and not derivable

| | |
|---|---|
| **SMART-DS field** | `Storage.dss` dispatch / charge / discharge power (timeseries) |
| **SMART-DS description** | **Does not exist in the dataset** |
| **Domain field** | `storage.charge_power_kw` / `discharge_power_kw` |
| **Transformation** | `UNAVAILABLE` — no derivation justified |
| **Loss of information** | Total |
| **Confidence** | `UNKNOWN` |
| **Origin** | `UNKNOWN` |
| **Evidence** | Exhaustive inspection of `Storage.dss` and `LoadShapes.dss`: no storage loadshape, no SOC column, no dispatch column, no discharge curve |

**The Phase 2 handoff suggested dispatch "may need to be derived from state of
charge". Verification shows that route is also unavailable.** Deriving power
requires SOC(t) and SOC(t+1); SMART-DS supplies a *single* `kWhStored` scalar,
so there is nothing to difference. **No battery dispatch was derived, and no
`StorageState` was constructed.** Producing a dispatch series would have meant
inventing one.

## EV

| | |
|---|---|
| **SMART-DS field** | `placements/ev_residential=*.json`, `ev_commercial=*.json` |
| **SMART-DS description** | Which customer loads are selected for EV adoption at a penetration level |
| **Domain field** | `ev.connected_vehicles` |
| **Transformation** | none |
| **Loss of information** | **LOCATIONS ONLY.** No EV charging load, power, energy demand or session timeseries anywhere in the dataset. A count of adopted sites is not a charging profile |
| **Confidence** | `PARTIAL` |
| **Origin** | `OBSERVED` |
| **Evidence** | The User Guide describes these purely as "charging locations"; the sub-region layout has no EV timeseries folder |

No `EVCharger` asset was created, because a charging profile cannot be invented.

## Topology

| | |
|---|---|
| **SMART-DS field** | `Lines.dss:bus1` / `bus2` |
| **Domain field** | network element `from_node` / `to_node` |
| **Loss of information** | Bus names carry phase suffixes (`.1.2.3`); Phase 2 has no per-phase node, so the base bus name is taken |
| **Confidence** | `DIRECT` |
| **Origin** | `OBSERVED` |
| **Evidence** | Observed: 4,555 Line objects with explicit endpoints |

| | |
|---|---|
| **SMART-DS field** | `Buscoords.dss` |
| **SMART-DS description** | Bare `<bus> <longitude> <latitude>` lines with **no directives at all**. Parsing it with a generic `.dss` parser yields **zero elements** |
| **SMART-DS unit** | degrees |
| **Domain field** | `node.metadata.latitude` / `longitude` |
| **Loss of information** | Phase 2 has no lat/lon field; recorded as asset metadata |
| **Confidence** | `DIRECT` |
| **Origin** | `OBSERVED` |

| | |
|---|---|
| **SMART-DS field** | `Master.dss:basekV` |
| **Domain field** | `node.voltage_kv` |
| **Loss of information** | one value for the circuit, not per node |
| **Confidence** | `PARTIAL` |
| **Origin** | `OBSERVED` |
| **Evidence** | Observed: `basekV=12.47` |

`Transformers.dss` → network element, `DIRECT`/`OBSERVED` (642 objects, ratings only).

## Time

| | |
|---|---|
| **SMART-DS field** | `Master.dss:Solve mode=yearly stepsize=15m number=35040` |
| **SMART-DS unit** | minutes |
| **Domain field** | `time_base.timestep_minutes` |
| **Transformation** | none |
| **Loss of information** | **NO resampling is required.** The load profiles are themselves native 15-minute (`interval=0.25` h, `npts=35040`), matching the Phase 2 grid exactly |
| **Confidence** | `DIRECT` |
| **Origin** | `OBSERVED` |
| **Evidence** | Confirmed **three independent ways**: `Master.dss` stepsize/number; `LoadShapes` npts=35040 interval=0.25; profile files contain exactly 35040 values and `solar_data` 35040 data rows |

### Timestamps — an explicit assumption

SMART-DS carries **no timezone whatsoever**. Observed time is a position in a
synthetic 35040-point yearly vector. Phase 2 requires timezone-aware datetimes
and rejects naive ones, so a choice is unavoidable. It is made explicitly and
recorded on every provenance chain:

> The synthetic clock is anchored to `<year>-01-01T00:00:00+00:00` and treated
> as UTC.

Why acceptable: the dataset is internally consistent on one synthetic clock, so
anchoring it anywhere preserves every **relative** time relationship — which is
what forecasting, flexibility and balance validation depend on. What it does
*not* preserve is true wall-clock local time (Austin is CST/CDT). Recorded as a
known limitation in `docs/gaps_report.md`.

## Recorded as `UNMAPPED` (understood, not carried)

| Source | Why not mapped |
|---|---|
| `Loads.dss:model` | All 3690/3690 are `model=1` (constant PQ). Phase 2 has no ZIP / power-factor modelling concept |
| `Lines.dss:Length` / `Units` | Phase 2 has no length field. Observed `Units=km` |
| `Loads.dss:Phases` | Recorded as asset metadata; 3638 single-phase, 52 three-phase |
