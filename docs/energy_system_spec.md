# Energy System Specification — Phase 2

The formal contract for **what energy system this project models**. Every later
phase (3-20) depends on this document. Where a value is not yet decided it says
`TBD`, and where it is not yet known it says `UNKNOWN` — never a plausible guess.

Companion documents:

* `decisions.md` — why each choice was made (D-022 … D-039)
* `flow.md` — the implemented call flow, by real function name
* `docs/agent_responsibility_matrix.md` — future agent responsibilities
* `docs/architecture.md` — target architecture and component boundaries

---

## 1. What is being modelled

A **renewable-integrated distribution-level environment containing distributed
energy resources**, connected to the utility grid at a point of common coupling.

```text
                        UTILITY GRID
                             │
                             ▼
                  DISTRIBUTION SYSTEM
                             │
        ┌────────────────────┼────────────────────┐
        ▼                    ▼                    ▼
     DEMAND              RENEWABLES              DERs
        │                    │                    │
   ┌────┼────┐        ┌──────┴──────┐      ┌──────┴──────┐
   ▼    ▼    ▼        ▼             ▼      ▼             ▼
Residential Commercial Industrial  Solar   Wind       Battery
                                                 │            │
                                       Flexible      EV
                                        Loads      chargers
```

**In scope:** demand by category, solar and wind generation, battery storage,
EV charging, flexible/HVAC load, the utility connection, and the network
topology connecting them.

**Out of scope:** transmission, wholesale markets, upstream generation beyond the
point of common coupling, and anything not listed above.

### Why this boundary

It was chosen because it makes the project's target applications simultaneously
addressable rather than sequentially bolted on: renewable integration, grid
reliability, demand and renewable forecasting, DER coordination, energy
flexibility, storage optimisation, peak management, curtailment, uncertainty,
resilience, anomaly detection, simulation, and autonomous decision support.

Critically, it yields a real **decision problem** rather than a forecasting
exercise. The question the eventual system answers is approximately:

> Given the current state of a renewable-integrated distribution system,
> uncertain future demand and renewable generation, available flexible resources,
> and system constraints, what action should the system take?

No asset count, bus count, feeder count or rating is hard-coded. All come from
configuration. A single lumped node and a large multi-feeder system use the same
types.

---

## 2. State variables

Every variable is classified by **role**. A variable has exactly one role, and the
role determines which container may hold it (`VariableRole`, enforced at
construction):

| Role | Meaning |
|---|---|
| `OBSERVATION` | Measured or inferred system state; read-only to control |
| `ACTION` | A decision variable the system may set |
| `DERIVED` | Computable from observations |
| `CONSTRAINT` | A bound or relation the solution must respect |
| `OBJECTIVE` | A quantity to be optimised |

The action/observation boundary is enforced in both directions:
`ObservationRecord` rejects `ACTION`; `Action` rejects anything but `ACTION`.
Battery SOC is therefore an observation and cannot be commanded; battery
charge/discharge power is an action and cannot be observed as a given.

### 2.1 Demand

| | |
|---|---|
| **Name** | `total_demand` |
| **Role** | OBSERVATION |
| **Meaning** | Total electrical demand at a node |
| **Unit** | kW |
| **Range** | ≥ 0 |
| **Resolution** | 15 min (chosen; see `decisions.md` D-027) |
| **Source** | simulation / telemetry |
| **Required** | yes |
| **Can be missing / noisy** | yes / yes |
| **Used by** | forecasting, flexibility, optimization, digital twin |

Also `demand_by_category` (DERIVED, kW, `{"residential": …, "commercial": …}`),
which must sum to `total_demand`.

### 2.2 Renewable generation

| | |
|---|---|
| **Name** | `renewable_available_power`, `renewable_actual_power` |
| **Role** | OBSERVATION |
| **Unit** | kW |
| **Range** | ≥ 0 |
| **Resolution** | 15 min |
| **Used by** | forecasting, flexibility, optimization, red team, digital twin |

`renewable_curtailment_kw` is **DERIVED** (`available − actual`) and
`renewable_curtailment_fraction` is DERIVED in [0, 1]. A solar asset additionally
declares `tilt_degrees` and `azimuth_degrees`; a wind asset declares
`cut_in_speed`, `rated_speed`, `cut_out_speed`, which must satisfy
`cut_in < rated < cut_out`.

### 2.3 Storage

| | |
|---|---|
| **Name** | `battery_state_of_charge` |
| **Role** | OBSERVATION |
| **Meaning** | Stored energy as a fraction of usable capacity |
| **Unit** | fraction |
| **Range** | 0 ≤ SOC ≤ 1 |
| **Resolution** | 15 min |
| **Source** | simulation / telemetry (BMS) |
| **Required** | if a battery exists |
| **Can be missing / noisy** | yes / yes |
| **Used by** | forecasting, flexibility, optimization, digital twin |

`battery_charge_power` and `battery_discharge_power` are **ACTION** (see §3).
`battery_available_energy` is **DERIVED** from `usable_energy` and the SOC floor.
A battery may not charge and discharge simultaneously — enforced at construction.

Static battery parameters, all required: `usable_energy` (kWh), `min_soc`,
`max_soc` (with `min_soc ≤ max_soc`), `max_charge_power`, `max_discharge_power`
(kW), `round_trip_efficiency` in (0, 1].

### 2.4 EV and flexible resources

| | |
|---|---|
| **Name** | `ev_connected_vehicles`, `ev_available_connectors`, `ev_charging_power` |
| **Role** | OBSERVATION |
| **Unit** | count, kW |
| **Used by** | flexibility, optimization |

`ev_deferrable_power_kw` is **DERIVED** and must not exceed
`ev_charging_power_kw` — deferring more than is being drawn is not a deferral.

`ev_min_return_soc` (fraction) and `ev_departure_deadline` are the constraints
that make EV flexibility non-trivial: energy must come back.

Flexible loads declare `min_power`, `max_power` (kW), `shiftable_energy`
(fraction of consumption that may move), `priority` (higher = less deferrable)
and `is_hvac` (marks thermal load, which carries comfort coupling).

### 2.5 Grid state

| | |
|---|---|
| **Name** | `grid_import_kw`, `grid_export_kw` |
| **Role** | OBSERVATION |
| **Unit** | kW |
| **Range** | each ≥ 0 |
| **Used by** | optimization, constraint checking, digital twin |

`grid_frequency_hz` and `voltage_by_node` are **OBSERVATION**, with per-value
quality flags. They are **not derived**: Phase 2 runs no power flow, so a
computed voltage would be a fabrication. They exist in the vocabulary because the
chosen topology admits them; computing them is Phase 11's job.

---

## 3. Action space

Each action declares the authority level sufficient to perform it. The check is
an explicit set membership test, **not** a ranking of authority levels, because
curtailment is a direct output reduction rather than a shift in time (D-028).

| Action | Role | Control variable | Unit | Min | Max | Authority sufficient |
|---|---|---|---|---|---|---|
| `battery_charge` | ACTION | charge power setpoint | kW | 0 | `max_charge_power` | `DISPATCHABLE` |
| `battery_discharge` | ACTION | discharge power setpoint | kW | 0 | `max_discharge_power` | `DISPATCHABLE` |
| `ev_charging` | ACTION | charging power setpoint | kW | 0 | charger rating | `DISPATCHABLE`, `SCHEDULEABLE` |
| `ev_defer_charging` | ACTION | charging power setpoint | kW | 0 | charger rating | `DISPATCHABLE`, `SCHEDULEABLE` |
| `flexible_load_shift` | ACTION | power delta | kW | signed, ≠ 0 | ±`shiftable_energy` | `DISPATCHABLE`, `SCHEDULEABLE` |
| `hvac_setpoint_adjustment` | ACTION | temperature setpoint | degC | signed, ≠ 0 | comfort envelope | `DISPATCHABLE`, `SCHEDULEABLE` |
| `renewable_curtailment` | ACTION | curtailed power | kW | 0 | available power | `DISPATCHABLE`, `CURTAILABLE` |
| `demand_response` | ACTION | shed/restore power | kW | 0 | `shiftable_energy` | `DISPATCHABLE`, `SCHEDULEABLE` |
| `grid_import_limit` | ACTION | import ceiling | kW | signed, ≠ 0 | connection rating | `DISPATCHABLE` |

Every action also carries `start_step` (grid step index, may be negative for
retrospective replay), `duration_steps` (≥ 1), and mandatory `provenance`.

**Signed vs non-negative.** Non-negative actions are power magnitudes.
Signed actions are ones where "more" and "less" are both meaningful. A signed
action with a **zero** setpoint is rejected: zero is the absence of an action,
not an action.

### 3.1 Control authority

Selected model: **tiered per-asset authority, carried on the action** (D-028).

| Level | Meaning |
|---|---|
| `DISPATCHABLE` | Full control of output |
| `SCHEDULEABLE` | May shift in time |
| `CURTAILABLE` | May reduce output |
| `ADVISORY_ONLY` | May be reasoned about, never dispatched |
| `NOT_CONTROLLABLE` | An input, never an action target |

This is realistic: a distribution operator genuinely cannot dispatch a
customer's EV or HVAC unilaterally. It also makes Phase 13's adaptive autonomy
meaningful — autonomy is high for self-owned assets and degrades to advisory
where authority is absent. An `Action` cannot be constructed under
`ADVISORY_ONLY`; that is advisory output, not an action.

### 3.2 Ramp and duration constraints

`ramp_rate_kw` and `minimum_up_time` exist as **constraint categories** but are
**not populated in Phase 2**: a ramp limit is only meaningful relative to a
measured ramp history, which requires the dataset (Phase 3). Marked `TBD`.

---

## 4. Constraints

Declared in Phase 2, **not enforced**. `Constraint.is_enforceable_in_phase_2`
returns `False` for every constraint, so nothing implies enforcement that does not
exist. No algebraic formulation is committed, because that would prejudge the
Phase 10 optimizer choice.

| Category | Unit | Scope | Notes |
|---|---|---|---|
| `power_balance` | kW | system / node | Enforced structurally by `NodePower` |
| `state_of_charge_limits` | fraction | battery | From `min_soc` / `max_soc` |
| `charge_discharge_power_limits` | kW | battery | From rated charge/discharge |
| `renewable_availability` | kW | generator | Cannot exceed available |
| `flexible_load_bounds` | kW | load | `min_power` ≤ P ≤ `max_power` |
| `grid_connection_capacity` | kW | PCC | Import/export ceiling |
| `voltage_limits` | V | node | **Requires power flow — Phase 11** |
| `thermal_loading_limits` | A | network element | **Requires power flow — Phase 11** |
| `minimum_up_time` | min | asset | TBD pending Phase 3 data |
| `ramp_rate_limits` | kW | asset | TBD pending Phase 3 data |
| `ev_departure_deadline` | fraction | EV | Vehicle must return to `min_return_soc` |
| `demand_flexibility_limit` | kW | load | Ceiling on deferrable energy |

A constraint must be scoped to at least one asset or node — an unscoped
constraint cannot be checked against anything, and is rejected.

### 4.1 Power balance

The one structural invariant Phase 2 **does** enforce, because it is what
distinguishes a distribution model from a lumped one:

```text
generation_kw + storage_kw + grid_kw − load_kw + unaccounted_kw ≈ 0
```

per node, within `POWER_BALANCE_TOLERANCE` (0.5 kW). The tolerance is
deliberately generous enough for unmodelled feeder losses while still catching a
sign error, a double count or a missing asset. Phase 11 may tighten it once a
loss model exists. `unaccounted_kw` records an explicit residual rather than
hiding it.

Sign convention throughout: generation and grid import positive; load and grid
export negative; storage a signed injection.

---

## 5. Objectives

Phase 2 defines the categories and, critically, **the conflicts between them**.
It does **not** combine them into a weighted sum: choosing weights is a policy
decision that encodes a preference between cost and comfort, and making it now
would either bake in an arbitrary answer or hide the trade-off inside a scalar
nobody can audit. There is deliberately **no `weight` field**, asserted by
`test_objective_has_no_weight_field`.

| Objective | Direction | Unit | Conflicts with |
|---|---|---|---|
| `energy_cost` | minimize | currency (TBD) | renewable_utilization, battery_degradation |
| `peak_demand` | minimize | kW | energy_cost, load_shed_amount |
| `grid_import_peak` | minimize | kW | energy_cost |
| `renewable_curtailment` | minimize | kWh | energy_cost |
| `renewable_utilization` | maximize | percent | energy_cost, peak_demand |
| `battery_degradation` | minimize | currency/TBD | energy_cost, renewable_utilization |
| `load_shed_amount` | minimize | kWh | energy_cost, renewable_utilization |
| `unmet_demand` | minimize | kWh | — (independent) |
| `constraint_violation` | minimize | count | — (independent) |
| `emissions` | minimize | kgCO2e/TBD | energy_cost |

Direction and unit live in `OBJECTIVE_METADATA`, keyed by category, so two
objectives of the same category cannot disagree about what "better" means.
`conflicts()` is symmetric, so adding objectives in a different order cannot
change the answer.

The currency and emissions units are `TBD`: a tariff model is a Phase 3 concern
and an emissions factor depends on the grid mix, which is not yet fixed.

---

## 6. Temporal model

```text
t−N … t−2  t−1  t  │  t+1 … t+H
   history  now   │   forecast / planning horizon
```

| Property | Value | Source |
|---|---|---|
| Native timestep | **15 minutes** | chosen, D-027 |
| Horizon | **96 steps = 24 hours (day-ahead)** | chosen, D-027 |
| Hourly view | derivable (4 steps/hour) | derived |
| Timezone | timezone-aware UTC; naive rejected | chosen |

Rationale for 15 minutes rather than hourly: battery cycling and EV charging are
only meaningfully schedulable at sub-hourly resolution, and PV ramp on a
distribution feeder is materially non-representative inside an hour. Hourly was
the source forecasting repository's resolution, but that repository models
regional aggregate load/wind/PV with no storage or EVs, so its resolution was
never constrained by the flexibility this project needs to study.

`TimeBase` is constructed from configuration, not hard-coded, so 5-minute or
hourly data requires no code change. The allowed timestep set is constrained to
values for which integer aggregation to coarser grids is well defined.

Timestamps that do not fall exactly on a grid step are **rejected** rather than
rounded: silent rounding would shift the whole system by a partial step.

---

## 7. Topology

Chosen model: **topology-aware, node-granular, physics deferred** (D-026).

* `Node` — `node_id`, `name`, `voltage_kv`, `parent_node_id`, `is_grid_connection`.
* `NetworkElement` — `element_id`, `kind` (line / transformer / switch /
  capacitor / regulator / feeder head), `from_node_id`, `to_node_id`, `rating`,
  `is_in_service`, `parameters`.
* `NetworkTopology` — nodes plus elements, plus the required statement of where
  the utility attaches.

`is_in_service` is how an outage or switching state enters the model, which the
Phase 12 Red Team will need to perturb.

Enforced invariants: non-empty node set; no duplicate ids; every parent and
element endpoint resolves; exactly one boundary per grid connection (a grid
connection node must have no parent); at least one `is_grid_connection` node.
Elements may not connect a node to itself.

`NetworkTopology.is_aggregate` is surfaced rather than hidden. A conclusion drawn
on a single lumped node cannot support per-location claims, and `configs/domain.toml`
declares `topology_kind` explicitly for the same reason.

**Not implemented:** no power flow. Voltage, current and thermal loading are
observations and constraint categories only.

---

## 8. Provenance

Every observation, state, forecast, action, constraint and objective carries a
mandatory `Provenance`. It cannot be omitted: `Provenance` has no valid
construction without a `SourceReference` and at least one `ProvenanceEvent`.

```text
SourceReference ──► Provenance ──► { ProvenanceEvent × ≥1 }
                                └──► { ProcessingStep × ≥0 }
```

* `SourceReference` — `source_id`, `dataset`, `locator`, `version`, optional
  `checksum`.
* `ProvenanceEvent` — `timestamp` (ISO-8601), `event`, `detail`.
* `ProcessingStep` — `name`, `version`, `detail`. `version` is required so a
  result remains reproducible after the code changes.

The checksum is optional in Phase 2 because no dataset has been ingested yet. It
should become mandatory in Phase 3 — SAT-SA makes content digests central, and
Phase 2 deliberately adopts the *contract* without copying its ledger machinery
(D-032).

Motivation is taken from SAT-SA: data must have provenance rather than appearing
magically inside the system.

---

## 9. Data quality

Phase 2 makes the model *capable* of expressing data quality; it detects none of
it. Flags: `ok`, `missing`, `stale`, `outlier`, `invalid_range`, `sensor_error`,
`estimated`, `interpolated`.

Invariants:

* At least one flag required — absence of flags means *unknown*, which must be
  stated, not implied.
* `ok` cannot be combined with anything. Mixing `ok` with `missing` would be
  self-contradictory and would defeat a naive `OK in flags` check.
* Flags are a set, because conditions genuinely co-occur (a value can be both
  stale and interpolated).

`is_usable_for_decision` is **strict**: any adverse flag disqualifies a value
from backing an action. Staleness is included deliberately — an out-of-date state
of charge is exactly the input that produces a confidently wrong dispatch.
Phase 13 may admit estimated values under a tightened assurance level; this is the
safe default.

This is the one area where SAT-SA is a **negative** reference: it has no staleness
detector, no sensor-error flag and no estimated/imputed marker, and its own
research notes acknowledge that gap (D-029).

---

## 10. Uncertainty

Uncertainty is a first-class concept; the mathematics is deliberately deferred.

`UncertaintyEstimate` carries `kind`, `unit`, `method`, and **exactly one** of
`lower_bound` / `upper_bound` / `standard_deviation` / `relative_std`.

| `kind` | Meaning | Reducible? |
|---|---|---|
| `measurement` | Sensor error | with better instrumentation |
| `model` | Epistemic | with more data |
| `parametric` | Aleatoric / scenario spread | no |
| `combined` | Several sources | depends |
| `unknown` | Not yet characterised | — |

Rules:

* A non-`unknown` kind **must** quantify something. An uncertainty that asserts
  nothing is not an estimate.
* Supplying more than one numeric summary is **rejected**, because how an interval
  relates to a standard deviation depends on a distributional assumption Phase 8
  has not yet made.
* `method` defaults to `"TBD"` and is reported as not-yet-selected, so the
  outstanding choice is visible rather than hidden.
* `unknown` is a legal value, so a model can honestly say "not characterised yet".

`Forecast` carries `issued_at` and `target_time`, and **rejects** a forecast
issued at or after the moment it describes: such a record is hindsight, and
admitting one would contaminate every later evaluation.

Neither source repository models uncertainty — the forecasting repository
contains no reference to quantiles, conformal prediction, ensembles or
aleatoric/epistemic decomposition. Getting the container right while the
methodology is open is what lets Phase 8 choose a method without changing every
domain object (D-033).

---

## 11. Digital-twin compatibility

`StateOrigin` is mandatory on every state and distinguishes
`observed` / `simulated` / `predicted` / `assimilated`. These are not
interchangeable, because the project's central research signal is comparing a
learned model's prediction against a simulation against reality. With origin
implicit, that comparison would be meaningless.

The same `EnergyState` type carries all four origins, so comparing observed
against simulated requires **no change to the domain representation** — only a
check that the origins differ.

---

## 12. Model-input compatibility

`EnergyState` → `ObservationRecord` → `Forecast` is the consumption path a future
Energy World Model will read. The representation is deliberately generic:
`ObservationRecord.variable` is a validated name, not a closed enum, so the set of
tracked variables stays data-driven rather than fixed at Phase 2.

Serialization is the stable seam a later tokenisation or feature pipeline can
build on: deterministic JSON with an explicit schema version
(`SCHEMA_VERSION = "2.0.0-phase2"`), no hidden fields, and round-trip enforced per
type. No model integration is attempted in Phase 2.

---

## 13. Open questions

Explicitly undecided. Each is `TBD`, not assumed.

| Question | Blocked phase | Note |
|---|---|---|
| Concrete asset inventory and siting | 3 | Depends on the dataset |
| Tariff model and currency unit | 3 | Needed for `energy_cost` |
| Grid emission factors | 3 | Needed for `emissions` |
| Ramp limits and minimum up/down times | 3 | Needs measured ramp history |
| Uncertainty method (quantile / ensemble / conformal) | 8 | Container is ready |
| Optimizer family and formulation | 10 | Must follow Phase 2's constraint set |
| Power-flow solver and fidelity | 11 | Determines voltage/thermal realism |
| Decision-assurance scoring | 13 | No formula invented |
| Whether the ~100M edge model shares weights with the ~1.7B model | 5-6 | Must be decided explicitly |
| MoE expert count and routing objective | 7 | Regime list remains a hypothesis |
| Final Phase 3 dataset download | 3 | SMART-DS is the representability anchor only |
