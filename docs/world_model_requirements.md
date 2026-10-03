# World Model Requirements

What an **action-conditioned Energy World Model** would actually require, and what
SMART-DS provides.

This document exists to stop an overclaim. Phase 5 built a temporal **Energy Demand
Dynamics Model** — a forecasting model over demand. It is deliberately *not* called a
world model, because a world model must learn `f(state, action) -> next state`, and
SMART-DS supplies no actions, no transitions and no system responses. The line between
"a model of a quantity over time" and "a model of a system responding to
intervention" is the line between Phase 5 and a future phase, and this document
records exactly where it falls.

Phase 4's companion is `ml_data_gap_report.md`, which covers what the ML layer lacks.
This one covers the world-model layer.

---

## The definition being tested

A world model needs six things. Each is checked against the acquired dataset rather
than assumed.

| Requirement | Meaning | In SMART-DS v1.0 (AUS/P1U) |
|---|---|---|
| **State** | The variables that determine the system's next state | **PARTIAL** — demand series exist per customer and per feeder; PV exists for ONE array that is not on this feeder; no voltage, current or reactive state |
| **Action** | Something an operator or controller can change | **ABSENT** — no setpoint, dispatch, tariff, price or control signal anywhere in the dataset |
| **Transition** | The rule linking state and action to the next state | **ABSENT** — derivable only by simulation, never observed |
| **System response** | The measured consequence of an action | **ABSENT** — no counterfactual or interventional record |
| **Storage dynamics** | How storage moves between states | **ABSENT** — 93 batteries with ratings and one `kWhStored` scalar; no SOC(t), no dispatch, no efficiency curve over time |
| **Control variables** | The levers an optimiser would move | **ABSENT** — Phase 2's domain model *declares* them (setpoints, schedules, curtailment); the dataset contains none of their values |

**Five of six are absent.** That is the central finding, and it is why Phase 5 is a
demand model and not a world model.

---

## What SMART-DS does provide, precisely

### State-like series (the usable part)

| Quantity | Granularity | Resolution | Coverage | Origin |
|---|---|---|---|---|
| Customer demand | 1,871 customers | 15 min | 35,040 steps, complete | `DERIVED` (rating × per-unit profile) |
| Feeder demand | 1 feeder | 15 min | 35,040 steps, complete | `DERIVED` |
| PV generation | ONE 1000 kW array, not on this feeder | 15 min | 35,040 steps, 52.2% exact zeros | `OBSERVED` |
| Weather at that array | 1 location | 15 min | 35,040 steps | `OBSERVED` (actual, not forecast) |
| Topology | 5,196 nodes, 4,555 line elements | static | — | `OBSERVED` (OpenDSS model) |
| Coordinates | partial | static | — | `OBSERVED` |

### What is conspicuously missing

1. **The feeder's own PV fleet cannot be reconstructed.** The 1,216 `PVSystem`
   objects reference 20 irradiance loadshapes that are not in the dataset, and the one
   `solar_data` file that exists is a *different* location (30.3459 N vs 30.4196 N).
   So even the "renewables" half of the state vector is unavailable for this feeder.
2. **No electrical state.** No per-node voltage, current or reactive power time
   series. `Loads.dss` carries `kvar` and a phase count, but nothing is measured over
   time at a node.
3. **No availability history.** Every asset is modelled as in service;
   `Storage.State=IDLING` is a static modelling state. Degradation and outages cannot
   be learned.
4. **No intervention record.** Not one action was ever observed, so no transition
   model could be fitted to data — only to a simulator.
5. **One year, one region.** No cross-year or cross-region validation is possible.

---

## The Phase 3 balance residual, in this context

The 20.4506 kW residual is a **feeder-level constant at the published peak**. It is
not a per-step quantity and not a per-customer quantity.

* It does **not** enter any Phase 5 target or feature, because both are per-customer
  per-step series.
* It **is** direct evidence about the missing-state problem: it is exactly the kind of
  quantity a power-flow model would explain (real losses, 3.2175% at the peak), and the
  dataset provides no model that can. A world model asked to predict feeder power
  would have to account for it with a loss model that does not exist here.

So the residual is *orthogonal* to Phase 5's forecasting target but *diagnostic* of the
gap in this document.

---

## What a true action-conditioned world model would need

Ordered by how much each would change the project's architecture.

### 1. An environment that responds to intervention

Minimum viable: an OpenDSS or CYME power flow over the same feeder, with controls
exposed (battery charge/discharge setpoints, PV curtailment, load switching). This
yields a **simulated** transition function. Every value it produces is `SIMULATED` and
must be labelled so — Phase 3's vocabulary already has the label.

Without it, the only honest source of transitions is a second dataset with observed
control actions.

### 2. Storage that actually dispatches

Minimum viable: a storage control strategy plus a solver run, producing SOC(t) and
charge/discharge power. Phase 9's flexibility work cannot start on ratings alone
(G-01).

### 3. A weather forecast, not an observation

Measured consequence: PV at 24 hours is 74.3 kW MAE without weather and 8.57 kW with
it at 15 minutes. A world model conditioned on weather needs a **forecast** product;
SMART-DS ships observations only.

### 4. Feeder PV it can actually see

Either the 20 missing irradiance loadshapes, or a second PV dataset on the same
feeder. Without this, net load and renewable curtailment cannot be modelled for this
feeder at all (G-03).

### 5. Per-node electrical state

Voltage and reactive power, either measured (SCADA) or solved. Without them there is
no constraint-aware dispatch, no Volt-VAR, and no way to check that a control action
is feasible rather than merely arithmetically valid.

### 6. Availability and disturbance records

Outages, derates and maintenance. `availability = 1.0` is currently a **modelling
assumption**, recorded as such in every dataset manifest, not a fact.

---

## Architecture consequence

| Question | Answer from this dataset |
|---|---|
| Can we learn `state -> next demand`? | **Yes.** Phase 5 measured this against persistence and gradient boosting. |
| Can we learn `state, action -> next state`? | **No.** No action has ever been observed, so no transition can be fitted to data. |
| Would a foundation model change that? | **No.** A larger model cannot learn a relationship the data does not contain. |
| What would a foundation model buy instead? | Multi-task transfer across horizons, assets and (later) datasets; better few-shot generalisation; and a shared representation for regimes. That is a *different* claim from world modelling, and `qwen_energy_requirements.md` states it precisely. |

The last row is the honest Phase 5 → Phase 6 bridge. **A Qwen-based energy model
built on this dataset would be a demand model.** Calling it a world model would be a
naming error that propagates into every later claim, so it is not made here.

---

## Recommendation

Do not build an action-conditioned Energy World Model on SMART-DS alone.

Sequence it as:

1. **Now → Phase 6**: a foundation model over the demand dynamics that *do* exist,
   evaluated against the Phase 4 and Phase 5 baselines recorded in this repository.
2. **In parallel**: acquire or simulate the missing inputs — weather forecasts, a PV
   dataset on this feeder, battery dispatch, per-node state.
3. **Only then**: an action-conditioned model, trained on data that contains actions,
   with `SIMULATED` labelling wherever a simulator supplied the transition.

Step 3 is blocked by data, not by architecture. That should be established now, from
evidence, rather than discovered in Phase 9.