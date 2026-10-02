# Agent Responsibility Matrix

**Status: DESIGN DOCUMENT ONLY. No agent is implemented in Phase 2.**

This matrix maps responsibilities that exist in the two source repositories onto
the energy domain, and records what this project would reuse and what it would
**redesign**. It exists so that Phases 14-15 have a traceable starting point
rather than an invented agent roster.

## How to read the `Source` column

Sources are cited by repository and file, from direct inspection:

* **SAT-SA** = `PrathamKapoor/SAT-SA-with-PQC`
* **Guard** = `PrathamKapoor/Guardrailed-Agentic-MLOps-for-Self-Adaptive-Energy-Forecasting-in-Renewable-Integrated-Smart-Grids`

`Source` records where a responsibility **actually exists in code**. A
responsibility with no cited source is marked **NEW (energy-specific)** and
exists only because the energy domain demands it. Nothing here is presented as
pre-existing fact without evidence.

---

## 0. What the source repositories actually are

This matters more than the roster, because it determines what can be reused.

| | SAT-SA | Guard |
|---|---|---|
| Actual domain | **Cybersecurity SOC supervision** (Smart India Hackathon) | **Forecasting MLOps** for smart grids |
| Energy assets | **none** — `battery`, `feeder`, `voltage`, `storage` → 0 hits | `LOAD`/`WIND`/`PV` regional aggregate; `battery`, `EV`, `HVAC`, `curtail`, `feeder`, `voltage`, `storage` → **0 hits** |
| Dataset | — | RTS-GMLC, 2020, **hourly**, 8,784 rows, targets LOAD/WIND/PV |
| Uncertainty | `ConfidenceVector` (min-aggregation) | **none** — `quantile`, `conformal`, `ensemble`, `aleatoric` → 0 hits |
| Red team | `RedTeamAgent` (name only) | **none** — `red.?team`, `adversar` → 0 hits |
| Digital twin | none | **none** — `digital.?twin`, `simulat` → 0 hits |
| Optimizer | none | **none** — `optimiz`, `MILP`, `MPC`, `cvxpy` → 0 hits |
| Provenance | **Strong**: `SourceRecord`, hash-chained ledger, canonical form | **Weak**: SHA-256 checksums only |
| Agent contract | README advertises a 13-field `Observation`, but **no implementation exists** — the fields appear only in docstrings | **Real**: `AgentOutput` + capability firewall |

**Conclusion, stated plainly:** neither repository contains the domain this
project needs. SAT-SA contributes the *architecture of evidence and
governance*; Guard contributes the *real agent contract, capability firewall and
forecasting lifecycle*. The energy domain — assets, topology, actions,
constraints, uncertainty — must be built here, and Phase 2 has built the
data contract for it.

---

## 1. Reusable patterns (adopted, not copied)

These are the concrete inheritances. Each is evidenced and each is already
reflected in Phase 2 code.

| # | Pattern | Source evidence | Where it lives here |
|---|---|---|---|
| R1 | **Provenance is mandatory, not optional** | SAT-SA `satsa/domain/evidence.py`: `SourceRecord`, `ProvenanceRecord` (`:24-37`, `:275-290`) | `domain/provenance.py` — `Provenance` is unconstructible without a source and ≥1 event (D-032) |
| R2 | **Abstain ≠ clean** (five-state result vocabulary) | SAT-SA `satsa/contracts/worker.py:21` `BATCH_STATES` = signal / no_signal / insufficient_data / not_applicable / error | Adopts the principle via `DataQuality` (an adverse flag is never "clean") and `UncertaintyKind.UNKNOWN`. A dedicated agent-result enum is **Phase 14** work |
| R3 | **Multi-dimensional confidence, min-aggregated, `None` ≠ 0** | SAT-SA `satsa/domain/evidence.py:64-97` `ConfidenceVector` | Not adopted in Phase 2. `DataQuality.is_usable_for_decision` is a strict boolean gate. A confidence *vector* is **Phase 13** work |
| R4 | **Mandatory evidence citation on a positive claim** | SAT-SA `satsa/domain/evidence.py:187-197` — a `signal` finding must cite ≥1 evidence ref | Adopted as provenance being mandatory on every domain object |
| R5 | **Fail-closed everywhere** | SAT-SA `qsmlops/supervisor/supervisor.py:161-237` — agent exception, validator crash and dropped evidence all escalate | Adopted: health check fails closed; invalid config rejects rather than warns; degraded data never backs a decision |
| R6 | **Capability firewall: agents advise, governance acts** | Guard `agents/firewall.py:20-29` — 5 allowed advisory types, 7 blocked lifecycle types, unknown blocked by default | Strongly endorsed. Extended in Phase 2 as the tiered authority model (`domain/actions.py`) |
| R7 | **Deterministic authoritative backend; LLM output is untrusted advisory** | Guard `agents/base.py:21-52` — `LocalRuleBackend` is "the designed production path"; `LLMBackendInterface` raises `NotImplementedError` and its output "can never execute, modify, train, deploy" | Endorsed as policy for Phase 15. Relevant to the eventual Qwen Energy World Model: its output must be advisory until verified |
| R8 | **Governed threshold change, graded against real labels** | Guard `governance/calibration` / `retraining/policy.py` `IDENTITY_KEYS` fingerprinting | Adopted in spirit: `AppConfig.fingerprint()` and `DomainConfig.fingerprint()` pin a run to its settings |
| R9 | **Ref-only inputs to components** | SAT-SA `satsa/contracts/worker.py:24-53` (`SnapshotRef`, `BaselineRef`, `PolicyRef`) — never a live DB handle | Adopted: domain objects are frozen dataclasses with no I/O handles |
| R10 | **Explicit uncertainty of the uncertainty** | SAT-SA `agent_version` / `detector_version` fields | Adopted: `ProvenanceEvent`, `ProcessingStep.version`, `SCHEMA_VERSION` |
| R11 | **Structural discriminator vocabulary** | Guard `agents/schemas.py:7-26` (`ALLOWED_RECOMMENDATIONS`, `BLOCKED_RECOMMENDATIONS`, `QUERY_TYPES`, `AGENT_IDS`) | Adopted: every domain enum is a closed, inspectable vocabulary; `energy-intel domain vocabulary` prints it at runtime |
| R12 | **Data-quality checks return a report, not an exception** | Guard `monitoring/data_quality.py:2-3` returns missing/extra/nan/inf counts and a `critical` flag | Adopted: `DataQuality` and `CheckResult` are report types |

---

## 2. Responsibility matrix

`Phase` is when the responsibility is expected to be implemented. "Contract
only" means the interface is defined in Phase 2 and the behaviour is later.

| Agent | Source | Responsibility | Energy adaptation | Consumes | Produces | Phase | Dependencies |
|---|---|---|---|---|---|---|---|
| **Supervisor Agent** | SAT-SA `satsa/supervisor/engine.py:148-379` (5-stage `observe→reason→act→verify→learn`); Guard `agents/orchestrator.py` | Coordinate all other agents; own the run lifecycle | Extend to the energy loop `observe→understand→predict→attack→simulate→verify→decide→execute→learn` | Agent observations, decision report | Routed decision, lineage log | 15 | All agents below |
| **Data Quality Agent** | SAT-SA `satsa/analysis/workers/negative_space.py`, `coverage_gap.py`; Guard `monitoring/data_quality.py` | Schema, missingness, freshness, range validity | Detect and label `DataQuality` flags on energy observations; **staleness detection is new** (SAT-SA has none) | `ObservationRecord`, raw series | Quality-flagged observations, coverage report | 14 | Phase 3 ingestion |
| **Drift Agent** | SAT-SA `satsa/analysis/workers/drift.py`; Guard `monitoring/feature_drift.py` (PSI + normalised Wasserstein) | Feature and prediction distribution drift | Same mechanisms applied to demand/PV/wind/net-load distributions | Reference vs current distributions, `Provenance` | Drift signals with evidence refs | 14 | Phase 4 baselines |
| **Demand Forecast Agent** | NEW (energy-specific) — Guard forecasts LOAD but has no agent for it | Load forecast health and calibration | Track `total_demand`, `demand_by_category` | Load history, weather | Forecast health signals | 14 | Phase 4 |
| **Renewable Forecast Agent** | NEW (energy-specific) — Guard forecasts PV/WIND but has no agent for it | Solar and wind forecast health | Track `renewable_available_power` and forecast error by technology | Irradiance/wind, PV/wind history | Per-technology forecast health | 14 | Phase 4 |
| **Forecast Performance Agent** | SAT-SA `qsmlops/agents/performance_agent.py`; Guard `monitoring/performance_drift.py` | Forecast accuracy and calibration | Evaluate MAE/RMSE **and calibration** of `Forecast.uncertainty` — calibration is impossible without the Phase 2 uncertainty container | Issued forecasts, realised outcomes | Accuracy + calibration report | 14 | Phase 4, Phase 8 |
| **Anomaly Agent** | SAT-SA `satsa/analysis/workers/anomaly.py` (median/MAD outliers, `min_samples` gate) | Telemetry and system anomalies | Detect infeasible states, impossible SOC, unbalanced nodes, flatline sensors | `EnergyState` history | Anomaly findings with evidence | 14 | Phase 2 domain, Phase 3 |
| **Asset Health Agent** | NEW (energy-specific) | Battery/DER availability and degradation | Own `Asset.availability`; track `round_trip_efficiency` degradation | `StorageState`, asset ratings | Availability updates | 14 | Phase 2 domain |
| **Flexibility Agent** | NEW (energy-specific) — this is the core energy contribution | Quantify available controllable capacity | Own `DERIVED` flexibility: `battery_available_energy`, `ev_deferrable_power_kw`, `shiftable_energy` | State, SOC, EV deadlines, asset authority | Declared flexibility envelope | 9 | Phase 2 domain, Phase 8 |
| **Constraint Agent** | NEW (energy-specific) | Voltage / thermal / operational constraint tracking | Report constraint margins from declared `Constraint` set; flag physics-dependent ones needing Phase 11 | `Constraint`, network ratings, state | Margin report, violation list | 14 | Phase 10, Phase 11 |
| **Red Team Agent** | SAT-SA `qsmlops/agents/redteam.py` (exists by name only) | Attempt to break a proposed decision | Attack a `CandidateAction`: missing telemetry, renewable loss, demand spike, battery unavailable, comms failure, distribution shift | `Action`, `EnergyState`, assumptions | Surviving/failed attack set | 12 | Phase 2 domain, Phase 10 |
| **Optimization Agent** | SAT-SA `qsmlops/agents/optimization_agent.py`; Guard has **none** | Optimisation quality and feasibility | Own the action search within state + constraints + objectives | `EnergyState`, `Forecast`, `Constraint`, `Objective` | `CandidateAction` set | 10 | Phase 10 optimizer |
| **Simulation Agent** | NEW (energy-specific) — neither source has a twin | Digital-twin validation | Produce a `StateOrigin.SIMULATED` outcome for a proposed action; report model–physics disagreement | `CandidateAction`, `EnergyState`, topology | Simulated state, disagreement metric | 11 | Phase 2 domain, Phase 11 twin |
| **Decision Assurance Agent** | NEW (energy-specific); concept adapted from Guard's firewall | Approve or reject a proposed action | Gate on forecast confidence, data quality, uncertainty, model–physics disagreement, constraint margins, authority level | Everything above | APPROVE / REJECT with rationale | 13 | Phase 12, Phase 13 |
| **Performance Agent** | SAT-SA `qsmlops/agents/performance_agent.py`; Guard `final_evaluation/` | Real-world energy outcome | Evaluate cost, peak, curtailment, unmet demand, degradation — **energy outcomes, not generic ML metrics** | Executed outcome, `Objective` set | Outcome scorecard | 14 | Phase 16 |
| **Incident Response Agent** | SAT-SA `qsmlops/agents/incident_response_agent.py` | Unexpected system behaviour | Handle divergence between expected and realised energy behaviour | Alerts, outcome, policy | Incident record, fallback action | 14 | Phase 13 |
| **Governance Agent** | SAT-SA `qsmlops/agents/governance_agent.py`; Guard `governance/state_machine.py` (legal transitions, terminal states) | Policy and deployment rules | Own the lifecycle state machine for models *and* for autonomy level | Policy, decision requests | Permit / deny with reason | 14-15 | Phase 15 |
| **Training Agent** | Guard `retraining/agent` family; SAT-SA has no training agent (Guard does) | Model training and retraining | Trigger retraining on drift, using the retraining policy | Drift signals, dataset | New model candidate | 14-15 | Phase 14 |
| **Training Optimisation Agent** | SAT-SA `qsmlops/agents/training_optimization_agent.py` | Training efficiency | Tune training cost/latency without weakening evaluation | Training runs, budget | Cheaper training plan | 14-15 | Phase 15 |
| **Deployment Agent** | NEW (energy-specific) — Guard has governance but no separate deployment agent | Controlled model deployment | Deploy only after assurance; roll back safely | Approved model, governance state | Deployment record | 14-15 | Phase 13, Phase 15 |
| **Report Generation Agent** | SAT-SA `satsa/analysis/report.py`; Guard `agents/report_agent.py` | Human-readable reporting | Render energy decisions with provenance and limitations | Any agent output | Report | 19 (UI phase) | Phase 16 |
| **Evidence Retrieval Agent** | Guard `agents/evidence_agent.py` | Resolve evidence references | Fetch the `Provenance` behind any claim | Evidence refs | Provenance records | 14 | Phase 2 provenance |
| **Security Agent** | SAT-SA `qsmlops/agents/security_agent.py` | Security posture | Optional for an energy system; likely unnecessary at hackathon scale | — | — | **TBD** | — |
| **Quantum Security Agent** | SAT-SA `qsmlops/agents/quantum_agent.py` | Post-quantum security | **Not applicable.** No cryptographic requirement has been identified for this project | — | — | **Excluded** | — |

### Agents deliberately **not** carried over from SAT-SA

SAT-SA's 23 supervisory agents exist for SOC evidence workflows. These have **no
energy analogue** and are excluded rather than translated cosmetically:
`EntityAssetResolution`, `Ingestion`, `Normalization`, `ExecutionGap`,
`NegativeSpace`, `WorkflowReconstruction`, `PeerBenchmark`, `CoverageGap`,
`CaseSimilarity`, `EvidenceCompleteness`, `CorrelationSignalFusion`,
`EntityRiskScoring`, `Prioritization`, `ReviewWorkflow`, `TrustProvenance`,
`EvidenceAssembly`, `MetaAudit`, `Validation`.

The energy system has no alerts/cases/investigation/escalation/disposition
evidence model, so the workers that reason over it do not transfer. SAT-SA's own
`AGENT_GOVERNANCE.md` states its 32-agent count is a **consequence of
responsibility separation, not a target**; this matrix follows the same
principle. The energy roster above is **23 candidates**, of which several are
marked `TBD` or `Excluded` — and the final count should emerge from Phase 14
separation of concerns, not from this list.

---

## 3. Integration points already fixed by Phase 2

The brief asks which domain object feeds which agent. These are now concrete:

```text
EnergyState          → Data Quality, Anomaly, Asset Health, Flexibility, Constraint Agents
ObservationRecord    → Data Quality Agent
Forecast             → Demand Forecast, Renewable Forecast, Forecast Performance Agents
Forecast.uncertainty → Decision Assurance Agent      (container exists; method is Phase 8)
Asset.availability   → Asset Health Agent
Asset.authority      → Decision Assurance Agent      (cannot dispatch what you do not control)
Action               → Red Team, Optimization, Simulation Agents
Constraint           → Constraint, Decision Assurance Agents
Objective            → Performance Agent
Provenance           → Evidence Retrieval Agent
StateOrigin          → Simulation Agent             (observed vs simulated must differ)
```

---

## 4. Future agent contract (documented, **not implemented**)

SAT-SA's advertised contract — `observe(context) → Observation` with severity,
confidence, scope, subjects, evidence references, rationale, statistics,
threshold, limitations, recommended action, analysis period and provenance — is
**declared in that repository's README but has no implementation**. The richest
*actually implemented* version is `satsa/domain/evidence.py` `Finding`.

The energy equivalent, when Phase 14 defines it, should combine the two proven
sources:

```text
EnergyObservation (proposed — Phase 14, NOT implemented in Phase 2)
├── agent_id, agent_version          # from Guard AgentOutput + SAT-SA detector_version
├── severity                         # from SAT-SA Finding.severity
├── confidence                       # multi-dimensional; NOT a bare float (R3)
├── scope, subjects                  # asset_ids / node_ids from the Phase 2 domain
├── state                            # signal | no_signal | insufficient_data |
│                                    #   not_applicable | error      (R2, mandatory)
├── rationale, statistics, threshold # from SAT-SA Finding
├── limitations                      # mandatory, from SAT-SA Finding
├── recommended_action               # MUST be advisory; firewall-gated (R6)
├── evidence_refs                    # mandatory on any positive claim (R4)
│                                    #   → Phase 2 Provenance objects
└── provenance, analysis_period
```

Hard rules carried over from both sources:

1. `state` **must** distinguish `insufficient_data` from `no_signal`
   (R2). An energy agent that had no telemetry must not report "no anomaly".
2. Any `signal` **must** cite evidence (R4), and the evidence must resolve to a
   Phase 2 `Provenance`.
3. `recommended_action` **must** pass the capability firewall (R6). Agents advise;
   deterministic governance acts. An agent may never promote a model, alter a
   policy or execute a physical action.
4. A confidence **vector**, not a scalar (R3), with `None` meaning "not
   applicable" rather than zero.
5. Every failure path **escalates** rather than passing silently (R5).
6. Agent versions are recorded so a result can be tied to the code that produced
   it (R10).
