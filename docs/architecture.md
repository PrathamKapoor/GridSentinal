# Architecture

## Intended system

The project targets an adaptive, self-verifying energy management system. Its
conceptual architecture:

```text
              PHYSICAL / SIMULATED ENERGY SYSTEM
                            │
                            ▼
                     DATA / TELEMETRY
                            │
                            ▼
                   DATA QUALITY ENGINE
                            │
          ┌─────────────────┴─────────────────┐
          ▼                                   ▼
   ANOMALY DETECTION                   DATA VALIDATION
          │                                   │
          └─────────────────┬─────────────────┘
                            ▼
                    ENERGY WORLD MODEL
                            │
              ┌─────────────┴─────────────┐
              ▼                           ▼
        FORECASTING                  REGIME
       /TRAJECTORIES                 DETECTION
              │                           │
              └─────────────┬─────────────┘
                            ▼
                    ENERGY MoE MODEL
                            │
                    UNCERTAINTY MODEL
                            │
                    FLEXIBILITY ENGINE
                            │
                      OPTIMIZATION
                            │
                    CANDIDATE ACTION
                            │
          ┌─────────────────┴─────────────────┐
          ▼                                   ▼
    RED TEAM ENGINE                       SCENARIOS
          │                                   │
          └─────────────────┬─────────────────┘
                            ▼
                      DIGITAL TWIN
                            │
                    PHYSICS VALIDATION
                            │
                    SAFETY / ASSURANCE
                            │
          ┌─────────────────┴─────────────────┐
          ▼                                   ▼
       EXECUTE                              REJECT
          │                                   │
          ▼                                   ▼
   ENERGY SYSTEM                       RE-OPTIMIZE
          │
          ▼
       OUTCOME
          │
          ▼
   PERFORMANCE DATA
          │
          ▼
    ENERGY-AWARE MLOps
          │
          ▼
     MODEL EVALUATION
          │
          ▼
    DEPLOYMENT GATE
          │
          ▼
       UPDATE
          └───────────────► LOOP
```

The system philosophy, as a closed loop:

```text
OBSERVE → UNDERSTAND → PREDICT → GENERATE OPTIONS → ATTACK OPTIONS
        → SIMULATE OPTIONS → VERIFY → DECIDE → EXECUTE → OBSERVE RESULT → LEARN
```

The distinguishing property: **the system does not merely produce predictions, it
converts predictions into verified, risk-aware energy decisions.**

## Current implementation: Phase 1

None of the components above exist yet. Phase 1 implements only the foundation
they will be built on:

```text
cli.main()
   │
   ├── health.check_health()
   │      ├── check_python_version()
   │      ├── check_package_import()
   │      ├── check_config()  → config.load_config()
   │      │                          → config.coerce_app_config()  → AppConfig
   │      ├── check_directories()   → paths.ProjectPaths
   │      ├── check_test_infrastructure()
   │      └── check_logging()       → logging_setup.initialize_logging()
   │
   ├── config.load_config() / coerce_app_config() / validate_app_config()
   ├── logging_setup.initialize_logging() / get_logger() / reset_logging()
   ├── paths.ProjectPaths
   └── health.render_report()
```

There is no energy-domain code in the repository. This is deliberate: Phase 1
exists so later components can be added cleanly, not so the pipeline can appear
to work. See `decisions.md` → D-019.

## Component responsibility boundaries

These distinctions are architectural commitments, not implementation detail. They
are recorded now so that later phases do not merge responsibilities merely
because several of them use AI.

| Component | Responsibility | Explicitly not its job |
|---|---|---|
| Energy World Model | Learned representation of system behaviour over time; "if state X and action Y, what happens next?" | Not the simulator. Not the optimizer. |
| Energy MoE | Regime-specialised experts inside the learned model, routed by energy-system state | Not agent orchestration. |
| Agents | Lifecycle and orchestration: drift, retraining, red team, deployment | Not the learned model. Not a substitute for MoE. |
| Edge (~100M) model | Fast local inference, anomaly detection, short-horizon prediction | Not automatically the same model as the central (~1.7B) model. Do not merge without an explicit architectural decision. |
| Optimizer | Convert state + forecasts + uncertainty + flexibility + constraints into an action | Not the assurance mechanism. |
| Red Team | Determine under what conditions a proposed decision becomes unsafe, ineffective or suboptimal | Not the digital twin. |
| Digital Twin | Independent simulation of a proposed action; produces a simulated outcome to compare against the learned model's prediction | Not a decorative 3D visualisation. |
| Decision Assurance | Gate between proposal and execution; approve or reject | Not the place for a final scoring formula until Phase 13 defines one. |

### MoE versus agents

These are frequently conflated and must not be.

```text
                 ENERGY MODEL
                     │
              ┌──────┴──────┐
              ↓             ↓
          Base Model       MoE
              │              │
              └──────┬───────┘
                     ▼
             Energy prediction


                  MLOps
                   │
            Supervisor Agent
                   │
      ┌────────────┼────────────┐
      ↓            ↓            ↓
 Drift Agent  Training     Red Team
                 Agent        Agent
```

The MoE is part of the **learned model**. Agents are part of **system
orchestration and lifecycle management**. They are separate layers.

## Phase map

| Phase | Scope | Status |
|---|---|---|
| 1 | Foundation / scaffolding | **Complete** |
| 2 | Energy-system definition + data model | Not started |
| 3 | Dataset ingestion + preprocessing | Not started |
| 4 | Baseline forecasting models | Not started |
| 5 | Energy World Model | Not started |
| 6 | Qwen3-1.7B domain specialization | Not started |
| 7 | Energy-MoE | Not started |
| 8 | Uncertainty estimation | Not started |
| 9 | Flexibility modelling | Not started |
| 10 | Optimization / decision engine | Not started |
| 11 | Digital twin | Not started |
| 12 | Red Team / adversarial scenarios | Not started |
| 13 | Decision assurance + adaptive autonomy | Not started |
| 14 | Energy-aware MLOps | Not started |
| 15 | Agentic orchestration + Jenkins | Not started |
| 16 | End-to-end integration | Not started |
| 17 | Evaluation / ablation studies | Not started |
| 18 | Demo scenarios | Not started |
| 19 | UI / visualization | Not started |
| 20 | Hackathon packaging | Not started |

Phase numbers may change if implementation reveals a better dependency order.
Any such change must be recorded in `decisions.md`.

## Planned target applications

Grid reliability · renewable-energy integration · energy optimization ·
distributed energy resources · demand/production forecasting ·
battery/storage management · energy flexibility · resilience · AI/ML ·
agentic MLOps · simulation/digital twins · sustainability.

## Open architectural questions

Per the project's "no assumptions" rule, these are `TBD`, not decided:

* Which energy system is modelled. Phase 2 decides this, and it determines the
  data model, the optimizer formulation and the digital twin's scope.
* Which energy-domain regimes deserve dedicated experts. Current candidates —
  normal operation, renewable ramp, peak demand, abnormal conditions — are
  hypotheses, **not** confirmed expert definitions, and must not be hard-coded.
* How many experts, and what routing objective. Phase 7.
* Whether the optimizer is MPC, LP, MILP, NLP or RL. Phase 10, and the choice
  must follow from the problem formulation defined in Phase 2 — not the reverse.
* How model–physics disagreement translates into reduced confidence. A research
  hypothesis only; no formula or threshold exists yet.
* The decision-assurance scoring function. Phase 13.
* Whether the ~100M edge model and the ~1.7B central model share any weights.
  Phase 5 or 6 must decide explicitly.
