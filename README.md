# Energy Intelligence

**Adaptive, self-verifying energy management** — Yuva Yodha Energy Tech Hackathon 2026, Schneider Electric.

The project aims to answer one question:

> Can an autonomous energy-management system determine what action should be taken, understand how uncertain that decision is, attempt to break its own decision, simulate the consequences before execution, and only execute the decision when sufficient evidence supports it?

The system philosophy is a closed loop:

```text
OBSERVE → UNDERSTAND → PREDICT → GENERATE OPTIONS → ATTACK OPTIONS
        → SIMULATE OPTIONS → VERIFY → DECIDE → EXECUTE → OBSERVE RESULT → LEARN
```

---

## Current status: Phase 2 of 20 — ENERGY SYSTEM DEFINITION + DATA MODEL

**Phase 1 (foundation) is complete. Phase 2 (the energy contract) is complete.
There is still no AI.** No model, no mixture-of-experts, no optimizer, no digital
twin, no red-team engine, no MLOps agent and no UI exist. Those belong to later
phases.

| Phase | Capability | Status |
|---|---|---|
| 1 | Foundation / scaffolding | **Complete** — config, logging, CLI, health check, tests |
| 2 | Energy-system definition + data model | **Complete** — the formal contract below |
| 3 | Dataset ingestion + preprocessing | Not started |
| 4 | Baseline forecasting models | Not started |
| 5 | Energy World Model | Not started |
| 6 | Qwen3-1.7B domain specialization | Not started |
| 7 | Energy-MoE | Not started |
| 8 | Uncertainty estimation | Container only; method TBD |
| 9 | Flexibility modelling | Representation only; method TBD |
| 10 | Optimization / decision engine | Not started |
| 11 | Digital twin | Not started |
| 12 | Red team / adversarial scenarios | Not started |
| 13 | Decision assurance + adaptive autonomy | Not started |
| 14 | Energy-aware MLOps | Not started |
| 15 | Agentic orchestration + Jenkins | Not started |
| 16 | End-to-end integration | Not started |
| 17 | Evaluation / ablation studies | Not started |
| 18 | Demo scenarios | Not started |
| 19 | UI / visualization | Not started |
| 20 | Hackathon packaging | Not started |

---

## What energy system is modelled

A **renewable-integrated distribution-level environment containing distributed
energy resources**, connected to the utility grid at a point of common coupling.

```text
                        UTILITY GRID
                             │
                  DISTRIBUTION SYSTEM
        ┌────────────────────┼────────────────────┐
        ▼                    ▼                    ▼
     DEMAND              RENEWABLES              DERs
   ┌────┼────┐        ┌──────┴──────┐      ┌──────┴──────┐
   ▼    ▼    ▼        ▼             ▼      ▼             ▼
Residential Commercial Industrial  Solar   Wind       Battery
                                                 │            │
                                       Flexible      EV
                                        Loads      chargers
```

Key choices (full rationale in `decisions.md`, D-022 … D-039):

| Question | Decision |
|---|---|
| Boundary | Renewable-integrated distribution-level DER environment (D-022) |
| Topology | Node-granular and configurable; power flow deferred to Phase 11 (D-023) |
| Time base | **15-minute** native timestep, **96-step** day-ahead horizon (D-024) |
| Control | **Tiered per-asset authority**, carried on each action (D-028) |
| Dataset anchor | **SMART-DS** as a representability anchor; model stays dataset-agnostic (D-038) |

No bus count, feeder count, asset count or rating is hard-coded. A single lumped
node and a large multi-feeder system use the same types.

---

## The domain contract

`src/energy_intelligence/domain/` defines the formal contract every later phase
depends on. Nothing in it presumes a model, optimizer, simulator or agent exists.

| Concern | Type | Key guarantee |
|---|---|---|
| Action vs observation | `VariableRole` | Enforced in both directions at construction |
| Assets | `Asset` + 5 typed subclasses | Each subtype carries only its own fields |
| Authority | `AuthorityLevel` | An advisory-only asset can never be dispatched |
| Topology | `Node`, `NetworkElement`, `NetworkTopology` | All references resolve; boundary must be declared |
| State | `EnergyState` | Per-node power balance enforced; `StateOrigin` mandatory |
| Actions | `Action` | Role, unit, sign, duration and authority all validated |
| Constraints | `Constraint` | Declared and bound to real assets; **never enforced in Phase 2** |
| Objectives | `Objective` | Declared **with their conflicts**; **no weight field** |
| Uncertainty | `UncertaintyEstimate` | Container ready; exactly one numeric summary; `UNKNOWN` legal |
| Provenance | `Provenance` | Mandatory everywhere; cannot be constructed without a source |
| Quality | `DataQuality` | Strict: any adverse flag blocks decision use |

The **per-node power balance** is the one invariant Phase 2 enforces structurally:

```text
generation_kw + storage_kw + grid_kw − load_kw + unaccounted_kw ≈ 0   (per node)
```

Two design choices are load-bearing and deliberately *not* implemented early:
constraints are **declared but never enforced** (a formulation would prejudge the
Phase 10 optimizer), and objectives are **never weighted** (choosing weights would
hide the cost-versus-comfort trade-off inside an unauditable scalar).

---

## Requirements

* Python **3.12** (pinned in `.python-version`; enforced by `requires-python`)
* [`uv`](https://docs.astral.sh/uv/) for environment management

Python 3.12 rather than 3.13 because Phases 6-7 target
`Qwen3-1.7B-Base` customisation and a custom MoE, where quantization and kernel
wheels (`bitsandbytes`, `flash-attn`, `xformers`, `deepspeed`) historically lag
the newest Python. See `decisions.md` → D-002.

## Setup

```powershell
uv venv --python 3.12
uv sync --extra dev
```

## Verification

```powershell
# Full health check (exit code 0 when healthy, 1 otherwise)
uv run python -m energy_intelligence health

# Or, after activating the environment:
energy-intel health

# Test suite
uv run pytest

# Run configuration (Phase 1)
uv run python -m energy_intelligence config validate

# Energy-domain configuration and contract (Phase 2)
uv run python -m energy_intelligence domain validate
uv run python -m energy_intelligence domain vocabulary
```

## Commands

| Command | Purpose |
|---|---|
| `energy-intel health` | Run every health check (Phase 1 + Phase 2) |
| `energy-intel health --json` | Same, machine-readable |
| `energy-intel health --create-dirs` | Create missing directories first |
| `energy-intel config show` | Print resolved run configuration as JSON |
| `energy-intel config validate` | Validate run configuration, non-zero exit if invalid |
| `energy-intel domain show` | Print resolved energy-domain configuration |
| `energy-intel domain validate` | Validate the energy-domain configuration |
| `energy-intel domain vocabulary` | Print the whole domain vocabulary from the enums |
| `energy-intel init-dirs` | Create the standard directory layout |
| `energy-intel paths` | Print the standard directory layout |

Global flags: `--config PATH`, `--log-level LEVEL`, `--experiment-name NAME`,
`--seed INT`, `--domain-config PATH`. A run-config path can also come from
`$ENERGY_INTEL_CONFIG`.

## Layout

```text
.
├── src/energy_intelligence/
│   ├── config/
│   │   ├── domain.py       Phase 2 energy-domain configuration
│   │   ├── loader.py       project root discovery, run-config loading
│   │   └── schema.py       run configuration (Phase 1)
│   ├── domain/             Phase 2: the formal energy contract
│   │   ├── actions.py      Action space + authority enforcement
│   │   ├── assets.py       typed asset hierarchy
│   │   ├── constraints.py  declared constraint categories
│   │   ├── enums.py        the controlled vocabulary
│   │   ├── errors.py       DomainValidationError
│   │   ├── forecasts.py    Forecast + uncertainty attachment
│   │   ├── identifiers.py  typed identifiers
│   │   ├── objectives.py   objective categories + conflicts
│   │   ├── observations.py ObservationRecord + role enforcement
│   │   ├── provenance.py   mandatory traceability
│   │   ├── quality.py      data-quality flags
│   │   ├── quantities.py   Quantity(value, unit)
│   │   ├── serialization.py deterministic versioned JSON
│   │   ├── state.py        EnergyState + per-node power balance
│   │   ├── system.py       EnergySystem + referential integrity
│   │   ├── timebase.py     15-min grid arithmetic
│   │   ├── topology.py     Node / NetworkElement / NetworkTopology
│   │   └── uncertainty.py  UncertaintyEstimate
│   ├── cli.py              command line entry point
│   ├── health.py           environment health check
│   ├── logging_setup.py    structured JSON Lines logging
│   └── paths.py            canonical filesystem layout
├── tests/
│   ├── domain/             Phase 2 domain tests
│   └── *.py                Phase 1 tests
├── configs/
│   ├── default.toml        run configuration
│   ├── domain.toml         energy-domain configuration
│   └── test.toml
├── data/{raw,processed,external}/   dataset conventions (empty in Phase 2)
├── models/                model checkpoints (empty in Phase 2)
├── experiments/           per-run outputs (empty in Phase 2)
├── artifacts/             machine-readable run outputs
├── logs/                  JSON Lines logs (git-ignored)
└── docs/
    ├── architecture.md              target architecture, component boundaries
    ├── energy_system_spec.md        the Phase 2 contract in full
    └── agent_responsibility_matrix.md  future agent responsibilities
```

Domain subpackages for later phases (`forecasting/`, `moe/`, `optimization/`,
`simulation/`, `agents/`, `mlops/`, `data/`) are **deliberately not created**
until their phase implements them. An empty package would imply capability that
does not exist.

## Configuration

Two independent configuration files:

* `configs/default.toml` — **a run**: seed, device, paths, experiment name,
  logging level. Loaded by `load_config()`.
* `configs/domain.toml` — **the environment under study**: time base, system
  identity and boundary, dataset anchor. Loaded by `load_domain_config()`.

They are separate on purpose (D-037): a run config describes how to execute, a
domain config describes what is being studied. Both reject unknown keys, so a
typo fails fast instead of being silently ignored. All paths resolve to absolute
paths against the project root. Every config object is a frozen dataclass, so it
cannot be mutated mid-run and silently break reproducibility.

## Dependencies

Runtime dependencies: **none**, in both phases. Dev extras: `pytest`,
`pytest-cov`. The scientific/ML stack is not committed until a phase actually
needs it. See `decisions.md` → D-004 and D-005.

## Documentation

| File | Contents |
|---|---|
| `decisions.md` | Every architectural decision D-001 … D-039, with alternatives and consequences |
| `flow.md` | How the system actually works, by real function and file name |
| `docs/energy_system_spec.md` | The Phase 2 contract: state, actions, constraints, objectives, time, topology, provenance, quality, uncertainty |
| `docs/agent_responsibility_matrix.md` | Future agent responsibilities, sourced from the two reference repositories |
| `docs/architecture.md` | Target architecture and component responsibility boundaries |
| `handoff.md` | Current state, what is done and not done, next phase, critical context |


## License

Proprietary. All rights reserved.
