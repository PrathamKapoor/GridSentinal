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

## Current status: Phase 1 of 20 — FOUNDATION AND PROJECT SCAFFOLDING

**This repository currently contains no AI.** There is no model, no
mixture-of-experts, no optimizer, no digital twin, no red-team engine and no
MLOps agent. Those belong to later phases.

What exists today is the foundation those components will be built on:
configuration, structured logging, filesystem conventions, tests, a health
check, and documentation.

| Capability | Phase | Status |
|---|---|---|
| Foundation / scaffolding | 1 | **In progress** |
| Energy-system definition + data model | 2 | Not started |
| Dataset ingestion + preprocessing | 3 | Not started |
| Baseline forecasting models | 4 | Not started |
| Energy World Model | 5 | Not started |
| Qwen3-1.7B domain specialization | 6 | Not started |
| Energy-MoE | 7 | Not started |
| Uncertainty estimation | 8 | Not started |
| Flexibility modelling | 9 | Not started |
| Optimization / decision engine | 10 | Not started |
| Digital twin | 11 | Not started |
| Red team / adversarial scenarios | 12 | Not started |
| Decision assurance + adaptive autonomy | 13 | Not started |
| Energy-aware MLOps | 14 | Not started |
| Agentic orchestration + Jenkins | 15 | Not started |
| End-to-end integration | 16 | Not started |
| Evaluation / ablation studies | 17 | Not started |
| Demo scenarios | 18 | Not started |
| UI / visualization | 19 | Not started |
| Hackathon packaging | 20 | Not started |

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
# Full Phase 1 health check (exit code 0 when healthy, 1 otherwise)
uv run python -m energy_intelligence health

# Or, after activating the environment:
energy-intel health

# Test suite
uv run pytest

# Configuration
uv run python -m energy_intelligence config validate
uv run python -m energy_intelligence config show
```

## Commands

| Command | Purpose |
|---|---|
| `energy-intel health` | Run every Phase 1 health check |
| `energy-intel health --json` | Same, machine-readable |
| `energy-intel health --create-dirs` | Create missing directories first |
| `energy-intel config show` | Print resolved configuration as JSON |
| `energy-intel config validate` | Validate configuration, non-zero exit if invalid |
| `energy-intel init-dirs` | Create the standard directory layout |
| `energy-intel paths` | Print the standard directory layout |

Global flags: `--config PATH`, `--log-level LEVEL`, `--experiment-name NAME`,
`--seed INT`. A config path can also come from `$ENERGY_INTEL_CONFIG`.

## Layout

```text
.
├── src/energy_intelligence/
│   ├── config/            validated, immutable configuration (dataclasses + tomllib)
│   ├── cli.py             command line entry point
│   ├── health.py          Phase 1 environment health check
│   ├── logging_setup.py   structured JSON Lines logging
│   └── paths.py           canonical filesystem layout
├── tests/                 pytest suite (configuration, logging, health, layout)
├── configs/               versioned TOML configuration
├── data/{raw,processed,external}/   dataset conventions (empty in Phase 1)
├── models/                model checkpoints (empty in Phase 1)
├── experiments/           per-run outputs (empty in Phase 1)
├── artifacts/             machine-readable run outputs
├── logs/                  JSON Lines logs (git-ignored)
└── docs/                  extended documentation
```

Domain subpackages (`data/`, `models/`, `forecasting/`, `moe/`,
`optimization/`, `simulation/`, `assurance/`, `agents/`, `mlops/`) are
**deliberately not created** until their phase implements them. An empty package
would imply capability that does not exist.

## Configuration

`configs/default.toml` is the base; a run may layer a file on top via
`--config`, or override individual keys via CLI flags. Unknown keys are
rejected so typos fail fast rather than being silently ignored. All paths are
resolved to absolute paths against the project root. `AppConfig` is a frozen
dataclass — a config cannot be mutated mid-run, which would silently break
reproducibility.

## Dependencies

Runtime dependencies: **none**. Dev extras: `pytest`, `pytest-cov`. The
scientific/ML stack is not committed until a phase actually needs it. See
`decisions.md` → D-004 and D-005.

## Documentation

| File | Contents |
|---|---|
| `decisions.md` | Every architectural and implementation decision, with alternatives and consequences |
| `flow.md` | How the Phase 1 system actually works, by function and file name |
| `handoff.md` | Current state, what is done and not done, next phase, critical context |
| `docs/` | Extended documentation as it accumulates |

## License

Proprietary. All rights reserved.
