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

## Current status: Phase 5 of 20 — ENERGY DEMAND DYNAMICS MODEL

**Phases 1-4 complete. Phase 5 complete — and its answer is mixed, not a win.** A
learned temporal model beats the classical baseline at two of three horizons and loses
at the third. No Energy World Model, no Qwen, no Energy-MoE, no optimiser, no digital
twin, no agent, no UI exists yet, by design — and `docs/world_model_requirements.md`
records why an action-conditioned model is not reachable from this dataset.

| Phase | Capability | Status |
|---|---|---|
| 1 | Foundation / scaffolding | **Complete** |
| 2 | Energy-system definition + data model | **Complete** |
| 3 | Dataset ingestion, normalization, reality validation | **Complete — PARTIAL fidelity** |
| 4 | ML dataset + naive/classical/neural baselines | **Complete — measured** |
| 5 | Energy Demand Dynamics Model (temporal) | **Complete — mixed result** |
| 6 | Qwen3-1.7B domain specialization | Not started — bar defined in `docs/qwen_energy_requirements.md` |
| 7 | Energy-MoE | Not started — requirements in `docs/moe_design_requirements.md` |
| 8 | Uncertainty estimation | Container only; method TBD |
| 9 | Flexibility modelling | Representation only; **battery dispatch unavailable (G-01)** |
| 10 | Optimization / decision engine | Not started |
| 11 | Digital twin | Not started; needs a loss model (G-06) and produced state |
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

## The Phase 5 answer

> Does the **order** of a customer's demand history carry information that a flat
> vector of lags discards, and does a learned temporal model beat the classical
> baseline on the same rows?

**Partly. Two horizons of three, yes; the 24-hour horizon, no.**

A 74,099-parameter dilated causal TCN was trained on one week of ordered context and
compared against Phase 4's baselines on the **identical** `179,520` chronological test
rows. kW MAE, lower is better:

| Model | h=1 (15 min) | h=4 (1 h) | h=96 (24 h) |
|---|---|---|---|
| persistence | **0.4043** | 0.8897 | 2.4343 |
| Phase 4 classical GBM | 0.4242 | 0.8579 | **1.5648** |
| **Phase 5 temporal TCN** | 0.4178 | **0.8307** | 1.6510 |
| Phase 5 vs GBM | **+1.5%** | **+3.2%** | **-5.5%** |

### Three findings that change Phase 6

1. **The delta parameterisation is the single biggest design effect.** Predicting the
   change from `y(t)` instead of the level is worth **+253% / +52% / +20%** at the
   three horizons. Persistence is within 3% of the best 15-minute result, so most of
   the short-horizon difficulty is not in the history at all.
2. **Temporal order helps, but not everywhere.** The TCN beats gradient boosting at
   15 min and 1 h and loses at 24 h. A future model must be judged per horizon; a
   single average hides the one place it has to win.
3. **Attention did not beat convolution on this data.** A 112k-parameter patch
   Transformer was 49.8% worse at 15 min at a matched budget. That is the evidence
   base for `docs/qwen_energy_requirements.md`, and it does not support assuming a
   larger model will help.

### Two things Phase 5 got wrong, on the record

* The first shipped regime table labelled **h=1** regimes with **h=96** errors, because
  the per-horizon metrics loop overwrote the error arrays. Headline metrics were
  unaffected, so nothing else would have caught it. Analysis now runs once per horizon
  with the horizon recorded, two regression tests assert the invariant, and a
  checkpoint replay path re-derives the analysis and **fails if the recorded metrics
  do not reproduce** (D-078).
* The `no_cyclic` ablation was **better** than the shipped configuration at all three
  horizons. The channels stayed, because dropping them after seeing the test result
  would be tuning against the test split. So the shipped configuration is **not the
  best this study found**, and Phase 6 should start by testing that at full budget
  (D-077).

**Not claimed:** any statistical significance (one seed, no confidence intervals), any
action-conditioned capability, and any world-model behaviour.

---

## The Phase 4 answer

> Can real SMART-DS data become a temporally correct, leakage-free ML dataset, and
> how hard is forecasting before any custom model exists?

**YES — and the baselines are now measured, so every future claim has something to
beat.**

### What could be forecast at all

Of six candidate targets, three are real and four absences are recorded rather than
filled:

| Target | Status | Basis |
|---|---|---|
| Per-customer load | **SUPPORTED** | 1,871 customers x 35,040 points, no missing values |
| Feeder load | **SUPPORTED** | sum over 3,690 Load objects |
| PV generation | **PARTIALLY_SUPPORTED** | one real 1000 kW array; **not** the feeder's 1,216 PV (G-03) |
| Net load, battery SOC/power, EV demand, wind | **UNSUPPORTED** | no series exists; see `docs/ml_data_gap_report.md` |

### Measured baselines — test split, chronological, never shuffled

Load task, 40 customers, 1,309,440 samples, MAE in kW:

| Model | h=1 (15 min) | h=4 (1 h) | h=96 (24 h) | Train time |
|---|---|---|---|---|
| naive last value | **0.4043** | 0.8897 | 2.4343 | — |
| naive seasonal (day) | 2.3367 | 2.3375 | 2.4343 | — |
| naive seasonal (week) | 1.8673 | 1.8638 | 1.8306 | — |
| classical ridge | 0.5193 | 1.0878 | 2.1920 | 0.5 s |
| **classical hist GBM** | 0.4242 | **0.8579** | **1.5648** | 20.5 s |
| neural MLP | 0.4638 | 0.9315 | 6.4850 | 389 s |
| neural GRU | 0.5691 | 0.8976 | 2.2856 | 1233 s |

PV task, 15-minute nowcast with real weather at the origin, MAE in kW on a 1000 kW
array: **hist GBM 8.57** (0.86% of capacity), ridge 10.39, MLP 9.90, persistence
13.58, seasonal-naive 79-85.

### Six findings that change the plan

1. **Persistence wins at 15 minutes.** Household load on a 15-minute grid is so
   smooth that repeating the last value beats every fitted model. Machine learning
   earns its keep only as the horizon grows.
2. **Gradient boosting wins at 1 h and 24 h** (0.8579 and 1.5648 kW against
   persistence's 0.8897 and 2.4343). Classical ML is the right default for this data.
3. **The neural baseline does not beat classical here.** The MLP collapses at 24 hours
   (6.485 kW, worse than persistence) and the GRU costs 3x the MLP's time for no gain.
   Untrained-but-documented baselines, per D-056.
4. **Regime structure is real, and measured.** Worst-to-best regime error ratios run
   12.7-18.5 on the load task and 15.8 across solar phases. Error concentrates in
   high-demand and high-ramp periods, and in the evening/morning PV ramps. This is
   the first data-driven argument for regime-specialised modelling.
5. **Day-ahead PV is a missing-input problem, not a modelling problem.** Without a
   weather forecast, PV at 24 hours reaches MAE 74.3 kW against persistence's 79.6 kW
   (R2 0.62) — barely better than doing nothing. With weather at the origin it is
   8.57 kW. The dataset has no weather *forecast*, so Phase 4 refuses to use weather
   beyond 15 minutes, structurally.
6. **Percentage metrics are wrong for PV.** sMAPE reads 126% while MAE is 0.86% of
   capacity, because 52.2% of the target is exactly zero. MAPE is never used.

Full analysis: `artifacts/ml/*/reports/` and `docs/ml_data_gap_report.md`.

---

## The Phase 3 answer---

## The Phase 3 answer

> Can a real SMART-DS instance be transformed into our `EnergySystem` without
> inventing information or silently changing its meaning?

**YES, WITH EXPLICIT DERIVATIONS** — and with the balance criterion failing,
reported rather than worked around.

| Question | Answer | Evidence |
|---|---|---|
| **Dataset** | SMART-DS **v1.0**, AUS/P1U, 2018, `base_timeseries`, feeder `p1uhs0_1247--p1udt12703` | OEDI public S3 bucket; `SMART-DS_version.txt` = `1.0` |
| **Provenance** | 359 files, 228.5 MiB, every file SHA-256'd | `docs/smart_ds_acquisition.md` |
| **Time** | **Native 15-minute, 35,040 points** — matches the Phase 2 grid exactly, **no resampling** | Confirmed 3 independent ways |
| **Assets** | 3,690 load objects / **1,871 individual customer assets**, 4,555 line edges, 5,196 nodes, 1,216 PV, 93 batteries | Real files |
| **Topology** | Full node graph constructible from `Lines.dss` | Real `bus1`/`bus2` |
| **Units** | Known and normalized; no inference from magnitude | User Guide + observed magnitudes |
| **Quality** | Residential demand reconstructs to **8 decimal places**; commercial is **−20.45 kW** short | vs dataset's own published figures |
| **Derivation** | 1 derivation: `P(t) = Σ kW_rating × profile_pu(t)` | Verified against published values |
| **Battery** | Dispatch **UNAVAILABLE and not derivable**; usable energy and directional limits **UNKNOWN** | No SOC series; `kWhRated` is a nameplate only (D-050) |
| **Domain mapping** | A real `EnergySystem` instantiates: `sys-smartds-aus-p1u` | 5,196 nodes, 1,871 real load assets, 2 observations, mandatory provenance |
| **Balance** | **FAILS the 0.5 kW criterion** — 2 of 5 identities | Reported, tolerance unchanged |

### Three findings that change the project plan

1. **Battery dispatch does not exist and cannot be derived** (G-01). Phase 2's
   handoff said dispatch "may need to be derived from state of charge".
   Verification shows that route is *also* unavailable — deriving power needs
   SOC(t) and SOC(t+1), and SMART-DS supplies a single `kWhStored` scalar. No
   battery flexibility can come from this dataset. This **corrects** the Phase 2
   document.

2. **Per-node power balance cannot be evaluated over time** (G-02). Grid
   import/export is not provided as a time series, so there is nothing to close
   against. Phase 11 must *produce* flows, not read them.

3. **The 0.5 kW tolerance cannot close while network losses are unmodelled**
   (G-06). Real feeders lose **3.2175 %** (501.67 kW at 15,090.72 kW). Phase 2
   deferred physics, so this is a physics gap. The tolerance was **not**
   adjusted.

Full detail: `docs/gaps_report.md` — 12 gaps, with 4 fixable by better ingestion.

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

# SMART-DS ingestion (Phase 3) - requires the dataset under data/raw/smart_ds
uv run python -m energy_intelligence data inspect
uv run python -m energy_intelligence data ingest
uv run python -m energy_intelligence data validate   # exits 1: balance FAILS
uv run python -m energy_intelligence data mapping

# ML dataset and baselines (Phase 4) - requires the dataset under data/raw/smart_ds
uv run python -m energy_intelligence ml targets      # what can be forecast, and why
uv run python -m energy_intelligence ml features     # feature catalogue + leakage policy
uv run python -m energy_intelligence ml dataset      # build the dataset, no training
uv run python -m energy_intelligence ml run          # full experiment (~37 min, CPU)
uv run python -m energy_intelligence ml experiments  # the recorded registry

# the two PV experiments (see the configs for why they differ)
uv run python -m energy_intelligence ml --ml-config configs/ml_pv_nowcast.toml run
uv run python -m energy_intelligence ml --ml-config configs/ml_pv_horizon.toml run
```

## Commands

| Command | Purpose |
|---|---|
| `energy-intel health` | Run every health check (Phases 1-3) |
| `energy-intel health --json` | Same, machine-readable |
| `energy-intel health --create-dirs` | Create missing directories first |
| `energy-intel config show` / `validate` | Run configuration |
| `energy-intel domain show` / `validate` | Energy-domain configuration |
| `energy-intel domain vocabulary` | The whole domain vocabulary from the enums |
| `energy-intel data config` | Resolved dataset selection |
| `energy-intel data inspect` | Schema, assets and data quality, no mapping |
| `energy-intel data ingest` | Full pipeline; writes all artifacts |
| `energy-intel data validate` | Balance report; **exit 1 when it fails** |
| `energy-intel data mapping` | SMART-DS to domain field mapping |
| `energy-intel data manifest` | Manifest summary with content digest |
| `energy-intel ml config` | Resolved experiment configuration |
| `energy-intel ml targets` | Every forecasting target with support status and evidence |
| `energy-intel ml features` | Feature catalogue with availability and leakage policy |
| `energy-intel ml dataset` | Build the ML dataset and its manifest, without training |
| `energy-intel ml run` | Dataset + all baselines + regime and error analysis + registry |
| `energy-intel ml experiments` | The experiment registry |
| `energy-intel temporal config` | Resolved temporal experiment configuration |
| `energy-intel temporal run` | Train one temporal model, evaluate once, analyse, record |
| `energy-intel temporal ablate` | The ablation suite and its measured table |
| `energy-intel temporal experiments` | The temporal experiment registry |
| `energy-intel init-dirs` / `paths` | Directory layout |

Global flags: `--config`, `--log-level`, `--experiment-name`, `--seed`.
Sub-command flags: `--domain-config` (domain), `--data-config` (data), `--ml-config` (ml), `--out` (artifact directory).

## Layout

```text
.
├── src/energy_intelligence/
│   ├── config/
│   │   ├── data.py         Phase 3 dataset selection
│   │   ├── domain.py       Phase 2 energy-domain configuration
│   │   ├── loader.py       project root discovery, run-config loading
│   │   ├── ml.py           Phase 4 experiment shape
│   │   └── schema.py       run configuration (Phase 1)
│   ├── ml/                 Phase 4 ML dataset and baselines
│   │   ├── targets.py          which quantities can be forecast, and which cannot
│   │   ├── features.py         feature catalogue + leakage guard
│   │   ├── splits.py           chronological splits, no window crosses a boundary
│   │   ├── dataset.py          windowing, schema, versioning, artifact
│   │   ├── scaling.py          train-only preprocessing statistics
│   │   ├── metrics.py          generic + energy-specific error measures
│   │   ├── analysis.py         regime and error analysis
│   │   ├── baselines/          naive.py, classical.py, neural.py
│   │   ├── experiment.py       one experiment end to end
│   │   ├── registry.py         reproducible experiment records
│   │   ├── runner.py           artifact production
│   │   └── pipeline.py         config to real data bridge
│   ├── data/               Phase 3 ingestion
│   │   ├── manifest.py     checksummed dataset manifest
│   │   └── smartds/        the only layer that knows SMART-DS
│   │       ├── adapter.py       facade: parse + normalise + quality
│   │       ├── balance.py       energy balance validation (fixed tolerance)
│   │       ├── buscoords.py     bare coordinate-list format
│   │       ├── domain_mapper.py normalised records -> Phase 2 objects
│   │       ├── dss.py           generic OpenDSS parser (dataset-agnostic)
│   │       ├── layout.py        dataset paths
│   │       ├── mapping.py       semantic field mapping + confidence
│   │       ├── normalize.py     per-unit -> kW, index -> timestamp
│   │       ├── pipeline.py      end-to-end orchestration
│   │       └── schema.py        programmatic schema discovery
│   ├── domain/             Phase 2: the formal energy contract (20 modules)
│   ├── cli.py              command line entry point
│   ├── health.py           environment health check
│   ├── logging_setup.py    structured JSON Lines logging
│   └── paths.py            canonical filesystem layout
├── tests/
│   ├── data/               Phase 3 tests + fixtures extracted from real data
│   ├── domain/             Phase 2 domain tests
│   ├── ml/                 Phase 4 tests (leakage, splits, metrics, baselines)
│   └── *.py                Phase 1 tests
├── scripts/extract_fixtures.py   builds test fixtures from the real dataset
├── configs/{default,domain,data,ml,ml_pv_nowcast,ml_pv_horizon,test}.toml
├── data/raw/smart_ds/       downloaded SMART-DS subset (git-ignored)
├── artifacts/data/         Phase 3 reports (git-ignored)
├── artifacts/ml/           Phase 4 datasets and reports (git-ignored)
├── experiments/registry.jsonl   the one committed experiment record
└── docs/
    ├── architecture.md              target architecture, component boundaries
    ├── energy_system_spec.md        the Phase 2 contract in full
    ├── agent_responsibility_matrix.md  future agent responsibilities
    ├── smart_ds_acquisition.md      source, version, checksums, reproducibility
    ├── smart_ds_mapping.md          field-by-field mapping with confidence
    ├── gaps_report.md               what SMART-DS does NOT provide (domain view)
    └── ml_data_gap_report.md        what it does NOT provide (ML view)
```

Subpackages for later phases (`moe/`, `optimization/`, `simulation/`, `agents/`,
`mlops/`) are **deliberately not created** until their phase implements them. An
empty package would imply capability that does not exist. `ml/` exists because
Phase 4 implements it - and it contains baselines only, no custom model.

## Configuration

Three independent files, each a distinct concern:

* `configs/default.toml` — **a run**: seed, device, paths, experiment name, logging.
* `configs/domain.toml` — **the environment**: time base, system identity, boundary.
* `configs/data.toml` — **the dataset**: region, sub-region, year, scenario, feeder.

All three reject unknown keys, so a typo fails fast instead of being silently
ignored. All paths resolve to absolute paths against the project root. Every
config object is a frozen dataclass, so it cannot be mutated mid-run and silently
break reproducibility.

## Dependencies

Phases 1-3 had **no** runtime dependencies: configuration, logging, testing, health
checks and the entire SMART-DS ingestion pipeline run on the standard library alone.

Phase 4 adds exactly three, because it is the first phase that needs a numerical
stack (D-055): `numpy`, `scikit-learn`, and `torch` from the **CPU-only** index
(the default Windows wheel bundles CUDA and is several hundred megabytes). pandas was
deliberately **not** adopted, and a test asserts it is never imported.

A test also asserts that `energy_intelligence/domain/` imports **no** ML library, so
the Phase 2 contract stays a dependency-free vocabulary that everything else speaks.

Dev extras: `pytest`, `pytest-cov`.

## Documentation

| File | Contents |
|---|---|
| `decisions.md` | Every decision D-001 … D-079, with alternatives and consequences |
| `flow.md` | How the system actually works, by real function and file name |
| `docs/energy_system_spec.md` | The Phase 2 contract: state, actions, constraints, objectives, time, topology, provenance, quality, uncertainty |
| `docs/agent_responsibility_matrix.md` | Future agent responsibilities, sourced from two reference repositories |
| `docs/smart_ds_acquisition.md` | Source, version, checksums, subset, reproducibility |
| `docs/smart_ds_mapping.md` | Field-by-field mapping with confidence and origin |
| `docs/gaps_report.md` | 12 gaps: what SMART-DS does not provide (domain view) |
| `docs/ml_data_gap_report.md` | 15 gaps: what it does not provide for machine learning |
| `docs/world_model_requirements.md` | What an action-conditioned world model needs, and why SMART-DS cannot supply it |
| `docs/qwen_energy_requirements.md` | The bar a Qwen3-1.7B energy model must clear, derived from measured Phase 4/5 results |
| `docs/moe_design_requirements.md` | Measured regime heterogeneity, the observable-gate constraint, and which MoE designs the evidence supports |
| `docs/architecture.md` | Target architecture and component boundaries |
| `handoff.md` | Current state, what is done and not done, next phase, critical context |


## License

Proprietary. All rights reserved.
