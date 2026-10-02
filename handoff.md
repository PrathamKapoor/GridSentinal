# Handoff — Phase 4

**Project:** Energy Intelligence — adaptive, self-verifying energy management
**Event:** Yuva Yodha Energy Tech Hackathon 2026, Schneider Electric
**Current phase:** **Phase 4 of 20 — ML-ready dataset + baseline forecasting — COMPLETE**
**Previous phases:** Phase 1 foundation, Phase 2 domain contract, Phase 3 SMART-DS ingestion — **all preserved, zero regressions**
**Date:** 2026-10-03
**Author of all work:** PrathamKapoor <prathamkapoor027@gmail.com>

---

## 1. Work completed

Phase 4 answered two questions with measurements:

> Can real SMART-DS data become a temporally correct, leakage-free ML dataset?

**Yes.** And:

> How difficult is forecasting, before any custom model exists?

**Measured, and recorded below.** No Energy World Model, no Qwen, no Energy-MoE, no
optimiser, no agent and no UI was built. This was a baseline phase, and it stayed one.

### Steps 1-14 — inspection

Full re-inspection before any change: repository state, branch, identity, all four
Phase 1-3 documents, the Phase 2 specification, the SMART-DS acquisition and mapping
documents, every processed artifact under `artifacts/data/`, the Phase 3 validation
reports, all configuration files, the test suite, and the real files themselves.

The decisive inspection finding came from reading the actual PV scenario files rather
than the documentation: **the 20 irradiance loadshapes the 1,216 `PVSystem` objects
reference do not exist anywhere in the dataset** (M-04). The one `solar_data` file that
*is* present is a different location (30.3459 N, not 30.4196 N). This closes Phase 3's
G-03: the feeder PV link is not merely unresolved, it is unresolvable from this source.

### Steps 15-20 — targets, dataset, baselines

| Step | Result |
|---|---|
| Target registry | 7 candidates declared, **3 real / 4 refused** with evidence |
| Extraction | per-customer load, feeder total, PV generation — all on the native 15-minute grid, no resampling |
| Cross-phase proof | Phase 4's per-profile feeder total equals Phase 3's per-customer sum to **5e-12 kW** |
| Leakage guard | poisons the future **per sample** and proves every feature is unchanged; the guard is itself tested against a deliberately leaking feature |
| Dataset | `ds-16ff1dabe80b`, **1,309,440 samples**, 13 features, 40 series, 3 horizons |
| Splits | chronological 70/15/15; **0** samples dropped, because windows are constructed per split |
| Baselines | 4 naive, 2 classical, 2 neural — all measured, none tuned |
| Regime analysis | regimes derived from the data, ratios measured, no expert invented |
| Error analysis | worst rows with timestamps, worst series, error by hour and month |

### Steps 21-26 — test, document, commit

737 tests passing, 89% coverage.

---

## 2. Key findings

### F-01 — Persistence beats machine learning at 15 minutes

| Model | h=1 (15 min) MAE | h=4 (1 h) | h=96 (24 h) |
|---|---|---|---|
| naive last value | **0.4043 kW** | 0.8897 | 2.4343 |
| naive seasonal (week) | 1.8673 | 1.8638 | 1.8306 |
| classical ridge | 0.5193 | 1.0878 | 2.1920 |
| **classical hist GBM** | 0.4242 | **0.8579** | **1.5648** |
| neural MLP | 0.4638 | 0.9315 | 6.4850 |
| neural GRU | 0.5691 | 0.8976 | 2.2856 |

Household and commercial load on a 15-minute grid is so smooth that repeating the last
value is hard to beat at one step ahead. **Classical ML earns its keep only as the
horizon grows**, where it beats persistence by 36% at 24 hours.

### F-02 — The neural baseline does not beat classical here

The MLP collapses at 24 hours (6.4850 kW, *worse than persistence*), in **per-unit
terms too** — MAE 0.1628 against gradient boosting's 0.0481 — so it is a genuine
model limitation and not a unit-weighting artefact. The GRU costs 1233 s against the
MLP's 389 s and gains nothing.

Both were left untuned by decision (D-056): a baseline tuned until it wins is not a
baseline. This is an **untuned-baseline** result, and it bounds what an
off-the-shelf neural baseline achieves on this data — not what a tuned one could.

### F-03 — Regime structure is real, and it was measured rather than assumed

Worst-to-best regime error ratios: **12.7-18.5** on the load task, **15.8** across
solar phases. On the load task, high-demand periods cost gradient boosting 0.98 kW
against 0.09 kW in low-demand periods — an order of magnitude.

Error concentrates in **high-demand and high-ramp** periods, and in the **evening and
morning PV ramps**. This is the first data-driven support for the premise a Phase 7
Energy-MoE would rest on.

*Caveat, stated because it matters:* the ratio is unstable for rules whose error
approaches zero in some regimes — `naive_seasonal_day` reports 1860, which is an
artefact of a 0.12 kW denominator, not a finding. Read the per-regime MAEs, not the
ratio, for the naive rules.

### F-04 — Day-ahead PV is a missing-input problem

| PV configuration | h=1 | h=96 | R² at 96 h |
|---|---|---|---|
| weather at the origin (nowcast) | **8.57 kW** (0.86% of capacity) | — | — |
| no weather | 8.48 kW | 74.34 kW vs persistence 79.59 | 0.62 |

With actual weather at the origin, PV is easy. Without it, a 24-hour forecast is
**barely better than doing nothing**. The dataset contains no weather *forecast*, so
Phase 4 withdraws weather features structurally beyond 15 minutes rather than quietly
using the future.

### F-05 — Percentage metrics are actively misleading for PV

sMAPE reads **126%** while MAE is 0.86% of array capacity, because 52.2% of the target
(18,304 of 35,040 steps) is exactly zero at night. MAPE is never used anywhere in this
phase. sMAPE defines `0/0 = 0` as a perfect prediction and is bounded to `[0, 200]`.

The zero-period metric earned its keep immediately: at night the ridge model errs by
**5.99 kW** where persistence errs by 0.91 kW. A linear model cannot represent
"generation is exactly zero at night".

### F-06 — Two bugs of my own, both found by the tests and both material

1. **The row layout was wrong.** The dataset stored one row ordering while
   `build_feature_matrix` assumed the opposite, so every feature was read from the
   wrong customer. Nothing crashed; targets and features stayed *internally*
   consistent, so all metrics were silently meaningless. Correlation between the target
   and `lag_1` was **0.029** where it should be ~0.99. Caught by asking why a model
   scored worse than the mean predictor.
2. **The feature set was weaker than the baseline it was measured against.** It had
   `y[t-1]` but not `y[t]`, while persistence reads `y[t]`. Ridge scored 0.48 kW
   against persistence's 0.27 kW — a rigged comparison. Adding `value_at_origin`
   (D-064) reversed the honest result.

Both are now regression-tested. The general lesson, recorded for the next phase: a
pipeline that is internally consistent can still be entirely wrong, so cross-checks
against an *independent* implementation are not optional.

### F-07 — The Phase 3 balance failure does not affect any ML target

The 20.4506 kW residual is a **feeder-level constant at the published peak**. It is
not a per-customer quantity, so it cannot enter a per-customer target, and it is not a
per-step quantity, so it cannot enter a time series. It is **orthogonal** to this
phase's forecasting tasks.

It was not "repaired", the 0.5 kW criterion was not touched, and no data was
normalised to make it disappear. The note is written into every Phase 4 summary
artifact so the next reader cannot mistake the two findings for one.

---

## 3. Files changed

```text
NEW  src/energy_intelligence/ml/__init__.py
NEW  src/energy_intelligence/ml/targets.py       which targets exist, and which do not
NEW  src/energy_intelligence/ml/features.py      feature catalogue + leakage guard
NEW  src/energy_intelligence/ml/splits.py        chronological splits, boundary rules
NEW  src/energy_intelligence/ml/dataset.py       windowing, schema, versioning, artifact
NEW  src/energy_intelligence/ml/scaling.py       train-only preprocessing
NEW  src/energy_intelligence/ml/metrics.py       generic + energy-specific measures
NEW  src/energy_intelligence/ml/analysis.py       regime and error analysis
NEW  src/energy_intelligence/ml/baselines/{__init__,naive,classical,neural}.py
NEW  src/energy_intelligence/ml/experiment.py    one experiment, end to end
NEW  src/energy_intelligence/ml/registry.py      reproducible experiment records
NEW  src/energy_intelligence/ml/runner.py        artifact production
NEW  src/energy_intelligence/ml/pipeline.py      config-to-data bridge
NEW  src/energy_intelligence/ml/config_ml_bridge.py
NEW  src/energy_intelligence/config/ml.py        fourth configuration file
NEW  configs/ml.toml, ml_pv_nowcast.toml, ml_pv_horizon.toml
NEW  tests/ml/{conftest,test_targets,test_features,test_dataset,test_models,test_pipeline}.py
NEW  docs/ml_data_gap_report.md

EXTENDED  src/energy_intelligence/cli.py               + ml {config,targets,features,dataset,run,experiments}
EXTENDED  src/energy_intelligence/data/smartds/adapter.py   loads(names=…), profile_filename(), source_bus()
EXTENDED  src/energy_intelligence/data/smartds/normalize.py CustomerLoad.bus, .is_commercial
EXTENDED  pyproject.toml      + numpy, scikit-learn, torch (CPU index)
EXTENDED  .gitignore          + experiments/registry.jsonl exception
EXTENDED  tests/test_package.py  Phase 1 dependency guard tightened, not deleted
EXTENDED  README.md, decisions.md (D-055…D-070), flow.md
```

**Not modified:** any Phase 2 `domain/` module, the 0.5 kW balance criterion, or the
Phase 3 reconstruction arithmetic.

---

## 4. Current architecture and state

```text
SMART-DS files
   └─ ml/targets.py ────────► TargetSeries (values, scale, value_origin, provenance)
        └─ ml/dataset.py ───► MlDataset (features, targets, origin_index, series_index, split)
             ├─ ml/features.py   availability policy + leakage guard
             ├─ ml/splits.py     chronological boundaries
             └─ ml/scaling.py    train-only, asserted
                  └─ ml/experiment.py ─► ml/baselines/{naive,classical,neural}
                       └─ ml/metrics.py ─► ml/analysis.py ─► artifacts + registry
```

Artifacts (all git-ignored except the registry):

```text
artifacts/ml/load-40/         dataset ds-16ff1dabe80b (.npz + .json), reports/
artifacts/ml/pv-nowcast/      dataset ds-d381ee913098, reports/
artifacts/ml/pv-no-weather/   dataset ds-77f04fd46328, reports/
experiments/registry.jsonl    15 records: 8 models x 3 horizons, load task
                              + 6 PV nowcast + 12 PV no-weather
```

---

## 5. Decisions made

Full detail in `decisions.md` (D-055 … D-070). The ones that bind later phases:

| ID | Decision | Binding constraint |
|---|---|---|
| D-055 | Exactly three runtime dependencies; pandas rejected; **domain layer imports no ML library** | tested, not merely intended |
| D-056 | Baselines are evaluated, never tuned | the neural failure below is a real bound on an *untuned* baseline |
| D-057 | Adapter gains a customer filter; feeder total accumulated per profile | cross-phase agreement asserted to 1e-6 kW |
| D-058 | 3 targets supported, 4 refused by name with evidence | no empty datasets, no fabricated tasks |
| D-059 | Experiment shape is a **fourth** config file | changing a lookback must not edit the acquisition record |
| D-060 | Lookback 672 steps (7 days) | the shortest window containing a full weekly cycle |
| D-061 | Horizons 1/4/96 steps; chronological 70/15/15 | ramp error is `n/a` unless horizons are consecutive |
| D-062 | Stratified 40-customer sample, stride 1 | 65.6M values is a scale problem, not a baseline problem |
| D-063 | Per-unit normalisation by rated kW; **weather withdrawn beyond 15 min** | `WEATHER_AVAILABLE_THROUGH_STEP = 1`, enforced in code |
| D-064 | `value_at_origin` is a feature | the feature set may not be weaker than the baseline |
| D-065 | Flat feature matrices, not nested windows | every temporal invariant is directly assertable |
| D-066 | Split-dependent features are made at fit time, never stored | a dataset must be reusable under another split |
| D-067 | MAE/RMSE in kW, sMAPE, **no MAPE**, plus energy metrics | no composite score is produced |
| D-068 | Regimes are **measured**, never assumed | the premise for Phase 7 is evidence, not faith |
| D-069 | One fixed seed; **no significance claims** | determinism is tested, statistics are not invented |
| D-070 | Registry committed; artifacts regenerable | re-running replaces a record, never duplicates it |

---

## 6. Requirements and constraints in force

1. **Still no custom model.** No Energy World Model, no Qwen download or fine-tuning,
   no Energy-MoE, no experts, no routing, no optimiser, no digital twin, no red-team
   agent, no MLOps, no Jenkins, no UI. The only networks in this phase are two
   conventional baselines.
2. **No invented data.** Where SMART-DS has no series, the target is `UNSUPPORTED`
   and the runner refuses it by name.
3. `UNKNOWN` / `PARTIALLY_SUPPORTED` / `n/a` are correct answers; a number is not
   better than an honest absence.
4. The Phase 3 balance failure stays visible: 20.4506 kW, 0.5 kW tolerance unchanged.
5. **No leakage.** No shuffled split, no feature reading at or after its target, no
   weather forecast invented, no scaler fitted on test data.
6. One seed; **no claim of statistical significance from a single run.**
7. Do not commit `paper/`, `research/`, `*.tex`, the 228.5 MiB raw dataset, model
   checkpoints, or the ML dataset arrays.
8. Every dependency and every domain-model change needs a decision entry.

---

## 7. Testing and verification

Every figure below was produced in this session on Windows, Python 3.12.5, 16 threads.

```powershell
uv run pytest
  → 737 passed in 113 s          (was 584 after Phase 3; 153 Phase 4 tests added)

uv run pytest --cov
  → 89% total statement coverage (5618 statements)

uv run python -m energy_intelligence health
  → 7 passed, 0 warning(s), 0 failed - HEALTHY   (exit 0)

uv run python -m energy_intelligence ml targets
  → 7 targets, 3 usable, 4 UNSUPPORTED with evidence

uv run python -m energy_intelligence ml run
  → exit 0; dataset ds-16ff1dabe80b; 1,309,440 samples; 15 registry records
    total_seconds 2203.0   (dataset 8.6 s; the rest is model training)

uv run python -m energy_intelligence ml --ml-config configs/ml_pv_nowcast.toml run
  → exit 0; dataset ds-d381ee913098; 33,021 samples

uv run python -m energy_intelligence ml --ml-config configs/ml_pv_horizon.toml run
  → exit 0; dataset ds-77f04fd46328; 32,736 samples; ramp metric present at h=1

uv run pytest tests/ml -q --deselect <the 6 real-data tests>
  → 145 passed  (clean-checkout path, no dataset required)
```

The six real-data tests skip when `data/raw/smart_ds` is absent, exactly as Phase 3's
do, so a clean checkout runs the whole suite.

### Measured cost

| Quantity | Value |
|---|---|
| Load dataset build (1.3M samples, 40 series) | 8.6 s |
| Ridge, all 3 horizons | 0.5 s |
| Histogram gradient boosting, all 3 horizons | 20.5 s |
| Neural MLP, all 3 horizons | 389 s |
| Neural GRU, all 3 horizons | 1233 s |
| PV datasets (33k samples) | < 20 s each |
| Full load experiment, 8 models | 2203 s |
| Dataset arrays | 40 customers x 35,040 float32 per-unit = 5.4 MB per target matrix |

Hardware: 16 logical cores, CPU only. No GPU is used or required.

### Test coverage of the risky parts

| Property | Test |
|---|---|
| Cross-phase consistency | `test_feeder_extraction_matches_the_phase3_reconstruction` |
| The leakage guard has teeth | `test_the_guard_catches_a_deliberately_leaking_feature` |
| Weather withdrawn past 15 min | `test_weather_is_withdrawn_beyond_the_first_step` |
| Windows never cross a boundary | `test_no_window_crosses_a_split_boundary` |
| Scaler fitted on training only | `test_dataset_scaler_is_fitted_on_training_rows_only` |
| Targets align with origin+horizon | `test_targets_are_the_values_at_origin_plus_horizon` |
| Row layout correctness | `test_each_row_belongs_to_the_series_its_features_came_from` |
| Seasonal offsets are exact | `test_seasonal_offsets_are_exact_not_approximate` |
| sMAPE zero handling | `test_smape_treats_zero_over_zero_as_zero` |
| Unsupported targets refused | `test_unsupported_target_is_refused_with_evidence` |
| Experiment reproducibility | `test_experiment_is_reproducible`, `verify_reproducible` |

---

## 8. Known issues and risks

| # | Issue | Severity |
|---|---|---|
| 1 | **No weather forecast**, so day-ahead PV is unforecastable (F-04) | **Critical** for Phase 5 renewable work |
| 2 | **Feeder PV fleet underivable** — 20 irradiance shapes absent (G-03, M-04) | **Critical** for net load |
| 3 | **Neural baseline fails at 24 h** (F-02) — untuned, so a bound not a ceiling | High |
| 4 | Battery / EV / wind unsupported (M-07..M-09) | High for Phase 9 |
| 5 | **One year of data only** — seasonal models cannot be validated across years (M-10) | High for Phase 5 |
| 6 | **No holiday calendar** (M-11) — holidays hide inside the weekly pattern | Moderate, cheap to fix |
| 7 | No per-node voltage/current/reactive time series (M-12) | Moderate for Phase 11 |
| 8 | `availability = 1.0` assumed for all assets (M-13) | Moderate |
| 9 | Result covers 40 of 1,871 customers (D-062) | Moderate |
| 10 | `load_data/*.parquet` end-use detail not acquired (M-14) | Moderate, cheap to fix |
| 11 | Regime ratio unstable for near-zero-error rules (F-03 caveat) | Low, documented |
| 12 | GRU costs 1233 s for no gain | Low |
| 13 | No linter/type checker configured | Low |
| 14 | Branch is `master` | Low |

**Not carried forward as a blocker:** the Phase 3 balance residual is orthogonal to
every ML task here (F-07).

---

## 9. Unfinished work

Phase 4 deliberately did not implement: any custom model, Qwen, the Energy-MoE, an
optimiser, a digital twin, red-team agents, MLOps, Jenkins or a UI.

Cheap, high-value follow-ups identified but not done:

1. **Acquire `load_data/*.parquet`** (34 columns of end-use breakdown: heating,
   cooling, lighting, motors). Highest-value cheap acquisition available.
2. **Acquire a holiday calendar** for the region as a legitimate external input.
3. **Decide the wind scope question** — SMART-DS has no wind, so either drop it or
   source a second dataset. This should be an explicit decision, not a drift.
4. **Widen the customer sample** from 40 to all 1,871 and re-measure.

---

## 10. Next subphase

### Phase 4 follow-up (cheap, do before Phase 5)

1. Acquire the end-use parquet data and the holiday calendar.
2. Decide the wind scope explicitly (D-0xx).
3. Optionally re-run the load experiment with more seeds, before anyone claims a
   difference is significant.

### Phase 5 — Energy World Model

Phase 4 tells Phase 5 what it can and cannot learn from this data:

- **It will be a demand world model.** That is the only target this dataset supports
  well, and the measured baselines to beat are 0.4043 kW at 15 minutes and 1.5648 kW
  at 24 hours.
- **Do not attempt feeder PV or net load on this dataset.** The inputs do not exist
  (M-04, M-06). Attempting it means inventing data, which is the one thing this
  project has refused at every step so far.
- **Regime structure is real** (F-03) — high-demand, high-ramp and PV-ramp regimes
  carry an order of magnitude more error. A world model with a regime-aware
  representation is the first architecture change the evidence supports.
- **The hard series are high-volatility residential customers**, not large ones:
  `load_p1ulv6662` (54 kW rated, 11 kW mean, load factor 0.21, per-unit std 0.12) has
  7x the MAE of the median series and accounts for the worst rows, all in November and
  December evenings.
- **PV without a weather forecast is a dead end.** If Phase 5 wants renewable
  forecasting, the first action is a weather-forecast source, not a model.

**Baseline to beat, before anything else:** `classical_hist_gbm` at MAE 0.4242 kW
(15 min), 0.8579 kW (1 h), 1.5648 kW (24 h), on the chronological test split of
dataset `ds-16ff1dabe80b`.

---

## 11. Critical context

1. **The baselines exist to be beaten, and beating them is now cheap.** A future claim
   of "our model is better" must state which baseline, which horizon, which split,
   and which dataset version.

2. **A neural network is not obviously justified on this data.** That is a finding,
   not a failure. It was recorded rather than tuned away, and if Phase 5 wants to
   justify an Energy World Model it must justify it against gradient boosting, not
   against persistence.

3. **Read `decisions.md` before changing anything.** D-055 … D-070 carry
   alternatives and consequences, and several have tests that fail if reversed without
   a new entry.

4. **The two self-inflicted bugs (F-06) are the most important lesson of the phase.**
   An internally consistent pipeline was entirely wrong, and the only thing that
   caught it was a cross-check against an independent implementation and a "why is
   this worse than the mean?" question. Keep asking both.

5. **Watch the units.** The dataset stores per-unit values; metrics are reported in
   kW. `mae` and `mae_per_unit` are both in the comparison JSON, and a per-unit MAE
   silently reported as kW would be wrong by two orders of magnitude.

6. **`ml run` takes about 37 minutes on CPU**, dominated by the two neural baselines.
   Use `ml dataset` for dataset-only work and `ml --ml-config ... run` for the PV
   variants, which take seconds.

7. **Verify before claiming.** Run `uv run pytest` and `energy-intel health` and
   report real output. Note that `energy-intel data validate` exits **1** by design —
   that is the honest Phase 3 balance result, not a crash.

8. **Git hygiene.** Commits authored solely as
   `PrathamKapoor <prathamkapoor027@gmail.com>`. No `Co-Authored-By`, no AI or tool
   attribution anywhere. Never add Claude or any bot as a collaborator. Do not commit
   `paper/`, `research/`, `*.tex`, the raw dataset, or `artifacts/`.

9. **Undecided means `TBD`.** Every reasonable choice needs a `decisions.md` entry
   stating alternatives, selection, reason and consequence.

---

## 12. Agent instructions

If you are an automated agent continuing this project:

- **Do not fabricate energy data.** The single hardest-won property of this codebase
  is that every number is traceable to a real file. Four targets are `UNSUPPORTED`
  precisely because the alternative was inventing them.
- **Do not relax a tolerance, a split or a leakage guard to make a metric look
  better.** The 0.5 kW balance criterion fails on purpose; the chronological split and
  the weather withdrawal exist for the same reason.
- **Before adding a dependency, add a decision.** `tests/test_package.py` asserts the
  exact set and will fail.
- **Before changing `domain/`, add a decision and a test.** The Phase 4 rule is that
  the domain layer must import no ML library; a test enforces it.
- **When you change a feature, target, lookback, horizon, split, stride or
  normalisation, the dataset version changes automatically.** Do not overwrite a
  previous version; produce a new one.
- **Run `uv run pytest` before claiming anything is finished.** 737 tests, ~2 minutes.