# Handoff — Phase 5

**Project:** Energy Intelligence — adaptive, self-verifying energy management
**Event:** Yuva Yodha Energy Tech Hackathon 2026, Schneider Electric
**Current phase:** **Phase 5 of 20 — Energy Demand Dynamics Model — COMPLETE (mixed result)**
**Previous phases:** Phase 1 foundation, Phase 2 domain contract, Phase 3 SMART-DS ingestion, Phase 4 ML dataset and baselines — **all preserved, zero regressions**
**Date:** 2026-10-03
**Author of all work:** PrathamKapoor <prathamkapoor027@gmail.com>

---

## 1. Work completed

Phase 5 asked one question:

> Does the **order** of a customer's demand history carry information that a flat
> vector of lags discards, and does a learned temporal model beat the classical
> baseline on the same rows?

**Partly: two horizons of three beat gradient boosting, the 24-hour horizon does not.**
That mixed verdict is the phase's finding, and it is recorded as mixed rather than
rounded up.

No Energy World Model, no Qwen, no Energy-MoE, no optimiser, no agent and no UI was
built. Three requirement documents were written for those phases instead
(`docs/world_model_requirements.md`, `docs/qwen_energy_requirements.md`,
`docs/moe_design_requirements.md`), each derived from what Phase 5 measured.

### Pipeline

```text
TargetSeries (kW per customer)
  -> normalise by each series' own rated kW
  -> build_sequence_index()      ordered windows, per-series expansion,
                                 chronological split, causality guard
  -> token stride 4, INPUTS ONLY  672 native steps -> 168 tokens; targets native
  -> TCN, dilations 1..64, GroupNorm within the window, per-series embedding
  -> 3 heads -> deltas at h = 1, 4, 96
  -> yhat(t+h) = y(t) + delta(t+h)
  -> per-unit -> kW, evaluate, analyse, record
```

### Result on the shared test split

`179,520` chronological test rows, identical for every row of this table. kW MAE:

| Model | h=1 (15 min) | h=4 (1 h) | h=96 (24 h) |
|---|---|---|---|
| persistence | **0.4043** | 0.8897 | 2.4343 |
| Phase 4 classical GBM | 0.4242 | 0.8579 | **1.5648** |
| **Phase 5 temporal TCN** | 0.4178 | **0.8307** | 1.6510 |
| Phase 5 vs GBM | **+1.5%** | **+3.2%** | **-5.5%** |
| Phase 5 vs persistence | -3.3% | **+6.6%** | **+32.2%** |

The sequence dataset version is `sq-437c16b5d925`: 596,640 windows (237,600 train /
179,520 validation / 179,520 test), 40 series, 168 tokens, 6 channels, 74,099
parameters, best epoch 7 of 8, 6,524 s of CPU training.

---

## 2. Findings that change Phase 6

### F-01 — The delta parameterisation is the biggest single design effect

Predicting the **change from `y(t)`** rather than the level is worth
**+253% / +52% / +20%** at the three horizons. Persistence is within 3% of the best
15-minute result, so most short-horizon difficulty is not in the history at all. Any
future model should start from `y(t)`.

### F-02 — Temporal order helps, but not at every horizon

The TCN beats gradient boosting at 15 min and 1 h and **loses at 24 h**. Judging a
future model by a single averaged number would hide the only horizon where it has to
win. `h=96` is the open problem.

### F-03 — Attention did not beat convolution here

A 112,131-parameter patch Transformer was **+49.8%** worse at h=1, -6.0% at h=4 and
+0.6% at h=96, at a matched reduced budget. This is the evidence base for
`docs/qwen_energy_requirements.md`, and it argues against assuming parameter count is
the bottleneck. The budget caveat is real: undertrained attention is the usual reason it
loses.

### F-04 — Context length matters, monotonically

One day of context costs +38.3% at h=1 and +34.6% at h=96; two days still cost +13.6%
and +9.7%. No saturation was observed up to a week.

### F-05 — Shared multi-horizon representation earns its place

A single horizon instead of three is 38.5% worse at h=1.

### F-06 — Error concentrates by regime, and no single model wins everywhere

At h=1, the high **demand** tercile costs 12x the low tercile (0.928 vs 0.077 kW) and
is 74% of total error; the high **ramp** tercile is 90% of total error. But the
per-regime winners differ: the TCN wins demand terciles, persistence wins low-ramp
periods by an order of magnitude (0.0034 vs 0.0324 kW), GBM wins high-ramp periods.
A regime-dependent best predictor is the precondition for a router — see
`docs/moe_design_requirements.md` §1.

### F-07 — The cyclic channels were not earning their place

The `no_cyclic` ablation was **better** at all three horizons (-3.6% / -28.7% / -18.6%).
The channels stayed in the shipped configuration (D-077): removing them after seeing
the test result would be tuning against the test split, and the ablation budget
(2 epochs, training stride 32) is too small to settle it — its reference reads 0.5435 kW
against the main run's 0.4178.

**Consequence to carry forward: the shipped main configuration is not the best
configuration this study found.** Removing the cyclic channels is the first thing
Phase 6 should test, at full budget, pre-registered.

---

## 3. Two errors Phase 5 made, and the fixes

Both are recorded because the artifacts looked healthy in both cases.

### E-01 — The regime table mixed horizons

The per-horizon metrics loop rebuilt the error arrays, so the regime table ended up
labelled with the **shortest** horizon's regime axes while carrying the **longest**
horizon's errors. Per-regime MAEs came out ~4x the headline figure; a reader would have
taken them at face value. **No headline metric was affected**, so no cross-check
caught it — it was found by noticing the per-regime numbers could not average to the
reported total.

Fix: analysis factored into one shared function that takes the horizon explicitly and
emits a table **per horizon** plus a headline naming its horizon; two regression tests
assert each horizon is judged on its own errors; a checkpoint replay path re-derives the
analysis from the saved checkpoint and **raises if the recomputed MAE differs from the
recorded one**. The replay confirmed the shipped checkpoint reproduces its metrics
exactly. (D-078)

### E-02 — A list where the contract wanted a tuple

`horizons = [1]` from TOML failed validation while `horizons = (1,)` passed — an
ablation crashed for a reason unrelated to the data. Fixed by normalising in
`SequenceConfig.__post_init__`, with a regression test.

### Also worth knowing

* An early TCN run was invalidated by **CPU contention** and a wall-clock cap (one
  epoch took 20,178 s). It was discarded, not reported. Do not run heavy commands
  concurrently with training.
* `_to_kw` broadcast a 1-D column against the per-row scale twice, inflating kW
  errors; the runner rewrote it defensively and it is tested for both ranks.

---

## 4. What is deliberately NOT claimed

* **Not** statistical significance. One seed (`20260101`), no confidence intervals, no
  p-values. `verify_reproducible` proves determinism, not generalisation.
* **Not** that a temporal model beats the classical baseline. It does at two of three
  horizons.
* **Not** an action-conditioned world model. No actions, transitions or system
  response exist in the data; `docs/world_model_requirements.md` records this.
* **Not** that a larger model would help. Nothing measured suggests parameter count is
  the bottleneck — a 74k-parameter TCN beat gradient boosting at two horizons while
  Phase 4's larger MLP and GRU lost.
* **Not** that the ablation budget supports absolute comparisons. Valid **within**
  `artifacts/phase5/ablations.md`, not against the main run.

---

## 5. Blocked or impossible on this dataset

| Item | Status | Evidence |
|---|---|---|
| Feeder PV output, net load | **Unresolvable** | the 20 referenced irradiance loadshapes do not exist (M-04); net load additionally needs battery dispatch |
| Battery dispatch / SOC | **Unavailable** | no dispatch trace; Phase 3 G-01 |
| EV demand | **Unavailable** | no EV assets in the subset |
| Wind generation | **Unavailable** | no wind assets |
| Weather forecasts | **Unavailable** | no forecast product in SMART-DS |
| Action-conditioned dynamics | **Unavailable** | no actions, transitions or state response |
| Commercial residual `-20.4506 kW` | **Unresolved** | Phase 3 G-04 |
| SMART-DS licence terms | **Unknown** | Phase 3 G-05 |
| Phase 3 balance | **Intentionally failing** | max residual 20.4506 kW vs 0.5 kW tolerance; orthogonal to per-customer forecasting |

---

## 6. Repository state

Commits, in order: `2571c48` Phase 1, `ff17a35` Phase 2, `a255887` Phase 3,
`7351e8c` Phase 4, and Phase 5 as the newest commit.

```text
src/energy_intelligence/ml/sequence.py     SequenceConfig, index, causality guard
src/energy_intelligence/ml/temporal.py     TCN + Transformer builders
src/energy_intelligence/ml/training.py     fit, checkpoint, early stop, evaluate
src/energy_intelligence/ml/phase5.py       runner, analyse_predictions, replay_evaluation
src/energy_intelligence/ml/ablation.py     ablation suite + table
src/energy_intelligence/config/temporal.py strict config loader
configs/temporal.toml                       the shipped experiment
```

Artifacts (git-ignored, regenerable): `artifacts/phase5/tcn-main/` holds
`checkpoint.pt`, `dataset_manifest.json`, `training_log.jsonl`, `result.json`,
`regimes.json`, `error_analysis.json`; `artifacts/phase5/ablations.md` and
`ablations.json` hold the suite. `experiments/registry.jsonl` is the tracked record.

Dependencies are unchanged from Phase 4: `numpy`, `scikit-learn`, CPU-only `torch`.
No new dependency was added in Phase 5.

---

## 7. Tests and verification

```powershell
uv run pytest -q                # 810 passed, 3m
uv run pytest -q --cov=energy_intelligence --cov-report=term-missing
                                # 88% overall; ml package 91%
uv run python -m energy_intelligence health          # 7 passed, 0 failed
energy-intel temporal config
energy-intel temporal experiments
```

Phase 5 added `tests/ml/test_sequence.py` and `tests/ml/test_temporal.py`, and
extended `tests/ml/test_phase5.py`: **224 tests** in `tests/ml` (up from 190), full
suite **810** (up from 737). Module coverage: `temporal.py` 97%, `training.py` 95%,
`phase5.py` 96%, `ablation.py` 99%, `sequence.py` 91%.

Two of the new tests exist because Phase 5 shipped a real defect: the regime analysis
mixed horizons (E-01), and a config list failed where a tuple was required (E-02).

---

## 8. Phase 6 recommendation

Start with the **cheap, pre-registered experiment**: retrain the TCN at full budget with
the cyclic channels removed (F-07). It is the one change Phase 5 already measured as
helpful, it costs one run, and deciding it after the fact would be test-set tuning.

Then, in order of measured value:

1. **Attack h=96.** It is the only horizon where the temporal model loses, and the one
   an operator most wants. Long context helped monotonically (F-04), so extending
   lookback past a week is the obvious first probe.
2. **Re-test attention at full budget** before scaling it (F-03), and record a negative
   result as a legitimate outcome.
3. **Multi-seed evaluation** before any "significantly better" claim.
4. **Qwen only after** the bar in `docs/qwen_energy_requirements.md` §1 is met: beat
   **1.5648 kW at 24 hours**, with at least three seeds, on more than one year of data
   or an explicit statement that annual generalisation is out of scope.

If the next phase builds an MoE, note the blocking constraint in
`docs/moe_design_requirements.md` §2: the measured regime terciles are computed from
the **future** demand and cannot be used by a router. A gate must run on features
observable at the forecast origin, and its inputs need their own leakage test.

---

## 9. Critical context

* **Never invent data.** No synthetic energy values, weather, PV output, battery
  dispatch, EV demand, wind or topology. Unsupported targets are refused with evidence,
  not approximated.
* **The chronological split is the contract.** Never shuffled; validation and test
  origins keep stride 1 so the `179,520` test rows stay identical to Phase 4's. Any
  Phase 6 comparison is meaningless on different rows.
* **Scaling is train-only**, by each series' own rated kW — a physical quantity known
  for the whole horizon.
* **One seed, no significance claims.**
* **Do not run heavy commands concurrently with training.** CPU contention has already
  invalidated one run.
* **Commit identity:** `PrathamKapoor <prathamkapoor027@gmail.com>`. No attribution
  lines, no AI or bot collaborators, on any commit, PR, tag or release note.
