# Handoff - Phase 10

**Project:** Energy Intelligence — GridSentinal
**Event:** Yuva Yodha Energy Tech Hackathon 2026, Schneider Electric
**Current phase:** **Phase 10 of 20 — Controlled feature ablation — PARTIAL**
**Previous phases:** Phases 1-9 complete and preserved, zero regressions
**Date:** 2026-10-04
**Author of all work:** PrathamKapoor <prathamkapoor027@gmail.com>

> Phase 10 was built and executed inside a one-hour budget, ship-first. The implementation, the
> console, the tests, the documentation and the classical grid all shipped. One arm — the MLP
> robustness check — was deliberately not run and is recorded as NOT RUN rather than
> approximated. **This phase is PARTIAL, not COMPLETE**, and the exact boundary is in §2.

---

## 1. The answer

> Which predefined feature families actually contribute predictive information when model
> configurations are held fixed?

| target | model | selected | mean MAE (F01-F04) | best absolute | tie-break |
|---|---|---|---|---|---|
| `customer_load` | `classical_hist_gbm` | **A** (4 features) | 0.16023 | B, 0.15978 | **applied** |
| `pv_generation` | `classical_ridge` | **B** (5 features) | 0.15850 | B, 0.15850 | no |

Two findings worth carrying forward, and one of them is uncomfortable:

**1. For load, almost no feature family matters.** All five sets fall inside 0.9% of each other:

```text
A 0.16023   B 0.15978   C 0.16118   D 0.16056   E 0.16100
```

B's mean MAE was 0.28% better than A — inside the 0.5% tolerance frozen before any result was
visible — so the simplicity tie-break selected the 4-feature calendar set. **The full 13-feature
set E is worse than the 4-feature set A.** Feature count is not the lever it looks like.

**2. For PV, lags dominate and calendar actively hurts:**

```text
A 0.21831 → B 0.15850  (−27.40%)   the single largest effect in the phase
A 0.21831 → C 0.19607  (−10.19%)   calendar + lags recovers only a third of it
D 0.19659 → E 0.19576  (−0.42%)    ramp adds nothing measurable
```

Calendar features cost PV 27% relative to lags alone. That is the largest honest effect
measured anywhere in this project.

**3. Load's selection did not hold on confirmation.** On F05-F06 the best set was D, not A:

```text
A 0.17320   B 0.17757   C 0.17297   D 0.17016   E 0.17215
```

The chosen set was 1.78% off the best. With a single seed and differences of this size across
folds, that is exactly the kind of result that cannot be distinguished from noise — which is
why it is reported rather than buried, and why no significance claim is made.

## 2. Status, stated exactly

| gate | state |
|---|---|
| implementation | **COMPLETE** |
| tested | **COMPLETE** — 65 new tests, 1356 total |
| smoke | **PASS** — 78.6s, stamped `NON_EVIDENCE_SMOKE` |
| official classical | **COMPLETE** — 2 targets × 5 sets × 6 folds, 105.0s |
| official MLP | **NOT RUN** — recorded, not approximated |
| feature selection (F01-F04) | **COMPLETE** — both targets |
| confirmation (F05-F06) | **COMPLETE** — both targets |
| WIND | **UNSUPPORTED** — no wind asset in SMART-DS v1.0 |

The single incomplete item is the MLP robustness arm. Training it properly costs more than the
classical grid, and a half-trained MLP ablation would produce a number that looks like a
robustness result and is not one. Its frozen configuration checksum is already in the protocol
freeze, so it can be run later without renegotiating the protocol.

## 3. What was held fixed

Per the protocol freeze, hashed `472ff9c1b723787f` and committed at
`artifacts/experimental_design/phase_10_ablation_protocol_freeze.yaml`:

model implementation, hyperparameters, training procedure, scaling, seed (20260101), horizon
(H24 = 96 steps), the six folds, the metrics, the target definition, the timestamp policy.

No Optuna, no GridSearch, no RandomSearch, no feature-specific tuning. So **no difference in
this phase is attributable to a different amount of tuning** — which is the point of an
ablation.

Feature sets, built only from the existing Phase 4 catalogue, weather columns excluded:

```text
A calendar                4   hour_of_day, day_of_week, day_of_year, is_weekend
B autoregressive lags     5   value_at_origin, lag_1, lag_4, lag_96, lag_672
C = A + B                 9
D = C + rolling          11   + roll_mean_96, roll_std_96
E = D + ramp             13   + ramp_1, roll_mean_same_hour_7d  (= every non-weather feature)
```

No feature was invented. `E` terminates at the whole available non-weather catalogue rather
than at an arbitrary stopping point.

## 4. Integrity — the final test is still locked

```text
FINAL_TEST_TRAINING_ACCESS          = NO
FINAL_TEST_HPO_ACCESS                = NO
FINAL_TEST_MODEL_SELECTION_ACCESS    = NO
FINAL_TEST_FEATURE_SELECTION_ACCESS  = NO
FINAL_TEST_PERFORMANCE_EVALUATION    = NO
FINAL_TEST_INTEGRITY_AUDIT_ACCESS    = historical P9-DEV-002 only
```

Three independent controls, not one:

1. **No CLI option exists.** `--final-test` and `--test-split` are absent from the parser. An
   option that does not exist cannot be reached for by a well-meaning flag.
2. **The folds cannot reach it.** `ablation_folds()` takes the validation range as its outer
   bound, so no fold it emits extends past validation.
3. **The runner refuses it at runtime.** `assert_final_test_denied()` raises
   `FinalTestAccessError`, and a test proves that corrupting every test-split row leaves the
   scored numbers bit-identical.

Selection is restricted to F01–F04 in code, not by convention: `select()` takes the fold ids as
an argument and **raises** if handed F05 or F06.

## 5. Evidence that was kept, and evidence that was not

Timestamp-level rows are retained for every cell — `forecast_origin`, `target_timestamp`,
`actual`, `prediction`, `error`, `absolute_error`, `squared_error` — because they are the
evidence base a future paired significance analysis would need and they cannot be reconstructed
afterwards.

`artifacts/paper_evidence/evidence_registry.yaml` registers 7 artifacts and explicitly lists 3
as **not** registered:

| not registered | why |
|---|---|
| `phase_10_smoke` | `NON_EVIDENCE_SMOKE`; excluded from every research table by test |
| `phase_10_mlp_robustness` | NOT RUN — recorded rather than approximated |
| `phase_10_wind` | UNSUPPORTED — no wind asset exists; not substituted |

## 6. Defects this phase found in its own code

Each was caught by a measurement or a test, and each has a named regression test.

| defect | how it surfaced | consequence had it shipped |
|---|---|---|
| tie-break compared `len(set_id)` — the **ID string**, not the feature count | a test asserting the tie-break fires on a near-tie | `"A"` and `"B"` both have length 1, so the tie-break could **never** fire; the selected set was wrong |
| selection reported a `mean_mae` key that does not exist | `KeyError` after the whole official grid had run | results computed, then discarded — a 96-second run thrown away |
| `run_one` read `targets[:, horizon_steps]` on a single-column array | `IndexError` | single-fold callers broken |
| the tie-break loop skipped the simplest set | the same test | it skipped exactly the set it was meant to prefer |
| `ml/ablation.py` shadowing | `ModuleNotFoundError` | package could not be named `ablation`; renamed (D-121) |
| PV table filenames `_pv_pv` | inspecting the written paths | duplicated suffix |

The tie-break bug is the one that mattered. It changed the selected feature set for
`customer_load` from B to A, and it could only have been caught by a test that asserts the
*rule's* behaviour rather than by reading the code.

## 7. The console

`energy-intel console …` and `energy-intel ablation …` are the operator path. The console is
read-only over artifacts earlier phases wrote: it never recomputes a result, and it never
renders a missing measurement as a zero.

```text
console health      system, config, data, models, integrity, phase
console status      every phase with the verdict it actually reached
console data        SMART-DS availability, per target, unknowns
console config      active config, config dir, the five feature sets
console flexibility  PHYSICAL / STATISTICAL / ASSUMED — three lines, never merged
console integrity  final-test state and the protocol freeze
ablation smoke      non-evidence pipeline check
ablation run        the official grid, selection, confirmation
ablation report     the recorded result; regenerates the tables
```

**Flexibility is printed as three lines on purpose.** A console that printed
`flexibility: 4 kW` would undo Phase 9. A test asserts that the word "dispatchable" appears
only inside a negation, and that an unrun Phase 9 shows UNKNOWN per basis rather than one
missing line.

Phases 6 and 7 appear in `console status` as **NEGATIVE**. That is deliberate: the ledger that
hides its own failures is not a ledger.

## 8. Blocked, out of scope, or not done

- **MLP robustness arm** — NOT RUN. The frozen config is in the freeze; the arm is runnable.
- **WIND** — UNSUPPORTED. No wind asset in SMART-DS v1.0. Reported, not substituted.
- **Formal significance testing** — explicitly out of scope. Single seed; a difference smaller
  than seed-to-seed variation cannot be separated from it, and no claim of that kind is made.
- **Figures** — not generated. Correct tables and reproducibility were worth more than plots in
  the remaining budget, so `artifacts/research_figures/phase_10/` and its manifest were
  **not** created rather than being created empty.
- **A properly conditioned calendar reference** — the loser on PV was not disproven, it was
  worse by 27%. Worth investigating with more calibration data; it needs its own measurement.

## 9. Console commands — all verified

```powershell
cd C:\Projects\Schneider
.venv\Scripts\python.exe -m energy_intelligence console health
.venv\Scripts\python.exe -m energy_intelligence console status
.venv\Scripts\python.exe -m energy_intelligence console data
.venv\Scripts\python.exe -m energy_intelligence console config
.venv\Scripts\python.exe -m energy_intelligence console flexibility
.venv\Scripts\python.exe -m energy_intelligence console integrity
.venv\Scripts\python.exe -m energy_intelligence ablation smoke
.venv\Scripts\python.exe -m energy_intelligence ablation run
.venv\Scripts\python.exe -m energy_intelligence ablation report
.venv\Scripts\python.exe scripts\run_feature_ablation.py --smoke
.venv\Scripts\python.exe scripts\run_feature_ablation.py --all
.venv\Scripts\python.exe -m pytest -q --no-header -p no:cacheprovider
.venv\Scripts\python.exe -m compileall -q src scripts tests
```

Full walkthrough in `README.md` → **Running GridSentinal**.

## 10. Repository state

- Branch `main`, remote `https://github.com/PrathamKapoor/GridSentinal.git`.
- **Tests: 1356 passed, 0 failed.** Compile: PASS.
- `.gitignore` now tracks `artifacts/experimental_design/`, `artifacts/research_tables/` and
  `artifacts/paper_evidence/`. A protocol freeze that is not committed governs nothing, and a
  research table that is not committed cannot be checked against its run. Large binaries under
  `artifacts/` remain ignored.
- **Frontend untouched.** The uncommitted `web/` work belongs to the user and was not staged.
- No credentials, caches or raw data tracked.

## 11. Phase 11 readiness

**READY, with the final test still available as a benchmark.**

Available to Phase 11: the hashed protocol freeze, the selection freeze, the six fold
definitions, the per-cell timestamp rows, the common-sample accounting, and a recorded
selection that did not fully survive confirmation — which is itself worth knowing before a
model is chosen on it.

Phase 11 must not re-select features on the final split. The final test performance evaluation
is still LOCKED and Phase 10 did not touch it.

## 12. License

No `LICENSE` file exists. Carried forward from earlier phases; settle before any external
distribution.