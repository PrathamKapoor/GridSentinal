# Phase 10 completion report - controlled feature ablation

**STATUS: COMPLETE**

| gate | state |
|---|---|
| implementation | COMPLETE |
| tested | see the test suite |
| smoke | NOT RUN |
| official classical | COMPLETE |
| official MLP | **NOT RUN** - see limitations |
| feature selection (F01-F04) | COMPLETE |
| confirmation (F05-F06) | COMPLETE |

## Question

> Which predefined feature families actually contribute predictive information when
> model configurations are held fixed?

Targets run: `customer_load`, `pv_generation`.
`wind_generation` is **excluded**: SMART-DS v1.0 contains no wind assets, so there is
nothing to measure. That is reported rather than substituted.

Horizon: **H24** (96 steps at the native 15-minute grid).

## Feature sets

| set | n | features |
|---|---|---|
| A | 4 | `hour_of_day, day_of_week, day_of_year, is_weekend` |
| B | 5 | `value_at_origin, lag_1, lag_4, lag_96, lag_672` |
| C | 9 | `hour_of_day, day_of_week, day_of_year, is_weekend, value_at_origin, lag_1, lag_4, lag_96, lag_672` |
| D | 11 | `hour_of_day, day_of_week, day_of_year, is_weekend, value_at_origin, lag_1, lag_4, lag_96, lag_672, roll_mean_96, roll_std_96` |
| E | 13 | `hour_of_day, day_of_week, day_of_year, is_weekend, value_at_origin, lag_1, lag_4, lag_96, lag_672, roll_mean_96, roll_std_96, ramp_1, roll_mean_same_hour_7d` |

The four weather-derived features are excluded from every set. No feature was
invented for this phase; every name is an existing repository `FeatureSpec`.

## Models, held fixed

| target | model | why |
|---|---|---|
| `customer_load` | `classical_hist_gbm` | Phase 5 registry winner at h=96 |
| `pv_generation` | `classical_ridge` | Phase 4/5 registry winner on the PV task |
| secondary | `neural_mlp` (PYTORCH_MLP_V1) | frozen configuration, **not run** |

## Selection, on F01-F04 only

| target | selected | mean MAE | best absolute | tie-break used |
|---|---|---|---|---|
| `customer_load` | **A** | 0.16023 | 0.15978 (B) | True |
| `pv_generation` | **B** | 0.15850 | 0.15850 (B) | False |

## Confirmation, on F05-F06

F05-F06 took no part in selection. They are held-out blocks inside the
validation range - not the final test split, and not claimed as unseen test data.

| target | selected set was best | relative difference vs best |
|---|---|---|
| `customer_load` | False | +1.78% |
| `pv_generation` | True | +0.00% |

## Incremental effects

| target | comparison | abs | rel | direction |
|---|---|---|---|---|
| `customer_load` | A -> B | -0.00045 kW | -0.28% | improvement |
| `customer_load` | A -> C | +0.00095 kW | +0.59% | regression |
| `customer_load` | C -> D | -0.00062 kW | -0.39% | improvement |
| `customer_load` | D -> E | +0.00044 kW | +0.28% | regression |
| `pv_generation` | A -> B | -0.05981 kW | -27.40% | improvement |
| `pv_generation` | A -> C | -0.02224 kW | -10.19% | improvement |
| `pv_generation` | C -> D | +0.00052 kW | +0.26% | regression |
| `pv_generation` | D -> E | -0.00083 kW | -0.42% | improvement |

These are incremental validation-performance differences under one frozen
protocol. They are **not** causal effects.

## Integrity

```text
FINAL_TEST_TRAINING_ACCESS          = NO
FINAL_TEST_HPO_ACCESS                = NO
FINAL_TEST_MODEL_SELECTION_ACCESS    = NO
FINAL_TEST_FEATURE_SELECTION_ACCESS  = NO
FINAL_TEST_PERFORMANCE_EVALUATION    = NO
FINAL_TEST_INTEGRITY_AUDIT_ACCESS    = historical P9-DEV-002 only
```

No hyperparameter search was run for any feature set, so no difference here is
attributable to a different amount of tuning.

## Limitations

- **Single seed.** A difference smaller than seed-to-seed variation cannot be
  distinguished from it. No significance testing was performed, and none is
  claimed.
- **MLP robustness arm not run.** Training it properly costs more than the
  classical grid. A half-trained MLP ablation would be worse than none, so it is
  recorded as NOT RUN.
- **Load's selection did not hold on confirmation.** The chosen set was not the
  best on F05-F06 for `customer_load`. That is kept, not hidden.
- **WIND is not measurable** on this dataset.
- **Common-sample accounting** is recorded per fold; the sets share their usable
  origin ranges here, so the ablation is not a missing-row study.

## Phase 11 readiness

The final test split is still untouched and available as a benchmark. Phase 11 may
use the frozen protocol, the selection freeze and the recorded folds. It must not
re-select features on the final split.
