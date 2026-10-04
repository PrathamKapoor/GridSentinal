# Incremental feature-family effects

Each row is an **incremental validation-performance difference** between two
feature sets evaluated under an identical, frozen protocol. These are **not**
causal effects. Single seed; no significance testing was performed.

| target | comparison | baseline MAE | candidate MAE | abs diff | rel diff | direction | features added |
|---|---|---|---|---|---|---|---|
| customer_load | A -> B | 0.16023 | 0.15978 | -0.00045 | -0.28% | improvement | value_at_origin, lag_1, lag_4, lag_96, lag_672 |
| customer_load | A -> C | 0.16023 | 0.16118 | +0.00095 | +0.59% | regression | value_at_origin, lag_1, lag_4, lag_96, lag_672 |
| customer_load | C -> D | 0.16118 | 0.16056 | -0.00062 | -0.39% | improvement | roll_mean_96, roll_std_96 |
| customer_load | D -> E | 0.16056 | 0.16100 | +0.00044 | +0.28% | regression | ramp_1, roll_mean_same_hour_7d |
| pv_generation | A -> B | 0.21831 | 0.15850 | -0.05981 | -27.40% | improvement | value_at_origin, lag_1, lag_4, lag_96, lag_672 |
| pv_generation | A -> C | 0.21831 | 0.19607 | -0.02224 | -10.19% | improvement | value_at_origin, lag_1, lag_4, lag_96, lag_672 |
| pv_generation | C -> D | 0.19607 | 0.19659 | +0.00052 | +0.26% | regression | roll_mean_96, roll_std_96 |
| pv_generation | D -> E | 0.19659 | 0.19576 | -0.00083 | -0.42% | improvement | ramp_1, roll_mean_same_hour_7d |
