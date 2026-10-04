# Post-ablation confirmation (F05-F06)

F05-F06 took no part in selecting the feature set. They are contiguous blocks
inside the validation range: held out from selection, but **not** the final
test split and carrying no unseen-test claim.

| target | selected set | selected set was best | rel diff vs best | per-set confirmation mean MAE |
|---|---|---|---|---|
| customer_load | A | False | +1.78% | A 0.17320, B 0.17757, C 0.17297, D 0.17016, E 0.17215 |
| pv_generation | B | True | +0.00% | A 0.27067, B 0.17097, C 0.24405, D 0.23898, E 0.23662 |
