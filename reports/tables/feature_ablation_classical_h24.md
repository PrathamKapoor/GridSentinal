# Feature ablation - customer_load - H24 (OFFICIAL)

Model held fixed at `classical_hist_gbm`. Feature set is the only variable.
Protocol hash `472ff9c1b723787f`.

F01-F04 select the feature set. F05-F06 are held out from selection and are
post-ablation confirmation, not unseen test data.

| feature set | n feats | fold | role | n | MAE kW | RMSE kW |
|---|---|---|---|---|---|---|
| A | 4 | F01 | ABLATION | 22440 | 0.15596 | 0.18753 |
| A | 4 | F02 | ABLATION | 22440 | 0.16168 | 0.18669 |
| A | 4 | F03 | ABLATION | 22440 | 0.15895 | 0.19514 |
| A | 4 | F04 | ABLATION | 22440 | 0.16433 | 0.18959 |
| A | 4 | F05 | CONFIRMATION | 29920 | 0.17291 | 0.19010 |
| A | 4 | F06 | CONFIRMATION | 59840 | 0.17349 | 0.19123 |
| B | 5 | F01 | ABLATION | 22440 | 0.16205 | 0.19559 |
| B | 5 | F02 | ABLATION | 22440 | 0.15568 | 0.19030 |
| B | 5 | F03 | ABLATION | 22440 | 0.16352 | 0.20963 |
| B | 5 | F04 | ABLATION | 22440 | 0.15785 | 0.18789 |
| B | 5 | F05 | CONFIRMATION | 29920 | 0.18300 | 0.20386 |
| B | 5 | F06 | CONFIRMATION | 59840 | 0.17214 | 0.19362 |
| C | 9 | F01 | ABLATION | 22440 | 0.15651 | 0.18762 |
| C | 9 | F02 | ABLATION | 22440 | 0.16270 | 0.18698 |
| C | 9 | F03 | ABLATION | 22440 | 0.16012 | 0.19578 |
| C | 9 | F04 | ABLATION | 22440 | 0.16538 | 0.19000 |
| C | 9 | F05 | CONFIRMATION | 29920 | 0.17212 | 0.18888 |
| C | 9 | F06 | CONFIRMATION | 59840 | 0.17381 | 0.19115 |
| D | 11 | F01 | ABLATION | 22440 | 0.15587 | 0.18758 |
| D | 11 | F02 | ABLATION | 22440 | 0.16084 | 0.18705 |
| D | 11 | F03 | ABLATION | 22440 | 0.16064 | 0.19815 |
| D | 11 | F04 | ABLATION | 22440 | 0.16487 | 0.18983 |
| D | 11 | F05 | CONFIRMATION | 29920 | 0.16816 | 0.18394 |
| D | 11 | F06 | CONFIRMATION | 59840 | 0.17217 | 0.18957 |
| E | 13 | F01 | ABLATION | 22440 | 0.15699 | 0.18725 |
| E | 13 | F02 | ABLATION | 22440 | 0.16118 | 0.18692 |
| E | 13 | F03 | ABLATION | 22440 | 0.16057 | 0.19690 |
| E | 13 | F04 | ABLATION | 22440 | 0.16527 | 0.18968 |
| E | 13 | F05 | CONFIRMATION | 29920 | 0.17071 | 0.18706 |
| E | 13 | F06 | CONFIRMATION | 59840 | 0.17359 | 0.19101 |
