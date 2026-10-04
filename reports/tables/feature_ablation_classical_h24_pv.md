# Feature ablation - pv_generation - H24 (OFFICIAL)

Model held fixed at `classical_ridge`. Feature set is the only variable.
Protocol hash `472ff9c1b723787f`.

F01-F04 select the feature set. F05-F06 are held out from selection and are
post-ablation confirmation, not unseen test data.

| feature set | n feats | fold | role | n | MAE kW | RMSE kW |
|---|---|---|---|---|---|---|
| A | 4 | F01 | ABLATION | 22440 | 0.21055 | 0.23577 |
| A | 4 | F02 | ABLATION | 22440 | 0.22522 | 0.24875 |
| A | 4 | F03 | ABLATION | 22440 | 0.20891 | 0.23423 |
| A | 4 | F04 | ABLATION | 22440 | 0.22859 | 0.25659 |
| A | 4 | F05 | CONFIRMATION | 29920 | 0.27167 | 0.30256 |
| A | 4 | F06 | CONFIRMATION | 59840 | 0.26967 | 0.30107 |
| B | 5 | F01 | ABLATION | 22440 | 0.15748 | 0.19640 |
| B | 5 | F02 | ABLATION | 22440 | 0.15541 | 0.19007 |
| B | 5 | F03 | ABLATION | 22440 | 0.16441 | 0.21156 |
| B | 5 | F04 | ABLATION | 22440 | 0.15672 | 0.18888 |
| B | 5 | F05 | CONFIRMATION | 29920 | 0.17723 | 0.19589 |
| B | 5 | F06 | CONFIRMATION | 59840 | 0.16471 | 0.18440 |
| C | 9 | F01 | ABLATION | 22440 | 0.19173 | 0.21560 |
| C | 9 | F02 | ABLATION | 22440 | 0.19978 | 0.22067 |
| C | 9 | F03 | ABLATION | 22440 | 0.19059 | 0.21547 |
| C | 9 | F04 | ABLATION | 22440 | 0.20219 | 0.22541 |
| C | 9 | F05 | CONFIRMATION | 29920 | 0.24955 | 0.27925 |
| C | 9 | F06 | CONFIRMATION | 59840 | 0.23854 | 0.26579 |
| D | 11 | F01 | ABLATION | 22440 | 0.19141 | 0.21426 |
| D | 11 | F02 | ABLATION | 22440 | 0.19974 | 0.22052 |
| D | 11 | F03 | ABLATION | 22440 | 0.19083 | 0.21611 |
| D | 11 | F04 | ABLATION | 22440 | 0.20437 | 0.22662 |
| D | 11 | F05 | CONFIRMATION | 29920 | 0.24001 | 0.26962 |
| D | 11 | F06 | CONFIRMATION | 59840 | 0.23794 | 0.26508 |
| E | 13 | F01 | ABLATION | 22440 | 0.19061 | 0.21326 |
| E | 13 | F02 | ABLATION | 22440 | 0.19862 | 0.21917 |
| E | 13 | F03 | ABLATION | 22440 | 0.18934 | 0.21492 |
| E | 13 | F04 | ABLATION | 22440 | 0.20446 | 0.22707 |
| E | 13 | F05 | CONFIRMATION | 29920 | 0.23738 | 0.26662 |
| E | 13 | F06 | CONFIRMATION | 59840 | 0.23585 | 0.26267 |
