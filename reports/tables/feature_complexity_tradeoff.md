# Feature-count versus error tradeoff (selection folds F01-F04)

Mean MAE on the four selection folds against feature count, with the cost of
each step away from the best absolute set.

| target | set | n features | mean MAE | cost vs best | note |
|---|---|---|---|---|---|
| customer_load | A | 4 | 0.16023 | +0.28% | **selected** |
| customer_load | B | 5 | 0.15978 | +0.00% |  |
| customer_load | C | 9 | 0.16118 | +0.88% |  |
| customer_load | D | 11 | 0.16056 | +0.49% |  |
| customer_load | E | 13 | 0.16100 | +0.77% |  |
| pv_generation | A | 4 | 0.21831 | +37.73% |  |
| pv_generation | B | 5 | 0.15850 | +0.00% | **selected** |
| pv_generation | C | 9 | 0.19607 | +23.70% |  |
| pv_generation | D | 11 | 0.19659 | +24.03% |  |
| pv_generation | E | 13 | 0.19576 | +23.50% |  |
