# Benchmark Comparison Table: small_50x30 (30 Seeds)

| method                            |   hv_mean |   hv_std |   hv_median |   feasibility_rate |   z1_mean |   z1_std |   z2_mean |   z2_std |   z3_mean |   z3_std |   spacing_mean |   runtime_mean |   runtime_std |   pct_dominates_random |   pct_dominates_greedy |   pct_dominates_mcf |   pct_dominates_mcf_repaired |
|:----------------------------------|----------:|---------:|------------:|-------------------:|----------:|---------:|----------:|---------:|----------:|---------:|---------------:|---------------:|--------------:|-----------------------:|-----------------------:|--------------------:|-----------------------------:|
| Random                            |  0.011796 | 0.008512 |    0.00913  |             0      |   44.7921 |   1.2367 |    0.5468 |   0.0307 |      7.07 |     0.64 |        0       |         0.0658 |        0.0412 |                    0   |                      0 |                   0 |                            0 |
| Greedy                            |  0.093186 | 0        |    0.093186 |             1      |   72.615  |   0      |    1.023  |   0      |      8    |     0    |        0       |         0.0065 |        0      |                   23.3 |                      0 |                   0 |                          100 |
| Min-Cost Flow (raw: infeasible)   |  0.095203 | 0        |    0.095203 |             0      |   73.3523 |   0      |    1.023  |   0      |      8    |     0    |        0       |         0.105  |        0      |                   23.3 |                    100 |                   0 |                          100 |
| mcf_repaired (repaired: feasible) |  0.092909 | 0        |    0.092909 |             1      |   72.5135 |   0      |    1.023  |   0      |      8    |     0    |        0       |         0.1091 |        0      |                   23.3 |                      0 |                   0 |                            0 |
| NSGA-II                           |  0.365117 | 0.031291 |    0.36764  |             1      |   65.1137 |   0.8538 |    0.9449 |   0.0292 |      6    |     0    |        0.38147 |        13.5361 |        3.5612 |                  100   |                      0 |                   0 |                            0 |
| SPEA2                             |  0.374278 | 0.038433 |    0.372151 |             1      |   65.1754 |   1.031  |    0.9536 |   0.0345 |      5.97 |     0.18 |        0.48072 |        12.8055 |        3.502  |                  100   |                      0 |                   0 |                            0 |
| MOEA/D                            |  0.224454 | 0.0553   |    0.220176 |             0.8936 |   63.0208 |   1.3772 |    0.7814 |   0.0755 |      6.1  |     0.31 |        0.00791 |        18.2047 |        4.5584 |                   96.7 |                      0 |                   0 |                            0 |
| Hybrid                            |  0.772709 | 0.115765 |    0.825396 |             0.7771 |   72.637  |   0.0484 |    1.1142 |   0.0251 |      5.43 |     0.5  |        0.65984 |        21.2465 |        5.6588 |                  100   |                    100 |                   0 |                          100 |

### Wilcoxon Statistical Hypothesis Tests on Hypervolume

- **Hybrid_vs_NSGA-II**: p-value = `1.8626e-09`, rank-biserial $r$ = `1.0`, significant = `True`
- **Hybrid_vs_SPEA2**: p-value = `1.8626e-09`, rank-biserial $r$ = `1.0`, significant = `True`
- **Hybrid_vs_MOEA/D**: p-value = `1.8626e-09`, rank-biserial $r$ = `1.0`, significant = `True`
- **NSGA-II_vs_Random**: p-value = `1.8626e-09`, rank-biserial $r$ = `1.0`, significant = `True`
- **NSGA-II_vs_Greedy**: p-value = `1.8626e-09`, rank-biserial $r$ = `1.0`, significant = `True`
- **NSGA-II_vs_Min-Cost Flow (raw: infeasible)**: p-value = `1.8626e-09`, rank-biserial $r$ = `1.0`, significant = `True`
- **NSGA-II_vs_mcf_repaired (repaired: feasible)**: p-value = `1.8626e-09`, rank-biserial $r$ = `1.0`, significant = `True`
