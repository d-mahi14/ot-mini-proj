# Benchmark Comparison Table: small_50x30 (30 Seeds)

| method                            |   hv_mean |   hv_std |   hv_median |   feasibility_rate |   z1_mean |   z1_std |   z2_mean |   z2_std |   z3_mean |   z3_std |   spacing_mean |   runtime_mean |   runtime_std |   pct_dominates_random |   pct_dominates_greedy |   pct_dominates_mcf |   pct_dominates_mcf_repaired |
|:----------------------------------|----------:|---------:|------------:|-------------------:|----------:|---------:|----------:|---------:|----------:|---------:|---------------:|---------------:|--------------:|-----------------------:|-----------------------:|--------------------:|-----------------------------:|
| Random                            |  0.011657 | 0.008432 |    0.009004 |             0      |   44.7921 |   1.2367 |    0.5468 |   0.0307 |      7.07 |     0.64 |        0       |         0.148  |        0.0967 |                    0   |                      0 |                   0 |                            0 |
| Greedy                            |  0.093181 | 0        |    0.093181 |             1      |   72.615  |   0      |    1.023  |   0      |      8    |     0    |        0       |         0.003  |        0      |                   23.3 |                      0 |                   0 |                          100 |
| Min-Cost Flow (raw: infeasible)   |  0.095203 | 0        |    0.095203 |             0      |   73.3523 |   0      |    1.023  |   0      |      8    |     0    |        0       |         0.0467 |        0      |                   23.3 |                    100 |                   0 |                          100 |
| mcf_repaired (repaired: feasible) |  0.092902 | 0        |    0.092902 |             1      |   72.5135 |   0      |    1.023  |   0      |      8    |     0    |        0       |         0.053  |        0      |                   23.3 |                      0 |                   0 |                            0 |
| NSGA-II                           |  0.364785 | 0.031305 |    0.367287 |             1      |   65.1137 |   0.8538 |    0.9449 |   0.0292 |      6    |     0    |        0.38147 |        17.8864 |        5.9326 |                  100   |                      0 |                   0 |                            0 |
| SPEA2                             |  0.373939 | 0.03843  |    0.371815 |             1      |   65.1754 |   1.031  |    0.9536 |   0.0345 |      5.97 |     0.18 |        0.48072 |        17.9585 |        4.6854 |                  100   |                      0 |                   0 |                            0 |
| MOEA/D                            |  0.224193 | 0.055245 |    0.219878 |             0.8936 |   63.0208 |   1.3772 |    0.7814 |   0.0755 |      6.1  |     0.31 |        0.00791 |        26.3169 |        4.957  |                   96.7 |                      0 |                   0 |                            0 |
| Hybrid                            |  0.765035 | 0.133484 |    0.688431 |             0.7594 |   72.6237 |   0.0186 |    1.1276 |   0.0273 |      5.53 |     0.51 |        0.62242 |        31.9637 |        9.3348 |                  100   |                    100 |                   0 |                          100 |

### Wilcoxon Statistical Hypothesis Tests on Hypervolume

- **Hybrid_vs_NSGA-II**: p-value = `1.8626e-09`, rank-biserial $r$ = `1.0`, significant = `True`
- **Hybrid_vs_SPEA2**: p-value = `1.8626e-09`, rank-biserial $r$ = `1.0`, significant = `True`
- **Hybrid_vs_MOEA/D**: p-value = `1.8626e-09`, rank-biserial $r$ = `1.0`, significant = `True`
- **NSGA-II_vs_Random**: p-value = `1.8626e-09`, rank-biserial $r$ = `1.0`, significant = `True`
- **NSGA-II_vs_Greedy**: p-value = `1.8626e-09`, rank-biserial $r$ = `1.0`, significant = `True`
- **NSGA-II_vs_Min-Cost Flow (raw: infeasible)**: p-value = `1.8626e-09`, rank-biserial $r$ = `1.0`, significant = `True`
- **NSGA-II_vs_mcf_repaired (repaired: feasible)**: p-value = `1.8626e-09`, rank-biserial $r$ = `1.0`, significant = `True`
