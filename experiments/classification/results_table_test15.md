# Test15 Results Table

Latest run per dataset for `test15_alpha_pruning_two_pass`.

| Model | Mean Pass1 (%) | Mean Pass2 (%) | Mean Delta (pp) | Improved | Degraded | Unchanged |
|---|---:|---:|---:|---:|---:|---:|
| Archi + All RBF | 86.847 | 86.972 | 0.125 | 4 | 3 | 3 |
| Archi + Half RBF/Half Linear | 87.812 | 87.280 | -0.533 | 2 | 4 | 4 |
| Archi + All Linear | 84.887 | 85.203 | 0.316 | 2 | 5 | 3 |
| Full-Vector RBF | 88.709 | 88.446 | -0.264 | 3 | 5 | 2 |
| Full-Vector Linear | 79.043 | 81.392 | 2.349 | 6 | 3 | 1 |

Heatmap: `experiments/classification/results_test15_delta_heatmap.png`
