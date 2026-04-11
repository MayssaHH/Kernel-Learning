# Test15 Full Report

This report summarizes the completed `test15_alpha_pruning_two_pass` sweep across the 10 UCI datasets.

## What Test15 Does

1. Pass 1 trains models on all original features.
2. A feature mask is built from the source alpha vector (`alpha_source=all_rbf`) using cumulative alpha mass (`keep_mass=0.95`).
3. Pass 2 retrains the same local models on the reduced feature set.
4. Delta is reported as `Pass2 - Pass1` in percentage points.

Paper methods (EasyMKL, AverageMKL, CKA, SMKL) are still reference-only in Test15; they are not retrained here.

## Aggregate Model Summary

| Model | Mean Pass1 (%) | Mean Pass2 (%) | Mean Delta (pp) | Median Delta (pp) | Improved | Degraded | Unchanged |
|---|---:|---:|---:|---:|---:|---:|---:|
| Archi + All RBF | 86.847 | 86.972 | 0.125 | 0.000 | 4 | 3 | 3 |
| Archi + Half RBF/Half Linear | 87.812 | 87.280 | -0.533 | 0.000 | 2 | 4 | 4 |
| Archi + All Linear | 84.887 | 85.203 | 0.316 | -0.109 | 2 | 5 | 3 |
| Full-Vector RBF | 88.709 | 88.446 | -0.264 | -0.869 | 3 | 5 | 2 |
| Full-Vector Linear | 79.043 | 81.392 | 2.349 | 2.374 | 6 | 3 | 1 |

## Per Dataset / Per Model Results

| Dataset | Model | Pass1 (%) | Pass2 (%) | Delta (pp) | Selected Features (mean) | Selected Ratio (mean) | Selected Mass (mean) | Run UUID |
|---|---|---:|---:|---:|---:|---:|---:|---|
| banknote | Archi + All RBF | 92.364 | 92.000 | -0.364 | 2.00 | 0.500 | 1.000 | 1d4da433-aa7b-4ba8-b767-3706e5a595a8 |
| banknote | Archi + Half RBF/Half Linear | 90.909 | 86.909 | -4.000 | 2.00 | 0.500 | 1.000 | 1d4da433-aa7b-4ba8-b767-3706e5a595a8 |
| banknote | Archi + All Linear | 88.000 | 87.273 | -0.727 | 2.00 | 0.500 | 1.000 | 1d4da433-aa7b-4ba8-b767-3706e5a595a8 |
| banknote | Full-Vector RBF | 100.000 | 92.727 | -7.273 | 2.00 | 0.500 | 1.000 | 1d4da433-aa7b-4ba8-b767-3706e5a595a8 |
| banknote | Full-Vector Linear | 97.455 | 87.273 | -10.182 | 2.00 | 0.500 | 1.000 | 1d4da433-aa7b-4ba8-b767-3706e5a595a8 |
| breastcancer | Archi + All RBF | 93.860 | 91.228 | -2.632 | 12.00 | 0.400 | 0.962 | 3e4a0e40-148e-4997-85ee-86c50938d5de |
| breastcancer | Archi + Half RBF/Half Linear | 92.982 | 92.105 | -0.877 | 12.00 | 0.400 | 0.962 | 3e4a0e40-148e-4997-85ee-86c50938d5de |
| breastcancer | Archi + All Linear | 97.368 | 94.737 | -2.632 | 12.00 | 0.400 | 0.962 | 3e4a0e40-148e-4997-85ee-86c50938d5de |
| breastcancer | Full-Vector RBF | 93.860 | 93.860 | 0.000 | 12.00 | 0.400 | 0.962 | 3e4a0e40-148e-4997-85ee-86c50938d5de |
| breastcancer | Full-Vector Linear | 82.456 | 94.737 | 12.281 | 12.00 | 0.400 | 0.962 | 3e4a0e40-148e-4997-85ee-86c50938d5de |
| haberman | Archi + All RBF | 79.032 | 79.032 | 0.000 | 2.00 | 0.667 | 0.980 | b3d9155f-2b77-41c0-b039-a07d749df88a |
| haberman | Archi + Half RBF/Half Linear | 82.258 | 82.258 | 0.000 | 2.00 | 0.667 | 0.980 | b3d9155f-2b77-41c0-b039-a07d749df88a |
| haberman | Archi + All Linear | 80.645 | 83.871 | 3.226 | 2.00 | 0.667 | 0.980 | b3d9155f-2b77-41c0-b039-a07d749df88a |
| haberman | Full-Vector RBF | 69.355 | 80.645 | 11.290 | 2.00 | 0.667 | 0.980 | b3d9155f-2b77-41c0-b039-a07d749df88a |
| haberman | Full-Vector Linear | 79.032 | 82.258 | 3.226 | 2.00 | 0.667 | 0.980 | b3d9155f-2b77-41c0-b039-a07d749df88a |
| heart | Archi + All RBF | 78.689 | 78.689 | 0.000 | 8.00 | 0.615 | 0.981 | 1a945d22-d69c-4937-a4c2-1e534683b83c |
| heart | Archi + Half RBF/Half Linear | 78.689 | 78.689 | 0.000 | 8.00 | 0.615 | 0.981 | 1a945d22-d69c-4937-a4c2-1e534683b83c |
| heart | Archi + All Linear | 83.607 | 83.607 | 0.000 | 8.00 | 0.615 | 0.981 | 1a945d22-d69c-4937-a4c2-1e534683b83c |
| heart | Full-Vector RBF | 78.689 | 70.492 | -8.197 | 8.00 | 0.615 | 0.981 | 1a945d22-d69c-4937-a4c2-1e534683b83c |
| heart | Full-Vector Linear | 81.967 | 83.607 | 1.639 | 8.00 | 0.615 | 0.981 | 1a945d22-d69c-4937-a4c2-1e534683b83c |
| ionosphere | Archi + All RBF | 90.141 | 87.324 | -2.817 | 7.00 | 0.206 | 0.952 | 13a340de-c9de-4e60-b77c-8a6ff4fbab46 |
| ionosphere | Archi + Half RBF/Half Linear | 87.324 | 84.507 | -2.817 | 7.00 | 0.206 | 0.952 | 13a340de-c9de-4e60-b77c-8a6ff4fbab46 |
| ionosphere | Archi + All Linear | 87.324 | 85.915 | -1.408 | 7.00 | 0.206 | 0.952 | 13a340de-c9de-4e60-b77c-8a6ff4fbab46 |
| ionosphere | Full-Vector RBF | 81.690 | 88.732 | 7.042 | 7.00 | 0.206 | 0.952 | 13a340de-c9de-4e60-b77c-8a6ff4fbab46 |
| ionosphere | Full-Vector Linear | 74.648 | 84.507 | 9.859 | 7.00 | 0.206 | 0.952 | 13a340de-c9de-4e60-b77c-8a6ff4fbab46 |
| iris | Archi + All RBF | 90.000 | 93.333 | 3.333 | 2.00 | 1.000 | 1.000 | eddc0e63-3923-4bd4-8f6c-fd2bd9064fce |
| iris | Archi + Half RBF/Half Linear | 93.333 | 93.333 | 0.000 | 2.00 | 1.000 | 1.000 | eddc0e63-3923-4bd4-8f6c-fd2bd9064fce |
| iris | Archi + All Linear | 86.667 | 86.667 | 0.000 | 2.00 | 1.000 | 1.000 | eddc0e63-3923-4bd4-8f6c-fd2bd9064fce |
| iris | Full-Vector RBF | 100.000 | 100.000 | 0.000 | 2.00 | 1.000 | 1.000 | eddc0e63-3923-4bd4-8f6c-fd2bd9064fce |
| iris | Full-Vector Linear | 86.667 | 86.667 | 0.000 | 2.00 | 1.000 | 1.000 | eddc0e63-3923-4bd4-8f6c-fd2bd9064fce |
| mammographic | Archi + All RBF | 83.938 | 84.456 | 0.518 | 4.00 | 0.800 | 0.991 | f7010e98-5b4a-4c87-885d-fc8b45a2f577 |
| mammographic | Archi + Half RBF/Half Linear | 86.010 | 85.492 | -0.518 | 4.00 | 0.800 | 0.991 | f7010e98-5b4a-4c87-885d-fc8b45a2f577 |
| mammographic | Archi + All Linear | 83.420 | 83.420 | 0.000 | 4.00 | 0.800 | 0.991 | f7010e98-5b4a-4c87-885d-fc8b45a2f577 |
| mammographic | Full-Vector RBF | 77.202 | 81.347 | 4.145 | 4.00 | 0.800 | 0.991 | f7010e98-5b4a-4c87-885d-fc8b45a2f577 |
| mammographic | Full-Vector Linear | 82.902 | 86.010 | 3.109 | 4.00 | 0.800 | 0.991 | f7010e98-5b4a-4c87-885d-fc8b45a2f577 |
| parkinsons | Archi + All RBF | 76.923 | 79.487 | 2.564 | 8.00 | 0.364 | 0.957 | 902786ba-2d3d-4a1f-9e50-5543ca1e6410 |
| parkinsons | Archi + Half RBF/Half Linear | 84.615 | 84.615 | 0.000 | 8.00 | 0.364 | 0.957 | 902786ba-2d3d-4a1f-9e50-5543ca1e6410 |
| parkinsons | Archi + All Linear | 56.410 | 64.103 | 7.692 | 8.00 | 0.364 | 0.957 | 902786ba-2d3d-4a1f-9e50-5543ca1e6410 |
| parkinsons | Full-Vector RBF | 97.436 | 92.308 | -5.128 | 8.00 | 0.364 | 0.957 | 902786ba-2d3d-4a1f-9e50-5543ca1e6410 |
| parkinsons | Full-Vector Linear | 66.667 | 64.103 | -2.564 | 8.00 | 0.364 | 0.957 | 902786ba-2d3d-4a1f-9e50-5543ca1e6410 |
| spambase | Archi + All RBF | 91.857 | 92.508 | 0.651 | 37.00 | 0.649 | 0.951 | 7a754804-42a1-4034-bfe3-7160ed3fc0c9 |
| spambase | Archi + Half RBF/Half Linear | 90.337 | 90.445 | 0.109 | 37.00 | 0.649 | 0.951 | 7a754804-42a1-4034-bfe3-7160ed3fc0c9 |
| spambase | Archi + All Linear | 90.988 | 90.771 | -0.217 | 37.00 | 0.649 | 0.951 | 7a754804-42a1-4034-bfe3-7160ed3fc0c9 |
| spambase | Full-Vector RBF | 91.640 | 89.902 | -1.737 | 37.00 | 0.649 | 0.951 | 7a754804-42a1-4034-bfe3-7160ed3fc0c9 |
| spambase | Full-Vector Linear | 44.191 | 53.094 | 8.903 | 37.00 | 0.649 | 0.951 | 7a754804-42a1-4034-bfe3-7160ed3fc0c9 |
| wine | Archi + All RBF | 91.667 | 91.667 | 0.000 | 6.00 | 0.462 | 0.968 | da697598-4b3a-4f4b-8209-ba8699749168 |
| wine | Archi + Half RBF/Half Linear | 91.667 | 94.444 | 2.778 | 6.00 | 0.462 | 0.968 | da697598-4b3a-4f4b-8209-ba8699749168 |
| wine | Archi + All Linear | 94.444 | 91.667 | -2.778 | 6.00 | 0.462 | 0.968 | da697598-4b3a-4f4b-8209-ba8699749168 |
| wine | Full-Vector RBF | 97.222 | 94.444 | -2.778 | 6.00 | 0.462 | 0.968 | da697598-4b3a-4f4b-8209-ba8699749168 |
| wine | Full-Vector Linear | 94.444 | 91.667 | -2.778 | 6.00 | 0.462 | 0.968 | da697598-4b3a-4f4b-8209-ba8699749168 |

## Paper Table-2 Reference (Context Only)

| Dataset | EasyMKL | AverageMKL | CKA | SMKL |
|---|---:|---:|---:|---:|
| banknote | 100.0 | 100.0 | 85.1 | 100.0 |
| breastcancer | 93.0 | 92.1 | 94.7 | 98.3 |
| haberman | 61.3 | 62.9 | 66.1 | 67.7 |
| heart | 85.2 | 85.2 | 86.9 | 93.4 |
| ionosphere | 73.2 | 74.6 | 85.9 | 93.0 |
| iris | 100.0 | 100.0 | 96.7 | 100.0 |
| mammographic | 80.8 | 79.3 | 75.1 | 84.5 |
| parkinsons | 82.1 | 82.1 | 74.4 | 89.7 |
| spambase | 90.4 | 87.6 | 81.0 | 90.9 |
| wine | 97.2 | 97.2 | 91.7 | 100.0 |

## Artifacts

- CSV table: `experiments/classification/results_table_test15.csv`
- Delta heatmap: `experiments/classification/results_test15_delta_heatmap.png`
- Compact summary: `experiments/classification/results_table_test15.md`
