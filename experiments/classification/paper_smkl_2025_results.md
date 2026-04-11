# Reference Results From Paper (SMKL, TMLR 2025)

This file stores benchmark numbers from:

- Paper: *Sparse Multiple Kernel Learning: Alternating Best Response and Semidefinite Relaxations*
- Link: https://arxiv.org/html/2511.21890
- Section: `5.2 Predictive Accuracy and Runtime`

Purpose:

1. Keep a local reference of paper results.
2. Make comparison against our runs immediate.

## Accuracy Table (Paper Table 2, test accuracy %)

Columns:

1. `EasyMKL`
2. `AverageMKL`
3. `CKA`
4. `Algorithm1_SMKL` (paper's method)

| Dataset       | EasyMKL | AverageMKL | CKA  | Algorithm1_SMKL |
|---------------|---------|------------|------|-----------------|
| iris          | 100.0   | 100.0      | 96.7 | 100.0           |
| wine          | 97.2    | 97.2       | 91.7 | 100.0           |
| breastcancer  | 93.0    | 92.1       | 94.7 | 98.3            |
| ionosphere    | 73.2    | 74.6       | 85.9 | 93.0            |
| spambase      | 90.4    | 87.6       | 81.0 | 90.9            |
| banknote      | 100.0   | 100.0      | 85.1 | 100.0           |
| heart         | 85.2    | 85.2       | 86.9 | 93.4            |
| haberman      | 61.3    | 62.9       | 66.1 | 67.7            |
| mammographic  | 80.8    | 79.3       | 75.1 | 84.5            |
| parkinsons    | 82.1    | 82.1       | 74.4 | 89.7            |

## Runtime Table (Paper Table 4, training time in seconds)

| Dataset       | EasyMKL | AverageMKL | CKA   | Algorithm1_SMKL |
|---------------|---------|------------|-------|-----------------|
| iris          | 0.030   | 0.002      | 0.026 | 0.022           |
| wine          | 0.045   | 0.002      | 0.029 | 0.005           |
| breastcancer  | 0.244   | 0.004      | 0.061 | 0.110           |
| ionosphere    | 0.072   | 0.003      | 0.036 | 0.070           |
| spambase      | 33.671  | 0.631      | 8.859 | 8.330           |
| banknote      | 0.337   | 0.006      | 0.301 | 0.519           |
| heart         | 0.062   | 0.003      | 0.035 | 0.188           |
| haberman      | 0.029   | 1.907      | 0.033 | 0.025           |
| mammographic  | 0.228   | 13.027     | 0.125 | 0.436           |
| parkinsons    | 0.038   | 0.002      | 0.028 | 0.245           |

## Sparse Kernel Count Table (Paper Table 3, number of non-zero kernel weights)

| Dataset       | EasyMKL | AverageMKL | CKA | Algorithm1_SMKL |
|---------------|---------|------------|-----|-----------------|
| iris          | 10      | 10         | 9   | 1               |
| wine          | 4       | 10         | 7   | 2               |
| breastcancer  | 3       | 10         | 7   | 2               |
| ionosphere    | 3       | 10         | 6   | 1               |
| spambase      | 10      | 10         | 7   | 5               |
| banknote      | 10      | 10         | 7   | 1               |
| heart         | 3       | 10         | 7   | 4               |
| haberman      | 8       | 10         | 8   | 1               |
| mammographic  | 7       | 10         | 7   | 1               |
| parkinsons    | 2       | 10         | 7   | 2               |

## Notes

1. These are copied from the paper's reported tables (not recomputed locally).
2. Use this file as the fixed reference when comparing our `test14` outputs.
3. If we later add official reproduction scripts for EasyMKL/AverageMKL/CKA, keep this file unchanged as the "paper baseline" and add a separate "our reproduction baseline" file.

## Fair Comparison Protocol (Implemented In `test14.py`)

To generate table-comparable numbers in this repo, run `test14` with:

- `protocol=paper_strict`
- `paper_seed=123`
- `train_ratio=0.8`
- `n_splits=1`

Example:

```bash
python -m experiments.classification.test14 --dataset breastcancer --architectures all_rbf --protocol paper_strict --paper-seed 123 --train-ratio 0.8 --n-splits 1
```

Expected total sample counts from the paper repo loader:

- `iris=150`, `wine=178`, `breastcancer=569`, `ionosphere=351`, `spambase=4601`
- `banknote=1372`, `heart=303`, `haberman=306`, `mammographic=961`, `parkinsons=195`
