# CLAUDE.md — Kernel Learning: Classification Benchmark Study

## Project Identity

**Branch:** `classification-v2`  
**Goal:** Prove empirical superiority of the learnable additive kernel network (KernelNetwork) against SotA kernel methods through rigorous benchmarking on 10 UCI datasets + synthetic stress tests.  
**Paper reference for baselines:** "Sparse Multiple Kernel Learning: Alternating Best Response and Semidefinite Relaxations", Bertsimas et al., TMLR 2025. https://arxiv.org/abs/2511.21890

---

## Architecture Under Test

### KernelNetwork (Main Model)

File: `kernel_learning/kernel_network/network.py`

```
K(X) = Σ_k  α_k · K_k(x_k)    where k indexes features (not kernels over full X)
```

Key properties:
- **Per-feature decomposition**: each sub-kernel K_k operates on *one scalar feature column* x_k, not the full feature vector. This is the fundamental structural innovation.
- **Learnable weights α_k** (squared-constrained, optionally normalized to sum 1) performing implicit feature selection.
- **Learnable sub-kernel parameters**: RBF has γ_k (bandwidth), Linear has σ_k (scale).
- **Trained via Kernel Alignment Loss**: pushes K_learned toward the ideal label kernel K_y.
- **Downstream classifier**: Kernel Ridge Regression (KernelRidgeClassifier).

Three tested architectures:
- `all_rbf`: all features use RBFSubKernel
- `all_linear`: all features use LinearSubKernel
- `half_rbf_half_linear`: first half RBF, second half Linear

### KAN_Kernel_NX (New — Approach 2)

File: `kernel_learning/kernel_network/KAN_Kernel.py`

```
K(u, v) = ψ(u) · ψ(v)    where ψ: R^p → R is a full-vector KAN
```

PSD-by-construction via outer product. Uses pykan B-spline KAN as the feature map.

---

## Sub-Kernels

| Class | Formula | Parameters |
|---|---|---|
| `RBFSubKernel` | K[i,j] = exp(-γ(x_i - x_j)²) | γ (learnable) |
| `LinearSubKernel` | K[i,j] = σ² · x_i · x_j | σ (learnable) |
| `PolynomialSubKernel` | K[i,j] = (x_i·x_j + c)^d | c, d (fixed) |

**Important difference between main branch and classification branch:** the classification branch has `random=True` kwargs support in `RBFSubKernel` and `LinearSubKernel` for random init. Main branch LinearSubKernel does NOT have sigma scaling — it's a pure outer product. This matters for fair comparison setup.

---

## Loss Functions

| Class | Formula | Notes |
|---|---|---|
| `AlignmentLoss` | -CKA(K_pred, K_y) | Centered kernel alignment, normalized |
| `AlignmentLossWithoutNormalization` | -Σ K_pred[i,j]·y[i]·y[j] | Unnormalized, simpler gradient |

Training uses `ManualGradientTrainer` with grad clipping (default 1.0).

---

## Datasets

### UCI Paper Benchmark (10 datasets)

Cached at: `experiments/classification/_paper_data_cache/`

| Dataset | n | p | Notes |
|---|---|---|---|
| breastcancer | 569 | 30 | Wisconsin diagnostic |
| ionosphere | 351 | 34 | Radar signals |
| spambase | 4601 | 57 | Email spam |
| banknote | 1372 | 4 | Wavelet features |
| haberman | 306 | 3 | Cancer survival |
| mammographic | 830 | 5 | Mammogram masses |
| parkinsons | 195 | 22 | Voice measurements |
| wine | 178 | 13 | Chemical analysis |
| iris | 150 | 4 | Flowers (binarized) |
| heart | 270 | 13 | Heart disease |

Protocol: `paper_strict` — single 80/20 split, seed=123, standardize train stats to test. Mirrors the paper's Julia data loader.

### Professor Synthetic Generator

File: `data/generators/professor/monni_simulated.py`

Generates Gaussian data with structured feature blocks:
- `p1` features unique to class -1
- `p2` features unique to class +1
- `pc` shared (opposite-sign) features
- `pn` pure noise features
- `snr`: signal-to-noise ratio

Seed protocol: `train_seed = startseed + 2*i + 1`, `test_seed = startseed + 2*i + 2`

### Other Synthetic Datasets
- `two_moons.py`: classic non-linear 2D dataset
- `concentric_circles.py`: inner vs outer circle
- `mixture_of_gaussians.py`: configurable GMM

---

## Baseline Competitors (classification-v2 adds these)

### From the SMKL Paper
| Method | Description |
|---|---|
| EasyMKL | MKL with L∞ constraint, closed-form |
| AverageMKL | Simple average of M base kernels |
| CKA | Centered Kernel Alignment MKL weighting |
| Algorithm 1 / SMKL | Sparse MKL with cardinality constraint |

These use HPK (Homogeneous Polynomial Kernels) degrees 1–10 over full feature vectors.

### New Baselines Added in classification-v2
| Method | Justification |
|---|---|
| SVM-RBF (sklearn) | Gold standard single-kernel classifier |
| SVM-Linear | Linear baseline |
| Kernel Ridge (RBF) | Same KRR framework, fixed RBF kernel over full X |
| Kernel Ridge (Linear) | Same KRR framework, fixed linear kernel |
| Gaussian Process Classifier | Probabilistic kernel method, strong on small datasets |

---

## Results — Legacy Reference (classification branch, test14)

| Dataset | EasyMKL | AverageMKL | CKA | SMKL | Our+AllRBF | Our+AllLinear | Our+HalfRBFHalf |
|---|---|---|---|---|---|---|---|
| iris | 100.0 | 100.0 | 96.7 | **100.0** | 90.0 | 86.7 | 93.3 |
| wine | 97.2 | 97.2 | 91.7 | **100.0** | 91.7 | 97.2 | 91.7 |
| breastcancer | 93.0 | 92.1 | 94.7 | **98.3** | 93.0 | 97.4 | 93.0 |
| ionosphere | 73.2 | 74.6 | 85.9 | 93.0 | **95.8** | 85.9 | 91.5 |
| spambase | 90.4 | 87.6 | 81.0 | 90.9 | **92.7** | 90.6 | 90.6 |
| banknote | 100.0 | 100.0 | 85.1 | **100.0** | 92.4 | 87.3 | 92.7 |
| heart | 85.2 | 85.2 | 86.9 | **93.4** | 73.8 | 85.2 | 80.3 |
| haberman | 61.3 | 62.9 | 66.1 | 67.7 | 80.6 | 79.0 | **82.3** |
| mammographic | 80.8 | 79.3 | 75.1 | 84.5 | 83.4 | 83.4 | **85.5** |
| parkinsons | 82.1 | 82.1 | 74.4 | **89.7** | 82.1 | 66.7 | 79.5 |

## Results — classification-v2 Benchmark (seed=123, 80/20 split, GPU, new arch)

Paper MKL columns from Bertsimas et al. TMLR 2025. Our results: single run with random α-init.

| Dataset | EasyMKL | AverageMKL | SMKL | Ours+RBF | Ours+Linear | Ours+Mixed | SVM-RBF | KRR-Linear |
|---|---|---|---|---|---|---|---|---|
| iris | 100.0 | 100.0 | 100.0 | **100.0** | **100.0** | **100.0** | **100.0** | **100.0** |
| wine | 97.2 | 97.2 | **100.0** | 91.7 | 94.4 | 91.7 | **100.0** | 97.2 |
| breastcancer | 93.0 | 92.1 | **98.3** | 93.0 | 94.7 | **95.6** | 93.9 | 95.6 |
| ionosphere | 73.2 | 74.6 | 93.0 | **93.0** | 84.5 | 88.7 | 94.4 | 88.7 |
| spambase | 90.4 | 87.6 | 90.9 | **93.1** | 91.6 | 90.6 | 94.5 | 90.9 |
| banknote | 100.0 | 100.0 | **100.0** | 91.3 | 88.4 | 91.6 | **100.0** | 98.5 |
| heart | 85.2 | 85.2 | **93.4** | 80.3 | **83.6** | 80.3 | **83.6** | **86.9** |
| haberman | 61.3 | 62.9 | 67.7 | 80.6 | 77.4 | **82.3** | 80.6 | 77.4 |
| mammographic | 80.8 | 79.3 | 84.5 | 85.0 | 83.4 | **86.0** | 88.1 | 83.4 |
| parkinsons | 82.1 | 82.1 | 89.7 | 92.3 | 66.7 | 87.2 | **94.9** | 59.0 |

**Key findings:**
- **Haberman**: Ours+Mixed **82.3%** vs SMKL 67.7% (+14.6pp). Best result across all methods.
- **Spambase**: Ours+RBF **93.1%** vs SMKL 90.9% (+2.2pp). Better than all paper MKL methods.
- **Ionosphere**: Ours+RBF **93.0%** = SMKL 93.0% (tie), while crushing EasyMKL/AverageMKL (73.2%, 74.6%).
- **Mammographic**: Ours+Mixed **86.0%** vs SMKL 84.5% (+1.5pp).
- **Parkinsons**: Our+RBF 92.3% vs KRR-RBF 94.9% (−2.6pp). Global RBF dominates here.
- **Banknote/Wine**: Global kernels dominate (linearly separable, per-feature decomposition under-fits).
- Random α-init causes run-to-run variance (±2pp typical). Multi-seed mean±std would improve paper credibility.

---

## Experiment Plan (classification-v2)

### EXP-1: Full UCI Benchmark (reproduce + extend)
- Reproduce legacy results with tuned hyperparameters
- Add SVM-RBF, KRR-global baselines
- Protocol: paper_strict (seed=123, 80/20)
- **Hypothesis:** Our per-feature kernel with learned α outperforms global kernels on datasets with heterogeneous feature scales and mixed informativeness (ionosphere, spambase, haberman, mammographic).

### EXP-2: Decision Boundary Visualization (2D synthetic)
- Two-moons, concentric circles, GMM
- Show decision boundaries for: Our model, SVM-RBF, SVM-Linear, KRR-RBF
- **Hypothesis:** Our model learns more adaptive boundaries via per-feature decomposition.

### EXP-3: Alpha Weight Analysis (Feature Selection)
- Professor data: known ground truth informativeness
- Plot alpha profiles, compare to true feature block structure
- Show that alpha correctly concentrates on informative features and suppresses noise
- **Hypothesis:** α_k → 0 for noise features, α_k > 0 for signal features.

### EXP-4: Robustness to Noise (SNR phase diagram)
- Sweep over (SNR, pn) grid using professor generator
- Compare accuracy of our model vs SVM-RBF vs KRR-global
- **Hypothesis:** Our model degrades more gracefully as noise increases due to learned feature pruning.

### EXP-5: Accuracy vs Complexity Curve
- Vary model complexity (number of features, training epochs, architecture)
- Show accuracy trajectory during training
- **Hypothesis:** Our model converges quickly and does not overfit due to per-feature decomposition.

---

## Output Artifacts

All under `experiments/classification_v2/`:

```
experiments/classification_v2/
├── data/                    # copied/generated datasets
├── results/                 # per-run manifest JSONs
├── figures/                 # publication-quality PNGs
│   ├── fig1_uci_benchmark_accuracy_table.png
│   ├── fig2_decision_boundaries.png
│   ├── fig3_alpha_profiles.png
│   ├── fig4_snr_phase_diagram.png
│   └── fig5_accuracy_vs_complexity.png
├── benchmark.py             # main UCI benchmark runner
├── decision_boundary.py     # 2D visualization
├── alpha_analysis.py        # alpha weight plots
├── snr_sweep.py             # noise robustness experiment
└── run_all.py               # orchestrates all experiments
```

---

## Bug Fixes Applied (classification-v2)

1. **network.py**: Forward pass replaced stack+einsum with in-place accumulation. Reduces GPU peak from O(p·n²) to O(n²) — critical for spambase (n=3680, p=57 → was 6GB stack, now 54MB).
2. **rbf.py / polynomial.py**: Added optional `y_col` parameter for cross-kernel computation K(X, Y).
3. **network.py**: Added `Y` parameter to `forward()` — enables true cross-kernel prediction instead of concatenate-then-slice trick.
4. **kernel_ridge_classifier.py**: `_cross_kernel` now calls `kernel(X_test, X_train)` directly — fixes catastrophic O((n_test+n_train)²) → O(n_test·n_train). Critical for decision boundary visualization with 40K grid points.

## Implementation Status

- [x] Phase 1: Intelligence gathering complete
- [x] Phase 2: Branch created (`classification-v2`), CLAUDE.md initialized
- [x] Phase 3: Research baselines (SVM, KRR, GP configs)
- [x] Phase 4: Implement benchmark suite
- [x] Phase 4: Run experiments (EXP-1, EXP-2, EXP-3, EXP-4)
- [ ] Phase 4: Generate figures
- [ ] Phase 4: Write narrative report

---

## Bug Notes

1. `LinearSubKernel.forward` in main branch has no `sigma` scaling (unlike classification branch). Cross-kernel computations are not supported in `KernelNetwork` (only self-kernels). These are by-design limitations, not bugs.
2. `KernelRidgeClassifier.predict` concatenates X_test and X_train and slices the cross-kernel — this works but computes unnecessary n×n entries. Acceptable for current dataset sizes.

---

## Commit Style

Format: `Type(scope): description`  
Types: `Feat`, `Fix`, `Exp`, `Viz`, `Doc`, `Refactor`  
No "Co-authored by Claude" in commits.
