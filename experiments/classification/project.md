# Kernel Learning Project Guide (Classification)

This document is a self-contained technical guide to what this project is doing, how the architecture works, and what each classification experiment tests.

It is written to match the current code in:

- `kernel_learning/`
- `data/generators/`
- `experiments/classification/`

## 1) What This Project Is Trying To Do

The project is learning a kernel function from data, instead of fixing one global kernel upfront.

The main design goal is:

1. Keep the kernel matrix valid and stable during training.
2. Learn which input features matter (via learned alpha weights).
3. Compare this learned additive kernel against standard full-vector baselines.

So the idea is not only "get high accuracy", but also "understand where the model puts weight", especially when irrelevant/noisy dimensions are added.

## 2) Formal Model Definition

### 2.1 Data and notation

Let:

- `X in R^(n x p)` be data with `n` samples and `p` features.
- `x^(k) in R^n` be feature column `k` (all samples for one coordinate).

For each feature `k`, there is a sub-kernel `kappa_k` producing:

- `K^(k) in R^(n x n)`, where `K^(k)_ij = kappa_k(x_i^(k), x_j^(k))`.

### 2.2 Sub-kernel formulas used in this repo

From `kernel_learning/sub_kernels/`:

1. RBF sub-kernel (`rbf.py`) on one scalar feature:
   `K^(k)_ij = exp(-gamma_k * (x_i^(k) - x_j^(k))^2)`
2. Linear sub-kernel (`linear.py`) on one scalar feature:
   `K^(k)_ij = sigma_k^2 * x_i^(k) * x_j^(k)`
3. Polynomial sub-kernel (`polynomial.py`) on one scalar feature:
   `K^(k)_ij = (sigma_k^2 * x_i^(k) * x_j^(k) + c)^d`

Each sub-kernel has its own learnable parameter (`gamma_k` or `sigma_k`).

### 2.3 Alpha parameterization and constraints

The network stores raw parameters `raw_alpha_k`, then maps them to usable nonnegative weights:

- `none`: `alpha_k = raw_alpha_k`
- `square`: `alpha_k = raw_alpha_k^2`
- `exp`: `alpha_k = exp(raw_alpha_k)`
- `softplus`: `alpha_k = softplus(raw_alpha_k)`
- `abs`: `alpha_k = |raw_alpha_k|`

If `normalize_alphas=True`, then:

`alpha_k <- alpha_k / (sum_j alpha_j + eps)`

So alphas become a simplex-like distribution over features.

### 2.4 Final kernel (additive architecture)

From `kernel_learning/kernel_network/network.py`, the learned kernel is:

`K(X) = sum_(k=1..p) alpha_k * K^(k)`

Implementation detail:

- per-feature kernels are stacked to `K_stack` of shape `(p, n, n)`,
- then combined via `torch.einsum('p,pij->ij', alphas, K_stack)`.

### 2.5 Symmetry enforcement property

Even after numeric operations, kernel symmetry is explicitly enforced:

`K <- 0.5 * (K + K^T)`

The code also computes relative asymmetry and warns if above tolerance.

This is one of the key "preserve properties" decisions in the architecture.

## 3) Training Objective and Pipeline

### 3.1 Alignment loss used in experiments

Most classification experiments use `AlignmentLoss` (`kernel_learning/losses/alignment.py`).

Given labels `y`, it builds label kernel:

`(K_y)_ij = 1[y_i == y_j]`

Both `K_pred` and `K_y` are centered, then normalized alignment is:

`A = <Kc, Kyc>_F / (||Kc||_F * ||Kyc||_F + eps)`

Training minimizes:

`L = -A`

So training pushes predicted kernel structure toward same-class/ different-class label structure.

### 3.2 Parameter update rule

With `ManualGradientTrainer` (`kernel_learning/trainers/manual_gradient.py`):

1. Compute `K_pred = model(X_train)`.
2. Compute loss (`AlignmentLoss` + optional alpha lasso).
3. Backpropagate.
4. Update parameters manually:
   `theta <- theta - lr * grad`

This updates:

- all sub-kernel params (`gamma`, `sigma`, etc.),
- `raw_alphas`.

### 3.3 Optional regularization

`KernelNetwork.get_lasso_penalty(lambda)` adds:

`lambda * sum_k |alpha_k|`

In most classification scripts here it is set to `0.0` (disabled), but support exists.

### 3.4 Classification and evaluation (Kernel Ridge Classifier)

`kernel_learning/methods/KernelRidgeClassifier.py` does multiclass KRR.

Given train kernel `K` and one-hot `Y`:

`Alpha_dual = (K + lambda_ridge * I)^(-1) Y`

Prediction for test points:

1. Build cross-kernel `K_test_train`.
2. Scores `S = K_test_train @ Alpha_dual`.
3. Predicted label = class with largest score.

## 4) Why This Architecture Is Useful

Main practical properties:

1. Feature-level decomposition:
   You can inspect one alpha per feature/sub-kernel.
2. Interpretability:
   Alpha mass tells where model trust is going.
3. Flexibility:
   Mixed kernel types per feature are possible.
4. Stability:
   Symmetry fix, optional alpha normalization, explicit parameter constraints.
5. Fair comparisons:
   Many experiments compare trained architecture vs untrained and full-vector baselines.

## 5) What `p` Means

`p` is the total number of feature columns currently given to the model.

It is always the width of the matrix passed to `KernelNetwork.forward`.

Examples:

1. Two-moons raw data is 2D, so `p=2`.
2. If you expand signal to 10 dimensions, then `p=10`.
3. If you then add `m=20` useless features, final `p=30`.
4. In professor data, `p = pc + p1 + p2 + pn`.

The model expects exactly one sub-kernel per feature, so `len(sub_kernels) == p`.

## 6) Data Generators (All Used in Classification)

### 6.1 General generators in `data/generators/`

1. `mixture_of_gaussians.py`
   Multiclass Gaussian components in arbitrary dimension.
2. `two_moons.py`
   Standard 2D two-moons binary data.
3. `concentric_circles.py`
   Binary inner vs outer circle.
4. `two_outer_circles_middle.py`
   Class 1 = two outer circles, class 0 = middle cloud.
5. `two_moons_modified.py`
   Variant of two-moons (kept in repo but not central in latest tests).

### 6.2 Professor Gaussian generator (core)

File: `data/generators/professor/monni_simulated.py`

This is the most important generator for tests 7-10.

#### Feature block layout

Total dimension:

`p = p1 + p2 + pc + pn`

Blocks:

1. `pc` shared informative features.
2. `p1` class `-1` specific informative features.
3. `p2` class `+1` specific informative features.
4. `pn` pure noise features.

#### Mean vectors

Two class means are constructed as:

- `mu_minus = [ +shared, +p1, 0*p2, 0*pn ]`
- `mu_plus  = [ -shared, 0*p1, +p2, 0*pn ]`

Interpretation:

1. Shared block flips sign between classes.
2. `p1` block only helps class `-1`.
3. `p2` block only helps class `+1`.
4. Noise block has zero mean in both classes.

#### SNR scaling logic

Let `diff = mu_plus - mu_minus`.
The code rescales means by:

`scale = sqrt(snr * sigma^2 / ||diff||^2)`

Then:

- `mu_minus <- scale * mu_minus`
- `mu_plus <- scale * mu_plus`

So effective separation is controlled by target `snr`.

#### Sampling

For each class, draw Gaussian points with std `sigma`:

- class `-1`: `N(mu_minus, sigma^2 I)`
- class `+1`: `N(mu_plus, sigma^2 I)`

Labels are `-1` and `+1` (not `0/1`), which is fine because the classifier one-hot encodes unique classes.

#### Seed schedule (important)

`generate_professor_split` uses:

1. train seed = `startseed + 2*i + 1`
2. test seed = `startseed + 2*i + 2`
3. val seed = `startseed + 2*iter_count + 1 + i` (if validation enabled)

So each iteration has an independent generated train/test pair.

#### Validation parameter

`include_validation=True` makes the generator also return `X_val, y_val`.
Most current tests keep it `False`.

### 6.3 Professor Cauchy generator

File: `data/generators/professor/monni_simulated_cauchy.py`

Same feature/mean structure as Gaussian professor generator, but class samples are drawn from standard Cauchy (scaled by `sigma`) around class locations.

Why harder:

1. Heavy tails create more extreme outliers.
2. Kernel distances become less stable because some points are very far.
3. Separation controlled by mean locations still exists, but sample-level noise is more aggressive.

`cauchy_clip` optionally clips values to `[-clip, clip]` for numerical stability.

## 7) Classification Experiments: File-by-File

Important: the function name `run_experiement` is intentionally misspelled in many files; this is current API in the repo.

### 7.1 `experiments/classification/run.py`

Single exploratory script:

1. Builds a mixed architecture (with `p=2` currently, so one RBF and one Linear).
2. Trains with alignment loss.
3. Prints parameter changes:
   alphas, raw alphas, gamma/sigma changes.
4. Compares:
   trained architecture, untrained copy, fixed additive RBF, full-vector RBF.
5. Visualizes 2D training data.

Purpose: sanity check that parameters actually move and compare trained vs controls.

### 7.2 `experiments/classification/modular_experiment.py`

Main modular benchmark runner.

Architectures trained:

1. Archi + All RBF
2. Archi + All Linear
3. Archi + Alternating RBF/Linear
4. Archi + Alternating RBF/Linear/Poly

Also evaluates untrained copies and full-vector RBF/Linear baselines.

Outputs:

1. data scatter
2. loss curves
3. test accuracy curves
4. final accuracy bars
5. final alpha scatter
6. `manifest.json`

### 7.3 `experiments/classification/test2.py`

Thin wrapper around `modular_experiment.run_experiement` using two-moons data.

Purpose: run modular comparison on nonlinear 2D dataset quickly.

### 7.4 `experiments/classification/test3.py`

Core "useless feature stress test".

Protocol:

1. Start from a signal dataset.
2. Optionally expand signal dimensions.
3. Add `m` useless random features.
4. Train `Archi + All RBF`.
5. Compare against full-vector RBF and full-vector Linear.

Tracks:

1. accuracy vs `m`
2. signal alpha mass vs noise alpha mass
3. per-`m` alpha scatter

Also includes `generate_from_xy_txt` to load `X` and `y` from text files.

### 7.5 `experiments/classification/test4.py`

Wrapper over test3 core using concentric circles as signal dataset.

### 7.6 `experiments/classification/test5.py`

Wrapper over test3 core using 2-class Gaussian signal generator.

### 7.7 `experiments/classification/test6.py`

Wrapper over test3 core using "two outer circles vs middle" signal dataset.

### 7.8 `experiments/classification/test7.py`

Professor Gaussian protocol (no manual feature append; `pn` already in generator).

For each `pn` and each `iter`:

1. Generate train/test from professor split seeds.
2. Train `Archi + All RBF`.
3. Evaluate full-vector RBF and full-vector Linear baselines.
4. Record alpha masses and per-run metrics.

Then averages over `iter_count` for each `pn`.

Outputs include:

1. per-`pn` alpha plot (mean over iterations)
2. `accuracy_vs_pn`
3. `alpha_mass_vs_pn`
4. manifest with per-run and summary metrics

### 7.9 `experiments/classification/test8.py`

Same data protocol as test7, but architecture under test is half RBF and half Linear sub-kernels.

### 7.10 `experiments/classification/test9.py`

Professor Gaussian with larger signal blocks and explicit feature-type tracking.

Adds:

1. optional feature shuffling (`shuffle_features=True` by default),
2. tracking of shuffled feature type per coordinate,
3. alpha mass breakdown by feature type:
   shared vs class-1-specific vs class+1-specific vs noise.

### 7.11 `experiments/classification/test10.py`

Same as test9 but architecture under test is all Linear sub-kernels.

Useful for testing "linear-only architecture vs same baselines" under identical shuffled feature-type setup.

### 7.12 `experiments/classification/test11.py`

Same spirit as test9 but uses Cauchy professor data.

Architecture under test: all RBF.

Adds `cauchy_clip` for stability.

### 7.13 `experiments/classification/test12.py`

Same Cauchy data protocol as test11, but mixed architecture:

1. half RBF, half Linear sub-kernels.

Alpha plot encoding is richer:

1. color = feature type (shared / p1 / p2 / noise)
2. marker shape = kernel type (RBF or Linear)

Also adds summary plot:

1. alpha mass by kernel type (RBF mass vs Linear mass).

## 8) What "Epoch", "Iter", "m", and "pn" Mean

There are three nested scales in many experiments.

1. `epoch`
   One optimization update cycle on a fixed train split.
2. `iter`
   One full train/test run on a newly generated split.
3. `m` or `pn`
   Difficulty/stress axis controlling irrelevant feature count.

Typical nested structure:

1. choose stress value (`m` or `pn`)
2. repeat `iter_count` times (different seeds/splits)
3. inside each iteration train for many epochs

So repeated console lines at the same `pn` are expected; they correspond to different iterations.

## 9) How Averaging Is Reflected in Results

For tests 7-12:

1. `runs` in `manifest.json` stores each iteration result.
2. `summary_per_pn` stores means/std over iterations.
3. Summary plots (`accuracy_vs_pn`, `alpha_mass_vs_pn`, etc.) are plotted from these averages.
4. Per-`pn` alpha plots are mean alpha vectors over iterations.

So yes, iteration averaging is done and visible in both numbers and plots.

## 10) Plot Interpretation Guide

### 10.1 Accuracy curves

If architecture is robust:

1. its accuracy decays slower as irrelevant features increase.
2. it stays above weak baselines at high noise.

### 10.2 Signal/noise alpha mass

Good feature selection behavior:

1. high signal alpha mass,
2. low noise alpha mass,
3. degradation should be gradual when stress increases.

### 10.3 Per-feature alpha scatter

Look for:

1. whether known noise coordinates receive small alpha,
2. whether important blocks keep nontrivial alpha mass,
3. in test12, whether RBF vs Linear marker groups shift by stress level.

### 10.4 Feature-type alpha mass breakdown (tests 9-12)

Lets you ask more specific questions:

1. Is model using shared block most?
2. Is it relying more on class-specific blocks?
3. Is noise mass growing too fast?

## 11) Full-Vector Baselines vs Additive Architecture

Full-vector baselines compute kernels on the whole vector directly:

1. full-vector RBF:
   `exp(-gamma * ||x_i - x_j||^2)`
2. full-vector Linear:
   `x_i^T x_j`

Your architecture is additive across features:

`K = sum_k alpha_k K^(k)`

So the experiments are directly testing:

1. whether adaptive per-feature weighting helps with noisy dimensions,
2. whether explicit feature-level kernel composition gives robustness or interpretability advantages.

## 12) Result Folder Convention

Most scripts save under:

`experiments/classification/results/<uuid>_<experiment_slug>/`

Typical contents:

1. PNG plots.
2. `manifest.json` with:
   config, per-run metrics, summary metrics, and artifact paths.

This makes runs reproducible and comparable.

## 13) Practical Run Commands

From project root in Windows PowerShell:

```powershell
.venv\Scripts\activate
python -m experiments.classification.test7
python -m experiments.classification.test8
python -m experiments.classification.test11
python -m experiments.classification.test12
```

## 14) Bottom Line

This repository implements a learnable additive kernel network with constrained/normalized alpha weights and per-feature sub-kernels.

The classification experiments are designed to answer two core questions:

1. Predictive question:
   Does the learned kernel compete with full-vector baselines under different data regimes?
2. Selection question:
   Does alpha mass concentrate on informative coordinates when irrelevant/noisy features increase?

The professor generators (Gaussian and Cauchy) are especially valuable because they give controlled, interpretable feature blocks (`shared`, `class-specific`, `noise`) and seed-stable repeated splits, which makes the alpha analysis meaningful.
