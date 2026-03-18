# Test 7 Detailed Explanation

This document explains:
1. How the professor-style data generator works (`data/generators/professor/monni_simulated.py`)
2. How `test7.py` uses that data for training and evaluation
3. What each key parameter controls
4. How to interpret the outputs/plots

The goal is to make both the **data generation logic** and the **testing pipeline** fully clear.

---

## 1) Big Picture

`test7.py` is a stress test for feature selection behavior under a specific synthetic setting your professor designed:

1. Some predictors are informative for both classes (shared, opposite sign)
2. Some predictors are informative only for class `-1`
3. Some predictors are informative only for class `+1`
4. Some predictors are pure noise (`pn`)

Then we increase `pn` (number of irrelevant predictors) and compare:

1. `Archi + All RBF` (your additive architecture with per-feature alphas)
2. Full-vector RBF baseline
3. Full-vector linear baseline

This lets you answer: as irrelevant predictors increase, does the architecture keep focusing alpha mass on true signal predictors?

---

## 2) Files and Responsibilities

### `data/generators/professor/monni_simulated.py`

- `simulate_data(...)`:
  Generates one dataset from the professor's model and optionally writes text files.
- `generate_professor_split(...)`:
  Generates one train/test (and optional validation) split using professor seed logic.
- `generate_professor_splits(...)`:
  Convenience wrapper to build multiple splits.

### `experiments/classification/test7.py`

- Loops over `pn_values` and `iter_count`.
- For each split:
  - trains `Archi + All RBF`
  - evaluates 2 baselines
  - records alpha mass on signal vs noise blocks
- Saves plots + `manifest.json`.

---

## 3) Professor Data Generator: Code-Level Explanation

## 3.1 Feature Block Design

In `simulate_data(...)`, total predictors are:

`p = p1 + p2 + pc + pn`

with this order in each sample vector:

1. `pc` shared predictors
2. `p1` class `-1` specific predictors
3. `p2` class `+1` specific predictors
4. `pn` pure noise predictors

This ordering is important because later in `test7.py` we compute:
- signal block size = `pc + p1 + p2`
- noise block size = `pn`

and measure alpha mass on each block.

---

## 3.2 Class Means Construction (`mu_minus`, `mu_plus`)

The generator builds class means as:

`mu_minus = [ +shared, +p1, 0*p2, 0*pn ]`
`mu_plus  = [ -shared, 0*p1, +p2, 0*pn ]`

Interpretation:

1. Shared predictors are informative for both classes, with opposite sign.
2. `p1` predictors only separate class `-1`.
3. `p2` predictors only separate class `+1`.
4. Noise predictors have zero mean for both classes, so they carry no class signal.

---

## 3.3 SNR Scaling

After constructing raw means, the generator rescales them so the target SNR is met:

1. Compute mean-difference vector: `diff = mu_plus - mu_minus`
2. Compute squared norm: `||diff||^2`
3. Compute scale:

`scale = sqrt( snr * sigma^2 / ||diff||^2 )`

4. Multiply both means by `scale`.

Result:

`||mu_plus - mu_minus||^2 / sigma^2 ~= snr`

This is why you see `snr_true` reported in `generate_professor_split(...)`.

---

## 3.4 Sampling

Each class is sampled from a Gaussian with:

1. Mean = `mu_minus` or `mu_plus`
2. Per-feature standard deviation = `sigma`
3. Equal(ish) class sizes:
   - `n1 = n // 2`
   - `n2 = n - n1`

Then:

1. `X = [X_minus; X_plus]`
2. `y = [-1 ... -1, +1 ... +1]`

Note: labels are intentionally `-1/+1` (professor style), not forced to `0/1`.

---

## 3.5 Optional File Output

`simulate_data(...)` can write:

1. `filex`: predictors `X`
2. `filey`: labels `y`
3. `filem`: 2-row matrix with `mu_minus` and `mu_plus`

If file args are `None`, it only returns tensors. This keeps it usable both for experiments and for external file generation.

---

## 4) Split Generation (`generate_professor_split`)

This function mirrors professor seed logic:

1. Train seed: `startseed + 2*i + 1`
2. Test seed: `startseed + 2*i + 2`
3. Val seed: `startseed + 2*iter_count + 1 + i` (if validation is requested)

For each iteration `i`, it returns:

1. `X_train`, `y_train`
2. `X_test`, `y_test`
3. `mu_minus`, `mu_plus`
4. seed values used
5. `snr_true` check

This avoids random split leakage and keeps train/test generation independent by seed.

---

## 5) Test 7 Pipeline (`test7.py`)

## 5.1 Outer Loops

`test7.py` loops over:

1. `pn_values` (noise dimension stress)
2. `iter_count` (repeated seeds for averaging)

So each `(pn, iter)` gives one complete train/test run.

---

## 5.2 Model Trained

For each split, `Archi + All RBF` is built as:

1. One RBF sub-kernel per feature dimension
2. `alpha_constraint="square"`
3. `normalize_alphas=True`
4. `alpha_init="random"`

Training uses:

1. `AlignmentLoss`
2. `ManualGradientTrainer`
3. configured `epochs`, `lr`

---

## 5.3 Baselines Evaluated

On same split:

1. Full-vector RBF KRR (`full_vector_rbf_gamma` fixed)
2. Full-vector linear KRR

This isolates architecture-vs-baseline behavior under increasing pure-noise dimensions.

---

## 5.4 Alpha Mass Measurement

After training Archi model:

1. `alphas = model._get_alphas()`
2. Signal mass:
   `sum(alphas[:pc+p1+p2])`
3. Noise mass:
   `sum(alphas[pc+p1+p2:])`

If model is selecting useful features, signal mass should stay high as noise grows.

---

## 5.5 Aggregation and Plots

Per `pn`, test7 computes mean/std over iterations for:

1. Archi accuracy
2. Full-vector RBF accuracy
3. Full-vector linear accuracy
4. Signal/noise alpha mass
5. SNR check

Then it saves:

1. `accuracy_vs_pn.png`
2. `alpha_mass_vs_pn.png`
3. `manifest.json` with full config and per-run details

---

## 6) What This Test Should Show

If behavior is good:

1. Archi accuracy degrades slowly (or stays strong) as `pn` grows
2. Signal alpha mass remains large
3. Noise alpha mass stays smaller than signal mass

If behavior is weak:

1. Accuracy drops quickly with `pn`
2. Noise alpha mass grows too much
3. Model starts weighting irrelevant predictors heavily

---

## 7) Important Notes and Caveats

1. `full_vector_rbf_gamma` is fixed:
   performance can depend strongly on gamma; if needed, add gamma tuning for a fairer baseline.
2. Labels are `-1/+1`:
   this is okay because classifier code uses unique labels and one-hot internally.
3. No predictor expansion:
   this respects professor instruction "do not expand predictors."
4. `pn` already defines irrelevant predictors:
   unlike test3/test4/test5, no extra manual noise append step is needed here.

---

## 8) Typical Run Configuration

Current `test7.py` default main run:

1. `p1=4`, `p2=4`, `pc=2`
2. `pn_values=[5,10,20,50,100]`
3. `n_train=n_test=400`
4. `snr=2.0`, `sigma=1.0`
5. `iter_count=10`

So signal block has:

`pc + p1 + p2 = 10` informative predictors

and noise block varies with `pn`.

---

## 9) Quick Mapping: Parameter Meaning

1. `pc`: shared informative predictors (opposite means)
2. `p1`: informative only for class `-1`
3. `p2`: informative only for class `+1`
4. `pn`: pure-noise predictors
5. `snr`: target signal-to-noise ratio in mean-separation sense
6. `sigma`: per-feature noise std for Gaussian sampling
7. `startseed`: base seed used in professor sequence
8. `iter_count`: number of repeated train/test seed pairs per `pn`

---

## 10) If You Want to Match Professor File Output Style

`generate_professor_split(..., output_prefix=...)` can also write:

1. `X_i`, `Y_i`, `means_i`
2. `XT_i`, `YT_i`, `meansT_i`
3. optional `XV_i`, `YV_i`, `meansV_i`

while still returning tensors for direct use in Python experiments.

---

## 11) How to Read the Console Output (Very Important)

Your logs look like:

`[pn=  5] [iter= 1/10] Epoch 100 | Archi + Half RBF/Half Linear alignment loss: -0.120681`

and:

`[pn=  5] [iter= 1/10] Final | Archi + Half RBF/Half Linear: 0.7025, Full-Vector RBF: 0.6725, Full-Vector Linear: 0.6975, Signal alpha mass: 0.8811, Noise alpha mass: 0.1189, SNR(true): 2.0000`

### 11.1 Meaning of each token

1. `pn=5`
   - This run is using 5 pure-noise predictors.
   - Total dimension is `pc + p1 + p2 + pn`.

2. `iter=1/10`
   - This is repetition #1 out of 10 independent train/test splits for this same `pn`.
   - Different iterations use different seeds, so metrics differ across iterations.

3. `Epoch 100`
   - Training progress inside this iteration.
   - One epoch = one full parameter update pass on that iteration's training set.

4. `alignment loss: -0.120681`
   - Training objective value at that epoch.
   - Here lower/more negative is generally better because loss is negative alignment.
   - It is not expected to be directly comparable across different `pn`/`iter` with strict equality.

5. `Final | ...`
   - End-of-iteration test metrics (for that exact `pn`, `iter` split).

6. `Archi + ...: 0.7025`
   - Test accuracy of the architecture under test.

7. `Full-Vector RBF / Full-Vector Linear`
   - Baseline test accuracies on the same split.

8. `Signal alpha mass`
   - Sum of learned alphas over known informative block (`pc+p1+p2` dims).

9. `Noise alpha mass`
   - Sum of learned alphas over pure-noise block (`pn` dims).
   - With normalized alphas, signal mass + noise mass is approximately 1.

10. `SNR(true): 2.0000`
    - Measured SNR from means after scaling.
    - Should match target SNR (up to tiny numerical precision).

---

### 11.2 Why the output looks "weird" but is actually expected

From your sample:

1. `iter=1` gives one set of accuracies and alpha masses.
2. `iter=2` gives different values.
3. `iter=3` gives different values again.

This is expected because each iteration uses different random train/test draws (different seeds).  
The point of `iter_count=10` is to estimate average behavior, not to expect identical runs.

So do not judge by one iteration alone. Use:

1. `summary_per_pn` in `manifest.json` (mean/std across iterations),
2. aggregated plots (`accuracy_vs_pn.png`, `alpha_mass_vs_pn.png`).

---

### 11.3 Relationship between `pn`, `iter`, and `epoch`

There are three nested loops conceptually:

1. Outer loop: choose one `pn` value.
2. Middle loop: run `iter_count` independent splits for that `pn`.
3. Inner loop: train for `epochs` updates on that split.

So:

1. `pn` changes experiment difficulty (more irrelevant predictors),
2. `iter` changes random split instance,
3. `epoch` changes training progress inside one split.

---

## 12) Quick Glossary (pn / iter / epoch)

1. `pn`: number of pure-noise predictors in data generation.
2. `iter`: one full train+test run with fixed seeds for that repetition.
3. `epoch`: one training step cycle inside an iteration.

Rule of thumb:

1. Compare `epoch` lines only within same `(pn, iter)`.
2. Compare `Final` lines across iterations to see variability.
3. Use summary means/std to compare methods robustly.

---

## 13) Detailed Training Lifecycle (Step-by-Step)

This section explains exactly what happens in one `test7` run at execution time.

### 13.1 Outer Control Flow in `test7.py`

For each `pn` in `pn_values`, and for each `iter` in `iter_count`:

1. Generate one train/test split from professor generator.
2. Build architecture model (`KernelNetwork` with all-RBF sub-kernels in test7).
3. Train model for `epochs` using full-batch alignment loss.
4. Fit KRR on top of the learned kernel and evaluate test accuracy.
5. Evaluate two full-vector baselines on the same split.
6. Compute signal/noise alpha mass.
7. Store per-run metrics.

After all iterations for one `pn` are done:

1. Compute mean/std metrics over iterations.
2. Store one summary row for this `pn`.

After all `pn` values are done:

1. Build aggregate plots from summary rows.
2. Save `manifest.json`.

---

### 13.2 What is the input data exactly (per iteration)

From `generate_professor_split(...)`:

1. `X_train`: shape `(n_train, p)`
2. `y_train`: shape `(n_train,)`, labels in `{-1, +1}`
3. `X_test`: shape `(n_test, p)`
4. `y_test`: shape `(n_test,)`

where:

`p = pc + p1 + p2 + pn`

Example with defaults and `pn=10`:

1. `pc=2, p1=4, p2=4, pn=10` -> `p=20`
2. `X_train` is `(400, 20)`
3. `X_test` is `(400, 20)`

No random split is performed inside `test7`; train/test are generated independently using professor seed schedule.

---

### 13.3 Model construction (what parameters are trainable)

For the architecture-under-test in `test7`:

1. Build `KernelNetwork` with one sub-kernel per feature (`p` sub-kernels).
2. Each sub-kernel is RBF with trainable `gamma`.
3. `raw_alphas` are trainable and transformed by:
   - `alpha_constraint="square"`
   - `normalize_alphas=True`

So trainable parameters include:

1. all sub-kernel parameters (`gamma` for each RBF sub-kernel),
2. all alpha parameters (`raw_alphas`).

---

### 13.4 What happens in one epoch

Inside `ManualGradientTrainer.train_epoch(...)`:

1. Forward kernel build:
   - call `model(X_train)`
   - output is a kernel matrix `K_pred` of shape `(n_train, n_train)`

2. Loss computation:
   - `AlignmentLoss(K_pred, y=y_train)`
   - internally builds label kernel `K_y` from class equality
   - centers both kernels
   - returns negative normalized alignment

3. Backpropagation:
   - `loss.backward()` computes gradients wrt `raw_alphas` and kernel params

4. Parameter update:
   - manual GD: `param -= lr * grad`
   - gradients zeroed each step

Important:

1. Training is full-batch (all train samples in every epoch).
2. No minibatch loader is used.
3. One epoch means one full forward/backward/update cycle.

---

### 13.5 Kernel matrix construction in `KernelNetwork`

Given `X_train` with shape `(n, p)`:

1. For each feature column `X_train[:, k]`, compute one sub-kernel matrix `K_k` of shape `(n, n)`.
2. Stack all sub-kernels into `K_stack` shape `(p, n, n)`.
3. Compute constrained/normalized alphas shape `(p,)`.
4. Weighted sum:
   - `K = sum_k alpha_k * K_k`
5. Enforce symmetry of `K`.

So every epoch optimizes a differentiable weighted additive kernel over per-feature sub-kernels.

---

### 13.6 What happens at "Final" of one iteration

After training epochs are done for that split:

1. Architecture accuracy:
   - `evaluate_kernel_model_with_krr(...)`:
     - fit KRR on `(X_train, y_train)` using learned kernel
     - predict on `X_test`
     - compute accuracy

2. Full-vector RBF baseline:
   - build full pairwise RBF kernel directly on vectors
   - solve KRR system
   - compute test accuracy

3. Full-vector linear baseline:
   - same KRR flow with linear full-vector kernel

4. Alpha mass:
   - signal mass = sum first `pc+p1+p2` alphas
   - noise mass = sum last `pn` alphas

The printed `Final` line is exactly these end-of-iteration test metrics.

---

### 13.7 KRR fitting mechanics (why you see stable numbers)

For any kernel method in this pipeline, KRR does:

1. Build training kernel matrix `K_train` shape `(n_train, n_train)`
2. Solve linear system:
   - `(K_train + lambda * I) alpha = Y_one_hot`
3. Build test-vs-train kernel matrix `K_test_train` shape `(n_test, n_train)`
4. Scores:
   - `scores = K_test_train @ alpha`
5. Predicted class = argmax score

So every reported accuracy is the result of this deterministic solve on that split.

---

### 13.8 Why iterations differ even with same `pn`

Within one fixed `pn`, iterations still differ because:

1. train/test data are regenerated with different seeds,
2. the optimization landscape changes with sampled data,
3. model initialization is random unless you globally lock torch seed before each run.

That is why `iter_count` exists: to estimate average behavior and variability.

---

### 13.9 Minimal timeline for one `(pn, iter)`

1. Generate `X_train, y_train, X_test, y_test`.
2. Build kernel network.
3. Epoch loop:
   - compute `K_pred`
   - compute alignment loss
   - backward
   - update params
4. Fit KRR on trained kernel.
5. Evaluate three accuracies.
6. Compute alpha masses.
7. Print one `Final` line.
8. Store run in manifest `runs`.

This repeats for each iteration, then gets aggregated in `summary_per_pn`.
