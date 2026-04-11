# Previous Context

This file is a focused handoff note for the current state of the repo.

Scope:
- the architecture currently used in the project
- only the paper-facing benchmark path
- only the latest relevant results and what they seem to mean

Out of scope:
- the older synthetic experiments except where they explain why the paper experiments look the way they do
- unrelated result folders and exploratory one-off scripts

## 1. What We Are Actually Using

The reusable core lives in `kernel_learning/`.

Main idea:
- one sub-kernel per input feature
- learn a nonnegative weight `alpha_k` for each feature-level kernel
- sum them to get the final kernel

Formal shape:

`K(X) = sum_k alpha_k * K^(k)`

where:
- `K^(k)` is the Gram matrix produced by feature `k`
- `alpha_k` is learned

In code:
- `kernel_learning/kernel_network/network.py`
- `kernel_learning/sub_kernels/`
- `kernel_learning/losses/alignment.py`
- `kernel_learning/trainers/manual_gradient.py`
- `kernel_learning/methods/KernelRidgeClassifier.py`

### 1.1 Sub-kernels

Available sub-kernels:
- `RBFSubKernel`
- `LinearSubKernel`
- `PolynomialSubKernel`

For the paper benchmark path, the active architectures are:
- all RBF
- all Linear
- half RBF / half Linear

Polynomial is not part of the paper-facing benchmark.

### 1.2 Alpha parameterization

The model stores raw alpha parameters and transforms them into usable weights.

In the current experiments:
- `alpha_constraint="square"`
- `normalize_alphas=True`
- `alpha_init="random"`

So in practice:
- alphas are nonnegative
- alphas sum to roughly 1
- alpha mass can be interpreted as feature importance mass

### 1.3 What gets trained

The trainable parameters are:
- one raw alpha per feature
- one kernel parameter per feature

Examples:
- RBF feature `k` learns `gamma_k`
- Linear feature `k` learns `sigma_k`

### 1.4 Training objective

The paper-facing experiments train with `AlignmentLoss`.

That loss:
- builds a label kernel from the class labels
- centers both the predicted kernel and label kernel
- maximizes their normalized alignment

In practice the code minimizes negative alignment.

Important current fact:
- the optional L1 alpha penalty exists in the library
- it is not actually used in the paper-facing experiments

So the current feature selection story is mostly post hoc:
- train with alignment
- inspect learned alphas
- optionally prune features afterward

### 1.5 Evaluation

Training learns the kernel only.

Classification is done afterward with kernel ridge classification:
- fit on the learned train kernel
- predict on the test set via cross-kernel against the train set

This is implemented in:
- `kernel_learning/methods/KernelRidgeClassifier.py`

## 2. The Paper Benchmark Path

The repo currently uses the phrase "paper" to mean comparison against the SMKL paper:

- `experiments/classification/paper_smkl_2025_results.json`
- `experiments/classification/paper_smkl_2025_results.md`

There are three main paper-facing experiment stages:
- `test14.py`
- `test15.py`
- `test16.py`

### 2.1 Test14: direct UCI comparison

Purpose:
- run our local architectures on the 10 UCI datasets used by the SMKL paper
- mirror the paper loader/split behavior as closely as possible

Key file:
- `experiments/classification/test14.py`

Important detail:
- `test14` mirrors `_tmp_paper_repo/data/get_data.jl`
- the strict protocol is called `paper_strict`

What `paper_strict` means here:
- same UCI sources
- same binary label construction
- same one-shot split seed
- same standardization rule on numeric columns
- one split, not repeated CV

Architectures compared locally:
- `all_rbf`
- `all_linear`
- `half_rbf_half_linear`

Paper methods in `test14`:
- EasyMKL
- AverageMKL
- CKA
- SMKL

Important caveat:
- in `test14`, paper methods are reference numbers copied from the paper
- they are not retrained locally

### 2.2 Test15: two-pass alpha pruning

Purpose:
- use pass-1 alpha weights to build a feature mask
- retrain the same local models on the reduced feature set
- check whether pruning helps

Key file:
- `experiments/classification/test15.py`

Default pruning logic:
- source alpha vector = `all_rbf`
- keep the top features whose cumulative normalized alpha mass reaches `0.95`
- retrain on only those kept features

Important caveat:
- paper methods are still reference-only in `test15`
- they are not retrained locally

### 2.3 Test16: pruning plus real MKL baselines

Purpose:
- same two-pass pruning idea as `test15`
- but now EasyMKL, AverageMKL, and CKA are actually rerun locally with MKLpy

Key file:
- `experiments/classification/test16.py`

Paper methods in `test16`:
- `AverageMKL (MKLpy)`
- `EasyMKL (MKLpy)`
- `CKA (MKLpy)`

Still reference-only:
- paper SMKL / Algorithm1_SMKL

Current status:
- `test16` is not yet run across all 10 datasets in a finalized way
- the repo currently contains breast-cancer-only `test16` runs

## 3. Current Best Reading Of The Results

## 3.1 High-level pattern before the paper path

The older professor-data experiments already established the main pattern:
- the additive architecture is much stronger than full-vector RBF when irrelevant dimensions grow
- full-vector Linear is often the strongest baseline on Gaussian/tabular-like regimes
- alpha mass still drifts toward noise as noise dimensions increase

Interpretation:
- the additive structure is useful
- but the current training objective does not create strong intrinsic sparsity

This matters for the paper path because it predicts what we later see:
- alphas are informative
- but they are not cleanly sparse by themselves

## 3.2 Test14 results

Source summary:
- `experiments/classification/results_table_test14.md`

Best local architecture by dataset:

| Dataset | Best Local Result | Best Local Architecture | Paper SMKL |
|---|---:|---|---:|
| iris | 93.333 | Half RBF / Half Linear | 100.0 |
| wine | 97.222 | All Linear | 100.0 |
| breastcancer | 97.368 | All Linear | 98.3 |
| ionosphere | 95.775 | All RBF | 93.0 |
| spambase | 92.725 | All RBF | 90.9 |
| banknote | 92.727 | Half RBF / Half Linear | 100.0 |
| heart | 85.246 | All Linear | 93.4 |
| haberman | 82.258 | Half RBF / Half Linear | 67.7 |
| mammographic | 85.492 | Half RBF / Half Linear | 84.5 |
| parkinsons | 82.051 | All RBF | 89.7 |

My read:
- local models are competitive, not dominant
- best local result beats the paper SMKL number on `ionosphere`, `spambase`, `haberman`, and `mammographic`
- the strongest local architecture is often linear or partly linear, not always RBF
- this is a real signal: per-feature weighting is useful, but simple kernel families often match the data better

The cleanest local wins are:
- `ionosphere`: all RBF beats all paper baselines listed in the reference table
- `spambase`: all RBF slightly beats paper SMKL
- `haberman`: all local architectures beat paper SMKL by a wide margin
- `mammographic`: half-mixed slightly beats paper SMKL

The clear weak spots are:
- `iris`
- `banknote`
- `heart`
- `parkinsons`

## 3.3 Test15 results

Source summary:
- `experiments/classification/results_test15.md`

Aggregate mean delta from pruning:

| Model | Mean Pass1 (%) | Mean Pass2 (%) | Mean Delta (pp) |
|---|---:|---:|---:|
| Archi + All RBF | 86.847 | 86.972 | +0.125 |
| Archi + Half RBF/Half Linear | 87.812 | 87.280 | -0.533 |
| Archi + All Linear | 84.887 | 85.203 | +0.316 |
| Full-Vector RBF | 88.709 | 88.446 | -0.264 |
| Full-Vector Linear | 79.043 | 81.392 | +2.349 |

My read:
- pruning is not a universal win for our architectures
- it helps a little on average for `all_rbf` and `all_linear`
- it slightly hurts the mixed architecture on average
- the biggest beneficiary is `Full-Vector Linear`

What that probably means:
- the alpha vector is useful as a ranking signal
- but our additive models already exploit that information internally
- weaker baselines benefit more from being handed a cleaner feature subset

So the feature mask looks more useful as:
- a feature-ranking tool
- a preprocessing tool for simpler baselines

than as proof that the trained additive model itself has become intrinsically sparse.

## 3.4 Test16 results

Current scope:
- only breast cancer has `test16` runs in the repo

Important note:
- there are two breast-cancer `test16` result folders
- they are not the same configuration

Relevant runs:
- `f383ab98-026f-4d3c-87a2-70cce937c73d`: lighter exploratory run
- `90e8fa3f-9073-4b48-9fc1-f66234fbb7e0`: fuller run

Why they differ:
- the lighter run used `epochs=20`, `mkl_degrees=[1..5]`, `mkl_max_iter=200`
- the fuller run used `epochs=500`, `mkl_degrees=[1..10]`, `mkl_max_iter=500`

So they should not be treated as direct repeats.

### Latest fuller breast-cancer run

Using:
- `epochs=500`
- `mkl_degrees=1..10`
- selected features: `12 / 30`
- selected mass: about `0.962`

Results:

| Model | Pass1 (%) | Pass2 (%) | Delta (pp) |
|---|---:|---:|---:|
| Archi + All RBF | 93.860 | 91.228 | -2.632 |
| Archi + Half RBF/Half Linear | 92.982 | 92.105 | -0.877 |
| Archi + All Linear | 97.368 | 94.737 | -2.632 |
| Full-Vector RBF | 93.860 | 93.860 | 0.000 |
| Full-Vector Linear | 82.456 | 94.737 | +12.281 |
| AverageMKL (MKLpy) | 79.825 | 86.842 | +7.018 |
| EasyMKL (MKLpy) | 75.439 | 83.333 | +7.895 |
| CKA (MKLpy) | 92.105 | 93.860 | +1.754 |

My read:
- the local all-linear model is still the strongest pass-1 result here
- pruning hurts the stronger local models on this dataset
- pruning helps the weaker baselines a lot
- CKA becomes competitive after pruning, but still does not beat the best local pass-1 result

This reinforces the same interpretation as `test15`:
- alpha-based pruning finds signal
- but it is helping simpler or weaker downstream methods more than it helps the already-strong additive learner

## 4. What Seems To Be Going On

This is the current best summary of the evidence.

### Strong conclusions

- The additive feature-wise kernel idea is real and useful.
- The model is consistently more robust than full-vector RBF in noisy settings.
- On real UCI-style data, linear or partly linear versions are often the strongest local architecture.
- Learned alphas contain meaningful ranking information.

### Weak or not-yet-proven conclusions

- The current training setup does not yet prove strong feature-selection behavior during training.
- The model does not yet consistently beat the paper SMKL benchmark across the 10 datasets.
- The pruning story is not yet a clear "our model improves after pruning"; it is more nuanced than that.

### Best current interpretation

The repo is not showing:
- "RBF additive kernels dominate everything"

It is showing:
- "feature-wise kernel weighting is useful"
- "linear or mixed additive structures often fit tabular data best"
- "post hoc alpha ranking can identify a useful subset of features"
- "that subset can substantially help weaker baselines"

## 5. Practical Caveats

These matter when discussing results.

- `test14`, `test15`, and current `test16` mostly use `paper_strict`, which is one split.
- That is good for paper-table comparability.
- It is not enough by itself for strong statistical claims.

- `test16` is not yet consolidated across all 10 datasets.
- Right now it is still a breast-cancer-focused extension.

- `pyproject.toml` does not declare all benchmark dependencies.
- The paper path also needs `pandas`, `scikit-learn`, and for `test16`, `MKLpy`.

## 6. What To Run Next

If the goal is a clean next step, the most useful follow-up is:

1. Run `test16` across all 10 datasets with one fixed config.
2. Aggregate those results into a single report like `results_test15.md`.
3. Then compare:
   - our three architectures
   - full-vector RBF / Linear
   - MKLpy AverageMKL / EasyMKL / CKA
   - paper SMKL as reference only

That would turn the current direction into a complete paper-facing story.

## 7. Key Files

Core architecture:
- `kernel_learning/kernel_network/network.py`
- `kernel_learning/sub_kernels/linear.py`
- `kernel_learning/sub_kernels/rbf.py`
- `kernel_learning/losses/alignment.py`
- `kernel_learning/trainers/manual_gradient.py`
- `kernel_learning/methods/KernelRidgeClassifier.py`

Paper-facing experiments:
- `experiments/classification/test14.py`
- `experiments/classification/test15.py`
- `experiments/classification/test16.py`

Paper references:
- `experiments/classification/paper_smkl_2025_results.json`
- `experiments/classification/paper_smkl_2025_results.md`
- `_tmp_paper_repo/data/get_data.jl`

Result summaries:
- `experiments/classification/results_table_test14.md`
- `experiments/classification/results_test15.md`
- `experiments/classification/results.md`

## 8. Bottom Line

Current architecture:
- additive per-feature kernel learner with learned alpha weights

Current result:
- useful and competitive
- especially strong against full-vector RBF
- often best in linear or mixed form on real tabular datasets

Current limitation:
- alpha learning is informative but not yet strongly sparse by itself
- pruning helps baselines more than it helps the strongest local models

Current unfinished item:
- `test16` needs a full all-datasets run and summary before it should be treated as finalized
