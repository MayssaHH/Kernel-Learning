# Classification Tests Guide

This file explains the test scripts in `experiments/classification/` in a practical way:

1. What each test is trying to prove.
2. How each test is implemented.
3. Which configuration parameters matter most.
4. What plots and metrics each test writes.

The naming `run_experiement` (misspelled) is intentional in this codebase and is used consistently across these files.

## 1) Common Test Pattern

Most tests follow this structure:

1. Generate data (`X_train, y_train, X_test, y_test`) or split one generated dataset.
2. Train one or more kernel-network architectures with `AlignmentLoss` using `ManualGradientTrainer`.
3. Evaluate each learned kernel through `KernelRidgeClassifier`.
4. Evaluate fixed full-vector baselines:
   - Full-Vector RBF KRR
   - Full-Vector Linear KRR
5. Save metrics + plots + `manifest.json` under:
   `experiments/classification/results/<uuid>_<experiment_slug>/`

Shared low-level components:

1. `kernel_learning/kernel_network/network.py`
2. `kernel_learning/losses/alignment.py`
3. `kernel_learning/trainers/manual_gradient.py`
4. `kernel_learning/methods/KernelRidgeClassifier.py`

## 2) Quick Map of Scripts

1. `run.py`: one-off exploratory script (manual sanity checks).
2. `modular_experiment.py`: main modular benchmark (4 trained architectures + controls + baselines).
3. `test2.py`: runs modular experiment on two-moons.
4. `test3.py`: core useless-feature stress test (base implementation).
5. `test4.py`: test3 protocol on concentric circles.
6. `test5.py`: test3 protocol on Gaussian signal data.
7. `test6.py`: test3 protocol on two-outer-circles-middle data.
8. `test7.py`: professor Gaussian protocol, architecture under test = all RBF.
9. `test8.py`: professor Gaussian protocol, architecture under test = half RBF / half Linear.
10. `test9.py`: professor Gaussian with feature-type tracking + shuffled coordinates, architecture under test = all RBF.
11. `test10.py`: test9 setup, architecture under test = all Linear.
12. `test11.py`: test9 setup on Cauchy professor data, architecture under test = all RBF.
13. `test12.py`: test11 setup, architecture under test = half RBF / half Linear with marker-coded alpha plots.
14. `test13.py`: SNR x PN phase-diagram experiment with 5-model comparison.

## 3) `run.py`

### What it is

A direct, manual experiment script for quick sanity checks, parameter movement checks, and baseline comparison.

### Implementation highlights

1. Builds one architecture (mixed RBF/Linear by index for `p=2`).
2. Keeps an untrained deep copy as control.
3. Trains with alignment loss for many epochs.
4. Prints initial/final:
   - alphas,
   - raw alphas,
   - RBF gamma values,
   - linear sigma values.
5. Evaluates:
   - trained architecture KRR,
   - untrained architecture KRR,
   - fixed additive RBF architecture KRR,
   - full-vector RBF KRR.

### Why useful

It answers: "Are parameters actually changing, and does training matter?"

## 4) `modular_experiment.py`

### What it is

The general benchmark runner used for broader architecture comparisons.

### Compared trained architectures

1. Archi + All RBF
2. Archi + All Linear
3. Archi + Alternating RBF/Linear
4. Archi + Alternating RBF/Linear/Poly

### Additional controls

1. Untrained copies of each architecture.
2. Full-Vector RBF baseline.
3. Full-Vector Linear baseline.

### Main configuration knobs

1. `num_classes`, `samples_per_class`, `dimension`
2. `epochs`
3. `data_generator` and `data_generator_kwargs`
4. `run_uuid`, `results_root`, `save_results`, `show_plots`

### Plots written

1. `data_scatter_first2dims.png`
2. `loss_curves.png`
3. `test_accuracy_curves.png`
4. `final_accuracy_bars.png`
5. `final_alphas.png`

### Manifest content

1. full config snapshot
2. trained/untrained accuracies
3. baseline accuracies
4. loss and accuracy curves
5. final alphas
6. list of artifact paths

## 5) `test2.py`

### What it is

A thin wrapper that runs `modular_experiment.run_experiement` on two-moons data.

### Why it exists

Fast way to test the modular benchmark on a nonlinear binary dataset without rewriting config.

### Important config passed

1. `dataset_name="two_moons"`
2. `data_generator=generate_two_moons`
3. `num_classes=2`, `dimension=2`
4. `epochs=500`

## 6) `test3.py` (Core Useless-Feature Stress Test)

### What it is

The base implementation for "signal + useless features" experiments.

### Core idea

1. Start from signal data (`X_signal, Y`).
2. Optionally expand signal dimensions (`expand_signal_features`).
3. Append `m` useless random features (`append_useless_features`).
4. Train architecture under test (all RBF per feature).
5. Compare against full-vector RBF and full-vector Linear.
6. Track alpha mass concentration on signal dims vs noise dims.

### Extra utility in this file

`generate_from_xy_txt` loads `X` and `y` from text files. This is useful for external datasets.

### Main configuration knobs

1. `noise_feature_counts`: list of `m` values.
2. `signal_generator` + kwargs.
3. `num_signal_features`.
4. `allow_signal_expansion`, `signal_expansion_mode`, `signal_expansion_noise_std`.
5. `epochs`, `lr`, `lambda_ridge`, `full_vector_rbf_gamma`.

### Plots written

1. signal-data preview (`signal_data_first2dims` or `signal_data_1d`)
2. one alpha scatter per `m` (`alphas_m<m>.png`)
3. `accuracy_vs_useless_features.png`
4. `alpha_mass_vs_useless_features.png`

### Why this is the foundation

`test4`, `test5`, `test6` reuse this exact logic with different signal generators.

## 7) `test4.py`

### What it is

Wrapper around test3 core using `generate_concentric_circles` as signal source.

### Purpose

Checks useless-feature robustness on ring-structured nonlinear signal.

### Key defaults

1. `dataset_name="concentric_circles"`
2. `experiment_slug="test4_useless_features"`
3. `noise_feature_counts` includes large values up to 200 in main block.

## 8) `test5.py`

### What it is

Wrapper around test3 core with a Gaussian signal generator (`generate_gaussian_signal`).

### Purpose

Checks the same useless-feature stress protocol on a smoother Gaussian-separation signal.

### Key defaults

1. `dataset_name="gaussian_mixture"`
2. `experiment_slug="test5_useless_features"`
3. Custom signal generator controls `radius` and `noise_std`.

## 9) `test6.py`

### What it is

Wrapper around test3 core with `generate_two_outer_circles_with_middle`.

### Purpose

Signal class is a union of two outer circles versus middle class cloud, then adds useless dimensions.

### Key defaults

1. `dataset_name="two_outer_circles_middle"`
2. `experiment_slug="test6_useless_features"`
3. hard-ish geometry with configurable center/radius/middle spread.

## 10) `test7.py` (Professor Gaussian, All RBF)

### What it is

Professor-style split protocol with repeated iterations over `pn` values.

### Data source

`data/generators/professor/monni_simulated.py` (`generate_professor_split`).

### Model comparison

1. Archi + All RBF
2. Full-Vector RBF
3. Full-Vector Linear

### Loop structure

1. For each `pn` in `pn_values`
2. For each `iter` in `iter_count`
3. Generate split with deterministic seed schedule
4. Train architecture
5. Evaluate all models
6. Record alpha mass and accuracy
7. Aggregate mean/std over iterations

### Main config knobs

1. `pn_values`, `iter_count`
2. `p1`, `p2`, `pc`, `snr`, `sigma`, `startseed`
3. `epochs`, `lr`, `lambda_ridge`
4. `include_validation`, `n_validation`

### Plots

1. one data preview
2. one alpha plot per `pn`
3. `accuracy_vs_pn`
4. `alpha_mass_vs_pn`

## 11) `test8.py` (Professor Gaussian, Mixed Archi)

### What it is

Same protocol as test7, but architecture under test is half RBF / half Linear.

### Key implementation detail

`build_half_rbf_half_linear_kernel_network(dimension)` creates:

1. `ceil(p/2)` RBF sub-kernels
2. `floor(p/2)` Linear sub-kernels

### Comparison models

1. Archi + Half RBF/Half Linear
2. Full-Vector RBF
3. Full-Vector Linear

## 12) `test9.py` (Professor Gaussian, Shuffled Feature Types, All RBF)

### What it is

Professor Gaussian protocol with bigger signal blocks and explicit per-feature-type tracking.

### New things introduced

1. Feature categories:
   - shared
   - class_minus_only
   - class_plus_only
   - noise
2. Optional feature shuffling by deterministic `shuffle_seed_offset`.
3. Color-coded alpha plots by feature type after shuffle.
4. Additional alpha-mass-by-feature-type summaries.

### Why important

It moves from simple signal/noise split to richer attribution.

## 13) `test10.py` (Professor Gaussian, Shuffled Types, All Linear)

### What it is

Same setup as test9 but architecture under test is all Linear sub-kernels.

### Why useful

Directly tests whether purely linear additive kernel can compete in this structured setting.

### Outputs

Same style as test9:

1. per-`pn` alpha plot (feature-type colors)
2. accuracy summary
3. signal/noise alpha mass summary
4. per-feature-type alpha mass summary

## 14) `test11.py` (Professor Cauchy, Shuffled Types, All RBF)

### What it is

test9-style setup on Cauchy heavy-tailed data.

### Data source

`data/generators/professor/monni_simulated_cauchy.py`

### Additional config

`cauchy_clip` controls optional clipping for numerical stability.

### Why useful

Stresses robustness under outliers/heavy tails while retaining interpretable feature-type blocks.

## 15) `test12.py` (Professor Cauchy, Shuffled Types, Mixed Archi)

### What it is

test11-style setup with mixed architecture (half RBF / half Linear).

### Extra interpretability

Alpha plots encode two dimensions of meaning:

1. point color = feature type
2. marker shape = kernel type (RBF or Linear)

### Additional summary

Includes alpha mass by kernel type:

1. mean RBF alpha mass
2. mean Linear alpha mass

## 16) `test13.py` (SNR x PN Phase-Diagram, Multi-Model Comparison)

### What it is

A 2D sweep over:

1. `snr_values`
2. `pn_values`

for professor Gaussian data, comparing five models in one unified experiment.

### Compared models

1. Archi + All RBF
2. Archi + Half RBF/Half Linear
3. Archi + All Linear
4. Full-Vector RBF
5. Full-Vector Linear

### Implementation flow

For each `(snr, pn)`:

1. generate `iter_count` independent train/test splits
2. train all three architecture variants
3. evaluate all five models
4. collect per-run metrics
5. aggregate mean/std in `summary_per_snr_pn`
6. write per-setting alpha plots

Then it writes:

1. line plots sliced by SNR (`accuracy_vs_pn_snr...`)
2. alpha-mass plots by architecture and SNR
3. feature-type alpha-mass plots by architecture and SNR
4. accuracy heatmaps over SNR x PN for each model
5. advantage heatmaps vs best full-vector baseline

### Key defaults in `__main__`

1. `snr_values=[0.5, 1.0, 2.0, 4.0]`
2. `pn_values=[5, 10, 20, 50]`
3. `iter_count=5`
4. `epochs=300`

## 17) Important Configuration Switches Across Tests

### Plot behavior

1. `show_plots=True/False`
2. `show_plots_at_end=True/False`

For detached/background runs use:

1. `show_plots=False`
2. `show_plots_at_end=False`

### Reproducibility

1. `startseed` controls professor split seeds.
2. `shuffle_seed_offset` controls deterministic feature permutation seeds.
3. `run_uuid` controls result folder naming.

### Data hardness

1. `pn` increases irrelevant dimensions.
2. `snr` controls class-mean separation scale.
3. `sigma` controls noise scale.
4. Cauchy tests add heavy-tail stress.

## 18) Output and Manifest Conventions

Every mature test writes:

1. one result folder under `experiments/classification/results/`
2. plots as `.png`
3. one `manifest.json`

Manifest usually contains:

1. `config`: exact run parameters
2. `summary_*`: aggregated metrics
3. `runs`: per-iteration raw metrics
4. `artifacts.plots`: generated plot paths

## 19) Practical Advice When Running

1. Start with lower `iter_count` and `epochs` for quick validation.
2. Keep split protocol identical when comparing models.
3. Prefer reading `summary` metrics over single-run outputs.
4. Use per-feature alpha plots together with accuracy plots, not accuracy alone.

