# Recent Classification Results

This file documents the latest available results from the recent classification runs in:

`experiments/classification/results/`

The focus here is on tests `test7` to `test13` (professor-data family + phase-diagram test).

## 1) Run Registry (latest per test)

1. `test7`: `801a88b9-25ac-497c-875c-00f56d2ac645_test7_professor_data`
2. `test8`: `e33237ef-a7d3-4117-b9b5-574c35726c3e_test8_professor_data_half_rbf_half_linear`
3. `test9`: `2cc5aea4-289f-4a7f-b881-492e0d2cb388_test9_professor_data_shuffled_types`
4. `test10`: `9b501c0a-7e78-4a28-93e7-e3dc27bf10de_test10_professor_data_shuffled_types_all_linear`
5. `test11`: `d8926c1f-5bea-490a-92fc-64ee6e118c5a_test11_professor_data_cauchy_shuffled_types`
6. `test12`: `497084bd-b6fd-451e-8a20-baa02e6a5dbd_test12_professor_data_cauchy_mixed_shuffled_types`
7. `test13`: `00aac293-820d-4979-b4d4-962d0783c02a_test13_snr_pn_phase_diagram`

## 2) Quick Metric Snapshot by Test

Numbers below are from each run's `manifest.json`.

### Test 7 (Professor Gaussian, Archi + All RBF)

PN range: `5 -> 100`

1. `Archi + All RBF`: `0.7270 -> 0.6917`
2. `Full-Vector RBF`: `0.6810 -> 0.5525`
3. `Full-Vector Linear`: `0.7458 -> 0.6858`

Alpha mass (Archi + All RBF):

1. signal: `0.9518 -> 0.3774`
2. noise: `0.0482 -> 0.6226`

Interpretation:

1. Archi+RBF clearly beats Full-Vector RBF as `pn` grows.
2. Full-Vector Linear remains strongest in absolute accuracy.
3. Alpha allocation shifts strongly to noise at high `pn`.

### Test 8 (Professor Gaussian, Archi + Half RBF/Half Linear)

PN range: `5 -> 100`

1. `Archi + Half RBF/Half Linear`: `0.7333 -> 0.6810`
2. `Full-Vector RBF`: `0.6810 -> 0.5525`
3. `Full-Vector Linear`: `0.7458 -> 0.6858`

Alpha mass (mixed arch):

1. signal: `0.9418 -> 0.3261`
2. noise: `0.0582 -> 0.6739`

Interpretation:

1. Mixed arch beats Full-Vector RBF by a clear margin.
2. Full-Vector Linear still edges out mixed arch.
3. High-noise regime again drives substantial noise alpha mass.

### Test 9 (Professor Gaussian + shuffled feature types, Archi + All RBF)

PN range: `5 -> 100`

1. `Archi + All RBF`: `0.7128 -> 0.6645`
2. `Full-Vector RBF`: `0.6245 -> 0.5383`
3. `Full-Vector Linear`: `0.7410 -> 0.6953`

Alpha mass (Archi + All RBF):

1. signal: `0.9552 -> 0.4177`
2. noise: `0.0448 -> 0.5823`

Interpretation:

1. Shuffling does not break the relative pattern.
2. Archi+RBF remains much stronger than Full-Vector RBF.
3. Full-Vector Linear remains strongest overall.

### Test 10 (Professor Gaussian + shuffled types, Archi + All Linear)

PN range: `5 -> 100`

1. `Archi + All Linear`: `0.7315 -> 0.6953`
2. `Full-Vector RBF`: `0.6245 -> 0.5383`
3. `Full-Vector Linear`: `0.7410 -> 0.6953`

Alpha mass (Archi + All Linear):

1. signal: `0.9391 -> 0.3781`
2. noise: `0.0609 -> 0.6219`

Interpretation:

1. At high `pn`, Archi+AllLinear and Full-Vector Linear are effectively tied.
2. Linear family dominates this Gaussian-shuffled regime.

### Test 11 (Professor Cauchy + shuffled types, Archi + All RBF)

PN range: `5 -> 100`

1. `Archi + All RBF`: `0.6377 -> 0.5763`
2. `Full-Vector RBF`: `0.4997 -> 0.5000`
3. `Full-Vector Linear`: `0.5283 -> 0.5140`

Alpha mass (Archi + All RBF):

1. signal: `0.9023 -> 0.2645`
2. noise: `0.0977 -> 0.7355`

Interpretation:

1. Under heavy tails, Archi+RBF is still clearly above both full-vector baselines.
2. Full-Vector RBF is near chance-level in this setup.
3. Noise-alpha takeover is stronger than in Gaussian tests.

### Test 12 (Professor Cauchy + shuffled types, Archi + Half RBF/Half Linear)

PN range: `5 -> 100`

1. `Archi + Half RBF/Half Linear`: `0.5548 -> 0.5358`
2. `Full-Vector RBF`: `0.4997 -> 0.5000`
3. `Full-Vector Linear`: `0.5283 -> 0.5140`

Alpha mass (mixed arch):

1. signal: `0.7512 -> 0.1750`
2. noise: `0.2488 -> 0.8250`

Interpretation:

1. Mixed arch beats both full-vector baselines, but less strongly than test11's all-RBF arch.
2. Noise allocation becomes very dominant at large `pn`.

## 3) Test 13 (SNR x PN Phase Diagram)

Run: `00aac293-820d-4979-b4d4-962d0783c02a_test13_snr_pn_phase_diagram`

Grid:

1. `snr_values = [0.5, 1.0, 2.0, 4.0]`
2. `pn_values = [5, 10, 20, 50]`
3. `iter_count = 5` per cell

### Average accuracy across all 16 cells

1. `Archi + All RBF`: `0.6695`
2. `Archi + Half RBF/Half Linear`: `0.6782`
3. `Archi + All Linear`: `0.6955`
4. `Full-Vector RBF`: `0.5895`
5. `Full-Vector Linear`: `0.7059`

### Cell win counts (best mean accuracy in each SNR x PN cell)

1. `Full-Vector Linear`: `15 / 16` cells
2. `Archi + All Linear`: `1 / 16` cells
3. Others: `0` cell wins

### Per-SNR mean accuracy trend

SNR = `0.5`

1. Archi RBF: `0.5693`
2. Archi Mixed: `0.5798`
3. Archi Linear: `0.5983`
4. Full-RBF: `0.5251`
5. Full-Linear: `0.6074`

SNR = `1.0`

1. Archi RBF: `0.6225`
2. Archi Mixed: `0.6291`
3. Archi Linear: `0.6495`
4. Full-RBF: `0.5537`
5. Full-Linear: `0.6634`

SNR = `2.0`

1. Archi RBF: `0.6931`
2. Archi Mixed: `0.7072`
3. Archi Linear: `0.7209`
4. Full-RBF: `0.6012`
5. Full-Linear: `0.7341`

SNR = `4.0`

1. Archi RBF: `0.7930`
2. Archi Mixed: `0.7969`
3. Archi Linear: `0.8132`
4. Full-RBF: `0.6778`
5. Full-Linear: `0.8189`

### Hardest and easiest cells

1. Hardest: `snr=0.5, pn=50`, best-model accuracy `0.5995`
2. Easiest: `snr=4.0, pn=10`, best-model accuracy `0.8295`

### Alpha mass behavior (aggregated over grid)

1. Archi+AllRBF: signal avg `0.7476`, noise avg `0.2524`
2. Archi+Mixed: signal avg `0.7176`, noise avg `0.2824`
3. Archi+AllLinear: signal avg `0.7367`, noise avg `0.2633`

Interpretation:

1. In this phase-diagram setup, linear kernels are strongest overall.
2. Among architecture variants, all-linear performs best on average.
3. All architecture variants still strongly outperform Full-Vector RBF.

## 4) Cross-Test Summary

### Consistent patterns

1. Full-Vector RBF is the weakest baseline in recent professor-data runs.
2. Linear-based models are strongest in most Gaussian settings.
3. On Cauchy data, architecture variants remain above full-vector baselines, especially all-RBF in test11.
4. As `pn` increases, alpha mass shifts toward noise dimensions in all architectures.

### Current evidence of architecture strength

1. Strong, repeated margin over Full-Vector RBF across tests.
2. Competitive (sometimes near-tied) performance vs Full-Vector Linear.
3. Better robustness than baselines in heavy-tail (Cauchy) setups.

### Current limitation visible in results

1. Feature-selection behavior is not yet "noise-suppressing" at very high `pn` in these runs.
2. In test13, Full-Vector Linear dominates most SNR x PN cells.

## 5) Where to Inspect Raw Data

1. Per-test config and summary metrics:
   each run's `manifest.json`
2. Iteration-level details:
   `runs` array inside each manifest
3. Training progress logs:
   `experiments/classification/results/_detached_logs/*.out.log`
4. Visual summaries:
   PNG files in each run folder

