# DECISIONS

One line per judgement call. "Spec" = the project brief.

## Scope and setup
- Deepfake track (Parts A2, B2, C2; scripts 02/04/07; figures 05/12/17) dropped at the owner's request to focus on the cyber-security side. No Kaggle credentials were present either.
- Freed compute went into cyber extensions: 4-class primer (B1), per-Volatility-plugin detectors (A1), DP noise extended to σ = 4, 8 and an informed weight-decay attacker (B3), real-detector-quality → policy analysis and a hypothetical TPR-scaling sweep (C1). Extra figures fig18, fig19.
- MalMem is loaded with `datasets.load_dataset("bvk/CIC-MalMem-2022")` (owner's request) and cached as `data/MalMem2022.parquet`.
- Python 3.12 venv via `uv`; PyTorch 2.11 + CUDA 12.8 wheels, because an RTX 5060 (sm_120) needs cu128.
- Git author is "Basil Ebinesar" with the GitHub noreply address of account `basilll007`. The `gh` CLI was not authenticated, so commits are local and pushing is a documented final step.
- Added `scripts/00_sanity_reproductions.py`, which is not in the spec layout: the two README reproductions are a separate, citable step.

## Data / Part A1
- `MalwareType` = prefix of `Category` (`Ransomware-*`, `Spyware-*`, `Trojan-*`, `Benign`).
- Binary identity = the SHA-256 inside `Filename` (`<Type>-<Family>-<sha256>-<dump>.raw`). 2,887 binaries, about 10.1 dumps each.
- Benign rows carry no binary identity (`Filename` = "Benign" for all 29,298), so each benign row is its own group. Benign dumps may still be correlated, which is a residual optimism in the group split.
- 19 hashes appear under two family labels. Groups are stratified by the group's first label, so no binary straddles two splits (overlap verified as 0).
- Group split = GroupShuffleSplit (70/15/15) run inside each family stratum, which keeps family balance. Random split = stratified by `Category`.
- Constant columns dropped (computed on the full table, label-free): `pslist.nprocs64bit`, `handles.nport`, `svcscan.interactive_process_services`. 52 features remain, so the MLP is 52→64→32→k, not 55→64→32→k.
- Count features = the 37 integer-typed, non-negative columns. `log1p` is applied to them, then StandardScaler fit on train only.
- Seeds 0, 1, 2 control the split partition **and** model initialisation, so the std includes split variance.
- MLP detector: Adam lr 1e-3, batch 512, weight decay 1e-5, at most 40 epochs, early stopping on val loss (patience 6), trained on the GPU.
- RF: 300 trees, unlimited depth. LR: lbfgs, C = 1, max_iter 3000.
- Permutation importance = drop in test macro-F1 (5 repeats) for the 4-class group-split RF and MLP, averaged over seeds. Own implementation, so the PyTorch MLP is handled the same way as sklearn models.
- LOTO detectors are trained on group-split train/val rows of Benign + the two seen types. They are evaluated on group-split test rows of the unseen type (same rows the known-type detector is scored on) and also on all rows of the unseen type.
- Extension: one MLP per Volatility plugin (9 telemetry sources). The C1 "weak detector" is the plugin with the lowest *validation* AUC among those with validation AUC > 0.6, so the useless single-feature `modules` detector is excluded. All plugin detectors are also run through C1, so nothing is cherry-picked.

## Part B (Neural Inverse Optimization)
- Formulation copied from Ch10 attempt 4: shared θ₀, T full-batch SGD steps, soft-label cross-entropy, unrolled `torch.autograd.grad(create_graph=True)` + `torch.func.functional_call`, Adam on the primer only.
- NIO loss normalisation: Σ_l MSE(θ_P, θ_ref) / MSE(θ_ref, θ₀), the same as ‖θ_P−θ_ref‖²/‖θ_ref−θ₀‖² per tensor. Reported as "relative weight error" = loss / 6 tensors, where 1.0 means no closer than θ₀. The raw Σ MSE is also logged.
- Inner (reference and twin) SGD lr = 0.5. Tuned on seed 0 over {0.1, 0.3, 0.5, 1.0}: 0.5 gives 99.4% binary reference accuracy at T = 20 without instability. Both reference and twin use the same recipe (the Ch10 threat model).
- NIO (Adam) lr = 0.1, the Ch10 value; it beat 0.01 and 0.05 in a 100-iteration pilot (P = 16, T = 20). 300 iterations as specified.
- Primer initialised N(0, I) in standardised feature space, with balanced labels (row i → class i mod k). The "Gaussian-noise primer" baseline is exactly this starting point, so it is a paired control.
- Soft-label variant: label logits initialised at 3·one-hot (starting near the balanced labels) and learned jointly. Run for binary at T = 20 only.
- Random-subset baseline is class-balanced (same label budget as the primer) and drawn from the training split.
- The P × T grid is run in full for the binary task (6 × 3 × 3 seeds). The 4-class task (P ≥ 4, since k = 4) uses T = 20 only.
- B1/B3 run on the group split, so held-out rows come from unseen binaries and the membership test is not confounded by sibling dumps.
- NIO unrolled loops run on CPU: about 2.4 s vs 5.5 s per 100 iterations on the RTX 5060, because tiny tensors are kernel-launch-bound. The GPU is used for detector training, DP per-example gradients and reference training in B3.
- Membership test changed during the run (first version discarded, never reported): a distance-to-closest-record test against a *fixed* size-matched train reference set S (|S| = |test|). Averaging over random subsamples made true members undetectable, so it had no power.
- Positive controls: near-copies of rows in S (sd 0.005, below the typical real NN spacing of about 0.08) and perturbed copies (sd 0.05). Null controls: real validation rows and Gaussian noise. Tests: binomial (share closer to train), Mann–Whitney (spec), paired Wilcoxon on log ratio.
- Structure test: cosine of the (malware − benign) class-mean difference and overlap of the top-15 |difference| features. The same metric is computed for the noise primer (null) and the random real subset (reference).

## B3 (defence)
- DP-SGD-style: per-example clipping C = 1, lot size 256 sampled each step, Gaussian noise σ·C on the summed lot gradient, T = 20, lr 0.5. **No privacy accounting, no ε claimed.**
- σ grid = spec {0, 0.5, 1, 2} plus {4, 8}, added because accuracy barely moved up to σ = 2.
- Weight decay on full-batch SGD: wd ∈ {0.01, 0.03, 0.1}. Larger values (0.5) collapsed accuracy to 50% in a pilot.
- Attacker default = plain recipe (does not know the defence). For weight decay an "informed attacker" whose twin uses the same wd is also reported.
- Extra leakage metric: loss-threshold membership-inference AUC of the released model (train rows vs equal-size test rows).
- Primer sizes P ∈ {16, 64}.

## Part C (Neural Input Optimization)
- SIR exactly as in Ch7: Euler, dt 0.1, 500 steps, S₀ 0.99, I₀ 0.01, state recorded before each update, peak = max over the 500 recorded I values.
- β = β₀(1 − 0.8x₁), γ = 0.05 + 0.45·x₂·TPR(τ), TPR/FPR = mean sigmoid((s − τ)/0.02), L = max I + λ(x₁² + x₂²) + μ·x₂·FPR(τ). β₀ 0.5, λ 0.05, μ 0.5 are illustrative assumptions.
- Scores s = MLP malware probability on group-split test rows (malware rows of the type; all benign test rows).
- Policy init z = 0 (x = 0.5). Adam lr 0.05 for 300 iterations. Problems are batched; Adam is element-wise, so a batch is identical to separate runs.
- "Default policy" for before/after = the Ch7 Experiment-1 manual setting (firewall 0.3, response 0.4) with τ = 0.5.
- The grid check uses the same soft TPR/FPR as the objective, so NIO and grid optimise exactly the same function.
- Pareto x-axis = deployment cost x₁² + x₂² (unweighted). The grid front is the lower envelope of grid policies with the best τ per (x₁, x₂).
- Time-varying policy: 10 windows of 50 steps for x₁ and x₂, τ fixed. Cost terms are averaged over windows, so a constant schedule has the fixed-policy cost. Smoothness penalty ρ = 0.05·Σ(Δx)². Both a cold start (z = 0) and a warm start (from the fixed optimum) are reported.
- Detector-quality sweep: γ uses q·TPR(τ), q ∈ {1, .9, .75, .5, .25, .1}. This is explicitly hypothetical; the real-detector version uses the 9 plugin detectors.

## Reporting
- All numbers in RESULTS.md are read from `results/*.csv|json` written by the scripts. mean ± std uses population std (ddof = 0) over the 3 seeds.
- Figures are made only from `results/` (`scripts/08_make_figures.py`). Colour-blind-safe categorical palette in a CVD-validated fixed order (adjacent-pair ΔE ≥ 8; scatter plots use at most 3 hues) and a one-hue sequential ramp for ordinal P/T/λ. No baked-in titles.
