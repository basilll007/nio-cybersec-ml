# RESULTS — NIO for Cyber Defense on CIC-MalMem-2022

<!-- Rendered by scripts/09_render_results.py from results/*. Do not edit RESULTS.md by hand. -->

Scope: memory-forensics malware detection (CIC-MalMem-2022). The deepfake track (A2/B2/C2) was dropped at the owner's
request. All values are mean ± std over seeds 0, 1, 2 (population std); each seed re-draws the split *and* the model
initialisation.

## 1. TL;DR

- **Split honesty matters for the hard task, not the easy one.** Binary benign-vs-malware detection is saturated under
  both splits (group-split MLP accuracy 0.9998 ± 0.0002). 4-class type identification drops when all dumps of a
  binary are kept in one split: RF macro-F1 0.813 ± 0.007 (random) → **0.774 ± 0.010 (group)**.
  Detectors still catch an *unseen* malware type: leave-one-type-out MLP TPR is Ransomware 0.9995 ± 0.0007,
  Spyware 0.9953 ± 0.0034, Trojan 0.9998 ± 0.0003. 5 of 9 single Volatility plugins
  alone reach AUC ≥ 0.999.
- **A 2-row NIO primer retrains the released detector to 0.985 ± 0.007 accuracy** (full data 0.996 ± 0.001), against
  0.918 ± 0.036 for 2 random real rows and 0.362 ± 0.248 for the noise primer. Its mean accuracy exceeds a
  same-size random real subset at 6/6 primer sizes (binary, T = 20). The margin is large
  for P ≤ 16 (P = 16: 0.989 ± 0.004 vs 0.966 ± 0.030) and within seed noise at P = 64 (0.990 ± 0.004 vs
  0.987 ± 0.003).
- **The primer leaks structure, not records.** Primer rows sit 6.71 (median) from their nearest real
  record, while real records sit 0.109 from each other. The DCR membership test finds no signal: share closer to
  train 0.495, n = 378, binomial p = 0.6, Wilcoxon p = 0.66. The same test
  flags near-copies of training rows (share 0.922, p = 3.4e-32). Yet the primer's malware-minus-benign
  direction matches the real one (cosine 0.86 ± 0.06 at P = 16).
- **DP-style noise and weight decay do not stop primer extraction** of the detector's *function*. With σ = 2 the released
  model keeps 0.9858 ± 0.0033 accuracy and the primer twin still gets 0.963 ± 0.035. Undefended: 0.9958 ± 0.0010 /
  0.989 ± 0.004. Loss-based membership inference stays at chance (AUC 0.504–0.507) in every
  configuration.
- **Detector-in-the-loop NIO policies, checked by brute force.** With the real known-type and leave-one-type-out
  detectors, Adam through the differentiable SIR lands within +1.28% of the 21³ grid optimum in all
  378 runs (median -0.51%, i.e. usually *better* than the discrete grid). It cuts peak infection from
  9.74% (Ch7 default setting) to 1.071%. With the nine single-plugin detectors,
  187 of 1701 runs ended more than 1% above the grid optimum. Most are the uninformative
  `modules`-only detector, but misses also occur for pslist-, malfind-, psxview- and callbacks-only (worst
  +23.7%, callbacks-only, Spyware, λ = 0.01, μ = 0.1). Gradient descent from one start is local, so the grid check is necessary. A 10-window
  time-varying policy improves the objective by 4.17 ± 1.87%. Weaker detection makes the optimiser buy firewall: x₁
  goes from 0.468 (real detector) to 0.990 at 10% effective TPR.

## 2. What was developed

| component | file |
|---|---|
| MalMem loader (`datasets.load_dataset`), type/family/binary-hash parsing, log1p + scaler, random / group / LOTO splits | `src/data_malmem.py` |
| MLP detector (52→64→32→k), deterministic θ₀ | `src/models.py` |
| sklearn-style PyTorch trainer, metrics, permutation importance | `src/train.py` |
| Ch10 Neural *Inverse* Optimization: unrolled SGD (`create_graph=True`, `torch.func.functional_call`), normalised weight-matching loss, primer with fixed / learnable soft labels, DP-SGD-style reference training (per-example clipping via `torch.func.vmap`) | `src/nio_inverse.py` |
| Leakage audit: DCR membership test with calibrated positive/null controls, class-structure recovery, PCA | `src/leakage.py` |
| Batched differentiable SIR (Ch7 Euler scheme) + running-max version for grids | `src/sir.py` |
| Ch7 Neural *Input* Optimization with detector TPR/FPR in the loop, brute-force grid, Pareto sweep, time-varying policy | `src/nio_input.py` |
| Course README reproductions (Ch10 "abc", Ch7 SIR) | `scripts/00_sanity_reproductions.py` |
| A1 baselines + LOTO + per-plugin detectors + detector scores for Part C | `scripts/01_baselines_malmem.py` |
| B1 primer sweep + baselines + leakage | `scripts/03_primer_malmem.py` |
| B3 defences (DP-style σ, weight decay, informed attacker, loss-MIA) | `scripts/05_primer_defense.py` |
| C1 policy NIO (main, grid check, Pareto, time-varying, detector quality) | `scripts/06_policy_malware.py` |
| figures from `results/` only; RESULTS.md renderer | `scripts/08_make_figures.py`, `scripts/09_render_results.py` |

## 3. Work done (chronological, wall-clock from `results/runlog.csv`)

| step | wall-clock | finished | what ran |
|---|---|---|---|
| 00_sanity_reproductions | 34 s | 2026-11-23 23:56:16 | Ch10 abc primer + Ch7 SIR NIO; CPU |
| A1_baselines_malmem | 5.8 min | 2026-11-23 23:56:33 | 3 seeds x {random;group} x {binary;4class} x {LR;RF;MLP} + LOTO + perm. importance |
| B3_primer_defense | 10.1 min | 2026-11-24 00:00:01 | 78 NIO runs against DP-style / weight-decay defended references |
| B1_primer_malmem | 11.4 min | 2026-11-24 00:01:17 | 87 NIO runs; P sweep x T sweep x seeds; binary + 4-class; soft-label variant |
| C1_policy_malware | 1.6 min | 2026-11-24 00:03:50 | 99 detector configs; main + 21^3 grid + 20 lambda/mu combos + time-varying + q sweep |

Order of work: setup → reproductions → A1 → B1 → B3 → C1 → figures → RESULTS.md. Superseded runs were discarded and
re-run; they are not counted above. The first B1/B3 leakage test was replaced by the size-matched DCR test after its
positive control showed it had no power (see DECISIONS.md). Scripts that ran concurrently share the CPU, so their
wall-clock times overlap.

### Sanity reproductions (before extending the course code)

Ch10 NIO Dataset Primer, toy "abc" (README attempt 4 logic, seed 42, CPU):

| quantity | README | ours |
|---|---|---|
| weight difference @ iter 0 | 0.03518 | 0.03518 |
| weight difference @ iter 20 | 0.00559 | 0.00559 |
| weight difference @ iter 40 | 0.00390 | 0.00390 |
| weight difference @ iter 60 | 0.00227 | 0.00227 |
| weight difference @ iter 80 | 0.00138 | 0.00138 |
| weight difference @ iter 100 | 0.00111 | 0.00111 |
| weight difference @ iter 120 | 0.00100 | 0.00100 |
| weight difference @ iter 140 | 0.00092 | 0.00092 |
| weight difference @ iter 160 | 0.00087 | 0.00087 |
| weight difference @ iter 180 | 0.00082 | 0.00082 |
| final primer (argmax) | cabcac | cabcac |
| GPT 1 eval loss | 0.86926 | 0.86926 |
| GPT 2 (primer only) eval loss | 0.91245 | 0.91245 |
| reduction iter 0 → 180 | ≈ 97.7% | 97.7% |

Ch7 NIO Malware Dynamics (README NIO block, 200 Adam iterations, lr 0.02):

| quantity | README | ours |
|---|---|---|
| optimised firewall x₁ | 0.470 | 0.470 |
| optimised recovery x₂ | 0.493 | 0.493 |
| peak infection | 1.07% | 1.07% |
| final loss | 0.0339 | 0.0339 |
| manual policy (0.3, 0.4) peak | (not printed) | 6.32% |

Both match the README values to the printed precision.

## 4. Dataset

| statistic | value |
|---|---|
| rows (memory dumps) | 58,596 |
| features: raw / after dropping constants | 55 / 52 |
| constant columns dropped | `pslist.nprocs64bit`, `handles.nport`, `svcscan.interactive_process_services` |
| log1p-transformed count features | 37 |
| binary classes Benign : Malware | 29,298 : 29,298 (1 : 1) |
| 4-class counts Benign / Ransomware / Spyware / Trojan | 29,298 / 9,791 / 10,020 / 9,487 |
| 4-class imbalance (largest : smallest) | 3.09 : 1 |
| malware families | 15 |
| distinct malware binaries (SHA-256 in Filename) | 2,887 |
| dumps per binary (mean / max) | 10.15 / 20 |
| exact duplicate feature rows | 569 |
| random split train / val / test (mean over 3 seeds) | 41,016 / 8,790 / 8,790 |
| group split train / val / test (mean over 3 seeds) | 40,865 / 8,917 / 8,814 |
| face-detection fallback rate | n/a (deepfake track dropped) |

## 5. Methods

**Neural Inverse Optimization (Ch10 → B1/B3).** A reference MLP starts at θ₀ and takes T full-batch SGD steps
(lr η = 0.5) on the real training split: θ_ref = SGD_T(θ₀, D). A learnable primer (X_P ∈ ℝ^{P×52}, labels Y_P: fixed
balanced one-hot or softmax-parameterised soft labels) trains a twin from the *same* θ₀ with the same recipe, unrolled
with `torch.autograd.grad(create_graph=True)`:
θ_P^{(t+1)} = θ_P^{(t)} − η ∇_θ CE(f(X_P; θ_P^{(t)}), Y_P).
The NIO loss is L_NIO = Σ_l MSE(θ_P^{(l)}, θ_ref^{(l)}) / MSE(θ_ref^{(l)}, θ₀^{(l)}) over the 6 parameter tensors. This is
the per-tensor normalisation by ‖θ_ref − θ₀‖². We report *relative weight error* = L_NIO / 6, where 1.0 means "no closer
than θ₀". Adam (lr 0.1, 300 iterations) updates only (X_P [, Y_P]); the twin's weights are never optimised directly.
Threat model (as in Ch10): the auditor knows the architecture, θ₀, T and η, and sees θ_ref, but not the data.

**Neural Input Optimization (Ch7 → C1).** Policy x = σ(z) ∈ [0,1]³ (firewall x₁, response aggressiveness x₂, alert
threshold τ = x₃). The detector enters through differentiable rates computed from its held-out scores s:
TPR(τ) = mean_{malware of type} σ((s − τ)/0.02), FPR(τ) = mean_{benign} σ((s − τ)/0.02).
SIR: β = β₀(1 − 0.8x₁), γ = 0.05 + 0.45·x₂·TPR(τ),
S′ = −βSI, I′ = βSI − γI, R′ = γI (Euler, dt 0.1, 500 steps, S₀ 0.99, I₀ 0.01).
Loss: L = max_t I(t) + λ(x₁² + x₂²) + μ·x₂·FPR(τ). Detection is what triggers cleanup, and wrongly quarantining benign
hosts costs μ. **All constants are illustrative assumptions**, as in the Ch7 README. The time-varying policy uses
x₁(t), x₂(t) piecewise-constant over 10 windows, window-averaged costs, plus a 0.05·Σ(Δx)² smoothness penalty.

**Hyperparameters**

| block | setting |
|---|---|
| preprocessing | drop 3 constant cols; log1p on 37 count cols; StandardScaler fit on train |
| splits | random stratified 70/15/15 by family; group split 70/15/15 (GroupShuffleSplit per family stratum; group = binary SHA-256; benign rows singleton groups) |
| LR | lbfgs, C = 1, max_iter 3000 |
| RF | 300 trees, unlimited depth |
| MLP detector | 52→64→32→k ReLU, Adam 1e-3, batch 512, wd 1e-5, ≤ 40 epochs, early stop on val loss (patience 6), GPU |
| permutation importance | 5 repeats, Δ macro-F1 on test (4-class, group split) |
| B1 | P ∈ {2,4,8,16,32,64} (4-class: P ≥ 4), T ∈ {10,20,40} (4-class: T = 20), inner SGD lr 0.5, NIO Adam lr 0.1, 300 iterations, primer init N(0, I), soft-label variant at T = 20 |
| leakage | DCR vs fixed train subset S with \|S\| = \|test\|; controls: near-copies (sd 0.005), perturbed copies (sd 0.05), validation rows, noise; tests: binomial, Mann–Whitney, paired Wilcoxon |
| B3 | T = 20, P ∈ {16, 64}; DP-style: clip C = 1, lot 256, σ ∈ {0, 0.5, 1, 2, 4, 8}, **no privacy accounting**; weight decay ∈ {0.01, 0.03, 0.1}; plain vs wd-aware attacker |
| C1 | β₀ 0.5, λ 0.05, μ 0.5, temperature 0.02, Adam lr 0.05, 300 iterations, z₀ = 0; grid 21³; λ ∈ {0.01, 0.02, 0.05, 0.1, 0.2} × μ ∈ {0.1, 0.5, 1, 2}; 10 windows, ρ = 0.05; q ∈ {1, .9, .75, .5, .25, .1} |
| compute | RTX 5060 Laptop (CUDA 12.8) for detectors / DP gradients; CPU for the tiny unrolled NIO loops (faster than the GPU in a pilot; see DECISIONS.md) |

## 6. Results

### A1 — Baseline detectors

| task | split | model | accuracy | macro-F1 | ROC-AUC (4-class: OvR macro) |
|---|---|---|---|---|---|
| binary | random | LR | 0.9994 ± 0.0003 | 0.9994 ± 0.0003 | 0.9998 ± 0.0001 |
| binary | random | RF | 0.9999 ± 0.0001 | 0.9999 ± 0.0001 | 1.0000 ± 0.0000 |
| binary | random | MLP | 0.9999 ± 0.0002 | 0.9999 ± 0.0002 | 1.0000 ± 0.0000 |
| binary | group | LR | 0.9994 ± 0.0001 | 0.9994 ± 0.0001 | 0.9999 ± 0.0001 |
| binary | group | RF | 1.0000 ± 0.0001 | 1.0000 ± 0.0001 | 1.0000 ± 0.0000 |
| binary | group | MLP | 0.9998 ± 0.0002 | 0.9998 ± 0.0002 | 1.0000 ± 0.0000 |
| 4class | random | LR | 0.7499 ± 0.0018 | 0.6240 ± 0.0029 | 0.9097 ± 0.0003 |
| 4class | random | RF | 0.8757 ± 0.0048 | 0.8131 ± 0.0074 | 0.9705 ± 0.0014 |
| 4class | random | MLP | 0.7929 ± 0.0024 | 0.6894 ± 0.0032 | 0.9320 ± 0.0013 |
| 4class | group | LR | 0.7494 ± 0.0054 | 0.6239 ± 0.0108 | 0.9073 ± 0.0029 |
| 4class | group | RF | 0.8488 ± 0.0057 | 0.7738 ± 0.0097 | 0.9601 ± 0.0030 |
| 4class | group | MLP | 0.7686 ± 0.0098 | 0.6543 ± 0.0122 | 0.9204 ± 0.0057 |

Leave-one-malware-type-out (train on Benign + 2 types; detect the unseen third type; threshold 0.5):

| unseen type | model | TPR known-type detector | TPR unseen type (test rows) | TPR unseen type (all rows) | AUC unseen vs benign | FPR (benign test) |
|---|---|---|---|---|---|---|
| Ransomware | LR | 0.9993 ± 0.0010 | 0.9984 ± 0.0023 | 0.9991 ± 0.0003 | 0.9996 ± 0.0006 | 0.0004 ± 0.0002 |
| Ransomware | RF | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| Ransomware | MLP | 1.0000 ± 0.0000 | 0.9995 ± 0.0007 | 0.9997 ± 0.0002 | 1.0000 ± 0.0000 | 0.0002 ± 0.0002 |
| Spyware | LR | 0.9983 ± 0.0008 | 0.9935 ± 0.0042 | 0.9923 ± 0.0011 | 1.0000 ± 0.0000 | 0.0004 ± 0.0002 |
| Spyware | RF | 0.9998 ± 0.0003 | 0.9957 ± 0.0031 | 0.9940 ± 0.0000 | 0.9990 ± 0.0007 | 0.0000 ± 0.0000 |
| Spyware | MLP | 0.9996 ± 0.0006 | 0.9953 ± 0.0034 | 0.9937 ± 0.0005 | 1.0000 ± 0.0000 | 0.0003 ± 0.0002 |
| Trojan | LR | 1.0000 ± 0.0000 | 0.9981 ± 0.0027 | 0.9971 ± 0.0017 | 0.9999 ± 0.0002 | 0.0004 ± 0.0002 |
| Trojan | RF | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.9993 ± 0.0003 | 1.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| Trojan | MLP | 1.0000 ± 0.0000 | 0.9998 ± 0.0003 | 0.9987 ± 0.0001 | 1.0000 ± 0.0000 | 0.0001 ± 0.0001 |

Single-telemetry-source detectors (one MLP per Volatility plugin, binary, group split):

| Volatility plugin | # features | test ROC-AUC | TPR@0.5 | FPR@0.5 | accuracy | C1 weak detector |
|---|---|---|---|---|---|---|
| pslist | 4 | 1.0000 ± 0.0000 | 0.998 ± 0.000 | 0.003 ± 0.000 | 0.998 ± 0.000 |  |
| ldrmodules | 6 | 0.9999 ± 0.0001 | 0.997 ± 0.002 | 0.004 ± 0.003 | 0.996 ± 0.001 |  |
| handles | 12 | 0.9999 ± 0.0001 | 0.999 ± 0.001 | 0.003 ± 0.001 | 0.998 ± 0.001 |  |
| svcscan | 6 | 0.9999 ± 0.0001 | 0.999 ± 0.001 | 0.001 ± 0.000 | 0.999 ± 0.000 |  |
| dlllist | 2 | 0.9998 ± 0.0002 | 0.995 ± 0.003 | 0.005 ± 0.004 | 0.995 ± 0.002 |  |
| malfind | 4 | 0.9908 ± 0.0025 | 0.985 ± 0.004 | 0.030 ± 0.005 | 0.978 ± 0.004 |  |
| psxview | 14 | 0.9218 ± 0.0020 | 0.809 ± 0.010 | 0.138 ± 0.007 | 0.835 ± 0.003 |  |
| callbacks | 3 | 0.8741 ± 0.0073 | 0.890 ± 0.002 | 0.266 ± 0.003 | 0.812 ± 0.003 | **yes** |
| modules | 1 | 0.5190 ± 0.0026 | 0.054 ± 0.005 | 0.016 ± 0.000 | 0.518 ± 0.003 |  |

Top-15 permutation importances (4-class, group split, mean ± std over seeds):

| rank | RF feature | RF ΔmacroF1 | MLP feature | MLP ΔmacroF1 |
|---|---|---|---|---|
| 1 | `handles.nfile` | 0.0444 ± 0.0116 | `handles.nsection` | 0.0586 ± 0.0230 |
| 2 | `handles.nsemaphore` | 0.0299 ± 0.0023 | `handles.nfile` | 0.0575 ± 0.0282 |
| 3 | `handles.nkey` | 0.0151 ± 0.0025 | `ldrmodules.not_in_mem` | 0.0491 ± 0.0225 |
| 4 | `handles.nsection` | 0.0138 ± 0.0053 | `ldrmodules.not_in_load` | 0.0444 ± 0.0216 |
| 5 | `pslist.avg_threads` | 0.0108 ± 0.0044 | `handles.nsemaphore` | 0.0427 ± 0.0274 |
| 6 | `ldrmodules.not_in_mem_avg` | 0.0105 ± 0.0025 | `ldrmodules.not_in_init` | 0.0403 ± 0.0155 |
| 7 | `ldrmodules.not_in_load_avg` | 0.0088 ± 0.0016 | `handles.nthread` | 0.0403 ± 0.0310 |
| 8 | `handles.nmutant` | 0.0059 ± 0.0025 | `handles.nkey` | 0.0388 ± 0.0196 |
| 9 | `ldrmodules.not_in_init_avg` | 0.0052 ± 0.0028 | `ldrmodules.not_in_load_avg` | 0.0314 ± 0.0188 |
| 10 | `ldrmodules.not_in_load` | 0.0043 ± 0.0038 | `pslist.nppid` | 0.0311 ± 0.0193 |
| 11 | `svcscan.kernel_drivers` | 0.0037 ± 0.0012 | `handles.ntimer` | 0.0311 ± 0.0205 |
| 12 | `dlllist.avg_dlls_per_proc` | 0.0037 ± 0.0027 | `ldrmodules.not_in_mem_avg` | 0.0296 ± 0.0187 |
| 13 | `svcscan.shared_process_services` | 0.0034 ± 0.0015 | `handles.nmutant` | 0.0292 ± 0.0278 |
| 14 | `ldrmodules.not_in_init` | 0.0032 ± 0.0005 | `psxview.not_in_deskthrd` | 0.0284 ± 0.0163 |
| 15 | `malfind.protection` | 0.0032 ± 0.0028 | `handles.nevent` | 0.0270 ± 0.0252 |

Figures: fig01 (dataset), fig02 (4-class confusion, group split), fig03 (importance), fig04 (split honesty), fig19
(per-plugin detectors).
**Takeaway:** binary detection is saturated, and LOTO MLPs miss at most 0.47% of a never-seen type. MalMem's benign class
(one VM's normal behaviour) is trivially separable, so "novel malware" generalisation here is an upper bound. The honest
difficulty is *type attribution*, where the group split costs RF 0.813 ± 0.007 → 0.774 ± 0.010 macro-F1.
Handle counts (`handles.nfile`) dominate type attribution.

### A2 — Deepfake detector

Not run: the deepfake track was removed from scope at the project owner's request (cyber-security focus). No Kaggle
credentials were available either.

### B1 — Does the released MalMem detector leak its training data? (Neural Inverse Optimization)

Binary detector, T = 20 inner SGD steps (test accuracy of a twin retrained from θ₀ on each P-row dataset):

| P | NIO primer (fixed labels) | NIO primer (soft labels) | random real subset | Gaussian-noise primer | full data (θ_ref) | rel. weight error primer | rel. weight error subset |
|---|---|---|---|---|---|---|---|
| 2 | 0.985 ± 0.007 | 0.984 ± 0.009 | 0.918 ± 0.036 | 0.362 ± 0.248 | 0.996 ± 0.001 | 0.297 ± 0.229 | 0.910 ± 0.839 |
| 4 | 0.988 ± 0.002 | 0.971 ± 0.027 | 0.953 ± 0.023 | 0.362 ± 0.225 | 0.996 ± 0.001 | 0.244 ± 0.133 | 0.367 ± 0.113 |
| 8 | 0.988 ± 0.004 | 0.985 ± 0.009 | 0.978 ± 0.005 | 0.415 ± 0.252 | 0.996 ± 0.001 | 0.214 ± 0.136 | 0.814 ± 0.627 |
| 16 | 0.989 ± 0.004 | 0.988 ± 0.004 | 0.966 ± 0.030 | 0.513 ± 0.298 | 0.996 ± 0.001 | 0.140 ± 0.080 | 1.648 ± 1.737 |
| 32 | 0.988 ± 0.003 | 0.991 ± 0.003 | 0.986 ± 0.002 | 0.450 ± 0.182 | 0.996 ± 0.001 | 0.156 ± 0.052 | 0.133 ± 0.042 |
| 64 | 0.990 ± 0.004 | 0.990 ± 0.002 | 0.987 ± 0.003 | 0.410 ± 0.199 | 0.996 ± 0.001 | 0.201 ± 0.019 | 0.179 ± 0.020 |

ROC-AUC of the same twins:

| P | primer ROC-AUC | subset ROC-AUC | noise ROC-AUC | full-data ROC-AUC |
|---|---|---|---|---|
| 2 | 0.9941 ± 0.0025 | 0.9342 ± 0.0378 | 0.3206 ± 0.3243 | 0.9998 ± 0.0002 |
| 4 | 0.9954 ± 0.0008 | 0.9869 ± 0.0017 | 0.3028 ± 0.2382 | 0.9998 ± 0.0002 |
| 8 | 0.9956 ± 0.0015 | 0.9896 ± 0.0019 | 0.5065 ± 0.3279 | 0.9998 ± 0.0002 |
| 16 | 0.9960 ± 0.0014 | 0.9893 ± 0.0052 | 0.5883 ± 0.3684 | 0.9998 ± 0.0002 |
| 32 | 0.9959 ± 0.0017 | 0.9890 ± 0.0041 | 0.4473 ± 0.2330 | 0.9998 ± 0.0002 |
| 64 | 0.9961 ± 0.0015 | 0.9917 ± 0.0019 | 0.3533 ± 0.2394 | 0.9998 ± 0.0002 |

4-class detector, T = 20:

| P | NIO primer acc | NIO primer macro-F1 | random subset acc | noise acc | full data acc | full data macro-F1 | rel. weight error |
|---|---|---|---|---|---|---|---|
| 4 | 0.649 ± 0.010 | 0.425 ± 0.044 | 0.559 ± 0.097 | 0.214 ± 0.093 | 0.690 ± 0.004 | 0.525 ± 0.016 | 0.530 ± 0.071 |
| 8 | 0.667 ± 0.006 | 0.453 ± 0.045 | 0.579 ± 0.040 | 0.199 ± 0.068 | 0.690 ± 0.004 | 0.525 ± 0.016 | 0.470 ± 0.073 |
| 16 | 0.655 ± 0.015 | 0.435 ± 0.038 | 0.650 ± 0.010 | 0.169 ± 0.030 | 0.690 ± 0.004 | 0.525 ± 0.016 | 0.803 ± 0.595 |
| 32 | 0.658 ± 0.017 | 0.432 ± 0.026 | 0.654 ± 0.032 | 0.155 ± 0.048 | 0.690 ± 0.004 | 0.525 ± 0.016 | 0.641 ± 0.269 |
| 64 | 0.680 ± 0.010 | 0.461 ± 0.023 | 0.662 ± 0.007 | 0.165 ± 0.075 | 0.690 ± 0.004 | 0.525 ± 0.016 | 0.338 ± 0.248 |

Inner-step sweep (binary, fixed labels; cells = twin accuracy / relative weight error):

| T | full-data acc | P=2 acc / rel.err | P=4 acc / rel.err | P=8 acc / rel.err | P=16 acc / rel.err | P=32 acc / rel.err | P=64 acc / rel.err | NIO s/run |
|---|---|---|---|---|---|---|---|---|
| 10 | 0.989 ± 0.002 | 0.938 / 0.37 | 0.985 / 0.23 | 0.986 / 0.17 | 0.986 / 0.13 | 0.986 / 0.20 | 0.987 / 0.18 | 3.6 |
| 20 | 0.996 ± 0.001 | 0.985 / 0.30 | 0.988 / 0.24 | 0.988 / 0.21 | 0.989 / 0.14 | 0.988 / 0.16 | 0.990 / 0.20 | 7.0 |
| 40 | 0.997 ± 0.000 | 0.987 / 0.32 | 0.990 / 0.19 | 0.989 / 0.19 | 0.990 / 0.16 | 0.974 / 0.17 | 0.972 / 0.20 | 14.0 |

Leakage audit — distance-to-closest-record membership test (pooled over seeds and P):

| point set | n | median d_train | median d_held-out | median ratio | share closer to train | binomial p (>0.5) | Mann–Whitney p | paired Wilcoxon p |
|---|---|---|---|---|---|---|---|---|
| NIO primer, binary, T=10 (all P) | 378 | 6.810 | 6.796 | 1.000 | 0.495 | 0.6 | 0.48 | 0.38 |
| NIO primer, binary, T=20 (all P) | 378 | 7.296 | 7.312 | 1.000 | 0.495 | 0.6 | 0.52 | 0.66 |
| NIO primer soft labels, binary, T=20 | 378 | 6.654 | 6.677 | 1.000 | 0.492 | 0.64 | 0.51 | 0.47 |
| NIO primer, binary, T=40 (all P) | 378 | 7.644 | 7.655 | 1.000 | 0.500 | 0.52 | 0.52 | 0.33 |
| NIO primer, 4-class, T=20 (all P) | 372 | 10.921 | 10.938 | 1.000 | 0.513 | 0.32 | 0.51 | 0.47 |
| Gaussian-noise primer (NIO start), binary T=20 | 378 | 7.103 | 7.116 | 1.000 | 0.513 | 0.32 | 0.51 | 0.083 |
| **positive control**: near-copies of train rows (sd 0.005) | 192 | 0.035 | 0.075 | 0.458 | 0.922 | 1.3e-36 | 1.3e-40 | 3.4e-32 |
| positive control: perturbed train rows (sd 0.05) | 192 | 0.353 | 0.381 | 0.984 | 0.771 | 1.2e-14 | 4.2e-09 | 1.8e-20 |
| null control: real non-member rows (val) | 192 | 0.088 | 0.109 | 0.995 | 0.516 | 0.36 | 0.34 | 0.19 |
| null control: Gaussian noise | 192 | 7.073 | 7.089 | 0.999 | 0.526 | 0.26 | 0.48 | 0.45 |

Structure recovery: cosine between the (malware − benign) class-mean difference of the primer and of the real training
data, and overlap of the top-15 most different features (binary, T = 20):

| P | primer diff-cosine | primer top-15 overlap | random-subset diff-cosine | random-subset top-15 overlap | noise diff-cosine | noise top-15 overlap | 4-class primer diff-cosine |
|---|---|---|---|---|---|---|---|
| 2 | 0.76 ± 0.24 | 9.3 ± 3.1 | 0.88 ± 0.05 | 11.7 ± 1.7 | -0.05 ± 0.21 | 4.3 ± 1.2 | – |
| 4 | 0.82 ± 0.08 | 10.0 ± 0.8 | 0.86 ± 0.10 | 11.3 ± 1.2 | -0.01 ± 0.17 | 5.0 ± 0.0 | 0.61 ± 0.01 |
| 8 | 0.83 ± 0.03 | 10.3 ± 0.9 | 0.94 ± 0.04 | 13.3 ± 1.7 | -0.04 ± 0.23 | 4.7 ± 1.2 | 0.63 ± 0.05 |
| 16 | 0.86 ± 0.06 | 12.3 ± 1.2 | 0.94 ± 0.01 | 13.3 ± 0.5 | 0.05 ± 0.19 | 3.0 ± 0.8 | 0.59 ± 0.08 |
| 32 | 0.84 ± 0.06 | 11.0 ± 0.8 | 0.97 ± 0.01 | 13.7 ± 0.5 | -0.02 ± 0.09 | 3.3 ± 1.2 | 0.61 ± 0.08 |
| 64 | 0.72 ± 0.10 | 9.7 ± 1.2 | 0.96 ± 0.01 | 13.0 ± 0.8 | -0.08 ± 0.09 | 3.7 ± 1.2 | 0.80 ± 0.11 |

Recovered discriminative features (binary, T = 20, P = 16):

| seed | diff-cosine | top-15 overlap | sign agreement on real top-15 | recovered top features (in real top-15 order) |
|---|---|---|---|---|
| 0 | 0.78 | 12 | 1.00 | `handles.nsection`, `dlllist.avg_dlls_per_proc`, `handles.nmutant`, `handles.nevent`, `handles.nhandles`, `ldrmodules.not_in_load`, `ldrmodules.not_in_mem`, `pslist.avg_threads` |
| 1 | 0.88 | 11 | 1.00 | `dlllist.avg_dlls_per_proc`, `handles.nsection`, `handles.nthread`, `handles.nevent`, `handles.nmutant`, `pslist.avg_threads`, `ldrmodules.not_in_load`, `ldrmodules.not_in_mem` |
| 2 | 0.93 | 14 | 1.00 | `dlllist.avg_dlls_per_proc`, `handles.nsection`, `handles.nthread`, `handles.nmutant`, `handles.nevent`, `handles.nhandles`, `dlllist.ndlls`, `handles.avg_handles_per_proc` |

Cost of NIO vs normal training ("does it reduce total cost?"):

| task | T | NIO run (300 it., CPU), s | reference: T full-batch SGD steps (CPU), s | NIO / reference | A1 MLP (Adam, early stop, GPU), s |
|---|---|---|---|---|---|
| 4class | 20 | 6.4 | 0.306 | 21× | 4.0 |
| binary | 10 | 3.6 | 0.163 | 22× | 4.0 |
| binary | 20 | 7.0 | 0.331 | 21× | 4.0 |
| binary | 40 | 14.0 | 0.679 | 21× | 4.0 |

Figures: fig06 (convergence), **fig07 (primer vs baselines, key figure)**, fig08 (T sweep), fig09 (NN leakage), fig10
(PCA).
**Takeaway:** a released detector's weights *do* leak a compact, highly usable training set. 2–16 synthetic rows retrain
the detector to 0.989 ± 0.004 (P = 16) vs 0.966 ± 0.030 for 16 random real rows, and the primer encodes the real
class contrast. That set is far off the data manifold (primer row norm 8.53 vs real 5.98). It
is indistinguishable from noise in the record-level membership test, so the leakage is *population/structure* level, not
*record* level. NIO costs about 21× the T-step reference training (7.0 s vs 0.331 s),
so it does **not** reduce total training cost.

### B2 — Deepfake primer

Not run: deepfake track removed from scope (see A2).

### B3 — Defences: privacy / utility trade-off (P = 16)

| defence | attacker | released-model acc | loss-MIA AUC | NIO rel. weight error | primer-twin acc | struct diff-cosine | NN share closer to train (pooled) | NN Wilcoxon p (pooled) |
|---|---|---|---|---|---|---|---|---|
| none (plain SGD) | plain recipe | 0.9958 ± 0.0010 | 0.504 ± 0.006 | 0.152 ± 0.082 | 0.989 ± 0.004 | 0.86 ± 0.06 | 0.583 | 0.36 |
| DP σ=0 | plain recipe | 0.9857 ± 0.0034 | 0.507 ± 0.005 | 0.219 ± 0.263 | 0.944 ± 0.060 | 0.66 ± 0.41 | 0.542 | 0.65 |
| DP σ=0.5 | plain recipe | 0.9858 ± 0.0033 | 0.507 ± 0.005 | 0.273 ± 0.307 | 0.853 ± 0.190 | 0.61 ± 0.48 | 0.438 | 0.92 |
| DP σ=1 | plain recipe | 0.9858 ± 0.0033 | 0.507 ± 0.005 | 0.325 ± 0.313 | 0.887 ± 0.141 | 0.62 ± 0.46 | 0.604 | 0.15 |
| DP σ=2 | plain recipe | 0.9858 ± 0.0033 | 0.507 ± 0.005 | 0.378 ± 0.209 | 0.963 ± 0.035 | 0.69 ± 0.32 | 0.542 | 0.5 |
| DP σ=4 | plain recipe | 0.9855 ± 0.0033 | 0.507 ± 0.004 | 0.533 ± 0.073 | 0.981 ± 0.006 | 0.65 ± 0.12 | 0.562 | 0.36 |
| DP σ=8 | plain recipe | 0.9851 ± 0.0036 | 0.507 ± 0.004 | 0.722 ± 0.033 | 0.977 ± 0.005 | 0.50 ± 0.10 | 0.521 | 0.55 |
| wd=0.01 | plain recipe | 0.9957 ± 0.0010 | 0.504 ± 0.006 | 0.216 ± 0.077 | 0.989 ± 0.003 | 0.87 ± 0.06 | 0.500 | 0.46 |
| wd=0.01 | knows wd | 0.9957 ± 0.0010 | 0.504 ± 0.006 | 0.134 ± 0.070 | 0.989 ± 0.002 | 0.87 ± 0.06 | 0.562 | 0.26 |
| wd=0.03 | plain recipe | 0.9951 ± 0.0013 | 0.505 ± 0.007 | 0.485 ± 0.095 | 0.981 ± 0.013 | 0.80 ± 0.14 | 0.521 | 0.37 |
| wd=0.03 | knows wd | 0.9951 ± 0.0013 | 0.505 ± 0.007 | 0.137 ± 0.084 | 0.982 ± 0.009 | 0.80 ± 0.11 | 0.500 | 0.26 |
| wd=0.1 | plain recipe | 0.9908 ± 0.0011 | 0.505 ± 0.006 | 0.727 ± 0.061 | 0.969 ± 0.020 | 0.57 ± 0.21 | 0.458 | 0.8 |
| wd=0.1 | knows wd | 0.9908 ± 0.0011 | 0.505 ± 0.006 | 0.050 ± 0.047 | 0.988 ± 0.002 | 0.83 ± 0.09 | 0.521 | 0.47 |

Same, P = 64:

| defence | attacker | released-model acc | loss-MIA AUC | NIO rel. weight error | primer-twin acc | struct diff-cosine | NN share closer to train (pooled) | NN Wilcoxon p (pooled) |
|---|---|---|---|---|---|---|---|---|
| none (plain SGD) | plain recipe | 0.9958 ± 0.0010 | 0.504 ± 0.006 | 0.199 ± 0.018 | 0.989 ± 0.004 | 0.75 ± 0.13 | 0.531 | 0.13 |
| DP σ=0 | plain recipe | 0.9857 ± 0.0034 | 0.507 ± 0.005 | 0.250 ± 0.238 | 0.961 ± 0.038 | 0.67 ± 0.41 | 0.542 | 0.19 |
| DP σ=0.5 | plain recipe | 0.9858 ± 0.0033 | 0.507 ± 0.005 | 0.362 ± 0.241 | 0.937 ± 0.071 | 0.64 ± 0.43 | 0.490 | 0.67 |
| DP σ=1 | plain recipe | 0.9858 ± 0.0033 | 0.507 ± 0.005 | 0.407 ± 0.255 | 0.943 ± 0.063 | 0.64 ± 0.43 | 0.526 | 0.4 |
| DP σ=2 | plain recipe | 0.9858 ± 0.0033 | 0.507 ± 0.005 | 0.484 ± 0.164 | 0.968 ± 0.029 | 0.65 ± 0.35 | 0.568 | 0.014 |
| DP σ=4 | plain recipe | 0.9855 ± 0.0033 | 0.507 ± 0.004 | 0.554 ± 0.068 | 0.979 ± 0.012 | 0.64 ± 0.22 | 0.573 | 0.0049 |
| DP σ=8 | plain recipe | 0.9851 ± 0.0036 | 0.507 ± 0.004 | 0.665 ± 0.063 | 0.986 ± 0.002 | 0.49 ± 0.14 | 0.516 | 0.096 |
| wd=0.01 | plain recipe | 0.9957 ± 0.0010 | 0.504 ± 0.006 | 0.296 ± 0.036 | 0.968 ± 0.029 | 0.72 ± 0.12 | 0.521 | 0.28 |
| wd=0.01 | knows wd | 0.9957 ± 0.0010 | 0.504 ± 0.006 | 0.213 ± 0.038 | 0.975 ± 0.020 | 0.70 ± 0.10 | 0.516 | 0.27 |
| wd=0.03 | plain recipe | 0.9951 ± 0.0013 | 0.505 ± 0.007 | 0.546 ± 0.099 | 0.954 ± 0.051 | 0.65 ± 0.19 | 0.484 | 0.6 |
| wd=0.03 | knows wd | 0.9951 ± 0.0013 | 0.505 ± 0.007 | 0.199 ± 0.054 | 0.957 ± 0.046 | 0.65 ± 0.19 | 0.479 | 0.46 |
| wd=0.1 | plain recipe | 0.9908 ± 0.0011 | 0.505 ± 0.006 | 0.726 ± 0.090 | 0.959 ± 0.043 | 0.59 ± 0.08 | 0.521 | 0.18 |
| wd=0.1 | knows wd | 0.9908 ± 0.0011 | 0.505 ± 0.006 | 0.074 ± 0.049 | 0.989 ± 0.002 | 0.89 ± 0.02 | 0.536 | 0.046 |

Figure: fig11.
**Takeaway:** DP-style clipping and noise cost 1.0 points of released-model accuracy (0.9958 ± 0.0010 →
0.9858 ± 0.0033 at σ = 2); most of that comes from clipping and lot sampling, not from σ. They raise the attacker's
weight-matching error steadily (0.152 ± 0.082 → 0.378 ± 0.209 at σ = 2 → 0.722 ± 0.033 at σ = 8), because
the attacker's assumed full-batch recipe no longer matches. Primer usefulness does not fall in step. The twin still reaches
0.963 ± 0.035 at σ = 2 and recovers to ≥ 0.977 at σ ≥ 4 (table), with large seed-to-seed variance at small σ. Weight
decay only defeats a naive attacker (relative error 0.727 ± 0.061 at wd = 0.1). An attacker who knows wd matches
*better* than with no defence (0.050 ± 0.047), with twin accuracy 0.988 ± 0.002. Loss-based MIA
stays at chance in every configuration. Of 26 pooled DCR tests, 0 survive Bonferroni correction
(α = 0.0019); the smallest p is 0.0049 (DP σ=4, plain, P = 64). These defences target record-level membership,
which was not measurably leaking. They do not, and by design cannot, hide the population-level decision structure that NIO
extracts.

### C1 — Detector-in-the-loop network-defence policy (Neural Input Optimization)

Main run (λ = 0.05, μ = 0.5; default = Ch7 manual policy (0.3, 0.4) with τ = 0.5). Detectors: all-feature known-type MLP,
leave-one-type-out MLP, and the weakest informative single-plugin MLP (`callbacks`-only, AUC 0.874 ± 0.007):

| type | detector | detector AUC | x₁ firewall | x₂ response | x₃ τ | TPR@τ / FPR@τ | peak I (default policy) | peak I (NIO) | loss NIO | loss grid 21³ | gap NIO−grid |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Ransomware | known | 1.0000 | 0.468 ± 0.000 | 0.552 ± 0.000 | 0.679 ± 0.110 | 1.000 / 0.000 | 9.73% | 1.071% | 0.0369 ± 0.0000 | 0.0372 | -0.76% |
| Ransomware | novel (LOTO) | 1.0000 | 0.468 ± 0.000 | 0.552 ± 0.000 | 0.689 ± 0.109 | 1.000 / 0.000 | 9.74% | 1.071% | 0.0369 ± 0.0000 | 0.0372 | -0.79% |
| Ransomware | callbacks-only | 0.8959 | 0.813 ± 0.009 | 0.498 ± 0.004 | 0.792 ± 0.003 | 0.520 / 0.000 | 11.80% | 1.067% | 0.0562 ± 0.0005 | 0.0563 | -0.27% |
| Spyware | known | 1.0000 | 0.468 ± 0.001 | 0.551 ± 0.000 | 0.493 ± 0.183 | 1.000 / 0.000 | 9.74% | 1.071% | 0.0369 ± 0.0001 | 0.0372 | -0.76% |
| Spyware | novel (LOTO) | 1.0000 | 0.471 ± 0.002 | 0.552 ± 0.001 | 0.477 ± 0.073 | 0.995 / 0.000 | 9.84% | 1.072% | 0.0371 ± 0.0001 | 0.0374 | -0.94% |
| Spyware | callbacks-only | 0.8399 | 0.870 ± 0.035 | 0.465 ± 0.023 | 0.780 ± 0.009 | 0.454 / 0.000 | 14.49% | 1.058% | 0.0593 ± 0.0020 | 0.0598 | -0.71% |
| Trojan | known | 1.0000 | 0.468 ± 0.000 | 0.552 ± 0.000 | 0.665 ± 0.128 | 1.000 / 0.000 | 9.73% | 1.071% | 0.0369 ± 0.0000 | 0.0372 | -0.76% |
| Trojan | novel (LOTO) | 1.0000 | 0.467 ± 0.000 | 0.552 ± 0.000 | 0.503 ± 0.194 | 1.000 / 0.000 | 9.74% | 1.071% | 0.0369 ± 0.0000 | 0.0371 | -0.76% |
| Trojan | callbacks-only | 0.8883 | 0.845 ± 0.008 | 0.481 ± 0.005 | 0.792 ± 0.005 | 0.484 / 0.000 | 11.59% | 1.062% | 0.0579 ± 0.0005 | 0.0582 | -0.56% |

Pareto sweep (mean over 3 types × 3 seeds):

| λ | μ | x₁ / x₂ / τ (all-feature detector) | peak I | deploy cost | x₁ / x₂ / τ (callbacks-only) | peak I (callbacks-only) | false-quarantine cost (callbacks-only) |
|---|---|---|---|---|---|---|---|
| 0.01 | 0.1 | 0.486 / 0.556 / 0.608 | 1.003% | 0.546 | 0.895 / 0.367 / 0.677 | 1.004% | 0.0126 |
| 0.01 | 0.5 | 0.487 / 0.555 / 0.622 | 1.003% | 0.546 | 0.827 / 0.531 / 0.785 | 1.003% | 0.0000 |
| 0.01 | 1.0 | 0.489 / 0.554 / 0.628 | 1.003% | 0.546 | 0.817 / 0.550 / 0.793 | 1.003% | 0.0000 |
| 0.01 | 2.0 | 0.491 / 0.552 / 0.634 | 1.003% | 0.546 | 0.837 / 0.515 / 0.785 | 1.003% | 0.0000 |
| 0.02 | 0.1 | 0.482 / 0.555 / 0.603 | 1.011% | 0.540 | 0.859 / 0.400 / 0.646 | 1.010% | 0.0225 |
| 0.02 | 0.5 | 0.482 / 0.555 / 0.618 | 1.011% | 0.540 | 0.843 / 0.497 / 0.786 | 1.010% | 0.0000 |
| 0.02 | 1.0 | 0.483 / 0.554 / 0.624 | 1.011% | 0.540 | 0.833 / 0.515 / 0.792 | 1.011% | 0.0000 |
| 0.02 | 2.0 | 0.484 / 0.553 / 0.631 | 1.011% | 0.540 | 0.841 / 0.502 / 0.784 | 1.011% | 0.0000 |
| 0.05 | 0.1 | 0.467 / 0.552 / 0.597 | 1.071% | 0.523 | 0.663 / 0.431 / 0.398 | 1.080% | 0.1149 |
| 0.05 | 0.5 | 0.468 / 0.552 / 0.612 | 1.071% | 0.523 | 0.843 / 0.482 / 0.788 | 1.062% | 0.0000 |
| 0.05 | 1.0 | 0.468 / 0.551 / 0.619 | 1.071% | 0.523 | 0.840 / 0.486 / 0.791 | 1.064% | 0.0000 |
| 0.05 | 2.0 | 0.468 / 0.551 / 0.626 | 1.071% | 0.523 | 0.840 / 0.487 / 0.783 | 1.064% | 0.0000 |
| 0.1 | 0.1 | 0.444 / 0.545 / 0.590 | 1.285% | 0.494 | 0.570 / 0.493 / 0.379 | 1.338% | 0.1312 |
| 0.1 | 0.5 | 0.444 / 0.545 / 0.608 | 1.285% | 0.494 | 0.822 / 0.488 / 0.786 | 1.275% | 0.0000 |
| 0.1 | 1.0 | 0.444 / 0.545 / 0.615 | 1.291% | 0.494 | 0.822 / 0.488 / 0.788 | 1.276% | 0.0000 |
| 0.1 | 2.0 | 0.444 / 0.545 / 0.622 | 1.288% | 0.494 | 0.822 / 0.488 / 0.781 | 1.276% | 0.0000 |
| 0.2 | 0.1 | 0.397 / 0.529 / 0.585 | 2.144% | 0.437 | 0.480 / 0.512 / 0.366 | 2.391% | 0.1364 |
| 0.2 | 0.5 | 0.397 / 0.529 / 0.604 | 2.144% | 0.437 | 0.622 / 0.425 / 0.510 | 2.552% | 0.0897 |
| 0.2 | 1.0 | 0.397 / 0.529 / 0.611 | 2.144% | 0.437 | 0.767 / 0.500 / 0.778 | 2.422% | 0.0000 |
| 0.2 | 2.0 | 0.397 / 0.529 / 0.618 | 2.144% | 0.437 | 0.767 / 0.500 / 0.779 | 2.422% | 0.0000 |

Brute-force verification: relative gap (NIO loss − grid loss) / grid loss for every main and λ/μ run (negative =
NIO better than the discrete grid):

| detector | mean AUC | runs | median gap | worst gap | runs > +1% above grid |
|---|---|---|---|---|---|
| known | 1.0000 | 189 | -0.51% | +1.28% | 1 |
| novel (LOTO) | 1.0000 | 189 | -0.51% | +0.54% | 0 |
| pslist-only | 1.0000 | 189 | -0.33% | +14.16% | 22 |
| handles-only | 0.9999 | 189 | -0.51% | +0.25% | 0 |
| ldrmodules-only | 0.9999 | 189 | -0.43% | -0.02% | 0 |
| svcscan-only | 0.9999 | 189 | -0.51% | -0.09% | 0 |
| dlllist-only | 0.9998 | 189 | -0.41% | +1.01% | 1 |
| malfind-only | 0.9908 | 189 | -0.25% | +3.91% | 13 |
| psxview-only | 0.9217 | 189 | -0.23% | +3.12% | 9 |
| callbacks-only | 0.8747 | 189 | -0.19% | +23.66% | 7 |
| modules-only | 0.5186 | 189 | +2.12% | +7.49% | 135 |

Time-varying policy (10 windows) vs best fixed policy (min of NIO and grid), same objective without the smoothness term:

| detector | init | n configs | objective time-varying | objective best fixed | improvement | peak TV / fixed | deploy cost TV / fixed |
|---|---|---|---|---|---|---|---|
| known | cold | 9 | 0.0349 ± 0.0001 | 0.0369 ± 0.0000 | 5.43 ± 0.32% | 1.013% / 1.071% | 0.494 / 0.523 |
| known | warm | 9 | 0.0349 ± 0.0001 | 0.0369 ± 0.0000 | 5.46 ± 0.16% | 1.013% / 1.071% | 0.494 / 0.523 |
| novel (LOTO) | cold | 9 | 0.0349 ± 0.0002 | 0.0369 ± 0.0001 | 5.46 ± 0.30% | 1.005% / 1.071% | 0.497 / 0.524 |
| novel (LOTO) | warm | 9 | 0.0349 ± 0.0002 | 0.0369 ± 0.0001 | 5.45 ± 0.18% | 1.017% / 1.071% | 0.495 / 0.524 |
| callbacks-only | cold | 9 | 0.0572 ± 0.0018 | 0.0578 ± 0.0018 | 1.13 ± 0.19% | 1.016% / 1.062% | 0.940 / 0.944 |
| callbacks-only | warm | 9 | 0.0571 ± 0.0019 | 0.0578 ± 0.0018 | 1.15 ± 0.20% | 1.007% / 1.062% | 0.941 / 0.944 |

Hypothetical detector degradation (effective TPR = q·TPR(τ), known-type detectors, mean over types × seeds):

| q (TPR scale) | x₁ firewall | x₂ response | peak I | loss |
|---|---|---|---|---|
| 1.0 | 0.468 ± 0.000 | 0.552 ± 0.000 | 1.071% | 0.0369 |
| 0.9 | 0.525 ± 0.000 | 0.558 ± 0.000 | 1.077% | 0.0401 |
| 0.75 | 0.626 ± 0.000 | 0.555 ± 0.000 | 1.082% | 0.0458 |
| 0.5 | 0.824 ± 0.000 | 0.500 ± 0.001 | 1.068% | 0.0572 |
| 0.25 | 0.963 ± 0.000 | 0.516 ± 0.000 | 1.122% | 0.0709 |
| 0.1 | 0.990 ± 0.000 | 0.611 ± 0.000 | 2.915% | 0.0969 |

Real detectors → optimal policy (all 11 detectors, mean over 3 types × 3 seeds):

| detector | AUC | x₁ | x₂ | τ | TPR@τ | FPR@τ | peak I | loss |
|---|---|---|---|---|---|---|---|---|
| modules-only | 0.5186 | 0.994 | 0.227 | 0.507 | 0.270 | 0.235 | 2.488% | 0.1054 |
| callbacks-only | 0.8747 | 0.843 | 0.482 | 0.788 | 0.486 | 0.000 | 1.062% | 0.0578 |
| psxview-only | 0.9217 | 0.772 | 0.483 | 0.816 | 0.610 | 0.013 | 1.072% | 0.0556 |
| malfind-only | 0.9908 | 0.542 | 0.499 | 0.793 | 0.981 | 0.025 | 1.078% | 0.0442 |
| dlllist-only | 0.9998 | 0.475 | 0.550 | 0.818 | 0.993 | 0.001 | 1.072% | 0.0374 |
| svcscan-only | 0.9999 | 0.469 | 0.551 | 0.685 | 0.999 | 0.001 | 1.071% | 0.0371 |
| handles-only | 0.9999 | 0.473 | 0.548 | 0.792 | 0.998 | 0.002 | 1.071% | 0.0374 |
| ldrmodules-only | 0.9999 | 0.473 | 0.550 | 0.836 | 0.995 | 0.001 | 1.072% | 0.0373 |
| pslist-only | 1.0000 | 0.474 | 0.548 | 0.658 | 0.998 | 0.002 | 1.072% | 0.0375 |
| novel (LOTO) | 1.0000 | 0.469 | 0.552 | 0.557 | 0.998 | 0.000 | 1.071% | 0.0369 |
| known | 1.0000 | 0.468 | 0.552 | 0.612 | 1.000 | 0.000 | 1.071% | 0.0369 |

Figures: fig13 (SIR before/after), fig14 (convergence), fig15 (Pareto + grid), fig16 (time-varying), fig18 (detector
quality → policy).
**Takeaway:** for the all-feature known-type and LOTO detectors NIO through the simulator finds the brute-force optimum
(and usually beats it slightly, because the grid is discrete). For several single-plugin detectors, gradient descent from
a single start stalls in a worse basin (gap table), and only the grid reveals it. Known-type and LOTO detectors are so good on MalMem that the optimal policy is the same: Spyware
(0.468, 0.551, τ=0.493) vs (0.471, 0.552, τ=0.477). With a near-perfect detector the false-quarantine price μ never binds, and τ can
sit anywhere in a wide plateau. The weak `callbacks`-only detector makes τ and μ matter, giving Spyware
(0.870, 0.465, τ=0.780). Degrading TPR shifts spending from automated response to firewall (x₁ 0.468 → 0.824 at
q = 0.5 → 0.990 at q = 0.1).

### C2 — Deepfake spread on a platform

Not run: deepfake track removed from scope (see A2).

## 7. README questions answered

| source | question | what we found |
|---|---|---|
| Ch10 Exp. 1 | Increase training steps 20 → 40: does NIO still reproduce the reference weights? | Yes. Relative weight error at T = 40 is in the same range as T = 20 (table "Inner-step sweep"), and twin accuracy stays ≥ 0.938 at every P and T. A T = 40 unroll costs 2.0× a T = 20 NIO run. |
| Ch10 Exp. 2 | Larger primer → better weight matching? | Up to a point. Error falls from P = 2 (0.30 ± 0.23) to P = 16 (0.14 ± 0.08) and then plateaus (P = 64: 0.20 ± 0.02) at a fixed 300-iteration budget. Accuracy saturates already at P = 2–4. |
| Ch10 Exp. 3 | Different original dataset: does NIO still find a useful primer? | Yes, on a real security dataset (MalMem binary and 4-class), not just a new toy pattern. 4-class P = 64 primer 0.680 ± 0.010 vs full data 0.690 ± 0.004. |
| Ch10 Exp. 4 | Does NIO reveal the structure of private data? Recovery vs mere similar predictions? | Structure yes: class-difference cosine 0.72–0.86 across P and 9.3–12.3 of the real top-15 discriminative features recovered. Records no: DCR membership test at chance while the positive control is detected. This is how we separate "genuine recovery" from "similar predictions". |
| Ch10 RQ | Primers for larger language models? | Not tested (tabular MLP only). |
| Ch10 RQ | Reproduce behaviour with fewer training steps? | A twin trained for T = 20 steps on 16 synthetic rows is 0.7 accuracy points below the full-data model (P · T row-gradients instead of N · T). |
| Ch10 RQ | Can private training information be recovered from weights? | Population-level information (class contrast, discriminative features, a working training set) yes. Individual records no evidence. |
| Ch10 RQ | Effect of primer size on weight matching? | See Exp. 2 and fig08. |
| Ch10 RQ | Do architectures differ in what they reveal? | Not tested across architectures. Within the MLP, the 4-class head is harder to invert (higher relative error) than the binary head. |
| Ch10 RQ | Does NIO reduce total training cost? | No. One NIO run costs about 21× the T-step reference training it imitates. |
| Ch7 Exp. 1 | Policies 0 / 1 / combinations | No defence (0, 0): peak 48.29%. Full defence (1, 1): peak 1.00% (= I₀, never grows). Manual (0.3, 0.4): 6.32%. |
| Ch7 RQ1 | NIO for more complex propagation models? | Yes for well-behaved detectors: 3-D policy with detector-dependent recovery γ(x₂, TPR(τ)) and a false-quarantine cost, within +1.28% of a 9,261-point grid. With single-plugin detectors plain Adam from one start misses the optimum by more than 1% in 187/1701 runs (multi-start or grid initialisation would be needed). |
| Ch7 RQ2 | How do policy costs change the optimum? | Raising λ lowers both x₁ and x₂ and raises the peak. μ only matters when the detector has non-zero FPR at useful thresholds (weak detector); then it pushes τ up. |
| Ch7 RQ3 | Time-varying policies? | Yes: a 10-window schedule beats the best fixed policy by 4.17 ± 1.87% on the same objective. |
| Ch7 RQ4 | Different computer types / topologies / strains? | Partly: separate runs per malware type, each with its own detector TPR. No topology or multi-strain coupling. |
| Ch7 RQ (extra) | Learned surrogate when simulation is expensive? | Not needed: one 300-iteration Adam batch of 1980 policies took 26 s. Not tested. |

## 8. Limitations and negative results

- **Saturated dataset.** Binary MalMem accuracy is 99.94%–100.00% for every model and split, and LOTO detectors miss at most 0.47% of an unseen type. Benign dumps
  come from one VM and have no binary identity, so benign-side leakage across splits cannot be excluded. Novel-type
  results are therefore optimistic.
- **The primer does not beat real data everywhere.** At P = 64 it ties a random real subset, and at T = 40 with P ≥ 32
  it is slightly worse with higher variance (table "Inner-step sweep"). Its advantage is at very small P.
- **The Gaussian-noise primer is mostly worse than chance** (binary accuracy 0.36–0.51 across P). Random off-manifold
  points with balanced labels push θ₀ in a direction anti-correlated with the real boundary. It is reported as measured,
  not tuned.
- **Leakage conclusions are bounded by the test.** DCR with a size-matched reference detects near-copies (sd 0.005)
  reliably, but cannot rule out subtler memorisation. Only one architecture (small MLP) and full-batch SGD were audited.
  The attacker is assumed to know θ₀, T and η (Ch10 threat model). Without θ₀ the attack is harder.
- **B3 is "DP-style", not DP.** There is no privacy accounting and no ε. Lot sampling is without replacement per step
  (not Poisson). The utility drop is small because the task is easy, so this is not evidence about harder tasks.
- **Part C constants are illustrative assumptions** (β₀, the 0.8 firewall efficacy, the γ form, λ, μ, the 0.02
  temperature, q). The absolute peak-infection numbers are not predictions for any real network.
- **NIO for policies is a local method.** With single-plugin detectors it ended up to +23.7% above the
  brute-force optimum. Only one initialisation (z = 0) was used. Multi-start would be the obvious fix and was not run.
- **The LOTO "weaker detector" was not actually weaker** on this data. The weak-detector analysis uses a
  telemetry-restricted real detector (`callbacks`) and a hypothetical TPR scaling instead.
- **Deepfake track (A2, B2, C2) not run** (out of scope by owner request).

## 9. Figure index

| file | caption |
|---|---|
| `fig01_malmem_dataset` | CIC-MalMem-2022 composition: (a) benign vs malware dumps, (b) malware type, (c) rows per malware family coloured by type. |
| `fig02_malmem_confusion` | Row-normalised 4-class confusion matrices on the group split (sum over 3 seeds) for (a) Random Forest and (b) MLP. Most errors are between malware types; benign is almost never confused. |
| `fig03_malmem_importance` | Top-15 permutation importances (drop in test macro-F1, mean ± std over seeds) for 4-class type attribution, (a) RF and (b) MLP. |
| `fig04_split_honesty` | (a) 4-class macro-F1 under random vs group split; (b) binary test error under both splits; (c) missed-malware rate of the known-type vs leave-one-type-out MLP for each malware type. |
| `fig06_primer_convergence` | NIO convergence: relative weight error ‖θ_P−θ_ref‖²/‖θ_ref−θ₀‖² (mean over tensors) vs iteration for each primer size P, T = 20, (a) binary, (b) 4-class; mean ± std over seeds; dotted line = untrained θ₀. |
| `fig07_primer_vs_baselines` | Key result. Test accuracy of a twin retrained from θ₀ for T = 20 SGD steps on: the NIO primer (fixed or learnable soft labels), a class-balanced random real subset of the same size, the Gaussian-noise primer (NIO starting point), and the full training data; (a) binary, (b) 4-class; mean ± std over 3 seeds. |
| `fig08_primer_T_sweep` | Effect of the number of unrolled inner SGD steps T on (a) primer-twin accuracy (dashed: full-data reference at the same T) and (b) final relative weight error, binary detector. |
| `fig09_leakage_nn` | Record-level leakage test. (a) ECDF of each primer row's distance to the nearest training row (size-matched reference) vs nearest held-out row; (b) log distance ratio for NIO primers, the noise primer, positive controls (near-copies / perturbed copies of training rows) and the null control (real non-member rows). |
| `fig10_primer_pca` | Real training data in its first two principal components with (a) the P = 16 NIO primer (stars) and its noise starting point (crosses) and (b) the P = 64 primer overlaid; colour = label. |
| `fig11_defense_tradeoff` | Privacy/utility of defences against NIO: (a) released-model accuracy and primer-twin accuracy vs DP-style noise σ; (b) attacker weight-matching error vs σ (dotted: undefended); (c) utility vs leakage for every defence (P = 16), including the weight-decay-aware attacker. |
| `fig13_sir_before_after` | SIR trajectories under the Ch7 default policy (dashed) vs the NIO-optimised detector-in-the-loop policy (solid), per malware type (known-type detector, seed 0): (top) susceptible and recovered fractions, (bottom) infected %. |
| `fig14_policy_convergence` | NIO policy optimisation: (a) loss vs iteration for each malware type and detector; (b) firewall, response and alert-threshold trajectories for Spyware with the known-type (solid) and novel-type (dashed) detector. |
| `fig15_pareto` | Peak infection vs deployment cost x₁² + x₂² for the λ × μ sweep (colour = λ, marker = μ), on top of all 21³ grid policies (grey, best τ) and the grid's peak-vs-deployment-cost front (black); top row all-feature detector, bottom row weak single-plugin detector. The front ignores the false-quarantine term μ·x₂·FPR, so for the weak detector (bottom) NIO optima sit to its right: they buy extra firewall to avoid quarantining benign hosts. |
| `fig16_time_varying_policy` | (a) 10-window time-varying firewall/response schedule vs the best fixed policy (dashed); (b) resulting infected fraction; Spyware, known-type detector. |
| `fig18_detector_quality` | How detector quality shapes the optimal policy: (a, b) hypothetical TPR scaling q vs optimal firewall/response and peak; (c) real detectors (all-feature, LOTO, nine single-plugin MLPs) — optimal policy vs detector ROC-AUC. |
| `fig19_plugin_detectors` | One MLP per Volatility plugin (group split, binary): (a) test ROC-AUC, (b) missed-malware and false-alarm rates at threshold 0.5; mean ± std over seeds; (#) = number of features. |

Figures 05, 12 and 17 (deepfake ROC, deepfake primer, deepfake spread) were not produced because that track was dropped.

## 10. Reproduce

```bash
uv venv .venv --python 3.12
uv pip install --python .venv -r requirements.txt
bash run_all.sh            # 00 → 01 → 03 → 05 → 06 → 08 → 09; writes results/, figures/, RESULTS.md
```

Individual steps: `.venv/Scripts/python.exe scripts/03_primer_malmem.py [--quick] [--seeds 0 1 2]` (`.venv/bin/python` on Linux/macOS).
