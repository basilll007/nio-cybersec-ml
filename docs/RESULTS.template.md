# RESULTS — NIO for Cyber Defense on CIC-MalMem-2022

<!-- Rendered by scripts/09_render_results.py from results/*. Do not edit RESULTS.md by hand. -->

Scope: memory-forensics malware detection (CIC-MalMem-2022). The deepfake track (A2/B2/C2) was dropped at the owner's
request. All values are mean ± std over seeds 0, 1, 2 (population std); each seed re-draws the split *and* the model
initialisation.

## 1. TL;DR

- **Split honesty matters for the hard task, not the easy one.** Binary benign-vs-malware detection is saturated under
  both splits (group-split MLP accuracy {{a1_bin_group_MLP_acc}}). 4-class type identification drops when all dumps of a
  binary are kept in one split: RF macro-F1 {{a1_4c_random_RF_f1}} (random) → **{{a1_4c_group_RF_f1}} (group)**.
  Detectors still catch an *unseen* malware type: leave-one-type-out MLP TPR is Ransomware {{a1_loto_Ransomware_tpr}},
  Spyware {{a1_loto_Spyware_tpr}}, Trojan {{a1_loto_Trojan_tpr}}. {{n_plugins_auc_ge_0999}} of 9 single Volatility plugins
  alone reach AUC ≥ 0.999.
- **A 2-row NIO primer retrains the released detector to {{b1_P2_primer}} accuracy** (full data {{b1_full}}), against
  {{b1_P2_subset}} for 2 random real rows and {{b1_P2_noise}} for the noise primer. Its mean accuracy exceeds a
  same-size random real subset at {{b1_primer_beats_subset_n}}/6 primer sizes (binary, T = 20). The margin is large
  for P ≤ 16 (P = 16: {{b1_P16_primer}} vs {{b1_P16_subset}}) and within seed noise at P = 64 ({{b1_P64_primer}} vs
  {{b1_P64_subset}}).
- **The primer leaks structure, not records.** Primer rows sit {{b1_primer_dtrain}} (median) from their nearest real
  record, while real records sit {{b1_real_nn}} from each other. The DCR membership test finds no signal: share closer to
  train {{b1_leak_share}}, n = {{b1_leak_n}}, binomial p = {{b1_leak_binom}}, Wilcoxon p = {{b1_leak_wil}}. The same test
  flags near-copies of training rows (share {{b1_pos_share}}, p = {{b1_pos_wil}}). Yet the primer's malware-minus-benign
  direction matches the real one (cosine {{b1_P16_cos}} at P = 16).
- **DP-style noise and weight decay do not stop primer extraction** of the detector's *function*. With σ = 2 the released
  model keeps {{b3_dp2_ref}} accuracy and the primer twin still gets {{b3_dp2_twin}}. Undefended: {{b3_none_ref}} /
  {{b3_none_twin}}. Loss-based membership inference stays at chance (AUC {{b3_mia_min}}–{{b3_mia_max}}) in every
  configuration.
- **Detector-in-the-loop NIO policies, checked by brute force.** With the real known-type and leave-one-type-out
  detectors, Adam through the differentiable SIR lands within {{c1_kl_max}} of the 21³ grid optimum in all
  {{c1_kl_n}} runs (median {{c1_kl_median}}, i.e. usually *better* than the discrete grid). It cuts peak infection from
  {{c1_default_peak}} (Ch7 default setting) to {{c1_spy_known_peak}}. With the nine single-plugin detectors,
  {{c1_pl_n_gt1}} of {{c1_pl_n}} runs ended more than 1% above the grid optimum. Most are the uninformative
  `modules`-only detector, but misses also occur for pslist-, malfind-, psxview- and callbacks-only (worst
  {{c1_worst_rel}}, {{c1_worst_cfg}}). Gradient descent from one start is local, so the grid check is necessary. A 10-window
  time-varying policy improves the objective by {{c1_tv_impr}}. Weaker detection makes the optimiser buy firewall: x₁
  goes from {{c1_q1_x1}} (real detector) to {{c1_q01_x1}} at 10% effective TPR.

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

{{table:runtime}}

Order of work: setup → reproductions → A1 → B1 → B3 → C1 → figures → RESULTS.md. Superseded runs were discarded and
re-run; they are not counted above. The first B1/B3 leakage test was replaced by the size-matched DCR test after its
positive control showed it had no power (see DECISIONS.md). Scripts that ran concurrently share the CPU, so their
wall-clock times overlap.

### Sanity reproductions (before extending the course code)

Ch10 NIO Dataset Primer, toy "abc" (README attempt 4 logic, seed 42, CPU):

{{table:sanity_ch10}}

Ch7 NIO Malware Dynamics (README NIO block, 200 Adam iterations, lr 0.02):

{{table:sanity_ch7}}

Both match the README values to the printed precision.

## 4. Dataset

{{table:dataset}}

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

{{table:a1_main}}

Leave-one-malware-type-out (train on Benign + 2 types; detect the unseen third type; threshold 0.5):

{{table:a1_loto}}

Single-telemetry-source detectors (one MLP per Volatility plugin, binary, group split):

{{table:a1_plugins}}

Top-15 permutation importances (4-class, group split, mean ± std over seeds):

{{table:a1_importance}}

Figures: fig01 (dataset), fig02 (4-class confusion, group split), fig03 (importance), fig04 (split honesty), fig19
(per-plugin detectors).
**Takeaway:** binary detection is saturated, and LOTO MLPs miss at most {{a1_loto_max_miss}} of a never-seen type. MalMem's benign class
(one VM's normal behaviour) is trivially separable, so "novel malware" generalisation here is an upper bound. The honest
difficulty is *type attribution*, where the group split costs RF {{a1_4c_random_RF_f1}} → {{a1_4c_group_RF_f1}} macro-F1.
Handle counts (`{{top_rf_feature}}`) dominate type attribution.

### A2 — Deepfake detector

Not run: the deepfake track was removed from scope at the project owner's request (cyber-security focus). No Kaggle
credentials were available either.

### B1 — Does the released MalMem detector leak its training data? (Neural Inverse Optimization)

Binary detector, T = 20 inner SGD steps (test accuracy of a twin retrained from θ₀ on each P-row dataset):

{{table:b1_main}}

ROC-AUC of the same twins:

{{table:b1_auc}}

4-class detector, T = 20:

{{table:b1_4class}}

Inner-step sweep (binary, fixed labels; cells = twin accuracy / relative weight error):

{{table:b1_T}}

Leakage audit — distance-to-closest-record membership test (pooled over seeds and P):

{{table:b1_leak}}

Structure recovery: cosine between the (malware − benign) class-mean difference of the primer and of the real training
data, and overlap of the top-15 most different features (binary, T = 20):

{{table:b1_struct}}

Recovered discriminative features (binary, T = 20, P = 16):

{{table:b1_features}}

Cost of NIO vs normal training ("does it reduce total cost?"):

{{table:b1_timing}}

Figures: fig06 (convergence), **fig07 (primer vs baselines, key figure)**, fig08 (T sweep), fig09 (NN leakage), fig10
(PCA).
**Takeaway:** a released detector's weights *do* leak a compact, highly usable training set. 2–16 synthetic rows retrain
the detector to {{b1_P16_primer}} (P = 16) vs {{b1_P16_subset}} for 16 random real rows, and the primer encodes the real
class contrast. That set is far off the data manifold (primer row norm {{b1_primer_norm}} vs real {{b1_real_norm}}). It
is indistinguishable from noise in the record-level membership test, so the leakage is *population/structure* level, not
*record* level. NIO costs about {{b1_cost_ratio}}× the T-step reference training ({{b1_nio_secs}} s vs {{b1_ref_secs}} s),
so it does **not** reduce total training cost.

### B2 — Deepfake primer

Not run: deepfake track removed from scope (see A2).

### B3 — Defences: privacy / utility trade-off (P = 16)

{{table:b3}}

Same, P = 64:

{{table:b3_64}}

Figure: fig11.
**Takeaway:** DP-style clipping and noise cost {{b3_dp_drop}} points of released-model accuracy ({{b3_none_ref}} →
{{b3_dp2_ref}} at σ = 2); most of that comes from clipping and lot sampling, not from σ. They raise the attacker's
weight-matching error steadily ({{b3_none_relerr}} → {{b3_dp2_relerr}} at σ = 2 → {{b3_dp8_relerr}} at σ = 8), because
the attacker's assumed full-batch recipe no longer matches. Primer usefulness does not fall in step. The twin still reaches
{{b3_dp2_twin}} at σ = 2 and recovers to ≥ {{b3_twin_s4_min}} at σ ≥ 4 (table), with large seed-to-seed variance at small σ. Weight
decay only defeats a naive attacker (relative error {{b3_wd01_relerr}} at wd = 0.1). An attacker who knows wd matches
*better* than with no defence ({{b3_wd01_informed_relerr}}), with twin accuracy {{b3_wd01_informed_twin}}. Loss-based MIA
stays at chance in every configuration. Of {{b3_n_tests}} pooled DCR tests, {{b3_n_bonf}} survive Bonferroni correction
(α = {{b3_bonf}}); the smallest p is {{b3_min_p}} ({{b3_min_p_cfg}}). These defences target record-level membership,
which was not measurably leaking. They do not, and by design cannot, hide the population-level decision structure that NIO
extracts.

### C1 — Detector-in-the-loop network-defence policy (Neural Input Optimization)

Main run (λ = 0.05, μ = 0.5; default = Ch7 manual policy (0.3, 0.4) with τ = 0.5). Detectors: all-feature known-type MLP,
leave-one-type-out MLP, and the weakest informative single-plugin MLP (`{{weak_plugin}}`-only, AUC {{weak_plugin_auc}}):

{{table:c1_main}}

Pareto sweep (mean over 3 types × 3 seeds):

{{table:c1_pareto}}

Brute-force verification: relative gap (NIO loss − grid loss) / grid loss for every main and λ/μ run (negative =
NIO better than the discrete grid):

{{table:c1_gaps}}

Time-varying policy (10 windows) vs best fixed policy (min of NIO and grid), same objective without the smoothness term:

{{table:c1_time_varying}}

Hypothetical detector degradation (effective TPR = q·TPR(τ), known-type detectors, mean over types × seeds):

{{table:c1_quality}}

Real detectors → optimal policy (all 11 detectors, mean over 3 types × 3 seeds):

{{table:c1_detpolicy}}

Figures: fig13 (SIR before/after), fig14 (convergence), fig15 (Pareto + grid), fig16 (time-varying), fig18 (detector
quality → policy).
**Takeaway:** for the all-feature known-type and LOTO detectors NIO through the simulator finds the brute-force optimum
(and usually beats it slightly, because the grid is discrete). For several single-plugin detectors, gradient descent from
a single start stalls in a worse basin (gap table), and only the grid reveals it. Known-type and LOTO detectors are so good on MalMem that the optimal policy is the same: Spyware
{{c1_spy_known_x}} vs {{c1_spy_loto_x}}. With a near-perfect detector the false-quarantine price μ never binds, and τ can
sit anywhere in a wide plateau. The weak `{{weak_plugin}}`-only detector makes τ and μ matter, giving Spyware
{{c1_spy_weak_x}}. Degrading TPR shifts spending from automated response to firewall (x₁ {{c1_q1_x1}} → {{c1_q05_x1}} at
q = 0.5 → {{c1_q01_x1}} at q = 0.1).

### C2 — Deepfake spread on a platform

Not run: deepfake track removed from scope (see A2).

## 7. README questions answered

| source | question | what we found |
|---|---|---|
| Ch10 Exp. 1 | Increase training steps 20 → 40: does NIO still reproduce the reference weights? | Yes. Relative weight error at T = 40 is in the same range as T = 20 (table "Inner-step sweep"), and twin accuracy stays ≥ {{b1_T_min_acc}} at every P and T. A T = 40 unroll costs {{b1_T40_cost}}× a T = 20 NIO run. |
| Ch10 Exp. 2 | Larger primer → better weight matching? | Up to a point. Error falls from P = 2 ({{b1_P2_relerr}}) to P = 16 ({{b1_P16_relerr}}) and then plateaus (P = 64: {{b1_P64_relerr}}) at a fixed 300-iteration budget. Accuracy saturates already at P = 2–4. |
| Ch10 Exp. 3 | Different original dataset: does NIO still find a useful primer? | Yes, on a real security dataset (MalMem binary and 4-class), not just a new toy pattern. 4-class P = 64 primer {{b1_4c_P64_primer}} vs full data {{b1_4c_full}}. |
| Ch10 Exp. 4 | Does NIO reveal the structure of private data? Recovery vs mere similar predictions? | Structure yes: class-difference cosine {{b1_cos_min}}–{{b1_cos_max}} across P and {{b1_overlap_min}}–{{b1_overlap_max}} of the real top-15 discriminative features recovered. Records no: DCR membership test at chance while the positive control is detected. This is how we separate "genuine recovery" from "similar predictions". |
| Ch10 RQ | Primers for larger language models? | Not tested (tabular MLP only). |
| Ch10 RQ | Reproduce behaviour with fewer training steps? | A twin trained for T = 20 steps on 16 synthetic rows is {{b1_P16_gap}} accuracy points below the full-data model (P · T row-gradients instead of N · T). |
| Ch10 RQ | Can private training information be recovered from weights? | Population-level information (class contrast, discriminative features, a working training set) yes. Individual records no evidence. |
| Ch10 RQ | Effect of primer size on weight matching? | See Exp. 2 and fig08. |
| Ch10 RQ | Do architectures differ in what they reveal? | Not tested across architectures. Within the MLP, the 4-class head is harder to invert (higher relative error) than the binary head. |
| Ch10 RQ | Does NIO reduce total training cost? | No. One NIO run costs about {{b1_cost_ratio}}× the T-step reference training it imitates. |
| Ch7 Exp. 1 | Policies 0 / 1 / combinations | No defence (0, 0): peak {{c7_p00}}. Full defence (1, 1): peak {{c7_p11}} (= I₀, never grows). Manual (0.3, 0.4): {{c7_manual}}. |
| Ch7 RQ1 | NIO for more complex propagation models? | Yes for well-behaved detectors: 3-D policy with detector-dependent recovery γ(x₂, TPR(τ)) and a false-quarantine cost, within {{c1_kl_max}} of a 9,261-point grid. With single-plugin detectors plain Adam from one start misses the optimum by more than 1% in {{c1_pl_n_gt1}}/{{c1_pl_n}} runs (multi-start or grid initialisation would be needed). |
| Ch7 RQ2 | How do policy costs change the optimum? | Raising λ lowers both x₁ and x₂ and raises the peak. μ only matters when the detector has non-zero FPR at useful thresholds (weak detector); then it pushes τ up. |
| Ch7 RQ3 | Time-varying policies? | Yes: a 10-window schedule beats the best fixed policy by {{c1_tv_impr}} on the same objective. |
| Ch7 RQ4 | Different computer types / topologies / strains? | Partly: separate runs per malware type, each with its own detector TPR. No topology or multi-strain coupling. |
| Ch7 RQ (extra) | Learned surrogate when simulation is expensive? | Not needed: one 300-iteration Adam batch of {{c1_pareto_n}} policies took {{c1_pareto_secs}} s. Not tested. |

## 8. Limitations and negative results

- **Saturated dataset.** Binary MalMem accuracy is {{a1_bin_min}}–{{a1_bin_max}} for every model and split, and LOTO detectors miss at most {{a1_loto_max_miss}} of an unseen type. Benign dumps
  come from one VM and have no binary identity, so benign-side leakage across splits cannot be excluded. Novel-type
  results are therefore optimistic.
- **The primer does not beat real data everywhere.** At P = 64 it ties a random real subset, and at T = 40 with P ≥ 32
  it is slightly worse with higher variance (table "Inner-step sweep"). Its advantage is at very small P.
- **The Gaussian-noise primer is mostly worse than chance** (binary accuracy {{b1_noise_min}}–{{b1_noise_max}} across P). Random off-manifold
  points with balanced labels push θ₀ in a direction anti-correlated with the real boundary. It is reported as measured,
  not tuned.
- **Leakage conclusions are bounded by the test.** DCR with a size-matched reference detects near-copies (sd 0.005)
  reliably, but cannot rule out subtler memorisation. Only one architecture (small MLP) and full-batch SGD were audited.
  The attacker is assumed to know θ₀, T and η (Ch10 threat model). Without θ₀ the attack is harder.
- **B3 is "DP-style", not DP.** There is no privacy accounting and no ε. Lot sampling is without replacement per step
  (not Poisson). The utility drop is small because the task is easy, so this is not evidence about harder tasks.
- **Part C constants are illustrative assumptions** (β₀, the 0.8 firewall efficacy, the γ form, λ, μ, the 0.02
  temperature, q). The absolute peak-infection numbers are not predictions for any real network.
- **NIO for policies is a local method.** With single-plugin detectors it ended up to {{c1_worst_rel}} above the
  brute-force optimum. Only one initialisation (z = 0) was used. Multi-start would be the obvious fix and was not run.
- **The LOTO "weaker detector" was not actually weaker** on this data. The weak-detector analysis uses a
  telemetry-restricted real detector (`{{weak_plugin}}`) and a hypothetical TPR scaling instead.
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
