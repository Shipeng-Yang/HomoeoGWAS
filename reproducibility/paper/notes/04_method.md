# 04 — Statistical Method

> **Status note (2026-06-02).** Rewritten to match the implemented code in
> `src/homoeogwas/`. The previous version described variant-dosage `{0..6}`/`{0..8}`
> models, an allelic-series basis, saddlepoint SPA, graph-node tests, knockoff/Group-BH
> FDR, and a BSLMM engine — **none implemented** (see
> [`03_architecture.md`](03_architecture.md) appendix). It also carried "pre-registered
> falsifiable claims" that the real-trait analyses did not meet; those are now recorded
> factually in §10 as negative empirical outcomes, not as aims.

## Scope

What the method **supports**: subgenome-stratified mixed models on additive `{0,1,2}`
hardcalls; multi-kernel REML variance components; single-locus EMMAX/P3D scanning (with
LOCO); a gene-resolution homoeolog-pair burden-product **interaction** test; GBLUP
prediction; and a defensive homoeolog-kernel diagnostic path.

What it does **not** claim: no per-SNP power gain over a pooled GRM (observed: none — see
§10); no validated genome-wide `K_hom` variance-component discovery advantage on real traits
(it was null — §4); no graph/SV testing and no polyploid dosage beyond additive `{0,1,2}`.

## 1. Core model

For trait vector **y** (n × 1) on `n` individuals of an allopolyploid with subgenomes
`s ∈ {A,B,D}` (wheat) / `{A,D}` (cotton) / `{A,C}` (rapeseed) / 4 subgenomes (strawberry,
oat), the **subgenome-stratified LMM** is

```
y = Xβ + Σ_s g_s  (+ g_hom)  + e ,   g_s ~ N(0, σ²_s K_s),   e ~ N(0, σ²_e I)
```

| Term | Description |
|---|---|
| `Xβ` | fixed effects. In the v0.1 analyses **X = intercept only** (additional covariates/PCs are accepted by the API but were not used). |
| `g_s` | additive polygenic effect of subgenome `s`, with VanRaden GRM `K_s` |
| `g_hom` | *optional* homoeolog co-association effect with `K_hom = ⊙_s K_s` — **downgraded, see §4** |
| `e` | residual |

The single test variant enters as a **fixed effect** in the per-SNP scan (§5), not as a
random term.

## 2. Subgenome GRMs (`grm.py`)

For each subgenome `s`, the VanRaden additive GRM on its own variants:

```
K_s = Z_s Z_sᵀ / Σ_j 2 p_j (1 − p_j)
```

`Z_s` is centered dosage (`{0,1,2}`, NaN imputed by per-variant mean), `p_j` the imputed
allele frequency; `K_s` is trace-normalised so `trace(K_s) ≈ n`. **LOCO** GRMs are built by
storing the numerator `Z_sZ_sᵀ` and denominator `Σ2pq` per chromosome and subtracting the
left-out chromosome (`K_{s,−c}`), with a build-time checksum `Σ_c denom_c = denom_global`.

## 3. REML variance components (`lmm.py`)

- **Single-kernel** (`fit_reml`): eigendecompose `K = UΛUᵀ` once, profile out `β` and `σ²_g`,
  minimise the negative REML log-likelihood over `log δ`, `δ = σ²_e/σ²_g`, with bounded Brent.
- **Multi-kernel** (`fit_multi_reml`): `y = Xβ + Σ_j u_j + e`, optimised by box-constrained
  **L-BFGS-B over the raw variance components σ²** (deliberately *not* `exp(θ)`, so the σ²=0
  boundary is reachable for a correct boundary LRT). Reports paper-grade
  `PVE_j = σ²_j · trace(K_j)/n ÷ Σ`, kernel collinearity, condition number, and which
  components sit at the boundary. Multi-start (`n_starts`) for robustness.
- `fit_with_homoeolog_fallback` fits the full model and **drops `K_hom`** if it hits the
  boundary, is ill-conditioned (`cond > 1e4`), or contributes `PVE < 1e-3`.

## 4. Homoeolog kernels and the `K_hom` downgrade (`kernel.py`)

- `sum_kernel`: `K_sum = Σ_s w_s K_s` (additive baseline).
- `hadamard_kernel`: `K_hom[i,j] = ∏_s K_s[i,j]` — homoeolog co-association (PSD by the Schur
  product theorem).
- `pairwise_mean_kernel`: `K_pair = mean_{s<t} K_s ⊙ K_t`, the numerically-stable fallback for
  ≥4 subgenomes.

**Status — read before using `K_hom`.** The Hadamard homoeolog kernel is *implemented* but
is **supplementary / defensive only**. In GBLUP REML across four panels its variance component
was at the σ²=0 boundary or non-predictive (out-of-sample Δr² ≈ 0, negative in 2/4 panels), and
in simulation it identified injected homoeolog epistasis with low power. It is therefore **not**
the primary interaction method, and the package auto-drops it (§3). A charter §2.1 hard gate
prevents promoting it back to a main claim. The actual interaction method is §7.

## 5. Single-locus association scan (`scan.py`)

EMMAX/P3D: variance components are estimated **once** (genome-wide REML, §3), then each marker
is tested with `V` held fixed:

```
V   = σ²_e I + Σ_j σ²_j K_j
P   = V⁻¹ − V⁻¹X (XᵀV⁻¹X)⁻¹ XᵀV⁻¹                 (built once)
U_g = gᵀP y ,  I_g = gᵀP g ,  γ̂ = U_g/I_g ,  χ² = U_g²/I_g  ~  χ²₁
```

Batched over markers this is one `P @ G` GEMM (CPU NumPy or single-GPU Torch). A streaming
variant supports bounded-memory scanning for wheat-scale (~86 M SNP) panels. **LOCO**
(`scan_snps_loco` /
`scan_bed_stream_loco`) tests each marker against its own chromosome's `P_{−c}` to remove
proximal contamination; variance components are not re-fit per chromosome (EMMAX/BOLT-LOCO
convention). Phenotypes enter raw (intercept-centred); no rank transform is applied in the
single-locus path. Genomic inflation `λ_GC = median(χ²)/0.4549` is reported per subgenome.

**Multiplicity.** Bonferroni per subgenome (and per pair set for §7); ACAT to combine
dependent p-values within a set. No knockoff filter, no Group-BH.

## 6. Genotype representation

Additive `{0,1,2}` hardcalls from PLINK1 BED throughout. VCFs are converted to BED/pgen via
`plink2` before analysis. There is no expected-dosage `{0..6}`/`{0..8}` model and no
allelic-series (non-additive) basis.

## 7. Homoeolog-pair interaction method (`interact.py`) — primary contribution

This is the project's actual interaction method; it **replaces** the failed variance-component
`K_hom` test (§4) at gene resolution.

For a homoeolog pair `(X in subgenome S_x, Y in S_y)`:

1. **Gene burden** `b_X` = mean of column-standardised dosages over the gene's SNPs
   (subsampled to a cap for dense WGS), then standardised across genes.
2. **Whitening** from the subgenome-stratified null LMM: `W = V^{-1/2}` from a multi-kernel
   REML over `{K_s}` (the full `{K_A,K_B,K_D}` for hexaploids — the third subgenome is never
   dropped).
3. **Whitened GLS** `y_w ~ 1 + b_X + b_Y + b_X·b_Y` and a t-test on the interaction
   coefficient. Main effects are retained, so regional additivity cannot masquerade as
   interaction.
4. **Aggregation** of per-pair p-values with **ACAT**; for hexaploids the three within-triad
   pairwise tests (A–B, A–D, B–D) combine to a triad-level ACAT omnibus.
5. **Calibration** by y-shuffle permutation (re-whitening each replicate): empirical ACAT p
   and `λ_perm`. **INT** is the primary phenotype transform; **raw** is a sensitivity transform
   (raw burden-products are anti-conservative under phenotype outliers — a documented failure
   mode, not a tool defect).

**Conditional-on-marginals sanity** (`pair_conditional_diagnostics`, persisted to
`results/phase7/conditional_sanity_hits.json`): for each reported hit, the whitened design's
interaction column is checked for collinearity with the marginals (projection VIF /
R²_int|marg), nested F-tests (interaction|marginals; joint main effects; total pair model),
and single-gene burden-GLS marginal p — supporting that a hit is carried by the pair product,
not by either homoeolog alone.

**Extensions** (y-independent, frozen before association; reported alongside, never replacing,
the unweighted result):
- `--weights`: a y-independent prior (zero-shot DL LLR or homoeolog expression bias) drives
  weighted Bonferroni and weighted ACAT.
- `multi_trait`: ACAT across a **frozen, predeclared** trait set → one pleiotropy p per pair,
  with multiplicity over pairs (not pairs × traits) and shared-permutation calibration.

## 8. Calibration and diagnostics (`calibration.py`, `diagnostics.py`)

- Null-simulation χ̄² mixture LRT type-I with Wilson confidence intervals.
- Nested REML LRT (likelihood-comparable fits), Self–Liang boundary LRT for σ²=0 components,
  parametric bootstrap LRT, and a PVE sensitivity grid.
- Per-subgenome `λ_GC`; for the interaction engine, analytic ACAT p vs empirical permutation p
  and `λ_perm`.

## 9. Genomic prediction and simulation (`gp.py`, `sim.py`)

- `gp.run_cv_gblup`: GBLUP with subgenome-stratified GRM tiers (pooled vs per-subgenome vs
  per-subgenome+`K_hom`), stratified cross-validation, top-k enrichment, paired-bootstrap CIs.
  This is the path on which `K_hom` was found non-predictive (§4).
- `sim`: pure phenotype simulation (no external binaries / Torch) for the power / type-I /
  calibration benchmark.

## 10. DL functional prior (standalone, supplementary)

Computed **outside** the `homoeogwas` package by `scripts/phase3/m3_3_score_dl_prior.py`.
For each variant a zero-shot masked-LM log-likelihood-ratio is read from two **borrowed
pretrained** plant DNA models (PlantCaduceus 225 M, AgroNT 1 B), ensembled by robust z-score
and fused with the GWAS signal at candidate ranking:
`fusion = z(−log10 p_gwas) + β·z(|LLR|_ensemble)`. The prior itself is computed without
phenotype values.

**This channel is a capability + mechanistic demonstration only — not evidence of
prior-weighted discovery.** The historical recall "lift" was selection-biased: the fusion
weight β was tuned to maximise recall over the recoverable known-QTL set and the lift was then
reported at that β on the same set. A leakage-aware re-analysis
(`scripts/phase7/dl_honest_lift.py` → `results/phase7/dl_honest_lift.json`) recomputed, for all
12 panel × trait runs, a β-selection-honest **nested leave-one-QTL-out** lift (β chosen only on
the inner QTL) and a **permutation null** (shuffle `|LLR|` among the scored candidates, rerun the
whole nested procedure). Outcome: **no panel shows a permutation-significant nested lift (all
empirical p ≥ 0.43)**; small-N "lifts" collapse to 0 under nesting (a β-selection artifact), and
the large-N positive "lifts" are not distinguishable from the candidate/sentinel structure under
permutation. Two further leakages **remain and are not removed** by this analysis and must be
disclosed prominently: the candidate pool was seeded with known-QTL **sentinel SNPs**, and the DL
ensemble was scored **only on GWAS-derived candidate variants** — a y-dependent selection — so
this is **not** a genome-wide y-independent prior-weighted-discovery test. The clean
prior-weighted-discovery evidence in this work is the wheat HEB weighted-interaction demo (with
decisive reversed/permuted negative controls) and the power simulation, **not** the DL channel.
Genome-wide DL rescoring (to remove candidate-selection leakage) is future work.

## 11. Tested outcomes (replaces the former "pre-registered claims")

Earlier design notes listed falsifiable targets (detect ≥3/5 simulated epistasis QTL;
recapitulate Vrn-A1/B1/D1 + Ppd-A1/B1/D1; detect a Vrn×Ppd interaction). In the completed
real-trait analyses these were **not** met and are recorded here as negative empirical
outcomes, not as aims or success criteria:

- On wheat Watkins flowering/emergence, **no locus reached genome-wide significance** among
  the tested outcomes, and **6 of 7 expected Ppd/Vrn loci were not recovered** (only Vrn-B1
  cleared a suggestive p < 5e-5).
- The subgenome-stratified single-locus LMM showed **no per-SNP power gain over a pooled
  GRM** in simulation; its measured advantages are calibration (lower BH-FDP) and runtime.
- The variance-component `K_hom` homoeolog test was **null** on real traits (§4).

The positive methodological results obtained instead — the calibrated cross-polyploid
framework, the gene-pair interaction engine, and reported candidate interaction findings in
cotton and wheat — are documented in the Phase-7 results, not pre-registered here.

## Appendix — considered but not implemented

As in [`03_architecture.md`](03_architecture.md): saddlepoint SPA, REGENIE two-step ridge,
graph-node/SV set tests, `{0..6}`/`{0..8}` dosage and the allelic-series basis, knockoff and
Group-BH multi-resolution FDR, and a BSLMM/`hibayes` Bayesian engine were considered in early
design and are **not** part of the statistical method evaluated here.
