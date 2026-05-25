# 04 — Statistical Method

## 1. Core model

For trait vector **y** (n × 1) measured on `n` individuals of an allopolyploid species with subgenomes `s ∈ {A, B, D}` (wheat) / `{A_t, D_t}` (cotton) / `{A, C}` (canola) / `{A_1..A_4}` (strawberry octoploid), we propose the **subgenome-stratified mixed model**:

```
y = Xβ + Σ_s g_s + Σ_{(s,t)} g_st + Σ_v z_v u_v + ε
```

with:

| Term | Description | Distribution |
|---|---|---|
| `Xβ` | fixed effects (intercept, covariates, top PCs per subgenome) | — |
| `g_s` | additive polygenic effect from subgenome `s` | `N(0, σ²_s · K_s)` |
| `g_st` | homoeolog-pair epistasis between subgenomes `s` and `t` | `N(0, σ²_st · K_st)` |
| `z_v u_v` | fixed effect of test variant `v` (SNP dosage **or** graph-node dosage) | scalar |
| `ε` | residual | `N(0, σ²_e · I)` |

### Subgenome GRMs

For each subgenome `s`, compute additive GRM on standardized dosages restricted to variants assigned to subgenome `s`:

```
K_s = (1/m_s) · Z_s Z_s^T
```

where `Z_s` is centered & scaled dosage on subgenome-s variants (`m_s` variants).

### Homoeolog interaction kernel

```
K_st = K_s ⊙ K_t        (Hadamard product)
```

This is equivalent to the additive-by-additive epistatic variance covariance (Santantonio et al. *G3* 2019).

**Curse of dimensionality avoidance**: We do **not** test every pairwise homoeolog interaction at SNP level (10⁷ × 10⁷ is infeasible). Two options:

1. **Variance-component test** (recommended for v1): test `H₀: σ²_st = 0` via score test on REML, one test per subgenome pair. Cheap, principled, but low resolution.
2. **Ortholog-bin test** (v1.5): group SNP pairs into bins by orthogroup membership (EnsemblPlants Compara), one variance component per bin (~100–500 bins). Score-test each bin → BH-FDR across bins.
3. **Targeted 2D scan** (v2): only on bins flagged in (2), perform pairwise SNP-SNP scan within the bin. GPU 2D kernel.

## 2. Variant-level test

For each test variant `v` (SNP or graph-node), score test under the null fit at step 1:

```
score statistic T_v = (z_v^T (y - ŷ))² / Var₀(z_v^T (y - ŷ))
```

with **saddle-point approximation** for skewed traits / rare variants (REGENIE-SPA style).

### Dosage encoding

For polyploids, we extend dosage beyond {0, 1, 2}:

- Wheat hexaploid: `g ∈ {0, 1, 2, 3, 4, 5, 6}` (expected dosage from polyRAD/Updog posterior).
- Strawberry octoploid: `g ∈ {0, ..., 8}`.

To allow non-linear (e.g., simplex / duplex dominance) dosage response, encode as **allelic-series basis**:

```
b₁(g) = g
b₂(g) = g(g-1)
b₃(g) = g(g-1)(g-2)
...
```

Default model uses b₁ (additive). Optional non-additive flag adds b₂ and b₃.

## 3. Graph-node integration

Bubble dosage from `NodeGWAS` / `Varigraph` / `PanGenie`:

- Each node has count/coverage per individual → expected dosage scaled to ploidy.
- Treated as additional variants in `Σ_v z_v u_v` term, indistinguishable from SNP dosage at the LMM step.
- LD-clumping (`--node-clump` flag) removes nodes in r² > 0.99 with a tag SNP to avoid double counting.
- Optional: bubble-level set-based test (collapse all nodes within a bubble into one test, à la STAAR set).

## 4. DL functional prior (v2)

**Recommended insertion point: STAAR-style weight on set-based test.**

For variant `v`, compute zero-shot log-likelihood ratio score under a pretrained plant DNA-LM:

```
π_v = log P(REF | context) − log P(ALT | context)            using PlantCaduceus
```

Then variant-level weight `w_v = softplus(|π_v|)`. Insert into set-based score test:

```
T_set = (Σ_v w_v · z_v^T(y - ŷ))² / Var₀(...)
```

### Calibration

- DL prior is computed **without phenotype information** → no leakage.
- Report power AND type-I error on null simulation with/without prior — must pass λ_GC ≈ 1.

### Alternatives considered (not chosen for v1)

| Option | Reason rejected |
|---|---|
| DeepNull-style residualization | leakage risk on small (~500) cotton/canola panels |
| End-to-end DL replacing LMM (DeepGWAS) | high overfitting risk, hard to calibrate |
| DL embedding as kernel (SKAT-RandomKernel) | training cost too high for v1, possibly v2 |

## 5. Multiplicity control

Multi-resolution FDR over four hypothesis classes:

- H1: SNP main effects (~10⁷)
- H2: Graph-node main effects (~10⁶)
- H3: SV-set / bubble set tests (~10⁵)
- H4: Homoeolog-pair epistasis bins (~100–500)

We use:

- **Group-BH** (Hu & Hua 2024) across resolutions → unified FDR target = 0.05
- **ACAT** to combine p-values within the same resolution when tests are dependent
- **Knockoff filter** (Candes et al.) as a second, independent FDR controller for replication

Report both for each discovery.

## 6. Calibration diagnostics (mandatory in every report)

- λ_GC per subgenome (A, B, D)
- λ_GC for graph-node test
- QQ plot per resolution
- Permutation-null calibration on chromosome arms (Santantonio 2019 style)
- Replication rate across resampled halves of the panel

## 7. Bayesian alternative (optional ablation in v1, primary engine in v2)

`hibayes` / `BSLMM` spike-slab prior over (SNP, node, homoeolog-pair) for sanity check. Report concordance of top loci between LMM and BSLMM — disagreement is itself informative.

---

## Falsifiable claims (pre-registered)

We commit to the following predictions before running the wheat real-data analysis:

1. On AlphaSimR-simulated wheat (N=1000, h²=0.4, 5 homoeolog-pair epistasis QTL), **HomoeoGWAS detects ≥3** of the 5 epistasis QTL at FDR 5%, while GWASpoly, NodeGWAS, networkGWAS detect 0.
2. On Watkins flowering time (10 °C), **HomoeoGWAS recapitulates Vrn-A1/B1/D1 and Ppd-A1/B1/D1**, and detects **at least 1 Vrn × Ppd cross-subgenome interaction** at FDR 5% that replicates in the CIMMYT WAMI panel.
3. Compared to NodeGWAS on the sugarcane benchmark, **calibration (λ_GC) is comparable (within 0.05)** while reducing the discovery–replication FDR by ≥30%.

Failure on any of these is a publishable negative result and triggers method revision before submission.
