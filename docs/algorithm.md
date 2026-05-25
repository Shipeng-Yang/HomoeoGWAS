# Algorithm

This page summarizes the statistical model and the design choices. The frozen claim hierarchy is in [Project Charter](00_charter.md); revision history is in `docs/00_charter.md` §9.

## 1. Subgenome-partitioned LMM (main claim)

For an allopolyploid panel with subgenomes `S = {A, B, D, ...}`, the working model is

```
y = X β + Σ_{s ∈ S} u_s + ε
u_s ~ N(0, σ²_s · K_s)
ε   ~ N(0, σ²_ε · I)
```

where:
- `K_s` = per-subgenome additive GRM, computed from the SNP set assigned to subgenome `s` via `species_config.chrom_map`
- variance components `{σ²_s : s ∈ S} ∪ {σ²_ε}` fit by REML (`src/homoeogwas/lmm.py`)
- per-SNP association = score test against the residualised `y` under the null mixed model

**Why this is not "new LMM framework"**: per-subgenome random effects appear in Voss-Fels 2017 (wheat) and Mascher 2021 (barley). Our contribution is the operational implementation — generic schema-driven (`configs/species/*.yaml`), cross-ploidy unified, dual-GPU scalable, and shipped as a tested package.

### LOCO option

When `loco_unit: chrom` is set, the per-SNP scan recomputes the kinship excluding the SNPs on the chromosome of the test SNP — the classic Leave-One-Chromosome-Out correction for proximal contamination. We confirmed this helps:

- Wheat Watkins days_to_emerg: per-chrom λ_GC improved (B sub 0.872 → 1.075 = +0.20, A 1.078 → 1.106, D 1.027 → 1.059), with cost +0% on 2-GPU runtime
- Horvath2020 plant_height: λ_GC 0.9636 → 0.9941

LOCO benefit scales with per-chrom polygenic contribution — consistent with BOLT-LOCO theory.

## 2. Homoeolog Hadamard kernel K_hom (scope-conditional, supplementary)

`K_hom = K_A ⊙ K_B [⊙ K_D ⊙ ...]` would, in principle, capture cross-subgenome co-association beyond additive structure. Phase 5c tested this directly: a GBLUP REML paired bootstrap on 4 panels (wheat / cotton / rapeseed / strawberry), one additive-polygenic primary trait each, n_repeats=20.

**Result**: `σ²_h` boundary-collapsed in all 4 panels; Δr² range \|0.004–0.019\| **statistically detectable but direction-unstable**:

| Panel | n | Δr²(K_pool + K_hom vs K_pool) | sig 95% |
|---|---|---|---|
| Wheat Watkins | 827 | **−0.004** | ✓ (neg) |
| Cotton hebau | 419 | **+0.019** | ✓ (pos) |
| Rapeseed Horvath2020 | 297 | **−0.015** | ✓ (neg) |
| Strawberry Pincot 2018 | 564 | **+0.018** | ✓ (pos) |

**Interpretation** (verbatim from charter §2.1):

> Across four panels and paired bootstrap GBLUP-REML tests, the global K_hom showed statistically detectable but direction-unstable Δr² and boundary variance estimates, so it is not interpretable as a robust cross-panel positive variance component.

### Tier 1 / Tier 2 revival paths (Phase 4/5)

If you are extending this project, the charter lists three Tier-2 revival paths:

- **C + G**: synteny-aware **local K_hom** — Hadamard within syntenic ortholog blocks instead of global GRM; requires building `homoeolog_map`
- **B**: epistasis-rich trait class re-test — wheat heading date (Ppd × Vrn known digenic), cotton lint × fiber joint, disease R-gene panels
- See `docs/00_charter.md` §2.1 for the 5-item hard gate that revival must satisfy before K_hom can be promoted to main claim

### n-subgenome auto-fallback

`build_homoeolog_kernel` in `src/homoeogwas/kernel.py`:
- `n_sub = 1` → `None` (degenerate; LMM falls back to additive-only)
- `n_sub ∈ {2, 3}` → full Hadamard
- `n_sub ≥ 4` (strawberry 8n × 4 sub) → pairwise-mean `(1/C(n,2)) · Σ_{i<j} K_i ⊙ K_j` (PSD by Schur + sum-of-PSD; avoids 4-way Hadamard sparse collapse)

## 3. DL prior conditional lift (v1.5 extension)

After running per-SNP LMM, we score the suggestive-hit SNPs (typically `p < 5e-5`) plus their LD partners (`r² ≥ 0.2` within ±1 Mb) with two pretrained plant DNA language models:

- **PlantCaduceus** (225 M params, Caduceus architecture with mamba-ssm, ~5 plant species pretraining): SNP-position masked LLR over 512 bp window
- **AgroNT** (1 B params, NT architecture, ~70 plant species pretraining): 6144 bp window, 6-phase shifted median LLR

The two scores correlate moderately (Spearman 0.61 on wheat 31k SNPs), so they are combined as an equal-weight ensemble `mean(z(LLR_pc), z(LLR_agront))` then fused with GWAS:

```
fusion_score = z(-log10 p) + β · z(|LLR_ensemble|)
```

`β` is chosen per-panel by **LOQO grid search** over `{0.25, 0.5, 1, 2, 5}`. We pre-specify the LOQO procedure rather than the resulting β value — this is reported transparently in paper Methods.

**Cross-panel result** (Phase 3.3 / 3.4):

| Panel | Trait | β | recall@100 GWAS → fused | best single rank flip |
|---|---|---|---|---|
| Horvath2020 | bloom_50pct | 2 | 0.10 → **0.80** | *BnaC03.GA20ox* 221 → 2 |
| Horvath2020 | plant_height | 0.25 | 0.60 → 0.90 | *BnaC09.FLC* 148 → 72 |
| Cotton hebau | fiber_length | 5 | 0.00 → 0.20 | *GhMYB25* 763 → 44 |
| Cotton hebau | lint_percentage | 5 | 0.00 → 0.20 | *GhMYB25* 951 → 34 |
| Wheat Watkins (ref) | days_to_emerg | 0.25 | n/a | 0 (power-saturated) |

**Framing**: DL prior lift is **panel-size-dependent** — lift is largest where GWAS marginal power is most limited. Wheat 31k suggestive at n=827 is already power-saturated and shows no recall lift; this is a supplementary panel-size scaling story, not a failure.

## 4. Engineering: dual-GPU scan

The per-SNP scan in `src/homoeogwas/scan.py` runs on 2× RTX 3080 20 GB with no NVLink (`NCCL_P2P_DISABLE=1`). The wheat Watkins LOCO scan (83.3M markers × 827 samples × 21 chromosomes) completes in **~90 minutes** wall-clock. EMMAX backend is 0% slower than naive panmictic LMM at the same marker count.

This is the engineering contribution (Methods, not main scientific claim — per charter §2.3).

## References (selected)

- Voss-Fels K. et al. 2017 — per-subgenome kinship LMM in wheat
- Mascher M. et al. 2021 — barley reference + LMM
- Schur product theorem — elementwise product of PSD matrices is PSD (justifies K_hom validity)
- Self & Liang 1987 — boundary mixture χ² for variance component testing (Tier 1 A in charter §2.1)
- BOLT-LMM / BOLT-LOCO — Loh et al. 2015 (LOCO benefit ∝ per-chrom polygenic contribution)
- PlantCaduceus — Caduceus architecture + plant pretraining (charter §2.2)
- AgroNT — Nucleotide Transformer + plant pretraining (charter §2.2)
