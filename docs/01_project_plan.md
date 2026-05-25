# 01 — Project Plan & Milestones

Start: **2026-05-13**. v1 target submission: **~2027-Q2** (12 months).

---

## Scope discipline

The user's original ask included four breakthroughs. To keep v1 publishable in 12 months on a single workstation (2× RTX 3080 20 GB), we **stage them across two papers**:

### v1 — "main" paper, 12 months, target *Genome Biology* / *Nature Methods*

Core claims:
1. Subgenome-stratified LMM with **homoeolog Hadamard epistasis kernel**
2. **Graph-node dosage** (NodeGWAS-compatible) as multi-allelic SV feature in the same LMM
3. **GPU-native** Rust + JAX/Triton stack with end-to-end Watkins (1k+ samples) under 4 GPU-hours
4. Two real-data demos: **wheat (Vrn × Ppd flowering, FHB resistance) + cotton (A_t × D_t fiber)**

### v2 — "extension" paper, +6 months, target *Plant Cell* / *Molecular Plant*

5. **DL functional prior** (PlantCaduceus zero-shot, STAAR-style weight)
6. **Octoploid strawberry stress test** (USDA panel)
7. Long-tail / rare-allele set-based test

### Descope order (cut from the END if blocked)

1. Octoploid strawberry (v2 only) ← cut first
2. DL prior (v2 only) ← cut second
3. 3-way homoeolog epistasis (only do pairwise A↔B, A↔D, B↔D)
4. Bayesian VI fallback (keep AI-REML only)

Floor: **Rust I/O + GPU LMM + subgenome GRM + graph-node dosage + AlphaSimR benchmark**. These five alone are publishable in *Bioinformatics*.

---

## v1 schedule (12 months)

### Phase 0 — Foundation (Month 1, weeks 1–4)

| Week | Goal | Deliverable |
|---|---|---|
| W1 | Project setup, env scaffolding | conda env, cargo workspace, CI on GitHub |
| W1 | Reference genome downloads (wheat CS RefSeq v2.1, cotton TM-1, canola ZS11) | `data/reference/` populated |
| W2 | NodeGWAS hands-on reproduction on Arabidopsis | `benchmarks/NodeGWAS/` running on demo |
| W2 | Watkins WGS download initiated (background) | partial `data/raw/wheat/watkins/` |
| W3 | Pgen / Zarr I/O Rust prototype: read 1k samples × 100k SNP | `src/rust_io` v0.1 |
| W4 | AlphaSimR allopolyploid simulator + custom PolySimX homoeolog-epistasis module | `scripts/simulate_polypoly.R` + sanity-check VCF |

### Phase 1 — Core engine (Months 2–4)

| Month | Goal | Deliverable |
|---|---|---|
| M2 | Subgenome GRM construction on GPU (K_A, K_B, K_D); matrix-free CG | `src/polygwas/grm.py` + Triton kernel |
| M2 | REGENIE-style step-1 stacked ridge on GPU | passes regression test on simulated null |
| M3 | Step-2 score test + SPA for skewed traits | matches REGENIE p-values on diploid demo |
| M3 | Bubble-VCF reader (vg / minigraph-cactus / PanGenie) | accepts NodeGWAS-style node-count input |
| M4 | Homoeolog Hadamard kernel `K_AB = K_A ⊙ K_B` with ortholog-bin low-rank | passes simulated power test |

### Phase 2 — Real data (Months 5–8)

| Month | Goal | Deliverable |
|---|---|---|
| M5 | Wheat Watkins end-to-end run (flowering time) | recapitulates Vrn-1, Ppd-1 loci |
| M5 | Vrn × Ppd homoeolog interaction scan | first novel epistasis call |
| M6 | Cotton 1081 / 3724 panel pipeline | A_t × D_t fiber QTL scan |
| M7 | Rapeseed B. napus 991 / 2105 SV panel | glucosinolate / oil locus scan |
| M8 | Benchmark vs NodeGWAS, GWASpoly, REGENIE, networkGWAS | full table + Manhattan plots |

### Phase 3 — Writeup (Months 9–12)

| Month | Goal | Deliverable |
|---|---|---|
| M9 | Method paper draft (Intro + Results) | shareable draft |
| M10 | Figures (nature-figure skill, Python) + code freeze v0.9 | submission-quality figs |
| M11 | bioRxiv preprint + GitHub public release | DOI + tagged release |
| M12 | *Genome Biology* / *Nat Methods* submission | submitted manuscript |

---

## Risks & checkpoints

| Risk | Trigger | Action |
|---|---|---|
| Wheat WGS download too slow (~30 TB raw) | M2 still pulling | switch to VCF-only download (∼500 GB) or use VMap2.0 mirror |
| RAM < 128 GB cannot hold Watkins GRM | M2 fails | shard GRM by chromosome arm, accept 1.5× slowdown |
| AI-REML diverges on 7 variance components | M2 ablation | EM warm-start; fall back to 4-component (K_A, K_B, K_D, ε) |
| Homoeolog epistasis = LD artifact | M4 simulation null shows inflation | bin homoeolog pairs by orthogroup distance, report stratified λ_GC |
| NodeGWAS upstream too brittle | M3 integration | use Varigraph or PanGenie instead — same node-VCF schema |

---

## Decision points (need user input)

- [ ] **D1 (W1)**: Final tool name (HomoeoGWAS? AlloPolyGWAS? other?) → check GitHub availability.
- [ ] **D2 (W1)**: License — MIT vs BSD-3 vs Apache-2.0.
- [ ] **D3 (M3)**: Whether to keep raw WGS or work from VCF-only mirrors (disk budget).
- [ ] **D4 (M5)**: Lead wheat trait — flowering time (clean genetics) or yield (high impact, noisier).
- [ ] **D5 (M9)**: Journal target — *Genome Biology* (faster) vs *Nature Methods* (slower, prestige).
