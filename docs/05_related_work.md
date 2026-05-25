# 05 — Related Work & Positioning

## TL;DR

| Tool | Year | What it does | What it does NOT do |
|---|---|---|---|
| **NodeGWAS** | 2026 (Letter; Zhang Yixing et al.) | uses graph-pangenome **node coverage** as GWAS predictor; tested on Arabidopsis + sugarcane | no subgenome model, no homoeolog epistasis, no DL prior, no GPU, no LMM tricks |
| **networkGWAS** | 2023 *Bioinformatics* (Muzio et al.) | aggregates SNPs over PPI neighborhoods → set-based LMM | not polyploid, no SV, no dosage, no GPU |
| **GWASpoly** | 2016 *Plant Genome* (Rosyara & Endelman) | R single-thread polyploid GWAS, 8 gene-action models | small datasets only; no SV; no GPU |
| **MultiGWAS** | 2021 *Plant Cell Rep* | wraps GWASpoly + SHEsis + GAPIT + TASSEL, consensus vote | speed of slowest member |
| **Varigraph** | 2025 *Plant Commun* | pangenome graph genotyper for diploid + polyploid (incl. autopolyploid dosage) | genotyper only, no GWAS layer |
| **REGENIE** | 2021 *Nat Genet* (Mbatchou et al.) | two-step LMM with stacked ridge → score test + SPA | diploid only |
| **SAIGE-GENE+** | 2022 *Nat Genet* (Zhou et al.) | rare-variant set-based GLMM with SPA | diploid only |
| **fastGWA-GLMM** | 2021 *Nat Genet* | sparse GRM, biobank-scale binary GWAS | diploid only |
| **STAAR** | 2022 *Nat Methods* | annotation-weighted rare-variant set test | diploid only |
| **DeepRVAT** | 2024 *Nat Genet* (Clarke et al.) | deep-set network → per-gene impairment score | diploid only |
| **DeepNull** | 2021 *Nat Commun* (Hormozdiari et al.) | DNN residualizes nonlinear covariates | not for variant-level discovery |
| **PlantCaduceus** | 2025 *PNAS* (Zhai et al.) | Mamba-based plant DNA-LM, 16 species | not a GWAS tool — usable as variant-effect prior |
| **AgroNT** | 2024 *Commun Biol* (Mendoza-Revilla et al.) | 1B-param Nucleotide Transformer for 48 crops | same as above |

---

## How we position vs each

### NodeGWAS (Zhang et al. 2026, *Letter*) — **upstream, not competitor**

NodeGWAS contribution = **representation** (node count) — they still run a vanilla single-marker association on top.

- Their own benchmark says **power: k-mer > node > SNP** — i.e., they trade some power for interpretability vs k-mer GWAS.
- They do **not** address: subgenome partitioning, dosage allelic series, homoeolog epistasis, DL prior, GPU.

**Our positioning**: HomoeoGWAS **consumes** NodeGWAS output (or Varigraph / PanGenie output) as one type of dosage feature, then layers:

1. Subgenome-stratified LMM
2. Homoeolog Hadamard kernel
3. DL functional prior
4. GPU acceleration

Frame in paper intro: *"NodeGWAS demonstrated graph nodes as a powerful feature representation; here we ask: given those features, what is the right statistical engine for an allopolyploid?"*

### networkGWAS (Muzio et al. 2023) — true methodological competitor at the **statistical layer**

It also has a "non-trivial statistical layer beyond vanilla GWAS" (PPI-set test + permutation). But:

- Diploid only.
- No SV.
- Cohort scale limited by CPU LMM + permutation.

In benchmarks, run networkGWAS as a baseline on the diploid Arabidopsis demo. On polyploids, mark "N/A (not supported)".

### GWASpoly & MultiGWAS — standard polyploid baselines

Run on every dataset where they finish in <24 hr. Expect them to fail or take days on Watkins (1k samples × 10⁷ SNPs). Document the wall-clock differential — this is part of v1's selling point.

### REGENIE / SAIGE-GENE+ / fastGWA — diploid speed baselines

Run them with ploidy forced to 2 (collapse dosage to {0,1,2}). They will lose subgenome signal but show calibration baseline. We expect HomoeoGWAS to match their speed (within 2–3×) while gaining biology.

### STAAR — variant-set baseline

Reference for our DL-prior insertion. Our weight function (PlantCaduceus log-LR) is plug-compatible with STAAR's annotation weights.

### DeepRVAT — DL aggregator baseline

If v2 includes rare-variant set tests, DeepRVAT is the direct comparator. v1 does not benchmark against it explicitly.

### PlantCaduceus / AgroNT — DL backbone, not baseline

We **use** them; we do not benchmark **against** them. They produce per-variant priors; we use those priors.

---

## Citation-ready one-paragraph summary

> Polyploid GWAS has accumulated three independent lines of work: dedicated polyploid statistical tools that run on CPU (GWASpoly, MultiGWAS), modern graph-pangenome representations that improve variant coverage (NodeGWAS, Varigraph, PanGenie), and diploid biobank-scale fast LMM engines (REGENIE, SAIGE-GENE+, fastGWA-GLMM). None unite them. In particular, none model **homoeolog-pair epistasis**, none integrate **graph-node dosage and SNP dosage in a single subgenome-stratified mixed model**, and none plug in **zero-shot plant DNA-language-model priors** for rare-variant power. HomoeoGWAS is the first framework to do all three on a GPU-native Rust/JAX/Triton stack, demonstrated on wheat (Vrn × Ppd flowering, FHB), cotton (A_t × D_t fiber), rapeseed (glucosinolate SV-eQTL), and (v2) octoploid strawberry.

---

## Forward-looking references to track (RSS the search)

- *Genome Biology*, *Nat Methods*, *Nat Genet* keyword "polyploid GWAS" (monthly)
- bioRxiv keyword "subgenome epistasis" (weekly)
- GitHub watch: `zhangyixing3/NodeGWAS`, `jendelman/GWASpoly`, `BorgwardtLab/networkGWAS`
- HuggingFace: search "plant", "agronomy", "genome" — new DNA-LMs appear monthly
