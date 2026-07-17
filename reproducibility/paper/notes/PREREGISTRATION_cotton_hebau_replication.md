# Pre-registration — same-locus replication of the two CottonGVD homoeolog-pair interactions in the HBAU/NDM8 panel

> **Status: FROZEN on writing. Nothing below may be revised after the gene mapping is run.**
>
> This document exists because the replication p-values are ALREADY ON DISK. The HBAU panel was
> rebuilt at full density and scanned genome-wide for all 13 traits on 2026-06-29
> (`results/viz_preview/fig3_cotton_locus/rescue/gw13/`). Nobody has yet mapped the CottonGVD lead
> genes onto that panel's assembly, so nobody has yet looked up their p-values — but the moment the
> mapping runs, the answer is visible. Every criterion below is therefore fixed BEFORE the mapping,
> and the result is reported whatever it is.
>
> Modelled on `PREREGISTRATION_wheat_chr1BD_replication.md` (frozen 2026-07-13), which governs the
> equivalent wheat test.

## 1. What is being tested

Exactly **two** pre-specified homoeolog pairs, discovered in the CottonGVD panel (n=1,245,
Gh_CRI v1 = GCA_006980745.1):

| trait | pair (Gh_CRI v1) | discovery interaction P | trait-wise permutation FWER |
|---|---|---|---|
| fibre length | `Gh_A05G118800` × `Gh_D05G131200` (MYB-like / MYBS3) | 5.99e-6 | 0.018 |
| fibre elongation | `Gh_A01G025100` × `Gh_D01G022200` (glyoxalase I) | 1.23e-5 | 0.036 |

No other pair. No other trait. No re-selection of either. The two discovery traits map onto the HBAU
panel's `fiber_length_BLUE` and `elongation_BLUE`; no other of its 13 traits enters this test.

**Scope note carried from discovery.** These two leads are trait-wise genome-wide significant but do
NOT survive experiment-wide Bonferroni across the four-trait CottonGVD grid (they pass BH-FDR 0.10).
The manuscript presents them as proof-of-concept. This replication cannot promote them beyond what
the discovery supports; it can only corroborate or fail to corroborate them.

## 2. Replication panel

HBAU/NDM8 panel (`GCA_018997965.1`), n=419, phenotypes `data/processed/cotton/pheno_m3_4_blue.tsv`.

Genotypes: **`results/viz_preview/fig3_cotton_locus/rescue/{A,D}_wgs`** — 1,700,604 + 1,082,561 =
2,783,165 variants, built 2026-06-29 with `--geno 0.2 --maf 0.01 --max-alleles 2 --snps-only`.

The 498,106-variant set used elsewhere in this project is NOT used here and must not be: an
`--hwe 1e-6` filter removed 1,997,332 variants (71.2%) from a panel of inbred lines, where
Hardy-Weinberg is not a meaningful QC criterion, and heterozygote-deficiency at collapsed homoeologs
is precisely the artefact under study. This is also why the manuscript's description of this panel as
a "4.98e5-SNP array" is wrong on two counts and must be corrected independently of this test.

## 3. Gene mapping (run AFTER this document is frozen)

Discovery is on Gh_CRI v1; the replication panel is on NDM8. DIAMOND reciprocal best hits between the
two full proteomes (`data/reference/cotton/CRI_TM1_v1/*.gff.pep.gz`, 79,703 seqs; and
`/mnt/nvme/cotton_hbau/cottongen/NDM8.pep.fa`, 80,124 seqs), restricted to the four lead genes.

**A pair is declared NON-TESTABLE, and is reported as such rather than substituted, if any of:**

- either lead gene has no reciprocal best hit in NDM8;
- the RBH is not subgenome-consistent (A→A, D→D);
- the best hit is not separated from the next-best by a clear margin, i.e. a paralog is in contention;
- either mapped gene carries fewer than 3 callable SNPs in the `{A,D}_wgs` set.

**Substituting a paralog for an absent ortholog is the exact error that retired this panel's own
previous hits** (`GhM_A11G2420|GhM_D11G2742` and `GhM_A06G1605|GhM_D06G1557`, dropped as MATE/RLK
paralog mispairings). A non-testable pair is an honest outcome; a substituted one is not.

`rbh_ndm8_to_nau.tsv` on disk maps NDM8↔**NAU v1.1**, a different assembly whose gene IDs share the
`Gh_` prefix but not the namespace (4-digit vs 6-digit suffix). It must not be reused here.

## 4. Test

Identical estimand and statistic to discovery, via the same engine (`src/homoeogwas`):

- subgenome-stratified GRM (K_A, K_D) whitened LMM; INT-transformed phenotype;
- per-gene burden cap 150, min 3 SNPs, MAF ≥ 0.01;
- primary statistic **omniB** = ACAT(minor-allele burden product, PC1×PC1, low-rank kernel-Hadamard);
- calibration by covariate-aware Freedman-Lane permutation, `perm_B ≥ 2000` (the existing `gw13`
  scans were run with `perm_B: 0` and are used only to read the asymptotic p; the permutation run is
  part of this test, not a later addition).

## 5. Direction anchor

The omnibus is unsigned and a minor-allele burden's sign is not portable between panels. Direction is
taken from the **REF-allele-coded** burden-product interaction coefficient. Unlike the wheat case,
the two cotton panels are called against **different assemblies** (Gh_CRI v1 vs NDM8), so the REF
allele is not guaranteed to be the same physical allele.

**Therefore, before the test, the REF allele at each mapped gene's SNPs must be verified to be the
same physical allele in both assemblies.** If it is not, and cannot be made so by strand/allele
harmonisation against the two reference FASTAs, **the direction criterion is dropped and this is
recorded**, leaving a magnitude-only test — which is weaker, and must be labelled so.

The anchor itself is computed from the DISCOVERY panel only
(`discovery_direction_anchor.py --species cotton_cottongvd`), before the replication p-values are
read, and frozen to `results/phase7/cotton_direction_anchor_{FibLen,FibElo}.json`.

## 6. Success criteria — fixed here, in advance

Per pair, with a Bonferroni threshold over the two pairs of **α = 0.05/2 = 0.025**:

- **CORROBORATED** = omniB P < 0.025 **AND** direction concordant with the frozen anchor.
- **DISCORDANT** = omniB P < 0.025 but the sign is opposite. This is *same-locus association with
  discordant direction*, and is NOT a replication.
- **NOT CORROBORATED** = omniB P ≥ 0.025, *and* the pre-computed power (§7) shows the test was
  adequately powered.
- **INCONCLUSIVE** = omniB P ≥ 0.025 while underpowered by §7. Reported as inconclusive, never as
  refutation.
- **NON-TESTABLE** = the mapping conditions in §3 fail.

## 7. Power bound — computed BEFORE the p-values are read

n=419 here against n=1,245 at discovery, and the discovery P-values are only 5.99e-6 and 1.23e-5.
The minimum detectable interaction PVE at 80% power is computed by exact noncentral-t
(`replication_power_bound.py`) using the mapped genes' actual SNP counts, and written to
`results/phase7/cotton_replication_power_bound.json` before §6 is evaluated.

**If the discovery effect size lies below that bound, this test cannot refute the discovery and a
null result carries no evidential weight against it.** Stating this in advance is what makes a null
interpretable; it is the discipline that converted the wheat 355-panel result from "failed
replication" into "non-evaluable" after its Vrn/Ppd positive controls came back silent.

## 8. Independence — the weakest link, and it is not yet established

The two panels' sample IDs do not intersect (HBAU: `B001–B083`, `D…`, `F…`, `L…`; CottonGVD:
`GH0086…`, `ZGH…`). **This is absence of evidence, not evidence of disjointness**: the schemes are
different namespaces, so a genuine germplasm overlap would be invisible to string matching. Both are
Chinese upland-cotton diversity panels; overlap is plausible.

Independence is therefore established **genomically or not at all**: IBS between the panels on
markers lifted to a common assembly; any HBAU accession with IBS > 0.99 to a CottonGVD accession is
dropped, and the count dropped is reported.

**Until that check has run, the word "independent" is not used for this test.** Should the check
prove infeasible (no common marker set), the result is reported as *same-locus corroboration in a
second panel of unverified independence* — which is weaker than replication and must be labelled so.

## 9. Reporting commitment

Both pairs are reported with omniB P, the permutation FWER, the REF-coded interaction coefficient and
its sign, the mapped NDM8 gene IDs with their RBH identity/coverage and next-best margin, the SNP
counts on both copies, the power bound, and the independence verdict — **whatever the outcome, and
including any pair declared non-testable**.

Per-environment sensitivity: the panel carries only **two** years (2014, 2015), not the twelve an
earlier note claimed. A two-point meta-analysis is reported as a sensitivity check, never as the
primary. Note the elongation BLUE's year-to-year correlation is only r=0.64 in this panel — its own
reproducibility bounds what any replication of the elongation pair can show.

## 10. Stopping rule

This is one test of two pre-specified pairs. If it does not corroborate them, we do not re-map, do
not widen to neighbouring genes, do not switch to a local-haplotype variant of the statistic, and do
not scan the panel for something else to report. Those are separate hypotheses requiring their own
pre-registration and their own denominator.
