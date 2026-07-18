# Pre-registration — molecular expression-product test of the cotton elongation homoeolog-pair interaction

> **Status: FROZEN on writing, committed to git BEFORE any expression data is downloaded or read.**
> The RNA-seq for these accessions exists only as raw reads (NCBI PRJNA891378); nothing about the
> per-accession expression of our lead genes is known yet. Every criterion below is fixed before the
> expression matrix exists on disk.
>
> Companion to `PREREGISTRATION_cotton_hebau_replication.md` (the genetic same-locus replication,
> which returned INCONCLUSIVE). This tests a DIFFERENT, molecular analogue of the same interaction in
> a DIFFERENT, independent population.

## 1. Question

Our genetic finding is a homoeolog-**pair-only** interaction: fibre elongation depends on the
combination of the A- and D-copy genotype burdens, while neither single copy is marginally associated
(discovery marginal P>0.5 for all copies; interaction P=1.2e-5 in CottonGVD). The molecular analogue:
does the **product of the two homoeologs' expression** predict fibre elongation, while neither
single-copy expression does — in an independent 376-accession population, at the trait-relevant
developmental stage?

This is SUPPORTING molecular evidence, not a replication of the genetic test. It cannot promote the
lead beyond proof-of-concept; it can corroborate or fail to corroborate the pair-interaction logic at
the expression level.

## 2. Population, genes, stage — all fixed here

- **Panel:** Zhang/Wang/Tu et al., *Nat. Genet.* 2023 (DOI 10.1038/s41588-023-01530-8), 376 upland
  cotton accessions (S001–S376), HAU TM-1 (`Ghir_`) reference. Genetically independent of both
  CottonGVD (discovery) and HBAU (the genetic-replication panel) — a third population.
- **Lead pair (PRIMARY):** `Ghir_A01G002400` (A) × `Ghir_D01G002310` (D) = glyoxalase I / lactoyl-
  glutathione lyase. Confirmed = our discovery elongation lead `Gh_A01G025100`×`Gh_D01G022200` by
  diamond blastp, 100% identity, 100% coverage, reciprocal homoeolog, paralog margin ≥71 bits
  (`results/phase7/cotton_multiomic/hau_ref/hits_v11.tsv`).
- **Lead pair (SECONDARY):** `Ghir_A05G013600` × `Ghir_D05G013340` = MYB-like, our fibre-length lead.
  Reported but not primary: the length pair had near-silent molecular support in the paper's own SI
  (no cis-eQTL, no TWAS), so it is a weaker a-priori candidate; a Bonferroni factor of 2 applies if it
  is counted (see §6).
- **Stage:** **20 DPA (primary).** Fixed from biology BEFORE any expression is read: fibre elongation
  is a late-fibre process, and the paper's own cis-eQTL for `Ghir_A01G002400` peaks at 20 DPA
  (R²=0.246, P=7.9e-28). 16 DPA is a pre-specified secondary/sensitivity stage. No stage is chosen
  after seeing the expression-product result. Only 20 DPA (and, for sensitivity, 16 DPA) reads are
  downloaded (369 runs, 1.57 TB); the earlier stages are deliberately not fetched.

## 3. Phenotype

Fibre elongation (FE) from Supplementary Table 13 of the same paper (373 S### accessions with
FL/FS/FE/FU; already in hand at `results/phase7/cotton_multiomic/MOESM4.xlsx`, sheet "Supp Table 13").
Matched to expression by the S### accession id. No other trait enters the primary test; fibre length
(FL) is tested only for the secondary MYB pair.

## 4. Expression pipeline (fixed)

- Download the 20 DPA (primary) and 16 DPA (secondary) RNA-seq runs of PRJNA891378 to
  `/mnt/lldata/cotton_multiomic_376/`, md5-verified against the ENA manifest
  (`results/phase7/cotton_multiomic/manifest_20DPA.tsv`, frozen).
- Quantify each run with **salmon** against the HAU TM-1 (`Ghir_`) transcriptome (the paper's own
  reference), TPM. Same tool convention as the strawberry population-expression pipeline.
- Extract TPM of the four lead genes per accession at 20 DPA. INT-transform each gene's TPM across
  accessions (robust to the heavy-tailed expression common in these data).
- One run per (accession, stage); where an accession has replicate runs at a stage, average TPM.

## 5. Test and success criteria — fixed in advance

Per pair, at 20 DPA, with the FE phenotype (INT-transformed):

    FE ~ 1 + ExprA + ExprD + ExprA:ExprD + (expression PCs 1..K)

where ExprA, ExprD are the INT-transformed per-accession TPM of the two copies, and the top K
expression principal components (K pre-set = 5) absorb batch/structure — necessary because the
RNA-only design has no genotype and therefore no kinship matrix (a stated limitation, §7).

- **CORROBORATED** = the interaction coefficient `ExprA:ExprD` is significant at α (below) **AND**
  neither single-copy main effect (ExprA, ExprD) is significant on its own — i.e. the molecular signal
  is pair-only, mirroring the genetic result.
- **MARGINAL-ONLY** = a single-copy expression main effect is significant (with or without the
  interaction). This does NOT support the pair-interaction logic and is reported as such.
- **NOT CORROBORATED** = no significant interaction and adequate power (§7).
- **INCONCLUSIVE** = no significant interaction while underpowered by §7.

Direction: the genetic interaction anchor is POSITIVE, but burden-product and expression-product are
different scales and the sign is NOT portable between them, so direction is reported descriptively,
not as a hard criterion (unlike the genetic replication).

## 6. Multiplicity

Primary = the FE glyoxalase pair at 20 DPA: one pair, one stage, one trait → α = 0.05. The FE lead is
pre-specified as primary on the strength of its independent cis-eQTL support in this same panel. If
the secondary MYB/FL pair is also counted as a discovery-grade test, a Bonferroni factor of 2
(α = 0.025) applies to the family of two; it is reported at both α=0.05 (nominal) and α=0.025
(family-corrected). The 16 DPA stage is a sensitivity check, not an independent test, and is not
added to the multiplicity family.

## 7. Power / limitation — stated before the read-out

n ≈ 369 accessions at 20 DPA. This is comparable to the HBAU genetic-replication panel (n=419), so
the test is not high-powered for a small interaction; a null is INCONCLUSIVE rather than a refutation.
The specific limitations of the RNA-only design, all fixed here:
- No genotype for these accessions in this path → no kinship control; structure is absorbed only by
  expression PCs, which is weaker.
- Expression-product is a correlational, cross-sectional analogue of the genetic interaction, not a
  causal or genetic test; a positive result is molecular corroboration, not proof.
- The reference is HAU TM-1 (`Ghir_`), not the Gh_CRI v1 discovery assembly; gene identity is fixed by
  the frozen 100%-identity crosswalk (§2), not re-decided.

## 8. Reporting commitment

The FE glyoxalase pair result is reported whatever the outcome — CORROBORATED, MARGINAL-ONLY, NOT
CORROBORATED, or INCONCLUSIVE — with the interaction coefficient, its p, the two single-copy main-
effect p-values, n, the covariate set, and the 16 DPA sensitivity. The secondary MYB/FL pair is
reported alongside. A null is stated as such and does not weaken the paper, whose spine is the method.

## 9. Stopping rule

This is one pre-specified test (FE glyoxalase pair, 20 DPA primary, 16 DPA sensitivity) plus one
secondary pair. If it does not corroborate, we do not sweep other stages, other genes, other traits,
other covariate sets, or other expression transforms in search of significance. Those are separate
hypotheses requiring their own pre-registration. The already-banked, download-free evidence — that
`Ghir_A01G002400` is a strong all-stage cis-eQTL fibre gene invisible to marginal/GWAS methods
(SI Table 5) — stands on its own regardless of this test's outcome.
