# Pre-registration — independent replication of the wheat chr1 B–D homoeolog interaction

**Frozen on 2026-07-13, before any replication-panel phenotype was joined to genotype.**
Everything below is fixed. Deviations, if any are forced, will be reported as deviations.

## 1. The hypothesis being replicated

A single, pre-specified homoeolog gene pair shows an interaction affecting heading date:

| | |
|---|---|
| B-subgenome gene | `TraesCS1B02G143800` — chr1B:195,552,668–195,560,543 (IWGSC RefSeq v1.0) |
| D-subgenome gene | `TraesCS1D02G128400` — chr1D:141,545,585–141,548,899 |
| Window | gene body ± 2 kb |
| Discovery panel | Watkins landraces, WGS, n = 827, trait `days_to_emerg` (heading) |
| Discovery result | strictly-invariant omnibus `omniB` P = 3.93e-6; top hit of a genome-wide rescan of 12,411 homoeolog pairs |

**Exactly one pair is tested.** No other pair, no other trait, no re-selection.

## 2. Replication panel

355 bread-wheat accessions, whole-genome sequenced at 14.5×, called against IWGSC RefSeq v1.0
(NGDC GVM000315; Niu et al. 2023, *Plant Cell* 35:4199). Chromosome naming and coordinates are
identical to the discovery panel, so no liftover is applied.

Marker content at the two genes was checked before this document was written (this is a technical
feasibility check, not a result): 204 SNPs in the B gene and 13 in the D gene at MAF ≥ 0.01 — enough
to build every component of the test. The WHEALBI exome panel is *not* used: its D gene carries a
single SNP, so the test is **non-evaluable** there. That is a genotyping limitation, not a failed
replication, and it will be described as such.

## 3. Phenotype

Two independent sources, analysed separately, both pre-specified:

**(a) GRIN (available now, primary for this round).** USDA GRIN descriptor `Days to Flowering`
— defined as the day (from 1 January) when 50 % of the spikes are fully exserted from the boot,
expressed relative to a check — from the `WHEAT.AGRON.MARICOPA.*` trials. This is the same
biological event as the discovery trait (spike exsertion). 227 of the 355 accessions carry a USDA PI
number; those with a Maricopa observation are used.
Harmonisation, fixed in advance: observations are **z-scored within trial (study × year)** and then
pooled; trial is not otherwise modelled. `Days to Anthesis` in absolute Julian days is **excluded**
because it is not comparable across locations (Maricopa ≈ 82 d vs Aberdeen ≈ 190 d).

**(b) The panel's own heading date (if the authors provide it).** Heading days (HD), Zhao County,
2013–2016. If obtained, this becomes the primary phenotype (larger n, single trial) and (a) becomes
an independent second environment.

Either way the phenotype is rank-inverse-normal transformed before testing, as in discovery.

## 4. Independence

Verified genomically, not from metadata: cross-panel identity-by-state is computed between the 355
replication accessions and the 827 Watkins accessions on shared SNPs, and **any replication accession
with IBS > 0.99 to a Watkins accession is dropped**. Country of origin is *not* used as the
independence criterion — the Watkins collection was itself assembled from Turkey, Iran, Pakistan,
Afghanistan and India, which are also represented among these landraces.

Pre-specified sensitivity analysis: repeat on the 175 modern cultivars only (103 Chinese + 72 US),
which cannot overlap a 1920s landrace collection by construction.

## 5. The test

Identical to discovery: subgenome-stratified GRM (K_B, K_D) whitened LMM, phenotype INT-transformed.

- **Primary statistic**: `omniB = ACAT(minor-allele burden product, PC1×PC1, low-rank
  kernel-Hadamard)` — the strictly REF/ALT-encoding-invariant omnibus.
- **Direction anchor**: the omnibus is unsigned, and a minor-allele burden's sign is not portable
  across populations (the minor allele may differ). Direction is therefore taken from the
  **REFERENCE-allele-coded** burden-product interaction coefficient, which *is* portable because both
  panels are called against the same reference. The discovery sign is frozen in
  `results/phase7/replication_direction_anchor.json`, computed before the replication phenotype was
  touched.

## 6. Success criteria (fixed)

The replication is declared **successful** only if BOTH hold:

1. `omniB` P < 0.05 on the single pre-specified pair (one test — no multiplicity correction is owed,
   and none is claimed beyond it), and
2. the REF-coded burden-product interaction coefficient has the **same sign** as the frozen discovery
   anchor.

If (1) holds but (2) does not, we report a **same-locus association with discordant direction** and
do **not** claim replication.

If (1) fails, we report a **failed replication** together with the achieved power.

## 7. Power, acknowledged in advance

Scaling the discovery effect (|t| ≈ 4.6 at n = 827):

| phenotype source | usable n (expected) | expected \|t\| | expected P | power at α=0.05 |
|---|---|---|---|---|
| GRIN (Maricopa) | ~130–170 | ≈ 2.0 | ≈ 0.05 | **~50 %** |
| authors' HD | 355 | ≈ 3.0 | ≈ 0.003 | ~80 % |

The GRIN round is therefore **underpowered by design**, and a negative result from it will be
reported as *inconclusive*, not as a refutation. It also samples a different environment (Arizona
desert vs. UK/China), so a genotype-by-environment failure is a live alternative explanation for a
negative. Both of these are stated here, in advance, so that a negative cannot be reinterpreted after
the fact.

## 8. What is reported regardless of outcome

The result of the single frozen test, the achieved n, the achieved power, the direction, and the
genome-wide rescan of the replication panel as a **secondary** (not confirmatory) analysis.
