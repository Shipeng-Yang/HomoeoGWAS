# Pre-registration — multi-locus replication of CottonGVD homoeolog-pair interactions in the hebau panel

Frozen before any hebau phenotype was read for these loci. Committed prior to running the test.

## Why this is not the earlier single-lead attempt

An earlier pre-registration (`PREREGISTRATION_cotton_hebau_replication.md`, blob 3b675e8ca696)
tested the two published leads in this panel. Its fibre-length lead was **NON-TESTABLE** (the A copy
has zero callable SNPs within ±2 kb even at full 2.8 M-SNP density) and its fibre-elongation lead was
**INCONCLUSIVE** (interaction P = 0.37, power ~62–78 %). That outcome is reported unchanged; nothing
here supersedes or reinterprets it.

This protocol tests **different loci that have never been examined in the hebau panel**. It exists
because a discovery-side audit — which read no hebau phenotype — showed that the discovery panel's
signal is sparse but real in the extreme tail (6 pairs at P < 1e-4 against 0.35 expected under a
1000-replicate calibrated null, empirical P = 0.004), and that several of those pairs reconstruct
cleanly in hebau while the published lead does not.

## Candidate selection (discovery-side only, no hebau outcome involved)

Candidates were the CottonGVD FibLen pairs with interaction P < 1e-3 (9 pairs) plus the published
FibElo lead. Reconstructibility in hebau was assessed by the criteria of the earlier protocol,
verbatim: protein-level reciprocal best hit, subgenome-consistent, no same-subgenome paralog within
the bitscore margin, and ≥ 3 callable SNPs within ±2 kb on **each** copy, counted on the full-density
(2.78 M SNP) hebau genotypes. Five of ten pairs passed
(`results/phase7/cotton_multilocus_reconstructibility.json`).

## Loci tested (frozen)

Primary — three independent loci, none previously examined in hebau:

| ID | A copy | D copy | discovery P | hebau SNPs A/D |
|----|--------|--------|-------------|----------------|
| L1 | Gh_A08G064900 | Gh_D08G060500 | 6.1e-6 | 13 / 15 |
| L2 | Gh_A06G029400 | Gh_D06G029100 | 2.1e-5 | 21 / 15 |
| L3 | Gh_A09G109300 | Gh_D09G103000 | 1.5e-4 | 8 / 10 |

Secondary — **not an independent test**, reported as a within-region sensitivity for L1 because it
sits in the same chromosome-8 region and is in linkage with it:

| S1 | Gh_A08G065900 | Gh_D08G061400 | 1.4e-5 | 47 / 11 |

Excluded and why: three pairs have < 3 callable SNPs on at least one copy, one fails reciprocal best
hit, and the published FibLen lead has zero SNPs on its A copy. The published FibElo lead is
testable but was already examined; its P = 0.37 stands and is **not** counted as new evidence.

## Phenotype

`fiber_length_BLUE` in the hebau panel (n = 419), the trait corresponding to the discovery scan
(FibLen). No other hebau trait is examined under this protocol; testing further traits would require
a separate protocol and its own multiplicity correction.

## Test

Identical engine and settings as discovery: per-subgenome GRMs, REML whitening, burden = capped mean
of column-standardised dosages (cap 150, min 3 SNPs), GLS t-test of the burden product with both
main effects in the model. Burdens are rebuilt from the hebau panel's own markers (the discovery
SNPs are not required to be present) — this is the same-gene-pair, not the same-SNP, replication
target, which is the target our own framework predicts should be replicable.

## Direction

The discovery interaction coefficient sign is frozen per locus before testing. Replication requires
the **same sign**; an opposite-sign result at any P value is a failure, not a replication.

## Success criteria (frozen, in advance)

Primary: **at least 2 of the 3 independent loci show the frozen direction with one-sided P < 0.05.**

Secondary, reported regardless: Stouffer combination of the three one-sided P values (equal
weights), and per-locus Bonferroni (P < 0.0167).

Declared outcomes:
- **REPLICATED** — primary criterion met.
- **PARTIAL** — exactly 1 of 3 loci meets direction + P < 0.05.
- **NOT REPLICATED** — 0 of 3, with power reported.
- **INCONCLUSIVE** — pre-test power < 50 % for the assumed effect size; in that case a null result
  is reported as uninformative rather than as evidence of absence.

## Power, computed before the test

Power is simulated by injecting the discovery-panel effect into the **real hebau genotypes and real
kinship**, at the discovery partial R² and at half of it (winner's-curse haircut), and is reported
alongside the result whatever the outcome. If power at the halved effect is below 50 % the study is
declared INCONCLUSIVE in advance.

## Positive control

The pipeline's ability to detect a real interaction in this panel was established previously
(injected PVE 0.5 recovered at P = 1.9e-64; a genome-wide scan of this panel returns ~10 pairs at
P < 1e-3; the tested pair's own statistic is non-degenerate). No new positive control is required,
but the previous one is cited in the report.

## Anti-circularity commitments

1. Loci, trait, direction, statistic, criteria and power protocol are fixed by this document.
2. The hebau phenotype is read exactly once, under the model specified here.
3. No locus may be added, dropped or re-weighted after the phenotype is read.
4. If the result is negative, this document is reported in full alongside it; the earlier
   single-lead outcomes and the pseudo-pair negative are reported in the same place, so the reader
   sees every replication attempt rather than only the surviving one.
5. The hebau panel has been used before (single-lead protocol), so it is **not** a pristine blinded
   panel. That is disclosed here; the defence is that these three loci specifically have never been
   examined in it.

---

# Amendment 1 — written before any hebau phenotype was read

## The fault

Candidates were selected on the `aggregate_perpair` table, whose p-value is **omniB** — an ACAT
combination of (minor-allele burden product, gene-PC1 × PC1 product, low-rank kernel-Hadamard) —
but the test specified above is the **burden-product GLS**. Selection statistic and test statistic
were therefore not the same quantity. This is an error in the original protocol, not a property of
the data.

## What the component diagnosis showed (discovery side only, no hebau phenotype read)

| locus | omniB | minor-burden | PC1×PC1 | kernel | driver |
|-------|-------|--------------|---------|--------|--------|
| L1 | 6.09e-6 | 1.69e-2 | 1.60e-2 | 2.03e-6 | kernel |
| L2 | 2.10e-5 | 1.13e-5 | 1.85e-5 | 3.02e-3 | **burden** |
| L3 | 1.55e-4 | 0.974 | 0.129 | 5.16e-5 | kernel (burden null) |
| S1 | 1.37e-5 | 0.905 | 0.899 | 4.55e-6 | kernel (burden null) |
| published FibLen lead | 8.98e-6 | 5.98e-6 | 6.00e-6 | 4.10e-3 | **burden** |

Three of the four frozen loci carry no burden-product signal at all. This project has already
ruled on that situation: a rapeseed hit driven solely by the kernel component was declared
NOT_SUPPORTED and was not reported as a discovery. The same standard applies here, so L1, L3 and S1
were never burden-product candidates and must not enter a burden-product replication.

## Pre-test power (real hebau genotypes, real kinship, injected discovery effect)

| locus | full effect | half effect (winner's-curse haircut) |
|-------|-------------|--------------------------------------|
| L1 | 0.440 | 0.275 |
| L2 | 0.897 | 0.625 |
| L3 | 0.056 | 0.052 |
| S1 | 0.056 | 0.059 |

Mean power over the three originally frozen independent loci at the halved effect is 0.317, below
the 50% gate this protocol set in advance. The original three-locus design is therefore declared
**INCONCLUSIVE BY DESIGN** and is not run.

## Amended test

**One locus, L2 (Gh_A06G029400 | Gh_D06G029100).** Everything else in the protocol is unchanged:
same engine, same coding, `fiber_length_BLUE`, frozen direction (discovery beta = +0.1194, so the
replication must be positive), one-sided P < 0.05, single test so no multiplicity adjustment.
Pre-test power 0.897 at the discovery effect and 0.625 at half of it, which clears the 50% gate.

This is a **single-locus** replication. It cannot support any "multi-locus" claim, and the report
must say so. L1, L3, S1 and the previously examined FibElo lead are reported as excluded, with the
reason, so that the reader sees the full candidate set rather than the surviving member.

## What is NOT being done, deliberately

Switching the test statistic to omniB would make all four loci "live" again and would align the
test with the selection rule. It is not done, because omniB has no signed direction (the kernel
component is unsigned), the pre-registered directional criterion could not then be applied, and
choosing the statistic that rescues the most loci after seeing which loci are null is exactly the
circularity this protocol exists to prevent.
