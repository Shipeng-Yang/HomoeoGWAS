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
