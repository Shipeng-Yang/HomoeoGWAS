#!/usr/bin/env python
"""H7 Part A — marker-density -> callable-homoeolog-pairs predictive design tool.

Polyploid genomes are huge, so WGS is expensive and many labs use cheap GBS. This module predicts,
BEFORE sequencing, how many homoeolog gene-PAIRS will be "callable" (both genes carry >= min_snp SNPs
in body/+-2kb flank, the minimum to form a burden) at a given marker density — the quantity that
drives this method's power and multiple-testing budget.

Approach: take a dense WGS base (wheat chr1, per-gene SNP counts up to 893), DOWN-SAMPLE (thin) SNPs to
mimic lower densities, and trace callable genes / callable pairs vs density. Two thinning models:
  - uniform (per-SNP Bernoulli retention) = OPTIMISTIC upper bound (spreads SNPs evenly);
  - clustered (whole-gene hit/miss at the same retained fraction) = PESSIMISTIC bound, closer to GBS
    where restriction-site SNPs cluster and miss whole genes.
Real GBS at a given total count lies between these. The deliverable is a PLOIDY-STRATIFIED design
guideline (species-agnostic): per-gene callability is ploidy-independent (from wheat thinning), and
ploidy enters via pairs-per-ortholog-group = C(n_subgenomes,2) (4n x1 / 6n x3 / 8n x6, via the 2-of-N
pairwise pivot) plus budget dilution (a fixed marker budget is spread over more gene copies at higher
ploidy). The callable-pair feasibility heuristic is checked for CONSISTENCY against the 4 real panels
(their exact callable-pair G vs observed discovery/null outcome) and the per-gene callable-fraction
curve is held-out validated (train/test gene split). A Poisson shortcut is reported only as an
optimistic upper bound (it overpredicts under real overdispersion).

Output: results/phase7/h7_marker_density.json"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path("/mnt/7302share/fast_ysp/U7_GWAS")
WDIR = ROOT / "results/phase7/interact_validate/wheat_chr1"
OUT = ROOT / "results/phase7/h7_marker_density.json"

FRACTIONS = [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01]
MIN_SNP_GRID = [3, 4, 5]
N_BOOT = 200

# feasibility classification on callable homoeolog-pair count G (from the 4-species empirical boundary)
BANDS = [(0, 70, "underpowered/null-likely"), (70, 300, "borderline/risky"),
         (300, 1000, "discovery-feasible"), (1000, np.inf, "strong-design")]


def _classify(G):
    for lo, hi, lab in BANDS:
        if lo <= G < hi:
            return lab
    return "strong-design"


def _gene_counts(sub):
    z = np.load(WDIR / f"snp_to_gene_{sub}.npz", allow_pickle=True)
    return {g: int(np.asarray(z["snp_idx"][i]).size) for i, g in enumerate(z["gene_ids"].tolist())}


def main():
    t0 = time.time()
    rng = np.random.default_rng(7)
    cB, cD = _gene_counts("B"), _gene_counts("D")
    triads = pd.read_csv(WDIR / "triads_chr1.tsv", sep="\t")
    # catalog = all chr1 triads with BOTH B and D genes annotated (regardless of SNP count)
    catalog = [(r.gene_B, r.gene_D) for r in triads.itertuples()
               if isinstance(r.gene_B, str) and isinstance(r.gene_D, str)]
    fullB = np.array([cB.get(gB, 0) for gB, _ in catalog])
    fullD = np.array([cD.get(gD, 0) for _, gD in catalog])
    n_pairs = len(catalog)
    mean_full = float(np.mean(np.concatenate([fullB[fullB > 0], fullD[fullD > 0]])))
    print(f"  catalog B-D pairs={n_pairs} mean in-gene SNPs/gene(full WGS)={mean_full:.1f} "
          f"({time.time()-t0:.0f}s)", flush=True)

    # ---- Part A: thinning curve (uniform + clustered) x min_snp x fractions, bootstrap CI
    curve = {}
    for ms in MIN_SNP_GRID:
        rows = []
        for f in FRACTIONS:
            Gu, Gc, Pg = [], [], []
            for _ in range(N_BOOT):
                # uniform per-SNP Bernoulli retention -> Binomial(full, f) retained SNPs per gene
                rB = rng.binomial(fullB, f) >= ms
                rD = rng.binomial(fullD, f) >= ms
                Gu.append(int(np.sum(rB & rD)))
                # per-GENE callable fraction (ploidy-agnostic; both subgenome gene pools pooled)
                Pg.append(float(np.mean(np.concatenate([rB, rD]))))
                # clustered: keep fraction f of genes wholesale (gene hit w.p. f), else 0
                hitB = rng.random(n_pairs) < f
                hitD = rng.random(n_pairs) < f
                Gc.append(int(np.sum(hitB & (fullB >= ms) & hitD & (fullD >= ms))))
            lam = f * mean_full
            rows.append(dict(
                fraction=f, lambda_gene=float(lam),
                P_gene_callable=float(np.mean(Pg)),       # ploidy-agnostic per-gene callability
                G_uniform=float(np.mean(Gu)), G_uniform_ci=[float(np.percentile(Gu, 2.5)),
                                                            float(np.percentile(Gu, 97.5))],
                G_clustered=float(np.mean(Gc)), G_clustered_ci=[float(np.percentile(Gc, 2.5)),
                                                                float(np.percentile(Gc, 97.5))],
                callable_frac_uniform=float(np.mean(Gu) / n_pairs),
                band_uniform=_classify(np.mean(Gu)), band_clustered=_classify(np.mean(Gc)),
                P_callable_poisson=float(stats.poisson.sf(ms - 1, lam))))
        curve[f"min_snp={ms}"] = rows
        print(f"  min_snp={ms} done ({time.time()-t0:.0f}s)", flush=True)

    # ---- transferable predictor. PRIMARY = empirical thinned-wheat callable-FRACTION vs lambda
    # (carries the real overdispersed per-gene SNP distribution); reported as a RANGE
    # [clustered=pessimistic, uniform=optimistic]. Poisson(mean lambda) is a SECONDARY optimistic
    # shortcut that OVERpredicts at high density (ignores overdispersion) -> upper bound only.
    base = curve["min_snp=3"]
    lam_grid = np.array([r["lambda_gene"] for r in base])[::-1]            # ascending
    pgene_grid = np.array([r["P_gene_callable"] for r in base])[::-1]      # ploidy-agnostic per-gene
    pgene_clu_grid = np.array([r["G_clustered"] / n_pairs for r in base])[::-1]  # clustered pair-frac

    # ploidy STRUCTURE: pairs_per_ortholog_group = C(n_subgenomes, 2) (the method tests pairwise,
    # so higher ploidy yields MORE candidate pairs per ortholog group via the 2-of-N pivot).
    PLOIDY = {
        "4n": dict(subgenomes=2, pairs_per_group=1, example="cotton AADD / rapeseed AACC"),
        "6n": dict(subgenomes=3, pairs_per_group=3, example="wheat AABBDD / oat AACCDD"),
        "8n": dict(subgenomes=4, pairs_per_group=6, example="strawberry (octoploid)"),
    }

    def predict_G(lambda_gene, n_ortholog_groups, ploidy):
        """Predict callable homoeolog-PAIR count at a per-gene density ``lambda_gene`` for a given
        ploidy. P_gene from the wheat thinning curve (per-gene callability is ploidy-agnostic);
        a pair is callable iff both its genes are -> P_pair ~ P_gene^2; ploidy enters via
        pairs_per_group. Optimistic (uniform thinning) + pessimistic (clustered) reported as a
        range. ``lambda_gene = total_in_gene_SNPs / (n_ortholog_groups * n_subgenomes)`` for a
        fixed marker budget -> higher ploidy dilutes lambda (more gene copies)."""
        ppg = PLOIDY[ploidy]["pairs_per_group"]
        P_gene = float(np.interp(lambda_gene, lam_grid, pgene_grid))
        P_pair_clu = float(np.interp(lambda_gene, lam_grid, pgene_clu_grid))
        G_uni = n_ortholog_groups * ppg * P_gene ** 2
        G_clu = n_ortholog_groups * ppg * P_pair_clu
        return dict(ploidy=ploidy, lambda_gene=float(lambda_gene), pairs_per_group=ppg,
                    P_gene_callable=P_gene, G_pred_range=[float(G_clu), float(G_uni)],
                    band_optimistic=_classify(G_uni), band_pessimistic=_classify(G_clu))

    # Poisson-vs-empirical diagnostic on wheat (shows Poisson OVERpredicts at high density)
    poisson_check = []
    for r in base:
        G_poisson = n_pairs * r["P_callable_poisson"] ** 2
        poisson_check.append(dict(lambda_gene=r["lambda_gene"], G_empirical_uniform=r["G_uniform"],
                                  G_empirical_clustered=r["G_clustered"], G_poisson=float(G_poisson),
                                  poisson_overpredicts=bool(G_poisson > r["G_uniform"] * 1.1)))

    # ---- HELD-OUT calibration: fit the callable-fraction(lambda) curve on a TRAIN gene-pair subset,
    # predict callable-pair count on a DISJOINT TEST subset, report calibration error. Tests whether
    # the fraction-curve generalizes across gene sets (a proxy for cross-species transfer) rather than
    # being a circular re-fit of the same pairs.
    perm = rng.permutation(n_pairs)
    tr, te = perm[: n_pairs // 2], perm[n_pairs // 2:]
    heldout = {}
    for ms in MIN_SNP_GRID:
        rows = []
        for f in FRACTIONS:
            errs_u = []
            for _ in range(N_BOOT):
                rB = rng.binomial(fullB, f) >= ms
                rD = rng.binomial(fullD, f) >= ms
                cal = rB & rD
                frac_tr = float(np.mean(cal[tr]))            # callable fraction fit on TRAIN
                G_pred_te = frac_tr * te.size                # predict TEST callable-pair count
                G_act_te = float(np.sum(cal[te]))            # actual TEST callable-pair count
                if G_act_te >= 5:                            # only score where the test count is meaningful
                    errs_u.append(abs(G_pred_te - G_act_te) / G_act_te)
            rows.append(dict(fraction=f, lambda_gene=float(f * mean_full),
                             median_rel_error=(float(np.median(errs_u)) if errs_u else None),
                             n_scored=len(errs_u)))
        heldout[f"min_snp={ms}"] = rows
    # overall calibration error (median over scored fractions, min_snp=3)
    he3 = [r["median_rel_error"] for r in heldout["min_snp=3"] if r["median_rel_error"] is not None]
    heldout_median_rel_error = float(np.median(he3)) if he3 else None
    print(f"  held-out calibration median rel-error (min_snp=3) = {heldout_median_rel_error} "
          f"({time.time()-t0:.0f}s)", flush=True)

    # ---- 4-species anchor validation: exact callable-pair G vs observed outcome
    anchors = [
        dict(species="wheat AABBDD", ploidy="6n", tech="WGS", genome_gb=16.0, snps="86.5M",
             G_callable=1759, scope="chr1 flank B-D", outcome="discovery (2 hits)"),
        dict(species="cotton AADD", ploidy="4n", tech="SNP-array", genome_gb=2.3, snps="498k",
             G_callable=372, scope="flank same-subgenome", outcome="discovery (2 hits)"),
        dict(species="oat AACCDD", ploidy="6n", tech="sparse GBS", genome_gb=11.0,
             snps="sparse(~4-6% gene cov)", G_callable=69, scope="flank pairs (AC+AD+CD)",
             outcome="null (underpowered)"),
        dict(species="rapeseed AACC", ploidy="4n", tech="GBS+imputation", genome_gb=1.1, snps="291k",
             G_callable=14, scope="flank same-subgenome", outcome="null (underpowered)"),
    ]
    for a in anchors:
        a["predicted_band"] = _classify(a["G_callable"])
        a["consistent_with_outcome"] = bool(
            ("discovery" in a["outcome"]) == (a["G_callable"] >= 300))
    anchors_all_consistent = all(a["consistent_with_outcome"] for a in anchors)
    anchor_framing = ("HEURISTIC consistency check, NOT validation of a universal threshold: 4 systems "
                      "differing in technology/species/genome/architecture/sample-size, with the cutoff "
                      "read off after seeing outcomes. Honest reading: a practical transition lies "
                      "around tens-vs-hundreds of callable pairs (G<70 high-risk; G>300 has precedent "
                      "for discovery); the exact cutoffs are heuristic, not calibrated.")

    # ---- PLOIDY-STRATIFIED design guideline (the generalizable paper deliverable, species-agnostic).
    # (A) STRUCTURAL view: callable pairs per 1000 ortholog groups vs per-gene density lambda, by
    #     ploidy (isolates the pairs-per-group multiplier: 4n x1, 6n x3, 8n x6).
    ploidy_curve = {}
    for name in PLOIDY:
        ploidy_curve[name] = [
            {**predict_G(r["lambda_gene"], 1000, name), "fraction": r["fraction"]}
            for r in base]
    # (B) BUDGET view: fixed in-gene SNP budget over a fixed ortholog-group count; lambda is DILUTED
    #     by the number of subgenomes (more gene copies) -> shows the competing ploidy effects
    #     (more pairs per group vs more gene regions to cover). Generic scenarios, no specific crop.
    N_GROUPS = 20000                          # generic ortholog-group count (per-subgenome gene set)
    budget_view = []
    for in_gene_snps in (50_000, 150_000, 500_000, 1_500_000):
        for name, st in PLOIDY.items():
            n_gene_regions = N_GROUPS * st["subgenomes"]      # higher ploidy = more gene copies
            lam = in_gene_snps / n_gene_regions
            budget_view.append(dict(in_gene_snps=in_gene_snps, n_ortholog_groups=N_GROUPS,
                                    n_gene_regions=n_gene_regions, **predict_G(lam, N_GROUPS, name)))

    out = dict(
        tool="homoeogwas", analysis="h7_marker_density_design",
        base="wheat_chr1_WGS (per-gene SNP counts, thinned)", n_pairs_catalog=n_pairs,
        mean_in_gene_snps_per_gene_full=mean_full, fractions=FRACTIONS, min_snp_grid=MIN_SNP_GRID,
        n_boot=N_BOOT,
        feasibility_bands=[dict(min_G=lo, max_G=(None if hi == np.inf else hi), label=lab)
                           for lo, hi, lab in BANDS],
        thinning_scenarios_note=("uniform and clustered are two explicit THINNING SCENARIOS, not "
                                 "formal bounds: real GBS can fall BELOW clustered (homoeolog-biased "
                                 "restriction-site/mappability loss) and capture/genic arrays can "
                                 "exceed uniform. Use the pair as an optimistic/pessimistic envelope."),
        thinning_curve=curve,
        heldout_calibration=heldout, heldout_median_rel_error_min_snp3=heldout_median_rel_error,
        poisson_vs_empirical_wheat=poisson_check,
        prediction_formula=("lambda_gene = total_SNPs * fraction_in_gene_regions / n_gene_regions. "
                            "PRIMARY predictor = empirical thinned-wheat callable-FRACTION(lambda) "
                            "interpolated -> G_pred RANGE [clustered/pessimistic, uniform/optimistic] "
                            "= n_homoeolog_pairs * callable_fraction(lambda). SECONDARY (optimistic "
                            "upper bound only) = Poisson: P(callable)=P(Pois(lambda)>=min_snp), "
                            "G=n_pairs*P^2 -- this OVERpredicts at high density (real per-gene SNP "
                            "counts are overdispersed), so design to the pessimistic/clustered end."),
        species_anchors=anchors, anchors_all_consistent=anchors_all_consistent,
        anchor_framing=anchor_framing,
        ploidy_structure=PLOIDY,
        ploidy_design_guideline=dict(
            label=("PLOIDY-STRATIFIED design guideline (species-agnostic). Per-gene callability is "
                   "ploidy-independent (from wheat thinning); ploidy enters via pairs-per-ortholog-"
                   "group = C(n_subgenomes,2) (4n x1, 6n x3, 8n x6, using the 2-of-N pairwise pivot). "
                   "Two competing ploidy effects: higher ploidy gives MORE candidate pairs per group "
                   "(+) but, for a fixed usable in-gene SNP budget, spreads markers over MORE gene "
                   "copies so per-copy lambda is DILUTED by the subgenome count (-)."),
            estimand_and_assumptions=[
                "G_pred counts callable PAIRWISE TESTS (homoeolog-pair opportunities), NOT independent "
                "ortholog groups -- do not read it as independent biological evidence",
                "C(n,2)*P_gene^2 is an EXPECTATION; the C(n,2) pairwise indicators within an ortholog "
                "group SHARE genes (A-B and A-D share A) so they are POSITIVELY CORRELATED -> this "
                "affects variance/CI/effective-independent-evidence, NOT the mean pair count; cluster "
                "uncertainty by ortholog group",
                "lambda = USABLE, in-gene, COPY-ASSIGNED SNP density per gene copy (i.e. AFTER mapping, "
                "multi-mapping removal, homoeolog/subgenome assignment, missingness) -- NOT raw "
                "genome-wide SNP count and NOT 'same sequencing effort'; higher ploidy makes obtaining "
                "usable copy-assigned in-gene SNPs HARDER (a data-quality cost outside this model)",
                "the 8n (octoploid) row is STRUCTURAL EXTRAPOLATION -- no empirical anchor here "
                "(strawberry is parked); 4n/6n are anchored by cotton/rapeseed and wheat/oat",
                "ploidy does NOT change per-gene callability; it enters only via the pairs-per-group "
                "multiplier and the per-copy budget dilution"],
            structural_view_pairs_per_1000_groups=ploidy_curve,
            budget_view_fixed_in_gene_snps=budget_view,
            headline=("Within a pairwise marker-density model, higher ploidy does NOT necessarily "
                      "reduce expected callable-pair YIELD: the C(n,2) increase in candidate homoeolog "
                      "pairs can offset per-copy marker dilution. The MODELED binding constraint is "
                      "usable per-gene copy-assigned marker density (lambda), not ploidy alone. This is "
                      "a yield/feasibility statement only -- higher ploidy can still be harder via read "
                      "mapping, dosage ambiguity, homoeolog collapse and SNP assignment, which are "
                      "outside this density model.")),
        recommendation=("Binding constraint = SNPs landing IN genes (lambda per gene region), which "
                        "drops as genome/ploidy grows for a fixed budget. Recommendation hierarchy: "
                        "(1) exome/target-capture -> directly targets gene regions, best cost/benefit "
                        "as genome size grows; (2) gene-targeted SNP array if a variant catalog exists "
                        "(cotton proves a 498k 4n array, NOT WGS, suffices: G=372 -> discovery); "
                        "(3) moderate-coverage WGS; (4) high-density GBS only if a pilot shows enough "
                        "gene-region SNPs; (5) GBS+imputation helps but is risky in polyploids "
                        "(homoeolog/subgenome confusion). Bare GBS on a large (>5 Gb) genome is "
                        "high-risk (oat 6n ~11 Gb GBS gave only G=69 -> null), independent of ploidy."),
        caveats=["uniform and clustered are two THINNING SCENARIOS (optimistic/pessimistic envelope), "
                 "not formal bounds; real GBS can fall below clustered, capture can exceed uniform",
                 "wheat chr1-only thinning base; per-gene SNP heterogeneity assumed comparable across "
                 "species/ploidy on transfer (the 8n=octoploid row has NO empirical anchor here -- "
                 "strawberry is parked -- so it is structural extrapolation only)",
                 "ploidy enters ONLY through the pairs-per-group multiplier + budget dilution; it does "
                 "NOT change per-gene callability; higher ploidy also makes homoeolog/subgenome marker "
                 "assignment harder (a data-quality cost not modeled here)",
                 "Poisson rule overpredicts (overdispersion) -> demoted to optimistic upper bound",
                 "homoeolog-pair catalog QUALITY (synteny/ortholog annotation) can dominate G in a "
                 "non-model species; marker density alone is necessary not sufficient",
                 "G drives BOTH power and the alpha/G multiple-testing burden (Part B power sim, separate)"],
        runtime_sec=round(time.time() - t0, 1))
    OUT.write_text(json.dumps(out, indent=2, default=float))
    print(f"✅ wrote {OUT}  (anchors_all_consistent={anchors_all_consistent}; "
          f"held-out rel-error={heldout_median_rel_error})  ({out['runtime_sec']}s)")


if __name__ == "__main__":
    main()
