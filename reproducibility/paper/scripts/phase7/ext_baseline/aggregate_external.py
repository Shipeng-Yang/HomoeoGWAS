"""Aggregate external published-tool recall of the two cotton homoeolog-pair hits.

Collects per-tool results (GWASpoly single-marker, SKAT/STAAR/regenie single-gene burden,
PLINK SNP x SNP epistasis), computes each tool's BEST evidence in the hit region, and both a
CANDIDATE-region threshold (primary: recall of 2 already-discovered hits) and a GLOBAL/genome-wide
threshold (supplement). Compares to homoeogwas. No circular hardcoding: every verdict is best-P vs a
prespecified, method-native multiple-testing threshold.

Output: results/phase7/ext_baseline/external_baselines.json + external_baselines_table.tsv
"""
import csv
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path("/mnt/7302share/fast_ysp/U7_GWAS")
EB = ROOT / "results/phase7/ext_baseline"

# ours (the homoeogwas discoveries) + hit geometry, from the committed realdata benchmark
OURS = {
    "Hit1_fiber_length": dict(trait="fiber_length_BLUE", ours_p=2.1359e-4,
                              genes=["GhM_A06G1605", "GhM_D06G1557"], n_cross=42),
    "Hit2_length_uniformity": dict(trait="length_uniformity_BLUE", ours_p=4.7470e-5,
                                   genes=["GhM_A11G2420", "GhM_D11G2742"], n_cross=48),
}
# genome-wide gene count per subgenome for the burden global threshold (callable body+flank gene maps)
N_GENES_GW = 3983  # snp_to_gene_body gene_ids count (order of magnitude for gene-burden multiplicity)


def manifest():
    rows = list(csv.DictReader(open(EB / "candidate_manifest.tsv"), delimiter="\t"))
    by_hit_gene = defaultdict(list)
    for r in rows:
        by_hit_gene[(r["hit"], r["gene"])].append(r)
    return rows, by_hit_gene


def read_tsv(p):
    return list(csv.DictReader(open(p), delimiter="\t")) if p.exists() else []


def epistasis_best(hit, man_rows):
    """Best A-gene x D-gene cross-pair interaction P (filtered to the homoeolog pair)."""
    role = {(r["hit"], r["snp_id"]): r["gene_role"] for r in man_rows}
    best = None
    n = 0
    f = EB / f"epistasis/{hit}.epi.qt"
    if not f.exists():
        return None, 0
    with open(f) as fh:
        fh.readline()
        for line in fh:
            c = line.split()
            if len(c) < 7 or c[6] == "nan":
                continue
            r1, r2 = role.get((hit, c[1])), role.get((hit, c[3]))
            if r1 and r2 and r1 != r2:
                p = float(c[6])
                n += 1
                if best is None or p < best:
                    best = p
    return best, n


def regenie_assoc_best(sub, trait, gene_snps):
    """Best single-variant regenie P among a gene's SNPs (LOG10P column)."""
    f = EB / f"regenie/{sub}/{sub}_assoc_{trait}.regenie"
    if not f.exists():
        return None
    best = None
    want = set(gene_snps)
    with open(f) as fh:
        hdr = None
        for line in fh:
            if line.startswith("#"):
                continue
            c = line.split()
            if hdr is None:
                hdr = c
                idx_id = hdr.index("ID")
                idx_lp = hdr.index("LOG10P")
                continue
            if c[idx_id] in want:
                lp = c[idx_lp]
                if lp not in ("NA", "nan"):
                    p = 10 ** (-float(lp))
                    if best is None or p < best:
                        best = p
    return best


def regenie_burden_best(sub, trait, gene):
    """regenie gene-burden ACATO omnibus P for a gene (one omnibus test/gene -> clean multiplicity).

    ACATO internally combines the burden/SKAT/SKATO masks, so we extract exactly that single
    per-gene omnibus P (codex-check: avoids min-selecting over ~6 mask/test variants).
    """
    f = EB / f"regenie/{sub}/{sub}_burden_{trait}.regenie"
    if not f.exists():
        return None
    best = None
    with open(f) as fh:
        hdr = None
        for line in fh:
            if line.startswith("#"):
                continue
            c = line.split()
            if hdr is None:
                hdr = c
                idx_id, idx_lp, idx_test = hdr.index("ID"), hdr.index("LOG10P"), hdr.index("TEST")
                continue
            if gene in c[idx_id] and c[idx_test] == "ADD-ACATO":
                lp = c[idx_lp]
                if lp not in ("NA", "nan"):
                    p = 10 ** (-float(lp))
                    if best is None or p < best:
                        best = p
    return best


def verdict(best_p, cand_thr, glob_thr):
    return dict(best_p=None if best_p is None else float(f"{best_p:.4g}"),
                candidate_threshold=float(f"{cand_thr:.4g}"),
                candidate_pass=bool(best_p is not None and best_p < cand_thr),
                global_threshold=float(f"{glob_thr:.2g}"),
                global_pass=bool(best_p is not None and best_p < glob_thr))


def main():
    man_rows, by_hit_gene = manifest()
    gp = {(r["subgenome"], r["gene"]): r for r in read_tsv(EB / "gwaspoly_A.tsv") + read_tsv(EB / "gwaspoly_D.tsv")}
    skat = {(r["subgenome"], r["gene"]): r for r in read_tsv(EB / "skat_A.tsv") + read_tsv(EB / "skat_D.tsv")}
    staar = {(r["subgenome"], r["gene"]): r for r in read_tsv(EB / "staar_A.tsv") + read_tsv(EB / "staar_D.tsv")}

    records = []
    for hit, info in OURS.items():
        trait = info["trait"]
        genes = info["genes"]
        # n candidate SNPs across the hit's 2 genes (single-marker candidate denominator)
        n_cand_snp = sum(len(by_hit_gene[(hit, g)]) for g in genes)
        gene_sub = {g: by_hit_gene[(hit, g)][0]["subgenome"] for g in genes}
        gene_snps = {g: [r["snp_id"] for r in by_hit_gene[(hit, g)]] for g in genes}
        n_mk_sub = {sub: int(rows[0]["n_markers_subgenome"])
                    for sub in set(gene_sub.values())
                    for rows in [read_tsv(EB / f"gwaspoly_{sub}.tsv")] if rows}

        # --- GWASpoly single-marker: best across both genes ---
        gp_ps = []
        for g in genes:
            r = gp.get((gene_sub[g], g))
            if r and r["best_p"] not in ("NA", ""):
                gp_ps.append(float(r["best_p"]))
        gp_best = min(gp_ps) if gp_ps else None
        gw_thr = 0.05 / max(n_mk_sub.values()) if n_mk_sub else 1e-7
        records.append(dict(tool="GWASpoly", estimand="single-marker additive (polyploid GLM)",
                            hit=hit, trait=trait, n_tests_candidate=n_cand_snp,
                            **verdict(gp_best, 0.05 / n_cand_snp, gw_thr)))

        # --- regenie single-variant ---
        rg_ps = [p for g in genes
                 for p in [regenie_assoc_best(gene_sub[g], trait, gene_snps[g])] if p is not None]
        rg_best = min(rg_ps) if rg_ps else None
        records.append(dict(tool="regenie (single-variant)", estimand="single-marker marginal GWAS",
                            hit=hit, trait=trait, n_tests_candidate=n_cand_snp,
                            **verdict(rg_best, 0.05 / n_cand_snp, gw_thr)))

        # --- single-gene burden tools (SKAT / STAAR / regenie-burden): best across both genes ---
        for tool, tab, pcol in [("SKAT", skat, "p_skato"), ("STAAR", staar, "p_STAAR_O")]:
            ps = [float(tab[(gene_sub[g], g)][pcol]) for g in genes
                  if (gene_sub[g], g) in tab and tab[(gene_sub[g], g)][pcol] not in ("NA", "")]
            best = min(ps) if ps else None
            records.append(dict(tool=tool, estimand="single-gene burden/SKAT/SKAT-O",
                                hit=hit, trait=trait, n_tests_candidate=len(genes),
                                **verdict(best, 0.05 / len(genes), 0.05 / N_GENES_GW)))
        rgb = [p for g in genes for p in [regenie_burden_best(gene_sub[g], trait, g)] if p is not None]
        rgb_best = min(rgb) if rgb else None
        records.append(dict(tool="regenie (gene-burden)", estimand="single-gene ACATO omnibus (burden/SKAT/SKATO)",
                            hit=hit, trait=trait, n_tests_candidate=len(genes),
                            **verdict(rgb_best, 0.05 / len(genes), 0.05 / N_GENES_GW)))

        # --- PLINK SNPxSNP epistasis: best A-gene x D-gene cross pair ---
        ep_best, ep_n = epistasis_best(hit, man_rows)
        n_gw_pairs = 1.2e11  # ~ (498k choose 2) order of magnitude for genome-wide SNPxSNP
        records.append(dict(tool="PLINK epistasis", estimand="SNP×SNP interaction (within homoeolog pair)",
                            hit=hit, trait=trait, n_tests_candidate=ep_n,
                            **verdict(ep_best, 0.05 / max(ep_n, 1), 0.05 / n_gw_pairs)))

    payload = dict(
        analysis="external_baseline_recall", date="2026-06-04",
        n_samples=419, species="cotton_AADD_hebau",
        ours=OURS,
        excluded_tools={
            "SAIGE": "single-variant + set burden estimand already covered by regenie; heavier C++ setup, no pair-interaction mode",
            "NodeGWAS": "requires graph-pangenome node inputs (reads + GFA); not the data type of this SNP-array panel",
            "networkGWAS": "requires a curated cotton PPI network + fastlmm; changes the tested data basis",
            "deeprvat": "GPU/days-scale deep rare-variant burden; disproportionate for a focused 2-hit recall test",
        },
        thresholds_note=("PRIMARY = candidate-region threshold (recall of two already-discovered hits: "
                         "Bonferroni over the method-native candidate tests — SNP count for single-marker, "
                         "gene count for single-gene omnibus tests SKAT-O/STAAR-O/ACATO, SNP-pair count for "
                         "epistasis). GLOBAL = genome-wide threshold (supplement). The ~10^11 genome-wide "
                         "SNP-pair count is a NOMINAL Bonferroni / tested-hypothesis-space upper bound "
                         "(conservative: LD reduces effective tests); homoeogwas tests only ~10^2-10^3 "
                         "synteny-guided gene-pairs, the ~8-order-of-magnitude search-space reduction."),
        framing_redlines=[
            "These tools were NOT detected under the prespecified thresholds for the two homoeolog-pair "
            "signals under their native marginal / single-gene / SNP×SNP estimands — NOT 'the tools are "
            "inferior', NOT 'no interaction exists', NOT 'homoeogwas is universally better'.",
            "STAAR/SKAT/regenie-burden are real published rare-variant machinery used here as single-GENE "
            "aggregation baselines on COMMON SNP-array variants (not their ideal variant spectrum); keep "
            "this caveat prominent.",
            "PLINK epistasis RECOVERED both frozen candidate regions under candidate-region multiplicity, "
            "but the same signals would NOT survive a genome-wide SNP-pair scan. State it this way (the "
            "interaction is real and local; unguided SNP-pair epistasis pays too large a multiplicity "
            "penalty) — a resolution/multiplicity result, not estimand-absence.",
            "homoeogwas 'genome-wide significant' = significant under its own transparent candidate-pair "
            "Bonferroni denominator (~10^2-10^3 synteny-guided pairs), reported explicitly; not an implicit "
            "claim.",
        ],
        caveats=[
            "Demonstration on two representative pair-only discoveries, NOT a comprehensive sensitivity "
            "ranking of the tools.",
            "External tools were given their BEST regional P (best SNP / gene / SNP-pair); homoeogwas's "
            "hit regions were frozen from the discovery snp->gene map (body for Hit1, flank2000bp for "
            "Hit2) BEFORE this comparison — no post-hoc region definition.",
            "Single-marker GWASpoly/regenie were run per-subgenome, native to homoeolog-specific SNP "
            "assignment (A and D variants are distinct loci); a pooled run is a possible sensitivity check.",
            "regenie step1 used an LD-pruned polygenic null (standard practice); step2 tested the full "
            "marker set / gene masks.",
        ],
        results=records,
    )
    (EB / "external_baselines.json").write_text(json.dumps(payload, indent=2, default=float))

    # tidy table
    cols = ["tool", "estimand", "hit", "trait", "n_tests_candidate", "best_p",
            "candidate_threshold", "candidate_pass", "global_threshold", "global_pass"]
    with open(EB / "external_baselines_table.tsv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        w.writerows(records)

    # console summary
    print(f"{'tool':24} {'hit':24} {'best_p':>10} {'cand_thr':>9} {'cand':>5} {'glob':>5}  ours_p")
    for r in records:
        o = OURS[r["hit"]]["ours_p"]
        print(f"{r['tool']:24} {r['hit']:24} {str(r['best_p']):>10} {r['candidate_threshold']:>9.2g} "
              f"{'PASS' if r['candidate_pass'] else 'miss':>5} {'PASS' if r['global_pass'] else 'miss':>5}  {o:.2g}")
    print(f"\nwrote {EB/'external_baselines.json'} + table")


if __name__ == "__main__":
    main()
