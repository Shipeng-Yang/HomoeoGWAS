# Snakefile — U7_GWAS / HomoeoGWAS three-panel one-click pipeline (Phase 4 entry)
#
# Goal (charter §3.3 v1.0 release):
#   `snakemake paper_figures` reproduces wheat_watkins + cotton_hebau +
#   horvath2020 LOCO + M3.2 known QTL recovery + M3.3 DL prior + key figures.
#
# Driver:
#   PANELS_TRAITS define the wave each panel runs (subgenomes / sumstats /
#   fasta / chrom_map / known_qtl). Wheat reuses the v1 wheat-specific scripts;
#   cotton and horvath2020 use the panel-general M3.3 v2 + species M3.2.
#
# Conventions:
#   - all paths relative to repo root /mnt/7302share/fast_ysp/U7_GWAS
#   - results/phase3/m3_1_loco_v2/<panel>/<trait>/  for LOCO
#   - results/phase3/m3_2_qtl_v2/<panel>/<trait>/   for M3.2 (cotton/horvath panel-general)
#   - results/phase3/m3_3_dl_prior_v2/<panel>/<trait>/  for M3.3 (panel-general)
#   - wheat retains its v1 paths (results/phase3/m3_1_loco_v2/wheat_watkins
#     and results/phase3/m3_3_dl_prior/wheat_watkins)

from pathlib import Path

ROOT = Path(workflow.basedir)
PY_CPU = str(Path.home() / ".local/share/mamba/envs/polygwas-cpu/bin/python")
PY_GPU = str(Path.home() / "miniconda3/envs/polygwas-gpu/bin/python")

# --------------------------------------------------------------------------
# Panel registry — single source of truth for paths/conventions
# --------------------------------------------------------------------------

PANELS = {
    "horvath2020": {
        "subgenomes": "A,C",
        "bed_root": "data/processed/rapeseed/horvath",
        "fasta": "data/reference/rapeseed/GCF_000686985.2_Bra_napus_v2.0_genomic.fa",
        "chrom_map": "data/reference/rapeseed/chrom_map_horvath_to_ncbi.tsv",
        "known_qtl": "data/reference/rapeseed/known_qtl_rapeseed.tsv",
        "traits": ["bloom_50pct", "plant_height"],
    },
    "cotton_hebau": {
        "subgenomes": "A,D",
        "bed_root": "data/processed/cotton",
        "fasta": "/mnt/nvme/cotton_hbau/GCA_018997965.1_ASM1899796v1_genomic.fa",
        "chrom_map": "data/reference/cotton/chrom_map_hebau_to_ncbi.tsv",
        "known_qtl": "data/reference/cotton/known_qtl_cotton.tsv",
        "traits": ["fiber_length_BLUE", "lint_percentage_BLUE"],
    },
    # Wheat goes through the v1 (wheat-only) M3.3 pipeline; we expose it here
    # for completeness of the three-panel sweep but reuse its existing artifacts.
    "wheat_watkins": {
        "subgenomes": "A,B,D",
        "bed_root": "data/processed/wheat_bed",
        "fasta": "/mnt/nvme/wheat_ref/iwgsc_refseqv1.0_all_chromosomes/iwgsc_refseqv1.0_all_chromosomes.fa",
        "chrom_map": "",   # wheat: BIM chrom == fasta chrom (identity)
        "known_qtl": "data/reference/wheat/known_qtl_wheat.tsv",
        "traits": ["days_to_emerg"],
    },
}

PANEL_GENERAL_PANELS = ["horvath2020", "cotton_hebau"]  # use m3_3_v2 prepare
WHEAT_PANELS = ["wheat_watkins"]                         # use v1 prepare


# --------------------------------------------------------------------------
# Top-level targets
# --------------------------------------------------------------------------

def _all_m3_3_summaries():
    out = []
    for p in PANEL_GENERAL_PANELS:
        for t in PANELS[p]["traits"]:
            out.append(f"results/phase3/m3_3_dl_prior_v2/{p}/{t}/m3_3_summary_{t}.json")
    # wheat: existing v1 path, already shipped — list for transparency
    for p in WHEAT_PANELS:
        for t in PANELS[p]["traits"]:
            out.append(f"results/phase3/m3_3_dl_prior/{p}/m3_3_summary_{t}.json")
    return out


rule paper_figures:
    """Top-level reproduction target — three-panel M3.3 outputs."""
    input:
        _all_m3_3_summaries()
    output:
        touch("results/phase4/paper_figures.done")


rule three_panel_m3_3:
    """All M3.3 summaries for cotton + horvath."""
    input:
        [f"results/phase3/m3_3_dl_prior_v2/{p}/{t}/m3_3_summary_{t}.json"
         for p in PANEL_GENERAL_PANELS for t in PANELS[p]["traits"]]


# --------------------------------------------------------------------------
# Per-panel M3.3 pipeline (panel-general v2)
# --------------------------------------------------------------------------

rule m3_3_v2_prepare:
    """Build candidate SNP set (panel-general)."""
    input:
        sumstats="results/phase3/m3_1_loco_v2/{panel}/{trait}/sumstats_{trait}_loco.tsv",
        leads="results/phase3/m3_2_qtl_v2/{panel}/{trait}/new_locus_candidates.tsv",
        known=lambda w: PANELS[w.panel]["known_qtl"],
        fasta=lambda w: PANELS[w.panel]["fasta"],
        chrom_map=lambda w: PANELS[w.panel]["chrom_map"],
    output:
        cand="results/phase3/m3_3_dl_prior_v2/{panel}/{trait}/candidates.tsv.gz",
        summary="results/phase3/m3_3_dl_prior_v2/{panel}/{trait}/candidates_summary.json",
    params:
        subgenomes=lambda w: PANELS[w.panel]["subgenomes"],
        bed_root=lambda w: PANELS[w.panel]["bed_root"],
        out_dir="results/phase3/m3_3_dl_prior_v2/{panel}/{trait}",
        chrom_map_arg=lambda w: f"--chrom-map {PANELS[w.panel]['chrom_map']}" if PANELS[w.panel]["chrom_map"] else "",
    shell:
        """
        {PY_GPU} scripts/phase3/m3_3_v2_prepare_candidates.py \\
            --panel {wildcards.panel} --trait {wildcards.trait} \\
            --subgenomes "{params.subgenomes}" \\
            --sumstats {input.sumstats} \\
            --leads-tsv {input.leads} \\
            --known-qtl {input.known} \\
            --bed-root {params.bed_root} \\
            --fasta {input.fasta} \\
            {params.chrom_map_arg} \\
            --out-dir {params.out_dir} \\
            --ld-leads-tier ALL
        """


rule m3_3_score_plantcad:
    """PlantCaduceus DL prior scoring (single GPU)."""
    input:
        cand="results/phase3/m3_3_dl_prior_v2/{panel}/{trait}/candidates.tsv.gz",
        fasta=lambda w: PANELS[w.panel]["fasta"],
    output:
        "results/phase3/m3_3_dl_prior_v2/{panel}/{trait}/dl_scores_plantcad.tsv.gz",
        "results/phase3/m3_3_dl_prior_v2/{panel}/{trait}/dl_scores_plantcad_summary.json",
    params:
        out_dir="results/phase3/m3_3_dl_prior_v2/{panel}/{trait}",
    shell:
        """
        CUDA_VISIBLE_DEVICES=0 {PY_GPU} scripts/phase3/m3_3_score_dl_prior.py \\
            --model plantcad --candidates {input.cand} \\
            --fasta {input.fasta} --out-dir {params.out_dir} --device cuda:0
        """


rule m3_3_score_agront:
    """AgroNT DL prior scoring (single GPU)."""
    input:
        cand="results/phase3/m3_3_dl_prior_v2/{panel}/{trait}/candidates.tsv.gz",
        fasta=lambda w: PANELS[w.panel]["fasta"],
    output:
        "results/phase3/m3_3_dl_prior_v2/{panel}/{trait}/dl_scores_agront.tsv.gz",
        "results/phase3/m3_3_dl_prior_v2/{panel}/{trait}/dl_scores_agront_summary.json",
    params:
        out_dir="results/phase3/m3_3_dl_prior_v2/{panel}/{trait}",
    shell:
        """
        CUDA_VISIBLE_DEVICES=1 {PY_GPU} scripts/phase3/m3_3_score_dl_prior.py \\
            --model agront --candidates {input.cand} \\
            --fasta {input.fasta} --out-dir {params.out_dir} --device cuda:0
        """


rule m3_3_fuse:
    """Ensemble fusion + recall@N evaluation."""
    input:
        cand="results/phase3/m3_3_dl_prior_v2/{panel}/{trait}/candidates.tsv.gz",
        plant="results/phase3/m3_3_dl_prior_v2/{panel}/{trait}/dl_scores_plantcad.tsv.gz",
        agr="results/phase3/m3_3_dl_prior_v2/{panel}/{trait}/dl_scores_agront.tsv.gz",
        known=lambda w: PANELS[w.panel]["known_qtl"],
    output:
        summary="results/phase3/m3_3_dl_prior_v2/{panel}/{trait}/m3_3_summary_{trait}.json",
    params:
        out_dir="results/phase3/m3_3_dl_prior_v2/{panel}/{trait}",
    shell:
        """
        {PY_GPU} scripts/phase3/m3_3_fuse_and_evaluate.py \\
            --trait {wildcards.trait} \\
            --out-dir {params.out_dir} \\
            --candidates {input.cand} \\
            --known-qtl {input.known} \\
            --scores-plantcad {input.plant} \\
            --scores-agront {input.agr} \\
            --top-n 100
        """


# --------------------------------------------------------------------------
# Wheat M3.3 v1 (already-shipped; rule documents the existing artifacts)
# --------------------------------------------------------------------------

rule wheat_v1_m3_3_summary:
    """Marker rule pointing at the wheat-v1 M3.3 summary (shipped Phase 3.3)."""
    output:
        "results/phase3/m3_3_dl_prior/wheat_watkins/m3_3_summary_{trait}.json"
    shell:
        "ls -la {output} >&2 || (echo 'wheat v1 M3.3 missing — see docs/06' >&2; exit 1)"
