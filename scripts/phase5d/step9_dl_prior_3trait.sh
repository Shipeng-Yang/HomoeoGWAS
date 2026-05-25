#!/bin/bash
# scripts/phase5d/step9_dl_prior_3trait.sh
# Step 9 — M3.3 v1.5 DL prior (PlantCaduceus + AgroNT) across 3 oat traits.
# Per trait: prepare candidates → plantcad score (GPU0) → agront score (GPU1) → fuse + evaluate.
# Codex review #2 Q5: framing = oat internal robustness across env traits,
# NOT NC main figure (main figure stays "三 panel × 一 trait" wheat/cotton/horvath).

set -uo pipefail
PROJ=/mnt/7302share/fast_ysp/U7_GWAS
GPU_PY=/home/yys05/.miniconda3/envs/polygwas-gpu/bin/python
[[ ! -x "$GPU_PY" ]] && GPU_PY=/home/yys05/miniconda3/envs/polygwas-gpu/bin/python
CPU_PY=/home/yys05/.local/share/mamba/envs/polygwas-cpu/bin/python
export PATH=/home/yys05/.local/share/mamba/envs/polygwas-cpu/bin:$PATH

LOG=$PROJ/scripts/phase5d/step9_dl_prior_3trait.log
exec > >(tee -a "$LOG") 2>&1
echo "=== Step 9 DL prior 3 trait started $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo "GPU_PY: $GPU_PY"
echo "CPU_PY: $CPU_PY"

cd $PROJ

FASTA=/mnt/nvme/oat_raw/ensembl_ot3098/oat_ot3098_v2.fa
CHROM_MAP=data/reference/oat/chrom_map_oat_logical.tsv
KNOWN_QTL=data/reference/oat/known_qtl_oat.tsv
BED_ROOT=data/processed/avena_sativa_logical

for TRAIT in BIO6 SOC BIO12; do
    echo ""
    echo "=========================================="
    echo "=== TRAIT: $TRAIT $(date -u +%H:%M:%S) ==="
    echo "=========================================="
    OUT=$PROJ/results/phase5d/m3_3_dl_prior/oat_old_rahman2025/$TRAIT
    mkdir -p "$OUT"

    SUMSTATS=results/phase5d/m3_1_loco_v2/oat_old_rahman2025/$TRAIT/sumstats_${TRAIT}_loco.tsv
    LEADS=results/phase5d/m3_2_qtl/oat_old_rahman2025/$TRAIT/new_locus_candidates.tsv

    # 1) prepare candidates (CPU)
    echo "--- [1/4] prepare candidates ---"
    $CPU_PY scripts/phase3/m3_3_v2_prepare_candidates.py \
        --panel oat_old_rahman2025 --trait $TRAIT --subgenomes A,C,D \
        --sumstats $SUMSTATS --leads-tsv $LEADS \
        --known-qtl $KNOWN_QTL \
        --bed-root $BED_ROOT --fasta $FASTA --chrom-map $CHROM_MAP \
        --out-dir $OUT

    if [[ ! -s "$OUT/candidates.tsv.gz" ]]; then
        echo "  ERROR: candidates.tsv.gz missing — skip $TRAIT"
        continue
    fi
    N_CAND=$(zcat $OUT/candidates.tsv.gz | tail -n +2 | wc -l)
    echo "  $N_CAND candidates"

    # 2) PlantCaduceus GPU0
    echo "--- [2/4] PlantCaduceus GPU0 ---"
    CUDA_VISIBLE_DEVICES=0 $GPU_PY scripts/phase3/m3_3_score_dl_prior.py \
        --model plantcad --candidates $OUT/candidates.tsv.gz \
        --fasta $FASTA --out-dir $OUT/plantcad --device cuda:0 &
    PC_PID=$!

    # 3) AgroNT GPU1 in parallel
    echo "--- [3/4] AgroNT GPU1 (parallel) ---"
    CUDA_VISIBLE_DEVICES=1 $GPU_PY scripts/phase3/m3_3_score_dl_prior.py \
        --model agront --candidates $OUT/candidates.tsv.gz \
        --fasta $FASTA --out-dir $OUT/agront --device cuda:0 &
    AG_PID=$!

    wait $PC_PID && echo "  plantcad DONE" || echo "  plantcad FAIL"
    wait $AG_PID && echo "  agront DONE" || echo "  agront FAIL"

    # 4) Fuse + evaluate (CPU)
    echo "--- [4/4] fuse + evaluate ---"
    PC_SCORES=$(ls $OUT/plantcad/scores*.tsv 2>/dev/null | head -1)
    AG_SCORES=$(ls $OUT/agront/scores*.tsv 2>/dev/null | head -1)
    if [[ -z "$PC_SCORES" || -z "$AG_SCORES" ]]; then
        echo "  ERROR: scores missing — pc=$PC_SCORES ag=$AG_SCORES"
        continue
    fi
    $CPU_PY scripts/phase3/m3_3_fuse_and_evaluate.py \
        --trait $TRAIT --out-dir $OUT \
        --candidates $OUT/candidates.tsv.gz \
        --known-qtl $KNOWN_QTL \
        --scores-plantcad $PC_SCORES --scores-agront $AG_SCORES
    echo "=== TRAIT $TRAIT DONE $(date -u +%H:%M:%S) ==="
done

echo "=== Step 9 FINISHED $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
