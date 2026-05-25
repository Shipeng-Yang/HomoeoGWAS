#!/usr/bin/env bash
# M3.3 DL prior for the RiceVarMap2 diploid panel (4 traits). Tests whether the
# PlantCaduceus + AgroNT recall@100 lift generalises to a diploid crop.
#
# Dense WGS panel (4.7M SNP): use p<1e-6 suggestive + cloned-gene sentinels with
# LD expansion restricted to ±10kb/r²≥0.9, giving ~10^3 candidates comparable to
# the array/GBS panels (the default ±1Mb/r²≥0.2 inflates dense-WGS sets ~90x).
set -uo pipefail
cd /mnt/7302share/fast_ysp/U7_GWAS
PY=~/miniconda3/envs/polygwas-gpu/bin/python
FASTA=/mnt/nvme/rice_ref/Oryza_sativa.IRGSP-1.0.dna.toplevel.fa
MAP=data/reference/rice/chrom_map_rice.tsv
BED=data/processed/rice

for trait in Heading_date Grain_length Grain_width Plant_height; do
  OUT=results/phase5e/m3_3_dl_prior/rice_ricevarmap2/$trait
  KNOWN=data/reference/rice/known_qtl_rice_${trait}.tsv
  SUM=results/phase5e/rice_ricevarmap2/$trait/sumstats_${trait}_loco_ALL.tsv.gz
  LEADS=results/phase5e/m3_2_qtl/rice_ricevarmap2/$trait/new_locus_candidates.tsv
  mkdir -p "$OUT"
  echo "================= M3.3 $trait ================="
  echo "[prepare]"
  $PY scripts/phase3/m3_3_v2_prepare_candidates.py \
    --panel rice_ricevarmap2 --trait "$trait" --subgenomes "ALL" \
    --sumstats "$SUM" --leads-tsv "$LEADS" --known-qtl "$KNOWN" \
    --bed-root "$BED" --fasta "$FASTA" --chrom-map "$MAP" \
    --out-dir "$OUT" --ld-leads-tier ALL \
    --p-max 1e-6 --ld-window-kb 10 --ld-min-r2 0.9 \
    2>&1 | grep -iE 'REF match|breakdown|merged unique' || true

  echo "[score] plantcad(GPU0) + agront(GPU1)"
  CUDA_VISIBLE_DEVICES=0 $PY scripts/phase3/m3_3_score_dl_prior.py \
    --model plantcad --candidates "$OUT/candidates.tsv.gz" \
    --fasta "$FASTA" --out-dir "$OUT" --device cuda:0 > "$OUT/score_plantcad.log" 2>&1 &
  PC=$!
  CUDA_VISIBLE_DEVICES=1 $PY scripts/phase3/m3_3_score_dl_prior.py \
    --model agront --candidates "$OUT/candidates.tsv.gz" \
    --fasta "$FASTA" --out-dir "$OUT" --device cuda:0 > "$OUT/score_agront.log" 2>&1 &
  AG=$!
  wait $PC; PCR=$?; wait $AG; AGR=$?
  echo "  plantcad rc=$PCR agront rc=$AGR"
  [ $PCR -ne 0 ] && tail -4 "$OUT/score_plantcad.log"
  [ $AGR -ne 0 ] && tail -4 "$OUT/score_agront.log"

  echo "[fuse]"
  $PY scripts/phase3/m3_3_fuse_and_evaluate.py \
    --trait "$trait" --out-dir "$OUT" \
    --candidates "$OUT/candidates.tsv.gz" --known-qtl "$KNOWN" \
    --scores-plantcad "$OUT/dl_scores_plantcad.tsv.gz" \
    --scores-agront "$OUT/dl_scores_agront.tsv.gz" --top-n 100 2>&1 | tail -3
done

echo "================= RICE recall@100 summary ================="
$PY - <<'PYEOF'
import json
for t in ["Heading_date","Grain_length","Grain_width","Plant_height"]:
    f=f"results/phase5e/m3_3_dl_prior/rice_ricevarmap2/{t}/m3_3_summary_{t}.json"
    try:
        s=json.load(open(f))
        fr="DL-enhanced" if s.get("absolute_lift",0)>=0.05 else "prioritization"
        print(f"{t:14s} beta={s.get('chosen_beta')} recall {s.get('recall_gwas'):.2f}->{s.get('recall_fused'):.2f} lift={s.get('absolute_lift'):+.3f} [{fr}]")
    except Exception as e:
        print(t,"ERR",e)
PYEOF
echo "RICE_M33_DONE"
