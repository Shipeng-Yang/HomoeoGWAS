#!/usr/bin/env bash
# M3.4 v3 — re-run DL-prior prepare/score/fuse after the REF-match RC fix.
# Strand-flipped array alleles are now harmonised by reverse-complement, raising
# REF concordance (~46-50% -> ~85-93%). We re-score the enlarged candidate set
# with the SAME prespecified LOQO beta grid and report before/after transparently.
set -uo pipefail

ROOT=/mnt/7302share/fast_ysp/U7_GWAS
cd "$ROOT"
PY=~/miniconda3/envs/polygwas-gpu/bin/python

HORV_FASTA=data/reference/rapeseed/GCF_000686985.2_Bra_napus_v2.0_genomic.fa
HORV_MAP=data/reference/rapeseed/chrom_map_horvath_to_ncbi.tsv
HORV_KNOWN=data/reference/rapeseed/known_qtl_rapeseed.tsv
HORV_BED=data/processed/rapeseed/horvath
COT_FASTA=/mnt/nvme/cotton_hbau/GCA_018997965.1_ASM1899796v1_genomic.fa
COT_MAP=data/reference/cotton/chrom_map_hebau_to_ncbi.tsv
COT_KNOWN=data/reference/cotton/known_qtl_cotton.tsv
COT_BED=data/processed/cotton

# trait | panel | subgenomes | sumstats | leads | fasta | map | known | bed
RUNS=(
"bloom_50pct|horvath2020|A,C|results/phase3/m3_1_loco_v2/horvath2020/bloom_50pct/sumstats_bloom_50pct_loco.tsv|results/phase3/m3_2_qtl/horvath2020/bloom_50pct/new_locus_candidates.tsv|$HORV_FASTA|$HORV_MAP|$HORV_KNOWN|$HORV_BED"
"plant_height|horvath2020|A,C|results/phase3/m3_1_loco/horvath2020/sumstats_plant_height_loco.tsv|results/phase3/m3_2_qtl/horvath2020/plant_height/new_locus_candidates.tsv|$HORV_FASTA|$HORV_MAP|$HORV_KNOWN|$HORV_BED"
"fiber_length_BLUE|cotton_hebau|A,D|results/phase3/m3_1_loco_v2/cotton_hebau/fiber_length_BLUE/sumstats_fiber_length_BLUE_loco.tsv|results/phase3/m3_2_qtl_v2/cotton_hebau/fiber_length_BLUE/new_locus_candidates.tsv|$COT_FASTA|$COT_MAP|$COT_KNOWN|$COT_BED"
"lint_percentage_BLUE|cotton_hebau|A,D|results/phase3/m3_1_loco_v2/cotton_hebau/lint_percentage_BLUE/sumstats_lint_percentage_BLUE_loco.tsv|results/phase3/m3_2_qtl_v2/cotton_hebau/lint_percentage_BLUE/new_locus_candidates.tsv|$COT_FASTA|$COT_MAP|$COT_KNOWN|$COT_BED"
)

BACKUP=/tmp/rcfix_before
mkdir -p "$BACKUP"

for row in "${RUNS[@]}"; do
  IFS='|' read -r trait panel subg sumstats leads fasta map known bed <<< "$row"
  OUT=results/phase3/m3_3_dl_prior_v2/$panel/$trait
  echo "================= $panel / $trait ================="
  # back up the shipped summary for before/after
  cp -f "$OUT/m3_3_summary_$trait.json" "$BACKUP/${panel}_${trait}_summary_OLD.json" 2>/dev/null || true
  cp -f "$OUT/candidates_summary.json" "$BACKUP/${panel}_${trait}_candsummary_OLD.json" 2>/dev/null || true

  echo "[prepare] $panel/$trait"
  $PY scripts/phase3/m3_3_v2_prepare_candidates.py \
    --panel "$panel" --trait "$trait" --subgenomes "$subg" \
    --sumstats "$sumstats" --leads-tsv "$leads" --known-qtl "$known" \
    --bed-root "$bed" --fasta "$fasta" --chrom-map "$map" \
    --out-dir "$OUT" --ld-leads-tier ALL 2>&1 | grep -iE 'REF match|breakdown|merged unique' || true

  echo "[score] plantcad(GPU0) + agront(GPU1) parallel"
  CUDA_VISIBLE_DEVICES=0 $PY scripts/phase3/m3_3_score_dl_prior.py \
    --model plantcad --candidates "$OUT/candidates.tsv.gz" \
    --fasta "$fasta" --out-dir "$OUT" --device cuda:0 \
    > "$OUT/score_plantcad.log" 2>&1 &
  PC_PID=$!
  CUDA_VISIBLE_DEVICES=1 $PY scripts/phase3/m3_3_score_dl_prior.py \
    --model agront --candidates "$OUT/candidates.tsv.gz" \
    --fasta "$fasta" --out-dir "$OUT" --device cuda:0 \
    > "$OUT/score_agront.log" 2>&1 &
  AG_PID=$!
  wait $PC_PID; PC_RC=$?
  wait $AG_PID; AG_RC=$?
  echo "  plantcad rc=$PC_RC  agront rc=$AG_RC"
  if [ $PC_RC -ne 0 ] || [ $AG_RC -ne 0 ]; then
    echo "  SCORE FAILED for $panel/$trait — see logs"; tail -5 "$OUT/score_agront.log"; continue
  fi

  echo "[fuse] $panel/$trait"
  $PY scripts/phase3/m3_3_fuse_and_evaluate.py \
    --trait "$trait" --out-dir "$OUT" \
    --candidates "$OUT/candidates.tsv.gz" --known-qtl "$known" \
    --scores-plantcad "$OUT/dl_scores_plantcad.tsv.gz" \
    --scores-agront "$OUT/dl_scores_agront.tsv.gz" --top-n 100 2>&1 | tail -3
done

echo "================= BEFORE / AFTER recall@100 ================="
$PY - <<'PYEOF'
import json, glob, os
rows=[]
for panel,trait in [("horvath2020","bloom_50pct"),("horvath2020","plant_height"),
                    ("cotton_hebau","fiber_length_BLUE"),("cotton_hebau","lint_percentage_BLUE")]:
    new=json.load(open(f"results/phase3/m3_3_dl_prior_v2/{panel}/{trait}/m3_3_summary_{trait}.json"))
    oldp=f"/tmp/rcfix_before/{panel}_{trait}_summary_OLD.json"
    old=json.load(open(oldp)) if os.path.exists(oldp) else {}
    rows.append((panel,trait,
        old.get("n_candidates"),new.get("n_candidates"),
        old.get("chosen_beta"),new.get("chosen_beta"),
        old.get("recall_gwas"),new.get("recall_gwas"),
        old.get("recall_fused"),new.get("recall_fused"),
        old.get("absolute_lift"),new.get("absolute_lift")))
print(f"{'panel/trait':32s} {'ncand o>n':12s} {'beta o>n':10s} {'rGWAS o>n':14s} {'rFus o>n':14s} {'lift o>n':14s}")
for p,t,co,cn,bo,bn,go,gn,fo,fn,lo,ln in rows:
    print(f"{p+'/'+t:32s} {str(co)+'>'+str(cn):12s} {str(bo)+'>'+str(bn):10s} "
          f"{str(round(go,2) if go is not None else None)+'>'+str(round(gn,2)):14s} "
          f"{str(round(fo,2) if fo is not None else None)+'>'+str(round(fn,2)):14s} "
          f"{str(round(lo,2) if lo is not None else None)+'>'+str(round(ln,2)):14s}")
PYEOF
echo "DONE"
