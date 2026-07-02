#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel 2>/dev/null || echo .)"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1

PLINK2=${PLINK2:-plink2}
DIAMOND=${DIAMOND:-diamond}
THREADS=${THREADS:-24}

BG=data/raw/rapeseed/bnagvd
PROC=data/processed/rapeseed/bnagvd
GFF=${GFF:-data/reference/rapeseed/darmor_v4.1/Darmor-bzh.gff3.gz}
D=reproducibility/paper/reproduce/rapeseed_bnagvd
MAP=$D/subgenome_map.tsv
PHENO_SRC=${PHENO_SRC:-"data/raw/rapeseed/phenotype/991 accessions (Wu D, et al. 2019)/ft.txt"}
CFG=reproducibility/paper/configs
CHRS="A01 A02 A03 A04 A05 A06 A07 A08 A09 A10 C01 C02 C03 C04 C05 C06 C07 C08 C09"
mkdir -p "$PROC"/{A,C,interact} "$BG/rehead"

for c in $CHRS; do
  out="$BG/rehead/chr$c.vcf.gz"
  [ -s "$out" ] && [ -s "$out.tbi" ] && continue
  cat "$D/header.txt" <(zcat "$BG/Darmor-bzh.SNP.anno_chr$c.vcf.gz") | bgzip -@ "$THREADS" > "$out"
  tabix -f -p vcf "$out"
done

tail -n +2 "$PHENO_SRC" | awk -F'\t' '$6!=""{print $1}' | sort -u > "$PROC/ft_samples.txt"
comm -12 "$PROC/ft_samples.txt" <(sort -u "$D/bnagvd_sample_order.txt") > "$PROC/cohort.txt"
awk '{print $1"\t"$1}' "$PROC/cohort.txt" > "$PROC/cohort_keep.txt"
python - "$PHENO_SRC" "$PROC/pheno_multi.tsv" <<'PY'
import sys, csv
rows = list(csv.reader(open(sys.argv[1]), delimiter="\t"))
with open(sys.argv[2], "w") as fh:
    fh.write("sample\tflowering_time\n")
    for r in rows[1:]:
        if len(r) >= 6 and r[5] != "":
            fh.write(f"{r[0]}\t{r[5]}\n")
PY

for SG in A C; do
  LIST=""; for c in $CHRS; do [ "${c:0:1}" = "$SG" ] && LIST="$LIST $BG/rehead/chr$c.vcf.gz"; done
  bcftools concat -Oz -o "$BG/rehead/${SG}_concat.vcf.gz" $LIST
  "$PLINK2" --vcf "$BG/rehead/${SG}_concat.vcf.gz" --double-id --allow-extra-chr --snps-only --max-alleles 2 \
    --keep "$PROC/cohort_keep.txt" --maf 0.05 --geno 0.2 --set-all-var-ids '@:#' \
    --make-bed --out "$PROC/$SG/all" --threads "$THREADS"
done

homoeogwas prep-snps --gff "$GFF" --subgenome-map "$MAP" \
  --bed A="$PROC/A/all" --bed C="$PROC/C/all" --feature mRNA --id-attr ID \
  --flank-bp 2000 --min-snp 3 --out-dir "$PROC/interact"

homoeogwas prep-homoeologs --mode pairwise --subgenomes A,C \
  --genes "$PROC/interact/genes_{S}.tsv" --method diamond-rbh \
  --proteins A="$D/pep_A.faa" --proteins C="$D/pep_C.faa" \
  --diamond "$DIAMOND" --threads "$THREADS" --min-snp-pair 3 \
  --out "$PROC/interact/pairs_AC.tsv"

homoeogwas interact -c "$CFG/runs/interact_rapeseed_bnagvd_flowering_time.yaml" --n-jobs "$THREADS"
