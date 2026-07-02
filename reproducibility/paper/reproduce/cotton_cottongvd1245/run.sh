#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel 2>/dev/null || echo .)"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1

PLINK2=${PLINK2:-plink2}
DIAMOND=${DIAMOND:-diamond}

VCF=data/raw/cotton/cottongvd/gwasSample.2allel.vcf.gz
REF_GFF=data/reference/cotton/CRI_TM1_v1/TM_1.Chr_genome_all_transcripts_final_gene.change.gff.gz
PROT=data/reference/cotton/CRI_TM1_v1/TM_1.Chr_genome_all_transcripts_final_gene.change.gff.pep.gz
MAP=reproducibility/paper/configs/panels/maps/cotton_cottongvd_subgenome_map.tsv
OUT=data/processed/cotton/cottongvd
CFG=reproducibility/paper/configs
mkdir -p "$OUT"/{A,D,interact}

A_CHR=A01,A02,A03,A04,A05,A06,A07,A08,A09,A10,A11,A12,A13
D_CHR=D01,D02,D03,D04,D05,D06,D07,D08,D09,D10,D11,D12,D13

for SG in A D; do
  CHR=$([ "$SG" = A ] && echo "$A_CHR" || echo "$D_CHR")
  "$PLINK2" --vcf "$VCF" --chr "$CHR" --allow-extra-chr --set-all-var-ids '@:#' \
    --max-alleles 2 --snps-only --make-bed --out "$OUT/$SG/all" --threads 16
done

python - "$OUT/pheno_cgvd1245.tsv" <<'PY'
import sys, pandas as pd
df = pd.read_excel("data/raw/cotton/cottongvd/phenotype.fiber.quality.xlsx", sheet_name="dxm_2019_blup")
df.columns = [c.split("_")[0] for c in df.columns]
df.to_csv(sys.argv[1], sep="\t", index=False)
PY

homoeogwas prep-snps --gff "$REF_GFF" --subgenome-map "$MAP" \
  --bed A="$OUT/A/all" --bed D="$OUT/D/all" --feature gene --id-attr ID \
  --flank-bp 2000 --min-snp 3 --out-dir "$OUT/interact"

python - "$PROT" "$OUT/interact/prot_A.clean.faa" "$OUT/interact/prot_D.clean.faa" <<'PY'
import sys, gzip, re
prot, outA, outD = sys.argv[1:4]
op = gzip.open if prot.endswith(".gz") else open
fa, name = {}, None
with op(prot, "rt") as fh:
    for line in fh:
        if line.startswith(">"):
            name = line[1:].split()[0]; fa[name] = []
        elif name:
            fa[name].append(line.strip())
wa, wd = open(outA, "w"), open(outD, "w")
for name, seq in fa.items():
    s = re.sub(r"[.*]", "X", "".join(seq))
    m = re.search(r"_([AD])\d", name)
    if m:
        (wa if m.group(1) == "A" else wd).write(f">{name}\n{s}\n")
wa.close(); wd.close()
PY

homoeogwas prep-homoeologs --mode pairwise --subgenomes A,D \
  --genes "$OUT/interact/genes_{S}.tsv" --method diamond-rbh \
  --proteins A="$OUT/interact/prot_A.clean.faa" --proteins D="$OUT/interact/prot_D.clean.faa" \
  --diamond "$DIAMOND" --threads 16 --restrict-base-group --subgenome-map "$MAP" \
  --min-snp-pair 3 --out "$OUT/interact/pairs_AD.tsv"

for T in FibElo FibLen FibStr FibMic; do
  homoeogwas interact -c "$CFG/runs/interact_cotton_cottongvd1245_${T}.yaml" --n-jobs 16
done
for T in FibElo FibLen; do
  homoeogwas interact -c "$CFG/runs/interact_cotton_cottongvd1245_${T}_perm2000.yaml" --n-jobs 16
done
