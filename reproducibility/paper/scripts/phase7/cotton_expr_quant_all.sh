#!/bin/bash
# Quantify all 20 DPA samples with salmon (truncated-gz-prefix aware) and build the per-accession
# TPM table for the four lead genes. Part of PREREGISTRATION_cotton_expression_product.md.
cd /mnt/7302share/fast_ysp/U7_GWAS
SALMON=~/.local/share/mamba/envs/straw_quant/bin/salmon
IDX=/mnt/lldata/cotton_multiomic_376/salmon_index
DEST=/mnt/lldata/cotton_multiomic_376/fastq_20DPA_400mb
OUT=/mnt/lldata/cotton_multiomic_376/salmon_quant
mkdir -p "$OUT"
quant(){
  f="$1"; acc=$(basename "$f" | grep -oE 'S[0-9]+' | head -1)
  [ -f "$OUT/$acc/quant.sf" ] && return 0
  $SALMON quant -i "$IDX" -l A -p 4 --validateMappings --quiet \
    -r <(zcat "$f" 2>/dev/null | awk 'NR%4==1{a=$0}NR%4==2{b=$0}NR%4==3{c=$0}NR%4==0{print a"\n"b"\n"c"\n"$0}') \
    -o "$OUT/$acc" 2>/dev/null || echo "QUANT FAIL $acc"
}
export -f quant; export SALMON IDX OUT
ls "$DEST"/*.400mb.fastq.gz | xargs -P 8 -n 1 bash -c 'quant "$0"'
# assemble TPM table: accession x 4 lead transcripts
python3 - <<'PY'
import glob,os
genes=["Ghir_A01G002400","Ghir_D01G002310","Ghir_A05G013600","Ghir_D05G013340"]
OUT="/mnt/lldata/cotton_multiomic_376/salmon_quant"
rows=[]
for d in sorted(glob.glob(OUT+"/S*")):
    acc=os.path.basename(d); sf=d+"/quant.sf"
    if not os.path.exists(sf): continue
    tpm={g:0.0 for g in genes}
    for l in open(sf):
        f=l.split("\t")
        tx=f[0].split(".")[0]
        if tx in tpm: tpm[tx]+=float(f[3])   # sum isoforms to gene
    rows.append([acc]+[tpm[g] for g in genes])
import csv
with open("/mnt/7302share/fast_ysp/U7_GWAS/results/phase7/cotton_multiomic/lead_tpm_20DPA.tsv","w") as o:
    w=csv.writer(o,delimiter="\t"); w.writerow(["accession"]+genes); w.writerows(rows)
print(f"wrote lead_tpm_20DPA.tsv: {len(rows)} accessions x {len(genes)} genes")
PY
echo QUANT_ALL_DONE
