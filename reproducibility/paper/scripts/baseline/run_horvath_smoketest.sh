#!/usr/bin/env bash
# Phase 1 Step E — baseline 烟雾测试.
# 在 Horvath2020 horvath/C 子集 + bloom_50pct trait 上跑 GEMMA + regenie,
# 输出 sumstats + Manhattan + QQ. 目的是验证数据流通, 不是出科学结果.

set -euo pipefail

ROOT=${ROOT:-/mnt/7302share/fast_ysp/U7_GWAS}
ENV=${ENV:-$HOME/.local/share/mamba/envs/polygwas-cpu}
export PATH=$ENV/bin:$PATH

PANEL=rapeseed_horvath2020
SUB=${SUB:-C}                        # 32k SNP, A 只 19k
TRAIT=${TRAIT:-bloom_50pct}
THREADS=${THREADS:-4}

PFILE=$ROOT/data/processed/rapeseed/horvath/$SUB/all
PHENO=$ROOT/data/processed/rapeseed/horvath/pheno_clean.tsv
OUTDIR=$ROOT/results/phase1/$PANEL/$SUB/$TRAIT
mkdir -p "$OUTDIR"/{gemma,regenie,tmp}

echo "[$(date)] Step E: $PANEL / $SUB / $TRAIT"

# 1) 写两种 pheno:
#    - pheno_plink.tsv (#IID trait) 给 plink2 (psam 只有 IID 列, 用 --no-input-fid)
#    - pheno_regenie.tsv (FID IID trait) 给 regenie (要求 FID IID)
python - << PYEOF
import pandas as pd
df = pd.read_csv("$PHENO", sep="\t")
# 重复 sample 取均值 (Horvath ars403 有两条 site 数据)
df = df.groupby("sample", as_index=False)["$TRAIT"].mean()
plink_out = pd.DataFrame({"#IID": df["sample"], "$TRAIT": df["$TRAIT"]})
plink_out.to_csv("$OUTDIR/tmp/pheno_plink.tsv", sep="\t", index=False, na_rep="NA")
reg_out = pd.DataFrame({"FID": "0", "IID": df["sample"], "$TRAIT": df["$TRAIT"]})  # FID=0 与 plink2 fam 一致
reg_out.to_csv("$OUTDIR/tmp/pheno_regenie.tsv", sep=" ", index=False, na_rep="NA")
print("pheno rows (after dedup):", len(df), "non-NA $TRAIT:", df["$TRAIT"].notna().sum())
PYEOF

# 2) pgen -> bed: drop pheno 缺失样本, 再重算 MAF/missing/HWE
plink2 --pfile "$PFILE" \
  --pheno "$OUTDIR/tmp/pheno_plink.tsv" --pheno-name "$TRAIT" \
  --prune --maf 0.05 --geno 0.1 --hwe 1e-6 \
  --make-bed --out "$OUTDIR/tmp/bfile" --threads "$THREADS" 2>&1 | tail -8
BFILE="$OUTDIR/tmp/bfile"

# 2b) 把 chrC01..chrC09 / chrA01..chrA10 改成纯数字 (regenie 要求, GEMMA/plot 也接受)
#     备份原 chr 名映射, 便于 plot 时 map 回去
python - << PYEOF
chrom_map = {}
with open("$BFILE.bim") as f:
    rows = [line.rstrip("\n").split("\t") for line in f]
seen = []
for r in rows:
    if r[0] not in chrom_map:
        seen.append(r[0])
        chrom_map[r[0]] = str(len(seen))
with open("$OUTDIR/tmp/chrom_map.tsv", "w") as f:
    for k, v in chrom_map.items():
        f.write(f"{v}\t{k}\n")
with open("$BFILE.bim", "w") as f:
    for r in rows:
        r[0] = chrom_map[r[0]]
        f.write("\t".join(r) + "\n")
print("chrom map:", chrom_map)
PYEOF

# 3) GEMMA: 计算 kinship + LMM
(cd "$OUTDIR/gemma" \
  && gemma -bfile "$BFILE" -gk 1 -o kinship -outdir output 2>&1 | tail -3 \
  && gemma -bfile "$BFILE" -k output/kinship.cXX.txt -lmm 4 -o lmm -outdir output 2>&1 | tail -3)

# 4) regenie step 1 + step 2 (quantitative trait)
(cd "$OUTDIR/regenie" \
  && regenie --step 1 --bed "$BFILE" \
       --phenoFile "$OUTDIR/tmp/pheno_regenie.tsv" --phenoCol "$TRAIT" --qt \
       --bsize 1000 --lowmem --threads "$THREADS" --minMAC 5 --out step1 2>&1 | tail -5 \
  && regenie --step 2 --bed "$BFILE" \
       --phenoFile "$OUTDIR/tmp/pheno_regenie.tsv" --phenoCol "$TRAIT" --qt \
       --pred step1_pred.list --bsize 200 --threads "$THREADS" --minMAC 5 --out step2 2>&1 | tail -5)

# 5) 画 Manhattan + QQ
python - << PYEOF
import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

OUT = "$OUTDIR"; TRAIT = "$TRAIT"; SUB = "$SUB"

def plot_pair(df, label, prefix):
    df = df.dropna(subset=["chrom", "pos", "p"]).copy()
    df["chrom"] = df["chrom"].astype(str)
    df["neglog10p"] = -np.log10(df["p"].clip(lower=1e-300))
    df = df.sort_values(["chrom", "pos"]).reset_index(drop=True)
    # cumulative position for manhattan
    offset, cums = 0.0, []
    last_chr = None
    for _, row in df.iterrows():
        if row["chrom"] != last_chr and last_chr is not None:
            offset += df.loc[df["chrom"] == last_chr, "pos"].max() + 1e6
        cums.append(row["pos"] + offset)
        last_chr = row["chrom"]
    df["cum"] = cums

    # manhattan
    fig, ax = plt.subplots(figsize=(10, 4))
    colors = ["#1f77b4", "#ff7f0e"]
    chr_list = sorted(df["chrom"].unique())
    for i, c in enumerate(chr_list):
        sub = df[df["chrom"] == c]
        ax.scatter(sub["cum"], sub["neglog10p"], s=4, c=colors[i % 2])
    ax.axhline(-np.log10(5e-8), color="red", ls="--", lw=0.8, label="5e-8")
    ax.axhline(-np.log10(1e-5), color="orange", ls=":", lw=0.6, label="1e-5")
    ax.set_xlabel("Cumulative position")
    ax.set_ylabel("-log10(p)")
    ax.set_title(f"{label} Manhattan — {TRAIT} ({SUB})  N_snp={len(df)}")
    ax.legend(loc="upper right", fontsize=8)
    fig.savefig(f"{prefix}_manhattan.png", dpi=120, bbox_inches="tight")
    plt.close()

    # QQ + lambda  (两数组都升序, 排在一起 -> 标准 QQ)
    n = len(df)
    expected = -np.log10((np.arange(1, n + 1) - 0.5) / n)
    expected.sort()                           # ascending
    observed = np.sort(df["neglog10p"].values)  # ascending
    chi2 = stats.chi2.ppf(1.0 - df["p"].clip(lower=1e-300), df=1)
    lam = float(np.median(chi2) / 0.4549)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.scatter(expected, observed, s=4)
    m = max(expected.max(), observed.max())
    ax.plot([0, m], [0, m], "r--", lw=0.8)
    ax.set_xlabel("Expected -log10(p)")
    ax.set_ylabel("Observed -log10(p)")
    ax.set_title(f"{label} QQ — lambda={lam:.3f}")
    fig.savefig(f"{prefix}_qq.png", dpi=120, bbox_inches="tight")
    plt.close()
    return lam

# 读 chrom_map
chrom_map = {}
try:
    with open(f"{OUT}/tmp/chrom_map.tsv") as f:
        for line in f:
            num, name = line.rstrip("\n").split("\t")
            chrom_map[num] = name
except FileNotFoundError:
    pass

# GEMMA: output/lmm.assoc.txt (chr 是数字 1..9, 转回 chrC0x 用于 manhattan ordering)
gem = pd.read_csv(f"{OUT}/gemma/output/lmm.assoc.txt", sep="\t")
gem = gem.rename(columns={"chr": "chrom", "ps": "pos", "p_wald": "p"})
gem["chrom"] = gem["chrom"].astype(str).map(lambda c: chrom_map.get(c, c))
gem.to_csv(f"{OUT}/gemma/sumstats.tsv", sep="\t", index=False)
lam_g = plot_pair(gem, "GEMMA", f"{OUT}/gemma/gwas")
print(f"GEMMA: N={len(gem)} lambda={lam_g:.3f}")

# regenie: step2_<TRAIT>.regenie (chr 也是数字, 同样映射回)
# regenie 输出可能空格或 tab, 用 \s+ 容错
reg_file = f"{OUT}/regenie/step2_{TRAIT}.regenie"
reg = pd.read_csv(reg_file, sep=r"\s+", engine="python")
reg = reg.rename(columns={"CHROM": "chrom", "GENPOS": "pos"})
reg["chrom"] = reg["chrom"].astype(str).map(lambda c: chrom_map.get(c, c))
reg["p"] = 10.0 ** (-reg["LOG10P"])
reg.to_csv(f"{OUT}/regenie/sumstats.tsv", sep="\t", index=False)
lam_r = plot_pair(reg, "regenie", f"{OUT}/regenie/gwas")
print(f"regenie: N={len(reg)} lambda={lam_r:.3f}")

with open(f"{OUT}/summary.txt", "w") as f:
    f.write(f"panel={'$PANEL'} sub={SUB} trait={TRAIT}\n")
    f.write(f"GEMMA: N={len(gem)} lambda={lam_g:.4f}\n")
    f.write(f"regenie: N={len(reg)} lambda={lam_r:.4f}\n")
PYEOF

echo "[$(date)] Step E DONE -> $OUTDIR"
ls -la "$OUTDIR"/{gemma,regenie}/*.png "$OUTDIR/summary.txt" 2>&1 | head -20
