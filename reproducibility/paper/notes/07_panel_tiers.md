# 07 — Panel 分级与 Benchmark 契约矩阵(冻结于 2026-05-20)

> 本文档冻结本期(v1)使用的 panel、trait 与 benchmark 工具集合。
> 任何要把 Tier 2 / blocked panel 拉进主结果的决定,都要在这里改并写明依据。

## 1. Panel 分级

| Tier | Panel | 物种 | 倍性 | 样本 | 状态 | 在 v1 的角色 |
|---|---|---|---|---|---|---|
| **Tier 1** | Watkins | wheat *T. aestivum* AABBDD | hexaploid | 1,035 | ✅ ready | **规模锚点**:大样本 / 三亚基因组 / 已知 QTL 回收验证 |
| **Tier 1** | hebau | cotton *G. hirsutum* AADD | tetraploid | ~419 | ✅ ready(需 CSV→VCF) | 二亚基因组 baseline,fiber 性状 |
| **Tier 1** | Horvath2020 (Zenodo 4302088) | rapeseed *B. napus* AACC | tetraploid | 428(geno+pheno 配齐) | ✅ ready | **开发锚点**:端到端先在这里跑通 |
| **Tier 2 / blocked** | Hu2022 (NGDC) | rapeseed *B. napus* AACC | tetraploid | 403 | ⚠️ blocked | 仅扩展基因型资源,**v1 不进主结果**;等通讯作者回信发原始 56-trait 矩阵 + sample key 后再决定是否提级 |
| **二期挂账** | (待选) | strawberry *F. ananassa* | octaploid | — | ⏸️ | 不在 v1 范围 |

### 1.1 Tier 升级条件(Hu2022)
- 必要:(a) 拿到 403 样本的 phenotype 矩阵 + (b) `bna.s1..bna.s403 ↔ 2AFxxx` ID key 同时齐全;
- 充分:还要满足至少 1 个 trait 在 Horvath2020 同名 trait 下方向一致(交叉验证)。

### 1.2 Tier 1 主 trait 冻结(每物种 1-2 个)

| Panel | 主 trait | 备选 | 选择依据 |
|---|---|---|---|
| Watkins | **heading date (HD)** | thousand kernel weight (TKW) | Vrn / Ppd 是教科书 QTL,易回收验证;phenotype 见 Watkins 表型库 |
| hebau | **fiber length** | fiber strength | hebau 原文最强信号;cotton 经典性状 |
| Horvath2020 | **seed oil content** 或 **glucosinolate (GSL)** | seed weight | Horvath2020 原文已报道位点,可作 ground truth 回收 |

→ 实际 trait 文件路径在 panel manifest(`configs/panels/{wheat,cotton,horvath2020}.yaml`,Phase 1 W2-3 落地)中。

---

## 2. Benchmark Applicability Matrix(8 baselines 分桶)

**Codex 提的关键观察**:8 个 baseline 不是同层级,不能强行同一张表。本期按工具类型分桶:

| Bucket | 工具 | 用法 / 是否进 v1 主表 | 备注 |
|---|---|---|---|
| **主桶 A — 单变异主效应 LMM(同层比较)** | **GEMMA** | ✅ 主表 baseline | 经典 LMM,跑得快,小群体稳;v1 必比 |
| | **regenie** | ✅ 主表 baseline | 主流大规模 GWAS,step1/step2 |
| | **SAIGE** | ✅ 主表 baseline | 混合模型,大样本 case-control 友好,wheat HD 二态化可用 |
| | **GWASpoly (R)** | ✅ 主表 baseline | **同源/异源多倍体专用**,与本方法直接对位 |
| **补充桶 B — Rare-variant / Burden(同层比较)** | **STAAR (R)** | 🟡 附录或补充图 | 不与单变异主效应混在主表,做 set-based burden 时单列 |
| | **DeepRVAT** | 🟡 附录或补充图 | rare-variant DL,同样不进单变异主表 |
| **输入表征桶 C — 可选对照** | **NodeGWAS** | 🟡 选 1 进补充 | 论文 repo 性质,集成成本高;若 1 个跑通即进 |
| | **networkGWAS** | 🟡 选 1 进补充 | 同上 |
| **不进 GWAS 主比较** | **pangenie** | ❌ N/A(永久) | **SV genotyper**,不是 GWAS 工具;原本预备作 SV extension 上游,但 SV extension 已 de-scoped(2026-05-20,docs/00 §2.2)— 无 reads + 无 graph,完全不可执行 |
| **本项目软件** | **HomoeoGWAS** | ✅(被评估对象) | Core = subgenome LMM + homoeolog Hadamard kernel |

### 2.1 主表(v1 主文 Table 1 / Fig 2-3)结构冻结

```
                wheat-A | wheat-B | wheat-D | cotton-A | cotton-D | horvath-A | horvath-C
GEMMA            -       -         -         -          -          -           -
regenie          -       -         -         -          -          -           -
SAIGE            -       -         -         -          -          -           -
GWASpoly         -       -         -         -          -          -           -
HomoeoGWAS (Hadamard)   -          -         -          -          -           -
```
评价维度:**power / FDR / λ_GC / runtime / peak RAM**。
**HomoeoGWAS 的卖点**必须在 power 或 runtime 任一维度上比 4 主桶中至少 2 个统计显著优胜,否则改 framing。

### 2.2 每个 baseline 的预算

- 每个主桶 baseline 限时 **3 工作日** 跑通 + 集成 + 一致输出格式;
- 超时 → 标 `blocked`,跳过,后续有空再补,**不堵主线**。
- 补充桶限时 5 工作日,超时直接降级到附录或 N/A。

---

## 3. Frozen Manifests(Phase 1 末产出物)

Phase 1 (W2-3) 必须产出:`configs/panels/{wheat_watkins,cotton_hebau,rapeseed_horvath2020}.yaml`,每个含:

```yaml
panel: wheat_watkins
species: wheat
ploidy: hexaploid
subgenomes: [A, B, D]
geno:
  vcf: data/raw/wheat/vcf/chr{1..7}{A,B,D}.*.vcf.gz
  reference: data/reference/wheat/iwgsc_refseqv1.0
  chrom_style: chr{1-7}{A,B,D}
pheno:
  source: <Watkins 表型库路径>
  primary_traits: [heading_date]
  secondary_traits: [tkw]
samples:
  n_geno: 1035
  n_pheno: <待 join 后填>
qc:
  maf_min: 0.01
  missing_max: 0.10
  hwe_pmin: 1e-6
v1_role: tier1_scale_anchor
```

---

## 4. 变更日志

| 日期 | 变更 | 依据 |
|---|---|---|
| 2026-05-20 | 文档创建,Tier 1 = 3 panels,Hu2022 标 blocked,8 baselines 分桶 | dual-plan (Claude + Codex) 评估结果;Codex 指出 Hu2022 缺 pheno + baseline 不同层级 |
| 2026-05-20 | pangenie 永久 N/A(SV extension de-scoped) | docs/00 §2.2;无 reads + 无 graph,1000 sample WGS ≈ 50 TB 不可行 |
