# 02 — Data Inventory(as-built, 2026-05-20)

> 此文档反映 NFS 上**实际下载完成的数据**(覆盖了项目启动期的规划版本)。
> 总盘点:`/mnt/7302share/fast_ysp/U7_GWAS/` 已用 **~382 GB**,剩 8.1 T。

---

## A. 参考基因组 `data/reference/`

| 物种 | 参考 | 文件 | 大小 |
|---|---|---|---|
| 小麦 | **IWGSC RefSeq v1.0** Chinese Spring | `wheat/iwgsc_refseqv1.0_all_chromosomes.zip` + `iwgsc_refseqv1.1_genes_2017July06.zip` | 8.4 G |
| 棉花 | **NCBI GCF_000987745.1**(TM-1 NAU v1.1) | `cotton/GCF_000987745.1_ASM98774v1_genomic.fna.gz` (+gff) | 658 M |
| 油菜 | **NCBI GCF_000686985.2**(Bra_napus_v2.0,实即 Darmor-bzh v4.1) | `rapeseed/GCF_000686985.2_Bra_napus_v2.0_genomic.fna.gz` (+gff) | 559 M |
| 油菜备份 | ZS11(GWHANRE00000000) | `rapeseed/_ZS11_BnIR_offline_backup/` | 已下,弃用 |

⚠️ 小麦用 **v1.0 不是 v2.1**,因为 Watkins VCF 的 CHROM 是 v1.0 风格 `chr1A..chr7D`。

---

## B. 基因型 + 表型 Panel

### B.1 小麦 — Watkins 1,035 panel ✅
- **来源**:Cheng et al. *Nature* 2024 "Harnessing landrace diversity empowers wheat breeding";ENA **PRJEB71453 / ERZ22157275**
- **VCF**:`data/raw/wheat/vcf/chr{1A..7D}.SNP.*.maf_retain.vcf.gz` 共 21 个染色体文件,**340 G**
- **CHROM**:`chr1A..chr7D`(IWGSC v1.0 风格,分析前 verify)
- **表型** `data/raw/wheat/phenotype/`:
  - `WISP_WGIN_Watkins_JIC_2006_2010_2011_Field_Data.xlsx`
  - `DFW_phenotype_data_CF_{2011..2017}.xlsx`(7 年田间)
  - `JIC_Wheat_Pre-Breeding_WISP_Watkins_{Core,Whole}_Collection.xlsx`
  - `DFW_ParWat_RIL_phenotype_data_for_Watseq/`(RIL 表型子目录)
  - `DFW_TKNILSet1-5_phenotype_data_for_Watseq/`(NIL 表型)
  - `Natural_Populations/`、`Other/`(细分)
  - 来源:WWWG2B portal(http://wwwg2b.com/)+ wisplandracepillar.jic.ac.uk
  - **137 个性状**

### B.2 棉花 — hebau panel ✅
- **VCF/SNP**:`data/raw/cotton/snp/genotype-data.zip`(275 M)
- **表型**:`data/raw/cotton/phenotype/`(2 个 zip)
- **CHROM**:hebau 原始坐标系,**需 verify** vs NCBI GCF_000987745.1 的 `NC_0300xx.1`

### B.3 油菜 — Horvath2020 winter panel ✅(本期主基准)
- **来源**:Anderson, Pasche, … D.P. Horvath et al. *Agronomy* **10(12):2006 (2020)** "A New Diversity Panel for Winter Rapeseed (B. napus) GWAS",DOI 10.3390/agronomy10122006
- **Zenodo**:record **4302088**,DOI 10.5281/zenodo.4302088,直链 `https://zenodo.org/api/records/4302088/files/new%20supplemental.tar/content`
- **路径**:`data/raw/rapeseed/Horvath2020_Zenodo4302088/`
  - `original.vcf.gz`(129 M) —— 429 样本 `ars001..ars433`,290,972 SNPs(稀疏 GBS)
  - `imputed.vcf.gz`(77 M) —— **GWAS 用这个**(密集填补)
  - `imputed_filtered.zip`(26 M)、`kinship.txt`(2.2 M,亲缘矩阵)
  - `pheno_lines_phenotypes.xlsx`(51 K) —— 433 行 × ~11 性状(WINTER SURVIVAL / GROWTH HABIT / 50% BLOOM / Fall vigor / Bloom / ave bloom date / Stem elongation / ave growth habit / Fall stand / Plant height / ARS#/NAME/SOURCE 元数据)
  - `Horvath2020_new_supplemental.tar`(372 M,源 tarball 存底)
- **CHROM**:Darmor v4.1 pseudomolecules `chrA01..chrA10 / chrC01..chrC09`(+ `*_random`、`chrAnn_random`、`chrCnn_random`、`chrUnn_random` 未锚定池)→ **A/C 亚基因组直接在 CHROM 名里**,无需 FASTA 做子集
- **ID join**:VCF 429 ∩ 表型 432 = **428 份 geno+pheno**(VCF-only:ARS402;pheno-only:ARS028 / ARS111 / ARS294 / ARS370)
- **唯一弱点**:单试验点(USDA-ARS Fargo 越冬试验),多性状但近单环境 → 方法学基准够用,多环境丰富度由小麦/棉花补

### B.4 油菜 — Hu2022 403 panel(额外大基因型集)⚠️ pheno 缺
- **来源**:Hu J, Chen B, …, Wu X. *Nat Genet* **54:694–704 (2022)** DOI 10.1038/s41588-022-01055-6 "Genomic selection and genetic architecture of agronomic traits during modern rapeseed breeding"
- **VCF**:NGDC GVM `Brassica_napus/SNP/detailed_vcf/all_snp.vcf.gz`(`download.cncb.ac.cn/gvm/...`)—— 403 样本(匿名化 `bna.s1..bna.s403`),~7.5 M SNPs,Darmor v4.1 坐标
- **CHROM**:`NC_027757.2..NC_027775.2`(== NCBI GCF_000686985.2 contig 名)
- **表型**:**全网无公开版本** —— 原 brassicanapusdata.cn 死站(Wayback 全是 ysjianzhan.cn 网站建设器 shell);cgris.net/rapedata 只有种质描述符;OMIX/BnIR 无对应条目;paper Data-availability 写 "available upon reasonable request"
- **唯一获取路径**:邮件 Jihong Hu / Xiaoming Wu / Shilin Tian,同时索要 `bna.s1..bna.s403` ↔ paper `2AFxxx` ID key(否则 pheno 与 geno 无法 join)
- **本期处理**:**不阻塞**;Horvath2020 已是油菜主基准 panel,Hu2022 仅作额外基因型集备用

### B.5 草莓 ⏸️
二期挂账,本期不动。

---

## C. 深度学习模型 ✅

`data/reference/models/`(总 8.3 G,字节级 == HF 官方)

| 模型 | 参数 | 路径 | 用 |
|---|---|---|---|
| **PlantCaduceus_l32** | ~225 M(Caduceus/Mamba) | `plantcaduceus/`(924 M,`pytorch_model.bin` 901,671,450 B) | 每变异零样本先验权重 |
| **agro-NT-1b** | 1 B | `agront/`(7.4 G,`pytorch_model.bin` 3,965,239,677 B + JAX ckpt 3,965,106,595 B) | 第二先验 / ablation |

下载教训(给未来 ref):HF `snapshot_download` 不带 timeout 会 hang(实测 10 h+),改用 aria2c 直拉 `https://huggingface.co/<repo>/resolve/main/<file>`,可复用脚本 `scripts/download/_models_grab.sh`。

---

## D. 基线工具(8 个)✅

`benchmarks/`,通过 GitHub tarball API 下载(**git 未安装,不能 clone**):

| 工具 | 大小 |
|---|---|
| NodeGWAS | 1.4 M |
| GWASpoly | 23 M |
| networkGWAS | 81 M |
| regenie | 23 M |
| SAIGE | 167 M |
| pangenie | 3.7 M |
| STAAR | 2.4 M |
| deeprvat | 290 M |

⚠️ `benchmarks/REGENIE` 与 `benchmarks/Varigraph` 是 4K 空占位目录,可清。

---

## E. 模拟框架(本期暂不需要,留 v2)

AlphaSimR(R 包)/ PolySimX(in-house, v2 才动)/ MoBPS

---

## F. 网络可达性(实测 2026-05-19)

| 域 | 状态 |
|---|---|
| NCBI / ENA(`ftp.sra.ebi.ac.uk`)/ EBI / NGDC(`download.cncb.ac.cn`)/ Zenodo / figshare / Dryad / Springer static-content / HuggingFace | ✅ 通(直连) |
| `yanglab.hzau.edu.cn`(BnIR)、`www.brassicadb.cn` | ❌ DNS sinkhole 198.18.0.149 + IP-block 211.69.141.171(DoH `--resolve` 也通不上) |
| `brassicanapusdata.cn` | ❌ 死站(Wayback 全是 ysjianzhan.cn 网站建设器 shell,从未托管真数据) |

→ Brassica 类数据**不要再走 BnIR/BRAD**;首选 Zenodo / NCBI / NGDC。
