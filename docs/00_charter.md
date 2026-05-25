# 00 — Project Charter(冻结于 2026-05-20)

> 这是项目的**最高级别冻结文档**。任何与本文 claim hierarchy / 硬指标 / exit criteria 冲突的开发决定,都要先回到这里改并写明依据。
> 阅读顺序:本文 → `07_panel_tiers.md` → `06_handoff_state.md`(运行进度)。

## 1. 项目一句话

异源多倍体 GWAS 方法学论文 + 开源工具 **HomoeoGWAS**,目标 **Nature Methods / Genome Biology**。

## 2. Claim Hierarchy(冻结)

### 2.1 Core(v1 必须做,论文卖点)— 2026-05-24 重构

**亚基因组分层 LMM(主卖点)** + **Homoeolog Hadamard 核 K_hom(scope-conditional supplementary)**

- 公式骨架:`y = Xβ + u_A + u_B [+ u_D] + ε`(per-subgenome 随机效应);Hadamard 核 `K_hom = K_A ⊙ K_B [⊙ K_D]` 作为**可选 epistasis 组分**,不强制 main claim。
- 实现位置:`src/homoeogwas/{kernel,lmm,infer,gpu}/*.py`。
- 与基线区分点:vs GEMMA / regenie 单核 LMM 假设 panmictic;vs GWASpoly *autopolyploid*(它不做 *allopolyploid* 亚基因组拆分)。区分点在 **subgenome-aware LMM** 上独立成立,**不依赖 K_hom 复活**。

**K_hom 实测降级依据**(Phase 5c 2026-05-24,dual-eval Claude+Codex locked):

> We originally hypothesized that a homoeolog Hadamard kernel, `K_hom = K_A ⊙ K_B [⊙ K_D]`, would capture cross-subgenome co-association beyond additive subgenome structure. Across four panels and paired bootstrap GBLUP-REML tests, however, the global `K_hom` showed **statistically detectable but direction-unstable Δr² and boundary variance estimates**, so it is **not interpretable as a robust cross-panel positive variance component**. We therefore retain `K_hom` only as a scope-conditional supplementary analysis (with two panels showing small positive Δr² and two showing small negative Δr² individually reported, not summed), and future extensions focus on synteny-aware, local, or trait-class-specific homoeolog kernels.

数据来源:`results/phase5c/aggregate/table1_*.tsv`(4 panel × 1 primary trait × n_repeats=20 stratified CV × paired bootstrap 95% CI);σ²_h 4 panel 全 boundary;Δr² range **|0.004-0.019|** 全 sig 但方向不稳定(Wheat **-0.004** / Cotton **+0.019** / Rapeseed **-0.015** / Strawberry **+0.018**)。

**paper main claim 转为**:**subgenome-aware scalable LMM + DL prior conditional lift across panels and ploidies**。K_hom 进 Methods + Supplementary 作 **transparent negative finding**,**不出现在 title/abstract**。

**Novelty wording 限制**(Codex Q3 tighten):避免 "**new LMM framework**" 大词 — Voss-Fels 2017 / Mascher 2021 等已有 per-subgenome kinship LMM 工作。Novelty 写在**组合 + 执行层**:(a) subgenome-partitioned random effects design;(b) polyploid panel cross-ploidy unified implementation(2n→4n→6n→8n);(c) DL prior conditional lift baseline-independent gain;(d) scalable workflow with ablation evidence;**不写"发明 LMM"**,改写 "**operational subgenome-aware extension + scalable implementation across allopolyploid panels**"。

**Phase 4/5 K_hom revival 候选**(dual-plan Claude+Codex 2026-05-24 合一,跑过后决定是否回 main):

**Tier 1 防御(1-2 周,必做,Phase 4 期间穿插)**:
- **F**: GWAS top-K recall@K with/without K_hom ablation(已有数据,~3-5 天)
- **A**: REML→LRT/score test 修正 σ²_h boundary 检验(Self-Liang 1987 mixture χ²,~1 周)
- **H' (Codex)**: spike-in simulation 证明 K_hom 在 n=500-1000 + additive-polygenic trait 上**理论上低于 detection limit**(强化 negative finding 科学可信度,~1 周)

**Tier 2 revival(选一,2-4 周;Phase 5 paper writing 前完成)**:
- **C+G (Codex)**: synteny-aware **local K_hom**(syntenic block ⊙ + per-block VC aggregation)— architecture 修复,需先建 `homoeolog_map`(原 charter §1 一直 ~ 未建)
- **B**: epistasis-rich trait class 重测(wheat heading date Ppd×Vrn / cotton lint×fiber joint / disease R-gene panel)— 依赖 phenotype 可用性

**Tier 3 不做**:per-SNP exhaustive epistasis(组合爆炸 + 多重检验 power 不可控);embedding-based K_hom(>4 周不进 critical path,除非主线转 foundation-model kernel — 而 §2.2 已锁 DL prior 用作后排序非 kernel)

**K_hom promote 回 main claim 的 hard gate**(2026-05-24 dual-eval tighten):
- **Tier 1 防御性结果(F/A/H')不构成 promote 依据**,只决定 supplementary 保留 vs 完全删除
- Promote 必须同时满足:(i) Tier 2(C+G 或 B)跑过 **σ²_h 非 boundary**;(ii) Δr² **正向**;(iii) **跨 bootstrap 稳定**(95% CI 不跨 0);(iv) ≥2 panel × ≥2 epistasis-rich trait 上 Δr² > 0.05 + FDR < 0.05;(v) **独立 trait class / held-out panel split 复现**
- 任一条件不满足:K_hom 永久作 scope-conditional supplementary,不进 main claim;positive panel 仅个案声明,不汇总

### 2.2 Extension(v1.5,**锁定 DL prior**,2026-05-20 由数据可行性决定)

**Option A — SNP×SV 联合**(**v1 不做**,2026-05-20 de-scoped)
- 路径不可行:PanGenie 等 SV genotyper 需要 short-read FASTQ + pan-genome graph,我们 Tier 1 panel 当前只有 array/VCF genotype(无 reads,无 graph)。
- 读测序数据规模:1000 sample WGS reads ≈ 50 TB,NFS 8.1 TB free 不够,且 Phase 1 时间不允许。
- 不"延期"而是"出 v1 范围",由后续版本(v1.5+)研究专门 SV panel(如 1000 Wheat Genomes SV release)再纳入。

**Option B — PlantCaduceus + AgroNT 零样本变异先验**(**v1.5 锁定为唯一 extension**)
- 触发条件:Core SNP-only 校准过关后启动。
- 价值:DL prior 用作 GWAS 后排序,**不做全量推理**,只对 suggestive hits + LD block 邻域(~10⁵ 量级)。
- Framing:若 power 提升 ≥5% → "DL-enhanced GWAS";否则改 framing 为 "variant prioritization"(不改架构)。
- 硬件就绪:本机 2× RTX 3080 20 GB + polygwas-gpu env(torch 2.4 + cu12.1 + transformers / peft / accelerate / bitsandbytes)已装。

### 2.3 不进 v1 主文(降级或附录)

- 双 GPU 原生栈 → 作为**软件工程贡献**写在 Methods 末尾,不作为科学卖点。
- NodeGWAS / networkGWAS / DeepRVAT / STAAR → 附录或补充材料。
- pangenie 作为 GWAS baseline → **永久 N/A**(它是 SV genotyper,不该对比)。
- 草莓八倍体 → 二期。
- Hu2022 panel 主结果 → 等 pheno 拿到再说(见 `07_panel_tiers.md`)。

## 3. 硬指标(go/no-go gates)

### 3.1 Phase 1 退出(4-6 周内)

必须全部 ✅:
- [ ] 3 个 Tier 1 panel 的 `configs/panels/*.yaml` manifest 完成,trait 冻结。
- [ ] 3 个 Tier 1 panel 的亚基因组拆分输出(`data/processed/<species>/<sub>/all.{pgen,vcf.gz}`)落地,SNP 数 / 样本数 / QC 通过率有报告。
- [ ] Benchmark applicability matrix(`07_panel_tiers.md` §2)4 个主桶 baseline **至少 2 个**在 Horvath2020 上 1 个 trait 跑出 manhattan(烟雾测试,不是结果声明)。
- [ ] `data/processed/{cotton,rapeseed/horvath,wheat}/pheno_clean.tsv` 落地,每个 panel sample ID 唯一,trait 数值化清洁。
- [ ] `results/phase1/wheat_watkins/closeout/qc_summary.tsv` 落地,显式记录 wheat raw VCF sample 数实测值 vs manifest 的 delta。
- [ ] `docs/06_handoff_state.md` 更新为 Phase 2 ready。

**已 de-scoped(不再阻塞 Phase 1 退出)**:pangenie SV calling — 见 §2.2 Option A。

### 3.2 Phase 2 退出(再 6-8 周,累计 3-4 个月)

必须全部 ✅:
- [ ] `src/homoeogwas/` core 引擎可装可调用,CLI `homoeogwas fit ...` 跑通。
- [ ] **开发锚点 Horvath2020**:end-to-end 至少 1 个 trait,QQ / λ_GC / null-simulation type-I error 报告生成。
- [ ] **规模锚点 wheat Watkins**:end-to-end 至少 1 个 trait,runtime / peak RAM / peak VRAM 表格生成。
- [ ] **Simulation benchmark**:per-subgenome causal SNP 注入,power vs FDR 曲线 vs ≥2 个主桶 baseline。
- [ ] HomoeoGWAS 在 power 或 runtime 任一维度上比 4 主桶中**至少 2 个**统计显著优胜(若不达到,框 storyline 重新定)。

### 3.3 v1.0 release(累计 6 个月)

必须全部 ✅:
- [ ] **≥1 个独立新位点**(三 Tier 1 panel 之一,真实数据,过经验阈值 + LD-fine-mapping 站得住)。**没有这个不送审**,故事缺真实数据 punch。
- [ ] GitHub repo + Zenodo DOI + tag v1.0。
- [ ] conda + pip 安装路径双通。
- [ ] 三 panel Snakemake/Nextflow 复现 pack 一键跑(`snakemake paper_figures`)。
- [ ] Benchmark notebook 出 Table 1 / Fig 2-3。
- [ ] mkdocs-material 站(Getting Started / API / Algorithm 三页)。
- [ ] GitHub Actions CI 小数据 smoke test + ruff/pytest。
- [ ] YAML 配置驱动,主流程命令行 ≤5 个 flag。

## 4. Anti-goals(明确不做)

- ❌ 在 v1 内试图同时做 SV 和 DL prior 两个 extension。**v1 已锁定 DL prior,SV de-scoped。**
- ❌ 在 v1 内强行让 8 个 baseline 全部上同一张主表。
- ❌ 在 v1 内做八倍体(草莓)。 **[2026-05-24 已破例:Phase 5b 草莓 Pincot 2018 panel 作 validation 不作 main]**
- ❌ 把 Hu2022 panel 当作主结果证据(在 pheno 拿到前)。
- ❌ 给 DL prior 跑全基因组千万 SNP 全量推理(改为 top-K + LD 邻域)。
- ❌ 在 Core 校准未过关前,启动 DL prior extension。
- ❌ 把 pangenie 作为 GWAS 主表 baseline(它是 SV genotyper,非同层方法)。
- ❌ **[2026-05-24 新加,dual-eval tighten]** K_hom 语言复活防护(直至 §2.1 hard gate 全过):
  - 不在 paper **title / abstract / figure caption / section heading** 宣称 global Hadamard `K_hom` 是 cross-panel **universal homoeolog epistasis evidence**
  - **禁词清单**(Discussion 与 Methods 也适用):"universal", "general", "robust across panels", "shared architecture", "broad homoeolog interaction", "across panels positive evidence" — 描述 global K_hom 时不得使用
  - Discussion 段写 K_hom 只能用 "**hypothesis-generating**" / "**scope-conditional signal**" / "**panel-specific Δr²**" 等限定语,并**同时报告 negative / opposite-direction panels**
  - K_hom positive Δr² panel(Cotton +0.019 / Strawberry +0.018)必须**逐 panel × trait-class 个案声明**,不得汇总成 "across panels positive evidence" 或 "2/4 panels"

## 5. 锚点策略(Codex 提出,采纳)

- **开发锚点 = Horvath2020**:428 样本 / 4-5 个 trait / 已有 published QTL → 端到端先在这里跑通,所有改动先在它身上验证。
- **规模锚点 = Watkins wheat**:1,035 样本 / 三亚基因组 / 大 VCF → 验证方法在最大规模上能跑、能加速、能出已知 QTL。
- 其余 panel(cotton)= 第三方验证,确保方法不是 Horvath2020-specific。

## 6. 硬件与运行约束

- 单机 2× RTX 3080 20GB + ~1 TiB RAM + 256 CPU(Codex 指出 RAM/CPU 不是瓶颈)。
- NFS `192.168.50.192:/mnt/7302share/fast_ysp/U7_GWAS`,8.1 T free。
- 双 GPU 无 NVLink → DDP/data-parallel 而非 model-parallel;`NCCL_P2P_DISABLE=1`。
- Wheat 单染色体最大 32 GB,拆分中间文件可达 100 GB 量级 → 验证通过后清 `data/processed/*/tmp/`。

## 7. 时间线(粗粒度)

| 阶段 | 周 / 月 | 内容 | 退出条件 |
|---|---|---|---|
| Phase 0 ✅ | (已完成) | 数据下载 + env yaml + 拆分脚本 | 见 `06_handoff_state.md` 旧版 |
| **Phase 1** | W1-6(~1.5 月) | 数据契约冻结 + 拆分跑完 + 2 个 baseline 烟雾测试 + pheno_clean.tsv + wheat closeout qc_summary | §3.1 全勾 |
| **Phase 2** | M2-4(~2.5 月) | SNP-only Core 引擎 + Horvath2020 端到端 + wheat pilot + simulation | §3.2 全勾 |
| Phase 3 | M4-6 | Extension = DL prior(SV 已 de-scoped 2026-05-20)+ 真实数据找新位点 | ≥1 新位点 |
| Phase 4 | M6-7 | 软件打包 + 文档 + CI + Snakemake 复现 pack | §3.3 全勾 |
| Phase 5 | M7-8 | 论文撰写 + 投稿 | 投出 |
| Phase 6 | (response) | 审稿响应 | 接收 |

## 8. 文档结构

```
docs/
├── 00_charter.md        ← 本文(claim hierarchy + 硬指标,冻结)
├── 01_project_plan.md   ← 早期总规划(留作历史)
├── 02_data_inventory.md ← 数据 as-built
├── 03_architecture.md   ← 软件架构
├── 04_method.md         ← 方法学细节
├── 05_related_work.md   ← 文献综述
├── 06_handoff_state.md  ← 当前阶段 / 跨会话交接(经常更新)
└── 07_panel_tiers.md    ← Panel 分级 + benchmark 矩阵(冻结)
```

`00` 和 `07` 是**冻结文档**,改动要在文末加变更日志。
`06` 是**滚动文档**,每次新会话开始读它。
其余可随实现演进。

## 9. 变更日志

| 日期 | 变更 | 依据 |
|---|---|---|
| 2026-05-20 | 文档创建,冻结 v1 claim hierarchy + Phase 1/2/v1.0 硬指标 + 二锚点策略 | dual-plan (Claude + Codex) 评估;hybrid 方案采用 |
| 2026-05-20 | SV extension de-scoped, DL prior 锁定唯一 v1.5 extension;Phase 1 退出条件移除 pangenie 后台启动,新增 pheno_clean.tsv + qc_summary.tsv 要求 | codex-loop 评估:Tier 1 panel 缺 reads/graph,1000-sample WGS ≈ 50 TB 超 NFS;Step F 永久 N/A |
| 2026-05-24 | §2.1 K_hom 降级:从 Core 卖点降级为 scope-conditional supplementary negative finding;paper main claim 转为 "subgenome-aware scalable LMM + DL prior conditional lift";Tier 1-3 revival paths(F/A/H' 防御 + C+G/B revival + D/E 拒做)由 dual-plan 列出;§4 anti-goal 同步加 K_hom universal 限定;§4 草莓 anti-goal 标记 Phase 5b 破例 | Phase 5c GBLUP Table 1 实测 σ²_h 4 panel 全 boundary + dual-check (Codex Q3 TOP_RISK="K_hom core-claim collapse") + dual-plan K_hom revival (Codex+Claude 合一 8 path 评分) |
| 2026-05-24 | dual-eval tighten:(a) §2.1 K_hom 实测描述精确化(Codex wording "direction-unstable Δr² and boundary variance estimates, not a robust cross-panel positive variance component"),Δr² range 修为 \|0.004-0.019\|;(b) §2.1 加 promote 回 main claim 的 5 项 hard gate(σ²_h 非 boundary + Δr²>0.05 + FDR<0.05 + 跨 bootstrap 稳定 + 独立 trait class/panel 复现);(c) §2.1 加 novelty wording 限制(不写 "new LMM framework",改 "operational subgenome-aware extension");(d) §4 K_hom anti-goal 扩展为禁词清单 + Discussion/Methods 范围限制 | dual-eval Codex+Claude VERDICT=PASS_WITH_TIGHTEN,TOP_LEAK="Tier 1 results being used to resurrect main claim";两人独立一致诊断 |
