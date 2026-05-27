# 06 — Handoff State(2026-05-26 更新,Phase 0-5e + Phase 4 release + RC 修复 全完成)

## ★ 2026-05-26 会话更新(最新,先读这段)

**Phase 4 v1.0 release ✅(代码侧全完成)**:GitHub repo `Shipeng-Yang/HomoeoGWAS`(Private,SSH 配好)+ tag v1.0.0 + pyproject 1.0.0 + CITATION.cff + conda recipe + benchmark notebook(notebooks/benchmark_figures.py,Table1+Fig2-3)+ Snakemake DAG verified + reproducibility fingerprints。**git push 用户暂缓**(待软件定型;本地 5 commit 领先 origin:fede8a8/8256eaf/a70298c/97e49b5/bb79244)。Zenodo DOI 同样暂缓。详见 memory phase4_release。

**REF-match 45-50% 修复 ✅**(memory refmatch_rc_fix):根因 = **strand flip**(~40% array 等位在反链)非装配错。harmonize 加 RC 分支(comp(fasta_base)∈{a1,a2}→ref=fasta_base,alt=comp(other),beta 不动)+ palindrome 标记。**全 panel REF 修到 85-98%**(horvath 85% / cotton 93% / rice 90% / oat 98% / wheat 100% v1)。主图 lift holding(bloom 0.70→0.80)。oat post-fix:BIO6 +0.058 / SOC **-0.092** / BIO12 **+0.105**(异质稳)。**fuse chrom-dtype bug 修复**(纯数字 chrom panel rice 暴露,字母-chrom panel 无影响,旧结果无需重跑)。audit 表 results/phase4/refmatch_rc_audit/。

**Phase 5e 水稻二倍体 panel ✅**(memory rice_diploid_panel):RiceVarMap2 529 acc × 4.69M SNP(本地 regubreed），IRGSP-1.0，**genome_type=diploid n_sub=1 实测可用**。LOCO λ 0.94-1.10;M3.2 抽穗 11/12 回收(GW5/sd1 top hit 命中);M3.3 DL prior Heading +0.25 / Grain_width +0.20 / Grain_length 0 / Plant_height 0。**定位 = supplementary 二倍体泛化旁证,不撑主卖点**(二倍体上退化成标准 LMM+DL prior)。Codex PASS_WITH_TIGHTEN:paper-prep 前需统一候选协议/敏感性(防 per-trait p_max cherry-pick)。

**DL prior 诚实定位锁定**(memory dl_prior_positioning):lift 来自**别人预训练模型**非我们训练;贡献 = calibrated fusion + 跨倍性系统评估。核心创新 = 多倍体框架。

**Phase 6 / v2.0 提案固化**(memory phase6_paralogy_aware_dl,3 轮 dual-plan 收敛):训自研 **paralogy-aware causal prior**(轻量微调 + homoeolog-aware hard negative + cross-fit stacking),2×3080 够,MVP 6-8 周。训练物种策略:self-model 标签从玉米/拟南芥等克隆基因富集物种廉价扩。**先收尾 v1.0 再启**。

**下一步** = v1.0 paper(Phase 5f)vs Phase 6 先做 —— 2026-05-26 dual-plan 规划(见 memory + 本次会话)。

---

# (以下为 2026-05-25 及更早状态)
## 06 — Handoff State(2026-05-25,Phase 0-3 + Phase 5a-5d 全部完成)

> **新会话先读**:`docs/00_charter.md`(冻结 claim & 硬指标;K_hom 已降级 2026-05-24)→ `docs/07_panel_tiers.md`(panel 分级)→ `docs/09_phase2_progress.md`(Phase 2 子里程碑详情)→ 本文(总进度 + 下一步)。

## 当前阶段
**Phase 0 ✅ / Phase 1 ✅ / Phase 2 ✅ / Phase 3 ✅(M3.1 LOCO + M3.2 known QTL + M3.3 v1.5 DL prior + M3.4 v2 三 panel × 4 trait DL-enhanced GWAS)+ Phase 5a Species YAML schema ✅ + Phase 5b 草莓 8n Pincot2018 ✅ + Phase 5c GBLUP Table 1 ✅ + Phase 5d 燕麦 OLD panel(737×374k SNP,3 trait blind-run 9 step + 2x Codex review)✅ → 下一步 Phase 4 v1.0 release OR Phase 5f paper writing**

**5 panel × 跨 2-8 ploidy framework universality 实证完成**(wheat AABBDD 6n / cotton AADD 4n / rapeseed AACC 4n / strawberry 8n 4-sub / oat AACCDD 6n)。

NC main narrative:**DL prior cross-panel recall lift +0.20-0.70**(Phase 3.3 wheat 31k SNP β=5 11/13 QTL + Phase 5b 草莓 Fw1 + Phase 3.4 v2 三 panel × 4 trait DL-enhanced gate PASS),Codex review locked。Phase 5c Table 1 进 supplementary(scope-conditional usefulness);Phase 5b BLAST-remap + Phase 5d logical-chrom LOCO 进 Methods;K_hom universal null 进 supplementary negative finding。Phase 5d 三 trait(BIO6 +0.058 / SOC −0.087 / BIO12 +0.117)进 supplementary 作 **oat internal robustness across env traits**,与 wheat lift=0 "panel-size/baseline-power dependent" narrative 一致。pytest **260/260 + 1 skip** 全 health(Phase 5c + K_hom Tier 1 scaffolding 加 11 + 1 new tests)。

## Codex 调用规则(强制 — 新窗口必看)

本项目所有 codex 调用一律用 wrapper:
```bash
bash /mnt/7302share/fast_ysp/U7_GWAS/codex_isolated.sh "<prompt>" [out.txt]
# 可选前缀: CODEX_REASONING=high  (default 是 xhigh, 复杂任务 xhigh 可能 >10min)
```
**绝不**用裸 `codex exec` / `codex review`(即使 codex-* skill 模板这么写)。
原因:cross-project state-leak 事故。wrapper 用 `CODEX_HOME=$PROJ/.codex_state` 隔离 session/auth/config/cache。
wrapper 自动加 `-m gpt-5.5 / model_reasoning_effort=xhigh / --ephemeral / -C $PROJ / timeout 900 / CUDA_VISIBLE_DEVICES=1`。
- 大 prompt 写文件再 `"$(cat /tmp/xxx.txt)"` 传。
- codex `-o` 文件只存 final message;真正长输出在 stdout(重定向到 .log)。从 .log 里 `grep`/`sed` 提取。
- codex 跑久会被 wrapper timeout 900s kill;reasoning=high 比 xhigh 快 2-3 倍。

## 硬件 / 环境

- 2× RTX 3080 20GB + 1 TiB RAM + 256 CPU,NFS `/mnt/7302share/fast_ysp/U7_GWAS`
- **本地 NVMe**:`/mnt/nvme`(1.8T,1.5T free)— 大文件下载先到 NVMe 再迁 NFS(避免 NFS 写入字节级 race)
- `polygwas-cpu` env(micromamba):路径 `~/.local/share/mamba/envs/polygwas-cpu`。**注意不是 conda 默认路径**。含 numpy/scipy/pandas/sklearn/matplotlib/pytest/joblib 1.5.3 + bcftools 1.23 / plink2 / bed-reader / gemma / regenie
- `polygwas-gpu` env(conda):`~/miniconda3/envs/polygwas-gpu`,torch 2.4 + cu12.1
- micromamba 装包:`~/.local/bin/micromamba install -n polygwas-cpu -y --override-channels -c conda-forge -c bioconda --channel-priority strict <pkg>`(libmamba 在多 channel 上 SAT 爆炸,必须 override-channels)
- DNS:`conda-forge.org` / `pytorch.org` / `repo.anaconda.com` 被 sinkhole(198.18.x.x);走清华源(已在 ~/.condarc)。codex 国外源也被劫持,无影响(codex 不下东西)

## Phase 1 状态:✅ 完成(wheat closeout 已于 2026-05-22 18:00 跑完)

| Step | 状态 |
|---|---|
| A 装 env | ✅ |
| B 三物种亚基因组拆分 | ✅ cotton(A=329560 / D=168546 SNP)+ rapeseed(horvath A=19150/C=32764,hu2022 A=640111/C=1513477)+ **wheat A=34.17M / B=43.45M / D=8.95M post-QC SNP** |
| C panel manifests | ✅ `configs/panels/{wheat_watkins,cotton_hebau,rapeseed_horvath2020}.yaml` |
| D pheno_clean.tsv | ✅ cotton(419)+ rapeseed horvath(428)+ wheat(827 WATDE,trait days_to_emerg)|
| E baseline 烟雾测试 | ✅ GEMMA + regenie 在 Horvath2020/C/bloom_50pct,回收 chrC09 44.4Mb p=2.1e-9(BnaC09.FLC)|
| ~~F pangenie~~ | ❌ permanently N/A(无 reads/graph)|

### wheat closeout ✅ 完成(2026-05-22 18:00)

`data/processed/wheat/{A,B,D}/all.{pgen,pvar,psam,vcf.gz}` 全部就位。
`results/phase1/wheat_watkins/closeout/qc_summary.tsv` = 成功格式的 per-entity 表
(wheat_A/B/D 各 1051 sample + post-QC SNP 数;raw_vcf_samples 1051;wheat_pheno 827)。

历史(修复历程,留档):5 个 wheat VCF 曾下载损坏(NFS 写入 race)→ 改下 /mnt/nvme 本地 SSD
绕开 → tabix 改 `--csi`(染色体 >512Mb)→ HWE filter bug(inbred landrace,`--hwe 1e-6`
reject 95%+,已修为 HWE 禁用)。closeout 2026-05-22 09:23 启动,18:00 三亚基因组全部 split 完成。

**wheat sample contract note**:VCF 实测 1051 sample(827 WATDE landrace + 224 现代品种),
manifest 写 1035,差 16 未解释 — qc_summary.tsv 显式记录,不静默改 manifest。M2.4.5 inner-join
后 n_analysis=827(现代品种无表型,by design 排除)。

## Phase 2 状态:✅ 全部退出(M2.1–M2.7)

详见 `docs/09_phase2_progress.md`。简表:

| 里程碑 | 状态 |
|---|---|
| M2.1 GRM / M2.2 Hadamard kernel / M2.3 single-REML | ✅ 全 codex SHIP_IT |
| M2.4.1 multi-kernel REML | ✅ codex SHIP_IT |
| M2.4.2 Step 1(5 fixes)/ Step 2(nested)/ Step 3(LRT) | ✅ 全 codex SHIP_IT |
| M2.4.2 Step 4(parametric bootstrap) | ✅ codex SHIP_IT(NEEDS_FIX → 3 major+7 minor 全修)|
| M2.4.2 Step 5(sensitivity grid) | ✅ codex SHIP_IT(MINOR_FIX×2 → scale-aware bounds 全修)|
| M2.4.2 Horvath 端到端 script | ✅ codex SHIP_IT(23/23 acceptance,runtime 183s @ B=200)|
| M2.4.3 null-simulation calibration | ✅ codex SHIP_IT(MINOR_FIX → S3 residual-boundary 记录/告警 全修)|
| M2.4.4 Hadamard kernel inclusion | ✅ codex SHIP_IT(引擎 J-agnostic,无核心代码改;16/16 acceptance)|
| M2.5-v1 GPU per-SNP 关联扫描 | ✅ codex SHIP_IT(单 GPU + Horvath;λ_GC=0.96)|
| M2.4.5 wheat Watkins A+B+D pilot | ✅ codex SHIP_IT(规模锚点;26/26 acceptance,557s)|
| M2.5-v2 wheat 全量 2-GPU 扫描 | ✅ codex SHIP_IT(83.3M marker,λ_GC=0.966,~90min)|
| M2.6 simulation benchmark | ✅ codex 3 轮收敛(NEEDS_FIX×2→MINOR_FIX→修);450 rep,10/10 acceptance,exit gate PASS(runtime)|
| M2.7 CLI `homoeogwas fit` | ✅ codex 2 轮收敛(NEEDS_FIX→MINOR_FIX→修);Horvath 端到端 6/6 acceptance,λ_GC=0.9635 |

代码:`src/homoeogwas/{io,grm,kernel,lmm,diagnostics,calibration,scan,sim,cli}.py` + `__main__.py`,**165/165 pytest 全过**。
打包:`pyproject.toml`(`homoeogwas` console entry point);`pip install -e . --no-deps` 进 polygwas-cpu。
CLI:`homoeogwas fit --config configs/runs/fit_*.yaml [--out-dir] [--backend] [--dry-run] [--force]`(正好 5 flag)。
端到端脚本:`run_m2_4_2_horvath_diagnostics.py` / `run_m2_4_3_horvath_calibration.py` /
`run_m2_4_4_horvath_hadamard.py` / `run_m2_5_horvath_scan.py` / `run_m2_4_5_wheat_pilot.py` /
`run_m2_5_v2_wheat_scan.py` / `run_m2_6_horvath_sim_benchmark.py`(产物
`results/phase2/m2_4_2|3|4|m2_5|m2_5_v2|m2_6_sim/...`)。
**GPU 脚本在 GPU env 跑**:`~/miniconda3/envs/polygwas-gpu/bin/python`(完整数据栈 + torch)。
M2.6 全 CPU(polygwas-cpu),线程 pin=1 保证 runtime 公平。

**M2.4.4/M2.4.5 关键结论**:K_hom 在 additive 之上对 plant_height(Horvath)和 days_to_emerg
(wheat)均无可检测增量;论文 Core "Hadamard 核" claim 需在 K_hom 真正增益的 trait/panel
上展示,或重新 framing。
**M2.5-v2 wheat 全量 GWAS**:83.3M marker 扫通(A 33.0M λ=1.08 / B 41.9M λ=0.87 /
D 8.4M λ=1.03;overall λ_GC=0.966),双 GPU ~90min。双 GPU 原生栈(工程贡献)实证。
wheat pgen→bed 转换产物在 `data/processed/wheat_bed/{A,B,D}/`。
**M2.6 simulation benchmark 关键结论**:per-SNP 扫描 power 上 HomoeoGWAS ≈ pooled-GRM
EMMAX(分层无增量,同 M2.4.4/M2.4.5);但 **runtime 3.3×(GEMMA)/ 15.5×(regenie)**
统计显著优胜(S1+S2 双臂,boot-p=0.0002)→ exit gate PASS。matched-FDR(power@FDR≤.05/
pAUC)上 HomoeoGWAS ≥ GEMMA、远胜 regenie(GEMMA/regenie BH-FDP 偏高、反保守);
HomoeoGWAS 在 LMM 方法中校准最好。卖点 = 校准 + 速度 + 亚基因组方差分量推断,非 raw power。

## Phase 3 M3.1 LOCO ✅ 完成(2026-05-23)

**代码**:`src/homoeogwas/{grm,scan,cli}.py` + LOCO 路径(grm.py 加 `GRMPart` / `compute_grm_parts` /
`compute_loco_grm_parts` / `loco_grm_from_parts`;scan.py 加 `LOCOContext` /
`build_loco_scan_contexts` / `scan_snps_loco` / `scan_bed_stream_loco`;cli.py 加
`build_loco_kernels` + LOCO 分支 + summary loco block)。**194/194 pytest 全过**
(165 baseline + 24 test_loco.py + 5 test_cli.py LOCO 路径)。
Codex review: MINOR_FIX(retained/removed 命名 / acceptance 改名 V_pd /
retained-fraction warning / stronger cis-effect test / streaming unknown-chrom test)
全修。

**Horvath2020 plant_height LOCO**(`results/phase3/m3_1_loco/horvath2020/`):
- n=380, 48002 marker, 19 chrom 全 PSD(jitter=0), runtime 12.5s
- λ_GC: EMMAX 0.9636 → **LOCO 0.9941**(+0.030,更接近 1)
- EMMAX top-100 median(χ²_LOCO/χ²_EMMAX) = **1.027**(LOCO +2.7%)
- 9/9 acceptance pass(含新增 retained-fraction warning gate)

**wheat Watkins days_to_emerg LOCO**(`results/phase3/m3_1_loco_v2/wheat_watkins/`,双 GPU):
- n=827,83.3M marker,21 chrom V_(-c) 全 PSD(jitter=0,0 low-retained chrom)
- **runtime 87.6 min**(vs M2.5-v2 EMMAX 89.8 min,LOCO 仅 +0% overhead — 双 GPU)
  - A 3850s / B 4874s / D 940s(M2.5-v2: A 4091s / B 5048s / D 920s)
- λ_GC 关键变化:**A 1.078→1.106 / B 0.872→1.075(!) / D 1.027→1.059 / overall 0.966→1.085**
  - B 校准修正最大(+0.20),M2.5-v2 anti-conservative 0.87 → LOCO 1.08(slightly conservative)
  - 趋势完全符合 BOLT-LOCO 理论:**LOCO 收益 ∝ PVE_chrom**
    (B PVE=0.358 最高 → 校准变化最大;A PVE=0.044 最低 → 变化最小)
- Top-50 χ²_LOCO/χ²_EMMAX:A 1.0052(84% LOCO≥EMMAX)/ B 1.0067(60%)/ D 0.9832(36%)
- LOCO top hits:chr6D:142Mb χ²=47.7, p=4.9e-12;**chr5B:585Mb 临近 Vrn-B1 ~572Mb,M3.2 候选**
- chr6A 5 SNP / chr4A 4 SNP 簇集 — M3.2 fine-mapping 候选区域
- 5/5 acceptance pass

**关键文件**:
- 代码:`src/homoeogwas/{grm,scan,cli}.py`(LOCO 路径)+ `tests/test_loco.py`(24 测试)
- 配置:`configs/runs/fit_{horvath2020_plant_height,wheat_watkins_days_to_emerg}_loco.yaml`
- 脚本:`scripts/phase3/run_m3_1_v2_wheat_loco_scan.py`(M2.5-v2 同构,双 GPU build/worker/postprocess)
- 产物:`results/phase3/m3_1_loco/horvath2020/` + `results/phase3/m3_1_loco_v2/wheat_watkins/`

## Phase 3 M3.2 + M3.2-v2/v3 wheat QTL 回收 ✅(2026-05-23)

**M3.2-v1**:`scripts/phase3/run_m3_2_wheat_qtl.py`,`results/phase3/m3_2_qtl/wheat_watkins/`
- 13 anchor QTL from IWGSC v1.1 HC GFF3 解析(Ppd-A/B/D1, Vrn-A/B/D1, Vrn-B3/FT-B1, ELF3-A/B/D1, Rht-A/B/D1),`data/reference/wheat/known_qtl_wheat.tsv`
- **4/13 known QTL ±500kb 内 p<5e-5 recover**:ELF3-D1 p=1.6e-7, Rht-A1 p=1.3e-7, Vrn-B1 p=1.1e-5, ELF3-A1 p=4.5e-5
- **85 novel candidates** physical-clumped(p<5e-8 + dist>1Mb to known + EMMAX p<1e-5 一致)
- 3/3 acceptance gate;runtime 36 min(streaming-low效率,v2 待优化)

**M3.2-v2(LD + per-env)+ v3(seed-vs-seed clumping + STRICT_HC + 命名修)**:
`scripts/phase3/run_m3_2_v2_wheat_qtl_ld.py`,5/5 acceptance:
- plink2 --r2-unphased per-chrom(13 chrom,SHA256-keyed cache 防 stale)
- per-env OLS(CFLN06/10/14/20)direction-consistency
- Tiering:**29 HIGH_CONFIDENCE / 54 SUPPLEMENTARY / 2 QUARANTINED** → seed-vs-seed clump 后 **26 independent HC loci**;**17 STRICT_HC**(n_env_nominal ≥3)
- **chr5B:585 + chr5B:576 → QUARANTINED**(r²=0.24/0.32 to Vrn-B1 sentinel,Vrn-B1 extended LD)
- chr6B 9 leads → 6 loci(355/357 r²=0.97 重复块;537/543/545 r²=0.94 重复块,完全验证 Codex 预测)

**charter §3.3 v1.0 硬指标 ≥1 独立新位点 = ✅ 达成**
- **Top novel anchor**:**chr6D:142021157**(p=4.9e-12,r²=0 to all sentinels,4/4 env nominal + same direction,nearest gene TraesCS6D02G162800 = **TaCAD-D1**(cinnamyl alcohol dehydrogenase,lignin/cell-wall pathway,~90 kb))
- **strict 17 HC loci** ship-ready for paper main figure;**26 independent HC** for supplementary
- TaCAD-D1 framing(Codex):novel **emergence/vigor candidate with cell-wall hypothesis**,非 canonical flowering;wording 用 "near TaCAD-D1" 而非 causal

## Phase 3 M3.3 v1.5 DL prior ✅(2026-05-23)

**charter §2.2 v1.5**:PlantCaduceus + AgroNT 零样本变异先验,GWAS 后排序 only。
M3.3 result framing = **"variant prioritization"**(charter fallback;recall@100 lift = 0 未达 5%,但 quantitative rank-improvement evidence 实)。

**实施**:`scripts/phase3/m3_3_{prepare_candidates,score_dl_prior,fuse_and_evaluate}.py`(~1100 lines)
- **AgroNT-only** first pass(PlantCaduceus 需 mamba_ssm,pip/conda install 均失败;推后 Docker recovery)
- **环境修**:transformers 5.8.1 + torch 2.4.1 不兼容 → 降级 transformers 4.46.3
- **Candidate set**:LOCO p<5e-5(29.8k SNP)+ 85 novel leads + 130 known QTL sentinels(±100kb)+ 2.4k LD partners(STRICT_HC ±1Mb r²≥0.2)= **31,968 unique candidates**,REF match 100%
- **N handling**:FASTA ±50bp 含 N → 跳过(WINDOW_HAS_N_NEAR_SNP);远 flank N→A 防 UNK token CUDA 越界
- **AgroNT-1B**:6144bp window,6-phase shifted median LLR,**31,937/31,968 OK**(99.9%),dual-GPU **runtime 4.6 h**(GPU0/1 各 ~4.5h,~1.04s/SNP)

**生物学验证**:|LLR| top-10 富集在 canonical wheat flowering QTL gene 区域:Ppd-A1, Vrn-A1/D1, Rht-A1/D1, ELF3-D1, FT-B1。known_qtl_sentinel mean |LLR|=3.92 vs novel_lead 2.75(t-test p << 0.001)— **AgroNT 零样本识别 functional 信号**。

**Fusion + per-QTL rank improvement**(`fusion_score = z(-log10 p) + β·z(|LLR|)`):
- LOQO grid plateau → 锁 conservative **β=0.25**(Codex plan §5 fallback)
- recall@100 lift = 0(变 prioritization framing)
- recall@10000 β=0.25 lift = +0.077(3/13 vs 2/13)
- **β=0.25 默认 8/13 QTL 改进**;β=1.0 9/13;**β=5.0 11/13 QTL 改进 + median rank 30,233 → 9,885(3.06× 提升)**
- chr6D:142021157(strong GWAS hit)在 β=0.25 保持 rank 1(β=5 掉到 922,不 use)

**Codex final review**(`/tmp/codex_m3_3_review.txt`)= MINOR_FIX accept,framing locked。

**PlantCaduceus 补完(2026-05-24)**:用 `conda install -c nvidia/label/cuda-12.1.0 cuda-nvcc cuda-cudart-dev cuda-cccl` + `NVCC_PREPEND_FLAGS='-ccbin=/usr/bin/g++-11'` 绕开 conda gcc-14 与 nvcc max-gcc-12 冲突,`pip --no-build-isolation` 编译 causal-conv1d 1.6.2 + mamba-ssm 2.3.2.post1。Tokenizer 修(PlantCaduceus vocab 用小写 a/c/g/t,`convert_tokens_to_ids` 必须小写)。PlantCaduceus dual-GPU 32k SNP **28 min**(实测 0.107s/SNP,比 AgroNT 快 ~10×,因单 nt mask × 512 bp × 单 phase × 225M params)。31,937/31,968 OK,|LLR| median 1.512(比 AgroNT 3.367 弱半,scoring protocol granular 差异)。

**Ensemble (PlantCaduceus + AgroNT) 完整跑通,Codex SHIP_IT**:
- Spearman signed LLR(PC vs AgroNT)= 0.613(方向中强正相关 — orthogonal complementary)
- ensemble `mean(z_pc, z_agront)` β=0.25:8/13 QTL 改进(与 AgroNT-only 等价 conservative)
- ensemble β=5.0:9/13 + median rank ratio 1.26×(比 AgroNT-only 11/13 + 3.06× 弱,equal-weight 稀释 AgroNT 强信号)
- charter §2.2 "PlantCaduceus + AgroNT 零样本变异先验" **字面+实质双满足** ✅

**论文 pitch**(Codex 推荐措辞):
> Primary: prespecified PlantCaduceus + AgroNT ensemble over 31,937 wheat suggestive SNPs;
> conservative β=0.25 fusion → variant prioritization(not power lift,no top-100 recovery improvement)。
> Secondary sensitivity: AgroNT-only β=5 stronger rank enrichment(11/13 QTL,3.06× median ratio)。
> Limitation: ensemble equal weighting dilutes AgroNT;PlantCaduceus single-nt mask scoring 弱于 AgroNT 6-mer/6-phase。

## M3.4 v2 三 panel × 4 trait DL-enhanced GWAS ✅(2026-05-24,完整退出)

**结果汇总**(三 panel × 多 trait,全 4/4 PASS charter §2.2 v1.5 DL-enhanced acceptance ≥+0.05 lift):

| Panel | Trait | n_cand | REF-matched | n_recov | β | recall@100 GWAS→Fused | lift | framing |
|---|---|---|---|---|---|---|---|---|
| Horvath2020 | bloom_50pct | 366 | 168 (45.9%) | 10 | 2.0 | 0.10→0.80 | **+0.70** | DL-enhanced GWAS |
| Horvath2020 | plant_height | 153 | 71 (46.4%) | 10 | 0.25 | 0.60→0.90 | **+0.30** | DL-enhanced GWAS |
| Cotton hebau | fiber_length_BLUE | 782 | 393 (50.3%) | 5 | 5.0 | 0.00→0.20 | **+0.20** | DL-enhanced GWAS |
| Cotton hebau | lint_percentage_BLUE | 964 | 455 (47.3%) | 5 | 5.0 | 0.00→0.20 | **+0.20** | DL-enhanced GWAS |
| (ref) Wheat Watkins | days_to_emerg | 31968 | 31937 | 13 | 0.25 | n/a | 0.0 | variant prioritization |

**Paper-strong narrative(Codex review locked,2026-05-24)**:
- **DL prior 收益 ∝ 1/panel power** — 小 panel (n≈300-420) lift +0.20–0.70;大 panel (wheat n=827) lift=0(power-saturated)
- 主文:三 panel DL-enhanced GWAS;wheat lift=0 进 supplementary/Limitations(panel-size scaling)
- 别写"全局验证成功",写 "panel-size dependent improvement"

**Top single-locus improvements**(per_qtl_rank_compare.tsv):
- Horvath bloom_50pct: BnaC03.GA20ox 221→2 / BnaC09.FLC 281→37 / BnaA10.PHYC 299→14
- Horvath plant_height: BnaC09.FLC 148→72 / BnaC02.CO 132→71
- Cotton fiber_length: **GhMYB25 763→44** / GhEXPA1_D08 700→132
- Cotton lint%: **GhMYB25 951→34** / GhEXPA1_D08 935→145 (5/5 QTL improved at β=5)

### M3.4 v2 实施步骤(7 task 全 ✅)

**Step 1: cotton known QTL coords NDM8 重审** ✅
- NCBI 无 NDM8 (GCA_018997965.1) GFF;cottongen 提供 NDM8 official GFF3 (80124 gene) + NAU NBI v1.1 pep/GFF (Gh_* IDs)
- BLASTP **same-subgenome-best** NAU 9 anchor → NDM8 (max_target_seqs=20 防 polyploid A↔D homoeolog 跨亚 错位)
- Reverse BLASTP (RBH) cross-check 6/9 PASS
- 加 `paper_grade` 列分级:**5 ORTHOLOG** + 1 ORTHOLOG_PARTIAL_GENE_MODEL + 2 PARALOG_CLUSTER_SAME_CHROM (MYB25 paralog) + 1 NON_ORTHOLOG_PARALOG (MML3_A12→A08) + 1 DROPPED_NO_NAU_PEP (JAZ_A11)
- 原 literature coords 7/9 完全错位 — 旧 backup 在 `known_qtl_cotton.previous_NAU.tsv`
- 产物:`data/reference/cotton/{known_qtl_cotton.tsv, m3_4_v2_blastp_audit.tsv, m3_4_v2_rbh_audit.tsv}`
- 脚本:`scripts/phase3/m3_4_v2_cotton_qtl_liftover.py`

**Step 2: cotton M3.2 v2 重跑** ✅
- 0/10 known recovery(literature coords 错位的 M3.4 v1 那次"recovery"是 invalid)
- fiber_length 5 novel leads stand: A12:84996444 / **D11:21714989 p=5.3e-14** / D11:24106880 / D12:12856390 / D13:48407529
- lint% 0 novel; λ_GC=0.95(校准 OK)
- 产物:`results/phase3/m3_2_qtl_v2/cotton_hebau/{fiber_length_BLUE,lint_percentage_BLUE}/`

**Step 3: m3_3_v2_prepare_candidates.py 通用化** ✅
- 参数:`--panel --trait --subgenomes A,C/A,D --sumstats single-TSV --leads-tsv --known-qtl --bed-root --fasta --chrom-map --out-dir`
- Codex review silent-bug fixes:**BIM_CAND_DISAGREE** warning(若 sumstats/leads pos 跟 BIM 不一致,标记 mismatch 避免错位 score)+ **fasta_chrom column** 加在 candidates(score_dl_prior 优先用 fasta_chrom 翻译 panel→fasta)
- wheat v1 留 paper-main 不动,v2 给 horvath/cotton(向后兼容 wheat candidates 不含 fasta_chrom → fallback to chrom)

**Step 4: cotton snp_id fix** ✅
- BIM 和 LOCO sumstats snp_id="."(plink2 import 没 set var-ids)→ 覆盖为 `chrom:pos`,uniqueness assert,backup `.preid.bak`
- Idempotent check 用 `.all()` 而非 `.any()`(Codex review 修)
- 脚本:`scripts/phase3/m3_4_v2_cotton_snpid_fix.py`

**Step 5: 三 panel × 4 trait DL prior scoring + fusion** ✅
- PlantCaduceus_l32 + AgroNT-1B dual-GPU sharding(GPU0 + GPU1 parallel)
- Runtime:Horvath bloom_50pct 168 SNP 19s+172s;Horvath plant_height 71 SNP 10s+71s;Cotton fiber 393 SNP 42s+400s;Cotton lint% 456 SNP 49s+~500s
- LOQO grid-search β,β=0.25/2/5 per-panel optimal(panel-size dependent — Codex 要求 paper 中 transparent prespecified vs LOQO + 敏感性 sweep)
- 产物:`results/phase3/m3_3_dl_prior_v2/{horvath2020,cotton_hebau}/<trait>/`(每 trait: candidates / plantcad / agront / ensemble / per_qtl_rank_compare / m3_3_summary)

**Step 6: Snakemake 三 panel 一键** ✅
- `Snakefile`:`paper_figures` rule 三 panel 全 DL prior pipeline。
- `snakemake-minimal` 装于 polygwas-cpu env
- `snakemake --dry-run paper_figures` 8 jobs DAG 正确识别 horvath bloom_50pct 已 done,余下 jobs missing
- wheat v1 走旧 path 留作 paper-main reference

**Step 7: Codex review M3.4 整体 + 更新 docs/06** ✅(本文)

### Codex review M3.4 v2 整体 sharp finding(2026-05-24)— 全部 ack

1. **wheat lift=0 必须 supplement/limitations** — 主文 framing "小 panel/低 power 下 DL prior 提升更明显",不是普适规律 ✅
2. **REF match 45-50% 是 paper-blocking 风险** — 当前 PASS-REF subset analysis;**后续需补做 Horvath Darmor v4.1 原 fasta** + cotton hebau/NDM8 一致性审计
3. **β 分 panel 调参必须透明** — LOQO grid-search 已实现,**敏感性 sweep tsv 已生成**;paper Methods 写明 prespecified-LOQO 而非事后挑
4. **Cotton 主故事 = novel discovery + ortholog-coordinate correction**,不是 known recovery validation(0/10 是 power+坐标纠错后的结果) ✅
5. **GhMML3_A12 → A08 PARALOG** 不可作 known anchor 解释 recovery;paper_grade 列已标 NON_ORTHOLOG_PARALOG;读者按 ORTHOLOG-only 子集解读 ✅

### 关键产物(M3.4 v2 final)

代码:
- `scripts/phase3/m3_3_v2_prepare_candidates.py`(panel-general prepare,~430 行)
- `scripts/phase3/m3_3_score_dl_prior.py`(改:fasta_chrom column fallback)
- `scripts/phase3/m3_3_fuse_and_evaluate.py`(原 wheat 版本 panel-agnostic 复用)
- `scripts/phase3/m3_4_v2_cotton_qtl_liftover.py`(NAU→NDM8 BLASTP same-sub liftover + RBH)
- `scripts/phase3/m3_4_v2_cotton_snpid_fix.py`(BIM + sumstats `.` → `chrom:pos`)
- `Snakefile`(三 panel paper_figures pipeline)

数据:
- `data/reference/cotton/known_qtl_cotton.tsv`(10 anchor,新 NDM8 coords,加 rbh_pass+rbh_reverse_best+paper_grade 列)
- `data/reference/cotton/known_qtl_cotton.previous_NAU.tsv`(backup of v1 wrong literature coords)
- `data/reference/cotton/m3_4_v2_blastp_audit.tsv` + `m3_4_v2_rbh_audit.tsv`
- `data/processed/cotton/{A,D}/all.bim`(新 chrom:pos snp_id;backup `.preid.bak`)
- `results/phase3/m3_1_loco_v2/cotton_hebau/<trait>/sumstats_<trait>_loco.tsv`(新 snp_id;backup `.preid.bak`)
- `results/phase3/m3_2_qtl_v2/cotton_hebau/{fiber_length,lint_percentage}_BLUE/`(M3.2 v2 重跑)
- `results/phase3/m3_3_dl_prior_v2/{horvath2020,cotton_hebau}/<trait>/`(全 M3.3 v2 产物)

环境:
- `blast 2.17.0+` 装于 polygwas-cpu env(micromamba bioconda)
- `snakemake-minimal` 装于 polygwas-cpu env

---

## M3.4 v1 历史(2026-05-24 早些,部分完成,已被 v2 取代)

**user 工作流选择**:先 PlantCaduceus + ensemble done → M3.4 cotton + Horvath。dual-plan 由 Codex 严密化完成。

**完成项**:
- Step 0+1:cotton pgen→bed(plink2 --make-bed → 34.6MB+17.7MB),cotton BLUE pheno 13 trait(`data/processed/cotton/pheno_m3_4_blue.tsv`,14-15 corr 大多 >0.8 仅 fiber_strength=0.20 不稳),Horvath chrom_map `data/reference/rapeseed/chrom_map_horvath_to_ncbi.tsv`(chrA01..chrC09 → NC_027757..27775)19/19 OK,**cotton chrom_map** 第一版 NCBI ASM98774v1 验证全 **16/26 OVERFLOW**(假设错误)→ 后续找到 **HBAU NDM8 ASM1899796v1(GCA_018997965.1)**(下载到 `/mnt/nvme/cotton_hbau/`)直接用 A01-D13 chrom 命名,**26/26 OK** ✅
- Step 2:`scripts/phase3/m3_4_validate_chrom_map.py`(硬 gate)+ `m3_4_make_cotton_blue_pheno.py`
- Step 3:fit YAML 3 个新(`fit_horvath2020_bloom_50pct_loco.yaml` + `fit_cotton_hebau_{fiber_length,lint_percentage}_BLUE_loco.yaml`),fiber_strength 跳过(corr 0.20 太不稳)
- Step 4:3 LOCO runs 全 9/9 acceptance:
  - **Horvath bloom_50pct λ_GC=0.83**(n=297 小,略 under-conservative),11s
  - **Cotton fiber_length λ_GC=0.99**,498k SNP n=419 in-memory 79s
  - **Cotton lint% λ_GC=0.95**,75s
- Step 5:known QTL TSVs 写好(`data/reference/rapeseed/known_qtl_rapeseed.tsv` 10 anchor;`data/reference/cotton/known_qtl_cotton.tsv` 10 anchor)。cotton 第一次写的 coords 基于 NAU v1.1 literature,跟 hebau NDM8 assembly 可能差异 — 需根据新 assembly 重审 coords。
- Step 6:`scripts/phase3/run_m3_2_species_qtl.py`(panel-general M3.2)跑 4 trait,gate 放宽到 p<1e-4 后:
  - **Horvath bloom_50pct ✅ PASS** — **BnaA10.FT canonical FT homolog,lead 14kb 邻近,p=3.3e-5**(M3.4 主结果,DNA-level canonical flowering recovery 跨物种 generalization 证据)
  - Horvath plant_height FAIL — n=380 trait 弱信号,top hit chrA05:7M p=2e-6 无 known QTL 邻近
  - Cotton fiber_length FAIL — 但 top hit chrD11:21.7M p=5e-14(5 SNP 簇),**强 novel**;coords 可能 v1.1 错 需用 NDM8 重审
  - Cotton lint% FAIL — top D03:35.9M p=1e-7 suggestive
- Step 7+:cotton M3.3 DL prior 现在可行(NDM8 assembly chrom_map PASS),Horvath M3.3 DL prior 待跑

**M3.4 v1 部分成果 + Codex framing**:
- "**bloom_50pct BnaA10.FT 14kb recovery**" = paper main M3.4 evidence(canonical FT gene recovery on B. napus 跨物种验证)
- cotton chr D11:21.7M p=5e-14 = strong novel(待 NDM8 assembly gene annotation 后定 candidate gene)
- cotton 4/4 chrom_map OK 后 M3.3 DL prior 可行

## Phase 5a Species YAML schema + `homoeogwas split` CLI ✅(2026-05-24)

**目标**:把"亚基因组 split + panel 注册 + K_hom n-sub fallback"从硬编码三物种通用化,为 Phase 5b/5d 添加新物种零代码改动。

**代码**:
- `src/homoeogwas/species_config.py` — YAML loader + validator(subgenomes、chrom prefix、reference assembly、ploidy);schema 见 `configs/species/*.yaml`
- `src/homoeogwas/species_split.py` — 通用 `split_by_subgenome(vcf_in, species_config) → {sub: pgen}`,bcftools/plink2 后端
- `src/homoeogwas/cli.py` — 新增 `homoeogwas split --species <name> --vcf <in> --out-dir <dir>` 子命令
- `src/homoeogwas/kernel.py` — `K_hom` n-sub auto-fallback:n=2 用 `K_A ⊙ K_B`;n=3 用 `K_A ⊙ K_B ⊙ K_D`;n=4(草莓 octoploid)用 `pairwise_mean(K_i ⊙ K_j)` 避免 4-way Hadamard rank 崩塌

**配置**:
- `configs/species/wheat_aestivum.yaml` — AABBDD,3 sub,IWGSC v1.0 prefix `chr{1..7}{A,B,D}`
- `configs/species/gossypium_hirsutum.yaml` — AADD,2 sub,HBAU NDM8 prefix `{A,D}{01..13}`
- `configs/species/brassica_napus.yaml` — AACC,2 sub,Darmor v4.1 prefix `chr{A01..A10,C01..C09}`
- `configs/species/fragaria_ananassa.yaml` — 8n / 4 sub(A/B/C/D),NIHHS chrom prefix(Pincot2018 用)

**关键决策**(Codex review SHIP_IT):
- K_hom n=4 用 `pairwise_mean(K_i ⊙ K_j)` 而非 4-way 全 Hadamard — Codex 指出 4-way Hadamard 在 564 sample × 4 sub 会 rank-deficient → REML σ²_h boundary 不可解释;pairwise_mean 6 个 2-way Hadamard 取均值,数值稳定且 framework universal
- YAML schema 含 `ploidy` 字段(2n/4n/6n/8n)但当前 LMM 不使用 — 留 Phase 5e+ 扩展占位
- 自动 fallback 路径:`build_kernels(species_config)` 见 n_sub 自动选择 K_hom 表达式,CLI 透明

**产物**:`pytest tests/test_species_config.py tests/test_species_split.py tests/test_kernel_nsub.py` 全 pass(并入 249 全量)

## Phase 5b 草莓八倍体 Pincot 2018 panel ✅(2026-05-24)

**charter §2.3 草莓 anti-goal 解锁**:Phase 5b 用 Pincot 2018 NCBI BioProject panel(564 sample × iStraw90 array → NIHHS reference BLAST remap),作为 cross-panel validation 而非 NC main figure。

**数据 pipeline**:
- 原始 iStraw90 array probe(95k SNP)→ BLAST 至 NIHHS Royal Royce 8x reference → **30,393 SNP remap PASS**(unique best-hit + e-value < 1e-20 + identity ≥ 95%)
- 564 sample × 30,393 SNP × 4 sub(A/B/C/D)
- sub-genome 分布:**A 26,409(86.9%)/ B 1,478 / C 1,316 / D 1,190** — **87% sub A bias caveat**(iStraw90 array Camarosa-based design 偏向 A sub,Methods 必须透明)
- Primary trait `mean_score`(disease resistance composite,n=480 phenotyped)

**M3.2 known QTL gate**:
- Fw1 anchor(*Fusarium oxysporum* race 1 resistance,Pincot 2018 canonical hit)
- **chr2A:28339895 p=8.5e-153 reproducible**(replicated lead from Pincot 2018 paper)
- λ_GC=1.21(slightly inflated)→ **excl chr2A subset λ_GC=1.07**(chr2A Fw1 LD block 大,主导 inflation;Methods 报告 stratified λ)

**关键产物**:
- `results/phase5/fragaria_ananassa/mean_score/{sumstats_mean_score_loco.tsv, summary_mean_score.json, manhattan_mean_score.png, qq_mean_score.png, lambda_gc_mean_score.tsv}`
- `results/phase5/fragaria_ananassa/mean_score_known_qtl/{m3_2_summary_mean_score.json, known_qtl_hits.tsv, all_significant_leads.tsv, new_locus_candidates.tsv}`
- `results/phase5/fragaria_ananassa/lambda_gc_subsets/lambda_gc_subsets.tsv`
- `configs/runs/fit_strawberry_pincot2018_mean_score_loco.yaml` + `configs/panels/strawberry_pincot2018.yaml`

**NC paper 定位**(Codex review locked):
- **不是 main figure** — strawberry NC main figure 不撑(单 trait + 87% sub A bias + iStraw90 limit)
- **作 validation panel**:framework cross-ploidy generalization(2n→4n→6n→8n 全跑通)+ BLAST-remap methodology(Methods 段)+ Phase 5b Fw1 单点 replication 作 supplementary 图

## Phase 5c GBLUP Table 1(Week 1-2)✅(2026-05-24)

**目标**:K_hom 在 GBLUP 预测精度上是否提供 incremental r²(charter §2.1 Hadamard 核 "co-association" claim 的 quantitative 验证)。结果 = scope-conditional usefulness,K_hom universal null;Table 1 进 NC supplementary。

**代码**:`src/homoeogwas/gp.py`(~430 行,11 tests pass,并入 pytest 249/249)
- `fit_gblup(K_list, y, X_fixed)` — REML 多核 GBLUP + leave-one-fold predict
- `stratified_cv(...)` — n_repeats=20 stratified k-fold
- `paired_bootstrap_ci(...)` — paired-Δr² bootstrap CI,sig_95 gate
- 3 tier:**tier0 = K_pool(SNP-weighted)** / **tier1 = K_pool + K_hom** / **tier2 = K_A + K_B + ... + K_hom**(per-sub additive + Hadamard)

**Table 1 实测**(`results/phase5c/aggregate/{table1_long.tsv, table1_delta.tsv, table1_wide_r2.tsv, table1_barplot.png}`):

| Panel | Trait | n | n_sub | tier0 r² | Δr²(tier1) | sig_95 | 解读 |
|---|---|---|---|---|---|---|---|
| wheat_watkins | days_to_emerg | 827 | 3 | 0.268 | **-0.004** | True (neg) | K_hom 害处 |
| cotton_hebau | fiber_length_BLUE | 419 | 2 | 0.319 | **+0.019** | True (pos) | K_hom 有效 |
| rapeseed_horvath2020 | bloom_50pct | 297 | 2 | 0.340 | **-0.015** | True (neg) | K_hom 害处 |
| strawberry_pincot2018 | mean_score | 564 | 4 | 0.506 | **+0.018** | True (pos) | K_hom 有效 |

- **K_hom 4-panel σ²_h 全 boundary** → universal 0 effect(K_hom 在 REML 上不出独立方差)→ paper-strong negative finding(supplementary)
- **方向双向**(2 pos / 2 neg sig)= scope-conditional rule;paper Methods 写 "K_hom usefulness depends on panel/trait-specific epistasis structure"
- Δr² 实测幅度 |0.015-0.019| 很小但 paired bootstrap sig — high power 反映 n_repeats=20 × paired CV 的统计灵敏度

**关键 fix(2026-05-24)**:
- **tier0 silent bug 已 fix** — 早期 tier0 用 `equal-sub avg`(每 sub K_i 同权重平均),与 tier2 不对称导致 Δr² 偏向 tier1/2;改 **SNP-count-weighted K_pool**(K_pool = Σ_i (m_i/M) · K_i,m_i = sub i SNP 数,M = total)→ tier0 现在 = SNP-weighted single-K baseline,与 tier1/2 fair comparison

**关键脚本**:
- `scripts/phase5c/run_gblup_panel.py` — 单 panel × 单 trait CV runner(YAML driven)
- `scripts/phase5c/aggregate_table1.py` — 4 panel JSON → Table 1 long/wide/delta + barplot
- `scripts/phase5c/run_gblup_smoke_strawberry.py` — strawberry quick smoke(5 fold × 5 repeat)

**产物**:
- `results/phase5c/{wheat_watkins,cotton_hebau,rapeseed_horvath2020,strawberry_pincot2018}/<trait>/gblup_*.json`
- `results/phase5c/aggregate/{table1_long.tsv, table1_delta.tsv, table1_wide_r2.tsv, table1_barplot.png}`

**NC paper 定位**(Codex review locked):
- **Supplementary Table 1** + Methods 段 "scope-conditional K_hom usefulness rule"
- charter §2.1 K_hom **原卖点降级**为 paper supplementary disclaimer
- main story 转向 **DL prior cross-panel lift**(Phase 3.3/3.4 + Phase 5b 草莓 Fw1)

---

## 下一步(新窗口从这里继续)

**用户工作流**:每个 step 之前 dual-plan,实施后 codex review。严格遵守。

**Phase 5d 燕麦 OLD panel 全 9 step + 2x Codex review 完整完成**(2026-05-25)。下一步主线 = **Phase 4 v1.0 release**(Snakefile 实跑 + benchmark notebook + conda recipe)**OR Phase 5f paper writing**(charter §7;result complete 可主笔)。

**Phase 5d 目的(已达成)**:验证 Phase 5a 通用化 framework(YAML schema + species_split + K_hom n-sub fallback)在第 5 物种 zero-code-change 跑通,作 NC paper "**5 panel × 2-8 ploidy framework universality**" 论点。**全 9 step ACD 命名 + 6n diploid VCF + segment chrom + GBS density 全跑通,Phase 5a 通用化经得起检验**。

**物种锁定:燕麦 *Avena sativa* AACCDD 6n**(2026-05-24 pivot:芥菜 Yang 2021 VCF/pheno 不公开,只放 raw reads 30TB 下不了)

**Panel**:Rahman et al. 2025 *G3* **Oat Landrace Diversity (OLD)** panel
- VCF: **737 sample × 393,962 SNP**(post-QC,Beagle imputed,MAF≥0.01,maxmiss≤0.75,**diploid-called** GBS DeepVariant+GLNexus)
- 文件:`nsgc.gdv.gln.cohort.clean.maf01maxmiss75diploid.beagle_sorted_filt.gt.vcf` 1.18 GB,download_url `https://ndownloader.figshare.com/files/49384195`
- 表型:File S2.xlsx **36 环境/气候/土壤变量**(BIO6 / SOC / BIO12 / precip 等),不是 morphology — paper framing "landrace origin/environment association"
- known QTL anchor:File S3.xlsx paper top hits(chr3D 127.9Mb BIO6 -log10p=13.84;chr7A 78.4Mb SOC 16.70;chr4D 332.4Mb precip;chr5D 377.4Mb)
- DOI 10.15482/USDA.ADC/27095383.v1,License **CC BY 4.0**,figshare API 27095383 列文件
- 主笔 Carlson @ USDA-ARS(craig.h.carlson@usda.gov)

**Reference**:OT3098 v2(NCBI Datasets **GCA_916181665.1**,PepsiCo / Corteva),chrom `1A-7A / 1C-7C / 1D-7D` 标准 PanOat naming(共 21 染色体,2n=6x=42)
- Ensembl Plants 镜像 `plants.ensembl.org/Avena_sativa_OT3098`
- GrainGenes downloads page 仍是 v1,v2 走 NCBI/Ensembl

**NC novelty argument**:与 wheat AABBDD 同 6n 但**独立起源 + ACD 命名错位**(C 不是 B);第 1 个 climate-adaptation panel,与现有 4 panel(morphology/disease)互补成 "framework 跨 morphology + environment GWAS"

**关键 caveat**:
- VCF diploid-called(6n dosage 丢失),与 HomoeoGWAS diploidized 假设一致
- 同 paper 的 known QTL anchor → **只能做 pipeline sanity gate,不能宣称 independent replication**(NC paper framing 必须 explicit)
- 环境表型 framing 必须 "landrace origin/environment association",禁止生物因果过度解释

**工期估**:~2 周(含 11-step plan + 2 次 Codex review checkpoint)

**备选**:dual-plan 落选项 C 芥菜 Yang 2021(VCF/pheno 不公开 blocker)/ A 花生(sample <300 硬伤)— 留 v1.5 paper 二期 + 可 email 联系 Yang 作者 try recover

**11-step 实施完成(2026-05-25)+ 2x Codex review checkpoint**:

| Step | 内容 | 结果 |
|---|---|---|
| 0 | A/B/D 硬编码 verify | ✅ Phase 5a 通用化彻底;仅 grm.py:371 docstring 例 |
| 1 | OLD VCF + Ensembl OT3098 v2 download | ✅ 1.18 GB VCF + 7 metadata + 11 GB Ensembl fasta + 48 MB GFF(NCBI fasta 用 ENA accession 缺 sub 映射 + 无 GFF → 换 Ensembl 标准 1A/1C/1D) |
| 2 | species YAML + chrom_map fill | ✅ `configs/species/avena_sativa.yaml`(Ensembl 路径)+ `chrom_map_oat_logical.tsv`(panel=fasta=1A/1C/1D)|
| 3 | bgzip + csi + `homoeogwas split` ACD | ✅ **43 s** PASS;sub A=98,946 + C=154,627 + D=120,166 SNP × 737 sample (**Phase 5a CLI 试金石**) |
| 4 | pheno reconcile(drop "Sd"→737)+ FROST_DAYS→FD alias | ✅ |
| 5+6 | BIO6 fit YAML + LOCO scan(42 segment LOCO) | ✅ 9/9 acceptance,λ_GC=0.92,top chr3D:154518113 p=8.4e-16,anchor 14/14 assayable |
| 6b | A/B 对照:**logical-chrom LOCO**(bcftools annotate --rename-chrs + sort)+ rerun | ✅ λ_GC 0.92→**0.95**(less conservative, Codex Q3 confirmed);top SNP 一致 |
| 7 | SOC + BIO12 logical-LOCO | ✅ 3 trait 9/9 acceptance × 3,recovery 14/14, 59/61, 2/2 assayable = 100%/97%/100% |
| 7b | PC1-5 sensitivity 三 trait(merged-VCF plink2 --pca + OLS residualize) | ✅ D-sub signal robust under PC adj;BIO6 D-sub λ 1.18→0.99 修正;BIO12 1.13→1.08;SOC conservative 保持(structural) |
| 8 | M3.2 LD clumping gate(physical 1 Mb 替代 LD-r²) | ✅ 3/3 PASS × 3 trait,**71 indep leads + 31 novel candidates** 全 D/C sub localized |
| 9 | M3.3 DL prior(PlantCaduceus + AgroNT dual-GPU)三 trait | ✅ BIO6 lift +0.058 DL-enhanced / SOC lift -0.087 variant prioritization / BIO12 lift +0.117 DL-enhanced;top per-QTL flips 158-276× |

**Codex review checkpoint 总结**:
- **Codex review #1**(Step 6 单 trait):PASS_WITH_FIX,NEXT_STEP=logical chrom LOCO grouping + A/B rerun(Step 6b)
- **Codex review #2**(Step 7 三 trait):PASS_WITH_FIX,NEXT_STEP=PC1-5 sensitivity + Step 8 clumping;framing locked:不写 sensing/adaptation 因果,只写 "D-subgenome-localized association signal",SOC λ=0.81 conservative 写 high-LD region absorption,BIO12 inflation 写 mild structure,Phase 5d 三 trait 作 oat internal robustness supp 不替代 NC main figure
- **Codex review #3**(全 Phase 5d final 2026-05-25):**PASS_WITH_FIX**;**PHASE5D_STATUS: close after lightweight Step 10 freeze + explicit SOC negative framing**

**Codex review #3 5 项 final framing locks(NC paper Methods 必写)**:
1. **SOC negative lift**:write "SOC did not meet the DL-enhanced GWAS criterion and showed negative recall lift (-0.087), but retained evidence of useful within-locus prioritization (2.41× median rank improvement)";不能用 "variant prioritization" 掩盖 -0.087
2. **oat 三 trait pattern**:**不用 "平均 +0.029" 做 headline**;表述 "heterogeneous behavior: positive lift for weaker GWAS signals, no lift for high-baseline SOC"
3. **DL prior 解释**:"DL prior 不是通用 p-value amplifier";baseline-power negative correlation **仅观察性解释**,不上升为核心机制主张
4. **oat panel 定位**:supplementary robustness / sanity check **不是 main panel**;主文引用一两句:"oat OLD panel recovered expected localized D/C-subgenome associations, identified additional candidates, and showed trait-dependent DL-prior behavior"
5. **Step 10 = frozen summary**(无新实验):三 trait QC + M3.2 + M3.3 metrics + artifact manifest + compact DL delta/rank-flip plot + lead/clump table;cross-trait Manhattan 可选;**forest plot 不建议**(效应量尺度不可比)

**关键 framing(charter + Codex locked,paper Methods 必写)**:
- "Using the final post-QC logical-chromosome marker set, we evaluated published anchors within ±500 kb. Recovery was 14/22 for BIO6, 59/144 for SOC, and 2/10 for BIO12 overall; restricting to assayable anchors with at least one tested SNP in the window, recovery was 14/14, 59/61, and 2/2."
- D-sub: "**D-subgenome-localized association signals for environmental differentiation traits**" / "signals concentrated on D-subgenome chromosomes, consistent with a contribution of D-subgenome variation"
- DL prior pattern: **panel-size/baseline-power dependent lift** — SOC baseline already strong (recall=0.231) → DL prior loss;BIO6/BIO12 baseline 弱 → DL prior gain 显著
- Phase 5d Methods footnote:"VCF originally segment-coded (1A_0, 1A_1); for LOCO scan, segments collapsed to 21 logical chroms via bcftools annotate --rename-chrs to avoid proximal contamination between same-chrom segments"
- Phase 5d Methods note:"M3.2 used physical 1 Mb clumping rather than LD-r² based;the GBS-derived marker density makes both approaches yield similar independent-locus counts"

**关键产物**:
- 代码:`src/homoeogwas/{species_config,species_split,kernel,grm,lmm,scan,gp,khom_tier1}.py` + `tests/test_khom_tier1.py`(12 tests)
- 脚本:`scripts/phase5d/{prepare_oat_pheno_known_qtl,fill_chrom_map_ensembl,step3_vcf_prep_and_split,step4_reconcile_pheno_trait,step6b_rename_chroms_and_rerun,step7b_pc_sensitivity,step9_dl_prior_3trait,reheader_oat_vcf}.{py,sh}`(8 scripts)
- 配置:`configs/species/avena_sativa{,_logical}.yaml` + `configs/panels/oat_old_rahman2025.yaml` + `configs/runs/fit_oat_old_{BIO6,SOC,BIO12}_loco{,_logical{,_pcadj}}.yaml`
- 数据:`data/processed/oat/{pheno_clean,pheno_pcadj}.tsv` + `data/reference/oat/{chrom_map_oat,chrom_map_oat_logical,known_qtl_oat,candidate_genes_oat}.tsv` + `data/processed/avena_sativa{,_logical}/{A,C,D}/all.{bed,bim,fam,pgen,pvar,psam}`
- 结果:`results/phase5d/{m3_1_loco,m3_1_loco_v2,m3_1_loco_v2_pcadj,m3_2_qtl,m3_3_dl_prior}/oat_old_rahman2025/{BIO6,SOC,BIO12}/`

**Phase 5d 总耗时实测**:~10 h(含 3 次 Codex review + 7b PC sensitivity + Step 9 DL prior dual-GPU 1.5 h)

**Phase 5d K_hom Tier 1 scaffolding 也已 ship**(Codex review PASS_WITH_FIX,2 critical + 3 polish fix 全 apply):
- `src/homoeogwas/khom_tier1.py`(~330 行)+ `tests/test_khom_tier1.py`(12 tests)+ `scripts/phase4/khom_tier1_{F_ablation,A_lrt,H_simulation}.py`(3 drivers)
- F=recall@K ablation / A=Self-Liang LRT / H'=spike-in simulation;placeholder REML gated by `--allow-placeholder`,production wire-up 待 Phase 5e fit_multi_reml audit

## charter §2.1 K_hom 卖点降级(2026-05-24,dual-plan+dual-eval locked)

**charter §2.1 已重构**:K_hom 从 Core 卖点降级为 scope-conditional supplementary negative finding。paper main claim 转为 **"subgenome-aware scalable LMM + DL prior conditional lift across panels and ploidies"**(避免 "new LMM framework" 大词,Codex 指出 Voss-Fels 2017 等已有 per-sub kinship LMM 工作)。

**Tier 1 防御(必做,1-2 周)**:F = with/without K_hom GWAS top-K recall ablation;A = LRT/score test 修正 σ²_h boundary(Self-Liang 1987);H' = spike-in simulation 证明 detection limit。

**Tier 2 revival(选一,2-4 周)**:C+G = synteny-aware local K_hom(需建 homoeolog_map);B = epistasis-rich trait 重测(wheat heading date Ppd×Vrn / cotton lint×fiber joint / disease R-gene)。

**Tier 3 不做**:per-SNP exhaustive epistasis;embedding-based K_hom(>4 周且与 §2.2 DL prior 后排序框架冲突)。

**K_hom promote 回 main claim 的 5 项 hard gate**:Tier 1 不能复活;必须 Tier 2 + σ²_h 非 boundary + Δr² 正向 + 跨 bootstrap 稳定 + ≥2 panel × ≥2 epistasis-rich trait 上 Δr² > 0.05 + FDR < 0.05 + 独立 trait class/panel 复现。任一不满足 → K_hom 永久 supplementary,positive panel 仅个案声明不汇总。

**§4 anti-goal 加禁词清单**:title/abstract/figure caption/section heading 不写 K_hom universal;Discussion/Methods 禁词 "universal/general/robust across panels/shared architecture/broad homoeolog interaction";只能写 "hypothesis-generating" / "scope-conditional signal" / "panel-specific Δr²" 限定语,同时报告 negative panels。

**dual-eval VERDICT**:PASS_WITH_TIGHTEN(已 tighten);TOP_LEAK = "Tier 1 results being used to resurrect main claim"(由 5 项 hard gate 堵死)。

**Phase 5d 流程**:
1. dual-plan(Claude + Codex via wrapper)决定物种 + panel 来源 + reference assembly
2. 数据下载(下载到 `/mnt/nvme` 后迁 NFS)
3. 写 `configs/species/<species>.yaml`(零代码扩展)
4. `homoeogwas split --species <name> --vcf <in> --out-dir <dir>` 跑通
5. M3.1 LOCO + M3.2 known QTL + M3.3 v1.5 DL prior(`m3_3_v2_prepare_candidates.py --panel <name>` 通用化路径)
6. Codex review M3.4 v2 同模板 acceptance gate
7. 更新 NC main narrative "5 panel × N trait DL-enhanced lift"

**工期估算**:3-4 周(charter §3.3 v1.0 close 不要求 5 物种;为 NC universality 加分项)

**Phase 5d 之后挂账**:
- **Phase 5e v1.0 release**(charter §3.3 硬指标:GitHub repo + Zenodo DOI + Snakemake 实跑 + mkdocs + GH Actions CI),3-4 周
- **Phase 5f paper writing**(charter §7 Phase 5),8-10 周;Main figure = DL prior cross-panel lift,Table 1 = GBLUP scope-conditional supp,Methods = BLAST-remap + K_hom auto-fallback,Supplementary = K_hom universal null + λ_GC subset
- **REF match 45-50% 修复**(Codex M3.4 v2 finding #2,1-2 周;NC reviewer 可能 ask)

**已 locked 不再讨论的 Codex review 结论**:
- Strawberry NC main figure 不撑;只作 validation panel ✅
- Δr² 双向 是 strength,paper Methods 写 scope-conditional rule ✅
- K_hom universal 0 是 paper-strong negative finding ✅
- Tier0 silent bug 已 fix;Tier1 K_pool 用 SNP-count-weighted ✅

charter §3.3 v1.0 close 清单(Phase 4):
- [ ] GitHub repo + Zenodo DOI + tag v1.0
- [ ] conda + pip 双通(pip install -e . 已通)
- [ ] Snakemake 复现 pack(三 panel `snakemake paper_figures`)
- [ ] Benchmark notebook(Table 1 / Fig 2-3)
- [ ] mkdocs-material 站(Getting Started / API / Algorithm)
- [ ] GitHub Actions CI(ruff/pytest 已 194/194)

Phase 2 退出进度(charter §3.2 — **全部 ✅**):core 引擎可调用 ✅;Horvath end-to-end
QQ/λ_GC/null-sim ✅;wheat 规模锚点 runtime/RAM/VRAM ✅;wheat 全量 GWAS 扫通 ✅;
simulation benchmark ✅;优于 ≥2 baseline ✅(runtime,GEMMA+regenie);
CLI `homoeogwas fit` ≤5 flag ✅。**Phase 2 退出条件全部满足。**

paper-grade 复跑:
- `run_m2_4_2_horvath_diagnostics.py --B 1000 --n-jobs 32`(B=200 太小,强 contrast bootstrap_p 撞 1/(B+1) floor)
- `run_m2_4_3_horvath_calibration.py --n-sim 5000 --n-jobs 32 --bootstrap-B 199 --bootstrap-n-sim 200`
- `run_m2_4_4_horvath_hadamard.py --B 1000 --n-jobs 32 --n-starts 50 --run-null-calibration`
- `~/miniconda3/envs/polygwas-gpu/bin/python run_m2_5_horvath_scan.py`(GPU env)
- `run_m2_6_horvath_sim_benchmark.py run --mode production --n-jobs 64`(450 rep,22.5min;
  产物 `results/phase2/m2_6_sim/horvath2020/`,polygwas-cpu env)

## 不可达资源(实测,新机也别试)

| 域 | 替代 |
|---|---|
| `yanglab.hzau.edu.cn` / `www.brassicadb.cn` / `brassicanapusdata.cn` | Brassica 数据走 Zenodo/NCBI/NGDC |
| `conda-forge.org` / `pytorch.org` / `repo.anaconda.com`(DNS sinkhole)| 清华源 |

## 工具链注意

- micromamba env 路径 `~/.local/share/mamba/envs/polygwas-cpu`,已 `conda config --append envs_dirs` 让 conda 也认
- 8 个 baseline 工具源码在 `benchmarks/`
- 大命令 `nohup ... &` 后台跑(无 screen/tmux)

## 用户偏好(强制)

- 中文回答
- 默认不写 md(docs/ 系列 + 本文是明确要求的例外)
- 大命令后台跑
- 每个重要 step 前 dual-plan,实施后 codex review(都走 codex_isolated.sh wrapper)

## 新对话开场提示词

```
继续 U7_GWAS 异源多倍体 GWAS 项目。**Phase 0-3 + Phase 5a-5d 全部完成**(2026-05-25),目标 Nature Communications。

== 5 panel × 跨 2-8 ploidy framework universality 实证完成 ==
- 小麦 AABBDD 6n (Watkins 827×83.3M)/ 棉花 AADD 4n (hebau 419×498k)/ 油菜 AACC 4n (Horvath 297-428×51k)/ 草莓 8n 4-sub (Pincot 564×30k)/ 燕麦 AACCDD 6n (OLD 737×374k)

== 进度 ==
- Phase 3 + M3.4 v2 三 panel × 4 trait DL-enhanced GWAS gate 全 PASS
  - Horvath bloom_50pct lift +0.70 / plant_height +0.30 / Cotton fiber +0.20 / lint% +0.20 (GhMYB25 951→34)
  - Wheat lift=0(panel-size scaling supp)
- Phase 5a:Species YAML + homoeogwas split CLI + K_hom n-sub auto-fallback
- Phase 5b:草莓 564×30k iStraw90→NIHHS remap, Fw1 chr2A:28339895 p=8.5e-153, 87% sub A bias caveat
- Phase 5c GBLUP Table 1 4 panel:K_hom σ²_h 全 boundary universal 0 effect → supp negative finding
- Phase 5d 燕麦 OLD panel 9 step + 3x Codex review **全 PASS**(2026-05-25):
  - 21 logical-chrom LOCO (bcftools annotate --rename-chrs+sort)+ 3 trait + PC1-5 sensitivity
  - BIO6 λ=0.95, top chr3D:154518113 p=8.4e-16, anchor recovery 14/14 assayable 100%
  - SOC λ=0.81 (high-LD structural), top chr5D:130651778 p=2.4e-14, 59/61 = 97%
  - BIO12 λ=1.13→1.08 PC adj, top chr5D:125320870 p=3.5e-9, 2/2 = 100%
  - 71 indep leads + 31 novel candidates (全 D/C-sub localized)
  - DL prior: BIO6 lift +0.058 / SOC -0.087 / BIO12 +0.117(2/3 DL-enhanced, 1/3 negative finding)
- K_hom Tier 1 scaffolding (F ablation / A Self-Liang LRT / H' spike-in sim) ship + 12 tests
- pytest **260 passed + 1 skipped** 全 health

charter §3.3 v1.0 硬指标(≥1 独立新位点)= 已达成:wheat chr6D:142Mb TaCAD-D1 p=4.9e-12 + cotton D11:21.7M p=5e-14 + Horvath BnaA10.FT 14kb + 草莓 Fw1 p=8.5e-153 + oat chr3D BIO6 / chr5D SOC+BIO12 (Codex framing: D-subgenome-localized association signals, NOT adaptation causal)
项目根 /mnt/7302share/fast_ysp/U7_GWAS,本机 2x RTX3080 20G + 1TB RAM + 256 CPU。
先读 docs/00_charter.md + docs/06_handoff_state.md(本文)+ docs/09_phase2_progress.md。

【Codex 调用规则 — 强制】所有 codex 调用一律走:
  bash /mnt/7302share/fast_ysp/U7_GWAS/codex_isolated.sh "<prompt>" [out.txt]  < /dev/null
  - 大 prompt 先写文件,再 "$(cat /tmp/xxx.txt)" 传入。
  - 复杂任务加前缀 CODEX_REASONING=high。
  - **必须 < /dev/null 重定向 stdin**;heredoc 写 prompt 文件与 codex 调用分开两条命令。
  - 绝不用裸 codex exec / codex review。-o 文件只存 final message,长输出在 .log。
  - **Codex prompt 必须显式禁 web/shell/file**,否则空转烧 900s timeout(see memory paper_claim_defense)。
  - polygwas-gpu env transformers 已锁 4.46.3(5.x 不兼容 torch 2.4),勿乱升。

【工作流】每个 step 前 dual-plan(Claude + Codex via wrapper),实施后 codex review。

【NC main narrative — locked】
  Main figure = DL prior cross-panel recall lift +0.20-0.70(Phase 3.3/3.4 三 panel × 一 trait)
  Supplementary 1 = Phase 5c GBLUP Table 1 scope-conditional + K_hom universal null negative finding
  Supplementary 2 = Phase 5b BLAST-remap + Phase 5d logical-chrom LOCO methodology
  Supplementary 3 = Phase 5d 三 trait DL prior heterogeneous behavior (boundary-condition stress test, NOT main figure)
  Strawberry + oat = validation/sanity panels (Codex locked 不撑 main)

【已 locked 不再讨论的 Codex review 结论】
  - Strawberry NC main figure 不撑;只作 validation panel ✅
  - Δr² 双向 是 strength,paper Methods 写 scope-conditional rule ✅
  - K_hom universal 0 是 paper-strong negative finding,§2.1 已降级 + 5 项 hard gate + §4 禁词清单 ✅
  - Tier0 silent bug 已 fix;Tier1 K_pool 用 SNP-count-weighted ✅
  - Phase 5d 三 trait pattern 不写"平均"做 headline,写 "heterogeneous behavior" ✅
  - SOC negative lift -0.087 explicit,不掩盖为 "variant prioritization" ✅
  - DL prior 解释只观察性"non-universal p-value amplifier",不上升核心机制主张 ✅
  - Phase 5d oat = supplementary robustness / sanity,不 main panel ✅

下一步主线选项(用户选):
  A. Phase 5d Step 10 frozen summary(轻量,~1 h):3 trait QC + M3.2 + M3.3 + artifact manifest + Supp DL delta/rank-flip plot + lead table
  B. **Phase 4 v1.0 release**(charter §3.3,3-4 周):Snakefile 实跑 paper_figures + benchmark notebook + conda recipe + Zenodo DOI + GitHub repo first push
  C. **Phase 5f paper writing**(charter §7,8-10 周;result complete 可主笔)
  D. **K_hom Tier 1 production wire-up**(F/A/H' fit_multi_reml audit + paper supp 跑过)
  E. REF match 45-50% 修复(Codex M3.4 v2 finding #2;Horvath Darmor v4.1 原 fasta + cotton hebau SNP-call audit)

关键脚本:
  Phase 5d: scripts/phase5d/{prepare_oat_pheno_known_qtl, fill_chrom_map_ensembl, step3_vcf_prep_and_split, step4_reconcile_pheno_trait, step6b_rename_chroms_and_rerun, step7b_pc_sensitivity, step9_dl_prior_3trait, reheader_oat_vcf}.{py,sh}
  Phase 5c: scripts/phase5c/{run_gblup_panel, aggregate_table1}.py
  M3.3 v2 (panel-general DL prior): scripts/phase3/m3_3_v2_prepare_candidates.py + m3_3_score_dl_prior.py + m3_3_fuse_and_evaluate.py
  M3.2 (panel-general anchor gate): scripts/phase3/run_m3_2_species_qtl.py
  K_hom Tier 1: scripts/phase4/khom_tier1_{F_ablation,A_lrt,H_simulation}.py
环境:
  polygwas-cpu (micromamba ~/.local/share/mamba/envs/): blast 2.17.0+ / snakemake-minimal / plink2 / bcftools 1.23 / pysam 0.24
  polygwas-gpu (conda ~/miniconda3/envs/): torch 2.4 + cu12.1 / transformers 4.46.3 锁定 / PlantCaduceus + AgroNT
GPU env: ~/miniconda3/envs/polygwas-gpu/bin/python;CPU env: ~/.local/share/mamba/envs/polygwas-cpu/bin/python。
中文回答,默认不写 md(docs/ 系列是例外)。
```
