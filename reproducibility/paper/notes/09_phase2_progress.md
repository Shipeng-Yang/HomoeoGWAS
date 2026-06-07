# 09 — Phase 2 Progress(滚动文档,2026-05-22)

> Phase 2 = Core 方法实现:亚基因组分层 LMM + homoeolog Hadamard 核。
> 本文记录 M2.1–M2.5 各子里程碑的实现状态、产物、关键结果。
> 配合 `docs/06_handoff_state.md`(总进度)+ `docs/00_charter.md`(claim hierarchy)读。

## 代码包结构 `src/homoeogwas/`

| 模块 | 内容 | 状态 |
|---|---|---|
| `__init__.py` | public API export + version 0.1.0.dev0 | ✅ |
| `io.py` | `GenoChunk` dataclass + `load_bed_hardcall` + `vcf_to_bed`(plink2 wrapper) | ✅ M2.1 |
| `grm.py` | `compute_grm`(VanRaden)+ `compute_grm_panel`(per-subgenome) | ✅ M2.1 |
| `kernel.py` | `hadamard_kernel` + `sum_kernel` + `normalize_kernel` | ✅ M2.2 |
| `lmm.py` | `fit_reml`(single-kernel profiled REML)+ `REMLResult`;`fit_multi_reml`(multi-kernel)+ `MultiREMLResult` | ✅ M2.3 + M2.4.1 |
| `diagnostics.py` | `residual_only_reml` / `compare_nested_reml` / `NestedREMLComparison`(Step 2);`boundary_lrt` / `lrt_boundary_pvalue` / `boundary_lrt_table` / `BoundaryLRTResult`(Step 3);`parametric_bootstrap_lrt` / `bootstrap_lrt_table` / `BootstrapLRTResult` / `_bh_adjust` / `_psd_project`(Step 4);`pve_sensitivity_grid` / `SensitivityGridResult`(Step 5) | ✅ M2.4.2 Step 1-5 |

| `calibration.py` | `run_null_lrt_calibration` / `NullSimulationScenario` / `NullCalibrationResult` / `scenario_from_reduced_fit` / `summarize_type1` + chi-bar helpers(M2.4.3) | ✅ M2.4.3 |
| `scan.py` | `build_scan_context` / `ScanContext` / `scan_snps` / `ScanResult` / `lambda_gc` / `write_scan_tsv`(M2.5 per-SNP scan,CPU+GPU)| ✅ M2.5-v1 |

测试:`tests/{test_grm,test_kernel,test_lmm,test_lmm_multi,test_lmm_diagnostics,test_calibration,test_j3_hardening,test_scan,test_sim,test_cli}.py` — **165/165 全过**(2026-05-22,test_sim 34 个 M2.6 模拟基准 + test_cli 14 个 M2.7 CLI;1 GPU 测试在 CPU env skip)。

## 子里程碑状态

| M | 内容 | 状态 | Codex review |
|---|---|---|---|
| M2.1 | per-subgenome VanRaden GRM | ✅ | SHIP_IT(修了 4 bug)|
| M2.2 | homoeolog Hadamard kernel | ✅ | SHIP_IT |
| M2.3 | single-kernel REML LMM | ✅ | SHIP_IT(REML 公式 vs EMMA/GEMMA 核对正确)|
| M2.4.1 | multi-kernel REML(`fit_multi_reml`) | ✅ | SHIP_IT(NEEDS_FIX → 5 issues 已修)|
| M2.4.2 Step 1 | M2.4.1 5 个 hardening fix | ✅ | SHIP_IT |
| M2.4.2 Step 2 | nested REML compare | ✅ | SHIP_IT |
| M2.4.2 Step 3 | boundary-corrected LRT(Stram-Lee mixture) | ✅ | SHIP_IT(2 minor 已修)|
| M2.4.2 Step 4 | parametric bootstrap LRT | ✅ | SHIP_IT(NEEDS_FIX → 3 major + 7 minor 全修)|
| M2.4.2 Step 5 | sensitivity table(n_starts / norm / z-score) | ✅ | SHIP_IT(MINOR_FIX×2 → scale-aware bounds + reference 校验 全修)|
| M2.4.2 | Horvath end-to-end script(整合 Step 2-5) | ✅ | SHIP_IT(一次过,23/23 acceptance)|
| M2.4.3 | null simulation calibration | ✅ | SHIP_IT(MINOR_FIX → S3 residual-boundary 记录/告警 + p_naive 移出 gate + parser 健壮化 全修)|
| M2.4.4 | Hadamard kernel inclusion(A+C+K_hom)+ multi-start | ✅ | SHIP_IT(MINOR_FIX=产物过时,重跑刷新;代码无需改)|
| M2.4.5 | wheat Watkins A+B+D pilot(规模锚点) | ✅ | SHIP_IT(MINOR_FIX → preflight + 4 收敛 gate 全修)|
| M2.5-v1 | GPU per-SNP 关联扫描(单 GPU + Horvath) | ✅ | SHIP_IT(2 hardening guard 已补)|
| M2.5-v2 | 2-GPU + wheat 86.6M SNP 全量流式扫描 | ✅ | SHIP_IT(v2a 引擎 MINOR_FIX→修;v2b 生产 SHIP_IT)|
| M2.6 | Simulation benchmark(power-vs-FDR vs GEMMA/regenie) | ✅ | NEEDS_FIX×2 → MINOR_FIX → 全修;10/10 acceptance,exit gate PASS(runtime)|
| M2.7 | CLI `homoeogwas fit`(YAML 驱动,≤5 flag) | ✅ | NEEDS_FIX → MINOR_FIX → 全修;Horvath 端到端 6/6 acceptance |

## 关键 Horvath2020 结果(plant_height, n=380)

### M2.4.1 multi-kernel REML(A+C)
- PVE_A = 0.341,PVE_C = 0.659,PVE_e ≈ 0(σ²_e hit lower bound)
- kernel_corr(A,C) = 0.686
- **生物学**:C 亚基因组对 plant_height 贡献 ≈ A 的 2 倍

### M2.4.2 Step 3 boundary-corrected LRT(5 contrasts)
| null→alt | added | df | T | p_naive | p_boundary |
|---|---|---|---|---|---|
| e→A+e | A | 1 | 88.2 | 6.0e-21 | 3.0e-21 |
| e→C+e | C | 1 | 97.2 | 6.2e-23 | 3.1e-23 |
| A+e→A+C+e | C | 1 | 13.7 | 2.2e-4 | **1.1e-4** |
| C+e→A+C+e | A | 1 | 4.65 | 0.031 | **0.0155** |
| e→A+C+e | A,C | 2 | 101.9 | 7.6e-23 | 2.2e-23 |

→ **C 在 A 之上贡献强显著(p=1e-4),A 在 C 之上贡献显著但弱(p=0.016)**。不能 collapse 成 single K_sum,multi-kernel 必要。

### M2.4.2 Step 4 bootstrap(B=60 smoke,codex SHIP_IT 后)
- 5 contrasts bootstrap_p 全 ≈ 1/(B_usable+1) = 0.0164(B=60 仍太小,0 个 T_b ≥ T_obs;
  观测 LRT 极强)→ **paper run 必须 B=1000**
- converged_rate ≥ 0.983(`require_converged=True` 不误杀),jitter_used=0,bootstrap_p 回填一致
- codex review NEEDS_FIX → 已修:① n_starts 默认继承 observed fit(防 anti-conservative)
  ② `require_converged` 过滤未收敛 replicate ③ `replace()` 回填 `boundary.bootstrap_p`
  ④ jitter_used 记录 + warn ⑤ `_psd_project` 对齐 fit_multi_reml ⑥ B<1/缺 kernel 显式 ValueError
  ⑦ BH finite-only ⑧ MCSE 公式 ⑨ per-replicate RNG(n_starts>1 仍 n_jobs 无关确定)⑩ 表格列补齐

## 关键产物路径

```
results/phase2/m2_1/horvath2020/grm_A_C.npz            # G_A G_C samples
results/phase2/m2_2/horvath2020/K_hom_AC.npz           # K_hom_AC K_hom_AC_raw K_sum_AC samples
results/phase2/m2_2/horvath2020/kernel_heatmap_AC.png
results/phase2/m2_3/horvath2020/reml_fit.tsv           # single-kernel REML
results/phase2/m2_4/horvath2020/multi_reml_A_C_plant_height.{tsv,json,png}   # M2.4.1
results/phase2/m2_4_2/horvath2020/lrt_table_plant_height.tsv               # Step 3
```

## 关键设计决策(已冻结)

- **REML 算法**:single-kernel = profiled REML 1D Brent;multi-kernel = box-constrained L-BFGS-B over raw σ²(NOT exp(θ) — true σ²=0 boundary 需要 raw 才能 LRT)
- **PVE 公式**:paper-grade `component_var_j = σ²_j · trace(K_j)/n`;PVE = component_var / Σ。scale-invariant
- **LRT**:Self-Liang/Stram-Lee binomial chi-bar mixture(k 个 boundary component → `C(k,j)/2^k` weights)
- **bootstrap**:simulate under fitted null,Phipson-Smyth `(1+#{T_b≥T_obs})/(B+1)`
- **boundary detection**:PVE_j < boundary_eps(default 1e-3),scale-invariant
- **GPU**:M2.4 全 CPU;GPU 推到 M2.5 per-SNP scan

### M2.4.2 Step 5 sensitivity grid(codex SHIP_IT 后)
- 12 cell 网格:y∈{raw,zscore} × norm∈{trace,frobenius,none} × n_starts∈{1,10}
- 真实 Horvath plant_height(n=380):**global genetic PVE drift = 4.97e-5**,12 cell 全收敛,
  genetic_upper_bound_hit 全 False → A/C 亚基因组拆分对预处理选择稳健(Nature Methods sensitivity claim)
- codex 两轮 review(MINOR_FIX→MINOR_FIX→SHIP_IT):① scale-aware init/bounds(per cell 按
  n/trace(K_j) 缩放,不动冻结 core)防 `none` 伪 drift ② `reference`/`n_starts` 严格整数校验
  ③ `genetic_upper_bound_hit` 诊断列按实际 bounds 判定
- 产物函数:`pve_sensitivity_grid` → `SensitivityGridResult(.table 12行 / .drift_table / .reference_cell)`

### M2.4.2 Horvath 端到端脚本(codex SHIP_IT)
- `scripts/phase2/run_m2_4_2_horvath_diagnostics.py` 整合 Step 2-5,产物落 `results/phase2/m2_4_2/horvath2020/`
- 完整 run `--B 200 --n-jobs 16`:**23/23 acceptance check PASS**,runtime 183s
- 产物:`nested_reml_` / `lrt_table_` / `bootstrap_lrt_table_` / `sensitivity_table_` /
  `sensitivity_drift_` `.tsv` + `m2_4_2_summary_plant_height.json` + 3 PNG
- bootstrap 5 contrasts success_rate 全 1.000;B=200 太小,强 contrast 的 bootstrap_p
  撞 floor 1/(B+1)≈0.005 → **paper run 用 `--B 1000`**
- kernel 纪律:Step 2-4 用 trace-normalized canonical,Step 5 用 RAW GRM(变量隔离,codex 确认无泄漏)

**M2.4.2 全部完成(Step 1-5 + 端到端脚本,全 codex SHIP_IT)。**

### M2.4.3 null-simulation calibration(codex SHIP_IT)
- 新模块 `src/homoeogwas/calibration.py`:`NullSimulationScenario` / `NullCalibrationResult` /
  `run_null_lrt_calibration` / `scenario_from_reduced_fit` / `summarize_type1` / chi-bar helpers
- 3 场景(σ² 取自 reduced model 实拟合):S1 全局 null(e→A+C+e df2)、S2 C-null-given-A
  (A+e→A+C+e df1)、S3 A-null-given-C(C+e→A+C+e df1)
- 校准 p_naive(χ²)/ p_mixture(Stram-Lee 边界混合)/ bootstrap_p(可选);指标 = type-I @ α
  + Wilson CI;弃 λ_GC 改 chi-bar `tail_inflation`;boundary p 有原子 → 不用 KS
- **S3 是 residual-boundary 退化 null**(plant_height 在 C+e 下 σ²_e 撞下界)→ 显式 warning +
  metadata 记录;empirical type-I 仍有效,Stram-Lee 解析混合标注为近似
- per-SNP QQ/λ_GC 推迟到 M2.5(尚无 per-SNP scan)
- 测试:`tests/test_calibration.py` 16 个(deterministic 逻辑 + 理论 chi-bar 已知良好 +
  n_jobs 不变性);**89/89 pytest 全过**
- 脚本 `scripts/phase2/run_m2_4_3_horvath_calibration.py`,产物 `results/phase2/m2_4_3/horvath2020/`

#### M2.4.3 实测结果(n_sim=1000,Horvath plant_height,9/9 acceptance,465s)
| 场景 | p_naive @0.05 | p_mixture @0.05 | tail_inflation q95 |
|---|---|---|---|
| S1 全局 null(df=2) | 0.009 conservative | 0.028 conservative | 0.73 |
| S2 C-null-given-A(df=1) | 0.015 conservative | 0.033 conservative | 0.83 |
| S3 A-null-given-C(df=1) | 0.016 conservative | 0.033 conservative | 0.73 |

**结论:三个 null 场景下 boundary-corrected LRT(p_mixture)全部 conservative,无一 anti-conservative**
—— 即使在相关核(corr 0.69)下。dual-plan 担心的"相关核 binomial chi-bar 反保守"未发生,实际轻度保守
(type-I ≈ 0.6× nominal,tail_inflation < 1)。M2.4.2 的边界 LRT **假阳受控**,是可发表的干净结论。
S3 为 residual-boundary 退化 null(e_boundary_rate 0.58),已显式标注、不影响 empirical type-I 有效性。

### M2.4.4 Hadamard kernel inclusion(codex SHIP_IT)
- 引擎 J-agnostic → 无新核心代码;补 `tests/test_j3_hardening.py`(8 测试:J=3 fit /
  8 models / 7 default pairs / 19 all_nested / bootstrap / sensitivity / hadamard 归一等价 /
  近共线不崩)→ **97/97 pytest**
- 脚本 `scripts/phase2/run_m2_4_4_horvath_hadamard.py`,产物 `results/phase2/m2_4_4/horvath2020/`
- 实跑 16/16 acceptance,215s

#### M2.4.4 实测结果(Horvath plant_height,n=380)
| 量 | 值 |
|---|---|
| full A+C+hom+e PVE | A=0.344 / C=0.656 / **hom=0.000** / e=0 |
| headline `A+C+e→A+C+hom+e` | T=0,p_boundary=1,**bootstrap_p=1** |
| `e→hom+e`(K_hom 单独) | T=45.9,p=6.3e-12(K_hom 自身有信号)|
| kernel_corr A-hom / C-hom | 0.40 / 0.42;design_cond=11.3(interpretable)|
| multi-start | n_starts=1 全 cell 与 50 一致(canonical);sensitivity 单个 (zscore,trace,n_starts=1) cell 撞局部最优 → per-axis drift 全 ~1e-6 |

**结论:K_hom(homoeolog Hadamard 核)在 additive A+C 之上无可检测增量** —— K_hom 单独携带表型结构
(e→hom+e 显著)但与 additive A+C 完全冗余。codex framing:这不证伪 Core claim,只说明
plant_height/Horvath 上 additive A/C 协方差已吸收全部信号,需跨 trait / wheat A+B+D pilot
继续验证。**论文 Core "Hadamard 核" claim 需在 K_hom 真正增益的 trait/panel 上展示。**

### M2.5-v1 GPU per-SNP 关联扫描(codex SHIP_IT)
- 新模块 `src/homoeogwas/scan.py`:EMMAX/P3D 式固定-V 扫描。`build_scan_context` 构投影
  P = V⁻¹−V⁻¹X(X'V⁻¹X)⁻¹X'V⁻¹;`scan_snps` 批量 `P@G` GEMM,CPU(numpy)+ 单 GPU(torch)backend
- `tests/test_scan.py` 14 测试(单 SNP vs 显式 GLS Wald 等价、translation/scaling 不变性、
  缺失/QC、CPU batch==loop、sample alignment、GPU==CPU);**110/110 pytest**
- 脚本 `scripts/phase2/run_m2_5_horvath_scan.py`,产物 `results/phase2/m2_5/horvath2020/`

#### M2.5-v1 实测结果(Horvath plant_height,n=380,GPU backend,15s)
| 量 | 值 |
|---|---|
| genome-wide REML | σ²_A=6.84 / σ²_C=13.07 / σ²_e=0(PVE 0.34/0.66/0)|
| 扫描 marker | 48002 kept(A 17698 + C 30304;低 MAF 过滤 3897)|
| **λ_GC** | **0.9635**(λ_A=0.98 / λ_C=0.95)—— 亚基因组分层 LMM 良好控制基因组膨胀,无 inflation |
| GPU==CPU | max\|Δχ²\| = 2.5e-14 |
| top hit | chrA05:7007244,p=2.5e-6(n=380 dev anchor,无 5e-8 显著位点,符合预期)|

**→ 满足 Phase 2 退出硬指标"Horvath end-to-end QQ / λ_GC 报告"。** proximal contamination
v1 不校正(标准 EMMAX);LOCO 推 v2。

### M2.4.5 wheat Watkins A+B+D pilot — 规模锚点(codex SHIP_IT)
- wheat closeout 完成(A 34.17M / B 43.45M / D 8.95M post-QC SNP,1051 sample)
- 脚本 `scripts/phase2/run_m2_4_5_wheat_pilot.py`(在 polygwas-gpu env 跑):
  preflight → 三亚基因组 psam ∩ pheno → plink2 LD-prune → `compute_grm` → A/B/D + K_hom 核
  → REML(J=3 additive 8 模型 + J=4 Hadamard 16 模型)→ boundary LRT → GPU canary → profiling
- pruned-GRM 方案:plink2 `--indep-pairwise 1000kb 1 0.2` → A 113871 / B 123920 / D 108295
  pruned SNP → `compute_grm`(m_used ~90-95k/亚基因组),复用 Horvath 同款 VanRaden 约定

#### M2.4.5 实测结果(wheat days_to_emerg,n=827 WATDE,26/26 acceptance)
| 量 | 值 |
|---|---|
| J=3 additive PVE | A=0.044 / **B=0.358** / D=0.134 / e=0.464(B 亚基因组主导 heading date)|
| J=4(+K_hom)PVE | A=0.040 / B=0.328 / D=0.088 / hom=0.125 / e=0.419;design_cond=8.93 |
| K_hom headline LRT | `A+B+D+e→A+B+D+hom+e` T=1.37,p_boundary=0.121 → **无显著增量**(同 M2.4.4)|
| GPU canary | D 88987 marker,backend=gpu,λ_GC=0.9145,**peak VRAM 0.33 GB** |
| **runtime / RAM / VRAM 表** | 总 **557s** / peak RSS **2.9 GB** / peak VRAM **0.33 GB**(stage_profile.tsv)|

stage 明细:prune A 93s / B 129s / D 26s,compute_grm 25s,REML J3 81s / J4 190s,canary 11s。
**→ 满足 Phase 2 退出硬指标"wheat Watkins end-to-end runtime/peak RAM/peak VRAM 表格"。**
GRM/REML 全 CPU(n=827 小问题,GPU 无收益);GPU 只在 canary 阶段。known QTL 回收推 M2.5-v2。

### M2.5-v2 wheat 全量 2-GPU per-SNP 扫描(codex SHIP_IT)
- **v2a 流式引擎**:`io.iter_bed_chunks`(bed_reader variant 切片分块读)+ `scan.scan_bed_stream`
  (流式拉块 → P@G → 增量写 gzip TSV,不进内存)+ `StreamScanSummary`;test_scan.py 加 5 个
  流式测试(流式==in-memory 等价 / chunk_size 不变 / gzip / sample_ids guard);codex
  MINOR_FIX(context.sample_ids 唯一性)→ 修
- **v2b 生产脚本** `scripts/phase2/run_m2_5_v2_wheat_scan.py`:4 模式(build/worker/
  postprocess/run);2-GPU 数据并行 = 两独立进程 GPU0=A,D / GPU1=B,无 NCCL;codex SHIP_IT
- 前置:wheat pgen→bed 转换(A 7.1G/B 9.0G/D 1.9G);V/P 复用 M2.4.5 additive A+B+D REML
  σ²(标准 EMMAX 全局 P);LOCO + known QTL 推 Phase 3

#### M2.5-v2 实测结果(wheat days_to_emerg,2× RTX 3080,~90min)
| 亚基因组 | markers | λ_GC | worker runtime |
|---|---|---|---|
| A | 33,049,044 | 1.078 | 4091s(GPU0)|
| B | 41,895,946 | 0.872 | 5048s(GPU1)|
| D | 8,351,057 | 1.027 | 920s(GPU0)|
| **全部** | **83,296,047** | **0.9661** | 总 5385s(双 GPU 并行)|

**→ 真正的六倍体全基因组 GWAS 扫完(83.3M marker)**,overall λ_GC=0.966 —— 亚基因组分层
LMM 在 wheat 全量扫描上良好控制基因组膨胀。产物 `results/phase2/m2_5_v2/wheat_watkins/`
(scan_{A,B,D}.tsv.gz 3.7G + Manhattan/QQ/lambda_gc.tsv)。双 GPU 原生栈(工程贡献)实证。

### M2.6 Simulation benchmark — Phase 2 退出硬指标(codex 3 轮收敛后达标)

- 新模块 `src/homoeogwas/sim.py`(纯函数:`SimArm`/`CausalSet`/`SimPhenotype`、
  `standardize_dosage`、`select_causal_snps`、`simulate_phenotype`(per-subgenome
  random-effect 生成模型 + 等幅 QTL 注入)、`build_truth_windows`(物理 ±500kb ∪
  LD r²≥0.2)、`evaluate_threshold`(discovery-locus 折叠 + 重复 locus 计 FP)、
  `power_fdr_curve`/`power_at_fdr`/`partial_auc`、`paired_bootstrap`、`holm_correct`)
- 脚本 `scripts/phase2/run_m2_6_horvath_sim_benchmark.py`(6 subcommand:prepare /
  simulate / run-homoeo / run-baselines / summarize / run);全 CPU(线程 pin=1,
  runtime 公平);merged A+C BED 给 GEMMA/regenie,共同 marker universe(MAF≥0.01 /
  call_rate≥0.95,51914 marker,所有方法 BH 分母一致)
- `tests/test_sim.py` 34 测试 → **151/151 pytest 全过**
- 6 arm:S1 C 主导 / S2 A 主导(mirror)/ S3 pooled-truth / S4 balanced /
  S5 null / S6 hadamard stress;方法:HomoeoGWAS(A+C 多核)+ pooled-GRM EMMAX
  (ablation)+ GEMMA + regenie。预注册参数:q=4 causal/亚基因组,h_qtl=0.18(等幅,
  per-SNP h²≈0.045,pilot 测得 χ²≈400·h²_snp → 信息量 mid-power)
- codex review **3 轮**:NEEDS_FIX(12 issue)→ NEEDS_FIX(6 issue)→ MINOR_FIX(1)
  → 全部修复(thread-pin 强制、both-arms exit gate、strict marker-universe、
  realized-PVE、sample-order 硬断言、duplicate-locus FP、`_join_meta` 断言、
  qc_params 缓存校验等)

#### M2.6 实测结果(Horvath2020,production R=100×S1/S2/S3 + R=50×S4/S5/S6,450 rep,22.5min,10/10 acceptance)

| arm | HomoeoGWAS | pooled-EMMAX | GEMMA | regenie |
|---|---|---|---|---|
| S1 BH-power / BH-FDP | 0.325 / **0.060** | 0.340 / 0.087 | 0.371 / 0.095 | 0.289 / 0.185 |
| S1 power@FDR≤.05 / pAUC | **0.266 / 0.266** | 0.250 / 0.239 | 0.266 / 0.253 | 0.184 / 0.144 |
| S2 power@FDR≤.05 / pAUC | **0.175 / 0.171** | 0.139 / 0.156 | 0.116 / 0.136 | 0.086 / 0.099 |
| S5 null BH-FDP | 0.040 | 0.040 | 0.040 | 0.020 |
| **runtime/rep(单线程)** | **6.6s** | 6.0s | 21.9s | 101.6s |

- **runtime win**:HomoeoGWAS 中位 6.6s vs GEMMA 21.9s(**3.3×**)vs regenie 101.6s
  (**15.5×**),paired bootstrap p=0.0002,S1+S2 双臂 → **exit gate PASS(runtime)**
- **power**:per-SNP BH-power 上 HomoeoGWAS ≈ pooled-EMMAX(Δ≈−0.01,亚基因组分层对
  per-SNP 扫描无增量,与 M2.4.4/M2.4.5 一致);raw BH-power 略低于 GEMMA(Δ≈−0.04)但
  **GEMMA/regenie BH-FDP 更高(反保守)**——matched-FDR 的 power@FDR≤0.05 / pAUC 上
  HomoeoGWAS ≥ GEMMA、远胜 regenie。HomoeoGWAS 在 LMM 方法中**校准最好**(BH-FDP 最低)
- **诚实性 arm**:S3 pooled-truth HomoeoGWAS 仍有竞争力(power@FDR .240 vs GEMMA .181);
  S4 balanced 持平;S5 null 三方法 FDP 0.04 受控;S6 K_hom 无特殊增益(同 M2.4.4)
- 产物 `results/phase2/m2_6_sim/horvath2020/`(summary.json + metrics/{power_fdr_curve,
  per_replicate,method_scores,paired_tests}.tsv + 11 figures)

**结论:Phase 2 退出硬指标「Simulation benchmark」达成** —— power-vs-FDR 曲线 vs 2 主桶
baseline 完成,HomoeoGWAS 在 runtime 维度对 GEMMA+regenie 统计显著优胜(≥2 baseline)。
诚实结论:亚基因组分层 LMM 的 per-SNP 扫描 power 与 pooled GRM 持平,卖点是
**校准 + 速度 + 亚基因组分辨的方差分量推断(M2.4 LRT/PVE)**,非 raw GWAS power。

### M2.7 CLI `homoeogwas fit` — Phase 2 退出最后一项硬指标(达成)

- 新模块 `src/homoeogwas/cli.py` + `src/homoeogwas/__main__.py`(`python -m homoeogwas`
  fallback);`pyproject.toml`(`homoeogwas` console_scripts entry point,src layout,
  PyYAML 等依赖)。`pip install -e .` 后 `homoeogwas` 上 PATH
- `homoeogwas fit --config X.yaml [--out-dir] [--backend] [--dry-run] [--force]`
  —— **正好 5 flag**(charter §3.3 ≤5)。trait/QC/kernel/scan 全进 YAML
- YAML run-config(`configs/runs/fit_*.yaml`),可 `panel_manifest:` 引用冻结的
  `configs/panels/*.yaml` 取 subgenomes 默认值(冻结 manifest 不动)
- 流程泛化 `run_m2_5_horvath_scan.py` 到 J 亚基因组:GRM(npz 缓存或 BED 现算)→
  `normalize_kernel`[+ 可选 K_hom]→ `fit_multi_reml` → `build_scan_context` →
  `scan_snps`(in-memory)/ `scan_bed_stream`(wheat-scale 流式,marker 数启发式 auto)
  → sumstats + QQ + Manhattan + λ_GC + summary JSON。引擎模块零改动
- `scan_summary` bounded-memory:χ² 流入单个预分配数组 + int16 亚基因组码,plot frame
  thin 到 ~200k + 所有 p≤1e-3 强信号 —— wheat 83M marker 不 OOM
- `tests/test_cli.py` 14 测试(flag budget / config 解析+校验 / dry-run / in-memory +
  streaming + J=3 Hadamard 端到端合成 panel / force / 错误处理)→ **165/165 pytest**
- codex review:NEEDS_FIX(streaming OOM 等 4)→ MINOR_FIX(3)→ 全修

#### M2.7 实测(Horvath2020 plant_height,`homoeogwas fit` 端到端)
- `homoeogwas fit --config configs/runs/fit_horvath2020_plant_height.yaml --backend cpu`
- n=380,REML σ²_A=6.84/σ²_C=13.07(PVE 0.34/0.66),51914→48002 marker,
  **λ_GC=0.9635**,6/6 acceptance,11.6s —— 与 M2.5-v1 结果**逐位一致**(CLI 是
  M2.5 流程的忠实泛化封装)。产物 `results/phase2/cli/horvath2020/`

**结论:Phase 2 退出最后一项硬指标「CLI homoeogwas fit 跑通」达成。Phase 2 全部退出。**

## 下一步(Phase 3)

1. LOCO(leave-one-chromosome-out)校正 proximal contamination
2. wheat known QTL 回收(Ppd-D1 / Ppd-B1 / Vrn)真实数据找新位点
3. v1.5 extension:PlantCaduceus + AgroNT 零样本变异先验(Core 校准过关后启动)

## 变更日志

| 日期 | 变更 |
|---|---|
| 2026-05-21 | M2.1-M2.3 完成 + M2.4.1 + M2.4.2 Step 1-3 |
| 2026-05-22 | M2.4.2 Step 4 bootstrap 完成,59/59 pytest |
| 2026-05-22 | M2.4.2 Step 4 codex review NEEDS_FIX → 3 major+7 minor 全修 → 复审 SHIP_IT,65/65 pytest |
| 2026-05-22 | M2.4.2 Step 5 sensitivity grid:dual-plan(Claude+Codex)→ 实施 → codex MINOR_FIX×2 全修 → SHIP_IT,73/73 pytest |
| 2026-05-22 | M2.4.2 Horvath 端到端脚本:dual-plan → 实施 → codex SHIP_IT;23/23 acceptance。**M2.4.2 里程碑完成** |
| 2026-05-22 | M2.4.3 null-simulation calibration:dual-plan → 实施(calibration.py + 16 测试 + Horvath 脚本)→ codex MINOR_FIX 全修 → SHIP_IT,89/89 pytest。**M2.4.3 里程碑完成** |
| 2026-05-22 | M2.4.4 Hadamard kernel inclusion:dual-plan → 实施(test_j3_hardening.py 8 测试 + Horvath 脚本,无核心代码改动)→ codex MINOR_FIX(产物过时,重跑)→ SHIP_IT,97/97 pytest,16/16 acceptance。**M2.4.4 里程碑完成** |
| 2026-05-22 | M2.5-v1 GPU per-SNP 扫描:dual-plan → 实施(scan.py + test_scan.py 14 测试 + Horvath 脚本)→ codex SHIP_IT + 2 hardening guard,110/110 pytest,7/7 acceptance,λ_GC=0.96。**M2.5-v1 完成,Phase 2 QQ/λ_GC 硬指标达成** |
| 2026-05-22 | M2.4.5 wheat Watkins A+B+D pilot:wheat closeout 完成 → dual-plan → 实施(run_m2_4_5_wheat_pilot.py + J=4 引擎测试)→ codex MINOR_FIX(preflight + 4 收敛 gate)→ 复审 SHIP_IT,112/112 pytest,26/26 acceptance,总 557s。**M2.4.5 完成,Phase 2 wheat 规模锚点硬指标达成** |
| 2026-05-22 | M2.5-v2 wheat 全量 2-GPU 扫描:dual-plan → v2a 流式引擎(scan_bed_stream + iter_bed_chunks,codex MINOR_FIX→修)→ v2b 生产脚本(codex SHIP_IT)→ canary D PASS → 全量 A+B+D run。117/117 pytest,83.3M marker,λ_GC=0.966,~90min 双 GPU。**M2.5-v2 完成,六倍体全基因组 GWAS 扫通** |
| 2026-05-22 | M2.6 Simulation benchmark:dual-plan(Claude+Codex)→ 实施(sim.py + run_m2_6 脚本 + test_sim.py 34 测试)→ codex review 3 轮(NEEDS_FIX 12 → NEEDS_FIX 6 → MINOR_FIX 1,全修)→ production 450 rep。151/151 pytest,10/10 acceptance,runtime 3.3×(GEMMA)/15.5×(regenie),exit gate PASS(runtime,S1+S2)。**M2.6 完成,Phase 2 退出硬指标「Simulation benchmark」达成** |
| 2026-05-22 | M2.7 CLI `homoeogwas fit`:dual-plan(Claude+Codex)→ 实施(cli.py + __main__.py + pyproject.toml + 2 run-config + test_cli.py 14 测试)→ codex review 2 轮(NEEDS_FIX 4 → MINOR_FIX 3,全修)→ Horvath 端到端 6/6 acceptance(λ_GC=0.9635,与 M2.5-v1 一致)。165/165 pytest。**M2.7 完成,Phase 2 退出硬指标「CLI ≤5 flag」达成 —— Phase 2 全部退出** |
