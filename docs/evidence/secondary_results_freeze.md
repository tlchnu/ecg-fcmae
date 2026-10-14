# Secondary Results Freeze

> 初次冻结：2026-08-22；架构组件消融补齐：2026-09-02  
> 冻结状态：**PASS**  
> 操作边界：本次只读检查既有结果；**未重新训练、未重新导出预测、未重新计算实验结果**。  
> 适用范围：模态/融合对照、架构组件消融、12 endpoint、少导联、亚组分析。Full/Strict 与低标注主冻结仍以 `docs/final_results_freeze.md` 为准。

## 1. 冻结结论

以下正式结果在服务器 272 的最终实验目录中实际存在，并通过结构与来源核验：

| 结果域 | 核验结论 | 已存在的正式证据 |
|---|---|---|
| Full / ECG-only / Concat / Tabular-only | PASS | 每配置 3 seeds、精确指标、val/test predictions、manifest；非 Full 配置有逐 seed 2,000 次 paired bootstrap |
| Strict / No ECA / GAP / Stage-4 only | PASS | 每配置 3 seeds、精确指标、val/test predictions、manifest、逐 seed 2,000 次 paired bootstrap |
| 渐进组件链 A–G | PASS | A–D 各有 3 seeds、精确指标、predictions、manifest、逐 seed 2,000 次 paired bootstrap；E/F/G 分别复用 ECG-only/Concat/Full 的唯一正式结果 |
| 12 endpoints | PASS | 3 seeds + ensemble 共 48 行逐标签结果；12 标签 × AUROC/AUPRC 共 24 行 ensemble 95% CI |
| Limb-6 / Precordial-6 / Anatomical-4 / Lead-I / Lead-II | PASS | 每配置 3 seeds、精确指标、val/test predictions、manifest、逐 seed 2,000 次 paired bootstrap |
| 年龄、性别、种族/族裔、临床场景亚组 | PASS | 16 个亚组、2 个模型、3 seeds，共 96 行逐 seed 指标；2,000 次患者级 bootstrap 与配对差值 CI |

结构校验器最终返回：`status=PASS`、`errors=[]`。Full 自身没有 `paired_bootstrap.csv` 是预期行为，因为它是配对参考；其绝对 95% CI 位于三 seed 统计目录。

## 2. 冻结协议与来源

- 服务器：`research server (connection details omitted)`（272）。
- 远端实验根：`/root/autodl-tmp/project_001_ecg_convnext_ssl/experiments/20260804_aug_jitter005_mask010_full_paper_v1`。
- 训练增强协议：`jitter0.05,mask0.1`，即标准差 0.05 的高斯噪声与随机置零 10% 时间点。
- 随机种子：`42`、`2026`、`3407`。
- 固定测试集：5,442 行、5,442 个唯一患者；固定验证集：4,626 行。
- 测试集 split SHA256：`8e4fa6df7f30565ea8aee4e06cb417637648bbce6f9d2a8c415d12b91c2faaba`。
- 验证集 split SHA256：`cf882636c4fcd7f94710c66c983d1a83626ecf1c50955e84357da511f5b3d34f1`。
- Primary endpoint：`shd_moderate_or_greater_flag`，输出索引 11。
- 三 seed 表中的 `mean ± SD` 是跨训练 seed 的均值与标准差；95% CI 是固定测试集上的患者级 bootstrap 区间，两者不可互换。
- 下列 paired delta 均定义为 `待比较模型/导联配置 − 同 seed Full`；负值表示低于 Full。

## 3. 模态与融合结果

### 3.1 三 seed 测试结果

| 模型 | SHD AUROC，mean ± SD | SHD AUPRC，mean ± SD | Macro-11 AUROC，mean ± SD | 状态 |
|---|---:|---:|---:|---|
| Full | 0.8521 ± 0.0008 | 0.8249 ± 0.0009 | 0.8374 ± 0.0020 | complete |
| ECG-only | 0.8466 ± 0.0006 | 0.8214 ± 0.0006 | 0.8297 ± 0.0025 | complete |
| Concat | 0.8505 ± 0.0008 | 0.8224 ± 0.0009 | 0.8348 ± 0.0002 | complete |
| Tabular-only | 0.7587 ± 0.0006 | 0.7002 ± 0.0018 | 0.7281 ± 0.0018 | complete |

### 3.2 与 Full 的逐 seed 配对 bootstrap

每行使用相同 seed 的 Full 预测作为参考，均为 2,000 次患者级 paired bootstrap。

| 模型 | Seed | Δ SHD AUROC（95% CI） | Δ SHD AUPRC（95% CI） |
|---|---:|---:|---:|
| ECG-only | 42 | -0.0054 (-0.0083, -0.0028) | -0.0021 (-0.0052, 0.0009) |
| ECG-only | 2026 | -0.0063 (-0.0093, -0.0033) | -0.0048 (-0.0084, -0.0013) |
| ECG-only | 3407 | -0.0046 (-0.0075, -0.0019) | -0.0035 (-0.0063, -0.0008) |
| Concat | 42 | -0.0016 (-0.0033, -0.0001) | -0.0007 (-0.0033, 0.0020) |
| Concat | 2026 | -0.0014 (-0.0031, 0.0003) | -0.0029 (-0.0060, 0.0001) |
| Concat | 3407 | -0.0015 (-0.0038, 0.0008) | -0.0038 (-0.0071, -0.0006) |
| Tabular-only | 42 | -0.0933 (-0.1043, -0.0824) | -0.1229 (-0.1395, -0.1072) |
| Tabular-only | 2026 | -0.0944 (-0.1052, -0.0839) | -0.1268 (-0.1424, -0.1109) |
| Tabular-only | 3407 | -0.0924 (-0.1029, -0.0818) | -0.1234 (-0.1384, -0.1075) |

解释边界：Concat 与 Full 的差异较小，且部分逐 seed CI 跨 0；不能仅凭点估计把 Concat 宣称为稳定劣于 Full。Tabular-only 明显低于 Full，说明 ECG 波形提供了主要判别信息，但这不等同于证明表格变量没有增量价值。

## 4. 架构组件消融

### 4.1 严格匹配消融的定义

Full 的正式结构为 `ECA + stage-3/4 attention pooling + FiLM + FCMAE(ECA)`。以下消融均使用十二导联、相同增强、相同数据划分和相同三个随机种子：

| 配置 | 相对 Full 的唯一变化 |
|---|---|
| Strict no-pretraining | 结构与 Full 完全相同，仅不加载预训练权重 |
| No ECA | 移除 ECA，其余结构与 Full 相同 |
| GAP | 用 GAP 替换 attention pooling，其余结构与 Full 相同 |
| Stage-4 only | 只使用 Stage-4 表征，其余结构与 Full 相同 |

Strict 是预训练效应锚点，不是网络组件删除；将其列在这里是为了保持相同结构下的因果对照边界。

### 4.2 严格匹配消融的三 seed 结果

| 配置 | SHD AUROC，mean ± SD | SHD AUPRC，mean ± SD | Macro-11 AUROC，mean ± SD | 状态 |
|---|---:|---:|---:|---|
| Full | 0.8521 ± 0.0008 | 0.8249 ± 0.0009 | 0.8374 ± 0.0020 | complete |
| Strict no-pretraining | 0.8448 ± 0.0001 | 0.8149 ± 0.0002 | 0.8258 ± 0.0032 | complete |
| No ECA | 0.8519 ± 0.0003 | 0.8262 ± 0.0014 | 0.8366 ± 0.0003 | complete |
| GAP | 0.8517 ± 0.0004 | 0.8240 ± 0.0012 | 0.8366 ± 0.0019 | complete |
| Stage-4 only | 0.8517 ± 0.0005 | 0.8253 ± 0.0017 | 0.8379 ± 0.0007 | complete |

### 4.3 严格匹配消融与 Full 的逐 seed 配对 bootstrap

差值均为 `消融配置 − 同 seed Full`，每行 2,000 次患者级 paired bootstrap。

| 配置 | Seed | Δ SHD AUROC（95% CI） | Δ SHD AUPRC（95% CI） |
|---|---:|---:|---:|
| Strict no-pretraining | 42 | -0.0078 (-0.0111, -0.0045) | -0.0088 (-0.0136, -0.0038) |
| Strict no-pretraining | 2026 | -0.0076 (-0.0112, -0.0041) | -0.0106 (-0.0156, -0.0056) |
| Strict no-pretraining | 3407 | -0.0065 (-0.0097, -0.0034) | -0.0101 (-0.0152, -0.0053) |
| No ECA | 42 | -0.0008 (-0.0025, 0.0009) | 0.0013 (-0.0016, 0.0042) |
| No ECA | 2026 | -0.0002 (-0.0020, 0.0016) | 0.0022 (-0.0004, 0.0051) |
| No ECA | 3407 | 0.0006 (-0.0014, 0.0025) | 0.0003 (-0.0027, 0.0034) |
| GAP | 42 | -0.0009 (-0.0021, 0.0003) | 0.0000 (-0.0019, 0.0020) |
| GAP | 2026 | -0.0003 (-0.0017, 0.0012) | -0.0002 (-0.0027, 0.0022) |
| GAP | 3407 | 0.0003 (-0.0010, 0.0015) | -0.0023 (-0.0051, 0.0001) |
| Stage-4 only | 42 | -0.0006 (-0.0020, 0.0007) | 0.0000 (-0.0021, 0.0020) |
| Stage-4 only | 2026 | -0.0003 (-0.0018, 0.0012) | 0.0016 (-0.0006, 0.0039) |
| Stage-4 only | 3407 | 0.0000 (-0.0011, 0.0010) | -0.0003 (-0.0019, 0.0012) |

Strict 在三个 seed 上的 SHD AUROC 与 AUPRC 配对区间均低于 0，支持相同结构下 FCMAE(ECA) 预训练的增益。No ECA、GAP、Stage-4 only 的所有逐 seed AUROC/AUPRC 区间均包含 0；本结果不能证明它们与 Full 等效，也不能声称 ECA、attention pooling 或 stage-3/4 聚合各自带来稳定显著增益。

### 4.4 论文 A–G 渐进组件链

| 组 | 初始化 | ECA | 池化/尺度 | 表格融合 | 正式结果来源 |
|---|---|---|---|---|---|
| A | Scratch | Off | Stage-4 GAP | None | `component_A` |
| B | FCMAE no-ECA checkpoint | Off | Stage-4 GAP | None | `component_B` |
| C | FCMAE(ECA) | On | Stage-4 GAP | None | `component_C` |
| D | FCMAE(ECA) | On | Stage-4 attention | None | `component_D` |
| E | FCMAE(ECA) | On | Stage-3/4 attention | None | 复用 ECG-only |
| F | FCMAE(ECA) | On | Stage-3/4 attention | Concat | 复用 Concat |
| G | FCMAE(ECA) | On | Stage-3/4 attention | FiLM | 复用 Full |

E/F/G 是唯一正式结果的别名，不是额外训练；因此不能把它们重复计入实验运行数。

| 组 | SHD AUROC，mean ± SD | SHD AUPRC，mean ± SD | Macro-11 AUROC，mean ± SD | 状态 |
|---|---:|---:|---:|---|
| A | 0.8390 ± 0.0021 | 0.8106 ± 0.0029 | 0.8236 ± 0.0032 | complete |
| B | 0.8459 ± 0.0008 | 0.8221 ± 0.0006 | 0.8290 ± 0.0011 | complete |
| C | 0.8466 ± 0.0003 | 0.8211 ± 0.0011 | 0.8299 ± 0.0006 | complete |
| D | 0.8470 ± 0.0006 | 0.8218 ± 0.0009 | 0.8307 ± 0.0005 | complete |
| E（ECG-only） | 0.8466 ± 0.0006 | 0.8214 ± 0.0006 | 0.8297 ± 0.0025 | complete / reused |
| F（Concat） | 0.8505 ± 0.0008 | 0.8224 ± 0.0009 | 0.8348 ± 0.0002 | complete / reused |
| G（Full） | 0.8521 ± 0.0008 | 0.8249 ± 0.0009 | 0.8374 ± 0.0020 | complete / reused |

### 4.5 A–D 与 Full 的逐 seed 配对 bootstrap

| 组 | Seed | Δ SHD AUROC（95% CI） | Δ SHD AUPRC（95% CI） |
|---|---:|---:|---:|
| A | 42 | -0.0135 (-0.0175, -0.0093) | -0.0129 (-0.0184, -0.0073) |
| A | 2026 | -0.0156 (-0.0198, -0.0112) | -0.0176 (-0.0236, -0.0115) |
| A | 3407 | -0.0102 (-0.0142, -0.0059) | -0.0116 (-0.0167, -0.0064) |
| B | 42 | -0.0076 (-0.0110, -0.0043) | -0.0017 (-0.0056, 0.0026) |
| B | 2026 | -0.0063 (-0.0097, -0.0029) | -0.0040 (-0.0082, -0.0001) |
| B | 3407 | -0.0045 (-0.0078, -0.0013) | -0.0027 (-0.0065, 0.0010) |
| C | 42 | -0.0062 (-0.0093, -0.0032) | -0.0037 (-0.0073, -0.0003) |
| C | 2026 | -0.0054 (-0.0082, -0.0027) | -0.0032 (-0.0062, -0.0001) |
| C | 3407 | -0.0046 (-0.0081, -0.0013) | -0.0045 (-0.0086, -0.0006) |
| D | 42 | -0.0061 (-0.0090, -0.0033) | -0.0022 (-0.0055, 0.0009) |
| D | 2026 | -0.0057 (-0.0086, -0.0029) | -0.0045 (-0.0077, -0.0014) |
| D | 3407 | -0.0034 (-0.0069, 0.0001) | -0.0025 (-0.0062, 0.0013) |

A–G 是累积配置链；当前正式 bootstrap 只比较各配置与同 seed Full，并未计算 A→B、B→C 等相邻步骤的配对区间。因此可以描述完整链的点估计变化，但不能把相邻点估计差直接解释为某个单独组件的显著边际贡献。E/F/G 相对 Full 的配对结果见第 3.2 节。

## 5. Full ensemble 的 12 endpoint 结果

三 seed/ensemble 明细表实际包含 48 行，即 `12 labels × (42, 2026, 3407, ensemble)`。下表冻结 ensemble 点估计及 2,000 次患者级 bootstrap 95% CI。

| Index | Endpoint | AUROC（95% CI） | AUPRC（95% CI） |
|---:|---|---:|---:|
| 0 | `lvef_lte_45_flag` | 0.9081 (0.8973, 0.9189) | 0.7470 (0.7194, 0.7728) |
| 1 | `lvwt_gte_13_flag` | 0.7775 (0.7628, 0.7921) | 0.4508 (0.4207, 0.4821) |
| 2 | `aortic_stenosis_moderate_or_greater_flag` | 0.8895 (0.8734, 0.9044) | 0.3146 (0.2665, 0.3719) |
| 3 | `aortic_regurgitation_moderate_or_greater_flag` | 0.7717 (0.7208, 0.8196) | 0.0514 (0.0301, 0.1047) |
| 4 | `mitral_regurgitation_moderate_or_greater_flag` | 0.8410 (0.8190, 0.8621) | 0.3015 (0.2580, 0.3554) |
| 5 | `tricuspid_regurgitation_moderate_or_greater_flag` | 0.8645 (0.8458, 0.8816) | 0.3378 (0.2930, 0.3906) |
| 6 | `pulmonary_regurgitation_moderate_or_greater_flag` | 0.8793 (0.7988, 0.9474) | 0.1759 (0.0243, 0.3579) |
| 7 | `rv_systolic_dysfunction_moderate_or_greater_flag` | 0.8995 (0.8838, 0.9145) | 0.5157 (0.4656, 0.5664) |
| 8 | `pericardial_effusion_moderate_large_flag` | 0.7752 (0.7221, 0.8258) | 0.0530 (0.0337, 0.0945) |
| 9 | `pasp_gte_45_flag` | 0.8162 (0.8004, 0.8326) | 0.4233 (0.3839, 0.4608) |
| 10 | `tr_max_gte_32_flag` | 0.8098 (0.7900, 0.8298) | 0.2728 (0.2311, 0.3216) |
| 11 | `shd_moderate_or_greater_flag` | 0.8529 (0.8431, 0.8627) | 0.8263 (0.8120, 0.8403) |

### 5.1 Ensemble 聚合指标

| 指标 | 点估计 | 95% CI |
|---|---:|---:|
| Primary SHD AUROC | 0.8529 | 0.8431–0.8627 |
| Primary SHD AUPRC | 0.8263 | 0.8120–0.8403 |
| Macro-11 AUROC | 0.8393 | 0.8279–0.8504 |
| Macro-12 AUROC | 0.8404 | 0.8296–0.8510 |
| Macro-12 AUPRC | 0.3725 | 0.3535–0.3976 |

注意：部分低患病率 endpoint 的 AUPRC 较低且区间较宽，尤其肺动脉瓣返流、主动脉瓣返流和心包积液；不得只依据 AUROC 把这些 endpoint 表述为已经获得稳定临床性能。

## 6. 少导联结果

### 6.1 配置与实现证据

少导联没有单独预训练。项目协议为从十二导联 FCMAE 预训练权重出发进行 stem-select；预测 manifest 均记录 `lead_input_mode=slice`。

| 配置 | 实际导联 | 索引 | Manifest setting |
|---|---|---|---|
| Limb-6 | I、II、III、aVR、aVL、aVF | `[0,1,2,3,4,5]` | `limb6` |
| Precordial-6 | V1–V6 | `[6,7,8,9,10,11]` | `precordial6` |
| Anatomical-4 | I、II、V1、V5 | `[0,1,6,10]` | `anatomical4` |
| Lead-I | I | `[0]` | `lead_i` |
| Lead-II | II | `[1]` | `lead_ii` |

本冻结中不存在也不声称存在 V1 单导联正式结果；正式少导联集合仅为上列五项。

### 6.2 三 seed 测试结果

| 配置 | SHD AUROC，mean ± SD | SHD AUPRC，mean ± SD | Macro-11 AUROC，mean ± SD | 状态 |
|---|---:|---:|---:|---|
| Limb-6 | 0.8322 ± 0.0016 | 0.8007 ± 0.0024 | 0.8087 ± 0.0043 | complete |
| Precordial-6 | 0.8463 ± 0.0003 | 0.8192 ± 0.0005 | 0.8306 ± 0.0011 | complete |
| Anatomical-4 | 0.8465 ± 0.0006 | 0.8208 ± 0.0013 | 0.8314 ± 0.0012 | complete |
| Lead-I | 0.8245 ± 0.0005 | 0.7962 ± 0.0002 | 0.7942 ± 0.0052 | complete |
| Lead-II | 0.8196 ± 0.0005 | 0.7783 ± 0.0005 | 0.7857 ± 0.0024 | complete |

### 6.3 与 Full 的逐 seed 配对 bootstrap

| 配置 | Seed | Δ SHD AUROC（95% CI） | Δ SHD AUPRC（95% CI） |
|---|---:|---:|---:|
| Limb-6 | 42 | -0.0219 (-0.0274, -0.0164) | -0.0254 (-0.0334, -0.0170) |
| Limb-6 | 2026 | -0.0203 (-0.0257, -0.0150) | -0.0249 (-0.0322, -0.0178) |
| Limb-6 | 3407 | -0.0173 (-0.0226, -0.0120) | -0.0217 (-0.0293, -0.0147) |
| Precordial-6 | 42 | -0.0060 (-0.0094, -0.0025) | -0.0042 (-0.0090, 0.0007) |
| Precordial-6 | 2026 | -0.0063 (-0.0100, -0.0024) | -0.0067 (-0.0127, -0.0009) |
| Precordial-6 | 3407 | -0.0052 (-0.0088, -0.0015) | -0.0063 (-0.0119, -0.0009) |
| Anatomical-4 | 42 | -0.0060 (-0.0092, -0.0028) | -0.0032 (-0.0076, 0.0011) |
| Anatomical-4 | 2026 | -0.0054 (-0.0088, -0.0021) | -0.0033 (-0.0076, 0.0011) |
| Anatomical-4 | 3407 | -0.0054 (-0.0086, -0.0022) | -0.0058 (-0.0101, -0.0013) |
| Lead-I | 42 | -0.0288 (-0.0350, -0.0221) | -0.0280 (-0.0372, -0.0184) |
| Lead-I | 2026 | -0.0275 (-0.0339, -0.0208) | -0.0291 (-0.0383, -0.0203) |
| Lead-I | 3407 | -0.0267 (-0.0331, -0.0206) | -0.0291 (-0.0377, -0.0204) |
| Lead-II | 42 | -0.0335 (-0.0406, -0.0263) | -0.0457 (-0.0566, -0.0354) |
| Lead-II | 2026 | -0.0325 (-0.0397, -0.0254) | -0.0464 (-0.0582, -0.0352) |
| Lead-II | 3407 | -0.0313 (-0.0385, -0.0240) | -0.0472 (-0.0582, -0.0367) |

所有少导联配置的 SHD AUROC 配对区间均低于 0。Precordial-6 与 Anatomical-4 的损失最小；这只能支持在本固定测试集上的相对性能描述，不能等同于少导联与十二导联临床等效。

## 7. 亚组分析

### 7.1 完整性

- 正式目录：`07_subgroups_three_seed_v1`。
- 状态：`complete`。
- 模型：同结构 Scratch 与 FCMAE(ECA) Full。
- Seeds：`42`、`2026`、`3407`。
- 16 个亚组 × 2 个模型 × 3 seeds = 96 行逐 seed 指标。
- 5,442 个测试患者，每位患者一行。
- 2,000 次患者级 bootstrap；两个模型使用相同重采样索引。
- 分组维度：年龄、性别、种族/族裔、临床场景。

### 7.2 冻结主表

| 类别 | 亚组 | n / positive n | 患病率 | Scratch AUROC（95% CI） | Full AUROC（95% CI） | Full AUPRC（95% CI） | Δ AUROC（Full − Scratch，95% CI） |
|---|---|---:|---:|---:|---:|---:|---:|
| Age | 18–59 | 2124 / 671 | 0.316 | 0.856 (0.839, 0.873) | 0.861 (0.844, 0.878) | 0.778 (0.746, 0.806) | 0.0048 (-0.0003, 0.0096) |
| Age | 60–69 | 1318 / 524 | 0.398 | 0.839 (0.817, 0.860) | 0.848 (0.826, 0.868) | 0.808 (0.776, 0.838) | 0.0086 (0.0025, 0.0145) |
| Age | 70–79 | 1154 / 578 | 0.501 | 0.801 (0.776, 0.825) | 0.815 (0.789, 0.838) | 0.838 (0.810, 0.863) | 0.0134 (0.0066, 0.0204) |
| Age | 80+ | 846 / 545 | 0.644 | 0.786 (0.754, 0.816) | 0.789 (0.757, 0.818) | 0.878 (0.853, 0.901) | 0.0029 (-0.0064, 0.0125) |
| Sex | Female | 2731 / 1001 | 0.367 | 0.847 (0.832, 0.862) | 0.852 (0.837, 0.866) | 0.789 (0.764, 0.812) | 0.0049 (0.0007, 0.0090) |
| Sex | Male | 2711 / 1317 | 0.486 | 0.841 (0.826, 0.855) | 0.848 (0.834, 0.863) | 0.852 (0.833, 0.869) | 0.0077 (0.0038, 0.0116) |
| Race/Ethnicity | Hispanic | 1649 / 620 | 0.376 | 0.847 (0.828, 0.866) | 0.854 (0.835, 0.871) | 0.805 (0.777, 0.832) | 0.0064 (0.0015, 0.0117) |
| Race/Ethnicity | White | 1569 / 656 | 0.418 | 0.837 (0.818, 0.858) | 0.843 (0.824, 0.863) | 0.811 (0.783, 0.838) | 0.0061 (0.0005, 0.0115) |
| Race/Ethnicity | Black | 846 / 405 | 0.479 | 0.847 (0.820, 0.871) | 0.852 (0.825, 0.877) | 0.855 (0.825, 0.884) | 0.0055 (-0.0011, 0.0123) |
| Race/Ethnicity | Asian | 153 / 70 | 0.458 | 0.848 (0.784, 0.907) | 0.846 (0.782, 0.905) | 0.824 (0.731, 0.901) | -0.0022 (-0.0210, 0.0152) |
| Race/Ethnicity | Other | 457 / 200 | 0.438 | 0.848 (0.813, 0.882) | 0.853 (0.820, 0.886) | 0.830 (0.781, 0.878) | 0.0055 (-0.0043, 0.0161) |
| Race/Ethnicity | Unknown | 768 / 367 | 0.478 | 0.854 (0.827, 0.880) | 0.863 (0.836, 0.887) | 0.853 (0.816, 0.887) | 0.0085 (0.0011, 0.0158) |
| Clinical context | Emergency | 1971 / 788 | 0.400 | 0.851 (0.835, 0.868) | 0.858 (0.842, 0.874) | 0.812 (0.785, 0.838) | 0.0064 (0.0018, 0.0108) |
| Clinical context | Inpatient | 2203 / 1156 | 0.525 | 0.822 (0.804, 0.840) | 0.829 (0.812, 0.845) | 0.856 (0.837, 0.873) | 0.0066 (0.0014, 0.0118) |
| Clinical context | Outpatient | 1059 / 271 | 0.256 | 0.830 (0.800, 0.858) | 0.837 (0.806, 0.865) | 0.714 (0.662, 0.764) | 0.0068 (-0.0003, 0.0137) |
| Clinical context | Procedural | 209 / 103 | 0.493 | 0.838 (0.785, 0.888) | 0.850 (0.795, 0.897) | 0.837 (0.764, 0.900) | 0.0115 (-0.0047, 0.0284) |

Full 在 15/16 个亚组的 AUROC 点估计为正差；最大差值位于 70–79 岁组，最小且为负的差值位于 Asian 亚组。9 个亚组的配对 AUROC 95% CI 排除 0。Asian 与 Procedural 等小样本亚组区间明显更宽。

亚组结果仅描述固定测试集上的模型行为。年龄和性别同时是分组变量与两个匹配模型的 FiLM 输入，因此不能把差异解释为人口学变量的因果效应；区间重叠也不能证明公平性或临床等效性。本结果来自单中心数据，不属于外部验证。

## 8. Predictions 与 bootstrap 证据清单

### 8.1 Prediction 完整性

- 17 个独立正式配置均有 3 seeds × val/test = 6 个 `.npz`，以及对应 6 个 manifest：原冻结中的 9 个配置，加 Strict、No ECA、GAP、Stage-4 only 与 component A–D 共 8 个目录。
- 合计：102 个 prediction NPZ + 102 个 manifest。A–G 中的 E/F/G 只复用 ECG-only/Concat/Full，不重复计数。
- 每个测试 prediction 为 5,442 行；每个验证 prediction 为 4,626 行。
- Full、ECG-only、Concat、五个少导联、Strict、三个匹配组件消融与 component A–D 的 manifest 明确记录 `one_row_per_patient=true`。
- Tabular-only manifest 没有 `one_row_per_patient` 字段，但记录 `n_rows=5442`、`n_unique_patients=5442`，且 test/val split hash 与其他模型完全一致；其逐 seed paired bootstrap 也明确引用相同 seed 的 Full prediction。
- 原冻结 9 个配置中，按路径排序的 54 个 prediction NPZ 的逐文件 SHA256 清单集合哈希为：`fe29750c5a54f153ed1460302077ed327e08f03c95a5a0b9e6431809cb43deb4`。该哈希的作用域不含本次补齐的 8 个消融目录；本次通过每目录计数、manifest、split hash、checkpoint hash 和 bootstrap 原始数组哈希核验新增证据。

### 8.2 Bootstrap 原始数组

| 证据 | Bytes | SHA256 |
|---|---:|---|
| 12-endpoint ensemble bootstrap | 229,491 | `c79df6c27c8ca41a93a92d42e48ba7f3b14c7b928cd342eca37289f3fb186183` |
| 三 seed subgroup bootstrap | 1,387,851 | `6728a26715efe09e204a64d40bf13e16f8b23c7fabec96c41bc9382bd798f727` |
| ECG-only paired bootstrap | 257,444 | `d0eed4e56f7049bf0003523723e59753a22100fd65f201686a8b27f475173984` |
| Concat paired bootstrap | 257,413 | `e751b1a543636219c163bb56f1e67ecb5a8a2c91d5fca0b02caa306fd096965f` |
| Tabular-only paired bootstrap | 260,541 | `b0db67edf5e0c430150daacc67ff997e6013370762450249072eb1a2868a8904` |
| Limb-6 paired bootstrap | 258,640 | `02b00a646ce83e42dd76cccac1d40a0cc4f881eb1bc0543da52dbeead8e6d73f` |
| Precordial-6 paired bootstrap | 258,328 | `3b4d1babd199dfd1835398650795c5bc59a0f38f361151b3a71449dd77cdc9c2` |
| Anatomical-4 paired bootstrap | 257,982 | `b6e52021c2bc219d4e37f5f0a7bdda7650011634399ad7f60a2a5d2eab4be408` |
| Lead-I paired bootstrap | 259,254 | `966d6cf7d25758abfab8ffcb954567a43e7077d89c847449358ad171e29412a5` |
| Lead-II paired bootstrap | 259,710 | `0ca3c48f01a772fcfe3495f6d1e0bdc970083d39db646027c894eff54af6773c` |
| Strict no-pretraining paired bootstrap | 257,888 | `08c74a22e88a34042e8467d96a991f051cf86c5caa1897d094141d2017ff1f4c` |
| No ECA paired bootstrap | 257,833 | `0ecd039e97b69c4ca7461673923ab42a6082250e0d12c88e1791b42f828894bd` |
| GAP paired bootstrap | 257,503 | `d5840e039d3ea1f9cce2cfbf081d7b37ff4d508c295e82c42f18b1c3a90df9b8` |
| Stage-4 only paired bootstrap | 257,641 | `4e6ba97192da2b15812a06875940ed6eda0b990220b6be84ff086fac878062d4` |
| Component A paired bootstrap | 258,428 | `03d7e8d7d768c256a886b3ddf05c053d0307d97bc8a3c280325f928f73519bde` |
| Component B paired bootstrap | 258,095 | `d03dd5378ad66cac83346589dce3abda180aaa975c6ac4e3b9dacf695740c6cf` |
| Component C paired bootstrap | 257,611 | `f09b2d1b77acd454d4846682fc09b2d07570210e855ab531f096e376cdcade32` |
| Component D paired bootstrap | 257,757 | `882b35dfa2d5d26c95426e113c18ffcc716dbc4b39b7d1b3907cd280637d4763` |

## 9. 远端唯一来源文件

### 模态/融合、架构消融与少导联

每个配置目录均以以下文件为正式来源：

- `exact_seed_metrics.csv`
- `exact_summary.csv`
- `review_status.json`
- `predictions/seed_{42,2026,3407}_{val,test}.npz`
- 对应的 `.manifest.json`
- 非 Full 配置：`paired_bootstrap.csv` 与 `paired_bootstrap_raw.npz`

架构消融独立目录：

- `01_core/strict_no_pretraining`
- `01_core/no_eca`
- `01_core/gap`
- `01_core/stage4_only`
- `02_component_chain/component_A`
- `02_component_chain/component_B`
- `02_component_chain/component_C`
- `02_component_chain/component_D`

组件链 E/F/G 分别指向 `01_core/ecg_only`、`01_core/concat`、`01_core/full`，不另设重复来源目录。

### 12 endpoint

目录：`10_three_seed_paper_statistics_v1/`

- `status.json`
- `full_per_label_seed_and_ensemble.csv`
- `full_ensemble_per_label_95ci.csv`
- `full_ensemble_aggregate_95ci.csv`
- `bootstrap_raw.npz`
- `prediction_alignment_sha256.csv`

### 亚组

目录：`07_subgroups_three_seed_v1/`

- `status.json`
- `subgroup_metrics_by_seed.csv`
- `subgroup_auroc_forest_data.csv`
- `subgroup_delta_forest_data.csv`
- `subgroup_bootstrap_raw.npz`
- `subgroup_table_main.md`
- `subgroup_analysis_summary.md`
- `data_integrity_report.md`

## 10. 论文可用声明边界

1. 可以声明上述结果文件实际存在、三 seed 完成、predictions 对齐且 bootstrap 产物存在。
2. 可以报告 Full、模态/融合、少导联的三 seed `mean ± SD`。
3. 可以报告 12 endpoint ensemble 的患者级 bootstrap 95% CI。
4. 可以报告逐 seed 的“替代配置 − Full”配对差值及 CI，但不能把三个逐 seed CI 冒充一个跨 seed 联合 CI。
5. 可以报告亚组描述性结果与配对差值；不能据此宣称公平、因果人口学效应、外部泛化或临床等效。
6. AUPRC 强烈依赖患病率；跨 endpoint 或跨亚组比较时必须同时报告 prevalence。
7. 本冻结不包含 V1 单导联，不应在论文表格或正文中暗示 V1 已完成。
8. 可以声明严格同结构 Scratch 明显低于 Full；不能声明 No ECA、GAP、Stage-4 only 与 Full 等效，也不能声称三项组件分别产生显著增益，因为其逐 seed 配对区间均跨 0。
9. A–G 是累积组件链。除非另行生成相邻配置的配对统计，否则不得把 A→B、B→C 等点估计差写成单组件的显著边际贡献。
