# EchoNext ECG-SHD final results freeze gate

> 审计时间：2026-08-21 03:29–03:40（Asia/Shanghai）  
> 审计服务器：272  
> 正式实验根：`/root/autodl-tmp/project_001_ecg_convnext_ssl/experiments/20260804_aug_jitter005_mask010_full_paper_v1`  
> 正式协议：`jitter0.05,mask0.1`；seeds `{42,2026,3407}`  
> 本文状态：**PASS — 5/5 个关键门全部通过**

## 1. 冻结结论

| 关键门 | 状态 | 结论 |
|---|---|---|
| Full/Strict 三 seed | **PASS** | 两组均有 `{42,2026,3407}` 的唯一 checkpoint、精确逐 seed 指标和同结构模型配置；唯一设计差异为是否加载 FCMAE 预训练。 |
| 精确 predictions | **PASS** | Full/Strict 各 3 seeds、val/test 共 12 个 NPZ 全部存在；行数、唯一患者数、标签、概率范围、患者顺序和 split hash 均通过独立核验。 |
| Macro-11 | **PASS** | 已从 11 个子标签独立重算，明确排除 overall SHD；与 `exact_seed_metrics.csv` 一致到 `1e-12`。 |
| paired bootstrap | **PASS WITH CORRECTION** | ensemble 和逐 seed 均有 2,000 次患者级配对 bootstrap；原始分布的 2.5/97.5 分位数与 CSV 一致。逐 seed CSV 的 `delta_*` 列是 bootstrap 均值而非观测点差，论文不得把它误写为 point estimate。 |
| low-label provenance | **PASS** | 已在 272 完成 24 次训练，并验证 subset、checkpoint、predictions、精确指标、配对 bootstrap 与逐文件 SHA256。 |

因此：

- **Full/Strict 三种子、精确 predictions、Macro-11、paired bootstrap 和 low-label provenance 全部冻结。**
- **论文最终结果总门通过。**
- `THREE_SEED_PAPER_COMPLETE.json` 与 `LOW_LABEL_COMPLETE.json` 共同构成完整完成门。

## 2. Full 与 Strict 三种子冻结结果

### 2.1 Full

| seed | SHD AUROC | SHD AUPRC | Macro-11 AUROC | Macro-12 AUROC |
|---:|---:|---:|---:|---:|
| 42 | 0.8526174105 | 0.8238206938 | 0.8363438492 | 0.8376999793 |
| 2026 | 0.8524366451 | 0.8254730654 | 0.8361755733 | 0.8375306626 |
| 3407 | 0.8511373165 | 0.8253572149 | 0.8397214384 | 0.8406727616 |
| **mean ± sample SD** | **0.8520637907 ± 0.0008074248** | **0.8248836581 ± 0.0009223747** | **0.8374136203 ± 0.0020003993** | **0.8386344678 ± 0.0017672431** |

### 2.2 Strict no-pretraining

| seed | SHD AUROC | SHD AUPRC | Macro-11 AUROC | Macro-12 AUROC |
|---:|---:|---:|---:|---:|
| 42 | 0.8448135673 | 0.8148649935 | 0.8236324975 | 0.8253975867 |
| 2026 | 0.8448088721 | 0.8147845862 | 0.8244300488 | 0.8261282841 |
| 3407 | 0.8446637350 | 0.8151040645 | 0.8294456144 | 0.8307137911 |
| **mean ± sample SD** | **0.8447620581 ± 0.0000851827** | **0.8149178814 ± 0.0001661759** | **0.8258360536 ± 0.0031513043** | **0.8274132206 ± 0.0028816322** |

### 2.3 结构与训练匹配门

6 个 checkpoint 的 `model_config` 完全一致：

- input/output：12 ECG channels、7 tabular features、12 labels；
- depths `(3,3,9,3)`，dims `(64,128,256,512)`；
- kernel sizes `(7,7,7,7)`，drop path `0.1`；
- ECA on，Stage-3/4 multiscale on，attention pooling，FiLM；
- head dropout `0.2`，layer scale off。

6 个 checkpoint 的 training config 均记录 `layer_decay=0.75`。Full 为 `pretrained=True`，Strict 为 `pretrained=False`。Full 的有效 YAML 中 `llrd:false` 是 CLI 显式标志值，但训练代码在预训练条件下自动启用 LLRD；Strict 通过 `--llrd` 显式启用。因此两者实际均使用 layer decay `0.75`。

Checkpoint SHA256：

| model | seed | SHA256 |
|---|---:|---|
| Full | 42 | `e924631cd0515c8a0f0fbc205fb29c0cf5f2244c6bb42693b5748b6adfdab305` |
| Full | 2026 | `7ece4b0d98805a819546153dc1053aaf29f8497ea994697e7bfbe30054f4f74c` |
| Full | 3407 | `1c9cf5830c592f7a93ef975e28ffef31ff94e04b81d14b93d9de8631913885e7` |
| Strict | 42 | `296fe5609a6a9c308db1647001a72d315968fcac84f644f8d607a3d63cabacd6` |
| Strict | 2026 | `3507c4da84dc010985f173abb42b06b6f58a9c39000e64121a9f4a0da33dccc6` |
| Strict | 3407 | `9501cbfd62496b418559873d4cff5a1944c232840d912446a5f38441470e28b3` |

seed 42 checkpoint 是指向原 seed-42 正式目录的符号链接；SHA256 已按实际解析后的文件计算。2026/3407 为当前实验目录内的实体文件。

## 3. 精确 predictions 门

### 3.1 完整性与对齐

- val：每个 prediction 文件 `4,626 × 12`；4,626 个唯一患者。
- test：每个 prediction 文件 `5,442 × 12`；5,442 个唯一患者。
- Full/Strict、三个 seeds 的同一 split 均具有完全相同的患者顺序、标签矩阵和标签名称。
- 所有概率均为有限值且位于 `[0,1]`。
- val split hash：`cf882636c4fcd7f94710c66c983d1a83626ecf1c50955e84357da51f5b3d34f1`。
- test split hash：`8e4fa6df7f30565ea8aee4e06cb417637648bbce6f9d2a8c415d12b91c2faaba`。
- test manifests 均确认一名患者一行。

### 3.2 Test prediction SHA256

| model | seed | SHA256 |
|---|---:|---|
| Full | 42 | `419eafbced7a627411654c8bef816397c21c5c28ba00b5dcb2ae3bd3d5eeafab` |
| Full | 2026 | `e2974da3669bea36ea793b9d85faafa43da8ac1e1c383182d58e710dedd4231b` |
| Full | 3407 | `e1f0d5cd4145df178e784917b928c2e6cd60eb35716d9605cc5d81fa5145840a` |
| Strict | 42 | `fb98b7be55eed7310cb4defca1ae6eda30645d71b67e94308b7d4be5ededf8a0` |
| Strict | 2026 | `b194161da5f2dc154a2e5cddfa03dc2062f47d34f5d255ee0c43b2568679b83c` |
| Strict | 3407 | `7c4309b4c063aac01a137c84fdbfba9c7849dad33e451460a8b3b8b3e5301749` |

### 3.3 Validation prediction SHA256

| model | seed | SHA256 |
|---|---:|---|
| Full | 42 | `74dfaca4c5f9ae162c20157a5f8a6dee114441a73224f62d9c5d152fed9945bf` |
| Full | 2026 | `c38b852259e5a71fdfb82e03b76ac40866bf6721afba2f3934f2c146dfe6c804` |
| Full | 3407 | `6394ab40ea309c95e12eed9cc7ed392a9e50e7eeebc2c070a6f9f60a7acc73d0` |
| Strict | 42 | `3a011e966b16476e45521efaf69061830e6932dc7c3de6809fab4379eae8ba91` |
| Strict | 2026 | `609a73c2b065b687d27482e34a539716688d9cfb60169dc5ab5b887480c89df9` |
| Strict | 3407 | `7a75918e61048b1fc402f358caf0420c2e6a7e7739c737ad432825cc09d6e69d` |

## 4. Macro-11 门

Macro-11 定义为标签索引 `0..10` 的 AUROC 算术平均，明确排除索引 11 的 overall moderate+ SHD。独立从 6 个 test NPZ 重新计算后，与两个 `exact_seed_metrics.csv` 中的 `macro_auroc_11` 一致到 `1e-12`。

不得将以下三种量混写：

- Macro-11：11 个具体异常标签；
- Macro-12：11 个具体标签加 overall SHD；
- primary SHD AUROC：只评估 overall SHD 标签。

三种子 Full ensemble 的 Macro-11 AUROC 为 `0.8393075625`，95% CI `[0.8279316328,0.8504094675]`。

## 5. Paired bootstrap 门

### 5.1 Full 三种子 ensemble 自身 95% CI

ensemble 定义为同一患者、同一标签的三个 seed 概率算术平均。患者级 percentile bootstrap `n=2,000`。

| 指标 | point | 95% CI |
|---|---:|---:|
| SHD AUROC | 0.8529395843 | [0.8431269761, 0.8626985756] |
| SHD AUPRC | 0.8262992769 | [0.8120285214, 0.8402974828] |
| Macro-11 AUROC | 0.8393075625 | [0.8279316328, 0.8504094675] |
| Macro-12 AUROC | 0.8404435643 | [0.8295978847, 0.8510250629] |
| Macro-12 AUPRC | 0.3725103420 | [0.3534904146, 0.3975782021] |

### 5.2 Full ensemble 与 Strict ensemble 配对比较

同一 bootstrap replicate 对 Full 和 Strict 使用相同患者索引。

| 指标 | Full ensemble | Strict ensemble | Full − Strict | 配对 95% CI |
|---|---:|---:|---:|---:|
| SHD AUROC | 0.8529395843 | 0.8468389954 | +0.0061005889 | [0.0033656221, 0.0088181097] |
| SHD AUPRC | 0.8262992769 | 0.8181262821 | +0.0081729948 | [0.0042186436, 0.0122249057] |
| Macro-11 AUROC | 0.8393075625 | 0.8294075092 | +0.0099000533 | [0.0049089601, 0.0153233231] |
| Macro-12 AUROC | 0.8404435643 | 0.8308601330 | +0.0095834313 | [0.0048521001, 0.0145451936] |
| Macro-12 AUPRC | 0.3725103420 | 0.3527812341 | +0.0197291079 | [0.0088959075, 0.0302031062] |

### 5.3 逐 seed 配对结果

下表 point 使用 prediction 原始观测指标直接相减；CI 来自 2,000 次配对 bootstrap。

| seed | Full−Strict SHD AUROC | 95% CI | Full−Strict SHD AUPRC | 95% CI |
|---:|---:|---:|---:|---:|
| 42 | +0.0078038432 | [0.0044551541, 0.0110778415] | +0.0089557003 | [0.0038473036, 0.0135716040] |
| 2026 | +0.0076277731 | [0.0040620257, 0.0111813151] | +0.0106884792 | [0.0056177750, 0.0155697906] |
| 3407 | +0.0064735815 | [0.0033737488, 0.0096745744] | +0.0102531504 | [0.0052650602, 0.0151750542] |

### 5.4 字段命名纠正

`01_core/strict_no_pretraining/paired_bootstrap.csv` 的 `delta_auroc` 和 `delta_auprc` 实际写入 bootstrap 分布均值，而不是原始样本上的观测点差。例如 seed 42：

- 观测 Strict−Full AUROC：`-0.0078038432`；
- CSV `delta_auroc` / bootstrap mean：`-0.0077557990`。

这不影响 raw bootstrap CI，但论文表中的 point estimate 必须使用 prediction 观测点差，不能直接复制该 CSV 的 `delta_*` 列。ensemble 文件已经把 `point_difference` 与 `bootstrap_mean_difference` 分开记录，字段正确。

原始 bootstrap NPZ 中每项均为 2,000 个样本；独立读取 raw arrays 后计算的 2.5/97.5 分位数与发布 CSV 全部一致。

## 6. Low-label provenance 门：PASS

低标注矩阵已在 272 按当前增强协议完成：1%/5%/10%/30% × SSL/Scratch × seeds `{42,2026,3407}`，共 24 次训练；100% 复用 Full/Strict。

### 6.1 三种子均值与样本 SD

| 训练记录比例 | SSL SHD AUROC | Scratch SHD AUROC | SSL SHD AUPRC | Scratch SHD AUPRC |
|---:|---:|---:|---:|---:|
| 1% | 0.800259 ± 0.003529 | 0.764065 ± 0.013009 | 0.760899 ± 0.005380 | 0.697763 ± 0.021543 |
| 5% | 0.824991 ± 0.000845 | 0.800149 ± 0.002254 | 0.798316 ± 0.001161 | 0.755430 ± 0.005370 |
| 10% | 0.835022 ± 0.001334 | 0.818083 ± 0.002765 | 0.808372 ± 0.001293 | 0.781841 ± 0.003744 |
| 30% | 0.844679 ± 0.000547 | 0.836479 ± 0.002714 | 0.818476 ± 0.000613 | 0.803662 ± 0.002924 |
| 100% | 0.852064 ± 0.000807 | 0.844762 ± 0.000085 | 0.824884 ± 0.000922 | 0.814918 ± 0.000166 |

### 6.2 三种子 ensemble 配对 bootstrap

患者级配对 percentile bootstrap，2,000 次；Δ 定义为 SSL−Scratch。

| 比例 | SSL AUROC | Scratch AUROC | Δ AUROC | 95% CI | Δ AUPRC | 95% CI |
|---:|---:|---:|---:|---:|---:|---:|
| 1% | 0.805300 | 0.772982 | +0.032317 | [0.023908, 0.040832] | +0.057935 | [0.045325, 0.070711] |
| 5% | 0.826529 | 0.806970 | +0.019560 | [0.013403, 0.025769] | +0.033879 | [0.025066, 0.042468] |
| 10% | 0.836221 | 0.823517 | +0.012705 | [0.007684, 0.017584] | +0.020417 | [0.012841, 0.028236] |
| 30% | 0.845346 | 0.839800 | +0.005545 | [0.001742, 0.009289] | +0.011161 | [0.005675, 0.016658] |
| 100% | 0.852940 | 0.846839 | +0.006101 | [0.003436, 0.008798] | +0.008173 | [0.004420, 0.012197] |

### 6.3 Provenance

- 固定 permutation SHA256：`180be8b31a936addd397c98b83a153b6929792650bc5b5234aa8b3ee83617240`；
- 1/5/10/30% 为同一 permutation 的嵌套前缀；SSL/Scratch 在每个比例使用完全相同的 ECG 行索引；
- 每个训练条件均有 checkpoint、effective config、subset manifest、val/test predictions、exact metrics 和 review status；
- test predictions 均为 5,442 行、5,442 名唯一患者，并通过同序标签与 split hash 对齐；
- `provenance/artifact_sha256.csv` SHA256：`b9e8d0278af14f57179b810fcf7d10f69541d5eb93a702ebd9a0ea3fedf66ed1`；
- 原始配对分布：`analysis/paired_bootstrap_raw.npz`；
- 完成门：`LOW_LABEL_COMPLETE.json`。

低标注比例按训练 ECG 记录行定义，不表述为唯一患者比例。100% SSL/Scratch 分别复用已冻结的 Full/Strict。

## 7. 冻结证据 SHA256

| 证据 | SHA256 |
|---|---|
| `THREE_SEED_PAPER_COMPLETE.json` | `33402e823cabc78ed83fccdfd8c772c838fedd25736b975bd1d0887688f56bb3` |
| Full `exact_seed_metrics.csv` | `7ba56577da9c1f01cecc09f37a25e48dabeb6524bfafdf81d3c1fcd7a12207b8` |
| Full `exact_summary.csv` | `81dc552382c4f0dd0e91d3c230c4038fab1ee3b82f8e5feaee7371598df9cd1b` |
| Strict `exact_seed_metrics.csv` | `9ea128081536ab68fb941dc07ddbbb1de1b20ad24bdd7d4267f2fb0b15cb3bca` |
| Strict `exact_summary.csv` | `fb08f7803bbe3a1f614de8204098c4147f33720a34278ab20441f6c1cf641467` |
| Strict `paired_bootstrap.csv` | `4e50b5066d563379403f24da4b42134f1640cddcd3bf0a86c255f38b8def3668` |
| Strict `paired_bootstrap_raw.npz` | `08c74a22e88a34042e8467d96a991f051cf86c5caa1897d094141d2017ff1f4c` |
| Three-seed statistics `status.json` | `1d341373b808b1af557f9c6082b174e96a1b69671ebaf6079fbb5b115370dc8c` |
| Full ensemble aggregate CI | `001e837392afd3276bf6a980aa523967548c4d950be15808c962e426615c4883` |
| Full-vs-Strict ensemble paired CI | `335a2ed9ae900835aa02c0b96339dd21ecdbc99a3be4cc0863193928125a30e0` |
| Three-seed `bootstrap_raw.npz` | `c79df6c27c8ca41a93a92d42e48ba7f3b14c7b928cd342eca37289f3fb186183` |
| Prediction alignment manifest | `5017e560b861752712627ba86eacf79c995ac640a7131e5c771ee82c02bef2b9` |

## 8. 唯一论文使用规则

1. Full/Strict 单模型三种子表使用第 2 节的精确值与 sample SD。
2. 主要配对结论优先使用第 5.2 节的三种子 ensemble 配对结果。
3. 单 seed 配对点差使用 prediction 直接差；CI 使用 raw paired bootstrap 分位数。
4. 论文中明确写 Macro-11 或 Macro-12，不使用模糊的“Macro AUROC”。
5. Low-label 使用第 6 节冻结结果，并保留 ECG 记录行比例的定义。
6. 当前总冻结状态为 **PASS**；Full/Strict 与 low-label 两个完成门均已闭环。

## 9. 远端唯一位置

本文件应位于：

`experiments/20260804_aug_jitter005_mask010_full_paper_v1/12_three_seed_completion_272_v1/final_results_freeze.md`

Full/Strict 和统计证据分别位于：

- `01_core/full/`
- `01_core/strict_no_pretraining/`
- `10_three_seed_paper_statistics_v1/`
- `12_three_seed_completion_272_v1/`

在 low-label provenance 解锁后，应新建版本化冻结文件（例如 `final_results_freeze_v2.md`），保留本文件，不覆盖本次阻塞审计。

> 冻结门由低标注完成队列于 2026-08-21T07:30:59 更新；原阻塞版本保存在 `final_results_freeze_blocked_v1.md`。
