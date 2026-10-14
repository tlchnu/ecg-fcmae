#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ecg_data.py —— 数据集与 DataLoader 的统一入口(从 runner / 预训练脚本里独立出来)。

支持两个数据集,由 --dataset 选择:
  * echonext : 你原来的 12 标签超声衍生任务,带 7 维表格特征 + split 列;主指标 = SHD AUROC。
  * ptbxl    : 直接读取 PTB-XL 原始数据集,无表格特征;诊断标签集由 --ptbxl-task 控制。
               主指标 = macro AUROC。

设计要点
--------
1. 数据集只负责「读波形/标签」,**不再在 __getitem__ 里做增强** —— 增强统一由
   ecg_aug.py 在训练循环里对 batch 施加(见 runner),这样 aug.py 的 batch 级算子才用得上。
2. 路径全部走参数:echonext 用 --data-dir + 各 *-name;ptbxl 用 --ptbxl-dir 指向原始目录。
3. forward 签名统一为 (wave, tab, labels);ptbxl 无表格特征时返回一个零向量占位
   (维度 = tabular_features,默认 0 表示模型完全不建表格分支)。
4. build_data_context(args) 返回 runner 需要的 ctx 字典;另有 build_waveform_dataset
   供自监督预训练(只取波形)复用同一套读取逻辑。
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
LEAD_SETTINGS = {
    "full12": list(range(12)),
    "limb6": [0, 1, 2, 3, 4, 5],
    "precordial6": [6, 7, 8, 9, 10, 11],
    "anatomical4": [0, 1, 6, 10],  # I, II, V1, V5
    "lead_i": [0],
    "lead_ii": [1],
    "lead_v1": [6],
}


class LeadSelectionDataset(Dataset):
    """Keep selected ECG leads using zero-fill or a true sliced input.

    ``zero_fill`` keeps the original 12-lead shape for robustness experiments.
    ``slice`` returns only the selected channels for a true reduced-input model.
    """

    def __init__(self, base_dataset, lead_setting="full12", input_mode="zero_fill"):
        if lead_setting not in LEAD_SETTINGS:
            raise ValueError(
                f"Unknown lead_setting={lead_setting!r}; "
                f"expected one of {sorted(LEAD_SETTINGS)}"
            )
        if input_mode not in {"zero_fill", "slice"}:
            raise ValueError("input_mode must be 'zero_fill' or 'slice'")
        self.base_dataset = base_dataset
        self.lead_setting = lead_setting
        self.input_mode = input_mode
        keep = torch.tensor(LEAD_SETTINGS[lead_setting], dtype=torch.long)
        keep_mask = torch.zeros(12, dtype=torch.bool)
        keep_mask[keep] = True
        self.keep_indices = keep
        self.keep_mask = keep_mask

    def __len__(self):
        return len(self.base_dataset)

    def _mask_wave(self, wave):
        if self.lead_setting == "full12":
            return wave
        if self.input_mode == "slice":
            return wave.index_select(0, self.keep_indices.to(wave.device))
        masked = wave.clone()
        masked[~self.keep_mask.to(masked.device)] = 0.0
        return masked

    def __getitem__(self, idx):
        item = self.base_dataset[idx]
        if isinstance(item, tuple):
            wave, *rest = item
            return (self._mask_wave(wave), *rest)
        return self._mask_wave(item)


class ShuffledLabelDataset(Dataset):
    """Break waveform-label association with one deterministic label permutation.

    Waveforms and tabular covariates remain at their original indices. Only the
    training-label rows are permuted, preserving marginal label prevalence.
    """

    def __init__(self, base_dataset, seed):
        self.base_dataset = base_dataset
        generator = torch.Generator().manual_seed(int(seed))
        self.permutation = torch.randperm(len(base_dataset), generator=generator)
        if hasattr(base_dataset, "labels"):
            source_labels = base_dataset.labels
        else:
            source_labels = torch.stack([base_dataset[index][2] for index in range(len(base_dataset))])
        self.labels = source_labels.index_select(0, self.permutation)
        self.shuffle_seed = int(seed)

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        wave, tabular, _ = self.base_dataset[idx]
        return wave, tabular, self.labels[idx]


# ===========================================================================
# 标签集定义
# ===========================================================================
ECHONEXT_LABEL_COLS = [
    "lvef_lte_45_flag", "lvwt_gte_13_flag",
    "aortic_stenosis_moderate_or_greater_flag",
    "aortic_regurgitation_moderate_or_greater_flag",
    "mitral_regurgitation_moderate_or_greater_flag",
    "tricuspid_regurgitation_moderate_or_greater_flag",
    "pulmonary_regurgitation_moderate_or_greater_flag",
    "rv_systolic_dysfunction_moderate_or_greater_flag",
    "pericardial_effusion_moderate_large_flag",
    "pasp_gte_45_flag", "tr_max_gte_32_flag",
    "shd_moderate_or_greater_flag",
]
ECHONEXT_PRIMARY = "shd_moderate_or_greater_flag"

# PTB-XL 各任务的预期类别数(用于核对载入的 y 是否对得上;名字仅 super 给全)
PTBXL_TASK_INFO = {
    "super":  {"n": 5,  "experiment": "exp1.1.1", "task": "superdiagnostic",
               "names": ["CD", "HYP", "MI", "NORM", "STTC"]},
    "sub":    {"n": 23, "experiment": "exp1.1",   "task": "subdiagnostic",  "names": None},
    "diag":   {"n": 44, "experiment": "exp1",     "task": "diagnostic",     "names": None},
    "form":   {"n": 19, "experiment": "exp2",     "task": "form",           "names": None},
    "rhythm": {"n": 12, "experiment": "exp3",     "task": "rhythm",         "names": None},
    "all":    {"n": 71, "experiment": "exp0",     "task": "all",            "names": None},
}


# ===========================================================================
# EchoNext
# ===========================================================================
class EchoNextNpyDataset(Dataset):
    """返回 (wave(12,L), tab(7,), labels(C,))。增强在外部 batch 级完成,这里不增强。"""
    def __init__(self, csv_path, wave_path, tab_path, split_name="train",
                 label_cols=None, limit=None):
        print(f"[echonext:{split_name}] 正在加载...")
        df = pd.read_csv(csv_path)
        df = df[df["split"] == split_name].reset_index(drop=True)
        if limit:
            df = df.iloc[:limit]

        self.label_cols = label_cols or ECHONEXT_LABEL_COLS
        self.labels = torch.tensor(df[self.label_cols].values, dtype=torch.float32)

        waves_raw = np.load(wave_path, mmap_mode="r")
        if limit:
            waves_raw = waves_raw[:limit]
        waves = np.squeeze(waves_raw, axis=1).transpose(0, 2, 1)   # (N,12,L)
        self.waves = torch.tensor(np.ascontiguousarray(waves), dtype=torch.float32)

        tab_raw = np.load(tab_path, mmap_mode="r")
        if limit:
            tab_raw = tab_raw[:limit]
        self.tabular = torch.tensor(np.ascontiguousarray(tab_raw), dtype=torch.float32)

        assert len(self.waves) == len(self.tabular) == len(self.labels), \
            f"[echonext:{split_name}] 波形/表格/标签长度不一致!"
        print(f"[echonext:{split_name}] 共 {len(self.waves)} 个样本 | wave={tuple(self.waves.shape)} "
              f"| tab={tuple(self.tabular.shape)} | labels={tuple(self.labels.shape)}")

    def __len__(self):
        return len(self.waves)

    def __getitem__(self, idx):
        return self.waves[idx], self.tabular[idx], self.labels[idx]


# ===========================================================================
# PTB-XL(从原始数据集预处理后的数组读取)
# ===========================================================================
class PTBXLArrayDataset(Dataset):
    """返回 (wave(12,L), tab(tab_dim,) 全零占位, labels(C,))。
    X 形如 (N,L,12),内部转成 (N,12,L);Y 形如 (N,C) 多热。"""
    def __init__(self, X, Y, split_name="train", tab_dim=0, limit=None):
        print(f"[ptbxl:{split_name}] 正在构建数组数据集...")
        if limit:
            X, Y = X[:limit], Y[:limit]
        X = np.ascontiguousarray(np.asarray(X).transpose(0, 2, 1))     # (N,12,L)
        self.waves = torch.tensor(X, dtype=torch.float32)
        self.labels = torch.tensor(np.ascontiguousarray(Y), dtype=torch.float32)
        if self.labels.ndim == 1:
            self.labels = self.labels.unsqueeze(1)
        self.tab_dim = tab_dim
        self._tab = torch.zeros(tab_dim, dtype=torch.float32) if tab_dim > 0 else torch.zeros(0)
        assert len(self.waves) == len(self.labels), f"[ptbxl:{split_name}] 波形/标签长度不一致!"
        print(f"[ptbxl:{split_name}] 共 {len(self.waves)} 个样本 | wave={tuple(self.waves.shape)} "
              f"| labels={tuple(self.labels.shape)}")

    def __len__(self):
        return len(self.waves)

    def __getitem__(self, idx):
        return self.waves[idx], self._tab, self.labels[idx]


# ===========================================================================
# 只取波形(自监督预训练用),两个数据集都支持
# ===========================================================================
class WaveformOnlyDataset(Dataset):
    def __init__(self, waves_tensor):
        self.waves = waves_tensor

    @classmethod
    def from_echonext(cls, csv_path, wave_path, split_name="train", limit=None):
        df = pd.read_csv(csv_path)
        df = df[df["split"] == split_name].reset_index(drop=True)
        n = len(df) if limit is None else min(limit, len(df))
        waves_raw = np.load(wave_path, mmap_mode="r")[:n]
        waves = np.squeeze(waves_raw, axis=1).transpose(0, 2, 1)
        w = torch.tensor(np.ascontiguousarray(waves), dtype=torch.float32)
        print(f"[echonext:{split_name}] 预训练波形: {tuple(w.shape)}")
        return cls(w)

    @classmethod
    def from_ptbxl(cls, x_path, split_name="train", limit=None):
        X = np.load(x_path, mmap_mode="r")
        if limit:
            X = X[:limit]
        X = np.ascontiguousarray(np.asarray(X).transpose(0, 2, 1))
        w = torch.tensor(X, dtype=torch.float32)
        print(f"[ptbxl:{split_name}] 预训练波形: {tuple(w.shape)}")
        return cls(w)

    def __len__(self):
        return len(self.waves)

    def __getitem__(self, idx):
        return self.waves[idx]


# ===========================================================================
# 路径解析
# ===========================================================================
def resolve_echonext_paths(args):
    d = args.data_dir
    return {
        "csv": os.path.join(d, args.csv_name),
        "train_wave": os.path.join(d, args.train_wave_name),
        "train_tab": os.path.join(d, args.train_tab_name),
        "val_wave": os.path.join(d, args.val_wave_name),
        "val_tab": os.path.join(d, args.val_tab_name),
        "test_wave": os.path.join(d, args.test_wave_name),
        "test_tab": os.path.join(d, args.test_tab_name),
    }


# ===========================================================================
# 顶层工厂:构建 runner 需要的 ctx
# ===========================================================================
def build_data_context(args):
    limit = (64 if getattr(args, "smoke_test", False) else None)
    lead_setting = getattr(args, "lead_setting", "full12")
    lead_input_mode = getattr(args, "lead_input_mode", "zero_fill")
    shuffle_train_labels_seed = getattr(args, "shuffle_train_labels_seed", None)

    if args.dataset == "echonext":
        p = resolve_echonext_paths(args)
        label_cols = ECHONEXT_LABEL_COLS
        train_set = EchoNextNpyDataset(p["csv"], p["train_wave"], p["train_tab"],
                                       "train", label_cols, limit)
        val_set = EchoNextNpyDataset(p["csv"], p["val_wave"], p["val_tab"],
                                     "val", label_cols, limit)
        test_set = EchoNextNpyDataset(p["csv"], p["test_wave"], p["test_tab"],
                                      "test", label_cols, limit)
        n_labels = len(label_cols)
        primary_idx = label_cols.index(ECHONEXT_PRIMARY)
        primary_name = "SHD"
        tab_dim = train_set.tabular.shape[1]
        label_names = label_cols

    elif args.dataset == "ptbxl":
        from .ptbxl import prepare_ptbxl_splits

        info = PTBXL_TASK_INFO[args.ptbxl_task]
        tab_dim = args.tabular_features        # 通常 0;>0 则喂零向量占位
        splits = prepare_ptbxl_splits(
            args.ptbxl_dir,
            task=args.ptbxl_task,
            sampling_rate=args.ptbxl_sampling_rate,
            min_samples=args.ptbxl_min_samples,
            train_fold=args.ptbxl_train_fold,
            val_fold=args.ptbxl_val_fold,
            test_fold=args.ptbxl_test_fold,
            drop_conflict=args.ptbxl_drop_conflict,
            use_cache=args.ptbxl_cache,
            standardize=args.ptbxl_standardize,
            smoke_limit=limit,
        )
        train_set = PTBXLArrayDataset(splits["X_train"], splits["y_train"], "train", tab_dim, limit)
        val_set = PTBXLArrayDataset(splits["X_val"], splits["y_val"], "val", tab_dim, limit)
        test_set = PTBXLArrayDataset(splits["X_test"], splits["y_test"], "test", tab_dim, limit)
        n_labels = train_set.labels.shape[1]
        if n_labels != info["n"]:
            print(f"[WARN] ptbxl-task={args.ptbxl_task} 名义 {info['n']} 类,"
                  f"实际 {n_labels} 类。可能由 --ptbxl-min-samples 或数据版本造成。")
        primary_idx = None                     # 无单一主标签 -> 用 macro
        primary_name = "Macro"
        label_names = splits["label_names"]
    else:
        raise ValueError(f"未知 dataset: {args.dataset}")

    if shuffle_train_labels_seed is not None:
        train_set = ShuffledLabelDataset(train_set, shuffle_train_labels_seed)
        print(
            "[label-randomization] training labels permuted only; "
            f"seed={shuffle_train_labels_seed}; validation/test unchanged"
        )

    if lead_setting != "full12":
        train_set = LeadSelectionDataset(train_set, lead_setting, lead_input_mode)
        val_set = LeadSelectionDataset(val_set, lead_setting, lead_input_mode)
        test_set = LeadSelectionDataset(test_set, lead_setting, lead_input_mode)
        kept = [LEAD_NAMES[idx] for idx in LEAD_SETTINGS[lead_setting]]
        print(f"[lead-setting] {lead_setting}: keep={kept}; input_mode={lead_input_mode}")

    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    print(f"[{args.dataset}] Train {len(train_set)} | Val {len(val_set)} | Test {len(test_set)} "
          f"| n_labels={n_labels} | tab_dim={tab_dim} | primary={primary_name}"
          + (f"(idx={primary_idx})" if primary_idx is not None else ""))

    return {
        "train_set": train_set,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "label_names": label_names,
        "n_labels": n_labels,
        "primary_idx": primary_idx,     # None => macro
        "primary_name": primary_name,
        "tab_dim": tab_dim,
        "input_channels": (
            len(LEAD_SETTINGS[lead_setting])
            if lead_setting != "full12" and lead_input_mode == "slice"
            else 12
        ),
        "lead_indices": (
            LEAD_SETTINGS[lead_setting]
            if lead_setting != "full12" and lead_input_mode == "slice"
            else None
        ),
    }


def build_waveform_dataset(args, split="train"):
    """自监督预训练:只取波形,支持两个数据集。"""
    limit = (64 if getattr(args, "smoke_test", False) else None)
    lead_setting = getattr(args, "lead_setting", "full12")
    if args.dataset == "echonext":
        p = resolve_echonext_paths(args)
        wave_name = {"train": p["train_wave"], "val": p["val_wave"], "test": p["test_wave"]}[split]
        dataset = WaveformOnlyDataset.from_echonext(p["csv"], wave_name, split, limit)
        return (
            LeadSelectionDataset(dataset, lead_setting, getattr(args, "lead_input_mode", "zero_fill"))
            if lead_setting != "full12"
            else dataset
        )
    elif args.dataset == "ptbxl":
        from .ptbxl import prepare_ptbxl_splits

        splits = prepare_ptbxl_splits(
            args.ptbxl_dir,
            task=args.ptbxl_task,
            sampling_rate=args.ptbxl_sampling_rate,
            min_samples=args.ptbxl_min_samples,
            train_fold=args.ptbxl_train_fold,
            val_fold=args.ptbxl_val_fold,
            test_fold=args.ptbxl_test_fold,
            drop_conflict=args.ptbxl_drop_conflict,
            use_cache=args.ptbxl_cache,
            standardize=args.ptbxl_standardize,
            smoke_limit=limit,
        )
        X = splits[f"X_{split}"]
        if limit:
            X = X[:limit]
        X = np.ascontiguousarray(np.asarray(X).transpose(0, 2, 1))
        w = torch.tensor(X, dtype=torch.float32)
        print(f"[ptbxl:{split}] 预训练波形: {tuple(w.shape)}")
        dataset = WaveformOnlyDataset(w)
        return (
            LeadSelectionDataset(dataset, lead_setting, getattr(args, "lead_input_mode", "zero_fill"))
            if lead_setting != "full12"
            else dataset
        )
    raise ValueError(f"未知 dataset: {args.dataset}")


# ===========================================================================
# 把数据相关的 argparse 选项集中加进来,runner / 预训练脚本共用
# ===========================================================================
def add_data_args(parser):
    parser.add_argument("--dataset", choices=["echonext", "ptbxl"], default="echonext",
                        help="选择数据集")
    parser.add_argument("--tabular-features", type=int, default=7,
                        help="表格特征维度;echonext 用 7,ptbxl 一般设 0(无表格分支)")

    # ---- echonext 路径 ----
    parser.add_argument("--lead-setting", choices=sorted(LEAD_SETTINGS), default="full12",
                        help="Reduced-lead ECG setting; unavailable leads are zero-filled.")
    parser.add_argument(
        "--lead-input-mode",
        choices=["zero_fill", "slice"],
        default="zero_fill",
        help="Keep a 12-channel zero-filled interface or use a true reduced-channel input.",
    )
    parser.add_argument(
        "--shuffle-train-labels-seed",
        type=int,
        default=None,
        help="Sanity-check only: deterministically permute training labels while leaving validation/test unchanged.",
    )

    g = parser.add_argument_group("echonext paths")
    g.add_argument("--data-dir", default=".", help="echonext 数据目录")
    g.add_argument("--csv-name", default="echonext_metadata_100k.csv")
    g.add_argument("--train-wave-name", default="EchoNext_train_waveforms.npy")
    g.add_argument("--train-tab-name", default="EchoNext_train_tabular_features.npy")
    g.add_argument("--val-wave-name", default="EchoNext_val_waveforms.npy")
    g.add_argument("--val-tab-name", default="EchoNext_val_tabular_features.npy")
    g.add_argument("--test-wave-name", default="EchoNext_test_waveforms.npy")
    g.add_argument("--test-tab-name", default="EchoNext_test_tabular_features.npy")

    # ---- ptbxl 原始数据路径 / 任务 ----
    h = parser.add_argument_group("ptbxl paths")
    h.add_argument("--ptbxl-dir", default="./ptbxl",
                   help="ptbxl 原始目录(含 ptbxl_database.csv / scp_statements.csv / records100 或 records500)")
    h.add_argument("--ptbxl-task", choices=list(PTBXL_TASK_INFO.keys()), default="super",
                   help="诊断标签集:super=5 / sub=23 / diag=44 / form=19 / rhythm=12 / all=71")
    h.add_argument("--ptbxl-sampling-rate", type=int, choices=[100, 500], default=100)
    h.add_argument("--ptbxl-min-samples", type=int, default=0,
                   help="过滤出现次数 <= 该值的稀有标签")
    h.add_argument("--ptbxl-train-fold", type=int, default=8,
                   help="strat_fold <= 该值作为训练集")
    h.add_argument("--ptbxl-val-fold", type=int, default=9)
    h.add_argument("--ptbxl-test-fold", type=int, default=10)
    h.add_argument("--ptbxl-no-drop-conflict", dest="ptbxl_drop_conflict", action="store_false",
                   help="不过滤 NORM+异常诊断类冲突记录")
    h.add_argument("--ptbxl-no-cache", dest="ptbxl_cache", action="store_false",
                   help="不使用/不写入 raw100.npy 或 raw500.npy 波形缓存")
    h.add_argument("--ptbxl-no-standardize", dest="ptbxl_standardize", action="store_false",
                   help="不使用训练集统计量标准化波形")
    h.set_defaults(ptbxl_drop_conflict=True, ptbxl_cache=True, ptbxl_standardize=True)
    return parser
