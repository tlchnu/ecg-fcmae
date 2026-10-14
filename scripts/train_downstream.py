#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ECGConvNeXtV3 消融/训练 —— 多 seed 严谨版(支持 echonext / ptbxl 双数据集 + 自监督微调)
================================================================================

当前版本的要点:
1. 数据读取 / DataLoader 全部搬到 ecg_data.py,用 --dataset 切换 echonext / ptbxl,
   ptbxl 直接读原始目录,用 --ptbxl-task 选诊断标签集(super=5 / all=71 等),路径全走参数。
2. 数据增强搬到 ecg_aug.py(基于 aug.py),用 --augmentations 控制,例如
   --augmentations flip,frequency,jitter,mask,channel,drop;增强在训练循环里对 batch 施加。
3. 训练默认 epochs=100,以验证集 primary AUROC 做 Early Stopping(patience=10)。
4. ECA 由配置控制；论文正式预训练与完整下游模型均启用 ECA。
5. 自监督走 --pretrained(载入 FCMAE 骨干 + 默认开 LLRD),下游可在 echonext / ptbxl 上微调。

用法
----
    # echonext,单配置,带增强,SWA,自定义实验名
    python ecg_ablation_v3_runner.py --dataset echonext --data-dir /path \
        --augmentations flip,jitter,mask --swa --des Exp --seeds 42 1 2

    # ptbxl superdiagnostic(5 类),直接读原始数据集,启用 ECA 消融、用平均池化
    python ecg_ablation_v3_runner.py --dataset ptbxl --ptbxl-dir ./ptbxl \
        --ptbxl-task super --tabular-features 0 --use-eca --pool avg --des ptbxl_super

    # 自监督微调:载入 FCMAE 骨干
    python ecg_ablation_v3_runner.py --dataset echonext --data-dir /path \
        --pretrained fcmae_encoder_best.pt --layer-decay 0.75 --des pretrained

    # 经典 leave-one-out 架构消融(预设配置列表)
    python ecg_ablation_v3_runner.py --dataset echonext --data-dir /path --ablation --seeds 42 1 2

    # 第一次先跑通流程
    python ecg_ablation_v3_runner.py --smoke-test --num-workers 0
"""

import os
import sys
import math
import copy
import time
import random
import argparse
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from tqdm.auto import tqdm
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ecg_project.augmentations import build_augmentation
from ecg_project.data import add_data_args, build_data_context
from ecg_project.models import MODEL_PRESETS, build_model as build_registered_model
from ecg_project.utils.checkpoint import load_encoder_checkpoint
from ecg_project.utils.config import parse_args_with_config
from ecg_project.utils.experiment import prepare_experiment_directory
from ecg_project.utils.metrics import multilabel_metrics

warnings.filterwarnings("ignore")

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)


# ---------------------------------------------------------------------------
# 可复现
# ---------------------------------------------------------------------------
def seed_everything(seed, deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def worker_init_fn(worker_id):
    s = torch.initial_seed() % (2 ** 32)
    np.random.seed(s)
    random.seed(s)


# ---------------------------------------------------------------------------
# 损失:BCE vs Focal(多标签)
# ---------------------------------------------------------------------------
class MultiLabelFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        targets = targets.float()
        ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p = torch.sigmoid(logits)
        pt = torch.where(targets == 1, p, 1 - p)
        alpha_t = torch.where(targets == 1,
                              torch.as_tensor(self.alpha, device=logits.device),
                              torch.as_tensor(1 - self.alpha, device=logits.device))
        focal = alpha_t * (1 - pt).pow(self.gamma) * ce
        if self.reduction == "mean":
            return focal.mean()
        if self.reduction == "sum":
            return focal.sum()
        return focal


def aggregate_shd_probability(logits, shd_index=-1, eps=1e-6):
    """Aggregate subtype probabilities into SHD probability with noisy-OR.

    Args:
        logits: Tensor of shape (B, C). One column is the direct SHD output.
        shd_index: Index of the direct SHD label. All other labels are treated
            as subtype labels for the composite SHD target.
        eps: Numerical clamp for stable probability products.

    Returns:
        Tensor of shape (B,) with P(any subtype positive).
    """
    if logits.ndim != 2:
        raise ValueError(f"Expected logits with shape (B, C), got {tuple(logits.shape)}")
    n_labels = logits.shape[1]
    if n_labels < 2:
        raise ValueError("SHD aggregation requires at least one subtype and one SHD label")
    shd_index = shd_index % n_labels
    subtype_indices = [idx for idx in range(n_labels) if idx != shd_index]
    subtype_probs = torch.sigmoid(logits[:, subtype_indices]).clamp(eps, 1.0 - eps)
    return 1.0 - torch.prod(1.0 - subtype_probs, dim=1)


class SHDConsistencyLoss(nn.Module):
    """Base multi-label loss plus SHD hierarchy-consistency regularization."""

    def __init__(self, base_loss="bce", shd_index=-1, consistency_weight=0.0):
        super().__init__()
        if consistency_weight < 0:
            raise ValueError("consistency_weight must be non-negative")
        self.shd_index = shd_index
        self.consistency_weight = float(consistency_weight)
        if base_loss == "bce":
            self.base = nn.BCEWithLogitsLoss()
        elif base_loss == "focal":
            self.base = MultiLabelFocalLoss(alpha=0.25, gamma=2.0, reduction="mean")
        else:
            raise ValueError(f"Unknown base_loss: {base_loss}")

    def forward(self, logits, targets):
        base_loss = self.base(logits, targets.float())
        if self.consistency_weight <= 0:
            return base_loss
        n_labels = logits.shape[1]
        shd_index = self.shd_index % n_labels
        shd_target = targets[:, shd_index].float()
        subtype_shd_prob = aggregate_shd_probability(logits, shd_index=shd_index)
        direct_shd_prob = torch.sigmoid(logits[:, shd_index])
        subtype_supervision = F.binary_cross_entropy(
            subtype_shd_prob.clamp(1e-6, 1.0 - 1e-6),
            shd_target,
        )
        direct_consistency = F.mse_loss(direct_shd_prob, subtype_shd_prob)
        hierarchy_loss = subtype_supervision + 0.5 * direct_consistency
        return base_loss + self.consistency_weight * hierarchy_loss


def make_criterion(loss_type, shd_consistency_weight=0.0, shd_index=-1):
    if shd_consistency_weight > 0:
        return SHDConsistencyLoss(
            base_loss=loss_type,
            shd_index=shd_index,
            consistency_weight=shd_consistency_weight,
        )
    if loss_type == "bce":
        return nn.BCEWithLogitsLoss()
    if loss_type == "focal":
        return MultiLabelFocalLoss(alpha=0.25, gamma=2.0, reduction="mean")
    raise ValueError(f"Unknown loss_type: {loss_type}")


# ---------------------------------------------------------------------------
# 训练技巧:EMA / cosine warmup / LLRD(都做成可控)
# ---------------------------------------------------------------------------
class ModelEMA:
    def __init__(self, model, decay=0.999):
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay = decay

    @torch.no_grad()
    def update(self, model):
        d = self.decay
        for ema_p, p in zip(self.ema.parameters(), model.parameters()):
            ema_p.mul_(d).add_(p.detach(), alpha=1 - d)
        for ema_b, b in zip(self.ema.buffers(), model.buffers()):
            ema_b.copy_(b)


def cosine_warmup_lr(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, lr_lambda)


def _llrd_group_id(name, n_stages=4):
    if name.startswith("front."):
        return 0
    if name.startswith("downsample_layers."):
        return int(name.split(".")[1])
    if name.startswith("stages."):
        return int(name.split(".")[1])
    return n_stages


def build_llrd_param_groups(model, base_lr, weight_decay, layer_decay,
                            n_stages=4, verbose=False):
    top = n_stages
    groups = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        gid = _llrd_group_id(name, n_stages)
        no_wd = p.ndim <= 1
        scale = layer_decay ** (top - gid)
        key = (gid, no_wd)
        if key not in groups:
            groups[key] = {
                "params": [], "lr": base_lr * scale,
                "weight_decay": 0.0 if no_wd else weight_decay,
                "lr_scale": scale, "group_name": f"layer{gid}_{'nowd' if no_wd else 'wd'}",
            }
        groups[key]["params"].append(p)
    param_groups = list(groups.values())
    if verbose:
        for g in sorted(param_groups, key=lambda d: (d["lr_scale"], d["weight_decay"])):
            print(f"    [LLRD] {g['group_name']:14s} lr={g['lr']:.2e} "
                  f"(x{g['lr_scale']:.3f}) wd={g['weight_decay']:.3g} | {len(g['params'])} tensors")
    return param_groups


# ---------------------------------------------------------------------------
# 评估:返回主指标(SHD 单标签 或 macro)与 macro
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, loader, device, n_labels, primary_idx):
    model.eval()
    probs_list, target_list = [], []
    for wave, tab, labels in loader:
        wave, tab = wave.to(device), tab.to(device)
        probs_list.append(torch.sigmoid(model(wave, tab)).cpu().numpy())
        target_list.append(labels.numpy())
    probs = np.vstack(probs_list)
    targets = np.vstack(target_list)
    return multilabel_metrics(targets, probs, primary_idx)


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
def cfg(name, model_name="convnext_tiny", loss="bce", use_eca=False,
        drop_path_rate=0.2, multi_scale_feat=True, pool="attn", fusion="film",
        pretrained=False, layer_decay=None, model_overrides=None):
    return dict(name=name, loss=loss, use_eca=use_eca, drop_path_rate=drop_path_rate,
                multi_scale_feat=multi_scale_feat, pool=pool, fusion=fusion,
                model_name=model_name, pretrained=pretrained, layer_decay=layer_decay,
                model_overrides=model_overrides or {})


# 经典 leave-one-out 预设(--ablation 时用)
ARCH_CONFIGS = [
    cfg("baseline"),
    cfg("eca",              use_eca=True),
    cfg("avg_pool",         pool="avg"),
    cfg("concat_fusion",    fusion="concat"),
    cfg("no_fusion",        fusion="none"),
    cfg("single_scale",     multi_scale_feat=False),
]
ARCHITECTURE_CONFIGS = [
    cfg("convnext", model_name="convnext_tiny"),
    cfg("convnext_layerscale", model_name="convnext_layerscale"),
    cfg("reparam_lk13", model_name="reparam_lk13"),
    cfg("reparam_lk31_mid", model_name="reparam_lk31_mid"),
    cfg("inception1d", model_name="inception1d"),
]
BASELINE_NAME = "baseline"


def make_cli_config(args):
    """用命令行的模块开关拼一个配置。名字默认用 --des(没给则 'custom')。"""
    name = args.des if args.des else "custom"
    return cfg(name,
               model_name=args.model,
               loss=args.loss,
               drop_path_rate=args.drop_path_rate,
               use_eca=args.use_eca,
               multi_scale_feat=args.multi_scale_feat,
               pool=args.pool,
               fusion=args.fusion,
               pretrained=bool(args.pretrained),
               layer_decay=(args.layer_decay if (args.pretrained or args.llrd) else None),
               model_overrides={
                   key: value for key, value in {
                       "block_type": args.block_type,
                       "stage_kernel_sizes": tuple(args.stage_kernel_sizes)
                           if args.stage_kernel_sizes else None,
                       "layer_scale_init": args.layer_scale_init,
                       "stem_kernel_size": args.stem_kernel_size,
                       "stem_stride": args.stem_stride,
                   }.items() if value is not None
               })


def build_config_list(args):
    if args.ablation and args.architecture_ablation:
        raise ValueError("--ablation and --architecture-ablation are mutually exclusive")
    if args.architecture_ablation:
        base = ARCHITECTURE_CONFIGS
        if args.pretrained:
            print("[WARN] Architecture ablation with one shared checkpoint is not fully fair; "
                  "shape-incompatible tensors will be skipped.")
            base = [dict(x, pretrained=True, layer_decay=args.layer_decay) for x in base]
        if args.des:
            base = [dict(x, name=f"{x['name']}_{args.des}") for x in base]
        return base
    if args.ablation:
        base = ARCH_CONFIGS
        if args.pretrained:
            base = [dict(x, pretrained=True, layer_decay=args.layer_decay) for x in base]
        elif args.llrd:
            base = [dict(x, layer_decay=args.layer_decay) for x in base]
        if args.des:
            base = [dict(x, name=f"{x['name']}_{args.des}") for x in base]
        return base
    if args.pretrained:
        c = make_cli_config(args)
        c["name"] = (args.des or "baseline") + "_pretrained"
        c["pretrained"] = True
        c["layer_decay"] = args.layer_decay
        return [c]
    return [make_cli_config(args)]


# ---------------------------------------------------------------------------
# 模型
# ---------------------------------------------------------------------------
def build_model(config, n_labels, tab_dim, input_channels=12):
    overrides = dict(config.get("model_overrides", {}))
    overrides.update({
        "in_channels": input_channels,
        "num_classes": n_labels,
        "tabular_features": tab_dim,
        "drop_path_rate": config.get("drop_path_rate", 0.2),
        "use_eca": config.get("use_eca", False),
        "multi_scale_feat": config.get("multi_scale_feat", True),
        "pool": config.get("pool", "attn"),
        "fusion": config.get("fusion", "film"),
    })
    return build_registered_model(
        config.get("model_name", "convnext_tiny"),
        **overrides,
    )


def build_train_loader(train_set, seed, batch_size, num_workers):
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(train_set, batch_size=batch_size, shuffle=True,
                      num_workers=num_workers, pin_memory=True, drop_last=True,
                      generator=g, worker_init_fn=worker_init_fn)


# ---------------------------------------------------------------------------
# 训练(单 config + 单 seed)
# ---------------------------------------------------------------------------
def train_one(config, seed, epochs, ctx, verbose=True):
    n_labels = ctx["n_labels"]
    primary_idx = ctx["primary_idx"]
    tab_dim = ctx["tab_dim"]
    deterministic = ctx["deterministic"]

    seed_everything(seed, deterministic=deterministic)
    train_loader = build_train_loader(ctx["train_set"], seed,
                                       ctx["batch_size"], ctx["num_workers"])

    seed_everything(seed, deterministic=deterministic)
    input_channels = ctx.get("input_channels", 12)
    model = build_model(config, n_labels, tab_dim, input_channels).to(device)

    use_pretrained = config.get("pretrained", False) and bool(ctx.get("pretrained_encoder"))
    if use_pretrained:
        load_encoder_checkpoint(
            model,
            ctx["pretrained_encoder"],
            verbose=verbose,
            input_channel_indices=ctx.get("lead_indices"),
            stem_mode=ctx.get("pretrained_stem_mode", "select"),
        )
    n_params = sum(p.numel() for p in model.parameters()) / 1e6

    criterion = make_criterion(
        config["loss"],
        shd_consistency_weight=ctx.get("shd_consistency_weight", 0.0),
        shd_index=primary_idx,
    )

    # ---- 增强流水线(batch 级,仅训练时生效)----
    aug = build_augmentation(ctx["augmentations"]).to(device)
    aug.train()

    # ---- 优化器:可选 LLRD ----
    base_lr, weight_decay = ctx["base_lr"], ctx["weight_decay"]
    layer_decay = config.get("layer_decay")
    if layer_decay is None and (use_pretrained or ctx.get("force_llrd")):
        layer_decay = ctx.get("layer_decay", 0.75)
    if layer_decay is not None:
        param_groups = build_llrd_param_groups(model, base_lr, weight_decay,
                                                layer_decay, model.n_stages, verbose=verbose)
        optimizer = optim.AdamW(param_groups, betas=(0.9, 0.999))
        if verbose:
            print(f"  [opt] LLRD on | base_lr={base_lr:.2e} layer_decay={layer_decay} "
                  f"| pretrained={use_pretrained}")
    else:
        optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)

    total_steps = epochs * len(train_loader)
    warmup_steps = max(1, 2 * len(train_loader))
    scheduler = cosine_warmup_lr(optimizer, warmup_steps, total_steps)

    # ---- 可控训练技巧:EMA / SWA ----
    use_ema = ctx["use_ema"]
    ema = ModelEMA(model, decay=ctx["ema_decay"]) if use_ema else None

    use_swa = ctx["use_swa"]
    swa_model = None
    swa_start = max(1, int(ctx["swa_start_frac"] * epochs))
    if use_swa:
        from torch.optim.swa_utils import AveragedModel
        swa_model = AveragedModel(model)

    grad_clip = ctx["grad_clip"]
    track = (lambda: ema.ema) if use_ema else (lambda: model)   # 用于逐 epoch val 选择

    tag = ctx["setting_fn"](config, seed)
    save_path = os.path.join(ctx["experiment_dir"], f"{tag}.pth")

    best_val = -1.0
    best_epoch = -1
    patience = ctx["patience"]
    bad_epochs = 0
    t0 = time.time()

    for epoch in range(epochs):
        model.train()
        running = 0.0
        pbar = tqdm(train_loader, desc=f"[{tag}] Ep {epoch+1}/{epochs}", leave=False)
        for wave, tab, labels in pbar:
            wave, tab, labels = wave.to(device), tab.to(device), labels.to(device)
            wave = aug(wave)                              # batch 级增强
            optimizer.zero_grad()
            loss = criterion(model(wave, tab), labels)
            loss.backward()
            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()
            if use_ema:
                ema.update(model)
            running += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        if use_swa and (epoch + 1) >= swa_start:
            swa_model.update_parameters(model)

        m = evaluate(track(), ctx["val_loader"], device, n_labels, primary_idx)
        val_primary = m["primary_auroc"]
        if val_primary > best_val:
            best_val = val_primary
            best_epoch = epoch + 1
            bad_epochs = 0
            torch.save({
                "model": track().state_dict(),
                "model_config": model.model_config,
                "training_config": config,
                "epoch": best_epoch,
                "seed": seed,
                "val_primary_auroc": best_val,
            }, save_path)
        else:
            bad_epochs += 1

        if verbose:
            print(f"  Ep {epoch+1:02d} | loss={running/len(train_loader):.4f} | "
                  f"val {ctx['primary_name']}={val_primary:.4f} | "
                  f"best={best_val:.4f} @ ep{best_epoch} | "
                  f"early_stop={bad_epochs}/{patience}")

        if patience > 0 and bad_epochs >= patience:
            if verbose:
                print(f"  [early stop] val AUROC {patience} 个 epoch 未提升,停止于 ep{epoch+1}.")
            break

    # ---- 最终评估:SWA 优先用平均模型;否则用 val 最佳的 (EMA/raw) 权重 ----
    if use_swa:
        swa_model.eval()
        val_m = evaluate(swa_model, ctx["val_loader"], device, n_labels, primary_idx)
        test_m = evaluate(swa_model, ctx["test_loader"], device, n_labels, primary_idx)
        best_val = val_m["primary_auroc"]
        best_epoch = "swa"
        torch.save({
            "model": swa_model.module.state_dict(),
            "model_config": model.model_config,
            "training_config": config,
            "epoch": "swa",
            "seed": seed,
            "val_primary_auroc": best_val,
        }, save_path)
    else:
        eval_model = build_model(config, n_labels, tab_dim, input_channels).to(device)
        checkpoint = torch.load(save_path, map_location=device)
        eval_model.load_state_dict(checkpoint["model"])
        test_m = evaluate(eval_model, ctx["test_loader"], device, n_labels, primary_idx)

    elapsed = (time.time() - t0) / 60.0
    return {
        "Dataset": ctx["dataset"],
        "Config": config["name"],
        "Model": config.get("model_name", "convnext_tiny"),
        "Des": ctx["des"],
        "Loss": config["loss"],
        "Seed": seed,
        "Params(M)": round(n_params, 2),
        "EMA": use_ema, "SWA": use_swa,
        "EarlyStop_Patience": patience,
        "Best_Epoch": best_epoch,
        "Val_Primary_AUROC": round(float(best_val), 4),
        "Test_Primary_AUROC": round(float(test_m["primary_auroc"]), 4),
        "Test_Primary_AUPRC": round(float(test_m["primary_auprc"]), 4),
        "Test_Macro_AUROC": round(float(test_m["macro_auroc"]), 4),
        "Test_Macro_F1": round(float(test_m["macro_f1"]), 4),
        "Test_Micro_F1": round(float(test_m["micro_f1"]), 4),
        "Time(min)": round(elapsed, 1),
    }


# ---------------------------------------------------------------------------
# 跑实验(带断点续跑)
# ---------------------------------------------------------------------------
def run_experiments(ctx, seeds, epochs, results_csv, configs):
    if os.path.exists(results_csv):
        results_df = pd.read_csv(results_csv)
        keycols = ["Dataset", "Config", "Loss", "Des", "Seed"]
        done = set(tuple(r) for r in results_df[keycols].itertuples(index=False, name=None)) \
            if all(c in results_df.columns for c in keycols) else set()
        print(f"已有结果 {len(results_df)} 条")
    else:
        results_df = pd.DataFrame()
        done = set()

    for config in configs:
        for seed in seeds:
            key = (ctx["dataset"], config["name"], config["loss"], ctx["des"], seed)
            if key in done:
                print(f"[SKIP] {key}")
                continue
            print(f"\n{'='*80}\n[RUN] {key}\n{'='*80}")
            metrics = train_one(config, seed, epochs=epochs, ctx=ctx, verbose=True)
            print(f"[DONE] {key}: Test {ctx['primary_name']} AUROC = {metrics['Test_Primary_AUROC']:.4f}")
            results_df = pd.concat([results_df, pd.DataFrame([metrics])], ignore_index=True)
            results_df.to_csv(results_csv, index=False)

    print(f"\n全部完成,结果写入 {results_csv}")
    return results_df


# ---------------------------------------------------------------------------
# 汇总 + 配对检验
# ---------------------------------------------------------------------------
def analyze(results_csv, baseline=BASELINE_NAME):
    try:
        from scipy import stats
        have_scipy = True
    except Exception:
        have_scipy = False

    df = pd.read_csv(results_csv)
    arch_df = df[df["Loss"] == "bce"].copy()

    agg = (arch_df.groupby("Config")["Test_Primary_AUROC"]
           .agg(["mean", "std", "count"]).reset_index()
           .rename(columns={"mean": "AUROC_mean", "std": "AUROC_std", "count": "n_seeds"}))
    agg["AUROC_mean"] = agg["AUROC_mean"].round(4)
    agg["AUROC_std"] = agg["AUROC_std"].round(4)

    pivot = arch_df.pivot_table(index="Seed", columns="Config", values="Test_Primary_AUROC")

    print("\n===== 各配置 Test Primary AUROC: mean ± std =====")
    print(agg.to_string(index=False))

    if baseline in pivot.columns:
        rows = []
        for name in pivot.columns:
            if name == baseline:
                continue
            paired = pd.concat([pivot[baseline], pivot[name]], axis=1).dropna()
            if paired.empty:
                continue
            delta = (paired[name] - paired[baseline]).values
            mean_d = float(np.mean(delta))
            std_d = float(np.std(delta, ddof=1)) if len(delta) > 1 else float("nan")
            n = len(delta)
            if have_scipy and n > 1 and std_d > 0:
                t_stat, p_val = stats.ttest_rel(paired[name].values, paired[baseline].values)
            else:
                t_stat = mean_d / (std_d / math.sqrt(n)) if (n > 1 and std_d > 0) else float("nan")
                p_val = float("nan")
            rows.append({"Config": name, "mean_Δ_vs_base": round(mean_d, 4),
                         "std_Δ": round(std_d, 4) if not math.isnan(std_d) else None,
                         "n": n,
                         "t": round(float(t_stat), 3) if not math.isnan(t_stat) else None,
                         "p_value": round(float(p_val), 4) if not math.isnan(p_val) else None})
        if rows:
            print(f"\n===== 配对检验 (config - {baseline}, 同 seed) =====")
            print(pd.DataFrame(rows).sort_values("mean_Δ_vs_base").to_string(index=False))
    else:
        print(f"\n[INFO] 结果里没有基线配置 '{baseline}',跳过配对检验。")
    if not have_scipy:
        print("\n(未装 scipy,只给了手算 t 值;pip install scipy 可得 p 值)")
    return agg


def plot_chart(agg, baseline_msr=None, output_stem="ablation_results", show=False):
    try:
        import matplotlib
        if not show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[WARN] 无法导入 matplotlib,跳过画图: {e}")
        return
    plot_df = agg.sort_values("AUROC_mean")
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(plot_df["Config"], plot_df["AUROC_mean"],
           yerr=plot_df["AUROC_std"].fillna(0), capsize=4, color="#A23B72")
    if baseline_msr:
        ax.axhline(baseline_msr, color="gray", linestyle="--",
                   label=f"baseline ({baseline_msr:.4f})")
        ax.legend(loc="lower right")
    ax.set_ylabel("Test Primary AUROC (mean ± std)")
    ax.set_title("Multi-seed results")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    png_path = f"{output_stem}.png"
    pdf_path = f"{output_stem}.pdf"
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    if show:
        plt.show()
    print(f"图已保存 {png_path} / {pdf_path}")


# ---------------------------------------------------------------------------
# 命令行
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="ECGConvNeXtV3 多 seed 训练/消融(echonext + ptbxl + 自监督)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--config", default=None, help="YAML experiment configuration")

    add_data_args(p)   # --dataset / --tabular-features / echonext & ptbxl 路径

    # 训练基本量
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=10,
                   help="Early Stopping patience;监控验证集 primary AUROC,<=0 关闭")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 1, 2, 3, 4])
    p.add_argument("--loss", choices=["bce", "focal"], default="bce")
    p.add_argument("--shd-consistency-weight", type=float, default=0.0,
                   help="EchoNext only: weight for noisy-OR subtype-to-SHD hierarchy loss")
    p.add_argument("--base-lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--smoke-test", action="store_true")
    p.add_argument("--no-deterministic", dest="deterministic", action="store_false")
    p.set_defaults(deterministic=True)

    # 需求2:数据增强
    p.add_argument("--augmentations", default="none",
                   help="逗号分隔,基于 aug.py。例:flip,frequency,jitter,mask,channel,drop;none=不增强")

    # 需求3:SWA / 描述 / 把训练技巧改成可控
    p.add_argument("--swa", action="store_true", help="启用随机权重平均 (SWA)")
    p.add_argument("--swa-start-frac", type=float, default=0.75,
                   help="从第 frac*epochs 个 epoch 起开始累计 SWA 平均")
    p.add_argument("--des", default=None, help="实验描述,拼进保存名/结果")
    p.add_argument("--ema", dest="ema", action="store_true", help="启用权重 EMA(默认开)")
    p.add_argument("--no-ema", dest="ema", action="store_false", help="关闭 EMA")
    p.set_defaults(ema=True)
    p.add_argument("--ema-decay", type=float, default=0.999)
    p.add_argument("--grad-clip", type=float, default=1.0, help="梯度裁剪阈值;<=0 关闭")
    p.add_argument("--drop-path-rate", type=float, default=0.2, help="stochastic depth 比率")
    p.add_argument("--llrd", action="store_true", help="即使不带 --pretrained 也启用 LLRD")
    p.add_argument("--layer-decay", type=float, default=0.75)

    # 模型工厂与可选消融模块
    p.add_argument("--model", choices=sorted(MODEL_PRESETS), default="convnext_tiny")
    p.add_argument("--block-type", choices=["convnext", "reparam_lk", "inception1d"],
                   default=None, help="覆盖 preset 的 temporal mixer")
    p.add_argument("--stage-kernel-sizes", type=int, nargs=4, default=None,
                   help="覆盖四个 stage 的 kernel size")
    p.add_argument("--layer-scale-init", type=float, default=None)
    p.add_argument("--stem-kernel-size", type=int, default=None)
    p.add_argument("--stem-stride", type=int, default=None)
    p.add_argument("--use-eca", dest="use_eca", action="store_true", help="启用 ECA 消融模块")
    p.add_argument("--no-eca", dest="use_eca", action="store_false", help="关闭 ECA 消融模块")
    p.add_argument("--multi-scale", dest="multi_scale_feat", action="store_true",
                   help="启用 penultimate + last 多尺度特征拼接")
    p.add_argument("--single-scale", dest="multi_scale_feat", action="store_false",
                   help="仅使用最后 stage 特征")
    p.add_argument("--pool", choices=["attn", "avg"], default="attn",
                   help="池化头:注意力池化或平均池化")
    p.add_argument("--fusion", choices=["film", "concat", "none"], default="film",
                   help="表格特征融合:FiLM、拼接或忽略表格")
    p.set_defaults(use_eca=False, multi_scale_feat=True)

    # 自监督 / 模式
    p.add_argument("--pretrained", default=None, help="FCMAE 预训练骨干 .pt,提供则做自监督微调")
    p.add_argument(
        "--pretrained-stem-mode",
        choices=["select", "skip"],
        default="select",
        help="Reduced-lead loading: select source stem channels or leave the target stem random.",
    )
    p.add_argument("--ablation", action="store_true", help="跑预设 leave-one-out 架构消融列表")
    p.add_argument("--architecture-ablation", action="store_true",
                   help="运行 ConvNeXt/LayerScale/Reparam-LK/Inception1D 架构消融")

    # 输出
    p.add_argument("--output-root", default="experiments")
    p.add_argument("--experiment-name", default=None,
                   help="推荐 YYYYMMDD_method_note;未提供时自动生成")
    p.add_argument("--results-csv", default="metrics.csv")
    p.add_argument("--baseline-msr", type=float, default=None, help="画图基线虚线(可选)")
    p.add_argument("--show-plot", action="store_true")
    p.add_argument("--skip-train", action="store_true", help="只分析已有结果")

    args = parse_args_with_config(p)
    if args.smoke_test:
        args.epochs = 2
    return args


def make_setting_fn(args):
    """生成保存名/标签:{dataset}_{config}_{loss}_{des}_seed{seed}"""
    def fn(config, seed):
        parts = [args.dataset, config["name"], config["loss"]]
        if args.des:
            parts.append(args.des)
        parts.append(f"seed{seed}")
        return "_".join(str(x) for x in parts)
    return fn


def main():
    args = parse_args()
    if not args.experiment_name:
        date_prefix = datetime.now().strftime("%Y%m%d")
        note = args.des or args.model
        args.experiment_name = f"{date_prefix}_{args.dataset}_{note}"
    if not args.des:
        args.des = args.model
    experiment_dir = prepare_experiment_directory(
        project_root=PROJECT_ROOT,
        output_root=args.output_root,
        experiment_name=args.experiment_name,
        args=args,
        source_config=args.config,
    )
    results_path = Path(args.results_csv)
    if not results_path.is_absolute():
        results_path = experiment_dir / results_path
    args.results_csv = str(results_path)

    configs = build_config_list(args)
    print(f"设备: {device} | dataset={args.dataset} | DETERMINISTIC={args.deterministic} | "
          f"SMOKE_TEST={args.smoke_test}")
    print(f"训练技巧: EMA={args.ema}(decay={args.ema_decay}) SWA={args.swa} "
          f"grad_clip={args.grad_clip} drop_path={args.drop_path_rate} "
          f"LLRD={args.llrd or bool(args.pretrained)} patience={args.patience}")
    print(f"增强: {args.augmentations}")
    print(f"模块开关: use_eca={args.use_eca} multi_scale={args.multi_scale_feat} "
          f"pool={args.pool} fusion={args.fusion}")
    print(f"实验目录: {experiment_dir}")
    print(f"Configs: {[(c['name'], c['loss']) for c in configs]} | Seeds: {args.seeds} | "
          f"Epochs: {args.epochs} | 总训练次数 = {len(configs)*len(args.seeds)}")

    if not args.skip_train:
        ctx = build_data_context(args)
        ctx.update({
            "batch_size": args.batch_size, "num_workers": args.num_workers,
            "deterministic": args.deterministic,
            "dataset": args.dataset, "des": args.des,
            "augmentations": args.augmentations,
            "use_ema": args.ema, "ema_decay": args.ema_decay,
            "use_swa": args.swa, "swa_start_frac": args.swa_start_frac,
            "patience": args.patience,
            "grad_clip": args.grad_clip,
            "base_lr": args.base_lr, "weight_decay": args.weight_decay,
            "layer_decay": args.layer_decay, "force_llrd": args.llrd,
            "shd_consistency_weight": args.shd_consistency_weight,
            "pretrained_encoder": args.pretrained,
            "pretrained_stem_mode": args.pretrained_stem_mode,
            "experiment_dir": str(experiment_dir),
            "setting_fn": make_setting_fn(args),
        })
        run_experiments(ctx, args.seeds, args.epochs, args.results_csv, configs)
    else:
        print("[INFO] --skip-train:仅分析已有结果。")

    if not os.path.exists(args.results_csv):
        print(f"[WARN] 找不到 {args.results_csv},无法分析/画图。")
        return
    if args.architecture_ablation:
        baseline_name = f"convnext_{args.des}" if args.des else "convnext"
    else:
        baseline_name = f"{BASELINE_NAME}_{args.des}" if (args.ablation and args.des) else BASELINE_NAME
    agg = analyze(args.results_csv, baseline=baseline_name)
    plot_chart(
        agg,
        baseline_msr=args.baseline_msr,
        output_stem=str(experiment_dir / "ablation_results"),
        show=args.show_plot,
    )


if __name__ == "__main__":
    main()
