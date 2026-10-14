#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ECG-FCMAE-1D —— ConvNeXt V2 官方 FCMAE 的一维自监督改编(12 导联 ECG;稠密 + SimMIM 式掩码重建)

与旧版的区别:波形读取改用 ecg_data.py,因此同一套自监督预训练可直接用于
echonext 或 ptbxl(--dataset 切换;只用波形,无需标签 / 表格特征)。

编码器 = ECGConvNeXtV3 的骨干(stem + downsample + stages),通过 forward_encoder 复用。
ECA 由 --use-eca/--no-eca 控制；论文正式 FCMAE 预训练启用 ECA。

用法
----
  # echonext 预训练
  python ecg_fcmae_pretrain.py --dataset echonext --data-dir /path --epochs 100 \
         --out fcmae_echonext.pt
  # ptbxl 预训练
  python ecg_fcmae_pretrain.py --dataset ptbxl --ptbxl-dir ./ptbxl_super --epochs 100 \
         --out fcmae_ptbxl.pt
  # 微调:在 runner 里带 --pretrained,且 cfg 开关与预训练一致
"""

import argparse
import math
import os
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ecg_project.data import add_data_args, build_waveform_dataset
from ecg_project.data.datasets import LEAD_SETTINGS
from ecg_project.models import (
    MODEL_PRESETS,
    Block,
    ECGConvNeXt1D,
    LayerNorm1D,
    get_model_config,
)
from ecg_project.utils.checkpoint import encoder_state_dict, load_encoder_checkpoint
from ecg_project.utils.config import parse_args_with_config
from ecg_project.utils.experiment import prepare_experiment_directory
from ecg_project.utils.reproducibility import seed_everything

ENC_PREFIXES = ("downsample_layers.", "stages.")
ANATOMICAL_LEAD_GROUPS = (
    (0, 1, 2, 3, 4, 5),       # limb leads: I, II, III, aVR, aVL, aVF
    (6, 7, 8, 9, 10, 11),     # precordial leads: V1-V6
    (1, 2, 5),                # inferior: II, III, aVF
    (6, 7, 8, 9),             # septal/anterior precordial: V1-V4
    (0, 4, 10, 11),           # lateral: I, aVL, V5, V6
)


# ---------------------------------------------------------------------------
# 1D FCMAE / SimMIM
# ---------------------------------------------------------------------------
class ECGFCMAE1d(nn.Module):
    def __init__(self, v3_kwargs, in_chans=12, patch_size=25, mask_ratio=0.6,
                 decoder_dim=256, decoder_depth=2, norm_pix_loss=True,
                 spectral_loss_weight=0.0, spectral_loss_type="smooth_l1",
                 masking_strategy="time", time_mask_ratio=None,
                 lead_mask_ratio=None):
        super().__init__()
        self.in_chans = in_chans
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.masking_strategy = masking_strategy
        self.time_mask_ratio = mask_ratio if time_mask_ratio is None else time_mask_ratio
        self.lead_mask_ratio = mask_ratio if lead_mask_ratio is None else lead_mask_ratio
        self.norm_pix_loss = norm_pix_loss
        self.spectral_loss_weight = spectral_loss_weight
        self.spectral_loss_type = spectral_loss_type
        valid_strategies = {
            "time", "time_independent", "lead", "lead_time_block", "anatomical"
        }
        if masking_strategy not in valid_strategies:
            raise ValueError(
                f"masking_strategy must be one of {sorted(valid_strategies)}, "
                f"got {masking_strategy!r}"
            )
        if spectral_loss_weight < 0.0:
            raise ValueError("spectral_loss_weight must be non-negative")
        if spectral_loss_type not in {"smooth_l1", "l1"}:
            raise ValueError("spectral_loss_type must be 'smooth_l1' or 'l1'")
        self.last_loss_components = {
            "time_loss": 0.0,
            "spectral_loss": 0.0,
            "total_loss": 0.0,
        }

        self.backbone = ECGConvNeXt1D(**v3_kwargs)
        enc_out_dim = v3_kwargs.get("dims", (64, 128, 256, 512))[-1]

        self.proj = nn.Conv1d(enc_out_dim, decoder_dim, kernel_size=1)
        self.mask_token = nn.Parameter(torch.zeros(1, decoder_dim, 1))
        nn.init.normal_(self.mask_token, std=0.02)
        self.decoder = nn.Sequential(
            *[Block(decoder_dim, kernel_size=7, drop_path=0.0)
              for _ in range(decoder_depth)]
        )
        self.dec_norm = LayerNorm1D(decoder_dim)
        self.pred = nn.Conv1d(decoder_dim, in_chans, kernel_size=1)

    def gen_patch_mask(self, B, n_patches, device):
        len_keep = int(n_patches * (1 - self.mask_ratio))
        noise = torch.rand(B, n_patches, device=device)
        ids = torch.argsort(noise, dim=1)
        mask = torch.ones(B, n_patches, device=device)
        mask.scatter_(1, ids[:, :len_keep], 0.0)        # 0=保留, 1=掩掉
        return mask

    def gen_mask(self, B, C, n_patches, device):
        if self.masking_strategy == "time":
            return self.gen_patch_mask(B, n_patches, device)
        if self.masking_strategy == "time_independent":
            return self._independent_time_mask(B, C, n_patches, device)
        if self.masking_strategy == "lead":
            return self._random_lead_mask(B, C, n_patches, device)
        if self.masking_strategy == "lead_time_block":
            return self._random_lead_time_block_mask(B, C, n_patches, device)
        if self.masking_strategy == "anatomical":
            return self._anatomical_group_time_block_mask(B, C, n_patches, device)
        raise ValueError(f"Unsupported masking_strategy={self.masking_strategy!r}")

    def _independent_time_mask(self, B, C, n_patches, device):
        """Sample the requested temporal mask ratio independently per lead."""
        len_keep = int(n_patches * (1 - self.mask_ratio))
        noise = torch.rand(B, C, n_patches, device=device)
        ids = torch.argsort(noise, dim=-1)
        mask = torch.ones(B, C, n_patches, device=device)
        mask.scatter_(2, ids[:, :, :len_keep], 0.0)
        return mask

    def _random_lead_mask(self, B, C, n_patches, device):
        n_mask_leads = max(1, min(C, int(round(C * self.lead_mask_ratio))))
        mask = torch.zeros(B, C, n_patches, device=device)
        for batch_idx in range(B):
            lead_ids = torch.randperm(C, device=device)[:n_mask_leads]
            mask[batch_idx, lead_ids, :] = 1.0
        return mask

    def _random_lead_time_block_mask(self, B, C, n_patches, device):
        n_mask_leads = max(1, min(C, int(round(C * self.lead_mask_ratio))))
        block_len = max(1, min(n_patches, int(round(n_patches * self.time_mask_ratio))))
        mask = torch.zeros(B, C, n_patches, device=device)
        for batch_idx in range(B):
            lead_ids = torch.randperm(C, device=device)[:n_mask_leads]
            start = torch.randint(0, n_patches - block_len + 1, (1,), device=device).item()
            mask[batch_idx, lead_ids, start:start + block_len] = 1.0
        return mask

    def _anatomical_group_time_block_mask(self, B, C, n_patches, device):
        block_len = max(1, min(n_patches, int(round(n_patches * self.time_mask_ratio))))
        mask = torch.zeros(B, C, n_patches, device=device)
        valid_groups = [
            tuple(idx for idx in group if idx < C)
            for group in ANATOMICAL_LEAD_GROUPS
            if any(idx < C for idx in group)
        ]
        for batch_idx in range(B):
            group = valid_groups[torch.randint(0, len(valid_groups), (1,), device=device).item()]
            start = torch.randint(0, n_patches - block_len + 1, (1,), device=device).item()
            mask[batch_idx, list(group), start:start + block_len] = 1.0
        return mask

    def _expand_mask_to_waveform(self, patch_mask, patch_size, channels):
        if patch_mask.ndim == 2:
            return patch_mask.repeat_interleave(patch_size, dim=1).unsqueeze(1)
        if patch_mask.ndim == 3:
            return patch_mask.repeat_interleave(patch_size, dim=2)
        raise ValueError(f"Expected patch mask with 2 or 3 dims, got {patch_mask.ndim}")

    def _temporal_mask_for_features(self, patch_mask):
        if patch_mask.ndim == 2:
            return patch_mask
        if patch_mask.ndim == 3:
            return patch_mask.amax(dim=1)
        raise ValueError(f"Expected patch mask with 2 or 3 dims, got {patch_mask.ndim}")

    def forward(self, wave):                              # wave: (B, 12, L)
        B, C, L = wave.shape
        P = self.patch_size
        n_patches = L // P
        Lc = n_patches * P
        wave = wave[:, :, :Lc]

        patch_mask = self.gen_mask(B, C, n_patches, wave.device)
        mask_full = self._expand_mask_to_waveform(patch_mask, P, C)
        masked_wave = wave * (1.0 - mask_full)

        feat = self.backbone.forward_encoder(masked_wave)
        Tprime = feat.shape[-1]

        x = self.proj(feat)
        temporal_mask = self._temporal_mask_for_features(patch_mask)
        feat_mask = F.interpolate(temporal_mask.unsqueeze(1), size=Tprime, mode="nearest")
        x = x * (1.0 - feat_mask) + self.mask_token * feat_mask
        x = self.dec_norm(self.decoder(x))
        x = F.interpolate(x, size=Lc, mode="linear", align_corners=False)
        pred = self.pred(x)

        time_loss = self._masked_loss(pred, wave, patch_mask, P)
        spectral_loss = pred.new_zeros(())
        if self.spectral_loss_weight > 0.0:
            recon_full = pred * mask_full + wave * (1.0 - mask_full)
            spectral_loss = self._spectral_loss(recon_full, wave)
        loss = time_loss + self.spectral_loss_weight * spectral_loss
        self.last_loss_components = {
            "time_loss": float(time_loss.detach().cpu()),
            "spectral_loss": float(spectral_loss.detach().cpu()),
            "total_loss": float(loss.detach().cpu()),
        }
        return loss, pred, mask_full

    def _masked_loss(self, pred, target, patch_mask, P):
        B, C, Lc = target.shape
        n = Lc // P
        t = target.reshape(B, C, n, P)
        p = pred.reshape(B, C, n, P)
        if self.norm_pix_loss:
            mu = t.mean(dim=-1, keepdim=True)
            var = t.var(dim=-1, keepdim=True)
            t = (t - mu) / torch.sqrt(var + 1e-6)
        per_patch = ((p - t) ** 2).mean(dim=-1)
        m = patch_mask
        if m.ndim == 2:
            m = m.unsqueeze(1).expand(-1, C, -1)
        return (per_patch * m).sum() / (m.sum() + 1e-8)

    def _spectral_loss(self, recon_full, target):
        recon_spec = torch.log1p(torch.abs(torch.fft.rfft(recon_full.float(), dim=-1)))
        target_spec = torch.log1p(torch.abs(torch.fft.rfft(target.float(), dim=-1)))
        if self.spectral_loss_type == "smooth_l1":
            return F.smooth_l1_loss(recon_spec, target_spec)
        return F.l1_loss(recon_spec, target_spec)


def load_pretrained_encoder(model, ckpt_path, verbose=True):
    """Backward-compatible alias for the shared checkpoint loader."""
    load_encoder_checkpoint(model, ckpt_path, verbose=verbose)
    return model


def cosine_lr(step, total, base_lr, warmup, min_lr=1e-6):
    if step < warmup:
        return base_lr * step / max(1, warmup)
    prog = (step - warmup) / max(1, total - warmup)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * prog))


def build_encoder_config(args):
    """Build the transferable ECG ConvNeXt encoder config for FCMAE."""
    model_overrides = {
        key: value for key, value in {
            "block_type": args.block_type,
            "stage_kernel_sizes": tuple(args.stage_kernel_sizes)
                if args.stage_kernel_sizes else None,
            "layer_scale_init": args.layer_scale_init,
        }.items() if value is not None
    }
    return get_model_config(
        args.model,
        in_channels=(
            len(LEAD_SETTINGS[args.lead_setting])
            if args.lead_setting != "full12" and args.lead_input_mode == "slice"
            else 12
        ),
        num_classes=12,
        tabular_features=0,
        drop_path_rate=0.0,
        use_eca=args.use_eca,
        multi_scale_feat=True,
        **model_overrides,
    )


def pretrain(args):
    seed_everything(args.seed, deterministic=args.deterministic)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = build_waveform_dataset(args, split=args.split)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                    num_workers=args.num_workers, pin_memory=True, drop_last=True)

    v3_kwargs = build_encoder_config(args)
    model = ECGFCMAE1d(
        v3_kwargs,
        in_chans=v3_kwargs["in_channels"],
        patch_size=args.patch_size,
        mask_ratio=args.mask_ratio,
        norm_pix_loss=not args.no_norm_pix,
        spectral_loss_weight=args.spectral_loss_weight,
        spectral_loss_type=args.spectral_loss_type,
        masking_strategy=args.masking_strategy,
        time_mask_ratio=args.time_mask_ratio,
        lead_mask_ratio=args.lead_mask_ratio,
    ).to(device)
    n_enc = sum(p.numel() for n, p in model.backbone.named_parameters()
                if n.startswith(ENC_PREFIXES)) / 1e6
    print(f"[{args.dataset}] 骨干可迁移参数: {n_enc:.2f}M | mask_ratio={args.mask_ratio} "
          f"patch={args.patch_size} use_eca={args.use_eca} "
          f"masking_strategy={args.masking_strategy} "
          f"spectral_loss_weight={args.spectral_loss_weight}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay, betas=(0.9, 0.95))
    amp_enabled = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    total_epochs = 2 if args.smoke_test else args.epochs
    steps_per_epoch = max(1, len(dl))
    total_steps = total_epochs * steps_per_epoch
    warmup = int(0.05 * total_steps)

    best = float("inf")
    gstep = 0
    for ep in range(total_epochs):
        model.train()
        running = 0.0
        for wave in dl:
            wave = wave.to(device, non_blocking=True)
            lr = cosine_lr(gstep, total_steps, args.lr, warmup)
            for g in opt.param_groups:
                g["lr"] = lr
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                loss, _, _ = model(wave)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            running += loss.item()
            gstep += 1
        avg = running / steps_per_epoch
        print(f"epoch {ep+1:3d}/{total_epochs} | recon_loss {avg:.4f} | lr {lr:.2e}")

        enc_state = encoder_state_dict(model.backbone)
        checkpoint = {
            "encoder": enc_state,
            "model_config": v3_kwargs,
            "epoch": ep + 1,
            "loss": avg,
            "seed": args.seed,
            "ssl_method": "fcmae",
            "masking_strategy": args.masking_strategy,
            "time_mask_ratio": args.time_mask_ratio,
            "lead_mask_ratio": args.lead_mask_ratio,
            "spectral_loss_weight": args.spectral_loss_weight,
            "spectral_loss_type": args.spectral_loss_type,
            "in_channels": v3_kwargs["in_channels"],
            "lead_setting": args.lead_setting,
            "lead_input_mode": args.lead_input_mode,
            "norm_pix_loss": not args.no_norm_pix,
        }
        torch.save(checkpoint, args.out)
        if avg < best:
            best = avg
            torch.save(checkpoint, args.out.replace(".pt", "_best.pt"))
    print(f"\n完成。骨干权重已存到 {args.out}(及 _best.pt)。"
          f"\n微调可直接带 --pretrained {args.out.replace('.pt', '_best.pt')};"
          f"匹配的 ECA 参数会随 encoder 权重加载；未匹配的模块按下游配置初始化。")


def parse_args():
    p = argparse.ArgumentParser(description="ECG FCMAE-1D 自监督预训练(echonext / ptbxl)",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--config", default=None)
    add_data_args(p)
    p.add_argument("--split", default="train")
    p.add_argument("--out", default="fcmae_encoder.pt")
    p.add_argument("--output-root", default="experiments")
    p.add_argument("--experiment-name", default=None)
    p.add_argument("--model", choices=sorted(MODEL_PRESETS), default="convnext_tiny")
    p.add_argument("--block-type", choices=["convnext", "reparam_lk", "inception1d"],
                   default=None)
    p.add_argument("--stage-kernel-sizes", type=int, nargs=4, default=None)
    p.add_argument("--layer-scale-init", type=float, default=None)
    p.add_argument("--use-eca", dest="use_eca", action="store_true",
                   help="Enable ECA in the FCMAE encoder.")
    p.add_argument("--no-eca", dest="use_eca", action="store_false",
                   help="Disable ECA in the FCMAE encoder.")
    p.set_defaults(use_eca=False)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=1.5e-4)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--patch-size", type=int, default=25, help="掩码 patch 长度(采样点)")
    p.add_argument("--mask-ratio", type=float, default=0.6)
    p.add_argument("--masking-strategy",
                   choices=["time", "time_independent", "lead", "lead_time_block", "anatomical"],
                   default="time")
    p.add_argument("--time-mask-ratio", type=float, default=None,
                   help="Temporal ratio for block masking; defaults to --mask-ratio.")
    p.add_argument("--lead-mask-ratio", type=float, default=None,
                   help="Lead ratio for lead-aware masking; defaults to --mask-ratio.")
    p.add_argument("--spectral-loss-weight", type=float, default=0.0)
    p.add_argument("--spectral-loss-type", choices=["smooth_l1", "l1"], default="smooth_l1")
    p.add_argument("--no-norm-pix", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-deterministic", dest="deterministic", action="store_false")
    p.set_defaults(deterministic=True)
    p.add_argument("--smoke-test", action="store_true")
    return parse_args_with_config(p)


if __name__ == "__main__":
    parsed_args = parse_args()
    if not parsed_args.experiment_name:
        parsed_args.experiment_name = (
            f"{datetime.now():%Y%m%d}_fcmae_{parsed_args.dataset}_{parsed_args.model}"
        )
    experiment_dir = prepare_experiment_directory(
        PROJECT_ROOT,
        parsed_args.output_root,
        parsed_args.experiment_name,
        parsed_args,
        parsed_args.config,
    )
    output_path = Path(parsed_args.out)
    if not output_path.is_absolute():
        output_path = experiment_dir / output_path
    parsed_args.out = str(output_path)
    pretrain(parsed_args)
