#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Train a tabular-only EchoNext baseline for SHD multi-label prediction."""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ecg_project.data import add_data_args, build_data_context
from ecg_project.utils.config import parse_args_with_config
from ecg_project.utils.experiment import prepare_experiment_directory
from ecg_project.utils.metrics import multilabel_metrics


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TabularMLP(nn.Module):
    """Small MLP baseline that consumes only tabular features."""

    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, tabular: torch.Tensor) -> torch.Tensor:
        return self.net(tabular)


def seed_everything(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_train_loader(train_set, seed: int, batch_size: int, num_workers: int) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=True,
        generator=generator,
        worker_init_fn=seed_worker,
    )


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, n_labels: int, primary_idx: int | None) -> dict[str, float]:
    model.eval()
    probabilities, targets = [], []
    for _wave, tabular, labels in loader:
        logits = model(tabular.to(DEVICE, non_blocking=True))
        probabilities.append(torch.sigmoid(logits).cpu().numpy())
        targets.append(labels.numpy())
    return multilabel_metrics(np.vstack(targets), np.vstack(probabilities), primary_idx)


def train_one_seed(args: argparse.Namespace, ctx: dict, seed: int, experiment_dir: Path) -> dict:
    seed_everything(seed, deterministic=args.deterministic)
    train_loader = make_train_loader(ctx["train_set"], seed, args.batch_size, args.num_workers)
    model = TabularMLP(ctx["tab_dim"], ctx["n_labels"], args.hidden_dim, args.dropout).to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=args.base_lr, weight_decay=args.weight_decay)
    total_steps = max(1, args.epochs * len(train_loader))
    warmup_steps = max(1, 2 * len(train_loader))

    def lr_factor(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    best_val = -1.0
    best_epoch = -1
    bad_epochs = 0
    checkpoint_path = experiment_dir / f"{args.experiment_name}_seed{seed}.pth"
    t0 = time.time()

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for _wave, tabular, labels in train_loader:
            tabular = tabular.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(model(tabular), labels)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()
            running_loss += float(loss.detach().cpu())
        val_metrics = evaluate(model, ctx["val_loader"], ctx["n_labels"], ctx["primary_idx"])
        val_primary = float(val_metrics["primary_auroc"])
        if val_primary > best_val:
            best_val = val_primary
            best_epoch = epoch + 1
            bad_epochs = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "model_config": {
                        "input_dim": ctx["tab_dim"],
                        "num_classes": ctx["n_labels"],
                        "hidden_dim": args.hidden_dim,
                        "dropout": args.dropout,
                    },
                    "seed": seed,
                    "epoch": best_epoch,
                    "val_primary_auroc": best_val,
                },
                checkpoint_path,
            )
        else:
            bad_epochs += 1
        print(
            f"seed={seed} epoch={epoch + 1:03d}/{args.epochs} "
            f"loss={running_loss / max(1, len(train_loader)):.4f} "
            f"val={val_primary:.4f} best={best_val:.4f}@{best_epoch} "
            f"early_stop={bad_epochs}/{args.patience}"
        )
        if args.patience > 0 and bad_epochs >= args.patience:
            break

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model = TabularMLP(**checkpoint["model_config"]).to(DEVICE)
    model.load_state_dict(checkpoint["model"])
    test_metrics = evaluate(model, ctx["test_loader"], ctx["n_labels"], ctx["primary_idx"])
    return {
        "Dataset": args.dataset,
        "Config": args.experiment_name,
        "Model": "tabular_mlp",
        "Des": args.des,
        "Loss": "bce",
        "Seed": seed,
        "Params(M)": round(sum(p.numel() for p in model.parameters()) / 1e6, 4),
        "EMA": False,
        "SWA": False,
        "EarlyStop_Patience": args.patience,
        "Best_Epoch": best_epoch,
        "Val_Primary_AUROC": round(float(best_val), 4),
        "Test_Primary_AUROC": round(float(test_metrics["primary_auroc"]), 4),
        "Test_Primary_AUPRC": round(float(test_metrics["primary_auprc"]), 4),
        "Test_Macro_AUROC": round(float(test_metrics["macro_auroc"]), 4),
        "Test_Macro_F1": round(float(test_metrics["macro_f1"]), 4),
        "Test_Micro_F1": round(float(test_metrics["micro_f1"]), 4),
        "Time(min)": round((time.time() - t0) / 60.0, 1),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tabular-only EchoNext baseline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default=None)
    add_data_args(parser)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 2024, 3407])
    parser.add_argument("--base-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--deterministic", action="store_true", default=True)
    parser.add_argument("--no-deterministic", dest="deterministic", action="store_false")
    parser.add_argument("--output-root", default="experiments")
    parser.add_argument("--experiment-name", default="tabular_only")
    parser.add_argument("--des", default="tabular_only")
    parser.add_argument("--results-csv", default="metrics.csv")
    parser.add_argument("--smoke-test", action="store_true")
    return parse_args_with_config(parser)


def main() -> None:
    args = parse_args()
    experiment_dir = prepare_experiment_directory(
        PROJECT_ROOT,
        args.output_root,
        args.experiment_name,
        args,
        args.config,
    )
    results_path = Path(args.results_csv)
    if not results_path.is_absolute():
        results_path = experiment_dir / results_path
    ctx = build_data_context(args)
    rows = []
    if results_path.exists():
        old = pd.read_csv(results_path)
        rows = old.to_dict("records")
        done = set(int(seed) for seed in old.get("Seed", []))
    else:
        done = set()
    for seed in args.seeds:
        if seed in done:
            print(f"[SKIP] seed={seed}")
            continue
        metrics = train_one_seed(args, ctx, seed, experiment_dir)
        rows.append(metrics)
        pd.DataFrame(rows).to_csv(results_path, index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"[done] metrics written to {results_path}")


if __name__ == "__main__":
    main()
