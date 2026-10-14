"""Deterministic execution helpers."""

import os
import random

import numpy as np
import torch


def seed_everything(seed: int = 42, deterministic: bool = True) -> None:
    """Seed Python, NumPy, PyTorch CPU, and PyTorch CUDA generators."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    if deterministic:
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except AttributeError:
            pass


def seed_worker(worker_id: int) -> None:
    """Seed a DataLoader worker from its PyTorch worker seed."""
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
