"""Experiment directory, environment, and configuration snapshots."""

from __future__ import annotations

import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import yaml


def collect_environment() -> dict[str, Any]:
    """Collect software and hardware metadata needed for reproduction."""
    cuda_available = torch.cuda.is_available()
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": cuda_available,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(0) if cuda_available else None,
    }


def _git_revision(project_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def prepare_experiment_directory(
    project_root: str | Path,
    output_root: str | Path,
    experiment_name: str,
    args: Any,
    source_config: str | None = None,
) -> Path:
    """Create an experiment folder and snapshot the effective configuration."""
    root = Path(project_root).resolve()
    output_path = Path(output_root)
    if not output_path.is_absolute():
        output_path = root / output_path
    experiment_dir = output_path / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    args_dict = vars(args) if hasattr(args, "__dict__") else dict(args)
    with (experiment_dir / "effective_config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(args_dict, handle, allow_unicode=True, sort_keys=True)

    environment = collect_environment()
    environment["git_revision"] = _git_revision(root)
    with (experiment_dir / "environment.json").open("w", encoding="utf-8") as handle:
        json.dump(environment, handle, ensure_ascii=False, indent=2)

    if source_config:
        shutil.copy2(source_config, experiment_dir / "source_config.yaml")
    return experiment_dir
