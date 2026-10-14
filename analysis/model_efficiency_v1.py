#!/usr/bin/env python
"""Measure parameters, scoped MACs/FLOPs, model size, and CUDA latency."""

from __future__ import annotations

import argparse
import io
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def conv1d_macs(module: nn.Conv1d, output: torch.Tensor) -> int:
    """Count multiply-accumulates for a Conv1d output tensor."""
    batch, output_channels, output_length = output.shape
    operations_per_output = (module.in_channels // module.groups) * module.kernel_size[0]
    return int(batch * output_channels * output_length * operations_per_output)


def linear_macs(module: nn.Linear, output: torch.Tensor) -> int:
    """Count multiply-accumulates for all vectors in a Linear output tensor."""
    vector_count = int(output.numel() // module.out_features)
    return int(vector_count * module.in_features * module.out_features)


def parameter_counts(model: nn.Module) -> dict[str, int]:
    """Return exact total and trainable parameter counts."""
    return {
        "total_parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        "trainable_parameters": int(
            sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        ),
    }


def bytes_to_megabytes(value: int) -> float:
    """Convert bytes to binary MiB."""
    return float(value / (1024**2))


def summarize_latency_ms(samples: np.ndarray) -> dict[str, float]:
    """Summarize synchronized latency samples in milliseconds."""
    values = np.asarray(samples, dtype=float)
    if values.size == 0:
        raise ValueError("At least one latency sample is required")
    return {
        "latency_median_ms": float(np.median(values)),
        "latency_mean_ms": float(np.mean(values)),
        "latency_std_ms": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
        "latency_p25_ms": float(np.percentile(values, 25)),
        "latency_p75_ms": float(np.percentile(values, 75)),
        "latency_p95_ms": float(np.percentile(values, 95)),
    }


def count_model_macs(
    model: nn.Module, waveform: torch.Tensor, tabular: torch.Tensor | None
) -> tuple[int, pd.DataFrame]:
    """Count Conv1d and Linear MACs with forward hooks for one inference."""
    rows: list[dict[str, int | str]] = []
    handles = []

    def hook(name: str):
        def record(module: nn.Module, _inputs, output) -> None:
            tensor = output[0] if isinstance(output, tuple) else output
            if isinstance(module, nn.Conv1d):
                macs = conv1d_macs(module, tensor)
            elif isinstance(module, nn.Linear):
                macs = linear_macs(module, tensor)
            else:
                return
            rows.append({"module": name, "type": type(module).__name__, "macs": macs})
        return record

    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv1d, nn.Linear)):
            handles.append(module.register_forward_hook(hook(name)))
    try:
        with torch.inference_mode():
            model(waveform, tabular)
    finally:
        for handle in handles:
            handle.remove()
    table = pd.DataFrame(rows)
    return int(table["macs"].sum()), table


def serialized_state_size_bytes(model: nn.Module) -> int:
    """Measure serialized state_dict size without writing a temporary file."""
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return int(buffer.tell())


def measure_latency(
    model: nn.Module,
    waveform: torch.Tensor,
    tabular: torch.Tensor | None,
    warmup: int,
    repetitions: int,
    use_amp: bool,
) -> np.ndarray:
    """Return synchronized batch-one inference latencies in milliseconds."""
    if waveform.device.type != "cuda":
        raise RuntimeError("CUDA is required for the paper latency protocol")
    model.eval()
    with torch.inference_mode():
        for _ in range(warmup):
            with torch.autocast("cuda", enabled=use_amp):
                model(waveform, tabular)
        torch.cuda.synchronize()
        samples = []
        for _ in range(repetitions):
            torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.autocast("cuda", enabled=use_amp):
                model(waveform, tabular)
            torch.cuda.synchronize()
            samples.append((time.perf_counter() - start) * 1000.0)
    return np.asarray(samples, dtype=float)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--signal_length", type=int, default=2500)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--repetitions", type=int, default=500)
    parser.add_argument("--batch_sizes", type=int, nargs="+", default=[1, 64])
    parser.add_argument("--amp", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.checkpoint.exists():
        raise FileNotFoundError(args.checkpoint)
    from ecg_project.models import ECGConvNeXt1D

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    if "model_config" not in checkpoint or "model" not in checkpoint:
        raise ValueError("Checkpoint must contain model_config and model")
    model = ECGConvNeXt1D(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    config = checkpoint["model_config"]
    input_channels = int(config.get("in_channels", 12))
    waveform = torch.zeros(1, input_channels, args.signal_length, device=device)
    tabular_dim = int(config.get("tabular_features", 0))
    tabular = torch.zeros(1, tabular_dim, device=device) if tabular_dim else None

    macs, module_table = count_model_macs(model, waveform, tabular)
    counts = parameter_counts(model)
    state_bytes = serialized_state_size_bytes(model.cpu())
    model = model.to(device)
    checkpoint_bytes = args.checkpoint.stat().st_size
    common_metrics = {
        **counts,
        "macs": macs,
        "flops_2x_macs": int(2 * macs),
        "checkpoint_size_bytes": int(checkpoint_bytes),
        "checkpoint_size_mib": bytes_to_megabytes(checkpoint_bytes),
        "state_dict_size_bytes": state_bytes,
        "state_dict_size_mib": bytes_to_megabytes(state_bytes),
        "warmup_iterations": args.warmup,
        "timed_iterations": args.repetitions,
        "signal_length": args.signal_length,
        "input_channels": input_channels,
        "tabular_features": tabular_dim,
        "precision": "amp" if args.amp else "fp32",
    }
    metric_rows = []
    latency_rows = []
    for batch_size in args.batch_sizes:
        batch_waveform = torch.zeros(
            batch_size, input_channels, args.signal_length, device=device
        )
        batch_tabular = (
            torch.zeros(batch_size, tabular_dim, device=device) if tabular_dim else None
        )
        torch.cuda.reset_peak_memory_stats(device)
        timing = measure_latency(
            model,
            batch_waveform,
            batch_tabular,
            args.warmup,
            args.repetitions,
            args.amp,
        )
        peak_memory_bytes = int(torch.cuda.max_memory_allocated(device))
        latency = summarize_latency_ms(timing)
        metric_rows.append(
            {
                **common_metrics,
                **latency,
                "batch_size": int(batch_size),
                "peak_gpu_memory_bytes": peak_memory_bytes,
                "peak_gpu_memory_mib": bytes_to_megabytes(peak_memory_bytes),
            }
        )
        latency_rows.extend(
            {"batch_size": int(batch_size), "iteration": index, "latency_ms": value}
            for index, value in enumerate(timing)
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metric_rows).to_csv(args.output_dir / "efficiency_metrics.csv", index=False)
    pd.DataFrame(latency_rows).to_csv(args.output_dir / "latency_samples.csv", index=False)
    module_table.to_csv(args.output_dir / "macs_by_module.csv", index=False)
    (args.output_dir / "efficiency_summary.json").write_text(
        json.dumps(
            {
                "common": common_metrics,
                "batch_results": metric_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    environment = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "checkpoint": str(args.checkpoint),
        "model_config": config,
        "mac_scope": "Conv1d and Linear only; normalization, activations, pooling reductions, interpolation, sigmoid, FiLM/GRN elementwise operations excluded",
        "flop_convention": "FLOPs = 2 * MACs",
        "data_transfer_timed": False,
        "latency_scope": "model forward pass with resident CUDA tensors and explicit synchronization",
    }
    (args.output_dir / "environment.json").write_text(
        json.dumps(environment, indent=2), encoding="utf-8"
    )
    report = [
        "# Model efficiency v1",
        "",
        f"- Trainable parameters: {common_metrics['trainable_parameters']:,}",
        f"- MACs: {common_metrics['macs']:,}",
        f"- FLOPs (2 x MACs): {common_metrics['flops_2x_macs']:,}",
        f"- Checkpoint size: {common_metrics['checkpoint_size_mib']:.3f} MiB",
        f"- State-dict size: {common_metrics['state_dict_size_mib']:.3f} MiB",
        "",
        *[
            f"- Batch {row['batch_size']}: median={row['latency_median_ms']:.4f} ms, "
            f"P95={row['latency_p95_ms']:.4f} ms, peak GPU memory={row['peak_gpu_memory_mib']:.1f} MiB"
            for row in metric_rows
        ],
        "",
        "MAC scope: Conv1d and Linear only. Normalization, activations, pooling reductions, interpolation, sigmoid, and elementwise FiLM/GRN operations are excluded.",
    ]
    (args.output_dir / "efficiency_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"common": common_metrics, "batch_results": metric_rows}, indent=2))


if __name__ == "__main__":
    main()
