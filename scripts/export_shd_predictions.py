#!/usr/bin/env python
"""Export aligned EchoNext validation/test predictions for formal SHD analysis."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for search_path in (PROJECT_ROOT, SRC_ROOT):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from ecg_project.data.datasets import (
    ECHONEXT_LABEL_COLS,
    LEAD_SETTINGS,
    EchoNextNpyDataset,
    LeadSelectionDataset,
)
from ecg_project.models import ECGConvNeXt1D


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return SHA-256 without loading a file into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def stable_split_hash(
    row_id: np.ndarray,
    labels: np.ndarray,
    patient_ids: np.ndarray | None = None,
) -> str:
    """Hash ordered row identifiers, patient identifiers, and labels."""
    digest = hashlib.sha256()
    digest.update(np.asarray(row_id, dtype=np.int64).tobytes())
    if patient_ids is not None:
        digest.update("\n".join(np.asarray(patient_ids, dtype=str).tolist()).encode("utf-8"))
    digest.update(np.asarray(labels, dtype=np.float32).tobytes())
    return digest.hexdigest()


def load_model(checkpoint_path: Path, device: torch.device) -> ECGConvNeXt1D:
    """Load the exact architecture embedded in a downstream checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "model" not in checkpoint or "model_config" not in checkpoint:
        raise ValueError(f"Checkpoint lacks model/model_config: {checkpoint_path}")
    model = ECGConvNeXt1D(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    return model


@torch.inference_mode()
def predict(
    model: ECGConvNeXt1D, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    """Return ordered probabilities and targets from a nonshuffled loader."""
    probabilities: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for waveform, tabular, target in loader:
        logits = model(
            waveform.to(device, non_blocking=True),
            tabular.to(device, non_blocking=True),
        )
        probabilities.append(torch.sigmoid(logits).cpu().numpy())
        labels.append(target.numpy())
    return np.vstack(probabilities), np.vstack(labels)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export an auditable EchoNext prediction bundle",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--split", choices=["val", "test"], required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data/echonext"))
    parser.add_argument("--csv-name", default="echonext_metadata_100k.csv")
    parser.add_argument("--wave-name", required=True)
    parser.add_argument("--tab-name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--lead-setting", choices=sorted(LEAD_SETTINGS), default="full12")
    parser.add_argument(
        "--lead-input-mode", choices=["zero_fill", "slice"], default="zero_fill"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    csv_path = args.data_dir / args.csv_name
    wave_path = args.data_dir / args.wave_name
    tab_path = args.data_dir / args.tab_name
    for input_path in (args.checkpoint, csv_path, wave_path, tab_path):
        if not input_path.exists():
            raise FileNotFoundError(input_path)

    metadata = pd.read_csv(csv_path)
    metadata = metadata.loc[metadata["split"] == args.split].reset_index(drop=True)
    dataset = EchoNextNpyDataset(
        csv_path,
        wave_path,
        tab_path,
        args.split,
        ECHONEXT_LABEL_COLS,
        None,
    )
    if args.lead_setting != "full12":
        dataset = LeadSelectionDataset(dataset, args.lead_setting, args.lead_input_mode)
    if len(metadata) != len(dataset):
        raise RuntimeError("Metadata and waveform split lengths differ")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    model = load_model(args.checkpoint, device)
    probabilities, labels = predict(model, loader, device)
    metadata_labels = metadata[ECHONEXT_LABEL_COLS].to_numpy(dtype=np.float32)
    if not np.array_equal(labels, metadata_labels):
        raise RuntimeError("Prediction loader labels do not align with metadata labels")

    row_id = np.arange(len(metadata), dtype=np.int64)
    patient_cluster_index = pd.factorize(metadata["patient_key"], sort=True)[0].astype(np.int64)
    patient_ids = metadata["patient_key"].fillna("missing").astype(str).to_numpy(dtype="U96")
    record_ids = metadata["ecg_key"].fillna("missing").astype(str).to_numpy(dtype="U96")
    split_hash = stable_split_hash(row_id, labels, patient_ids)
    checkpoint_sha256 = sha256_file(args.checkpoint)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        row_id=row_id,
        patient_id=patient_ids,
        record_id=record_ids,
        patient_cluster_index=patient_cluster_index,
        split=np.asarray([args.split] * len(metadata), dtype="U8"),
        labels=labels.astype(np.float32),
        probabilities=probabilities.astype(np.float32),
        label_names=np.asarray(ECHONEXT_LABEL_COLS, dtype="U96"),
        age_at_ecg=metadata["age_at_ecg"].to_numpy(dtype=np.float32),
        sex=metadata["sex"].fillna("missing").astype(str).to_numpy(dtype="U32"),
        location_setting=(
            metadata["location_setting"].fillna("missing").astype(str).to_numpy(dtype="U64")
        ),
        checkpoint_sha256=np.asarray(checkpoint_sha256),
        split_hash=np.asarray(split_hash),
        seed=np.asarray(args.seed, dtype=np.int64),
        lead_setting=np.asarray(args.lead_setting),
        lead_input_mode=np.asarray(args.lead_input_mode),
    )
    manifest = {
        "output": str(args.output),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "seed": args.seed,
        "split": args.split,
        "split_hash": split_hash,
        "n_rows": int(len(labels)),
        "n_unique_patients": int(np.unique(patient_cluster_index).size),
        "one_row_per_patient": bool(np.unique(patient_cluster_index).size == len(labels)),
        "n_labels": int(labels.shape[1]),
        "lead_setting": args.lead_setting,
        "lead_input_mode": args.lead_input_mode,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": str(device),
        "classification_rule": "probability >= threshold",
    }
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
