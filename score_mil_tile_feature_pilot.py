#!/usr/bin/env python3
"""Extract matched ResNet features from training-only existing JPEG tiles.

This is a feasibility pilot, not a model performance experiment. Image paths
are read as filesystem locations, never encoded as prediction features.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


def content_fraction_rgb(image) -> float:
    """Return an image-content proxy, not a cell or pathology label."""

    rgb = np.asarray(image)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or not rgb.size:
        raise ValueError("RGB image must be nonempty [height,width,3]")
    values = rgb.astype(np.float32)
    dark = values.mean(axis=2) < 180.0
    colored = (values.max(axis=2) - values.min(axis=2)) > 18.0
    return float(np.mean(dark | colored))


def validate_candidate_rows(rows: Sequence[Mapping[str, str]], expected_per_case: int) -> None:
    if not rows or expected_per_case <= 0:
        raise ValueError("nonempty candidates and positive per-case budget required")
    grouped = defaultdict(list)
    seen = set()
    for row in rows:
        if str(row.get("role", "")) != "train":
            raise ValueError("pilot accepts train candidates only")
        cid = str(row.get("case_id", ""))
        index = int(row["candidate_index"])
        key = (cid, index)
        if not cid or key in seen:
            raise ValueError("empty case ID or duplicate candidate index")
        if not str(row.get("image_path", "")):
            raise ValueError("image path is empty")
        seen.add(key)
        grouped[cid].append(index)
    for cid, indices in grouped.items():
        if sorted(indices) != list(range(expected_per_case)):
            raise ValueError(f"case {cid} does not have the expected candidate indices")


def write_case_features(
    rows: Sequence[Mapping[str, str]],
    features: np.ndarray,
    binary_by_case: Mapping[str, str],
    output_dir: Path,
) -> list[dict]:
    """Write per-case features and preserve candidate index to row alignment."""

    array = np.asarray(features, dtype=np.float32)
    if array.ndim != 2 or len(rows) != len(array) or not np.isfinite(array).all():
        raise ValueError("features must be finite 2-D data aligned with rows")
    by_case = defaultdict(list)
    for index, row in enumerate(rows):
        by_case[str(row["case_id"])].append((int(row["candidate_index"]), index, row))
    case_dir = output_dir / "case_features"
    case_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for case_id, entries in sorted(by_case.items()):
        entries.sort(key=lambda entry: entry[0])
        if [entry[0] for entry in entries] != list(range(len(entries))):
            raise ValueError("candidate indices must be contiguous from zero")
        if case_id not in binary_by_case:
            raise ValueError("case binary label missing from original protocol")
        filename = hashlib.sha256(case_id.encode("utf-8")).hexdigest() + ".npy"
        feature_path = case_dir / filename
        if feature_path.exists():
            raise FileExistsError(str(feature_path))
        np.save(feature_path, array[[entry[1] for entry in entries]], allow_pickle=False)
        for feature_row, (candidate_index, _, row) in enumerate(entries):
            x, y = int(row["grid_x"]), int(row["grid_y"])
            manifest.append({
                "case_id": case_id,
                "split": "train",
                "binary_label": binary_by_case[case_id],
                "wsl_class": row["wsl_class"],
                "hospital": row["hospital"],
                "candidate_index": candidate_index,
                "grid_row": y,
                "grid_col": x,
                "level": 0,
                "tile_size": 1024,
                "x_level0": x * 1024,
                "y_level0": y * 1024,
                "feature_row": feature_row,
                "feature_path": str(feature_path.resolve()),
                "feature_dim": int(array.shape[1]),
                "feature_status": "ok",
                "feature_error": "",
                "feature_source": "hospital_img_jpeg_resnet50_imagenet1k_v2",
            })
    return manifest


class TileDataset(Dataset):
    def __init__(self, rows: Sequence[Mapping[str, str]]) -> None:
        self.rows = list(rows)
        self.mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        with Image.open(self.rows[index]["image_path"]) as handle:
            rgb = handle.convert("RGB").resize((224, 224), resample=Image.Resampling.BILINEAR)
            array = np.asarray(rgb, dtype=np.uint8).copy()
        content = content_fraction_rgb(array)
        tensor = torch.from_numpy(array).permute(2, 0, 1).to(dtype=torch.float32).div(255.0)
        return (tensor - self.mean) / self.std, content


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _build_encoder(device: torch.device):
    from torchvision.models import ResNet50_Weights, resnet50

    weights = ResNet50_Weights.IMAGENET1K_V2
    checkpoint_path = Path(torch.hub.get_dir()) / "checkpoints" / Path(weights.url).name
    if not checkpoint_path.is_file():
        raise FileNotFoundError("required cached ResNet50 checkpoint not found; no automatic download attempted")
    model = resnet50(weights=weights)
    model.fc = torch.nn.Identity()
    model.to(device).eval()
    return model, checkpoint_path


def _extract_policy(rows, model, device, batch_size, workers, policy):
    loader = DataLoader(
        TileDataset(rows), batch_size=batch_size, shuffle=False, num_workers=workers,
        pin_memory=(device.type == "cuda"),
    )
    batches = []
    fractions = []
    processed = 0
    started = time.perf_counter()
    with torch.inference_mode():
        for tensors, content in loader:
            features = model(tensors.to(device=device, non_blocking=(device.type == "cuda")))
            batches.append(features.detach().cpu().numpy().astype(np.float32, copy=False))
            fractions.extend(float(value) for value in content.numpy())
            processed += len(tensors)
            if processed % 1024 == 0 or processed == len(rows):
                print(json.dumps({"event": "feature_progress", "policy": policy, "processed": processed, "total": len(rows)}, ensure_ascii=False), flush=True)
    array = np.concatenate(batches, axis=0)
    if array.shape != (len(rows), 2048) or not np.isfinite(array).all():
        raise RuntimeError(f"unexpected/nonfinite features: {array.shape}")
    elapsed = time.perf_counter() - started
    quality = np.asarray(fractions, dtype=np.float64)
    return array, quality, {
        "image_count": len(rows),
        "feature_dim": int(array.shape[1]),
        "elapsed_seconds": round(elapsed, 2),
        "images_per_second": float(len(rows) / elapsed),
        "mean_content_fraction": float(quality.mean()),
        "median_content_fraction": float(np.median(quality)),
        "low_content_fraction_below_002": float(np.mean(quality < 0.02)),
        "proxy_boundary": "RGB pixel heuristic on resized JPEG; not cell count or diagnosis",
    }


def run(args: argparse.Namespace) -> dict:
    manifest_root = args.candidate_dir
    candidate_summary = json.loads((manifest_root / "summary.json").read_text(encoding="utf-8"))
    if candidate_summary["error_count"]:
        raise ValueError("candidate manifest contains case errors")
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    train_rows = {str(row["case_id"]): row for row in protocol["train_rows"]}
    unknown = str(protocol["unknown_class"])
    if any(str(row["wsl_class"]) == unknown for row in train_rows.values()):
        raise ValueError("held-out label appears in training rows")
    binary_by_case = {cid: str(row["binary_label"]) for cid, row in train_rows.items()}
    rows_by_policy = {}
    for policy in ("random", "spatial"):
        rows = _read_csv(manifest_root / f"{policy}_candidates.csv")
        validate_candidate_rows(rows, args.expected_per_case)
        cases = {row["case_id"] for row in rows}
        if not cases.issubset(train_rows):
            raise ValueError("candidate set contains a non-training case")
        for row in rows:
            original = train_rows[row["case_id"]]
            if row["wsl_class"] != original["wsl_class"] or row["hospital"] != original["hospital"]:
                raise ValueError("candidate case metadata differs from protocol")
        rows_by_policy[policy] = sorted(rows, key=lambda row: (row["case_id"], int(row["candidate_index"])))
    if {r["case_id"] for r in rows_by_policy["random"]} != {r["case_id"] for r in rows_by_policy["spatial"]}:
        raise ValueError("sampling strategies do not cover exactly the same cases")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device is not available")
    model, checkpoint_path = _build_encoder(device)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    summary = {
        "scope": "known training-case image tiles only; feature-extraction pilot",
        "held_out_class_images_accessed": False,
        "validation_or_test_images_accessed": False,
        "model_training_performed": False,
        "encoder": "torchvision ResNet50 ImageNet1K V2, frozen",
        "encoder_checkpoint_sha256": _sha256(checkpoint_path),
        "script_sha256": _sha256(Path(__file__).resolve()),
        "device": str(device),
        "batch_size": args.batch_size,
        "workers": args.workers,
        "preprocessing": "JPEG RGB; bilinear resize to 224x224; ImageNet mean/std",
        "comparison_caveat": "both policies use the same JPEG preprocessing; historical NDPI level-2 preprocessing differs",
        "policies": {},
    }
    for policy in ("random", "spatial"):
        rows = rows_by_policy[policy]
        feature_array, quality, policy_stats = _extract_policy(rows, model, device, args.batch_size, args.workers, policy)
        policy_dir = args.out_dir / policy
        policy_dir.mkdir(parents=True, exist_ok=False)
        feature_manifest = write_case_features(rows, feature_array, binary_by_case, policy_dir)
        _write_csv(policy_dir / "feature_manifest.csv", feature_manifest)
        grouped_quality = defaultdict(list)
        for row, value in zip(rows, quality):
            grouped_quality[row["case_id"]].append(float(value))
        case_quality = [
            {"case_id": cid, "mean_content_fraction": float(np.mean(values)), "low_content_patch_fraction": float(np.mean(np.asarray(values) < 0.02))}
            for cid, values in sorted(grouped_quality.items())
        ]
        _write_csv(policy_dir / "case_content_proxy.csv", case_quality)
        policy_stats["case_count"] = len(grouped_quality)
        policy_stats["feature_manifest"] = str((policy_dir / "feature_manifest.csv").resolve())
        summary["policies"][policy] = policy_stats
        (args.out_dir / "partial_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        print(json.dumps({"event": "policy_complete", "policy": policy, **policy_stats}, ensure_ascii=False), flush=True)
    (args.out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--expected-per-case", type=int, default=64)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.batch_size <= 0 or args.workers < 0:
        raise ValueError("batch size must be positive and workers non-negative")
    run(args)


if __name__ == "__main__":
    main()
