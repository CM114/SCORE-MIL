#!/usr/bin/env python3
"""Create matched-budget training-only candidate manifests from existing tiles.

The sampler only considers numeric tile coordinates. It ignores normal/
abnormal selections, text annotations and prediction files. Paths include the
hospital's folder structure and must never be encoded as model features.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict, deque
from pathlib import Path
from statistics import mean, median
from typing import Sequence

import numpy as np


def _parse_coordinate(name: str):
    match = re.fullmatch(r"(\d+)_(\d+)\.(?:jpg|jpeg|png)", str(name), re.IGNORECASE)
    return (int(match.group(1)), int(match.group(2))) if match else None


def select_candidates(filenames: Sequence[str], mode: str, budget: int, bins: int, seed: int) -> list[dict]:
    """Select unique tiles from filenames, without case labels or images."""

    if budget <= 0 or bins <= 0 or mode not in {"random", "spatial"}:
        raise ValueError("invalid sampling mode, budget, or number of bins")
    records = []
    seen = set()
    for name in sorted(str(value) for value in filenames):
        coords = _parse_coordinate(name)
        if coords is None:
            continue
        if coords in seen:
            raise ValueError(f"duplicate tile coordinate {coords}")
        seen.add(coords)
        records.append({"filename": name, "grid_x": coords[0], "grid_y": coords[1]})
    records.sort(key=lambda row: (row["grid_y"], row["grid_x"], row["filename"]))
    if len(records) < budget:
        raise ValueError(f"insufficient candidate budget: {len(records)} < {budget}")
    width = max(row["grid_x"] for row in records) + 1
    height = max(row["grid_y"] for row in records) + 1
    for row in records:
        row["bin_x"] = min(bins - 1, row["grid_x"] * bins // width)
        row["bin_y"] = min(bins - 1, row["grid_y"] * bins // height)
    rng = np.random.default_rng(seed)
    if mode == "random":
        indices = rng.choice(len(records), size=budget, replace=False).tolist()
    else:
        by_bin = defaultdict(list)
        for index, row in enumerate(records):
            by_bin[(row["bin_y"], row["bin_x"])].append(index)
        bin_keys = sorted(by_bin)
        if len(bin_keys) > budget:
            selected_bins = sorted(rng.choice(len(bin_keys), size=budget, replace=False).tolist())
            bin_keys = [bin_keys[index] for index in selected_bins]
        indices = [choices[int(rng.integers(len(choices)))] for key in bin_keys for choices in [by_bin[key]]]
        if len(indices) < budget:
            selected = set(indices)
            remaining = [i for i in range(len(records)) if i not in selected]
            indices.extend(int(value) for value in rng.choice(remaining, size=budget - len(indices), replace=False))
    return [records[index] for index in indices]


def _select_training_cases(protocol: dict, max_cases: int) -> list[dict]:
    """Round-robin known train cases across source/class for the pilot audit."""

    unknown = str(protocol["unknown_class"])
    groups = defaultdict(list)
    for row in protocol["train_rows"]:
        if str(row["wsl_class"]) == unknown:
            raise ValueError("training protocol includes held-out label")
        groups[(str(row["hospital"]), str(row["wsl_class"]))].append(dict(row))
    queues = {key: deque(sorted(value, key=lambda row: str(row["case_id"]))) for key, value in sorted(groups.items())}
    selected = []
    limit = max_cases if max_cases > 0 else sum(len(v) for v in queues.values())
    while len(selected) < limit and any(queues.values()):
        for key in sorted(queues):
            if queues[key] and len(selected) < limit:
                selected.append(queues[key].popleft())
    return selected


def _case_seed(global_seed: int, case_id: str) -> int:
    digest = hashlib.sha256(f"{global_seed}:{case_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _write_csv(path: Path, rows: Sequence[dict]) -> None:
    if not rows:
        raise ValueError(f"no candidate rows to write: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> dict:
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    cases = _select_training_cases(protocol, args.max_cases)
    if not cases:
        raise ValueError("no eligible training cases")
    manifests = {"random": [], "spatial": []}
    audits = []
    errors = []
    for index, row in enumerate(cases, 1):
        case_id = str(row["case_id"])
        image_dir = args.wsl_root / str(row["wsl_class"]) / str(row["hospital"]) / case_id / "img"
        try:
            names = [path.name for path in image_dir.iterdir() if path.is_file()]
            selected = {
                policy: select_candidates(names, policy, args.budget, args.bins, _case_seed(args.seed, case_id))
                for policy in ("random", "spatial")
            }
            for policy, items in selected.items():
                for candidate_index, item in enumerate(items):
                    manifests[policy].append({
                        "case_id": case_id,
                        "role": "train",
                        "wsl_class": str(row["wsl_class"]),
                        "hospital": str(row["hospital"]),
                        "candidate_index": candidate_index,
                        "grid_x": item["grid_x"],
                        "grid_y": item["grid_y"],
                        "bin_x": item["bin_x"],
                        "bin_y": item["bin_y"],
                        "image_path": str(image_dir / item["filename"]),
                    })
            random_names = {item["filename"] for item in selected["random"]}
            spatial_names = {item["filename"] for item in selected["spatial"]}
            audits.append({
                "case_id": case_id,
                "hospital": str(row["hospital"]),
                "wsl_class": str(row["wsl_class"]),
                "available_coordinate_tiles": sum(_parse_coordinate(name) is not None for name in names),
                "random_occupied_bins": len({(item["bin_x"], item["bin_y"]) for item in selected["random"]}),
                "spatial_occupied_bins": len({(item["bin_x"], item["bin_y"]) for item in selected["spatial"]}),
                "overlap_count": len(random_names & spatial_names),
            })
        except (OSError, ValueError) as exc:
            errors.append({"case_id": case_id, "error": f"{type(exc).__name__}: {exc}"})
        if index % 25 == 0 or index == len(cases):
            print(json.dumps({"event": "progress", "processed_cases": index, "total_cases": len(cases), "case_errors": len(errors)}, ensure_ascii=False), flush=True)
    if not audits:
        raise RuntimeError("no training cases were successfully sampled")
    args.out_dir.mkdir(parents=True, exist_ok=False)
    _write_csv(args.out_dir / "random_candidates.csv", manifests["random"])
    _write_csv(args.out_dir / "spatial_candidates.csv", manifests["spatial"])
    _write_csv(args.out_dir / "case_audit.csv", audits)
    if errors:
        _write_csv(args.out_dir / "case_errors.csv", errors)
    summary = {
        "protocol": str(args.protocol),
        "selection_scope": "known training cases only",
        "target_unknown_label_images_accessed": False,
        "validation_or_test_images_accessed": False,
        "patient_id_independence_unconfirmed": True,
        "case_selection": "deterministic round-robin source/class selection for pilot only",
        "tile_selection": "filename coordinates only; no diagnostic labels, prediction scores, or image content",
        "path_policy": "class/source directory names are used only for file location and case metadata, never image features",
        "budget_per_case_per_policy": args.budget,
        "grid_bins": args.bins,
        "seed": args.seed,
        "selected_cases": len(cases),
        "successful_cases": len(audits),
        "error_count": len(errors),
        "successful_classes": dict(Counter(row["wsl_class"] for row in audits)),
        "successful_sources": dict(Counter(row["hospital"] for row in audits)),
        "random_candidate_rows": len(manifests["random"]),
        "spatial_candidate_rows": len(manifests["spatial"]),
        "median_available_tiles": float(median(row["available_coordinate_tiles"] for row in audits)),
        "mean_random_occupied_bins": float(mean(row["random_occupied_bins"] for row in audits)),
        "mean_spatial_occupied_bins": float(mean(row["spatial_occupied_bins"] for row in audits)),
        "mean_random_spatial_overlap": float(mean(row["overlap_count"] for row in audits)),
        "interpretation_boundary": "coordinate coverage only; no model training, cell count or lesion detection has been evaluated",
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wsl-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-cases", type=int, default=128)
    parser.add_argument("--budget", type=int, default=64)
    parser.add_argument("--bins", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261030)
    args = parser.parse_args()
    summary = run(args)
    print(json.dumps({"event": "complete", **summary}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
