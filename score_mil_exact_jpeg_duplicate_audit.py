#!/usr/bin/env python3
"""Directly SHA-256 all selected JPEG files across development train/validation.

Only selected spatial-64 parent tiles are included. Byte equality does not
prove common patients or near-image duplication. Outputs contain aggregate
counts only; image paths and case identifiers are not exported.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from score_mil_cross_split_duplicate_audit import _file_sha
from score_mil_dev_candidate_manifest import development_cases
from score_mil_scale_factorial import validate_subset_candidates
from score_mil_tile_feature_pilot import content_fraction_rgb


def cross_role_groups(records):
    groups = defaultdict(lambda: {"train": [], "validation": []})
    for row in records:
        if row["role"] not in ("train", "validation"):
            raise ValueError("unexpected role")
        groups[row["sha256"]][row["role"]].append(row)
    return [g for g in groups.values() if g["train"] and g["validation"]]


def run(args):
    outer = json.loads(args.protocol.read_text(encoding="utf-8"))
    development = {str(r["case_id"]): r for r in development_cases(outer)}
    selected = json.loads((args.experiment / "selected_cases.json").read_text(encoding="utf-8"))
    config = json.loads((args.experiment / "config.json").read_text(encoding="utf-8"))
    selected_by_id = {str(r["case_id"]): r for r in selected}
    if len(selected_by_id) != len(selected) or not set(selected_by_id).issubset(development):
        raise ValueError("selected cases are duplicated or outside development")
    digest = hashlib.sha256("\n".join(sorted(selected_by_id)).encode("utf-8")).hexdigest()
    if digest != config["selected_case_set_sha256"]:
        raise ValueError("selection hash mismatch")
    for cid, row in selected_by_id.items():
        if any(str(row[k]) != str(development[cid][k]) for k in ("role", "wsl_class", "hospital", "binary_label")):
            raise ValueError("selection metadata mismatch")
    with args.candidate_csv.open(encoding="utf-8", newline="") as handle:
        rows = [r for r in csv.DictReader(handle) if str(r["case_id"]) in selected_by_id]
    validate_subset_candidates(rows, selected, args.wsl_root, 64)
    records, bytes_read = [], 0
    for i, row in enumerate(rows, 1):
        path = Path(row["image_path"])
        digest = _file_sha(path)
        bytes_read += path.stat().st_size
        records.append({"role": row["role"], "case_id": row["case_id"], "wsl_class": row["wsl_class"],
                        "hospital": row["hospital"], "image_path": path, "sha256": digest})
        if i % 5000 == 0 or i == len(rows):
            print(json.dumps({"event": "jpeg_hash_progress", "files": i, "total": len(rows)}, ensure_ascii=False), flush=True)
    groups = cross_role_groups(records)
    matched_train_cases, matched_val_cases = set(), set()
    matched_train_tiles, matched_val_tiles = 0, 0
    content = Counter()
    for group in groups:
        matched_train_tiles += len(group["train"])
        matched_val_tiles += len(group["validation"])
        matched_train_cases.update(r["case_id"] for r in group["train"])
        matched_val_cases.update(r["case_id"] for r in group["validation"])
        with Image.open(group["train"][0]["image_path"]) as image:
            rgb = image.convert("RGB").resize((224, 224), resample=Image.Resampling.BILINEAR)
        fraction = content_fraction_rgb(np.asarray(rgb))
        content["below_0.02" if fraction < 0.02 else "at_least_0.02"] += 1
    def case_groups(ids):
        return {"by_label": dict(Counter(selected_by_id[cid]["wsl_class"] for cid in ids)),
                "by_source": dict(Counter(selected_by_id[cid]["hospital"] for cid in ids))}
    result = {
        "scope": "selected known development train and validation; direct SHA-256 of all spatial-64 parent JPEGs",
        "outer_unknown_or_outer_test_images_accessed": False,
        "train_cases": sum(r["role"] == "train" for r in selected),
        "validation_cases": sum(r["role"] == "validation" for r in selected),
        "parent_tiles_per_case": 64, "jpeg_files_hashed": len(rows), "jpeg_bytes_read": bytes_read,
        "cross_role_identical_jpeg_byte_groups": len(groups), "matched_train_tiles": matched_train_tiles,
        "matched_validation_tiles": matched_val_tiles, "matched_train_cases": len(matched_train_cases),
        "matched_validation_cases": len(matched_val_cases),
        "matched_train_case_groups": case_groups(matched_train_cases), "matched_validation_case_groups": case_groups(matched_val_cases),
        "matched_group_rgb_content_proxy": dict(content),
        "limitations": ["only selected spatial-64 tiles checked", "exact byte match is not patient identification", "same pixel content in different encodings and near duplicates are not detected", "repeated visits and other tiles remain unexamined", "RGB proxy is not lesion or cell annotation"],
    }
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(json.dumps({"event": "exact_jpeg_audit_complete", **result}, ensure_ascii=False), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--experiment", type=Path, required=True)
    p.add_argument("--candidate-csv", type=Path, required=True)
    p.add_argument("--wsl-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    run(p.parse_args())


if __name__ == "__main__":
    main()
