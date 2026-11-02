#!/usr/bin/env python3
"""Audit byte-identical JPEGs across selected development train/validation.

Exact old frozen feature equality is used as a cheap candidate index, followed
by JPEG SHA-256 verification.  This can miss duplicate images when numerical
feature outputs differ, and cannot establish patient independence. Only
aggregate counts are saved; no images or case IDs leave the server.
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

from ndpi_mil_baseline import group_feature_manifest, load_case_bag, read_csv
from score_mil_dev_candidate_manifest import development_cases
from score_mil_tile_feature_pilot import content_fraction_rgb


def feature_digest(vector):
    x = np.ascontiguousarray(vector, dtype=np.float32)
    if x.ndim != 1 or not np.isfinite(x).all():
        raise ValueError("feature vector must be finite and one-dimensional")
    return hashlib.sha256(x.tobytes()).digest()


def match_byte_groups(train_records, validation_records, byte_hash):
    train = defaultdict(list)
    validation = defaultdict(list)
    for record in train_records:
        train[byte_hash(record)].append(record)
    for record in validation_records:
        validation[byte_hash(record)].append(record)
    return [(train[key], validation[key]) for key in train.keys() & validation.keys()]


def _file_sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.digest()


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
    old_summary = json.loads((args.previous / "features" / "summary.json").read_text(encoding="utf-8"))
    if old_summary["outer_unknown_images_accessed"] or old_summary["outer_test_images_accessed"]:
        raise ValueError("old features violate development boundary")
    groups = group_feature_manifest(read_csv(args.previous / "features" / "spatial" / "feature_manifest.csv"))
    image_paths = {}
    with (args.previous / "candidates" / "spatial_candidates.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            cid = str(row["case_id"])
            if cid not in selected_by_id:
                continue
            ref = selected_by_id[cid]
            if any(str(row[k]) != str(ref[k]) for k in ("role", "wsl_class", "hospital", "binary_label")):
                raise ValueError("candidate metadata mismatch")
            key = (cid, int(row["candidate_index"]))
            if key in image_paths:
                raise ValueError("duplicate candidate index")
            image_paths[key] = Path(row["image_path"])
    if len(image_paths) != len(selected) * 64:
        raise ValueError("selected spatial candidates are incomplete")
    train_index = defaultdict(list)
    val_index = defaultdict(list)
    for i, case in enumerate(selected, 1):
        cid = str(case["case_id"])
        bag, _, _, meta = load_case_bag(groups[cid])
        if bag.shape != (64, 2048) or str(meta["split"]) != case["role"] or str(meta["wsl_class"]) != case["wsl_class"]:
            raise ValueError("old feature case alignment mismatch")
        target = train_index if case["role"] == "train" else val_index
        for index, feature in enumerate(bag):
            target[feature_digest(feature)].append((cid, index))
        if i % 100 == 0 or i == len(selected):
            print(json.dumps({"event": "duplicate_feature_index_progress", "cases": i, "total": len(selected)}), flush=True)
    overlap = set(train_index) & set(val_index)
    byte_cache = {}
    def byte_hash(key):
        if key not in byte_cache:
            byte_cache[key] = _file_sha(image_paths[key])
        return byte_cache[key]
    verified_groups = []
    for feat in overlap:
        verified_groups.extend(match_byte_groups(train_index[feat], val_index[feat], byte_hash))
    matched_train, matched_val = set(), set()
    train_cases, val_cases = set(), set()
    confirmed_group_content = Counter()
    for tr, va in verified_groups:
        matched_train.update(tr)
        matched_val.update(va)
        train_cases.update(x[0] for x in tr)
        val_cases.update(x[0] for x in va)
        with Image.open(image_paths[tr[0]]) as handle:
            rgb = handle.convert("RGB").resize((224, 224), resample=Image.Resampling.BILINEAR)
        proxy = content_fraction_rgb(np.asarray(rgb))
        confirmed_group_content["below_0.02" if proxy < 0.02 else "at_least_0.02"] += 1
    def group_counts(case_ids):
        return {"by_label": dict(Counter(selected_by_id[cid]["wsl_class"] for cid in case_ids)),
                "by_source": dict(Counter(selected_by_id[cid]["hospital"] for cid in case_ids))}
    result = {
        "scope": "selected known development train and validation; existing spatial-64 JPEG candidates",
        "outer_unknown_or_outer_test_images_accessed": False,
        "train_cases": sum(r["role"] == "train" for r in selected),
        "validation_cases": sum(r["role"] == "validation" for r in selected),
        "candidate_parent_tiles_per_case": 64,
        "cross_role_identical_feature_hash_groups": len(overlap),
        "jpeg_files_read_for_verification": len(byte_cache),
        "cross_role_identical_jpeg_byte_groups": len(verified_groups),
        "matched_train_tiles": len(matched_train), "matched_validation_tiles": len(matched_val),
        "matched_train_cases": len(train_cases), "matched_validation_cases": len(val_cases),
        "matched_train_case_groups": group_counts(train_cases), "matched_validation_case_groups": group_counts(val_cases),
        "confirmed_group_rgb_content_proxy": dict(confirmed_group_content),
        "limitations": ["exact feature equality is only a candidate filter and can miss byte-identical images with numerical differences", "images not selected in spatial-64 were not checked", "exact JPEG bytes do not prove two cases belong to one patient", "blank or shared background tiles can match", "RGB proxy is not lesion or cell annotation", "near duplicates and patient independence remain unverified"],
    }
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(json.dumps({"event": "duplicate_audit_complete", **result}, ensure_ascii=False), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--previous", type=Path, required=True)
    p.add_argument("--experiment", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    run(p.parse_args())


if __name__ == "__main__":
    main()
