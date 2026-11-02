#!/usr/bin/env python3
"""Small, training-only feature-scale feasibility audit on existing JPEGs.

For each selected parent tile, compare the same frozen encoder applied to the
whole tile resized to 224 with the mean of 25 native 224-pixel subtiles. No
model is trained, no outer unknown/test images are read, and aggregate output
cannot establish diagnostic accuracy or cell presence.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from ndpi_mil_baseline import group_feature_manifest, load_case_bag, read_csv
from score_mil_cytology_scale import UNICAS_SHA256, _sha256, build_unicas_model, extract_native_crops
from score_mil_tile_feature_pilot import _build_encoder, content_fraction_rgb


def choose_training_cases(outer: dict, per_group: int) -> list[dict]:
    if per_group <= 0:
        raise ValueError("per-group count must be positive")
    unknown = str(outer["unknown_class"])
    known = {str(x) for x in outer["known_classes"]}
    test_ids = {str(x) for x in outer["test_ids"]}
    groups = defaultdict(list)
    seen = set()
    for original in outer["train_rows"]:
        row = dict(original)
        cid, label = str(row["case_id"]), str(row["wsl_class"])
        if label == unknown or label not in known:
            raise ValueError("unknown or unsupported class in training rows")
        if cid in test_ids:
            raise ValueError("training case occurs in outer test")
        if cid in seen:
            raise ValueError("duplicate training case")
        seen.add(cid)
        groups[(str(row["hospital"]), label)].append(row)
    result = []
    for _, rows in sorted(groups.items()):
        rows.sort(key=lambda r: hashlib.sha256(str(r["case_id"]).encode("utf-8")).digest())
        result.extend(rows[:per_group])
    if not result:
        raise ValueError("no known training cases")
    return result


def choose_case_indices(case_id: str, seed: int, budget: int, count: int) -> list[int]:
    if budget <= 0 or count <= 0 or count > budget:
        raise ValueError("invalid fixed candidate budget")
    digest = hashlib.sha256(f"{seed}:{case_id}".encode("utf-8")).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
    return sorted(int(x) for x in rng.choice(budget, size=count, replace=False))


def cosine_similarity(first, second) -> float:
    a, b = np.asarray(first, dtype=np.float64), np.asarray(second, dtype=np.float64)
    if a.ndim != 1 or a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("embeddings must be finite aligned vectors")
    norm = float(np.linalg.norm(a) * np.linalg.norm(b))
    if norm == 0:
        raise ValueError("embedding norms must be nonzero")
    return float(np.clip(np.dot(a, b) / norm, -1.0, 1.0))


def cache_agreement(current, reference, relative_tolerance: float = 0.01, minimum_cosine: float = 0.9999) -> dict:
    a, b = np.asarray(current, dtype=np.float64), np.asarray(reference, dtype=np.float64)
    cosine = cosine_similarity(a, b)
    relative_l2 = float(np.linalg.norm(a - b) / np.linalg.norm(b))
    if relative_l2 > relative_tolerance or cosine < minimum_cosine:
        raise RuntimeError("whole-resized feature does not agree with cached feature")
    return {"max_abs_difference": float(np.max(np.abs(a - b))), "relative_l2": relative_l2, "cosine": cosine}


def _encode_views(model, views, device, batch_size):
    import torch

    # Match the original feature extractor's CPU-side normalization, and pad
    # the last batch to keep the numerical kernel shape fixed.
    mean = torch.tensor([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    outputs = []
    with torch.inference_mode():
        for offset in range(0, len(views), batch_size):
            part = views[offset:offset + batch_size]
            count = len(part)
            part = [*part, *([part[0]] * (batch_size - count))]
            tensors = []
            for image in part:
                array = np.asarray(image, dtype=np.uint8).copy()
                x = torch.from_numpy(array).permute(2, 0, 1).to(dtype=torch.float32).div(255.0)
                tensors.append((x - mean) / std)
            outputs.append(model(torch.stack(tensors).to(device=device)).cpu().numpy()[:count])
    return np.concatenate(outputs, axis=0)


def _summary(values):
    values = np.asarray(values, dtype=np.float64)
    return {"n": len(values), "min": float(np.min(values)), "q25": float(np.quantile(values, 0.25)),
            "median": float(np.median(values)), "q75": float(np.quantile(values, 0.75)), "max": float(np.max(values))}


def run(args):
    import timm
    import torch

    outer = json.loads(args.protocol.read_text(encoding="utf-8"))
    cases = choose_training_cases(outer, args.per_group)
    by_id = {str(c["case_id"]): c for c in cases}
    index_by_id = {cid: choose_case_indices(cid, args.seed, 64, args.tiles_per_case) for cid in by_id}
    rows = []
    with args.candidate_csv.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            cid = str(row["case_id"])
            if cid in by_id and int(row["candidate_index"]) in index_by_id[cid]:
                if row["role"] != "train" or any(str(row[k]) != str(by_id[cid][k]) for k in ("wsl_class", "hospital")):
                    raise ValueError("candidate metadata violates training-only scope")
                rows.append(row)
    expected = {(cid, idx) for cid in by_id for idx in index_by_id[cid]}
    if {(r["case_id"], int(r["candidate_index"])) for r in rows} != expected or len(rows) != len(expected):
        raise ValueError("sampled parent candidates are incomplete or duplicated")
    rows.sort(key=lambda r: (r["case_id"], int(r["candidate_index"])))
    original_summary = json.loads((args.previous / "features" / "summary.json").read_text(encoding="utf-8"))
    if original_summary["outer_unknown_images_accessed"] or original_summary["outer_test_images_accessed"]:
        raise ValueError("previous features violate outer development scope")
    groups = group_feature_manifest(read_csv(args.previous / "features" / "spatial" / "feature_manifest.csv"))
    cached = {}
    for cid in by_id:
        bag, _, _, meta = load_case_bag(groups[cid])
        if bag.shape != (64, 2048) or meta["split"] != "train" or meta["wsl_class"] != by_id[cid]["wsl_class"]:
            raise ValueError("cached feature metadata or shape mismatch")
        cached[cid] = bag
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the small encoder audit")
    torch.cuda.set_device(device)
    if torch.cuda.mem_get_info(device)[0] < 8 * 1024 ** 3:
        raise RuntimeError("less than 8 GiB free GPU memory")
    torch.set_num_threads(2)
    if _sha256(args.checkpoint) != UNICAS_SHA256:
        raise ValueError("UniCAS checkpoint SHA-256 mismatch")
    args.out_dir.mkdir(parents=True, exist_ok=False)
    parents = []
    for row in rows:
        with Image.open(row["image_path"]) as handle:
            rgb = handle.convert("RGB")
        if rgb.size != (1024, 1024):
            raise ValueError("sampled parent tile is not 1024×1024")
        parents.append((row, rgb))
    details = []
    reports = {}
    for backbone in ("resnet50", "unicas"):
        if backbone == "resnet50":
            model, ckpt = _build_encoder(device)
            if _sha256(ckpt) != original_summary["encoder_checkpoint_sha256"]:
                raise ValueError("ResNet checkpoint differs from cached feature source")
        else:
            model = build_unicas_model()
            state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
            model.load_state_dict(state, strict=True)
            del state
            model.to(device).eval()
        started = time.perf_counter()
        local = []
        for number, (row, rgb) in enumerate(parents, 1):
            whole = rgb.resize((224, 224), resample=Image.Resampling.BILINEAR)
            crops = extract_native_crops(rgb, crop_size=224, grid_size=5)
            effective_batch = 64 if backbone == "resnet50" else args.batch_size
            features = _encode_views(model, [whole, *crops], device, effective_batch)
            dimension = 2048 if backbone == "resnet50" else 1024
            if features.shape != (26, dimension) or not np.isfinite(features).all():
                raise ValueError("unexpected encoder output")
            agreement = {"max_abs_difference": None, "relative_l2": None, "cosine": None}
            if backbone == "resnet50":
                reference = cached[row["case_id"]][int(row["candidate_index"])]
                agreement = cache_agreement(features[0], reference)
            local.append({"case_id": row["case_id"], "candidate_index": int(row["candidate_index"]),
                          "wsl_class": row["wsl_class"], "hospital": row["hospital"], "backbone": backbone,
                          "whole_vs_native_mean_cosine": cosine_similarity(features[0], features[1:].mean(axis=0)),
                          "rgb_content_proxy": content_fraction_rgb(np.asarray(whole)),
                          "resnet_cache_max_abs_diff": agreement["max_abs_difference"],
                          "resnet_cache_relative_l2": agreement["relative_l2"], "resnet_cache_cosine": agreement["cosine"]})
            if number % 10 == 0 or number == len(parents):
                print(json.dumps({"event": "scale_pilot_progress", "backbone": backbone, "parents": number, "total": len(parents)}), flush=True)
        reports[backbone] = {
            "whole_vs_native_mean_cosine": _summary([r["whole_vs_native_mean_cosine"] for r in local]),
            "low_rgb_content_proxy_count": sum(r["rgb_content_proxy"] < 0.02 for r in local),
            "elapsed_seconds": time.perf_counter() - started,
            "max_resnet_cache_difference": max(r["resnet_cache_max_abs_diff"] for r in local) if backbone == "resnet50" else None,
            "max_resnet_cache_relative_l2": max(r["resnet_cache_relative_l2"] for r in local) if backbone == "resnet50" else None,
            "min_resnet_cache_cosine": min(r["resnet_cache_cosine"] for r in local) if backbone == "resnet50" else None,
        }
        details.extend(local)
        del model
        torch.cuda.empty_cache()
    with (args.out_dir / "case_details.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(details[0])); writer.writeheader(); writer.writerows(details)
    result = {
        "scope": "known outer-training JPEGs only; no model training or output to external services",
        "outer_unknown_images_accessed": False, "outer_test_images_accessed": False,
        "case_count": len(cases), "parent_tiles": len(parents), "tiles_per_case": args.tiles_per_case,
        "case_selection": "deterministic up to two per source/class; convenience sample",
        "tile_selection": "case-seeded indices from existing spatial-64 candidates, without image labels",
        "class_counts": dict(Counter(str(c["wsl_class"]) for c in cases)), "source_counts": dict(Counter(str(c["hospital"]) for c in cases)),
        "native_crops_per_parent": 25, "encoder_inputs_per_parent_per_model": 26,
        "checkpoint_sha256": UNICAS_SHA256, "timm_version": timm.__version__, "torch_version": torch.__version__,
        "device": str(device), "batch_size_unicas": args.batch_size, "batch_size_resnet50": 64, "seed": args.seed, "backbones": reports,
        "limitations": ["cosine distance is not accuracy or proof of lesion visibility", "image scores and labels not expert verified", "repeated parents from same case are not independent", "two backbone embedding spaces are not directly comparable", "native mean pooling can dilute rare cells"],
    }
    (args.out_dir / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"event": "scale_pilot_complete", **result}, ensure_ascii=False), flush=True)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--previous", type=Path, required=True)
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--candidate-csv", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--per-group", type=int, default=2)
    p.add_argument("--tiles-per-case", type=int, default=2)
    p.add_argument("--seed", type=int, default=20261031)
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--batch-size", type=int, default=8)
    args = p.parse_args()
    if args.tiles_per_case <= 0 or args.tiles_per_case > 64 or args.batch_size <= 0:
        raise ValueError("invalid pilot size")
    run(args)


if __name__ == "__main__":
    main()
