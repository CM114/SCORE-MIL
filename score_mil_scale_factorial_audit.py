#!/usr/bin/env python3
"""Independently check one parent's view encoding and paired pilot results."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from score_mil_cytology_scale import build_unicas_model, extract_native_crops
from score_mil_scale_disagreement_pilot import _encode_views
from score_mil_tile_feature_pilot import _build_encoder


def compare(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("invalid alignment")
    rel = float(np.linalg.norm(a-b) / max(np.linalg.norm(b), 1e-12))
    if rel > 0.001:
        raise ValueError("independent feature recomputation mismatch")
    return {"relative_l2": rel, "max_abs_difference": float(np.max(np.abs(a-b)))}


def run(args):
    import torch
    experiment = args.experiment
    summary = json.loads((experiment / "training_summary.json").read_text(encoding="utf-8"))
    config = summary["config"]
    cases = json.loads((experiment / "selected_cases.json").read_text(encoding="utf-8"))
    train = {r["case_id"]: r for r in cases if r["role"] == "train"}
    val = {r["case_id"]: r for r in cases if r["role"] == "validation"}
    if set(train) & set(val) or len(train) != config["train_cases"] or len(val) != config["validation_cases"]:
        raise ValueError("selection role mismatch")
    groups = config["groups"]
    order = defaultdict(dict)
    max_metric_difference = 0.0
    for result in summary["results"]:
        pseudo, seed, group = result["pseudo_unknown"], result["seed"], result["group"]
        order[(pseudo, seed)][group] = result["train_order_sha256"]
        result_path = experiment / "training" / pseudo / str(seed) / group / "result.json"
        detail = json.loads(result_path.read_text(encoding="utf-8"))
        if pseudo in detail["train_class_counts"] or sum(detail["train_class_counts"].values()) != result["train_n"]:
            raise ValueError("pseudo label training exclusion failed")
        with (result_path.parent / "validation_predictions.csv").open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != len(val) or {r["case_id"] for r in rows} != set(val):
            raise ValueError("validation case set differs")
        for row in rows:
            ref = val[row["case_id"]]
            if any(row[k] != str(ref[k]) for k in ("wsl_class", "hospital")) or int(row["is_pseudo_unknown"]) != int(row["wsl_class"] == pseudo):
                raise ValueError("validation metadata mismatch")
        truth = np.array([int(r["is_pseudo_unknown"]) for r in rows])
        score = np.array([float(r["ordinal_fused"]) for r in rows])
        known = [r for r in rows if not int(r["is_pseudo_unknown"])]
        recomputed = [roc_auc_score(truth, score), average_precision_score(truth, score),
                      f1_score([r["wsl_class"] for r in known], [r["predicted_class"] for r in known],
                               labels=detail["known_order"], average="macro", zero_division=0)]
        stored = [detail["metrics"]["ordinal_fused"]["unknown_auroc"], detail["metrics"]["ordinal_fused"]["unknown_aupr"],
                  detail["metrics"]["known_classification"]["macro_f1"]]
        error = float(np.max(np.abs(np.array(recomputed) - np.array(stored))))
        max_metric_difference = max(max_metric_difference, error)
        if error > 1e-10:
            raise ValueError("stored metrics differ from recomputed predictions")
    if any(set(g) != set(groups) or len(set(g.values())) != 1 for g in order.values()):
        raise ValueError("paired groups or training order hashes differ")
    if len(summary["results"]) != 2 * len(config["seeds"]) * len(groups):
        raise ValueError("incomplete model count")

    cid = sorted(train)[0]
    with args.candidate_csv.open(encoding="utf-8", newline="") as handle:
        row = next(r for r in csv.DictReader(handle) if r["case_id"] == cid and int(r["candidate_index"]) == 0)
    with Image.open(row["image_path"]) as handle:
        rgb = handle.convert("RGB")
    views = [rgb.resize((224, 224), resample=Image.Resampling.BILINEAR), *extract_native_crops(rgb)]
    filename = hashlib.sha256(cid.encode("utf-8")).hexdigest() + ".npz"
    device = torch.device(args.device)
    torch.set_num_threads(2)
    comparisons = {}
    for backbone in ("resnet", "unicas"):
        if backbone == "resnet":
            model, _ = _build_encoder(device)
            batch = 64
        else:
            model = build_unicas_model()
            model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True), strict=True)
            model.to(device).eval()
            batch = config["unicas_batch"]
        recomputed = _encode_views(model, views, device, batch)
        with np.load(experiment / "features" / backbone / filename, allow_pickle=False) as stored:
            comparisons[backbone] = {"whole": compare(recomputed[0], stored["whole"][0]),
                                     "native_mean": compare(recomputed[1:].mean(axis=0), stored["native_mean"][0]),
                                     "native_crops": compare(recomputed[1:], stored["native_crops"][0])}
        del model
        torch.cuda.empty_cache()
    report = {"scope": "one selected training parent recomputed independently; all saved validation metrics checked",
              "models": len(summary["results"]), "paired_seed_pseudo_groups": len(order),
              "all_training_orders_match": True, "all_validation_metadata_match": True,
              "pseudo_labels_excluded_from_training": True, "max_metric_absolute_difference": max_metric_difference,
              "parent_feature_recomputation": comparisons}
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--experiment", type=Path, required=True)
    p.add_argument("--candidate-csv", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda:1")
    run(p.parse_args())


if __name__ == "__main__":
    main()
