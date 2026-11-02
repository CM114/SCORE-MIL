#!/usr/bin/env python3
"""Paired case-level bootstrap and score-tie audit on existing development results.

No image, model training or outer-test result is used. Cases are treated as
independent; patient identity and source clustering are unverified, so the
resulting intervals are exploratory and conditional on the trained models.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


def paired_auc_bootstrap(y, baseline, candidate, replicates: int, seed: int) -> dict:
    y = np.asarray(y, dtype=np.int64)
    a = np.asarray(baseline, dtype=np.float64)
    b = np.asarray(candidate, dtype=np.float64)
    if y.ndim != 1 or a.ndim != 2 or a.shape != b.shape or a.shape[1] != len(y) or not np.isfinite(a).all() or not np.isfinite(b).all() or replicates < 1:
        raise ValueError("aligned finite model scores and positive replicates are required")
    if set(y.tolist()) != {0, 1}:
        raise ValueError("both known and pseudo-unknown cases are required")
    known, unknown = np.flatnonzero(y == 0), np.flatnonzero(y == 1)
    deltas = [float(roc_auc_score(y, b[i]) - roc_auc_score(y, a[i])) for i in range(len(a))]
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(replicates):
        index = np.r_[rng.choice(known, len(known), replace=True), rng.choice(unknown, len(unknown), replace=True)]
        values.append(float(np.mean([roc_auc_score(y[index], b[i, index]) - roc_auc_score(y[index], a[i, index]) for i in range(len(a))])))
    return {
        "known_n": len(known), "pseudo_unknown_n": len(unknown), "model_seed_differences": deltas,
        "mean_difference": float(np.mean(deltas)), "ci95": [float(x) for x in np.quantile(values, [0.025, 0.975])],
        "replicates": replicates, "resampling_seed": seed,
    }


def score_tie_summary(y, scores) -> dict:
    y = np.asarray(y, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if y.ndim != 1 or scores.shape != y.shape or set(y.tolist()) != {0, 1} or not np.isfinite(scores).all():
        raise ValueError("both groups and aligned finite scores are required")
    known, unknown = scores[y == 0], scores[y == 1]
    return {
        "cross_group_tie_fraction": float(np.mean(known[:, None] == unknown[None, :])),
        "unknown_at_upper_bound": int(np.sum(unknown == 1.0)),
        "unknown_strictly_above_known_max": int(np.sum(unknown > known.max())),
    }


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load(path: Path) -> dict:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    mapping = {str(row["case_id"]): row for row in rows}
    if len(rows) != len(mapping):
        raise ValueError("prediction file contains duplicate cases")
    return mapping


def run(args):
    data = {}
    file_hashes = {}
    for pseudo in args.pseudo_unknowns:
        data[pseudo] = {}
        for seed in args.seeds:
            for policy in ("random", "spatial"):
                path = args.previous / "inner_results" / pseudo / str(seed) / policy / "unweighted" / "validation_predictions.csv"
                data[pseudo][(seed, policy)] = _load(path)
                file_hashes[f"{pseudo}/{seed}/{policy}"] = _sha256(path)
    results = {}
    for fold_index, pseudo in enumerate(args.pseudo_unknowns):
        tables = data[pseudo]
        ids = sorted(tables[(args.seeds[0], "random")])
        reference = tables[(args.seeds[0], "random")]
        for table in tables.values():
            if set(table) != set(ids) or any(any(str(table[cid][field]) != str(reference[cid][field]) for field in ("is_pseudo_unknown", "wsl_class", "hospital")) for cid in ids):
                raise ValueError("paired prediction cases or labels differ")
        y = np.asarray([int(reference[cid]["is_pseudo_unknown"]) for cid in ids])
        score_arrays = {policy: np.asarray([[float(tables[(seed, policy)][cid]["ordinal_fused"]) for cid in ids] for seed in args.seeds]) for policy in ("random", "spatial")}
        comparison = paired_auc_bootstrap(y, score_arrays["random"], score_arrays["spatial"], args.replicates, args.seed + fold_index)
        tie_stats = {}
        for model_seed in args.seeds:
            for policy in ("random", "spatial"):
                table = tables[(model_seed, policy)]
                scores = np.asarray([float(table[cid]["ordinal_fused"]) for cid in ids])
                ties = score_tie_summary(y, scores)
                ties["known_low_uncertainty_by_label"] = {
                    label: {"n": sum(int(table[cid]["is_pseudo_unknown"]) == 0 and table[cid]["wsl_class"] == label for cid in ids),
                            "uncertainty_le_1e_6": sum(int(table[cid]["is_pseudo_unknown"]) == 0 and table[cid]["wsl_class"] == label and float(table[cid]["class_uncertainty"]) <= 1e-6 for cid in ids)}
                    for label in sorted({table[cid]["wsl_class"] for cid in ids if int(table[cid]["is_pseudo_unknown"]) == 0})
                }
                tie_stats[f"{model_seed}/{policy}"] = ties
        results[pseudo] = {"spatial_minus_random_unweighted_auroc": comparison, "score_ties": tie_stats}
        print(json.dumps({"event": "bootstrap_fold_complete", "pseudo_unknown": pseudo, **comparison}, ensure_ascii=False), flush=True)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    summary = {
        "scope": "existing inner validation predictions only; no training or image read",
        "design": "paired, stratified by known vs pseudo-unknown; same resampled cases across sampling strategies and model seeds",
        "limitations": ["patients unverified; case bootstrap can understate uncertainty", "source clustering not accounted for", "same development cases and previously selected strategies", "fixed trained models, not training-run uncertainty"],
        "input_prediction_hashes": file_hashes, "script_sha256": _sha256(Path(__file__).resolve()), "results": results,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"event": "bootstrap_complete", "out_dir": str(args.out_dir)}), flush=True)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--previous", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--pseudo-unknowns", nargs="+", default=["ASC-US", "LISL"])
    p.add_argument("--seeds", nargs="+", type=int, default=[20261030, 20261031, 20261032])
    p.add_argument("--replicates", type=int, default=2000)
    p.add_argument("--seed", type=int, default=20261031)
    args = p.parse_args()
    if args.replicates <= 0 or not args.seeds:
        raise ValueError("invalid bootstrap configuration")
    run(args)


if __name__ == "__main__":
    main()
