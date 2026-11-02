#!/usr/bin/env python3
"""Paired AUROC uncertainty for four predeclared development comparisons.

Case bootstrap intervals condition on the fixed trained models and fixed
validation-derived rank references.  Patients and source clusters are not
known to be independent.  They are exploratory, not final-test inference.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from score_mil_paired_case_bootstrap import _load, _sha256, paired_auc_bootstrap


COMPARISONS = (
    ("resnet_whole", "resnet_native_mean"),
    ("unicas_whole", "unicas_native_mean"),
    ("resnet_whole", "unicas_whole"),
    ("resnet_native_mean", "unicas_native_mean"),
)


def aligned_scores(tables, seeds, baseline, candidate):
    reference = tables[(seeds[0], baseline)]
    ids = sorted(reference)
    for seed in seeds:
        for group in (baseline, candidate):
            table = tables[(seed, group)]
            if set(table) != set(ids):
                raise ValueError("prediction cases do not match")
            for cid in ids:
                if any(str(table[cid][field]) != str(reference[cid][field]) for field in ("is_pseudo_unknown", "wsl_class", "hospital")):
                    raise ValueError("prediction metadata do not match")
    y = np.asarray([int(reference[cid]["is_pseudo_unknown"]) for cid in ids], dtype=np.int64)
    scores = {group: np.asarray([[float(tables[(seed, group)][cid]["ordinal_fused"]) for cid in ids] for seed in seeds])
              for group in (baseline, candidate)}
    return y, scores[baseline], scores[candidate]


def run(args):
    summary = json.loads((args.experiment / "training_summary.json").read_text(encoding="utf-8"))
    if summary["config"]["outer_test_images_accessed"] or summary["config"]["outer_unknown_images_accessed"]:
        raise ValueError("experiment violates outer development boundary")
    seeds = summary["config"]["seeds"]
    groups = summary["config"]["groups"]
    outputs, hashes = [], {}
    for fold_index, pseudo in enumerate(("ASC-US", "LISL")):
        tables = {}
        for seed in seeds:
            for group in groups:
                path = args.experiment / "training" / pseudo / str(seed) / group / "validation_predictions.csv"
                tables[(seed, group)] = _load(path)
                hashes[f"{pseudo}/{seed}/{group}"] = _sha256(path)
        for baseline, candidate in COMPARISONS:
            y, a, b = aligned_scores(tables, seeds, baseline, candidate)
            result = paired_auc_bootstrap(y, a, b, args.replicates, args.seed + fold_index)
            outputs.append({"pseudo_unknown": pseudo, "baseline": baseline, "candidate": candidate, **result})
    report = {
        "scope": "existing development validation predictions; no images or model retraining",
        "estimator": "candidate minus baseline fixed 0.5 fused AUROC; same bootstrap cases across groups and seeds",
        "interval": "paired percentile case bootstrap stratified by known/unknown; fixed trained models and fixed scores",
        "limitations": ["development validation repeatedly used", "conditional on selection and fixed rank references", "patients and source clustering unverified", "three seeds share cases and do not represent independent cohorts", "LISL pseudo-unknown support 18"],
        "input_prediction_sha256": hashes, "comparisons": outputs,
    }
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({"event": "paired_bootstrap_complete", "comparisons": len(outputs), "replicates": args.replicates}, ensure_ascii=False), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--experiment", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--replicates", type=int, default=2000)
    p.add_argument("--seed", type=int, default=20261031)
    args = p.parse_args()
    if args.replicates <= 0:
        raise ValueError("bootstrap replicates must be positive")
    run(args)


if __name__ == "__main__":
    main()
