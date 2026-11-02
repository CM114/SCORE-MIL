#!/usr/bin/env python3
"""Aggregate paired exploratory scale/encoder results without case identifiers."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean


SCORE_NAMES = ("class_uncertainty", "ordinal_residual", "ordinal_fused")


def summarize_result(row):
    metrics = row["metrics"]
    cal = [f["scores"]["known_rank_fused"] for f in row["known_calibration_folds"]]
    if len(cal) != 2:
        raise ValueError("expected two known calibration folds")
    known_count = sum(x["known_eval_n"] for x in cal)
    unknown_count = sum(x["unknown_eval_n"] for x in cal)
    hsil_support = sum(x["by_known_class"].get("HSIL", {}).get("n", 0) for x in cal)
    hsil_unflagged = sum(x["by_known_class"].get("HSIL", {}).get("misclassified_unflagged_count", 0) for x in cal)
    return {
        "train_n": row["train_n"], "known_validation_n": row["known_validation_n"],
        "pseudo_unknown_validation_n": row["unknown_validation_n"],
        "known_macro_f1": metrics["known_classification"]["macro_f1"],
        "hsil_recall": metrics["known_classification"]["per_class"].get("HSIL", {}).get("recall"),
        "hsil_support": hsil_support, "hsil_misclassified_unflagged_count": hsil_unflagged,
        "known_calibration_fpr": sum(x["known_false_alarms"] for x in cal) / known_count,
        "known_calibration_unknown_recall": sum(x["unknown_detected"] for x in cal) / unknown_count,
        "by_source": metrics["by_source"],
        "scores": {name: {"auroc": metrics[name]["unknown_auroc"], "aupr": metrics[name]["unknown_aupr"],
                           "fpr95": metrics[name]["fpr95"]} for name in SCORE_NAMES},
    }


def summarize(summary):
    models = summary["results"]
    buckets = defaultdict(dict)
    for row in models:
        key = (row["pseudo_unknown"], row["group"])
        seed = str(row["seed"])
        if seed in buckets[key]:
            raise ValueError("duplicate pseudo/group/seed result")
        buckets[key][seed] = summarize_result(row)
    output = []
    for (pseudo, group), seeds in sorted(buckets.items()):
        vals = list(seeds.values())
        if set(map(int, seeds)) != set(summary["config"]["seeds"]):
            raise ValueError("missing seed")
        output.append({"pseudo_unknown": pseudo, "group": group, "seeds": seeds,
                       "seed_means": {"known_macro_f1": mean(x["known_macro_f1"] for x in vals),
                                      "hsil_recall": mean(x["hsil_recall"] for x in vals) if all(x["hsil_recall"] is not None for x in vals) else None,
                                      "known_calibration_fpr": mean(x["known_calibration_fpr"] for x in vals),
                                      "known_calibration_unknown_recall": mean(x["known_calibration_unknown_recall"] for x in vals),
                                      "scores": {name: {field: mean(x["scores"][name][field] for x in vals)
                                                        for field in ("auroc", "aupr", "fpr95")} for name in SCORE_NAMES}}})
    return {"scope": "paired exploratory development; no independent patients or diagnostic claims",
            "known_calibration": "two known-only folds; identical unknown cases repeat in each fold",
            "seed_note": "same case set across seeds; seed means are descriptive",
            "source_note": "source AUROC shown only with at least five known and five pseudo-unknown cases; otherwise null",
            "groups": output}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(json.loads(args.input.read_text(encoding="utf-8")))
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({"event": "aggregate_complete", "groups": len(result["groups"]), "output": str(args.output)}))


if __name__ == "__main__":
    main()
