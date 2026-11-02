#!/usr/bin/env python3
"""Cross-fit known-only score references and thresholds on existing predictions.

Both folds are exploratory within previously examined validation cases.
Unknown labels are used only for reporting, not for thresholds or references.
Patient independence and real deployment exchangeability are not established.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score


def split_known_by_hash(rows, known_order, seed: int):
    ids = [str(row["case_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate validation case")
    known = [dict(r) for r in rows if not int(r["is_pseudo_unknown"])]
    classes = set(known_order)
    if {str(r["wsl_class"]) for r in known} != classes:
        raise ValueError("known validation classes differ from model classes")
    folds = [[], []]
    for label in known_order:
        group = [r for r in known if str(r["wsl_class"]) == label]
        if len(group) < 2:
            raise ValueError("each known class requires at least two cases")
        group.sort(key=lambda r: hashlib.sha256(f"{seed}:{r['case_id']}".encode("utf-8")).digest())
        for i, row in enumerate(group):
            folds[i % 2].append(row)
    return folds[0], folds[1]


def upper_tail_threshold(calibration_scores, alpha: float) -> float:
    values = np.asarray(calibration_scores, dtype=np.float64)
    if not (0 < alpha < 1):
        raise ValueError("alpha must be between zero and one")
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("finite calibration scores are required")
    order = math.ceil((len(values) + 1) * (1 - alpha))
    if order > len(values):
        return float("inf")
    return float(np.partition(values, order - 1)[order - 1])


def _scores(calibration, evaluation, name):
    def vals(rows, field):
        return np.asarray([float(row[field]) for row in rows], dtype=np.float64)
    if name in ("class_uncertainty", "ordinal_residual"):
        return vals(calibration, name), vals(evaluation, name)
    if name != "known_rank_fused":
        raise ValueError("unknown score name")
    ref_u = np.sort(vals(calibration, "class_uncertainty")); ref_r = np.sort(vals(calibration, "ordinal_residual"))
    if not len(ref_u) or not len(ref_r):
        raise ValueError("known score references are empty")
    def fused(rows):
        return 0.5 * np.searchsorted(ref_u, vals(rows, "class_uncertainty"), side="right") / len(ref_u) + 0.5 * np.searchsorted(ref_r, vals(rows, "ordinal_residual"), side="right") / len(ref_r)
    return fused(calibration), fused(evaluation)


def evaluate_known_calibration(calibration, known_eval, unknown_eval, score_name: str, alpha: float = 0.05) -> dict:
    if not calibration or not known_eval or not unknown_eval:
        raise ValueError("calibration, known evaluation and unknown evaluation must be nonempty")
    if any(int(r["is_pseudo_unknown"]) for r in calibration + known_eval) or any(not int(r["is_pseudo_unknown"]) for r in unknown_eval):
        raise ValueError("calibration/evaluation contains an unexpected class role")
    ids = [[str(r["case_id"]) for r in subset] for subset in (calibration, known_eval, unknown_eval)]
    if any(len(x) != len(set(x)) for x in ids) or set(ids[0]) & set(ids[1]) or set(ids[0]) & set(ids[2]) or set(ids[1]) & set(ids[2]):
        raise ValueError("calibration and evaluation cases must be disjoint")
    all_eval = known_eval + unknown_eval
    cal_scores, eval_scores = _scores(calibration, all_eval, score_name)
    threshold = upper_tail_threshold(cal_scores, alpha)
    flags = eval_scores > threshold
    nknown = len(known_eval)
    groups = {}
    for label in sorted({str(r["wsl_class"]) for r in known_eval}):
        idx = [i for i, r in enumerate(known_eval) if str(r["wsl_class"]) == label]
        wrongly_classified = sum(str(known_eval[i]["predicted_class"]) != label for i in idx)
        groups[label] = {"n": len(idx), "false_alarm_count": int(np.sum(flags[idx])),
                         "false_alarm_rate": float(np.mean(flags[idx])),
                         "misclassified_count": wrongly_classified,
                         "misclassified_unflagged_count": sum(str(known_eval[i]["predicted_class"]) != label and not flags[i] for i in idx),
                         "predicted_class_counts": dict(Counter(str(known_eval[i]["predicted_class"]) for i in idx)),
                         "unflagged_predicted_class_counts": dict(Counter(str(known_eval[i]["predicted_class"]) for i in idx if not flags[i]))}
    truth = np.r_[np.zeros(nknown, dtype=np.int64), np.ones(len(unknown_eval), dtype=np.int64)]
    return {
        "calibration_n": len(calibration), "known_eval_n": nknown, "unknown_eval_n": len(unknown_eval),
        "target_known_fpr": alpha, "threshold": float(threshold) if np.isfinite(threshold) else None,
        "threshold_is_above_all_scores": bool(np.isposinf(threshold)), "rule": "flag when score strictly exceeds threshold",
        "known_false_alarms": int(np.sum(flags[:nknown])), "known_false_alarm_rate": float(np.mean(flags[:nknown])),
        "unknown_detected": int(np.sum(flags[nknown:])), "unknown_detection_rate": float(np.mean(flags[nknown:])),
        "evaluation_auroc": float(roc_auc_score(truth, eval_scores)), "evaluation_aupr": float(average_precision_score(truth, eval_scores)),
        "by_known_class": groups,
    }


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load(path: Path, expected, pseudo: str) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != len(expected) or {str(r["case_id"]) for r in rows} != set(expected):
        raise ValueError("stored validation cases do not match the protocol")
    for row in rows:
        cid = str(row["case_id"]); ref = expected[cid]
        if str(row["wsl_class"]) != str(ref["wsl_class"]) or str(row["hospital"]) != str(ref["hospital"]) or int(row["is_pseudo_unknown"]) != int(str(row["wsl_class"]) == pseudo):
            raise ValueError("stored validation metadata does not match the protocol")
        row["is_pseudo_unknown"] = int(row["is_pseudo_unknown"])
    return rows


def run(args):
    outer = json.loads(args.protocol.read_text(encoding="utf-8"))
    validation = {str(r["case_id"]): r for r in outer["validation_rows"]}
    if len(validation) != len(outer["validation_rows"]) or not set(validation).isdisjoint({str(x) for x in outer["test_ids"]}):
        raise ValueError("validation and outer test case sets overlap")
    if any(str(r["wsl_class"]) == str(outer["unknown_class"]) for r in validation.values()):
        raise ValueError("outer unknown label in validation")
    previous_summary = json.loads((args.previous / "inner_results" / "summary.json").read_text(encoding="utf-8"))
    outputs = []
    for pseudo in args.pseudo_unknowns:
        known_order = previous_summary["results"][pseudo]["known_order"]
        if pseudo in known_order or str(outer["unknown_class"]) in known_order:
            raise ValueError("pseudo or outer unknown is included in model classes")
        for seed in args.seeds:
            for policy in ("random", "spatial"):
                for mode in ("unweighted", "sqrt_inverse"):
                    path = args.previous / "inner_results" / pseudo / str(seed) / policy / mode / "validation_predictions.csv"
                    rows = _load(path, validation, pseudo)
                    halves = split_known_by_hash(rows, known_order, args.split_seed)
                    unknown = [r for r in rows if r["is_pseudo_unknown"]]
                    folds = []
                    for i in (0, 1):
                        cal, known_eval = halves[i], halves[1 - i]
                        classification = {"n": len(known_eval),
                                          "macro_f1": float(f1_score([r["wsl_class"] for r in known_eval], [r["predicted_class"] for r in known_eval], labels=known_order, average="macro", zero_division=0))}
                        folds.append({"calibration_half": i, "known_classification": classification,
                                      "scores": {name: evaluate_known_calibration(cal, known_eval, unknown, name, args.alpha) for name in ("class_uncertainty", "ordinal_residual", "known_rank_fused")}})
                    outputs.append({"pseudo_unknown": pseudo, "seed": seed, "policy": policy, "mode": mode,
                                    "known_order": known_order, "folds": folds, "prediction_sha256": _sha256(path)})
        print(json.dumps({"event": "known_calibration_fold_complete", "pseudo_unknown": pseudo, "models": len(outputs)}, ensure_ascii=False), flush=True)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    summary = {
        "scope": "existing inner development validation predictions, no images or training",
        "split": "class-stratified case-ID hash, two complementary known calibration/evaluation halves",
        "split_seed": args.split_seed, "target_known_fpr": args.alpha,
        "threshold": "ceil((n+1)*(1-alpha)) order statistic of calibration known scores; strictly greater flags unknown",
        "pseudo_unknown_labels_used_to_fit_reference_or_threshold": False,
        "limitations": ["same exploratory validation set; two folds reuse the same pseudo unknown cases", "patients unverified; no independent clinical or exchangeability claim", "HSIL validation support 8 and LISL pseudo unknown support 18", "source and label semantics remain unconfirmed"],
        "protocol_sha256": _sha256(args.protocol), "script_sha256": _sha256(Path(__file__).resolve()), "models": outputs,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"event": "known_calibration_complete", "models": len(outputs), "out_dir": str(args.out_dir)}, ensure_ascii=False), flush=True)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--previous", type=Path, required=True)
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--pseudo-unknowns", nargs="+", default=["ASC-US", "LISL"])
    p.add_argument("--seeds", nargs="+", type=int, default=[20261030, 20261031, 20261032])
    p.add_argument("--split-seed", type=int, default=20261031)
    p.add_argument("--alpha", type=float, default=0.05)
    args = p.parse_args()
    if not 0 < args.alpha < 1:
        raise ValueError("alpha must be between zero and one")
    run(args)


if __name__ == "__main__":
    main()
