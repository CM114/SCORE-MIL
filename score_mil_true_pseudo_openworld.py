#!/usr/bin/env python3
"""True inner leave-one-known-label-out control for paired sampling/weighting.

The outer held-out label and outer test cases are never loaded. Each inner
pseudo-unknown class is removed from training and the model's ordinal order.
The held-out validation scores are exploratory development results, not a
clinical validation or an independent final test.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping, Sequence

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

from ndpi_mil_baseline import build_bag_manifest, load_case_bag, read_csv
from score_mil_dev_candidate_manifest import development_cases
from score_mil_ordinal_control import GLOBAL_ORDER, OrdinalAttentionMIL, _open_set_metrics, _validation_rank, build_ordinal_targets


def build_inner_partition(outer: Mapping[str, object], pseudo_unknown: str) -> dict:
    outer_unknown = str(outer["unknown_class"])
    known_outer = {str(label) for label in outer["known_classes"]}
    if pseudo_unknown not in known_outer or pseudo_unknown == outer_unknown:
        raise ValueError("pseudo unknown must be an outer known class")
    development_cases(dict(outer))
    inner_known = known_outer - {str(pseudo_unknown)}
    train = [dict(row) for row in outer["train_rows"] if str(row["wsl_class"]) in inner_known]
    known_validation = [dict(row) for row in outer["validation_rows"] if str(row["wsl_class"]) in inner_known]
    unknown_validation = [dict(row) for row in outer["validation_rows"] if str(row["wsl_class"]) == pseudo_unknown]
    if {str(row["wsl_class"]) for row in train} != inner_known or not known_validation or not unknown_validation:
        raise ValueError("inner training/validation support is insufficient")
    return {
        "outer_unknown": outer_unknown,
        "pseudo_unknown": str(pseudo_unknown),
        "known_classes": list(known_only_rank_order(inner_known)),
        "train_rows": train,
        "known_validation_rows": known_validation,
        "unknown_validation_rows": unknown_validation,
    }


def known_only_rank_order(known_classes: Sequence[str]) -> tuple[str, ...]:
    classes = set(str(x) for x in known_classes)
    if len(classes) < 2 or not classes.issubset(set(GLOBAL_ORDER)):
        raise ValueError("known-only ordinal order requires at least two supported known labels")
    return tuple(label for label in GLOBAL_ORDER if label in classes)


def class_loss_weights(counts: Mapping[str, int], known_classes: Sequence[str], mode: str) -> dict[str, float]:
    classes = list(known_classes)
    if not classes or any(int(counts.get(label, 0)) <= 0 for label in classes):
        raise ValueError("all known classes require training samples")
    if mode == "unweighted":
        return {label: 1.0 for label in classes}
    if mode != "sqrt_inverse":
        raise ValueError(f"unsupported class weight mode: {mode}")
    total = float(sum(int(counts[label]) for label in classes))
    raw = {label: np.sqrt(total / (len(classes) * int(counts[label]))) for label in classes}
    empirical_mean = sum(int(counts[label]) * raw[label] for label in classes) / total
    return {label: float(raw[label] / empirical_mean) for label in classes}


def openworld_scores(records: Sequence[Mapping[str, object]]) -> np.ndarray:
    if not records:
        raise ValueError("validation records are empty")
    u = np.asarray([float(row["class_uncertainty"]) for row in records], dtype=np.float64)
    r = np.asarray([float(row["ordinal_residual"]) for row in records], dtype=np.float64)
    y = np.asarray([int(row["is_pseudo_unknown"]) for row in records], dtype=np.int64)
    known = y == 0
    if not known.any() or not (y == 1).any():
        raise ValueError("both known and pseudo-unknown validation cases are required")
    return 0.5 * _validation_rank(u, u[known]) + 0.5 * _validation_rank(r, r[known])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def _write_csv(path: Path, records: Sequence[Mapping[str, object]]) -> None:
    if not records:
        raise ValueError("cannot write empty validation records")
    fields = ["case_id", "wsl_class", "hospital", "is_pseudo_unknown", "predicted_class", "probabilities", "class_uncertainty", "ordinal_residual", "ordinal_fused"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in records:
            output = dict(row)
            output["probabilities"] = json.dumps(output["probabilities"])
            writer.writerow(output)


def _known_metrics(records, known_classes):
    known = [row for row in records if not row["is_pseudo_unknown"]]
    mapping = {label: i for i, label in enumerate(known_classes)}
    truth = np.asarray([mapping[str(row["wsl_class"])] for row in known])
    pred = np.asarray([mapping[str(row["predicted_class"])] for row in known])
    pp, rr, ff, ss = precision_recall_fscore_support(truth, pred, labels=np.arange(len(mapping)), zero_division=0)
    return {
        "n": len(known),
        "accuracy": float(accuracy_score(truth, pred)),
        "macro_f1": float(np.mean(ff)),
        "per_class": {label: {"precision": float(pp[i]), "recall": float(rr[i]), "f1": float(ff[i]), "support": int(ss[i])} for i, label in enumerate(known_classes)},
    }


def _forward(model, items, known_order, pseudo_unknown, device):
    model.eval()
    result = []
    rank = torch.arange(len(known_order), dtype=torch.float32, device=device)
    with torch.inference_mode():
        for item in items:
            tensor = torch.from_numpy(item["features"]).to(device=device, dtype=torch.float32)
            out = model(tensor)
            prob = torch.softmax(out["class_logits"], dim=-1)
            ordinal = torch.sigmoid(out["ordinal_logits"]).sum()
            expected = torch.sum(prob * rank)
            result.append({
                "case_id": str(item["case_id"]),
                "wsl_class": str(item["wsl_class"]),
                "hospital": str(item["hospital"]),
                "is_pseudo_unknown": int(str(item["wsl_class"]) == pseudo_unknown),
                "predicted_class": known_order[int(prob.argmax().item())],
                "probabilities": prob.cpu().tolist(),
                "class_uncertainty": float((1.0 - prob.max()).item()),
                "ordinal_residual": float((torch.abs(ordinal - expected) / (len(known_order) - 1)).item()),
            })
    scores = openworld_scores(result)
    for row, score in zip(result, scores):
        row["ordinal_fused"] = float(score)
    return result


def _evaluate(records, known_order):
    known = [r for r in records if not r["is_pseudo_unknown"]]
    unknown = [r for r in records if r["is_pseudo_unknown"]]
    metrics = {field: _open_set_metrics(np.asarray([r[field] for r in known]), np.asarray([r[field] for r in unknown])) for field in ("class_uncertainty", "ordinal_residual", "ordinal_fused")}
    metrics["known_classification"] = _known_metrics(records, known_order)
    hospital_groups = defaultdict(list)
    for row in records:
        hospital_groups[row["hospital"]].append(row)
    metrics["by_source"] = {}
    for source, rows in sorted(hospital_groups.items()):
        truth = np.asarray([r["is_pseudo_unknown"] for r in rows])
        nknown, nunknown = int(np.sum(truth == 0)), int(np.sum(truth == 1))
        entry = {"known_count": nknown, "pseudo_unknown_count": nunknown, "fused_auroc": None}
        if nknown >= 5 and nunknown >= 5:
            entry["fused_auroc"] = float(roc_auc_score(truth, [r["ordinal_fused"] for r in rows]))
        metrics["by_source"][source] = entry
    return metrics


def _seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one(train_items, validation_items, known_order, pseudo_unknown, mode, seed, epochs, hidden_dim, device, out_dir):
    _seed_everything(seed)
    counts = Counter(str(row["wsl_class"]) for row in train_items)
    weights = class_loss_weights(counts, known_order, mode)
    mapping = {label: index for index, label in enumerate(known_order)}
    input_dim = int(train_items[0]["features"].shape[1])
    model = OrdinalAttentionMIL(input_dim, hidden_dim, len(known_order), len(known_order) - 1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    rng = random.Random(seed)
    order_hash = hashlib.sha256()
    history = []
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        indices = list(range(len(train_items)))
        rng.shuffle(indices)
        order_hash.update(np.asarray(indices, dtype=np.int32).tobytes())
        gradient_norm = defaultdict(list)
        losses = []
        for index in indices:
            item = train_items[index]
            label = str(item["wsl_class"])
            data = torch.from_numpy(item["features"]).to(device=device, dtype=torch.float32)
            target = torch.tensor([mapping[label]], dtype=torch.long, device=device)
            ordinal_target = torch.from_numpy(build_ordinal_targets([label], known_order)).to(device=device).squeeze(0)
            out = model(data)
            ce = F.cross_entropy(out["class_logits"].unsqueeze(0), target, reduction="sum") * weights[label]
            ordinal = F.binary_cross_entropy_with_logits(out["ordinal_logits"], ordinal_target)
            loss = ce + 0.5 * ordinal
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0).detach().cpu())
            if not np.isfinite(norm):
                raise RuntimeError("non-finite gradient")
            optimizer.step()
            gradient_norm[label].append(norm)
            losses.append(float(loss.detach().cpu()))
        history.append({
            "epoch": epoch,
            "mean_loss": float(np.mean(losses)),
            "gradient_clip_fraction_by_class": {label: float(np.mean(np.asarray(gradient_norm[label]) > 5.0)) for label in known_order},
        })
        if epoch in (1, epochs):
            print(json.dumps({"event": "train_epoch", "pseudo_unknown": pseudo_unknown, "mode": mode, "seed": seed, "epoch": epoch}, ensure_ascii=False), flush=True)
    records = _forward(model, validation_items, known_order, pseudo_unknown, device)
    metrics = _evaluate(records, known_order)
    out_dir.mkdir(parents=True, exist_ok=False)
    torch.save({"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()}, "known_order": list(known_order), "seed": seed, "epochs": epochs, "mode": mode, "input_dim": input_dim, "hidden_dim": hidden_dim}, out_dir / "fixed_epoch.pt")
    _write_csv(out_dir / "validation_predictions.csv", records)
    result = {
        "pseudo_unknown": pseudo_unknown,
        "known_order": list(known_order),
        "mode": mode,
        "seed": seed,
        "epochs": epochs,
        "train_class_counts": dict(counts),
        "class_weights": weights,
        "train_order_sha256": order_hash.hexdigest(),
        "metrics": metrics,
        "history": history,
        "elapsed_seconds": round(time.perf_counter() - started, 2),
    }
    _write_json(out_dir / "result.json", result)
    return result


def run(args: argparse.Namespace) -> dict:
    outer = json.loads(args.protocol.read_text(encoding="utf-8"))
    outer_cases = development_cases(outer)
    outer_ids = {str(row["case_id"]) for row in outer_cases}
    expected_meta = {str(row["case_id"]): row for row in outer_cases}
    feature_summary = json.loads((args.feature_root / "summary.json").read_text(encoding="utf-8"))
    if feature_summary["outer_test_images_accessed"] or feature_summary["outer_unknown_images_accessed"]:
        raise ValueError("features do not meet the declared outer development restriction")
    manifests = {policy: args.feature_root / policy / "feature_manifest.csv" for policy in ("random", "spatial")}
    data_by_policy = {}
    for policy, path in manifests.items():
        rows = read_csv(path)
        bags, groups, audit = build_bag_manifest(rows)
        if audit["error_count"] or set(groups) != outer_ids:
            raise RuntimeError("feature cases or arrays do not match complete outer development cases")
        for bag in bags:
            original = expected_meta[str(bag["case_id"])]
            if any(str(bag[field]) != str(original[field]) for field in ("wsl_class", "hospital", "binary_label")):
                raise ValueError("feature case metadata differs from outer protocol")
        items = {}
        for cid in sorted(outer_ids):
            features, _, _, metadata = load_case_bag(groups[cid])
            items[cid] = {"features": features, **metadata}
        data_by_policy[policy] = items
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    config = {
        "outer_unknown": str(outer["unknown_class"]),
        "pseudo_unknowns": args.pseudo_unknowns,
        "seeds": args.seeds,
        "epochs": args.epochs,
        "hidden_dim": args.hidden_dim,
        "device": str(device),
        "sampling_policies": ["random", "spatial"],
        "class_weight_modes": ["unweighted", "sqrt_inverse"],
        "sqrt_rule": "sqrt(inverse class frequency), empirical train case mean normalized to one",
        "ordinal_order": "known classes only; missing outer/pseudo unknown positions omitted",
        "ordinal_weight": 0.5,
        "fusion": "fixed 0.5 uncertainty rank + 0.5 residual rank, using known validation references",
        "checkpoint_selection": "fixed final epoch, no early stopping or test-based choice",
        "outer_test_features_loaded": False,
        "outer_unknown_case_features_loaded": False,
        "pseudo_unknown_features_used_for_training": False,
        "limitations": ["inner validation is exploratory", "pseudo-LISL has limited validation support", "patients may not be independent", "clinical ordinal order remains a hypothesis"],
        "protocol_sha256": _sha256(args.protocol),
        "feature_manifest_sha256": {p: _sha256(f) for p, f in manifests.items()},
        "script_sha256": _sha256(Path(__file__).resolve()),
    }
    _write_json(args.out_dir / "config.json", config)
    summary = {"config": config, "results": {}}
    for pseudo_unknown in args.pseudo_unknowns:
        part = build_inner_partition(outer, pseudo_unknown)
        train_ids = [str(row["case_id"]) for row in part["train_rows"]]
        val_ids = [str(row["case_id"]) for row in part["known_validation_rows"] + part["unknown_validation_rows"]]
        fold = {
            "known_order": part["known_classes"],
            "train_n": len(train_ids),
            "known_validation_n": len(part["known_validation_rows"]),
            "pseudo_unknown_validation_n": len(part["unknown_validation_rows"]),
            "results": {},
        }
        for seed in args.seeds:
            order_hashes = set()
            seed_data = {}
            for policy in config["sampling_policies"]:
                items = data_by_policy[policy]
                train = [items[cid] for cid in train_ids]
                validation = [items[cid] for cid in val_ids]
                for mode in config["class_weight_modes"]:
                    result = train_one(train, validation, part["known_classes"], pseudo_unknown, mode, seed, args.epochs, args.hidden_dim, device, args.out_dir / pseudo_unknown / str(seed) / policy / mode)
                    order_hashes.add(result["train_order_sha256"])
                    seed_data[f"{policy}__{mode}"] = {"metrics": result["metrics"], "class_weights": result["class_weights"], "train_order_sha256": result["train_order_sha256"]}
            if len(order_hashes) != 1:
                raise RuntimeError("paired sample orders differ")
            fold["results"][str(seed)] = seed_data
            print(json.dumps({"event": "seed_complete", "pseudo_unknown": pseudo_unknown, "seed": seed, "paired_orders_match": True}), flush=True)
            _write_json(args.out_dir / pseudo_unknown / "partial_summary.json", fold)
        summary["results"][pseudo_unknown] = fold
        _write_json(args.out_dir / "partial_summary.json", summary)
    _write_json(args.out_dir / "summary.json", summary)
    print(json.dumps({"event": "complete", "output_dir": str(args.out_dir)}), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--pseudo-unknowns", nargs="+", default=["ASC-US", "LISL"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[20261030, 20261031, 20261032])
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--device", default="cuda:1")
    args = parser.parse_args()
    if args.epochs <= 0 or args.hidden_dim <= 0 or not args.seeds:
        raise ValueError("invalid training settings")
    run(args)


if __name__ == "__main__":
    main()
