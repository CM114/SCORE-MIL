#!/usr/bin/env python3
"""Independent MIL baselines over features extracted directly from raw NDPI files."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler


FEATURE_DIM = 2048
CLASS_NAMES = ("NIML", "ASC-US", "ASC-H", "LISL", "HSIL")
CLASS_TO_ID = {name: index for index, name in enumerate(CLASS_NAMES)}
BINARY_TO_ID = {"normal": 0, "abnormal": 1}


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Mapping[str, object]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _records(case_rows) -> List[Dict[str, object]]:
    if hasattr(case_rows, "to_dict"):
        return [dict(row) for row in case_rows.to_dict(orient="records")]
    return [dict(row) for row in case_rows]


def _load_feature_array(path: Path, expected_dim: int | None = FEATURE_DIM) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(str(path))
    array = np.load(path, allow_pickle=False)
    if array.ndim != 2:
        raise ValueError(f"{path.name}: expected 2-D array, got shape {array.shape}")
    if expected_dim is not None and array.shape[1] != expected_dim:
        raise ValueError(f"{path.name}: expected width {expected_dim}, got {array.shape[1]}")
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"{path.name}: expected numeric dtype, got {array.dtype}")
    if not np.isfinite(array).all():
        raise ValueError(f"{path.name}: contains NaN or Inf")
    return np.asarray(array, dtype=np.float32)


def load_case_bag(case_rows) -> Tuple[np.ndarray, int, int, Dict[str, object]]:
    """Load one case's per-case NDPI feature array and validate row alignment."""

    rows = sorted(_records(case_rows), key=lambda item: int(item["candidate_index"]))
    if not rows:
        raise ValueError("empty case row group")
    case_ids = {str(row["case_id"]) for row in rows}
    if len(case_ids) != 1:
        raise ValueError(f"case group contains multiple case IDs: {sorted(case_ids)}")
    if any(str(row.get("feature_status", "")) != "ok" for row in rows):
        raise ValueError(f"case {rows[0]['case_id']} contains non-ok feature rows")
    feature_paths = {str(row["feature_path"]) for row in rows}
    if len(feature_paths) != 1:
        raise ValueError(f"case {rows[0]['case_id']} contains multiple feature paths")
    feature_dim_values = {int(row["feature_dim"]) for row in rows}
    if len(feature_dim_values) != 1 or next(iter(feature_dim_values)) <= 0:
        raise ValueError(f"case {rows[0]['case_id']} has feature_dim={sorted(feature_dim_values)}")
    feature_rows = [int(row["feature_row"]) for row in rows]
    if feature_rows != list(range(len(rows))):
        raise ValueError(f"case {rows[0]['case_id']} feature_row is not contiguous from zero")
    candidate_indices = [int(row["candidate_index"]) for row in rows]
    if len(set(candidate_indices)) != len(candidate_indices):
        raise ValueError(f"case {rows[0]['case_id']} has duplicate candidate_index")
    expected_dim = next(iter(feature_dim_values))
    features = _load_feature_array(Path(next(iter(feature_paths))), expected_dim=expected_dim)
    if features.shape[0] != len(rows):
        raise ValueError(f"case {rows[0]['case_id']}: array rows {features.shape[0]} != manifest rows {len(rows)}")
    wsl_class = str(rows[0].get("wsl_class", ""))
    binary_label = str(rows[0].get("binary_label", ""))
    if wsl_class not in CLASS_TO_ID:
        raise ValueError(f"unsupported wsl_class={wsl_class!r}")
    if binary_label not in BINARY_TO_ID:
        raise ValueError(f"unsupported binary_label={binary_label!r}")
    metadata = {
        "case_id": str(rows[0]["case_id"]),
        "split": str(rows[0].get("split", "")),
        "wsl_class": wsl_class,
        "binary_label": binary_label,
        "hospital": str(rows[0].get("hospital", "")),
        "patch_count": int(features.shape[0]),
        "feature_dim": int(features.shape[1]),
        "feature_path": str(next(iter(feature_paths))),
        "feature_source": str(rows[0].get("feature_source", "feature_manifest_encoder")),
    }
    return features, BINARY_TO_ID[binary_label], CLASS_TO_ID[wsl_class], metadata


def pool_mean(features: np.ndarray) -> np.ndarray:
    array = np.asarray(features)
    if array.ndim != 2 or array.shape[1] != FEATURE_DIM:
        raise ValueError(f"mean pooling expects [N,{FEATURE_DIM}], got {array.shape}; expected width {FEATURE_DIM}")
    if array.shape[0] == 0:
        raise ValueError("empty bag")
    if not np.isfinite(array).all():
        raise ValueError("mean pooling received NaN or Inf")
    return array.mean(axis=0, dtype=np.float64).astype(np.float32)


def group_feature_manifest(rows: Sequence[Mapping[str, str]]) -> Dict[str, List[Dict[str, str]]]:
    groups: Dict[str, List[Dict[str, str]]] = {}
    for row in rows:
        case_id = str(row.get("case_id", ""))
        if not case_id:
            raise ValueError("feature manifest contains empty case_id")
        groups.setdefault(case_id, []).append(dict(row))
    return {case_id: sorted(case_rows, key=lambda item: int(item["candidate_index"])) for case_id, case_rows in sorted(groups.items())}


def build_bag_manifest(rows: Sequence[Mapping[str, str]]) -> Tuple[List[Dict[str, object]], Dict[str, List[Dict[str, str]]], Dict[str, object]]:
    groups = group_feature_manifest(rows)
    output: List[Dict[str, object]] = []
    errors: List[Dict[str, str]] = []
    for case_id, case_rows in groups.items():
        try:
            _, binary_label, five_class_label, metadata = load_case_bag(case_rows)
            output.append({
                **metadata,
                "binary_label_id": binary_label,
                "five_class_label_id": five_class_label,
            })
        except Exception as exc:
            errors.append({"case_id": case_id, "error": f"{type(exc).__name__}: {exc}"})
    valid_feature_dims = sorted({int(item["feature_dim"]) for item in output})
    valid_feature_sources = sorted({str(item.get("feature_source", "")) for item in output})
    audit = {
        "input_feature_rows": len(rows),
        "input_cases": len(groups),
        "valid_bags": len(output),
        "error_count": len(errors),
        "errors": errors[:100],
        "feature_dim": valid_feature_dims[0] if len(valid_feature_dims) == 1 else None,
        "feature_dims": valid_feature_dims,
        "feature_source": valid_feature_sources[0] if len(valid_feature_sources) == 1 else valid_feature_sources,
        "excluded_inputs": ["hospital precomputed feature arrays", "wsi_classification.json", "classifation_wsi", "count_high_di"],
        "split_counts": dict(Counter(str(item["split"]) for item in output)),
        "class_counts": dict(Counter(str(item["wsl_class"]) for item in output)),
    }
    return output, groups, audit


def _metrics(y_true: np.ndarray, probabilities: np.ndarray, class_names: Sequence[str]) -> Dict[str, object]:
    predictions = probabilities.argmax(axis=1)
    result: Dict[str, object] = {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "macro_f1": float(f1_score(y_true, predictions, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, predictions, labels=list(range(len(class_names)))).tolist(),
        "class_names": list(class_names),
    }
    try:
        if len(class_names) == 2:
            result["roc_auc"] = float(roc_auc_score(y_true, probabilities[:, 1]))
        else:
            result["roc_auc_ovr_macro"] = float(
                roc_auc_score(y_true, probabilities, multi_class="ovr", average="macro", labels=list(range(len(class_names))))
            )
    except ValueError as exc:
        result["roc_auc_error"] = str(exc)
    return result


def _split_items(bag_rows: Sequence[Mapping[str, object]], split: str) -> List[Mapping[str, object]]:
    return [item for item in bag_rows if str(item["split"]) == split]


def run_mean_pooling(
    bag_rows: Sequence[Mapping[str, object]],
    groups: Mapping[str, Sequence[Mapping[str, str]]],
    output_dir: Path,
    seed: int,
) -> Dict[str, object]:
    pooled: Dict[str, np.ndarray] = {}
    for item in bag_rows:
        features, _, _, _ = load_case_bag(groups[str(item["case_id"])])
        pooled[str(item["case_id"])] = pool_mean(features)

    results: Dict[str, object] = {
        "model": "mean_pooling_logistic_regression",
        "feature_source": "raw_ndpi_resnet50_imagenet1k_v2",
        "tasks": {},
    }
    prediction_rows: List[Dict[str, object]] = []
    for task_name, label_key, class_names in (
        ("binary", "binary_label_id", ("normal", "abnormal")),
        ("five_class", "five_class_label_id", CLASS_NAMES),
    ):
        train_items = _split_items(bag_rows, "train")
        validation_items = _split_items(bag_rows, "validation")
        test_items = _split_items(bag_rows, "test")
        scaler = StandardScaler()
        x_train = scaler.fit_transform(np.stack([pooled[str(item["case_id"])] for item in train_items]))
        y_train = np.asarray([int(item[label_key]) for item in train_items], dtype=np.int64)
        model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)
        model.fit(x_train, y_train)
        task_result: Dict[str, object] = {"class_names": list(class_names), "split_metrics": {}}
        for split_name, split_items in (("validation", validation_items), ("test", test_items)):
            x_split = scaler.transform(np.stack([pooled[str(item["case_id"])] for item in split_items]))
            y_split = np.asarray([int(item[label_key]) for item in split_items], dtype=np.int64)
            probabilities = model.predict_proba(x_split)
            metrics = _metrics(y_split, probabilities, class_names)
            task_result["split_metrics"][split_name] = metrics
            predictions = probabilities.argmax(axis=1)
            for item, truth, prediction, probability in zip(split_items, y_split, predictions, probabilities):
                prediction_rows.append({
                    "model": "mean_pooling_logistic_regression",
                    "task": task_name,
                    "case_id": item["case_id"],
                    "split": split_name,
                    "ground_truth": int(truth),
                    "prediction": int(prediction),
                    "max_probability": float(probability.max()),
                })
        results["tasks"][task_name] = task_result
    write_json(output_dir / "mean_pooling_results.json", results)
    write_csv(output_dir / "mean_pooling_predictions.csv", prediction_rows, list(prediction_rows[0].keys()))
    return results


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def subsample_bag(features: np.ndarray, max_instances: int) -> np.ndarray:
    if max_instances <= 0 or features.shape[0] <= max_instances:
        return features
    indices = np.linspace(0, features.shape[0] - 1, max_instances, dtype=np.int64)
    return features[indices]


def run_attention_mil(
    bag_rows: Sequence[Mapping[str, object]],
    groups: Mapping[str, Sequence[Mapping[str, str]]],
    output_dir: Path,
    seed: int,
    epochs: int,
    max_instances: int,
    device_name: str,
) -> Dict[str, object]:
    import torch
    from torch import nn

    class GatedAttentionMIL(nn.Module):
        def __init__(self, input_dim: int, hidden_dim: int, num_classes: int) -> None:
            super().__init__()
            self.norm = nn.LayerNorm(input_dim)
            self.value = nn.Linear(input_dim, hidden_dim)
            self.gate = nn.Linear(input_dim, hidden_dim)
            self.attention = nn.Linear(hidden_dim, 1)
            self.classifier = nn.Linear(input_dim, num_classes)

        def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            x = self.norm(x)
            scores = self.attention(torch.tanh(self.value(x)) * torch.sigmoid(self.gate(x))).squeeze(-1)
            weights = torch.softmax(scores, dim=0)
            pooled = (weights.unsqueeze(1) * x).sum(dim=0)
            return self.classifier(pooled), weights

    device = torch.device(device_name if device_name else ("cuda" if torch.cuda.is_available() else "cpu"))
    results: Dict[str, object] = {
        "model": "gated_attention_mil",
        "feature_source": "raw_ndpi_resnet50_imagenet1k_v2",
        "device": str(device),
        "tasks": {},
    }
    prediction_rows: List[Dict[str, object]] = []
    for task_name, label_key, class_names in (
        ("binary", "binary_label_id", ("normal", "abnormal")),
        ("five_class", "five_class_label_id", CLASS_NAMES),
    ):
        model = GatedAttentionMIL(FEATURE_DIM, 128, len(class_names)).to(device)
        train_items = _split_items(bag_rows, "train")
        validation_items = _split_items(bag_rows, "validation")
        test_items = _split_items(bag_rows, "test")
        counts = np.bincount([int(item[label_key]) for item in train_items], minlength=len(class_names)).astype(np.float32)
        class_weights = counts.sum() / np.maximum(counts, 1.0) / len(class_names)
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32, device=device))
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
        best_state = None
        best_score = -float("inf")
        best_epoch = 0
        history = []

        def evaluate(items: Sequence[Mapping[str, object]]):
            model.eval()
            truths: List[int] = []
            probabilities: List[np.ndarray] = []
            with torch.no_grad():
                for item in items:
                    features, _, _, _ = load_case_bag(groups[str(item["case_id"])])
                    tensor = torch.from_numpy(subsample_bag(features, max_instances)).to(device)
                    logits, _ = model(tensor)
                    truths.append(int(item[label_key]))
                    probabilities.append(torch.softmax(logits, dim=0).cpu().numpy())
            probability_array = np.stack(probabilities)
            return _metrics(np.asarray(truths), probability_array, class_names), [
                {"case_id": item["case_id"], "truth": truth, "probability": probability}
                for item, truth, probability in zip(items, truths, probability_array)
            ]

        for epoch in range(1, epochs + 1):
            model.train()
            order = list(train_items)
            random.Random(seed + epoch).shuffle(order)
            losses = []
            for item in order:
                features, _, _, _ = load_case_bag(groups[str(item["case_id"])])
                tensor = torch.from_numpy(subsample_bag(features, max_instances)).to(device)
                target = torch.tensor([int(item[label_key])], dtype=torch.long, device=device)
                optimizer.zero_grad(set_to_none=True)
                logits, _ = model(tensor)
                loss = criterion(logits.unsqueeze(0), target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
            validation_metrics, _ = evaluate(validation_items)
            score = float(validation_metrics["macro_f1"])
            history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "validation": validation_metrics})
            print(json.dumps({"task": task_name, "epoch": epoch, "train_loss": float(np.mean(losses)), "validation_macro_f1": score}, ensure_ascii=False), flush=True)
            if score > best_score:
                best_score = score
                best_epoch = epoch
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

        if best_state is None:
            raise RuntimeError(f"attention MIL did not produce a checkpoint for {task_name}")
        model.load_state_dict(best_state)
        validation_metrics, validation_records = evaluate(validation_items)
        test_metrics, test_records = evaluate(test_items)
        results["tasks"][task_name] = {
            "class_names": list(class_names),
            "best_epoch": best_epoch,
            "best_validation_macro_f1": best_score,
            "validation": validation_metrics,
            "test": test_metrics,
            "history": history,
        }
        for split_name, records in (("validation", validation_records), ("test", test_records)):
            for record in records:
                probability = np.asarray(record["probability"])
                prediction_rows.append({
                    "model": "gated_attention_mil",
                    "task": task_name,
                    "case_id": record["case_id"],
                    "split": split_name,
                    "ground_truth": int(record["truth"]),
                    "prediction": int(probability.argmax()),
                    "max_probability": float(probability.max()),
                })
        torch.save({"task": task_name, "state_dict": best_state, "best_epoch": best_epoch}, output_dir / f"attention_mil_{task_name}.pt")

    results["config"] = {
        "input_dim": FEATURE_DIM,
        "hidden_dim": 128,
        "epochs": epochs,
        "max_instances": max_instances,
        "seed": seed,
        "device": str(device),
    }
    write_json(output_dir / "attention_mil_results.json", results)
    write_csv(output_dir / "attention_mil_predictions.csv", prediction_rows, list(prediction_rows[0].keys()))
    return results


def run(args: argparse.Namespace) -> Dict[str, object]:
    set_seed(args.seed)
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_csv(Path(args.feature_manifest))
    bag_rows, groups, bag_audit = build_bag_manifest(rows)
    write_json(output_dir / "bag_audit.json", bag_audit)
    write_csv(output_dir / "bag_manifest.csv", bag_rows, list(bag_rows[0].keys()) if bag_rows else ["case_id"])
    if len(bag_rows) != len(groups) or len(bag_rows) != args.expected_cases:
        raise RuntimeError(f"bag validation failed: valid_bags={len(bag_rows)}, groups={len(groups)}, expected={args.expected_cases}")
    mean_results = run_mean_pooling(bag_rows, groups, output_dir, args.seed)
    summary: Dict[str, object] = {"bag_audit": bag_audit, "mean_pooling": mean_results}
    if args.run_attention:
        summary["attention_mil"] = run_attention_mil(
            bag_rows,
            groups,
            output_dir,
            args.seed,
            args.attention_epochs,
            args.attention_max_instances,
            args.device,
        )
    summary["protocol"] = {
        "feature_manifest": str(Path(args.feature_manifest)),
        "source_policy": "raw NDPI pixels only; hospital-provided feature files excluded",
        "expected_cases": args.expected_cases,
        "split_counts": dict(Counter(str(item["split"]) for item in bag_rows)),
    }
    write_json(output_dir / "run_summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--expected-cases", type=int, default=2176)
    parser.add_argument("--seed", type=int, default=20261026)
    parser.add_argument("--run-attention", action="store_true")
    parser.add_argument("--attention-epochs", type=int, default=10)
    parser.add_argument("--attention-max-instances", type=int, default=64)
    parser.add_argument("--device", default="")
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    result = run(arguments)
    print(json.dumps(result, ensure_ascii=False, indent=2))
