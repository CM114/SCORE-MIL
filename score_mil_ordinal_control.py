#!/usr/bin/env python3
"""SCORE-MIL phase-1 control: ordinal scaffold versus one-hot MIL.

This is a frozen-feature diagnostic, not the final SCORE-MIL model. It tests
whether a provisional Bethesda-like label order improves known-class learning
and open-set novelty ranking without using held-out unknown labels for fitting.
The label order must be confirmed by the data owner before being used as a
clinical claim.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence

import numpy as np
import torch
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score, roc_curve
from torch import nn

GLOBAL_ORDER = ("NIML", "ASC-US", "LISL", "ASC-H", "HSIL")


def normalize_bag_features(features: np.ndarray, mode: str = "none") -> np.ndarray:
    """Apply source-agnostic normalization independently within each case."""

    array = np.asarray(features, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"features must be [instances, dim], got {array.shape}")
    if mode == "none":
        return array
    if mode not in {"case_center", "case_center_scale"}:
        raise ValueError(f"unsupported bag normalization: {mode}")
    centered = array - array.mean(axis=0, keepdims=True)
    if mode == "case_center":
        return centered.astype(np.float32, copy=False)
    scale = np.maximum(centered.std(axis=0, keepdims=True), 1e-6)
    return (centered / scale).astype(np.float32, copy=False)


def build_ordinal_targets(labels: Sequence[str], global_order: Sequence[str] = GLOBAL_ORDER) -> np.ndarray:
    """Build cumulative ordinal targets for the global diagnostic order."""

    rank = {name: index for index, name in enumerate(global_order)}
    if any(str(label) not in rank for label in labels):
        unknown = sorted({str(label) for label in labels if str(label) not in rank})
        raise ValueError(f"labels absent from global_order: {unknown}")
    output = np.zeros((len(labels), max(len(global_order) - 1, 1)), dtype=np.float32)
    for row_index, label in enumerate(labels):
        output[row_index, : rank[str(label)]] = 1.0
    return output


def ordinal_expected_rank(logits: torch.Tensor) -> torch.Tensor:
    """Return the expected ordinal rank from cumulative logits."""

    if logits.ndim == 1:
        logits = logits.unsqueeze(0)
    if logits.ndim != 2:
        raise ValueError(f"ordinal logits must be 1-D or 2-D, got {tuple(logits.shape)}")
    return torch.sigmoid(logits).sum(dim=-1)


def ordinal_residual_score(
    ordinal_score: torch.Tensor, class_expected_rank: torch.Tensor, num_levels: int = 4
) -> torch.Tensor:
    """Measure disagreement between ordinal and categorical evidence."""

    if ordinal_score.shape != class_expected_rank.shape:
        raise ValueError("ordinal_score and class_expected_rank must have the same shape")
    if num_levels <= 0:
        raise ValueError("num_levels must be positive")
    return torch.abs(ordinal_score - class_expected_rank) / float(num_levels)


def patch_residual_scores(
    patch_ordinal_logits: torch.Tensor,
    patch_class_logits: torch.Tensor,
    known_classes: Sequence[str],
    global_order: Sequence[str] = GLOBAL_ORDER,
) -> torch.Tensor:
    """Compute categorical--ordinal disagreement for every patch in a bag."""

    if patch_ordinal_logits.ndim != 2 or patch_class_logits.ndim != 2:
        raise ValueError("patch logits must both be 2-D tensors")
    if patch_ordinal_logits.shape[0] != patch_class_logits.shape[0]:
        raise ValueError("patch ordinal and class logits must have the same number of instances")
    probabilities = torch.softmax(patch_class_logits, dim=-1)
    class_rank = _class_expected_rank(probabilities, known_classes, global_order)
    ordinal_score = ordinal_expected_rank(patch_ordinal_logits)
    return ordinal_residual_score(ordinal_score, class_rank, len(global_order) - 1)


class OrdinalAttentionMIL(nn.Module):
    """Small attention MIL model with a categorical and ordinal head."""

    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, num_global_thresholds: int = 4) -> None:
        super().__init__()
        self.projection = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
        self.attention = nn.Linear(hidden_dim, 1)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.ordinal = nn.Linear(hidden_dim, num_global_thresholds)

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        if features.ndim != 2:
            raise ValueError(f"features must be [instances, dim], got {tuple(features.shape)}")
        hidden = self.projection(features)
        attention = torch.softmax(self.attention(hidden).squeeze(-1), dim=0)
        pooled = (attention.unsqueeze(-1) * hidden).sum(dim=0)
        patch_class_logits = self.classifier(hidden)
        patch_ordinal_logits = self.ordinal(hidden)
        return {
            "class_logits": self.classifier(pooled),
            "ordinal_logits": self.ordinal(pooled),
            "patch_class_logits": patch_class_logits,
            "patch_ordinal_logits": patch_ordinal_logits,
            "attention": attention,
            "embedding": pooled,
        }


class DecoupledOrdinalAttentionMIL(nn.Module):
    """Ordinal MIL with separate classification and severity pooling attention."""

    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, num_global_thresholds: int = 4) -> None:
        super().__init__()
        self.projection = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
        self.class_attention = nn.Linear(hidden_dim, 1)
        self.ordinal_attention = nn.Linear(hidden_dim, 1)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.ordinal = nn.Linear(hidden_dim, num_global_thresholds)

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        if features.ndim != 2:
            raise ValueError(f"features must be [instances, dim], got {tuple(features.shape)}")
        hidden = self.projection(features)
        class_attention = torch.softmax(self.class_attention(hidden).squeeze(-1), dim=0)
        ordinal_attention = torch.softmax(self.ordinal_attention(hidden).squeeze(-1), dim=0)
        class_pooled = (class_attention.unsqueeze(-1) * hidden).sum(dim=0)
        ordinal_pooled = (ordinal_attention.unsqueeze(-1) * hidden).sum(dim=0)
        return {
            "class_logits": self.classifier(class_pooled),
            "ordinal_logits": self.ordinal(ordinal_pooled),
            "patch_class_logits": self.classifier(hidden),
            "patch_ordinal_logits": self.ordinal(hidden),
            "attention": class_attention,
            "class_attention": class_attention,
            "ordinal_attention": ordinal_attention,
            "embedding": class_pooled,
        }


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _load_items(
    protocol: Mapping[str, object],
    groups: Mapping[str, Sequence[Mapping[str, str]]],
    role: str,
    bag_normalization: str = "none",
):
    from ndpi_mil_baseline import load_case_bag

    rows = protocol[f"{role}_rows"]
    output = []
    for row in rows:
        case_id = str(row["case_id"])
        features, _, _, metadata = load_case_bag(groups[case_id])
        features = normalize_bag_features(features, bag_normalization)
        output.append({"features": features, **metadata})
    return output


def _class_expected_rank(
    probabilities: torch.Tensor, known_classes: Sequence[str], global_order: Sequence[str] = GLOBAL_ORDER
) -> torch.Tensor:
    ranks = torch.tensor([global_order.index(name) for name in known_classes], dtype=probabilities.dtype, device=probabilities.device)
    return (probabilities * ranks.unsqueeze(0)).sum(dim=-1)


def _forward_items(model, items, known_classes, device, global_order: Sequence[str] = GLOBAL_ORDER):
    outputs = []
    model.eval()
    with torch.no_grad():
        for item in items:
            features = torch.from_numpy(item["features"]).to(device=device, dtype=torch.float32)
            result = model(features)
            probabilities = torch.softmax(result["class_logits"], dim=-1).unsqueeze(0)
            ordinal_score = ordinal_expected_rank(result["ordinal_logits"]).reshape(1)
            class_rank = _class_expected_rank(probabilities, known_classes, global_order)
            residual = ordinal_residual_score(ordinal_score, class_rank).item()
            uncertainty = (1.0 - probabilities.max(dim=-1).values).item()
            outputs.append(
                {
                    "case_id": item["case_id"],
                    "wsl_class": item["wsl_class"],
                    "hospital": item["hospital"],
                    "class_logits": result["class_logits"].cpu().numpy().tolist(),
                    "probabilities": probabilities.squeeze(0).cpu().numpy().tolist(),
                    "ordinal_score": float(ordinal_score.item()),
                    "class_expected_rank": float(class_rank.item()),
                    "ordinal_residual": float(residual),
                    "class_uncertainty": float(uncertainty),
                }
            )
    return outputs


def _validation_rank(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    reference = np.sort(np.asarray(reference, dtype=np.float64))
    values = np.asarray(values, dtype=np.float64)
    if reference.size == 0:
        raise ValueError("rank reference is empty")
    return np.searchsorted(reference, values, side="right") / float(reference.size)


def _fpr95(y_true: np.ndarray, scores: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(y_true, scores)
    candidates = fpr[tpr >= 0.95]
    return float(candidates.min()) if len(candidates) else 1.0


def _open_set_metrics(known_scores: np.ndarray, unknown_scores: np.ndarray) -> Dict[str, float]:
    scores = np.concatenate([known_scores, unknown_scores])
    labels = np.concatenate([np.zeros(len(known_scores), dtype=np.int64), np.ones(len(unknown_scores), dtype=np.int64)])
    return {
        "unknown_auroc": float(roc_auc_score(labels, scores)),
        "unknown_aupr": float(average_precision_score(labels, scores)),
        "fpr95": _fpr95(labels, scores),
    }


def _known_metrics(records: Sequence[Mapping[str, object]], known_classes: Sequence[str]) -> Dict[str, object]:
    class_to_id = {name: index for index, name in enumerate(known_classes)}
    truth = np.asarray([class_to_id[str(row["wsl_class"])] for row in records], dtype=np.int64)
    predictions = np.asarray([int(np.argmax(row["probabilities"])) for row in records], dtype=np.int64)
    return {
        "accuracy": float(accuracy_score(truth, predictions)),
        "macro_f1": float(f1_score(truth, predictions, average="macro", zero_division=0)),
        "n": int(len(truth)),
    }


def train_candidate(
    train_items,
    validation_items,
    known_classes,
    input_dim,
    hidden_dim,
    epochs,
    seed,
    ordinal_weight,
    device,
    global_order: Sequence[str] = GLOBAL_ORDER,
    model_variant: str = "shared",
):
    _seed_everything(seed)
    class_to_id = {name: index for index, name in enumerate(known_classes)}
    if model_variant == "shared":
        model_class = OrdinalAttentionMIL
    elif model_variant == "decoupled":
        model_class = DecoupledOrdinalAttentionMIL
    else:
        raise ValueError(f"unsupported model_variant: {model_variant}")
    model = model_class(input_dim, hidden_dim, len(known_classes), len(global_order) - 1).to(device)
    counts = np.bincount([class_to_id[item["wsl_class"]] for item in train_items], minlength=len(known_classes)).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0) / len(known_classes)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    ordinal_criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best_state = None
    best_f1 = -1.0
    for _ in range(epochs):
        model.train()
        order = list(range(len(train_items)))
        random.shuffle(order)
        for index in order:
            item = train_items[index]
            features = torch.from_numpy(item["features"]).to(device=device, dtype=torch.float32)
            output = model(features)
            target_class = torch.tensor([class_to_id[item["wsl_class"]]], device=device)
            ordinal_target = torch.from_numpy(build_ordinal_targets([item["wsl_class"]], global_order)).to(device=device).squeeze(0)
            loss = criterion(output["class_logits"].unsqueeze(0), target_class)
            loss = loss + float(ordinal_weight) * ordinal_criterion(output["ordinal_logits"], ordinal_target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        validation_records = _forward_items(model, validation_items, known_classes, device, global_order)
        validation_metrics = _known_metrics(validation_records, known_classes)
        if validation_metrics["macro_f1"] > best_f1:
            best_f1 = validation_metrics["macro_f1"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    if best_state is None:
        raise RuntimeError("training produced no checkpoint")
    model.load_state_dict(best_state)
    return model


def run_candidate(
    feature_manifest: Path,
    protocol_path: Path,
    output_dir: Path,
    args: argparse.Namespace,
    ordinal_weight: float,
    global_order: Sequence[str] = GLOBAL_ORDER,
):
    from ndpi_mil_baseline import build_bag_manifest, read_csv

    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    rows = read_csv(feature_manifest)
    _, groups, audit = build_bag_manifest(rows)
    if audit["error_count"]:
        raise RuntimeError(f"feature manifest has errors: {audit['errors'][:2]}")
    train_items = _load_items(protocol, groups, "train", args.bag_normalization)
    validation_items = _load_items(protocol, groups, "validation", args.bag_normalization)
    test_items = _load_items(protocol, groups, "test", args.bag_normalization)
    unknown_ids = {str(case_id) for case_id in protocol["test_unknown_ids"]}
    known_test_items = [item for item in test_items if item["case_id"] not in unknown_ids]
    unknown_test_items = [item for item in test_items if item["case_id"] in unknown_ids]
    known_classes = list(protocol["known_classes"])
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = train_candidate(
        train_items,
        validation_items,
        known_classes,
        input_dim=int(audit["feature_dim"]),
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        seed=args.seed,
        ordinal_weight=ordinal_weight,
        device=device,
        global_order=global_order,
        model_variant=args.model_variant,
    )
    validation_records = _forward_items(model, validation_items, known_classes, device, global_order)
    known_records = _forward_items(model, known_test_items, known_classes, device, global_order)
    unknown_records = _forward_items(model, unknown_test_items, known_classes, device, global_order)
    val_uncertainty = np.asarray([row["class_uncertainty"] for row in validation_records])
    val_residual = np.asarray([row["ordinal_residual"] for row in validation_records])
    known_uncertainty = np.asarray([row["class_uncertainty"] for row in known_records])
    unknown_uncertainty = np.asarray([row["class_uncertainty"] for row in unknown_records])
    known_residual = np.asarray([row["ordinal_residual"] for row in known_records])
    unknown_residual = np.asarray([row["ordinal_residual"] for row in unknown_records])
    known_fused = 0.5 * _validation_rank(known_uncertainty, val_uncertainty) + 0.5 * _validation_rank(known_residual, val_residual)
    unknown_fused = 0.5 * _validation_rank(unknown_uncertainty, val_uncertainty) + 0.5 * _validation_rank(unknown_residual, val_residual)
    result = {
        "unknown_class": protocol["unknown_class"],
        "known_classes": known_classes,
        "ordinal_weight": float(ordinal_weight),
        "bag_normalization": args.bag_normalization,
        "model_variant": args.model_variant,
        "label_order_hypothesis": list(global_order),
        "unknown_labels_used_for_fitting": False,
        "known_test": _known_metrics(known_records, known_classes),
        "class_uncertainty": _open_set_metrics(known_uncertainty, unknown_uncertainty),
        "ordinal_residual": _open_set_metrics(known_residual, unknown_residual),
        "ordinal_fused": _open_set_metrics(known_fused, unknown_fused),
        "counts": {"train": len(train_items), "validation": len(validation_items), "known_test": len(known_test_items), "unknown_test": len(unknown_test_items)},
        "records": {"known": known_records, "unknown": unknown_records},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-manifest", type=Path, required=True)
    parser.add_argument("--protocol-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--unknown-classes", nargs="+", default=["ASC-H", "ASC-US", "LISL"])
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20261030)
    parser.add_argument("--device", default="")
    parser.add_argument("--label-order", nargs="+", default=list(GLOBAL_ORDER))
    parser.add_argument("--bag-normalization", choices=["none", "case_center", "case_center_scale"], default="none")
    parser.add_argument("--model-variant", choices=["shared", "decoupled"], default="shared")
    return parser


def _make_variant_args(args: argparse.Namespace, seed: int) -> argparse.Namespace:
    """Clone CLI arguments while replacing the seed exactly once."""

    values = vars(args).copy()
    values["seed"] = int(seed)
    return argparse.Namespace(**values)


def main() -> None:
    args = build_parser().parse_args()
    summary = {"protocol": "source_mixed_leave_one_class_out", "results": {}}
    for index, unknown_class in enumerate(args.unknown_classes):
        protocol_path = args.protocol_root / "leave_one_class_out" / unknown_class / "protocol.json"
        if not protocol_path.is_file():
            raise FileNotFoundError(str(protocol_path))
        summary["results"][unknown_class] = {}
        for variant, ordinal_weight in (("onehot_control", 0.0), ("ordinal_control", 0.5)):
            result = run_candidate(
                args.feature_manifest,
                protocol_path,
                args.out_dir / variant / unknown_class,
                _make_variant_args(args, args.seed + index * 100 + (0 if variant == "onehot_control" else 50)),
                ordinal_weight,
                global_order=args.label_order,
            )
            summary["results"][unknown_class][variant] = {
                "known_test": result["known_test"],
                "class_uncertainty": result["class_uncertainty"],
                "ordinal_residual": result["ordinal_residual"],
                "ordinal_fused": result["ordinal_fused"],
                "counts": result["counts"],
            }
            print(json.dumps({"completed": unknown_class, "variant": variant}, ensure_ascii=False), flush=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# SCORE-MIL ordinal control",
        "",
        f"This is a provisional label-order diagnostic; order={args.label_order}. It is not a clinical semantic claim until NIML/LISL are confirmed.",
        "",
        "| Unknown label | Variant | Known macro-F1 | Class uncertainty AUROC | Ordinal residual AUROC | Ordinal fused AUROC |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for unknown_class, variants in summary["results"].items():
        for variant, values in variants.items():
            lines.append(
                f"| {unknown_class} | {variant} | {values['known_test']['macro_f1']:.3f} | "
                f"{values['class_uncertainty']['unknown_auroc']:.3f} | "
                f"{values['ordinal_residual']['unknown_auroc']:.3f} | "
                f"{values['ordinal_fused']['unknown_auroc']:.3f} |"
            )
    (args.out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
