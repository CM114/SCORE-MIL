#!/usr/bin/env python3
"""Run the ordinal-attention MIL training path on deterministic synthetic bags."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from score_mil_true_pseudo_openworld import train_one  # noqa: E402


KNOWN_ORDER = ("NIML", "LISL", "HSIL")
PSEUDO_UNKNOWN = "ASC-US"
FEATURE_DIM = 12
INSTANCES_PER_BAG = 16


def _make_item(case_id: str, label: str, center: np.ndarray, rng: np.random.Generator) -> dict:
    features = rng.normal(loc=center, scale=0.55, size=(INSTANCES_PER_BAG, FEATURE_DIM)).astype(np.float32)
    # Add a small number of high-salience instances to exercise bag attention.
    features[0, 3] += 1.1
    return {
        "case_id": case_id,
        "wsl_class": label,
        "hospital": "synthetic-source",
        "features": features,
    }


def make_synthetic_data(seed: int = 20261102):
    rng = np.random.default_rng(seed)
    centers = {
        "NIML": np.asarray([-1.1, 0.0] + [0.0] * (FEATURE_DIM - 2), dtype=np.float32),
        "LISL": np.asarray([0.0, 0.8] + [0.0] * (FEATURE_DIM - 2), dtype=np.float32),
        "HSIL": np.asarray([1.1, 0.0] + [0.0] * (FEATURE_DIM - 2), dtype=np.float32),
        # The pseudo-unknown is interpolated between known classes by design.
        "ASC-US": np.asarray([-0.35, 0.55] + [0.0] * (FEATURE_DIM - 2), dtype=np.float32),
    }
    training = []
    validation = []
    for label in KNOWN_ORDER:
        for index in range(10):
            training.append(_make_item(f"synthetic_train_{label}_{index:02d}", label, centers[label], rng))
        for index in range(5):
            validation.append(_make_item(f"synthetic_val_{label}_{index:02d}", label, centers[label], rng))
    for index in range(10):
        validation.append(_make_item(f"synthetic_val_unknown_{index:02d}", PSEUDO_UNKNOWN, centers[PSEUDO_UNKNOWN], rng))
    return training, validation


def run(output_dir: Path) -> dict:
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    training, validation = make_synthetic_data()
    result = train_one(
        training,
        validation,
        KNOWN_ORDER,
        PSEUDO_UNKNOWN,
        "unweighted",
        seed=20261102,
        epochs=8,
        hidden_dim=16,
        device=torch.device("cpu"),
        out_dir=output_dir,
    )
    metrics = result["metrics"]
    report = {
        "synthetic_only": True,
        "training_cases": len(training),
        "known_validation_cases": sum(row["wsl_class"] != PSEUDO_UNKNOWN for row in validation),
        "pseudo_unknown_validation_cases": sum(row["wsl_class"] == PSEUDO_UNKNOWN for row in validation),
        "seed": result["seed"],
        "epochs": result["epochs"],
        "unknown_detection": metrics["ordinal_fused"],
        "known_classification": metrics["known_classification"],
        "result_file": "result.json",
        "prediction_file": "validation_predictions.csv",
        "warning": "Synthetic software-path demonstration only; not a clinical performance estimate.",
    }
    summary_path = output_dir / "synthetic_demo_summary.json"
    summary_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True,
                        help="new directory for synthetic-only result files")
    args = parser.parse_args()
    run(args.out_dir)


if __name__ == "__main__":
    main()
