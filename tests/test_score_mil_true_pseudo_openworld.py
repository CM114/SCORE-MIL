import numpy as np
import pytest
import torch

from score_mil_true_pseudo_openworld import (
    build_inner_partition,
    class_loss_weights,
    known_only_rank_order,
    openworld_scores,
    train_one,
)


def _outer():
    return {
        "unknown_class": "ASC-H",
        "known_classes": ["NIML", "ASC-US", "LISL", "HSIL"],
        "train_rows": [
            {"case_id": "a", "wsl_class": "NIML"},
            {"case_id": "b", "wsl_class": "ASC-US"},
            {"case_id": "c", "wsl_class": "LISL"},
            {"case_id": "d", "wsl_class": "HSIL"},
        ],
        "validation_rows": [
            {"case_id": "e", "wsl_class": "NIML"},
            {"case_id": "f", "wsl_class": "ASC-US"},
            {"case_id": "g", "wsl_class": "LISL"},
            {"case_id": "h", "wsl_class": "HSIL"},
        ],
        "test_ids": ["x"],
    }


def test_inner_unknown_class_is_absent_from_training_and_known_order():
    part = build_inner_partition(_outer(), pseudo_unknown="ASC-US")
    assert {row["wsl_class"] for row in part["train_rows"]} == {"NIML", "LISL", "HSIL"}
    assert {row["wsl_class"] for row in part["unknown_validation_rows"]} == {"ASC-US"}
    assert known_only_rank_order(part["known_classes"]) == ("NIML", "LISL", "HSIL")
    with pytest.raises(ValueError, match="pseudo"):
        build_inner_partition(_outer(), pseudo_unknown="ASC-H")


def test_sqrt_weights_are_less_extreme_and_have_training_average_one():
    counts = {"A": 90, "B": 10}
    weights = class_loss_weights(counts, ["A", "B"], mode="sqrt_inverse")
    assert weights["B"] > weights["A"] > 0
    assert weights["B"] / weights["A"] == pytest.approx(3.0)
    assert (90 * weights["A"] + 10 * weights["B"]) / 100 == pytest.approx(1.0)
    assert class_loss_weights(counts, ["A", "B"], mode="unweighted") == {"A": 1.0, "B": 1.0}


def test_fused_scores_rank_against_known_only_validation_reference():
    records = [
        {"is_pseudo_unknown": 0, "class_uncertainty": 0.1, "ordinal_residual": 0.2},
        {"is_pseudo_unknown": 0, "class_uncertainty": 0.2, "ordinal_residual": 0.1},
        {"is_pseudo_unknown": 1, "class_uncertainty": 0.9, "ordinal_residual": 0.8},
    ]
    scores = openworld_scores(records)
    assert scores[2] > max(scores[:2])


def test_inner_training_runs_with_pseudo_unknown_only_in_validation(tmp_path):
    torch.set_num_threads(1)
    known = ("NIML", "LISL", "HSIL")
    rng = np.random.default_rng(8)

    def item(label, index):
        return {
            "case_id": f"{label}-{index}",
            "wsl_class": label,
            "hospital": "hospital-a",
            "features": rng.normal(size=(4, 8)).astype(np.float32),
        }

    train = [item(label, i) for label in known for i in range(2)]
    validation = [item(label, 5) for label in known + ("ASC-US",)]
    out = tmp_path / "run"
    result = train_one(train, validation, known, "ASC-US", "sqrt_inverse", 12, 1, 4, torch.device("cpu"), out)
    assert set(result["train_class_counts"]) == set(known)
    assert result["metrics"]["known_classification"]["n"] == 3
    assert 0.0 <= result["metrics"]["ordinal_fused"]["unknown_auroc"] <= 1.0
    assert (out / "fixed_epoch.pt").is_file()
    assert (out / "validation_predictions.csv").is_file()
