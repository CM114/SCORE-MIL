import argparse

import numpy as np
import torch

from score_mil_ordinal_control import (
    GLOBAL_ORDER,
    OrdinalAttentionMIL,
    _make_variant_args,
    build_ordinal_targets,
    ordinal_expected_rank,
    patch_residual_scores,
    ordinal_residual_score,
)


def test_build_ordinal_targets_uses_cumulative_global_order():
    labels = np.asarray(["NIML", "ASC-US", "HSIL"])
    targets = build_ordinal_targets(labels, GLOBAL_ORDER)
    assert targets.shape == (3, 4)
    np.testing.assert_array_equal(targets[0], [0, 0, 0, 0])
    np.testing.assert_array_equal(targets[1], [1, 0, 0, 0])
    np.testing.assert_array_equal(targets[2], [1, 1, 1, 1])


def test_build_ordinal_targets_accepts_a_control_order():
    labels = np.asarray(["NIML", "ASC-H"])
    targets = build_ordinal_targets(labels, ("NIML", "ASC-H", "ASC-US", "LISL", "HSIL"))
    np.testing.assert_array_equal(targets[1], [1, 0, 0, 0])


def test_ordinal_expected_rank_is_monotonic():
    logits = torch.tensor([[3.0, 3.0, 3.0, 3.0], [-3.0, -3.0, -3.0, -3.0]])
    scores = ordinal_expected_rank(logits)
    assert scores[0] > scores[1]
    assert torch.all(scores >= 0)
    assert torch.all(scores <= 4)


def test_residual_score_is_larger_when_ordinal_and_class_evidence_disagree():
    ordinal_score = torch.tensor([3.0, 1.0])
    class_expected_rank = torch.tensor([3.0, 3.0])
    residual = ordinal_residual_score(ordinal_score, class_expected_rank)
    assert residual[0] < residual[1]


def test_model_returns_class_and_ordinal_outputs_for_variable_bags():
    model = OrdinalAttentionMIL(input_dim=8, hidden_dim=4, num_classes=3, num_global_thresholds=4)
    output = model(torch.randn(5, 8))
    assert output["class_logits"].shape == (3,)
    assert output["ordinal_logits"].shape == (4,)
    assert output["patch_class_logits"].shape == (5, 3)
    assert output["patch_ordinal_logits"].shape == (5, 4)
    assert output["attention"].shape == (5,)
    assert torch.isfinite(output["class_logits"]).all()


def test_patch_residual_scores_align_categorical_and_ordinal_evidence():
    patch_class_logits = torch.tensor([[8.0, -8.0], [-8.0, 8.0]])
    patch_ordinal_logits = torch.tensor([[-8.0, -8.0, -8.0, -8.0], [-8.0, -8.0, -8.0, -8.0]])
    scores = patch_residual_scores(
        patch_ordinal_logits,
        patch_class_logits,
        known_classes=("NIML", "HSIL"),
        global_order=GLOBAL_ORDER,
    )
    assert scores.shape == (2,)
    assert scores[0] < scores[1]
    assert torch.isfinite(scores).all()


def test_variant_args_overrides_seed_without_duplicate_keyword():
    args = argparse.Namespace(seed=10, epochs=1)
    variant = _make_variant_args(args, 60)
    assert variant.seed == 60
    assert variant.epochs == 1
