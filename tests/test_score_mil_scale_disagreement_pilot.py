import numpy as np
import pytest

from score_mil_scale_disagreement_pilot import cache_agreement, choose_case_indices, choose_training_cases, cosine_similarity


def test_training_selection_is_deterministic_and_excludes_outer_unknown():
    outer = {
        "unknown_class": "ASC-H", "known_classes": ["NIML", "HSIL"], "test_ids": ["test"],
        "train_rows": [
            {"case_id": "a", "wsl_class": "NIML", "hospital": "A"},
            {"case_id": "b", "wsl_class": "NIML", "hospital": "A"},
            {"case_id": "c", "wsl_class": "HSIL", "hospital": "B"},
        ],
    }
    first = choose_training_cases(outer, per_group=1)
    second = choose_training_cases({**outer, "train_rows": list(reversed(outer["train_rows"]))}, per_group=1)
    assert [x["case_id"] for x in first] == [x["case_id"] for x in second]
    assert len(first) == 2
    with pytest.raises(ValueError, match="unknown"):
        choose_training_cases({**outer, "train_rows": outer["train_rows"] + [{"case_id": "u", "wsl_class": "ASC-H", "hospital": "A"}]}, per_group=1)
    with pytest.raises(ValueError, match="test"):
        choose_training_cases({**outer, "test_ids": ["a"]}, per_group=3)


def test_parent_indices_are_unique_reproducible_and_within_budget():
    first = choose_case_indices("case-a", seed=5, budget=64, count=2)
    assert first == choose_case_indices("case-a", seed=5, budget=64, count=2)
    assert len(set(first)) == 2 and all(0 <= x < 64 for x in first)


def test_cosine_rejects_zero_or_unaligned_embeddings():
    assert cosine_similarity(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == pytest.approx(0.0)
    with pytest.raises(ValueError, match="embedding"):
        cosine_similarity(np.zeros(2), np.ones(2))
    with pytest.raises(ValueError, match="embedding"):
        cosine_similarity(np.ones(2), np.ones(3))


def test_cache_agreement_tolerates_small_batch_arithmetic_but_rejects_wrong_tile():
    reference = np.array([4.0, 3.0, 2.0, 1.0])
    actual = reference + np.array([0.001, -0.001, 0.0, 0.0])
    result = cache_agreement(actual, reference)
    assert result["relative_l2"] < 0.01 and result["cosine"] > 0.9999
    with pytest.raises(RuntimeError, match="cached"):
        cache_agreement(np.array([1.0, 2.0, 3.0, 4.0]), reference)
