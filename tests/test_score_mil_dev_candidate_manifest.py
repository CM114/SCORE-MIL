from pathlib import Path

import pytest

from score_mil_dev_candidate_manifest import build_development_candidates, development_cases


def _protocol():
    return {
        "unknown_class": "U",
        "known_classes": ["A", "B"],
        "train_rows": [{"case_id": "a", "wsl_class": "A", "hospital": "H", "binary_label": "normal"}],
        "validation_rows": [{"case_id": "b", "wsl_class": "B", "hospital": "H", "binary_label": "abnormal"}],
        "test_ids": ["x"],
    }


def test_development_cases_prohibit_unknown_and_test_overlap():
    protocol = _protocol()
    rows = development_cases(protocol)
    assert {(row["case_id"], row["role"]) for row in rows} == {("a", "train"), ("b", "validation")}
    with pytest.raises(ValueError, match="unknown"):
        development_cases(dict(protocol, train_rows=[dict(protocol["train_rows"][0], wsl_class="U")]))
    with pytest.raises(ValueError, match="overlap"):
        development_cases(dict(protocol, test_ids=["a"]))


def test_candidate_policies_use_same_cases_and_budget_without_sidecars(tmp_path: Path):
    protocol = _protocol()
    for row in protocol["train_rows"] + protocol["validation_rows"]:
        image_dir = tmp_path / row["wsl_class"] / row["hospital"] / row["case_id"] / "img"
        image_dir.mkdir(parents=True)
        for x in range(4):
            for y in range(4):
                (image_dir / f"{x}_{y}.jpg").touch()
        (image_dir / "0_0.txt").touch()
    manifests, audits, errors = build_development_candidates(protocol, tmp_path, budget=4, bins=2, seed=9)
    assert not errors
    assert len(audits) == 2
    for policy in ("random", "spatial"):
        rows = manifests[policy]
        assert len(rows) == 8
        assert {(row["case_id"], row["role"]) for row in rows} == {("a", "train"), ("b", "validation")}
        assert all(row["image_path"].endswith(".jpg") for row in rows)
