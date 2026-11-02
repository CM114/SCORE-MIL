from pathlib import Path

import numpy as np
import pytest
from types import SimpleNamespace
from unittest.mock import patch

from score_mil_scale_factorial import _check_case_file, _save_case_file, collapse_views, dispatch_training, select_pilot_cases, validate_subset_candidates


def _protocol():
    return {
        "unknown_class": "ASC-H", "known_classes": ["NIML", "ASC-US", "LISL", "HSIL"],
        "test_ids": ["test"],
        "train_rows": [
            {"case_id": "a", "wsl_class": "NIML", "hospital": "A", "binary_label": "normal"},
            {"case_id": "b", "wsl_class": "ASC-US", "hospital": "A", "binary_label": "abnormal"},
            {"case_id": "c", "wsl_class": "LISL", "hospital": "B", "binary_label": "abnormal"},
            {"case_id": "d", "wsl_class": "HSIL", "hospital": "B", "binary_label": "abnormal"},
        ],
        "validation_rows": [
            {"case_id": "v", "wsl_class": "NIML", "hospital": "A", "binary_label": "normal"},
        ],
    }


def _rows(cases, root):
    rows = []
    for case in cases:
        for i in range(2):
            rows.append({**case, "candidate_index": str(i), "grid_x": str(i), "grid_y": "0",
                         "image_path": str(root / case["wsl_class"] / case["hospital"] / case["case_id"] / "img" / f"{i}_0.jpg")})
    return rows


def test_selected_cases_are_development_only_and_deterministic():
    first = select_pilot_cases(_protocol(), 4)
    second = select_pilot_cases(_protocol(), 4)
    assert first == second
    assert sum(r["role"] == "train" for r in first) == 4
    assert sum(r["role"] == "validation" for r in first) == 1
    assert {r["case_id"] for r in first}.isdisjoint({"test"})


def test_candidate_pairing_checks_index_label_and_path():
    root = Path("/synthetic/WSL")
    cases = select_pilot_cases(_protocol(), 4)
    rows = _rows(cases, root)
    validate_subset_candidates(rows, cases, root, 2)
    bad = [dict(x) for x in rows]
    bad[0]["image_path"] = str(root / "elsewhere.jpg")
    with pytest.raises(ValueError, match="path"):
        validate_subset_candidates(bad, cases, root, 2)
    bad = [dict(x) for x in rows]
    bad[0]["candidate_index"] = bad[1]["candidate_index"]
    with pytest.raises(ValueError, match="duplicate"):
        validate_subset_candidates(bad, cases, root, 2)
    bad = [dict(x) for x in rows]
    bad[0]["wsl_class"] = "ASC-H"
    with pytest.raises(ValueError, match="metadata"):
        validate_subset_candidates(bad, cases, root, 2)


def test_collapse_preserves_parent_and_native_view_order():
    x = np.arange(2 * 26 * 4, dtype=np.float32).reshape(2, 26, 4)
    whole, native = collapse_views(x)
    np.testing.assert_array_equal(whole, x[:, 0])
    np.testing.assert_allclose(native, x[:, 1:].mean(axis=1))
    with pytest.raises(ValueError, match="26"):
        collapse_views(x[:, :25])
    x[0, 3, 1] = np.nan
    with pytest.raises(ValueError, match="finite"):
        collapse_views(x)


def test_outer_unknown_and_test_are_rejected():
    protocol = _protocol()
    protocol["train_rows"][0]["case_id"] = "test"
    with pytest.raises(ValueError, match="overlap"):
        select_pilot_cases(protocol, 4)
    protocol = _protocol()
    protocol["train_rows"][0]["wsl_class"] = "ASC-H"
    with pytest.raises(ValueError, match="unknown"):
        select_pilot_cases(protocol, 4)


def test_saved_native_crop_features_are_recoverable_without_overwrite(tmp_path):
    crops = np.arange(2 * 25 * 4, dtype=np.float32).reshape(2, 25, 4)
    whole = np.ones((2, 4), dtype=np.float32)
    native_mean = crops.mean(axis=1)
    path = tmp_path / "case.npz"
    _save_case_file(path, whole, native_mean, crops)
    _check_case_file(path, 2, 4)
    with np.load(path, allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["native_crops"], crops)
        np.testing.assert_array_equal(saved["native_mean"], native_mean)
    with pytest.raises(FileExistsError):
        _save_case_file(path, whole, native_mean, crops)


def test_saved_crop_mean_mismatch_is_rejected(tmp_path):
    path = tmp_path / "bad.npz"
    _save_case_file(path, np.ones((2, 4)), np.ones((2, 4)), np.zeros((2, 25, 4)))
    with pytest.raises(ValueError, match="mean"):
        _check_case_file(path, 2, 4)


def test_dual_gpu_option_routes_training_to_separate_worker_output():
    args = SimpleNamespace(
        train_devices=["cuda:0", "cuda:1"],
        out_dir=Path("experiment"),
        protocol=Path("protocol.json"),
        seeds=[11, 12, 13],
        epochs=12,
    )
    multi_gpu = SimpleNamespace(train_from_features=lambda *args, **kwargs: {"args": args, "kwargs": kwargs})
    with patch.dict("sys.modules", {"score_mil_scale_factorial_multigpu": multi_gpu}):
        result = dispatch_training([{"case_id": "synthetic"}], {}, args, {})

    assert result["args"] == (
        Path("experiment"), Path("protocol.json"), Path("experiment") / "training_two_gpu"
    )
    assert result["kwargs"] == {
        "devices": ["cuda:0", "cuda:1"],
        "pseudo_unknowns": ("ASC-US", "LISL"),
        "seeds": [11, 12, 13],
        "epochs": 12,
    }
