import numpy as np

from ndpi_mil_baseline import build_bag_manifest


def test_build_bag_manifest_reports_feature_dimension_from_manifest(tmp_path):
    feature_path = tmp_path / "case.npy"
    np.save(feature_path, np.ones((2, 4), dtype=np.float32))
    rows = [
        {
            "case_id": "case-1",
            "candidate_index": "0",
            "feature_row": "0",
            "feature_path": str(feature_path),
            "feature_dim": "4",
            "feature_status": "ok",
            "wsl_class": "NIML",
            "binary_label": "normal",
            "split": "train",
            "hospital": "H1",
        },
        {
            "case_id": "case-1",
            "candidate_index": "1",
            "feature_row": "1",
            "feature_path": str(feature_path),
            "feature_dim": "4",
            "feature_status": "ok",
            "wsl_class": "NIML",
            "binary_label": "normal",
            "split": "train",
            "hospital": "H1",
        },
    ]
    _, _, audit = build_bag_manifest(rows)
    assert audit["error_count"] == 0
    assert audit["feature_dim"] == 4
