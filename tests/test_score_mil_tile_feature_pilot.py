import numpy as np
import pytest
import torch
from PIL import Image

from score_mil_tile_feature_pilot import _extract_policy, content_fraction_rgb, validate_candidate_rows, write_case_features


def test_content_fraction_distinguishes_light_blank_from_dark_pixels():
    image = np.full((4, 4, 3), 240, dtype=np.uint8)
    assert content_fraction_rgb(image) == 0.0
    image[:2, :2] = [50, 50, 50]
    assert content_fraction_rgb(image) == pytest.approx(0.25)


def test_candidate_validation_rejects_duplicate_indices_and_nontraining_roles():
    valid = [
        {"case_id": "a", "role": "train", "candidate_index": "0", "image_path": "/x"},
        {"case_id": "a", "role": "train", "candidate_index": "1", "image_path": "/y"},
    ]
    validate_candidate_rows(valid, expected_per_case=2)
    with pytest.raises(ValueError, match="duplicate"):
        validate_candidate_rows(valid + [valid[0]], expected_per_case=2)
    with pytest.raises(ValueError, match="train"):
        validate_candidate_rows([dict(r, role="test") for r in valid], expected_per_case=2)


def test_feature_bags_keep_candidate_order_and_original_labels(tmp_path):
    rows = [
        {"case_id": "case", "candidate_index": "1", "wsl_class": "NIML", "hospital": "h", "grid_x": "1", "grid_y": "0"},
        {"case_id": "case", "candidate_index": "0", "wsl_class": "NIML", "hospital": "h", "grid_x": "0", "grid_y": "0"},
    ]
    features = np.asarray([[10, 11], [20, 21]], dtype=np.float32)
    output = write_case_features(rows, features, {"case": "normal"}, tmp_path)
    assert len(output) == 2
    stored = np.load(output[0]["feature_path"], allow_pickle=False)
    np.testing.assert_array_equal(stored, [[20, 21], [10, 11]])
    assert [item["candidate_index"] for item in output] == [0, 1]
    assert all(item["binary_label"] == "normal" for item in output)


def test_actual_tile_loading_and_batched_extraction_preserve_order(tmp_path):
    paths = [tmp_path / "blank.png", tmp_path / "dark.png"]
    Image.fromarray(np.full((16, 16, 3), 240, dtype=np.uint8)).save(paths[0])
    Image.fromarray(np.full((16, 16, 3), 50, dtype=np.uint8)).save(paths[1])

    class DummyEncoder(torch.nn.Module):
        def forward(self, x):
            return x.mean(dim=(1, 2, 3)).unsqueeze(1).expand(-1, 2048)

    rows = [{"image_path": str(path)} for path in paths]
    features, fractions, stats = _extract_policy(rows, DummyEncoder(), torch.device("cpu"), 2, 0, "test")
    assert features.shape == (2, 2048)
    np.testing.assert_allclose(fractions, [0.0, 1.0])
    assert features[0, 0] > features[1, 0]
    assert stats["image_count"] == 2
