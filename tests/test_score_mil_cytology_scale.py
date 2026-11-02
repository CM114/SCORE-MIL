import numpy as np
import pytest
from PIL import Image

from score_mil_cytology_scale import compare_synthetic_embeddings, crop_boxes, extract_native_crops, grid_starts


def test_native_grid_covers_parent_and_has_expected_positions():
    assert grid_starts(1024, 224, 5) == [0, 200, 400, 600, 800]
    boxes = crop_boxes(1024, 1024, 224, 5)
    mask = np.zeros((1024, 1024), dtype=np.uint8)
    for x1, y1, x2, y2 in boxes:
        mask[y1:y2, x1:x2] += 1
    assert len(boxes) == 25
    assert np.min(mask) >= 1


def test_crop_pixels_keep_native_resolution_without_resize():
    image = np.arange(256 * 256 * 3, dtype=np.uint8).reshape(256, 256, 3)
    crops = extract_native_crops(Image.fromarray(image), crop_size=128, grid_size=2)
    assert len(crops) == 4
    assert all(crop.size == (128, 128) for crop in crops)
    np.testing.assert_array_equal(np.asarray(crops[0]), image[:128, :128])
    np.testing.assert_array_equal(np.asarray(crops[-1]), image[128:, 128:])


def test_rejects_small_images_and_grid_with_coverage_gaps():
    with pytest.raises(ValueError, match="crop"):
        grid_starts(100, 224, 5)
    with pytest.raises(ValueError, match="coverage"):
        grid_starts(1024, 224, 2)


def test_rectangular_parent_coverage():
    boxes = crop_boxes(256, 192, 128, 2)
    assert boxes == [(0, 0, 128, 128), (128, 0, 256, 128), (0, 64, 128, 192), (128, 64, 256, 192)]


def test_embedding_comparison_reports_difference_and_rejects_bad_shapes():
    a = np.array([[1.0, 0.0]], dtype=np.float32)
    b = np.array([[1.0, 0.1]], dtype=np.float32)
    summary = compare_synthetic_embeddings(a, b)
    assert summary["max_abs_difference"] == pytest.approx(0.1)
    assert 0.99 < summary["cosine_similarity"] < 1.0
    with pytest.raises(ValueError, match="shape"):
        compare_synthetic_embeddings(a, np.ones((2, 2)))
