import pytest

from score_mil_candidate_manifest import select_candidates


def _grid_files(width, height):
    return [f"{x}_{y}.jpg" for y in range(height) for x in range(width)]


def test_spatial_sampling_selects_one_patch_in_each_of_64_bins():
    selected = select_candidates(_grid_files(32, 32), mode="spatial", budget=64, bins=8, seed=3)
    bins = {(item["grid_x"] // 4, item["grid_y"] // 4) for item in selected}
    assert len(selected) == 64
    assert len(bins) == 64
    assert len({item["filename"] for item in selected}) == 64


def test_random_sampling_is_deterministic_and_ignores_label_sidecars():
    files = _grid_files(16, 16) + ["0_0.txt", "diagnosis.json"]
    first = select_candidates(files, mode="random", budget=64, bins=8, seed=11)
    second = select_candidates(list(reversed(files)), mode="random", budget=64, bins=8, seed=11)
    assert first == second
    assert all(item["filename"].endswith(".jpg") for item in first)


def test_sampling_rejects_incomplete_budget_and_duplicate_coordinates():
    with pytest.raises(ValueError, match="budget"):
        select_candidates(_grid_files(4, 4), mode="spatial", budget=64, bins=8, seed=1)
    with pytest.raises(ValueError, match="duplicate"):
        select_candidates(_grid_files(8, 8) + ["0_0.png"], mode="spatial", budget=64, bins=8, seed=1)
