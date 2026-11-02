from score_mil_exact_jpeg_duplicate_audit import cross_role_groups


def test_only_hashes_present_in_both_roles_are_reported():
    rows = [{"role": "train", "sha256": b"a"}, {"role": "validation", "sha256": b"a"},
            {"role": "train", "sha256": b"b"}, {"role": "validation", "sha256": b"c"}]
    groups = cross_role_groups(rows)
    assert len(groups) == 1
    assert len(groups[0]["train"]) == len(groups[0]["validation"]) == 1
