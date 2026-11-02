import numpy as np
import pytest

from score_mil_cross_split_duplicate_audit import feature_digest, match_byte_groups


def test_exact_feature_index_and_byte_verification():
    assert feature_digest(np.array([1, 2], dtype=np.float32)) == feature_digest(np.array([1, 2], dtype=np.float32))
    assert feature_digest(np.array([1, 2], dtype=np.float32)) != feature_digest(np.array([1, 3], dtype=np.float32))
    with pytest.raises(ValueError):
        feature_digest(np.array([np.nan], dtype=np.float32))
    values = {"t1": "same", "t2": "train-only", "v1": "same", "v2": "val-only"}
    groups = match_byte_groups(["t1", "t2"], ["v1", "v2"], lambda x: values[x])
    assert groups == [(["t1"], ["v1"])]
