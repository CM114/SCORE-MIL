import pytest

from score_mil_scale_factorial_bootstrap import aligned_scores


def _row(label, pseudo, score, hospital="A"):
    return {"is_pseudo_unknown": str(pseudo), "wsl_class": label, "hospital": hospital,
            "ordinal_fused": str(score)}


def test_bootstrap_alignment_preserves_case_pairing_and_rejects_mismatch():
    table = {"a": _row("NIML", 0, .1), "b": _row("LISL", 1, .8)}
    other = {"b": _row("LISL", 1, .9), "a": _row("NIML", 0, .2)}
    y, first, second = aligned_scores({(1, "left"): table, (1, "right"): other}, [1], "left", "right")
    assert y.tolist() == [0, 1]
    assert first.tolist() == [[.1, .8]]
    assert second.tolist() == [[.2, .9]]
    other["a"]["hospital"] = "B"
    with pytest.raises(ValueError, match="metadata"):
        aligned_scores({(1, "left"): table, (1, "right"): other}, [1], "left", "right")
