import numpy as np
import pytest

from score_mil_known_calibration_audit import evaluate_known_calibration, split_known_by_hash, upper_tail_threshold


def _row(cid, label, unknown, uncertainty, residual):
    return {"case_id": cid, "wsl_class": label, "is_pseudo_unknown": unknown, "class_uncertainty": uncertainty,
            "ordinal_residual": residual, "predicted_class": label if not unknown else "NIML"}


def test_known_split_is_class_stratified_disjoint_and_unknown_free():
    rows = [_row(f"n{i}", "NIML", 0, .1, .2) for i in range(4)] + [_row(f"h{i}", "HSIL", 0, .3, .4) for i in range(4)] + [_row("u", "LISL", 1, .9, .9)]
    a, b = split_known_by_hash(rows, ["NIML", "HSIL"], seed=10)
    assert len(a) == len(b) == 4
    assert {r["case_id"] for r in a}.isdisjoint({r["case_id"] for r in b})
    assert all(not r["is_pseudo_unknown"] for r in a + b)
    assert sum(r["wsl_class"] == "HSIL" for r in a) == 2
    c, d = split_known_by_hash(list(reversed(rows)), ["NIML", "HSIL"], seed=10)
    assert [r["case_id"] for r in a] == [r["case_id"] for r in c]
    assert [r["case_id"] for r in b] == [r["case_id"] for r in d]


def test_upper_tail_order_statistic_uses_only_known_scores():
    assert upper_tail_threshold([.1, .2, .3, .4], alpha=.4) == pytest.approx(.3)
    assert np.isposinf(upper_tail_threshold([.1, .2, .3, .4], alpha=.05))
    with pytest.raises(ValueError, match="alpha"):
        upper_tail_threshold([.1], alpha=1.0)


def test_unknown_scores_do_not_change_calibration_threshold():
    calibration = [_row("a", "NIML", 0, .1, .2), _row("b", "HSIL", 0, .4, .5), _row("c", "NIML", 0, .2, .3), _row("d", "HSIL", 0, .3, .4)]
    known_eval = [_row("e", "NIML", 0, .2, .2), _row("f", "HSIL", 0, .8, .7), _row("g", "HSIL", 0, .15, .2)]
    known_eval[1]["predicted_class"] = "NIML"
    known_eval[2]["predicted_class"] = "NIML"
    unknown_a = [_row("u", "LISL", 1, .9, .9)]
    unknown_b = [_row("u", "LISL", 1, .05, .05)]
    first = evaluate_known_calibration(calibration, known_eval, unknown_a, "class_uncertainty", alpha=.4)
    second = evaluate_known_calibration(calibration, known_eval, unknown_b, "class_uncertainty", alpha=.4)
    assert first["threshold"] == second["threshold"]
    assert first["unknown_detected"] == 1 and second["unknown_detected"] == 0
    assert first["by_known_class"]["HSIL"]["false_alarm_count"] == 1
    assert first["by_known_class"]["HSIL"]["misclassified_unflagged_count"] == 1
