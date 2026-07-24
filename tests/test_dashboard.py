"""Tests for the review queue and decision recording.
Run: pytest tests/test_dashboard.py -v

These test the parts a reviewer's trust depends on: that the queue does not show
finished work, that ordering puts the biggest money first, and that a recorded
decision actually survives.
"""
import tempfile
from pathlib import Path

import pytest

from app.review_logic import (
    build_queue, queue_summary, save_decision, load_decisions, VALID_DECISIONS,
)


def _v(cid, score=0.7, excess=1000.0, action="HUMAN_REVIEW", findings=2):
    return {"claim_id": cid, "fraud_score": score, "estimated_excess_inr": excess,
            "recommended_action": action, "n_findings": findings,
            "billed_total_inr": excess * 3}


@pytest.fixture
def tmp_decisions():
    with tempfile.TemporaryDirectory() as d:
        yield str(Path(d) / "decisions")


# ---------- queue ----------

def test_auto_approved_claims_are_not_in_the_review_queue():
    v = [_v("A", action="AUTO_APPROVE"), _v("B", action="HUMAN_REVIEW")]
    q = build_queue(v, {}, mode="needs_review")
    assert [x["claim_id"] for x in q] == ["B"]


def test_already_decided_claims_leave_the_queue():
    v = [_v("A"), _v("B")]
    q = build_queue(v, {"A": {"adjudicator_decision": "accepted"}}, mode="needs_review")
    assert [x["claim_id"] for x in q] == ["B"]


def test_decided_mode_shows_only_decided():
    v = [_v("A"), _v("B")]
    q = build_queue(v, {"A": {}}, mode="decided")
    assert [x["claim_id"] for x in q] == ["A"]


def test_all_mode_shows_everything_including_auto_approved():
    v = [_v("A", action="AUTO_APPROVE"), _v("B")]
    q = build_queue(v, {}, mode="all")
    assert len(q) == 2


def test_queue_sorts_biggest_money_first_by_default():
    v = [_v("small", excess=500.0), _v("big", excess=90000.0), _v("mid", excess=7000.0)]
    q = build_queue(v, {}, mode="all")
    assert [x["claim_id"] for x in q] == ["big", "mid", "small"]


def test_queue_can_sort_by_score():
    v = [_v("lo", score=0.2, excess=99999.0), _v("hi", score=0.95, excess=10.0)]
    q = build_queue(v, {}, mode="all", sort_by="score")
    assert [x["claim_id"] for x in q] == ["hi", "lo"]


def test_filters_apply():
    v = [_v("A", score=0.2, excess=100.0), _v("B", score=0.9, excess=50000.0)]
    assert [x["claim_id"] for x in build_queue(v, {}, mode="all", min_score=0.5)] == ["B"]
    assert [x["claim_id"] for x in build_queue(v, {}, mode="all", min_excess=1000)] == ["B"]


def test_empty_input_gives_empty_queue():
    assert build_queue([], {}, mode="needs_review") == []


# ---------- summary ----------

def test_summary_counts_and_money():
    v = [_v("A", excess=1000.0), _v("B", excess=2500.0),
         _v("C", excess=0.0, action="AUTO_APPROVE")]
    s = queue_summary(v, {"A": {}})
    assert s["total_claims"] == 3
    assert s["flagged"] == 2
    assert s["excess_at_stake_inr"] == 3500.0
    assert s["decided"] == 1
    assert s["outstanding"] == 1          # B still waiting


# ---------- decisions ----------

def test_decision_is_saved_and_reloaded(tmp_decisions):
    save_decision(tmp_decisions, "CLM-1", "rejected", "ayush", "clear overbilling")
    got = load_decisions(tmp_decisions)
    assert got["CLM-1"]["adjudicator_decision"] == "rejected"
    assert got["CLM-1"]["decided_by"] == "ayush"
    assert got["CLM-1"]["note"] == "clear overbilling"
    assert got["CLM-1"]["decided_ts"]


def test_redeciding_replaces_not_duplicates(tmp_decisions):
    save_decision(tmp_decisions, "CLM-1", "accepted", "a")
    save_decision(tmp_decisions, "CLM-1", "rejected", "b")
    got = load_decisions(tmp_decisions)
    assert len(got) == 1
    assert got["CLM-1"]["adjudicator_decision"] == "rejected"


def test_multiple_claims_coexist(tmp_decisions):
    save_decision(tmp_decisions, "CLM-1", "accepted", "a")
    save_decision(tmp_decisions, "CLM-2", "escalated", "b")
    got = load_decisions(tmp_decisions)
    assert set(got) == {"CLM-1", "CLM-2"}


def test_invalid_decision_is_refused(tmp_decisions):
    with pytest.raises(ValueError):
        save_decision(tmp_decisions, "CLM-1", "maybe", "a")


def test_missing_decisions_table_is_not_an_error():
    assert load_decisions("./data/reference/definitely_not_here") == {}


def test_valid_decisions_are_the_expected_three():
    assert VALID_DECISIONS == {"accepted", "rejected", "escalated"}
