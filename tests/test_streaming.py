"""Tests for the streaming layer. Run: pytest tests/test_streaming.py -v

The tests that matter here are idempotence and late events. A stream that double-counts
on replay, or silently drops stragglers, produces numbers that look fine and are wrong
— which is worse than an obvious crash.
"""
from datetime import datetime, timedelta

import pytest

from streaming.windows import WindowedCounters, floor_to_window
from streaming.stream_job import reconcile


def _ev(cid, provider="H01", ts=None, billed=10000.0, excess=0.0,
        action="AUTO_APPROVE", score=0.1, icu=False):
    return {"claim_id": cid, "provider_id": provider,
            "event_ts": (ts or datetime(2026, 7, 1, 10, 5)).isoformat(),
            "billed_total_inr": billed, "estimated_excess_inr": excess,
            "recommended_action": action, "fraud_score": score, "has_icu": icu}


# ---------- windowing ----------

def test_floor_to_window_buckets_correctly():
    assert floor_to_window(datetime(2026, 7, 1, 10, 5), 60) == datetime(2026, 7, 1, 10, 0)
    assert floor_to_window(datetime(2026, 7, 1, 10, 59), 60) == datetime(2026, 7, 1, 10, 0)
    assert floor_to_window(datetime(2026, 7, 1, 11, 0), 60) == datetime(2026, 7, 1, 11, 0)
    assert floor_to_window(datetime(2026, 7, 1, 10, 20), 15) == datetime(2026, 7, 1, 10, 15)


def test_claims_land_in_separate_windows():
    c = WindowedCounters(window_minutes=60)
    c.add(_ev("A", ts=datetime(2026, 7, 1, 10, 5)))
    c.add(_ev("B", ts=datetime(2026, 7, 1, 11, 5)))
    assert len(c.snapshot()) == 2


def test_same_window_aggregates():
    c = WindowedCounters(window_minutes=60)
    c.add(_ev("A", ts=datetime(2026, 7, 1, 10, 5), billed=1000.0))
    c.add(_ev("B", ts=datetime(2026, 7, 1, 10, 50), billed=3000.0))
    snap = c.snapshot()
    assert len(snap) == 1 and snap[0]["n_claims"] == 2
    assert snap[0]["billed_inr"] == 4000.0


def test_providers_are_counted_separately():
    c = WindowedCounters()
    c.add(_ev("A", provider="H01"))
    c.add(_ev("B", provider="H02"))
    assert {r["provider_id"] for r in c.snapshot()} == {"H01", "H02"}


# ---------- idempotence: the replay property ----------

def test_replaying_the_same_claim_changes_nothing():
    c = WindowedCounters()
    assert c.add(_ev("A", billed=5000.0)) == "counted"
    assert c.add(_ev("A", billed=5000.0)) == "duplicate"
    assert c.add(_ev("A", billed=5000.0)) == "duplicate"
    snap = c.snapshot()
    assert snap[0]["n_claims"] == 1 and snap[0]["billed_inr"] == 5000.0


def test_full_replay_gives_identical_counters():
    events = [_ev(f"C{i}", ts=datetime(2026, 7, 1, 10, i)) for i in range(10)]
    a, b = WindowedCounters(), WindowedCounters()
    for e in events:
        a.add(e)
    for e in events * 3:                      # broker restarts, everything replays
        b.add(e)
    assert a.snapshot() == b.snapshot()


# ---------- late events ----------

def test_late_event_is_counted_not_dropped_within_watermark():
    c = WindowedCounters(window_minutes=60, watermark_minutes=120)
    c.add(_ev("new", ts=datetime(2026, 7, 1, 12, 0)))
    status = c.add(_ev("straggler", ts=datetime(2026, 7, 1, 11, 0)))   # 1h late
    assert status == "counted"
    assert c.stats()["late_events"] == 0


def test_event_beyond_watermark_is_reported_as_late():
    c = WindowedCounters(window_minutes=60, watermark_minutes=60)
    c.add(_ev("new", ts=datetime(2026, 7, 1, 15, 0)))
    status = c.add(_ev("ancient", ts=datetime(2026, 7, 1, 9, 0)))      # 6h late
    assert status == "late"
    assert c.stats()["late_events"] == 1       # surfaced, never silently discarded


# ---------- alerts ----------

def test_no_alert_on_a_tiny_sample():
    """Two flagged claims is noise, not a spike — alerting on it destroys trust."""
    c = WindowedCounters()
    for i in range(2):
        c.add(_ev(f"C{i}", action="HUMAN_REVIEW", excess=9000.0, billed=10000.0))
    assert c.spikes(min_claims=5) == []


def test_alert_when_a_provider_window_is_abnormal():
    c = WindowedCounters()
    for i in range(8):
        c.add(_ev(f"C{i}", provider="H09", action="HUMAN_REVIEW",
                  billed=10000.0, excess=6000.0,
                  ts=datetime(2026, 7, 1, 10, i)))
    alerts = c.spikes(min_claims=5)
    assert alerts and alerts[0]["provider_id"] == "H09"
    assert "flagged" in alerts[0]["reason"]


def test_healthy_provider_raises_no_alert():
    c = WindowedCounters()
    for i in range(10):
        c.add(_ev(f"C{i}", provider="H10", action="AUTO_APPROVE",
                  billed=10000.0, excess=0.0, ts=datetime(2026, 7, 1, 10, i)))
    assert c.spikes(min_claims=5) == []


# ---------- reconciliation: one logic, two speeds ----------

def test_reconcile_reports_full_agreement():
    stream = [{"claim_id": "A", "estimated_excess_inr": 100.0,
               "recommended_action": "HUMAN_REVIEW"}]
    batch = [{"claim_id": "A", "estimated_excess_inr": 100.0,
              "recommended_action": "HUMAN_REVIEW"}]
    r = reconcile(stream, batch)
    assert r["agreement_pct"] == 100.0 and r["mismatches"] == []


def test_reconcile_catches_a_drifted_amount():
    stream = [{"claim_id": "A", "estimated_excess_inr": 100.0,
               "recommended_action": "HUMAN_REVIEW"}]
    batch = [{"claim_id": "A", "estimated_excess_inr": 250.0,
              "recommended_action": "HUMAN_REVIEW"}]
    r = reconcile(stream, batch)
    assert r["mismatches"] and r["agreement_pct"] < 100.0


def test_reconcile_catches_a_missing_claim():
    r = reconcile([{"claim_id": "GHOST", "estimated_excess_inr": 0.0,
                    "recommended_action": "AUTO_APPROVE"}], [])
    assert r["mismatches"][0]["reason"] == "missing in batch"
