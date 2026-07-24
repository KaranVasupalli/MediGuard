"""Tumbling-window aggregation for the live stream.

Batch answers "what happened last month". Streaming answers "what is happening right
now" — a hospital that suddenly starts billing three times its usual ICU rate should
surface within minutes, not at the next monthly run.

Two things that are easy to get wrong and are handled explicitly here:

  LATE EVENTS. Claims do not arrive in order. A claim stamped 10:59 can turn up at
  11:02, after its window closed. Dropping it silently corrupts the counts, so a
  WATERMARK defines how long a window stays open for stragglers. Anything later than
  the watermark is counted as late and reported, never silently discarded.

  IDEMPOTENCE. A stream can replay the same message after a restart. Counters keyed by
  (provider, window) and rebuilt from the events they contain give the same answer no
  matter how many times a message arrives — that is what `merge_idem` in the writer
  registry means.

This module is pure Python with no Kafka in it, so the logic can be tested directly.
"""
from collections import defaultdict
from datetime import datetime, timedelta


def floor_to_window(ts: datetime, window_minutes: int) -> datetime:
    """Start of the tumbling window containing ts."""
    total = ts.hour * 60 + ts.minute
    start = (total // window_minutes) * window_minutes
    return ts.replace(hour=start // 60, minute=start % 60, second=0, microsecond=0)


class WindowedCounters:
    """Per-(provider, window) rolling statistics, rebuilt idempotently from events."""

    def __init__(self, window_minutes: int = 60, watermark_minutes: int = 120):
        self.window_minutes = window_minutes
        self.watermark = timedelta(minutes=watermark_minutes)
        # (provider, window_start) -> aggregate state
        self._state: dict[tuple[str, datetime], dict] = {}
        self._seen: set[str] = set()          # claim ids already counted
        self.max_event_ts: datetime | None = None
        self.late_events = 0

    # ---------------------------------------------------------------- ingest
    def add(self, event: dict) -> str:
        """Add one scored claim. Returns 'counted', 'duplicate' or 'late'."""
        cid = event["claim_id"]
        if cid in self._seen:
            return "duplicate"               # idempotent: replays change nothing

        ts = event["event_ts"]
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts)

        if self.max_event_ts is None or ts > self.max_event_ts:
            self.max_event_ts = ts

        # too late to be counted, but never silently dropped
        if self.max_event_ts - ts > self.watermark:
            self.late_events += 1
            return "late"

        key = (event["provider_id"], floor_to_window(ts, self.window_minutes))
        s = self._state.setdefault(key, {
            "n_claims": 0, "n_flagged": 0, "billed_inr": 0.0, "excess_inr": 0.0,
            "icu_claims": 0, "score_sum": 0.0,
        })
        s["n_claims"] += 1
        s["billed_inr"] += float(event.get("billed_total_inr", 0.0))
        s["excess_inr"] += float(event.get("estimated_excess_inr", 0.0))
        s["score_sum"] += float(event.get("fraud_score", 0.0))
        if event.get("recommended_action", "AUTO_APPROVE") != "AUTO_APPROVE":
            s["n_flagged"] += 1
        if event.get("has_icu"):
            s["icu_claims"] += 1

        self._seen.add(cid)
        return "counted"

    # ---------------------------------------------------------------- output
    def snapshot(self) -> list[dict]:
        """Current counters, one row per (provider, window)."""
        out = []
        for (provider, start), s in sorted(self._state.items(), key=lambda kv: kv[0][1]):
            n = s["n_claims"]
            out.append({
                "provider_id": provider,
                "window_start": start,
                "window_end": start + timedelta(minutes=self.window_minutes),
                "n_claims": n,
                "n_flagged": s["n_flagged"],
                "flag_rate": round(s["n_flagged"] / n, 4) if n else 0.0,
                "billed_inr": round(s["billed_inr"], 2),
                "excess_inr": round(s["excess_inr"], 2),
                "avg_fraud_score": round(s["score_sum"] / n, 4) if n else 0.0,
                "icu_rate": round(s["icu_claims"] / n, 4) if n else 0.0,
                "excess_ratio": round(s["excess_inr"] / s["billed_inr"], 4)
                if s["billed_inr"] else 0.0,
            })
        return out

    # ---------------------------------------------------------------- alerts
    def spikes(self, min_claims: int = 5, flag_rate_threshold: float = 0.5,
               excess_ratio_threshold: float = 0.25) -> list[dict]:
        """Providers whose CURRENT window looks abnormal.

        `min_claims` matters: with two claims a 50% flag rate is noise, not a spike.
        Alerting on tiny samples is how a live system loses its reader's trust.
        """
        alerts = []
        for row in self.snapshot():
            if row["n_claims"] < min_claims:
                continue
            reasons = []
            if row["flag_rate"] >= flag_rate_threshold:
                reasons.append(f"{row['flag_rate']*100:.0f}% of claims flagged")
            if row["excess_ratio"] >= excess_ratio_threshold:
                reasons.append(f"{row['excess_ratio']*100:.0f}% of billing is excess")
            if reasons:
                alerts.append({
                    "provider_id": row["provider_id"],
                    "window_start": row["window_start"],
                    "n_claims": row["n_claims"],
                    "flag_rate": row["flag_rate"],
                    "excess_inr": row["excess_inr"],
                    "reason": "; ".join(reasons),
                })
        return sorted(alerts, key=lambda a: -a["excess_inr"])

    def stats(self) -> dict:
        return {"windows": len(self._state), "claims_counted": len(self._seen),
                "late_events": self.late_events}
