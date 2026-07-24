"""The streaming consumer: score claims as they arrive.

THE CRITICAL DISCIPLINE — one logic, two speeds.
This job does NOT contain its own copy of the fraud rules. It imports exactly the same
`evaluate_claim`, `score_claim` and `build_verdict` the batch layer uses. If the two
paths ever drifted apart, the same claim would get different verdicts depending on
whether it arrived live or was processed later, and nobody could trust either.

`reconcile()` proves they agree, and a test enforces it.
"""
import json
from datetime import datetime

from evidence.rules_baseline import evaluate_claim
from evidence.cost_model import score_claim
from agents.reasoner import build_verdict
from streaming.windows import WindowedCounters

ICU_CODES = {"HBP-ICU-002"}


def score_event(event: dict, ctx: dict, client=None) -> dict:
    """Score one arriving claim. Same code path as batch scoring."""
    lines = sorted(event["lines"], key=lambda x: x["line_no"])
    rules_res = evaluate_claim(lines, ctx["idx"])
    cost_res = score_claim(lines, ctx["cost_idx"])

    v = build_verdict(
        event["claim_id"], lines, rules_res=rules_res, cost_res=cost_res,
        history=ctx.get("history", {}).get(event["claim_id"], {}),
        provider_risk=ctx.get("provider_risk", {}).get(event["provider_id"], {}),
        client=client,
    )
    return {
        "claim_id": v.claim_id,
        "provider_id": event["provider_id"],
        "event_ts": event["event_ts"],
        "fraud_score": v.fraud_score,
        "billed_total_inr": v.billed_total_inr,
        "estimated_excess_inr": v.estimated_excess_inr,
        "recommended_action": v.recommended_action,
        "verdict": v.verdict,
        "has_icu": any(l.get("hbp_code") in ICU_CODES for l in lines),
        "n_findings": len(rules_res.get("findings", [])),
    }


def run_stream(events, ctx: dict, counters: WindowedCounters | None = None,
               client=None, on_alert=None) -> dict:
    """Consume an event iterable, score each, update windows, raise alerts.

    `events` is any iterable — a Kafka consumer, a corpus replay, or a test list.
    The scoring and windowing never know which, which is why the same job runs
    against Redpanda locally and Kafka in the cloud with only a config change.
    """
    counters = counters or WindowedCounters()
    scored, seen_alerts = [], set()

    for ev in events:
        out = score_event(ev, ctx, client=client)
        status = counters.add(out)
        out["window_status"] = status
        scored.append(out)

        if status == "counted":
            for a in counters.spikes():
                key = (a["provider_id"], a["window_start"])
                if key not in seen_alerts:
                    seen_alerts.add(key)
                    if on_alert:
                        on_alert(a)

    return {"scored": scored, "counters": counters,
            "alerts": counters.spikes(), "stats": counters.stats()}


def kafka_events(cfg: dict, timeout_ms: int = 5000):
    """Read claim events from a real Redpanda/Kafka topic."""
    from kafka import KafkaConsumer
    s = cfg["streaming"]["kafka"]
    consumer = KafkaConsumer(
        s.get("topic", "claims"),
        bootstrap_servers=s.get("bootstrap_servers", "localhost:19092"),
        value_deserializer=lambda v: json.loads(v.decode()),
        auto_offset_reset="earliest",
        enable_auto_commit=True,
        consumer_timeout_ms=timeout_ms,
        group_id="mediguard-stream",
    )
    for msg in consumer:
        yield msg.value


def reconcile(stream_results: list[dict], batch_results: list[dict]) -> dict:
    """Do the live path and the batch path agree, claim for claim?

    This is the property a two-speed architecture lives or dies on. Any mismatch means
    the layers have drifted and the system is producing two different truths.
    """
    b = {r["claim_id"]: r for r in batch_results}
    mismatches = []
    for s in stream_results:
        other = b.get(s["claim_id"])
        if other is None:
            mismatches.append({"claim_id": s["claim_id"], "reason": "missing in batch"})
            continue
        for field in ("estimated_excess_inr", "recommended_action"):
            if round(float(s.get(field, 0)), 2) != round(float(other.get(field, 0)), 2) \
                    if isinstance(s.get(field), (int, float)) else s.get(field) != other.get(field):
                mismatches.append({"claim_id": s["claim_id"], "field": field,
                                   "stream": s.get(field), "batch": other.get(field)})
    n = len(stream_results)
    return {"compared": n, "mismatches": mismatches,
            "agreement_pct": round((n - len(mismatches)) / n * 100, 2) if n else 100.0}
