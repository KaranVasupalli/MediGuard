"""Streaming step: score claims as they arrive, maintain live counters, raise alerts.

Runs against a real Redpanda broker if one is reachable; otherwise replays the corpus
through exactly the same code path so the layer is demonstrable without Docker.

Ends with the test that matters for a two-speed architecture: do the streaming and
batch paths produce the SAME verdict for the same claim?
"""
import shutil
from pathlib import Path

import pyarrow as pa
from deltalake import write_deltalake, DeltaTable

from config.spark_config import load_config
from score_all import build_context, _NoLLM
from streaming.producer import ClaimProducer, claims_from_corpus
from streaming.stream_job import run_stream, kafka_events, reconcile
from streaming.windows import WindowedCounters
from evidence.rules_baseline import evaluate_claim
from evidence.cost_model import score_claim
from agents.reasoner import build_verdict


def _write(path: str, rows: list[dict]):
    shutil.rmtree(path, ignore_errors=True)
    if not rows:
        return
    t = pa.Table.from_pylist(rows)
    for i, f in enumerate(t.schema):
        if pa.types.is_null(f.type):
            t = t.set_column(i, f.name, t.column(i).cast(pa.string()))
    write_deltalake(path, t, mode="overwrite")


def main(n_claims: int = 400):
    cfg = load_config()
    ref = cfg["paths"]["reference"]

    print("1) loading corpus + evidence context ...")
    rows = DeltaTable(cfg["paths"]["corpus"]).to_pyarrow_table().to_pylist()
    ctx = build_context(rows)

    # keep the demo quick: take the first N claims
    wanted = sorted({r["claim_id"] for r in rows})[:n_claims]
    subset = [r for r in rows if r["claim_id"] in set(wanted)]
    print(f"   streaming {len(wanted)} claims ({len(subset)} lines)")

    print("\n2) checking for a broker ...")
    producer = ClaimProducer(cfg)
    live = producer.available()
    print(f"   source={producer.source}  broker reachable: {live}")

    events = list(claims_from_corpus(subset))

    if live:
        print(f"   publishing {len(events)} claims to topic '{producer.topic}' ...")
        for ev in events:
            producer.send(ev)
        producer.flush()
        print("   consuming from the broker ...")
        stream_input = kafka_events(cfg)
    else:
        print("   no broker — replaying the same events through the same code path")
        stream_input = events

    print("\n3) running the stream ...")
    alerts_seen = []
    counters = WindowedCounters(window_minutes=60, watermark_minutes=120)
    result = run_stream(stream_input, ctx, counters=counters, client=_NoLLM(),
                        on_alert=alerts_seen.append)

    scored = result["scored"]
    stats = result["stats"]
    print(f"   scored {len(scored)} claims")
    print(f"   windows open: {stats['windows']}  late events: {stats['late_events']}")

    snap = counters.snapshot()
    _write(f"{ref}/streaming_counters", [
        {**s, "window_start": s["window_start"].isoformat(),
         "window_end": s["window_end"].isoformat()} for s in snap])
    print(f"   wrote {len(snap)} (provider, window) counter rows")

    print("\n4) live alerts raised during the run:")
    if result["alerts"]:
        for a in result["alerts"][:6]:
            print(f"   [{a['provider_id']}] {a['window_start'].strftime('%H:%M')}  "
                  f"{a['n_claims']} claims  excess INR {a['excess_inr']:,.0f}")
            print(f"       {a['reason']}")
    else:
        print("   none — no provider window crossed the alert thresholds")

    # ---- the property that matters ----
    print("\n" + "=" * 64)
    print("RECONCILIATION: does the LIVE path agree with the BATCH path?")
    print("=" * 64)
    batch = []
    by_claim: dict[str, list[dict]] = {}
    for r in subset:
        by_claim.setdefault(r["claim_id"], []).append(r)
    no_llm = _NoLLM()
    for cid, lines in by_claim.items():
        lines = sorted(lines, key=lambda x: x["line_no"])
        rr = evaluate_claim(lines, ctx["idx"])
        cr = score_claim(lines, ctx["cost_idx"])
        v = build_verdict(cid, lines, rules_res=rr, cost_res=cr,
                          history=ctx["history"].get(cid, {}),
                          provider_risk=ctx["provider_risk"].get(lines[0]["provider_id"], {}),
                          client=no_llm)
        batch.append({"claim_id": cid,
                      "estimated_excess_inr": v.estimated_excess_inr,
                      "recommended_action": v.recommended_action})

    rec = reconcile(scored, batch)
    print(f"  claims compared : {rec['compared']}")
    print(f"  agreement       : {rec['agreement_pct']}%")
    if rec["mismatches"]:
        print(f"  MISMATCHES ({len(rec['mismatches'])}):")
        for m in rec["mismatches"][:5]:
            print(f"    {m}")
        print("\n  -> the two layers have DRIFTED; this must be fixed.")
    else:
        print("\n  -> identical verdicts from both paths. One logic, two speeds.")

    print("\n=== streaming layer working ===")


if __name__ == "__main__":
    main()
