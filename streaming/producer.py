"""Send claims into the stream.

Three sources, chosen by `streaming.source` in config.yaml — the consumer code is
identical for all three:

  kafka  - a real Redpanda/Kafka broker (local Redpanda, or Kafka in the cloud)
  file   - replay claims from the corpus with no broker at all
  rate   - synthetic claims at a fixed rate, for load testing

The `file` mode is what makes the streaming layer testable without Docker running,
and it is why the switch exists in the config rather than in the code.
"""
import json
from datetime import datetime, timedelta

from config.spark_config import load_config


class ClaimProducer:
    def __init__(self, cfg: dict | None = None):
        full = cfg or load_config()
        s = full.get("streaming", {})
        self.source = s.get("source", "file")
        self.topic = s.get("kafka", {}).get("topic", "claims")
        self.servers = s.get("kafka", {}).get("bootstrap_servers", "localhost:19092")
        self._producer = None

    def _connect(self):
        if self._producer is not None:
            return self._producer
        from kafka import KafkaProducer
        self._producer = KafkaProducer(
            bootstrap_servers=self.servers,
            value_serializer=lambda v: json.dumps(v, default=str).encode(),
            key_serializer=lambda k: str(k).encode(),
            retries=3, acks="all",
        )
        return self._producer

    def send(self, claim: dict):
        """Key by claim_id so all lines of a claim land on the same partition —
        otherwise a claim could be split across consumers and scored on partial data."""
        if self.source != "kafka":
            return False
        p = self._connect()
        p.send(self.topic, key=claim["claim_id"], value=claim)
        return True

    def flush(self):
        if self._producer is not None:
            self._producer.flush()

    def available(self) -> bool:
        """True if a broker is actually reachable."""
        if self.source != "kafka":
            return False
        try:
            self._connect()
            return True
        except Exception:
            return False


def claims_from_corpus(rows: list[dict], start: datetime | None = None,
                       spread_minutes: int = 240):
    """Group corpus lines into claim events with synthetic arrival times.

    Real claims arrive spread through the day; replaying them all with one timestamp
    would put everything in a single window and prove nothing about windowing.
    """
    by_claim: dict[str, list[dict]] = {}
    for r in rows:
        by_claim.setdefault(r["claim_id"], []).append(r)

    start = start or datetime.now().replace(second=0, microsecond=0)
    n = max(len(by_claim), 1)
    step = spread_minutes / n

    for i, (cid, lines) in enumerate(sorted(by_claim.items())):
        lines = sorted(lines, key=lambda x: x["line_no"])
        yield {
            "claim_id": cid,
            "provider_id": lines[0]["provider_id"],
            "patient_hash": lines[0]["patient_hash"],
            "event_ts": (start + timedelta(minutes=i * step)).isoformat(),
            "lines": lines,
        }
