"""Review-queue logic, kept separate from the Streamlit UI so it can be tested.

The dashboard file is all layout; the decisions about WHICH claims a reviewer sees,
in what order, and how their verdict is recorded live here where a test can reach
them. UI code that hides business logic is untestable, and this is the part that
actually matters.
"""
import shutil
from datetime import datetime

import pyarrow as pa
from deltalake import DeltaTable, write_deltalake

SORT_FIELDS = {
    "excess": "estimated_excess_inr",
    "score": "fraud_score",
    "findings": "n_findings",
}
VALID_DECISIONS = {"accepted", "rejected", "escalated"}


def load_table(path: str) -> list[dict]:
    try:
        return DeltaTable(path).to_pyarrow_table().to_pylist()
    except Exception:
        return []


def load_decisions(path: str) -> dict:
    return {r["claim_id"]: r for r in load_table(path)}


def save_decision(path: str, claim_id: str, decision: str,
                  reviewer: str = "", note: str = "") -> dict:
    """Record a reviewer's decision. Re-deciding a claim replaces the old record."""
    if decision not in VALID_DECISIONS:
        raise ValueError(f"invalid decision {decision!r}")

    existing = load_decisions(path)
    existing[claim_id] = {
        "claim_id": claim_id,
        "adjudicator_decision": decision,
        "decided_by": reviewer or "unknown",
        "decided_ts": datetime.now().isoformat(timespec="seconds"),
        "note": note or "",
    }
    shutil.rmtree(path, ignore_errors=True)
    write_deltalake(path, pa.Table.from_pylist(list(existing.values())),
                    mode="overwrite")
    return existing[claim_id]


def build_queue(verdicts: list[dict], decisions: dict, *, mode: str = "needs_review",
                min_score: float = 0.0, min_excess: float = 0.0,
                sort_by: str = "excess") -> list[dict]:
    """The reviewer's worklist.

    'needs_review' deliberately excludes auto-approved claims AND anything already
    decided — a queue that shows finished work is a queue nobody trusts.
    """
    out = []
    for v in verdicts:
        decided = v["claim_id"] in decisions
        if mode == "needs_review":
            if v.get("recommended_action") == "AUTO_APPROVE" or decided:
                continue
        elif mode == "decided" and not decided:
            continue
        if float(v.get("fraud_score", 0)) < min_score:
            continue
        if float(v.get("estimated_excess_inr", 0)) < min_excess:
            continue
        out.append(v)

    field = SORT_FIELDS.get(sort_by, "estimated_excess_inr")
    return sorted(out, key=lambda r: float(r.get(field, 0)), reverse=True)


def queue_summary(verdicts: list[dict], decisions: dict) -> dict:
    flagged = [v for v in verdicts if v.get("recommended_action") != "AUTO_APPROVE"]
    return {
        "total_claims": len(verdicts),
        "flagged": len(flagged),
        "flagged_pct": round(len(flagged) / len(verdicts) * 100, 1) if verdicts else 0.0,
        "excess_at_stake_inr": round(sum(float(v.get("estimated_excess_inr", 0))
                                         for v in flagged), 2),
        "decided": len(decisions),
        "outstanding": len([v for v in flagged if v["claim_id"] not in decisions]),
    }
