"""MediGuard reviewer dashboard.

Run:  streamlit run app/dashboard.py

WHO WRITES WHAT (the single-writer rule from Step 1 still holds):
  score_all.py  writes  verdicts     - the scored claims
  this app      writes  decisions    - the human's accept/reject

They are separate tables joined on claim_id, so re-scoring never destroys a reviewer's
work and a reviewer never overwrites a score.
"""
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import pyarrow as pa
import streamlit as st
from deltalake import DeltaTable, write_deltalake

import config.storage as stg
from app.review_logic import (
    load_decisions as _load_decisions, save_decision as _save_decision,
    build_queue, queue_summary,
)

st.set_page_config(page_title="MediGuard AI — Claim Review", layout="wide")

SO = stg.deltalake_storage_options() or None
VERDICTS = stg.table_path("verdicts")
DECISIONS = stg.table_path("decisions")


# ---------------------------------------------------------------- data access
@st.cache_data(show_spinner=False)
def load_verdicts() -> pd.DataFrame:
    try:
        rows = DeltaTable(VERDICTS,
                          storage_options=SO).to_pyarrow_table().to_pylist()
    except Exception:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def load_decisions() -> dict:
    return _load_decisions(DECISIONS, SO)


def save_decision(claim_id: str, decision: str, reviewer: str, note: str):
    _save_decision(DECISIONS, claim_id, decision, reviewer, note,
                   storage_options=SO)


def money(x) -> str:
    try:
        return f"₹{float(x):,.0f}"
    except (TypeError, ValueError):
        return "—"


# ---------------------------------------------------------------- sidebar
st.sidebar.title("MediGuard AI")
st.sidebar.caption("Explainable claim fraud review")
st.sidebar.caption(f"Storage: {stg.describe()}")

df = load_verdicts()
if df.empty:
    st.error("No verdicts found. Run `python score_all.py` first.")
    st.stop()

decisions = load_decisions()
reviewer = st.sidebar.text_input("Reviewer name", value="reviewer1")

st.sidebar.subheader("Filters")
show = st.sidebar.radio("Queue", ["Needs review", "All claims", "Already decided"], index=0)
min_score = st.sidebar.slider("Minimum fraud score", 0.0, 1.0, 0.0, 0.05)
min_excess = st.sidebar.number_input("Minimum excess (₹)", value=0, step=1000)
sort_by = st.sidebar.selectbox("Sort by", ["Excess (₹)", "Fraud score", "Findings"])

df["decided"] = df["claim_id"].map(lambda c: c in decisions)
view = df.copy()
if show == "Needs review":
    view = view[(view["recommended_action"] != "AUTO_APPROVE") & (~view["decided"])]
elif show == "Already decided":
    view = view[view["decided"]]

view = view[(view["fraud_score"] >= min_score) &
            (view["estimated_excess_inr"] >= min_excess)]
sort_col = {"Excess (₹)": "estimated_excess_inr", "Fraud score": "fraud_score",
            "Findings": "n_findings"}[sort_by]
view = view.sort_values(sort_col, ascending=False)


# ---------------------------------------------------------------- header
st.title("Claim review queue")

flagged = df[df["recommended_action"] != "AUTO_APPROVE"]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total claims", f"{len(df):,}")
c2.metric("Flagged for review", f"{len(flagged):,}",
          f"{len(flagged)/len(df)*100:.1f}% of all")
c3.metric("Excess at stake", money(flagged["estimated_excess_inr"].sum()))
c4.metric("Decided", f"{len(decisions):,}")

st.caption(f"Showing {len(view):,} claims in this queue.")

if view.empty:
    st.success("Nothing matches these filters. Queue is clear.")
    st.stop()


# ---------------------------------------------------------------- queue table
cols = ["claim_id", "provider_id", "fraud_score", "billed_total_inr",
        "estimated_excess_inr", "n_findings", "recommended_action"]
table = view[cols].head(200).rename(columns={
    "claim_id": "Claim", "provider_id": "Provider", "fraud_score": "Score",
    "billed_total_inr": "Billed ₹", "estimated_excess_inr": "Excess ₹",
    "n_findings": "Findings", "recommended_action": "Action"})
st.dataframe(table, use_container_width=True, hide_index=True, height=280)


# ---------------------------------------------------------------- detail
st.divider()
choice = st.selectbox("Open a claim", view["claim_id"].tolist())
row = view[view["claim_id"] == choice].iloc[0]
verdict = json.loads(row["verdict_json"])

left, right = st.columns([2, 1])

with left:
    st.subheader(f"{choice}")
    a, b, c = st.columns(3)
    a.metric("Billed", money(row["billed_total_inr"]))
    b.metric("Justified", money(row["justified_total_inr"]))
    c.metric("Excess", money(row["estimated_excess_inr"]))

    st.markdown("**Explanation**")
    st.info(verdict.get("explanation", "—"))

    if st.button("Generate detailed explanation with the local model"):
        with st.spinner("Asking the local model…"):
            try:
                from agents.llm_client import LLMClient
                from agents.reasoner import SYSTEM
                from agents import numeric_guard
                facts = {
                    "claim_id": choice,
                    "billed_total_inr": float(row["billed_total_inr"]),
                    "allowed_total_inr": float(row["justified_total_inr"]),
                    "estimated_excess_inr": float(row["estimated_excess_inr"]),
                    "fraud_score": float(row["fraud_score"]),
                    "findings": verdict["evidence"].get("rules_baseline", []),
                }
                client = LLMClient()
                text = client.generate(
                    f"Write the reviewer explanation using ONLY these figures:\n"
                    f"{facts}\n\nExplanation:", system=SYSTEM)
                if not text:
                    st.warning("No model reachable — start Ollama to use this.")
                else:
                    ok, bad = numeric_guard.check(text, facts)
                    if ok:
                        st.success(text)
                        st.caption(f"Source: {client.last_source} · numeric guard passed")
                    else:
                        st.error("Model output rejected: it contained figures not in "
                                 f"the evidence ({bad}). The verified explanation above "
                                 "stands.")
            except Exception as e:      # never let the dashboard die on the LLM
                st.warning(f"Explanation unavailable: {type(e).__name__}")

    st.markdown("**Line-by-line adjudication**")
    adj = pd.DataFrame(verdict.get("line_adjudication", []))
    if not adj.empty:
        adj = adj.rename(columns={"line": "Line", "billed": "Billed ₹",
                                  "allowed": "Allowed ₹", "status": "Status",
                                  "reason": "Reason"})
        st.dataframe(adj, use_container_width=True, hide_index=True)

    findings = verdict.get("evidence", {}).get("rules_baseline", [])
    if findings:
        st.markdown(f"**Rule findings ({len(findings)})**")
        for f in findings:
            sev = f.get("severity", "low")
            icon = {"high": "🔴", "medium": "🟠", "low": "🟡"}.get(sev, "⚪")
            st.write(f"{icon} `{f.get('rule')}` line {f.get('line_no')} — "
                     f"{f.get('reason', '')}")

    cites = verdict.get("citations", [])
    if cites:
        st.markdown("**Citations from the discharge summary**")
        for c_ in cites:
            st.caption(f"[{c_['finding']}] “{c_['span'][:200]}”")

with right:
    st.subheader("Risk signals")
    st.metric("Fraud score", f"{row['fraud_score']:.2f}")
    st.progress(min(float(row["fraud_score"]), 1.0))
    st.write(f"**Recommended:** `{row['recommended_action']}`")

    ev = verdict.get("evidence", {})
    ml = ev.get("supervised", {}).get("ml_score")
    if ml is not None:
        st.write(f"**ML score:** {ml:.3f}")
    an = ev.get("anomaly", {}).get("score")
    if an is not None:
        st.write(f"**Anomaly score:** {an:.3f}")
    if row.get("ring_risk"):
        st.write(f"**Provider ring risk:** {float(row['ring_risk']):.2f}")
    if row.get("history_flags"):
        st.write(f"**Patient history:** {row['history_flags']}")

    drivers = ev.get("shap_top_features", [])
    if drivers:
        st.markdown("**Top model drivers**")
        for d in drivers:
            st.caption(f"{d.get('label', d.get('feature'))} = {d.get('value')} "
                       f"({d.get('contribution'):+.3f})")

    st.divider()
    st.subheader("Decision")
    prior = decisions.get(choice)
    if prior:
        st.success(f"Already **{prior['adjudicator_decision']}** by "
                   f"{prior['decided_by']} on {prior['decided_ts'][:10]}")
        if prior.get("note"):
            st.caption(f"Note: {prior['note']}")

    note = st.text_area("Reviewer note", value="", height=80)
    d1, d2, d3 = st.columns(3)
    if d1.button("✅ Accept", use_container_width=True):
        save_decision(choice, "accepted", reviewer, note)
        st.rerun()
    if d2.button("❌ Reject", use_container_width=True):
        save_decision(choice, "rejected", reviewer, note)
        st.rerun()
    if d3.button("⬆️ Escalate", use_container_width=True):
        save_decision(choice, "escalated", reviewer, note)
        st.rerun()

    st.divider()
    audit = verdict.get("audit", {})
    st.caption("**Audit trail**")
    st.caption(f"model: {audit.get('llm_model')} · temp {audit.get('temperature')}")
    refs = audit.get("reference_delta_versions", {})
    st.caption(f"explanation source: {refs.get('explanation_source', '—')}")
    st.caption(f"numeric guard passed: {refs.get('numeric_guard_passed', '—')}")
    st.caption(f"prompt hash: {audit.get('prompt_hash', '—')}")