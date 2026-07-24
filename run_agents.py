"""Agents step: produce a COMPLETE verdict combining every evidence layer.

  rules + cost + ML + anomaly + history + graph + discharge note
      -> deterministic score, money, action
      -> LLM writes the explanation
      -> numeric guard verifies the model invented nothing
      -> full Verdict object (the contract frozen in Step 1)

Works with Ollama running or not: without a model it uses the deterministic template,
clearly marked in the audit trail.
"""
import json
from pathlib import Path

from deltalake import DeltaTable

from config.spark_config import load_config
from batch.mine_baselines import mine_all
from batch.patient_history import build_patient_history
from batch.provider_graph import build_provider_graph, analyse, ring_risk_score
from evidence.rules_baseline import build_indexes, evaluate_claim
from evidence.cost_model import score_claim
from ml.features import build_features
from agents.llm_client import LLMClient
from agents.reader import read_discharge_note, find_unsupported_charges
from agents.reasoner import build_verdict

NOTE = """Patient admitted with acute exacerbation of COPD.
Required 2 days in the ICU with continuous oxygen support.
Daily physician review was carried out. CBC was performed on admission.
Length of stay 3 days. Discharged in stable condition on oral bronchodilators."""


def main():
    cfg = load_config()
    ref = cfg["paths"]["reference"]

    print("1) loading corpus + evidence layers ...")
    rows = DeltaTable(cfg["paths"]["corpus"]).to_pyarrow_table().to_pylist()
    b = mine_all(rows)
    idx = build_indexes(b["diag_procedure_norms"], b["procedure_cost_pctiles"])
    cost_idx = {(r["hbp_code"], r["provider_state"]): r
                for r in b["procedure_cost_pctiles"]}

    hist_by_claim = {h["claim_id"]: h for h in build_patient_history(rows)}

    G = build_provider_graph(rows, min_shared=2)
    stats = analyse(G)
    edges = sorted(s["shared_patient_edges"] for s in stats)
    med = edges[len(edges) // 2] if edges else 0
    for s in stats:
        s["ring_risk"] = ring_risk_score(s, med)
    risk_by_provider = {s["provider_id"]: s for s in stats}

    by_claim = {}
    for r in rows:
        by_claim.setdefault(r["claim_id"], []).append(r)
    print(f"   {len(by_claim)} claims, {len(stats)} providers")

    # optional ML score
    explainer = None
    if Path("./data/models/fraud_model.txt").exists():
        from ml.explain import FraudExplainer
        explainer = FraudExplainer()
        print("   ML model loaded")

    print("\n2) LLM availability ...")
    client = LLMClient()
    probe = client.generate("Reply with OK.", system="Reply with exactly: OK")
    src = client.last_source
    print(f"   provider={client.provider} model={client.describe()['llm_model']}")
    print("   status: " + ("LLM responded" if probe else
                           "no model reachable -> deterministic template will be used"))

    print("\n3) reading the discharge summary ...")
    extracted = read_discharge_note(NOTE, client)
    print(f"   method: {extracted['extraction_method']}")
    print(f"   procedures found: {extracted['procedures_mentioned']}")
    print(f"   ICU days documented: {extracted['icu_days_documented']}")

    # pick the highest-excess claim to demonstrate
    best_cid, best_excess = None, -1.0
    for cid, lines in by_claim.items():
        r = evaluate_claim(sorted(lines, key=lambda x: x["line_no"]), idx)
        if r["total_excess_inr"] > best_excess:
            best_cid, best_excess = cid, r["total_excess_inr"]

    lines = sorted(by_claim[best_cid], key=lambda x: x["line_no"])
    rules_res = evaluate_claim(lines, idx)
    cost_res = score_claim(lines, cost_idx)

    ml_score, shap_drivers = None, []
    if explainer:
        feats = build_features(lines, rules_res, cost_res)
        out = explainer.explain(feats, top_k=3)
        ml_score, shap_drivers = out["fraud_score"], out["top_drivers"]

    unsupported = find_unsupported_charges(lines, extracted)

    print(f"\n4) assembling verdict for {best_cid} ...")
    v = build_verdict(
        best_cid, lines, rules_res=rules_res, cost_res=cost_res,
        ml_score=ml_score, anomaly_score=None,
        history=hist_by_claim.get(best_cid, {}),
        provider_risk=risk_by_provider.get(lines[0]["provider_id"], {}),
        shap_drivers=shap_drivers, reader_out=extracted,
        unsupported=unsupported, client=client,
    )

    print("\n" + "=" * 66)
    print("FINAL VERDICT")
    print("=" * 66)
    print(f"  claim            {v.claim_id}")
    print(f"  verdict          {v.verdict}")
    print(f"  fraud score      {v.fraud_score}")
    print(f"  billed    INR    {v.billed_total_inr:,.2f}")
    print(f"  justified INR    {v.justified_total_inr:,.2f}")
    print(f"  excess    INR    {v.estimated_excess_inr:,.2f}")
    print(f"  action           {v.recommended_action}")
    print(f"\n  explanation:\n    {v.explanation}")
    print(f"\n  line adjudication ({len(v.line_adjudication)} lines):")
    for a in v.line_adjudication:
        print(f"    line {a.line}  {a.status:<9} billed {a.billed:>10,.0f}  "
              f"allowed {a.allowed:>10,.0f}   {a.reason[:52]}")
    if v.citations:
        print(f"\n  citations from the discharge summary ({len(v.citations)}):")
        for c in v.citations[:3]:
            print(f"    [{c.finding}] \"{c.span[:70]}...\"")
    print("\n  audit trail:")
    print(f"    provider={v.audit.llm_provider}  model={v.audit.llm_model}  "
          f"temperature={v.audit.temperature}")
    print(f"    explanation source : {v.audit.reference_delta_versions['explanation_source']}")
    print(f"    numeric guard passed: {v.audit.reference_delta_versions['numeric_guard_passed']}")
    print(f"    human review required: {v.audit.human_review_required}")

    out_dir = Path("./data/verdicts")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{v.claim_id}.json").write_text(v.model_dump_json(indent=2))
    print(f"\n  saved verdict to data/verdicts/{v.claim_id}.json")
    print("=" * 66)


if __name__ == "__main__":
    main()
