"""Score every claim and write the verdicts table the dashboard reads.

DESIGN NOTE — why no LLM here:
Running a language model over every claim would take minutes per thousand and produce
text nobody reads, because a reviewer only ever opens the handful that get flagged. So
batch scoring uses the deterministic template, and the LLM explanation is generated
ON DEMAND when a reviewer actually opens a claim in the dashboard. Same verdict, same
numbers; only the prose is deferred.

Everything else (score, money, action, evidence) is computed here in full.
"""
from pathlib import Path

import pyarrow as pa
from deltalake import write_deltalake, DeltaTable

import config.storage as stg
from config.spark_config import load_config
from batch.mine_baselines import mine_all
from batch.patient_history import build_patient_history
from batch.provider_graph import build_provider_graph, analyse, ring_risk_score
from evidence.rules_baseline import build_indexes, evaluate_claim
from evidence.cost_model import score_claim
from ml.features import build_features
from ml.anomaly import AnomalyDetector, build_profile
from agents.reasoner import build_verdict
from agents.llm_client import LLMClient


class _NoLLM(LLMClient):
    """Forces the deterministic template — used for bulk scoring."""

    def __init__(self):
        self.cfg = load_config().get("llm", {})
        self.provider = self.cfg.get("provider", "ollama")
        self.last_source = "offline"
        self.cache_dir = Path(self.cfg.get("cache_dir", "./data/_llm_cache"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def generate(self, prompt, system="", allow_offline=True):
        self.last_source = "offline"
        return ""


def _write(path: str, rows: list[dict]):
    """No rmtree — Delta's own overwrite works on any backend."""
    if not rows:
        return
    t = pa.Table.from_pylist(rows)
    for i, f in enumerate(t.schema):
        if pa.types.is_null(f.type):
            t = t.set_column(i, f.name, t.column(i).cast(pa.string()))
    write_deltalake(path, t, mode="overwrite",
                    storage_options=stg.deltalake_storage_options() or None)


def build_context(rows: list[dict]) -> dict:
    """All the shared evidence layers, computed once for the whole population."""
    b = mine_all(rows)
    idx = build_indexes(b["diag_procedure_norms"], b["procedure_cost_pctiles"])
    cost_idx = {(r["hbp_code"], r["provider_state"]): r
                for r in b["procedure_cost_pctiles"]}

    hist = {h["claim_id"]: h for h in build_patient_history(rows)}

    G = build_provider_graph(rows, min_shared=2)
    stats = analyse(G)
    if stats:
        edges = sorted(s["shared_patient_edges"] for s in stats)
        med = edges[len(edges) // 2]
        for s in stats:
            s["ring_risk"] = ring_risk_score(s, med)
    risk = {s["provider_id"]: s for s in stats}

    return {"idx": idx, "cost_idx": cost_idx, "history": hist, "provider_risk": risk}


def score_all(rows: list[dict], ctx: dict, explainer=None, detector=None) -> list[dict]:
    """One verdict row per claim, ready for the verdicts table."""
    by_claim: dict[str, list[dict]] = {}
    for r in rows:
        by_claim.setdefault(r["claim_id"], []).append(r)

    no_llm = _NoLLM()
    out = []
    for cid, lines in by_claim.items():
        lines = sorted(lines, key=lambda x: x["line_no"])
        rules_res = evaluate_claim(lines, ctx["idx"])
        cost_res = score_claim(lines, ctx["cost_idx"])

        ml_score, drivers = None, []
        if explainer:
            feats = build_features(lines, rules_res, cost_res)
            ex = explainer.explain(feats, top_k=3)
            ml_score, drivers = ex["fraud_score"], ex["top_drivers"]

        anom = None
        if detector:
            anom = float(detector.score([build_profile(lines, cost_res)])[0])

        v = build_verdict(
            cid, lines, rules_res=rules_res, cost_res=cost_res,
            ml_score=ml_score, anomaly_score=anom,
            history=ctx["history"].get(cid, {}),
            provider_risk=ctx["provider_risk"].get(lines[0]["provider_id"], {}),
            shap_drivers=drivers, client=no_llm,
        )

        out.append({
            "claim_id": v.claim_id,
            "provider_id": lines[0]["provider_id"],
            "patient_hash": lines[0]["patient_hash"],
            "verdict": v.verdict,
            "fraud_score": v.fraud_score,
            "billed_total_inr": v.billed_total_inr,
            "justified_total_inr": v.justified_total_inr,
            "estimated_excess_inr": v.estimated_excess_inr,
            "recommended_action": v.recommended_action,
            "n_findings": len(rules_res.get("findings", [])),
            "history_flags": ctx["history"].get(cid, {}).get("history_flags", ""),
            "ring_risk": ctx["provider_risk"].get(lines[0]["provider_id"], {}).get("ring_risk", 0.0),
            "adjudicator_decision": "",            # filled by the reviewer
            "decided_by": "",
            "verdict_json": v.model_dump_json(),
        })
    return out


def main():
    cfg = load_config()
    so = stg.deltalake_storage_options() or None
    corpus_path = cfg["paths"]["corpus"] if stg.backend() == "local" \
        else stg.table_path("corpus")

    print("1) loading corpus ...")
    rows = DeltaTable(corpus_path,
                      storage_options=so).to_pyarrow_table().to_pylist()
    print(f"   {len(rows)} lines, {len({r['claim_id'] for r in rows})} claims")

    print("2) building shared evidence layers ...")
    ctx = build_context(rows)

    explainer = None
    if Path("./data/models/fraud_model.txt").exists():
        from ml.explain import FraudExplainer
        explainer = FraudExplainer()
        print("   ML model loaded")

    print("3) fitting anomaly detector ...")
    by_claim: dict[str, list[dict]] = {}
    for r in rows:
        by_claim.setdefault(r["claim_id"], []).append(r)
    profiles = [build_profile(sorted(l, key=lambda x: x["line_no"]),
                              score_claim(l, ctx["cost_idx"]))
                for l in by_claim.values()]
    detector = AnomalyDetector(contamination=0.05).fit(profiles)

    print("4) scoring every claim ...")
    verdicts = score_all(rows, ctx, explainer, detector)
    _write(stg.table_path("verdicts"), verdicts)

    flagged = [v for v in verdicts if v["recommended_action"] != "AUTO_APPROVE"]
    money = sum(v["estimated_excess_inr"] for v in flagged)
    print(f"\n   {len(verdicts)} verdicts written")
    print(f"   flagged for review : {len(flagged)} ({len(flagged)/len(verdicts)*100:.1f}%)")
    print(f"   excess at stake    : INR {money:,.0f}")
    print(f"   auto-approved      : {len(verdicts)-len(flagged)}")
    print("\n=== verdicts table ready — run the dashboard ===")


if __name__ == "__main__":
    main()