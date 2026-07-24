"""Rebuild the whole data layer from the REALISTIC generator.

  realistic raw bills (+ separate labels)
    -> ingestion gate -> clean corpus (Delta)
    -> mine baselines -> reference tables (Delta)
    -> rules + cost model per claim -> features (Delta)
    -> labels joined by claim_id -> training table (Delta)

Labels live in their own table. They are NEVER part of the 27-column claim schema and
never become a feature — in reality they come from audit outcomes, not from the bill.

Paths come from config.storage, so the same script writes to local folders, Azurite,
or real ADLS Gen2 depending only on storage.backend in config.yaml.
"""
from pathlib import Path

import pyarrow as pa
from deltalake import write_deltalake, DeltaTable

import config.storage as stg
from config.spark_config import load_config
from eval.generate_realistic import generate
from data_quality.ingestion_gate import run_gate
from batch.mine_baselines import mine_all
from evidence.rules_baseline import build_indexes
from run_features import build_all_features

MAPPING = str(Path(__file__).parent / "data_quality" / "mappings" / "source_hospital_a.yaml")


def write_delta(path: str, rows: list[dict], partition_by=None):
    """Overwrite a Delta table. No rmtree — Delta's own overwrite works on any
    backend and keeps version history, which a folder delete does not."""
    if not rows:
        return
    t = pa.Table.from_pylist(rows)
    for i, f in enumerate(t.schema):
        if pa.types.is_null(f.type):
            t = t.set_column(i, f.name, t.column(i).cast(pa.string()))
    write_deltalake(path, t, mode="overwrite",
                    partition_by=partition_by,
                    storage_options=stg.deltalake_storage_options() or None)


def main(n_claims: int = 2000):
    cfg = load_config()
    corpus_path = cfg["paths"]["corpus"] if stg.backend() == "local" \
        else stg.table_path("corpus")
    print(f"   storage backend: {stg.describe()}")

    print(f"1) generating {n_claims} realistic claims ...")
    raw, labels, truth = generate(n_claims=n_claims)
    n_fraud = sum(l["is_fraud"] for l in labels)
    print(f"   {len(raw)} raw lines | {n_fraud} fraudulent ({n_fraud/len(labels)*100:.1f}%)")

    print("2) ingestion gate ...")
    res = run_gate(raw, MAPPING)
    print(f"   clean:{res['report']['clean_out']}  quarantined:{res['report']['quarantined']}")
    write_delta(corpus_path, res["clean"], partition_by=["claim_year", "claim_month"])
    write_delta(stg.table_path("claim_labels"), labels)
    write_delta(stg.table_path("ring_truth"),
                [{"provider_id": p, "in_ring": 1} for p in truth["ring_providers"]] +
                [{"provider_id": p, "in_ring": 0} for p in truth["honest_providers"]])
    print(f"   injected ring providers: {truth['ring_providers']}")

    print("3) mining baselines ...")
    rows = res["clean"]
    b = mine_all(rows)
    write_delta(stg.table_path("diag_procedure_norms"), b["diag_procedure_norms"])
    write_delta(stg.table_path("procedure_cost_pctiles"), b["procedure_cost_pctiles"])
    idx = build_indexes(b["diag_procedure_norms"], b["procedure_cost_pctiles"])
    cost_idx = {(r["hbp_code"], r["provider_state"]): r for r in b["procedure_cost_pctiles"]}
    print(f"   {len(b['diag_procedure_norms'])} norms, "
          f"{len(b['procedure_cost_pctiles'])} cost bands")

    print("4) building features ...")
    feats = build_all_features(rows, idx, cost_idx)
    write_delta(stg.table_path("claim_features"), feats)

    lab_by_id = {l["claim_id"]: l for l in labels}
    flagged = sum(1 for f in feats if f["n_findings"] > 0)
    fraud_flagged = sum(1 for f in feats
                        if f["n_findings"] > 0 and lab_by_id[f["claim_id"]]["is_fraud"])
    honest_flagged = flagged - fraud_flagged
    print(f"   {len(feats)} feature rows")
    print(f"   claims with >=1 finding: {flagged}/{len(feats)} "
          f"({flagged/len(feats)*100:.1f}%)")
    print(f"     of those, actually fraud: {fraud_flagged}  |  "
          f"false alarms: {honest_flagged}")
    print("\n=== data layer rebuilt on realistic, labelled claims ===")


if __name__ == "__main__":
    main()