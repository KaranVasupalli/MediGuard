"""Rules-and-baseline step, run locally (delta-rs stands in for Spark).

  clean corpus  ->  mine baselines (norms + cost pctiles)  ->  Delta reference tables
                ->  rules engine checks each claim  ->  explainable findings + rupee excess

Runs the checks on a real corpus claim AND on a crafted suspicious claim so every
rule visibly fires. On EC2 the mining runs in Spark; the rule logic is identical.
"""
import shutil

import pyarrow as pa
from deltalake import write_deltalake, DeltaTable

from config.spark_config import load_config
from batch.mine_baselines import mine_all
from evidence.rules_baseline import build_indexes, evaluate_claim


def _write_delta(path: str, rows: list[dict]):
    shutil.rmtree(path, ignore_errors=True)
    if not rows:
        return
    table = pa.Table.from_pylist(rows)
    for i, f in enumerate(table.schema):
        if pa.types.is_null(f.type):
            table = table.set_column(i, f.name, table.column(i).cast(pa.string()))
    write_deltalake(path, table, mode="overwrite")


def _crafted_suspicious_claim():
    """One claim engineered so all four rules fire, for demonstration."""
    base = dict(claim_id="CLM-DEMO-SUS", patient_hash="hDEMO", provider_id="H01",
                provider_state="MH", los_days=2, icd10_primary="A09")
    return [
        # R1: billed 3x the allowed rate; R3: 6 ICU days but LOS is 2
        {**base, "line_no": 1, "hbp_code": "HBP-ICU-002", "quantity": 6.0,
         "hbp_package_rate_inr": 4500.0, "unit_price_inr": 4500.0, "billed_inr": 81000.0},
        # R2: ICU procedure under a diabetes diagnosis — a pair never seen together
        {**base, "line_no": 2, "hbp_code": "HBP-ICU-002", "quantity": 1.0,
         "hbp_package_rate_inr": 4500.0, "unit_price_inr": 4500.0, "billed_inr": 4500.0,
         "icd10_primary": "E11"},
    ]


def main():
    cfg = load_config()
    corpus_path = cfg["paths"]["corpus"]
    ref = cfg["paths"]["reference"]

    print("1) reading clean corpus ...")
    rows = DeltaTable(corpus_path).to_pyarrow_table().to_pylist()
    print(f"   {len(rows)} clean lines, {len({r['claim_id'] for r in rows})} claims")

    print("2) mining population baselines ...")
    baselines = mine_all(rows)
    norms, pctiles = baselines["diag_procedure_norms"], baselines["procedure_cost_pctiles"]
    print(f"   {len(norms)} (diagnosis,procedure) norms; {len(pctiles)} cost buckets")
    _write_delta(f"{ref}/diag_procedure_norms", norms)
    _write_delta(f"{ref}/procedure_cost_pctiles", pctiles)
    print(f"   written to {ref}/diag_procedure_norms and /procedure_cost_pctiles")

    idx = build_indexes(norms, pctiles)

    print("\n3) rules on a REAL corpus claim ...")
    sid = sorted({r["claim_id"] for r in rows})[0]
    real_lines = sorted([r for r in rows if r["claim_id"] == sid],
                        key=lambda r: r["line_no"])
    res = evaluate_claim(real_lines, idx)
    print(f"   {sid}: {len(res['findings'])} findings, "
          f"excess INR {res['total_excess_inr']:,.2f}, rules {res['rule_hits']}")
    for f in res["findings"][:4]:
        print(f"     - [{f['rule']}] {f['reason']}")

    print("\n4) rules on a CRAFTED suspicious claim (all rules should fire) ...")
    res2 = evaluate_claim(_crafted_suspicious_claim(), idx)
    print(f"   CLM-DEMO-SUS: {len(res2['findings'])} findings, "
          f"excess INR {res2['total_excess_inr']:,.2f}, rules {res2['rule_hits']}")
    for f in res2["findings"]:
        print(f"     - [{f['rule']}] {f['reason']}")

    print("\n=== rules engine working — deterministic, explainable, rupee-exact ===")


if __name__ == "__main__":
    main()
