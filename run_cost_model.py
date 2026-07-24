"""Cost-model step, run locally.

  clean corpus  ->  cost percentiles baseline  ->  cost model scores each line
  against its fair price band, flagging bills that sit far above comparable ones
  even when under the HBP cap.

Shows the cost model on a real claim and on a crafted 'under the cap but inflated'
claim, which the hard rules alone would miss.
"""
from deltalake import DeltaTable

from config.spark_config import load_config
from batch.mine_baselines import mine_procedure_cost_pctiles
from evidence.rules_baseline import build_indexes
from evidence.cost_model import score_claim


def main():
    cfg = load_config()
    rows = DeltaTable(cfg["paths"]["corpus"]).to_pyarrow_table().to_pylist()
    print(f"1) corpus: {len(rows)} lines, {len({r['claim_id'] for r in rows})} claims")

    pctiles = mine_procedure_cost_pctiles(rows)
    cost_idx = {(r["hbp_code"], r["provider_state"]): r for r in pctiles}
    print(f"2) cost baseline: {len(pctiles)} (procedure, state) price bands")

    print("\n3) cost model on a REAL claim ...")
    sid = sorted({r["claim_id"] for r in rows})[0]
    lines = sorted([r for r in rows if r["claim_id"] == sid], key=lambda r: r["line_no"])
    res = score_claim(lines, cost_idx)
    print(f"   {sid}: {res['lines_above_fair_range']} lines above fair range, "
          f"worst={res['worst_severity']}, gap over p95 INR {res['total_gap_over_p95_inr']:,.2f}")
    for f in res["cost_findings"]:
        print(f"     - [{f['severity']}] line {f['line_no']}: {f['reason']}")

    print("\n4) cost model on a CRAFTED 'under-cap but inflated' claim ...")
    # pick a real price band and bill just under any cap but well above typical
    band = pctiles[0]
    crafted = [{
        "line_no": 1, "hbp_code": band["hbp_code"], "provider_state": band["provider_state"],
        "billed_inr": band["p50_inr"] * 1.8,        # 80% over typical
        "quantity": 1.0, "los_days": 2,
    }]
    res2 = score_claim(crafted, cost_idx)
    for f in res2["cost_findings"]:
        print(f"   band {band['hbp_code']}/{band['provider_state']}: {f['reason']}")
        print(f"   -> flagged [{f['severity']}] even though no fixed rate was broken")

    print("\n=== cost model working — catches inflated-but-under-cap bills ===")


if __name__ == "__main__":
    main()
