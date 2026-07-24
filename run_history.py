"""Patient-history step, with the same honest test as the other components:
can it catch fraud that per-claim checks structurally cannot see?
"""
import shutil
from pathlib import Path

import pyarrow as pa
from deltalake import write_deltalake, DeltaTable

from config.spark_config import load_config
from batch.patient_history import build_patient_history, summarise
from batch.mine_baselines import mine_all
from evidence.rules_baseline import build_indexes, evaluate_claim
from data_quality.ingestion_gate import run_gate
from eval.generate_realistic import generate_history_fraud

MAPPING = str(Path(__file__).parent / "data_quality" / "mappings" / "source_hospital_a.yaml")


def write_delta(path: str, rows: list[dict]):
    shutil.rmtree(path, ignore_errors=True)
    if not rows:
        return
    t = pa.Table.from_pylist(rows)
    for i, f in enumerate(t.schema):
        if pa.types.is_null(f.type):
            t = t.set_column(i, f.name, t.column(i).cast(pa.string()))
    write_deltalake(path, t, mode="overwrite")


def main():
    cfg = load_config()
    ref = cfg["paths"]["reference"]

    print("1) loading corpus ...")
    rows = DeltaTable(cfg["paths"]["corpus"]).to_pyarrow_table().to_pylist()
    print(f"   {len(rows)} lines, {len({r['patient_hash'] for r in rows})} patients")

    print("2) building patient history ...")
    hist = build_patient_history(rows)
    write_delta(f"{ref}/patient_history", hist)
    s = summarise(hist)
    print(f"   {s['total']} (patient, claim) rows | flagged {s['flagged']} "
          f"({s['flagged']/max(s['total'],1)*100:.1f}%)")
    for k, v in sorted(s["by_flag"].items(), key=lambda kv: -kv[1]):
        print(f"     {k:<28} {v}")

    top = sorted(hist, key=lambda h: -h["history_risk"])[:3]
    print("\n   highest-risk histories:")
    for h in top:
        print(f"     {h['claim_id']:<22} risk={h['history_risk']:.2f}  "
              f"prior={h['prior_claims']}  flags={h['history_flags'] or '-'}")

    # ---- held-out test: fraud only visible across visits ----
    print("\n" + "=" * 62)
    print("HELD-OUT TEST: CROSS-VISIT FRAUD (each claim individually clean)")
    print("=" * 62)
    raw, _ = generate_history_fraud(n_patients=30)
    clean = run_gate(raw, MAPPING)["clean"]

    b = mine_all(rows)
    idx = build_indexes(b["diag_procedure_norms"], b["procedure_cost_pctiles"])
    by_claim: dict[str, list[dict]] = {}
    for r in clean:
        by_claim.setdefault(r["claim_id"], []).append(r)

    rule_hits = sum(1 for lines in by_claim.values()
                    if evaluate_claim(sorted(lines, key=lambda x: x["line_no"]), idx)["findings"])

    hx = build_patient_history(clean)
    hist_hits = sum(1 for h in hx if h["history_flags"])
    n = len(by_claim)

    print(f"  {n} claims across {len({r['patient_hash'] for r in clean})} patients")
    print(f"  caught by PER-CLAIM RULES : {rule_hits}/{n} ({rule_hits/n*100:.0f}%)")
    print(f"  caught by PATIENT HISTORY : {hist_hits}/{n} ({hist_hits/n*100:.0f}%)")

    if hist_hits > rule_hits:
        print("\n  -> patient history EARNS ITS PLACE: the sequence is the evidence, "
              "not any single claim.")
    else:
        print("\n  -> patient history did not beat the per-claim rules here.")

    print("\n=== patient history evaluated honestly ===")


if __name__ == "__main__":
    main()
