"""Ingestion-gate step, run locally (delta-rs stands in for Spark/DuckDB).

  raw hospital bills  ->  INGESTION GATE  ->  clean corpus (Delta)  +  quality report
                                          ->  quarantine pile (rejected rows)
Then a stub verdict on one clean claim, to show the pipe still connects.

On AWS/EC2 later the same gate output is written by Spark; the gate logic is identical.
"""
import shutil
from pathlib import Path

import pyarrow as pa
from deltalake import write_deltalake, DeltaTable

from config.spark_config import load_config
from eval.generate_raw_source import generate_raw
from data_quality.ingestion_gate import run_gate
from agents.reasoner import make_verdict

MAPPING = str(Path(__file__).parent / "data_quality" / "mappings" / "source_hospital_a.yaml")


def _write_delta(path: str, rows: list[dict], partition_by=None):
    shutil.rmtree(path, ignore_errors=True)
    table = pa.Table.from_pylist(rows)
    for i, field in enumerate(table.schema):        # Delta rejects all-null columns
        if pa.types.is_null(field.type):
            table = table.set_column(i, field.name, table.column(i).cast(pa.string()))
    write_deltalake(path, table, mode="overwrite", partition_by=partition_by)


def main():
    cfg = load_config()
    corpus_path = cfg["paths"]["corpus"]
    quality_path = cfg["paths"]["reference"] + "/quality_report"

    print("1) generating messy RAW hospital bills ...")
    raw = generate_raw(n_claims=30)
    print(f"   {len(raw)} raw lines in Hospital-A format (non-canonical)")

    print("2) running the INGESTION GATE ...")
    result = run_gate(raw, MAPPING)
    rep = result["report"]
    print(f"   in:{rep['total_in']}  clean:{rep['clean_out']}  "
          f"quarantined:{rep['quarantined']}  flagged:{rep['clean_with_flags']}")
    print(f"   quarantine reasons: {rep['quarantine_reasons']}")

    print("3) proof the gate did its job ...")
    sample = result["clean"][0]
    print(f"   PII stripped:   patient_hash={sample['patient_hash']}  "
          f"(no name stored: {'provider_name' in sample and 'PT_NAME' not in sample})")
    print(f"   code translated: SNOMED {sample['snomed_src']} -> ICD-10 "
          f"{sample['icd10_primary']} | CPT {sample['cpt_code']} -> HBP {sample['hbp_code']}")
    print(f"   HBP rate attached: INR {sample['hbp_package_rate_inr']}")

    print("4) writing clean corpus + quality report to Delta ...")
    _write_delta(corpus_path, result["clean"], partition_by=["claim_year", "claim_month"])
    _write_delta(quality_path, [{
        "run_id": rep["run_id"], "total_in": rep["total_in"],
        "clean_out": rep["clean_out"], "quarantined": rep["quarantined"],
        "clean_with_flags": rep["clean_with_flags"],
    }])
    n = DeltaTable(corpus_path).to_pyarrow_table().num_rows
    print(f"   corpus rows written: {n}")

    print("5) stub verdict on one clean claim ...")
    rows = DeltaTable(corpus_path).to_pyarrow_table().to_pylist()
    sid = sorted({r["claim_id"] for r in rows})[0]
    lines = sorted([r for r in rows if r["claim_id"] == sid], key=lambda r: r["line_no"])
    v = make_verdict(sid, lines)
    print(f"\n=== VERDICT === {v.claim_id}  {v.verdict}  "
          f"billed INR {v.billed_total_inr:,.2f}  excess INR {v.estimated_excess_inr:,.2f}")
    print("=== ingestion gate working — messy in, clean+standardised+private out ===")


if __name__ == "__main__":
    main()
