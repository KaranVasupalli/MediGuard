"""SANDBOX-ONLY runner. Proves the walking skeleton end to end in an environment
with no Maven/DuckDB-extension network access.

It reuses the REAL parts unchanged:
  - eval.generate_fake_claims.make_claims   (real)
  - agents.reasoner.make_verdict            (real)
and substitutes only the two stations that need blocked downloads here:
  - Spark  Delta write   ->  delta-rs write_deltalake   (same Delta format)
  - DuckDB delta_scan read ->  delta-rs DeltaTable read (same Delta format)

On a normal machine, run orchestrate.py instead (Spark + DuckDB).
"""
import shutil
import pyarrow as pa
from deltalake import write_deltalake, DeltaTable

from config.spark_config import load_config
from eval.generate_fake_claims import make_claims
from agents.reasoner import make_verdict


def main():
    cfg = load_config()
    corpus_path = cfg["paths"]["corpus"]

    print("1) generating fake claims ...")
    rows = make_claims(n_claims=20)
    print(f"   {len(rows)} claim-lines, {len({r['claim_id'] for r in rows})} claims")

    print("2) write REAL Delta table (delta-rs stands in for Spark here) ...")
    shutil.rmtree(corpus_path, ignore_errors=True)
    table = pa.Table.from_pylist(rows)
    # Delta rejects pyarrow "null" columns (happens when a field is None in every
    # row, e.g. cpt_code here). Cast any such column to string.
    for i, field in enumerate(table.schema):
        if pa.types.is_null(field.type):
            table = table.set_column(i, field.name,
                                     table.column(i).cast(pa.string()))
    write_deltalake(corpus_path, table, mode="overwrite",
                    partition_by=["claim_year", "claim_month"])
    dt = DeltaTable(corpus_path)
    print(f"   Delta version {dt.version()}, {dt.to_pyarrow_table().num_rows} rows, "
          f"_delta_log present")

    print("3) read the Delta table back (delta-rs stands in for DuckDB here) ...")
    all_rows = DeltaTable(corpus_path).to_pyarrow_table().to_pylist()
    print(f"   read {len(all_rows)} rows from Delta")

    print("4) stub verdict for one claim ...")
    sample_id = sorted({r["claim_id"] for r in all_rows})[0]
    lines = sorted([r for r in all_rows if r["claim_id"] == sample_id],
                   key=lambda r: r["line_no"])
    verdict = make_verdict(sample_id, lines)

    print("\n=== WALKING SKELETON VERDICT ===")
    print(f"claim         {verdict.claim_id}")
    print(f"verdict       {verdict.verdict}")
    print(f"billed  INR   {verdict.billed_total_inr:,.2f}")
    print(f"justified INR {verdict.justified_total_inr:,.2f}")
    print(f"excess  INR   {verdict.estimated_excess_inr:,.2f}")
    print(f"action        {verdict.recommended_action}")
    print(f"lines         {len(verdict.line_adjudication)} adjudicated")
    print("=== END — pipe is connected end to end ===")


if __name__ == "__main__":
    main()
