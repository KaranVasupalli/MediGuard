"""Walking skeleton: runs the whole pipe end to end on fake data.

  fake claims  ->  stub batch (REAL Delta write)  ->  DuckDB read  ->  stub verdict

Proves integration. Replaces one stub with real code each following step.
Run from repo root:  python orchestrate.py
"""
from config.spark_config import get_spark, load_config
from eval.generate_fake_claims import make_claims
from batch.build_corpus import build
from serving.store import read_corpus, read_claim
from agents.reasoner import make_verdict


def main():
    cfg = load_config()
    corpus_path = cfg["paths"]["corpus"]

    print("1) generating fake claims ...")
    rows = make_claims(n_claims=20)
    print(f"   {len(rows)} claim-lines, "
          f"{len({r['claim_id'] for r in rows})} claims")

    print("2) stub batch -> REAL Delta write ...")
    spark = get_spark("skeleton")
    n = build(spark, rows, corpus_path)
    spark.stop()
    print(f"   wrote {n} rows to Delta at {corpus_path}")

    print("3) DuckDB reads the Delta table back ...")
    all_rows = read_corpus(corpus_path)
    print(f"   DuckDB read {len(all_rows)} rows via delta_scan")

    print("4) stub verdict for one claim ...")
    sample_id = sorted({r["claim_id"] for r in all_rows})[0]
    lines = read_claim(corpus_path, sample_id)
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
