"""Feature-engineering step, run locally.

  clean corpus + baselines  ->  rules engine + cost model per claim
                            ->  ONE numeric row per claim  ->  Delta feature table

This is the table the ML model will train on. Nothing here invents a number: every
feature is packed from evidence a deterministic computer already produced.
"""
import shutil

import pyarrow as pa
from deltalake import write_deltalake, DeltaTable

from config.spark_config import load_config
from batch.mine_baselines import mine_all
from evidence.rules_baseline import build_indexes, evaluate_claim
from evidence.cost_model import score_claim
from ml.features import build_feature_row, FEATURE_NAMES, FEATURE_VERSION


def _write_delta(path: str, rows: list[dict]):
    shutil.rmtree(path, ignore_errors=True)
    if not rows:
        return
    table = pa.Table.from_pylist(rows)
    for i, f in enumerate(table.schema):
        if pa.types.is_null(f.type):
            table = table.set_column(i, f.name, table.column(i).cast(pa.string()))
    write_deltalake(path, table, mode="overwrite")


def build_all_features(rows: list[dict], idx: dict, cost_idx: dict) -> list[dict]:
    by_claim: dict[str, list[dict]] = {}
    for r in rows:
        by_claim.setdefault(r["claim_id"], []).append(r)

    out = []
    for cid, lines in by_claim.items():
        lines = sorted(lines, key=lambda r: r["line_no"])
        rules_res = evaluate_claim(lines, idx)
        cost_res = score_claim(lines, cost_idx)
        out.append(build_feature_row(cid, lines, rules_res, cost_res))
    return out


def main():
    cfg = load_config()
    ref = cfg["paths"]["reference"]

    print("1) reading clean corpus ...")
    rows = DeltaTable(cfg["paths"]["corpus"]).to_pyarrow_table().to_pylist()
    print(f"   {len(rows)} lines, {len({r['claim_id'] for r in rows})} claims")

    print("2) mining baselines ...")
    b = mine_all(rows)
    idx = build_indexes(b["diag_procedure_norms"], b["procedure_cost_pctiles"])
    cost_idx = {(r["hbp_code"], r["provider_state"]): r
                for r in b["procedure_cost_pctiles"]}

    print("3) building features (one row per claim) ...")
    feats = build_all_features(rows, idx, cost_idx)
    print(f"   {len(feats)} feature rows x {len(FEATURE_NAMES)} features "
          f"(version {FEATURE_VERSION})")

    print("4) writing feature table to Delta ...")
    _write_delta(f"{ref}/claim_features", feats)
    n = DeltaTable(f"{ref}/claim_features").to_pyarrow_table().num_rows
    print(f"   {n} rows written to {ref}/claim_features")

    print("\n5) sample — the most suspicious claim by excess ratio:")
    worst = max(feats, key=lambda f: f["excess_ratio"])
    print(f"   claim {worst['claim_id']}")
    for k in ["total_billed_inr", "total_excess_inr", "excess_ratio",
              "n_findings", "n_high_severity", "share_lines_flagged",
              "cost_gap_ratio", "worst_cost_severity"]:
        print(f"     {k:<24} {worst[k]}")

    nonzero = sum(1 for f in feats if f["n_findings"] > 0)
    print(f"\n   claims with at least one finding: {nonzero}/{len(feats)}")
    print("=== features built — ready for the ML model ===")


if __name__ == "__main__":
    main()
