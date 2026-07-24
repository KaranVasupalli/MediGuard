"""ML step: train the fraud model, evaluate it honestly, and explain one prediction.

Run AFTER rebuild_data.py, which produces realistic labelled claims.
"""
from deltalake import DeltaTable

from config.spark_config import load_config
from ml.train import train
from ml.explain import FraudExplainer


def main():
    print("=" * 62)
    print("TRAINING")
    print("=" * 62)
    _, metrics = train()

    print("\n" + "=" * 62)
    print("EXPLAINING A PREDICTION (what a reviewer would actually see)")
    print("=" * 62)

    cfg = load_config()
    ref = cfg["paths"]["reference"]
    feats = DeltaTable(f"{ref}/claim_features").to_pyarrow_table().to_pylist()
    labels = {l["claim_id"]: l for l in
              DeltaTable(f"{ref}/claim_labels").to_pyarrow_table().to_pylist()}

    ex = FraudExplainer()

    # a known-fraud claim and a known-honest one, for contrast
    fraud_row = next(f for f in feats if labels[f["claim_id"]]["is_fraud"] == 1)
    honest_row = next(f for f in feats
                      if labels[f["claim_id"]]["is_fraud"] == 0 and f["n_findings"] == 0)

    for row, tag in [(fraud_row, "KNOWN FRAUD"), (honest_row, "KNOWN HONEST")]:
        lab = labels[row["claim_id"]]
        out = ex.explain(row)
        print(f"\n{tag}: {row['claim_id']}"
              + (f"  (patterns: {lab['fraud_patterns']})" if lab["fraud_patterns"] else ""))
        print(f"  fraud score : {out['fraud_score']:.4f}")
        print(f"  summary     : {FraudExplainer.to_sentence(out)}")
        print("  top drivers :")
        for d in out["top_drivers"]:
            print(f"     {d['direction']:<9} {d['label']:<38} "
                  f"value={d['value']:<10g} impact={d['contribution']:+.3f}")

    print("\n" + "=" * 62)
    print(f"verdict vs rules-only baseline: {metrics['verdict_vs_baseline']}")
    print("=" * 62)


if __name__ == "__main__":
    main()
