"""Train the fraud-scoring model.

The model does NOT detect fraud from scratch — it WEIGHS the deterministic signals the
rules engine and cost model already produced, and turns them into one ranked score.

Three things this script does that matter more than the training itself:
  1. Stratified train/test split, because fraud is rare and a random split could put
     almost no fraud in the test set.
  2. Reports PR-AUC, precision and recall — NOT accuracy. On 8% fraud, a model that
     predicts "never fraud" scores 92% accuracy and is worthless.
  3. Compares against a RULES-ONLY baseline. If the model cannot beat "flag anything
     the rules flagged", it is not earning its place, and the script says so.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from deltalake import DeltaTable
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, confusion_matrix,
)
import lightgbm as lgb

from config.spark_config import load_config
from ml.features import FEATURE_NAMES, FEATURE_VERSION

MODEL_DIR = Path("./data/models")


def load_training_data():
    cfg = load_config()
    ref = cfg["paths"]["reference"]
    feats = DeltaTable(f"{ref}/claim_features").to_pyarrow_table().to_pylist()
    labels = {l["claim_id"]: l["is_fraud"]
              for l in DeltaTable(f"{ref}/claim_labels").to_pyarrow_table().to_pylist()}

    X, y, ids = [], [], []
    for row in feats:
        if row["claim_id"] not in labels:
            continue
        X.append([float(row[n]) for n in FEATURE_NAMES])   # frozen feature order
        y.append(int(labels[row["claim_id"]]))
        ids.append(row["claim_id"])
    return np.array(X), np.array(y), ids


def rules_only_baseline(X: np.ndarray, y: np.ndarray) -> dict:
    """Baseline: flag a claim if the rules produced ANY finding."""
    i = FEATURE_NAMES.index("n_findings")
    pred = (X[:, i] > 0).astype(int)
    return {
        "precision": round(precision_score(y, pred, zero_division=0), 4),
        "recall": round(recall_score(y, pred, zero_division=0), 4),
        "f1": round(f1_score(y, pred, zero_division=0), 4),
        "flagged": int(pred.sum()),
    }


def train():
    X, y, ids = load_training_data()
    n_pos = int(y.sum())
    print(f"data: {len(y)} claims, {n_pos} fraud ({n_pos/len(y)*100:.1f}%), "
          f"{X.shape[1]} features")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y)   # stratify: rare class
    print(f"train {len(y_tr)} ({int(y_tr.sum())} fraud) | "
          f"test {len(y_te)} ({int(y_te.sum())} fraud)")

    # class imbalance: tell the model positives are rare
    pos_weight = (len(y_tr) - y_tr.sum()) / max(y_tr.sum(), 1)
    model = lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.05, num_leaves=15,
        min_child_samples=20, scale_pos_weight=pos_weight,
        random_state=42, verbose=-1,
    )
    model.fit(pd.DataFrame(X_tr, columns=FEATURE_NAMES), y_tr)

    prob = model.predict_proba(pd.DataFrame(X_te, columns=FEATURE_NAMES))[:, 1]
    pred = (prob >= 0.5).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_te, pred).ravel()
    metrics = {
        "model": {
            "precision": round(precision_score(y_te, pred, zero_division=0), 4),
            "recall": round(recall_score(y_te, pred, zero_division=0), 4),
            "f1": round(f1_score(y_te, pred, zero_division=0), 4),
            "roc_auc": round(roc_auc_score(y_te, prob), 4),
            "pr_auc": round(average_precision_score(y_te, prob), 4),
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        },
        "rules_only_baseline": rules_only_baseline(X_te, y_te),
        "feature_version": FEATURE_VERSION,
        "n_train": len(y_tr), "n_test": len(y_te),
    }

    print("\n--- MODEL (test set) ---")
    m = metrics["model"]
    print(f"  precision {m['precision']}   recall {m['recall']}   f1 {m['f1']}")
    print(f"  ROC-AUC   {m['roc_auc']}   PR-AUC {m['pr_auc']}  (PR-AUC is the honest one)")
    print(f"  caught {m['tp']} fraud, missed {m['fn']}, false alarms {m['fp']}")

    print("\n--- RULES-ONLY BASELINE (same test set) ---")
    b = metrics["rules_only_baseline"]
    print(f"  precision {b['precision']}   recall {b['recall']}   f1 {b['f1']}")

    lift = m["f1"] - b["f1"]
    verdict = ("model BEATS rules-only" if lift > 0.02 else
               "model ~ same as rules-only" if lift > -0.02 else
               "model WORSE than rules-only")
    print(f"\n  f1 difference: {lift:+.4f}  ->  {verdict}")
    metrics["verdict_vs_baseline"] = verdict

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(str(MODEL_DIR / "fraud_model.txt"))
    (MODEL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (MODEL_DIR / "feature_names.json").write_text(
        json.dumps({"feature_version": FEATURE_VERSION, "features": FEATURE_NAMES}, indent=2))
    print(f"\nsaved model + metrics to {MODEL_DIR}")
    return model, metrics


if __name__ == "__main__":
    train()
