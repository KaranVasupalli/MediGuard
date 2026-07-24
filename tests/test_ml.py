"""Tests for the ML layer + realistic data. Run: pytest tests/test_ml.py -v

These test the things that actually go wrong in ML projects: feature-order drift
between training and scoring, label leakage, and a model that looks good only because
the data is unrealistic.
"""
from pathlib import Path

import pytest

from eval.generate_realistic import generate
from ml.features import FEATURE_NAMES

MODEL = Path("./data/models/fraud_model.txt")
pytestmark_model = pytest.mark.skipif(not MODEL.exists(),
                                      reason="model not trained yet (run python run_ml.py)")


# ---------- data realism ----------

def test_fraud_is_rare_not_the_majority():
    _, labels, _t = generate(n_claims=500, seed=1)
    rate = sum(l["is_fraud"] for l in labels) / len(labels)
    assert 0.02 < rate < 0.20, f"fraud rate {rate:.2%} is not realistic"


def test_labels_are_separate_from_the_bill():
    raw, labels, _t = generate(n_claims=50, seed=2)
    bill_fields = set(raw[0])
    assert "is_fraud" not in bill_fields          # label must never ride on the claim
    assert "fraud_patterns" not in bill_fields
    assert {"claim_id", "is_fraud"} <= set(labels[0])


def test_every_claim_has_exactly_one_label():
    raw, labels, _t = generate(n_claims=100, seed=3)
    claim_ids = {r["BILL_NO"] for r in raw}
    label_ids = [l["claim_id"] for l in labels]
    assert set(label_ids) == claim_ids
    assert len(label_ids) == len(set(label_ids))


def test_honest_claims_are_billed_near_the_official_rate():
    raw, labels, _t = generate(n_claims=400, seed=4)
    honest = {l["claim_id"] for l in labels if not l["is_fraud"]}
    ratios = [r["AMOUNT_INR"] / (r["RATE_INR"] * r["QTY"])
              for r in raw if r["BILL_NO"] in honest and r["RATE_INR"] and r["QTY"]]
    over = sum(1 for x in ratios if x > 1.10) / len(ratios)
    assert over < 0.10, "too many honest lines far above rate — data unrealistic"


def test_fraud_patterns_are_recorded():
    _, labels, _t = generate(n_claims=300, seed=5)
    frauds = [l for l in labels if l["is_fraud"]]
    assert frauds and all(l["fraud_patterns"] for l in frauds)


# ---------- model + explainer ----------

@pytestmark_model
def test_model_feature_order_matches_contract():
    import json
    saved = json.loads(Path("./data/models/feature_names.json").read_text())
    assert saved["features"] == FEATURE_NAMES     # training order == scoring order


@pytestmark_model
def test_score_is_a_probability():
    from ml.explain import FraudExplainer
    ex = FraudExplainer()
    feats = {n: 0.0 for n in FEATURE_NAMES}
    s = ex.score(feats)
    assert 0.0 <= s <= 1.0


@pytestmark_model
def test_suspicious_claim_scores_higher_than_clean_claim():
    from ml.explain import FraudExplainer
    ex = FraudExplainer()
    clean = {n: 0.0 for n in FEATURE_NAMES}
    clean.update({"n_lines": 3, "los_days": 3, "total_billed_inr": 6000,
                  "avg_line_inr": 2000, "max_line_inr": 2500, "distinct_procedures": 3})
    bad = dict(clean)
    bad.update({"r1_overbilling_count": 3, "r3_stay_logic_count": 1, "n_findings": 5,
                "n_high_severity": 2, "total_excess_inr": 40000, "excess_ratio": 0.6,
                "share_lines_flagged": 1.0, "worst_cost_severity": 3})
    assert ex.score(bad) > ex.score(clean)


@pytestmark_model
def test_explanation_returns_readable_drivers():
    from ml.explain import FraudExplainer
    ex = FraudExplainer()
    feats = {n: 0.0 for n in FEATURE_NAMES}
    feats.update({"n_findings": 4, "total_excess_inr": 20000, "excess_ratio": 0.5})
    out = ex.explain(feats, top_k=3)
    assert 0.0 <= out["fraud_score"] <= 1.0
    assert len(out["top_drivers"]) == 3
    for d in out["top_drivers"]:
        assert d["label"] != d["feature"] or d["feature"] in FEATURE_NAMES
        assert d["direction"] in {"increases", "decreases"}
    assert isinstance(FraudExplainer.to_sentence(out), str)


@pytestmark_model
def test_metrics_report_pr_auc_not_just_accuracy():
    import json
    m = json.loads(Path("./data/models/metrics.json").read_text())
    assert "pr_auc" in m["model"] and "recall" in m["model"]
    assert "rules_only_baseline" in m          # honest comparison must be recorded
