"""Tests for feature engineering. Run: pytest tests/test_features.py -v"""
import pytest

from ml.features import (
    build_features, to_vector, build_feature_row, FEATURE_NAMES, FEATURE_VERSION,
)


def _lines(n=2, billed=1000.0, los=3):
    return [{"line_no": i + 1, "billed_inr": billed, "hbp_code": f"HBP-{i}",
             "los_days": los, "quality_flags": ""} for i in range(n)]


def _rules(excess=0.0, findings=None):
    return {"findings": findings or [], "total_excess_inr": excess,
            "rule_hits": sorted({f["rule"] for f in (findings or [])})}


def _cost(gap=0.0, n_above=0, worst="none", pct=0.0):
    return {"cost_findings": [{"pct_over_median": pct}] if pct else [],
            "total_gap_over_p95_inr": gap, "worst_severity": worst,
            "lines_above_fair_range": n_above}


def test_feature_set_matches_frozen_contract():
    f = build_features(_lines(), _rules(), _cost())
    assert set(f) == set(FEATURE_NAMES)
    assert len(FEATURE_NAMES) == len(set(FEATURE_NAMES))   # no duplicate names


def test_vector_order_is_stable():
    f = build_features(_lines(), _rules(), _cost())
    v = to_vector(f)
    assert len(v) == len(FEATURE_NAMES)
    assert v == [float(f[n]) for n in FEATURE_NAMES]       # order follows the contract


def test_clean_claim_has_all_risk_features_zero():
    f = build_features(_lines(), _rules(), _cost())
    for name in ["n_findings", "n_high_severity", "total_excess_inr", "excess_ratio",
                 "share_lines_flagged", "cost_lines_above_band", "worst_cost_severity"]:
        assert f[name] == 0.0, f"{name} should be 0 on a clean claim"


def test_rule_counts_are_split_by_rule():
    findings = [
        {"rule": "R1_overbilling", "line_no": 1, "severity": "high"},
        {"rule": "R1_overbilling", "line_no": 2, "severity": "medium"},
        {"rule": "R3_stay_logic", "line_no": 1, "severity": "high"},
    ]
    f = build_features(_lines(), _rules(5000.0, findings), _cost())
    assert f["r1_overbilling_count"] == 2.0
    assert f["r3_stay_logic_count"] == 1.0
    assert f["r2_dx_mismatch_count"] == 0.0
    assert f["n_findings"] == 3.0
    assert f["n_high_severity"] == 2.0


def test_excess_ratio_is_normalised_not_raw():
    """Same rupee excess must score worse on a small bill than a large one."""
    small = build_features(_lines(2, billed=3000.0), _rules(5000.0), _cost())
    large = build_features(_lines(2, billed=250000.0), _rules(5000.0), _cost())
    assert small["total_excess_inr"] == large["total_excess_inr"]   # same raw amount
    assert small["excess_ratio"] > large["excess_ratio"]            # but ratio differs


def test_share_lines_flagged_counts_distinct_lines():
    findings = [{"rule": "R1_overbilling", "line_no": 1, "severity": "low"},
                {"rule": "R4_cost_outlier", "line_no": 1, "severity": "low"}]
    f = build_features(_lines(4), _rules(10.0, findings), _cost())
    assert f["share_lines_flagged"] == 0.25      # 1 distinct line of 4, not 2 of 4


def test_zero_billed_does_not_divide_by_zero():
    f = build_features(_lines(1, billed=0.0), _rules(0.0), _cost())
    assert f["excess_ratio"] == 0.0 and f["cost_gap_ratio"] == 0.0


def test_severity_maps_to_ordinal():
    assert build_features(_lines(), _rules(), _cost(worst="high"))["worst_cost_severity"] == 3.0
    assert build_features(_lines(), _rules(), _cost(worst="low"))["worst_cost_severity"] == 1.0


def test_unmapped_codes_are_counted():
    lines = _lines(3)
    lines[0]["quality_flags"] = "icd10_unmapped,hbp_unmapped"
    f = build_features(lines, _rules(), _cost())
    assert f["n_unmapped_codes"] == 1.0        # one LINE carried unmapped flags


def test_feature_row_carries_id_and_version():
    row = build_feature_row("CLM-X", _lines(), _rules(), _cost())
    assert row["claim_id"] == "CLM-X"
    assert row["feature_version"] == FEATURE_VERSION
    assert all(n in row for n in FEATURE_NAMES)
