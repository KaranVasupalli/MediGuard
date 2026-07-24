"""Tests for baseline mining + rules engine. Run: pytest tests/test_rules.py -v"""
from batch.mine_baselines import (
    mine_diag_procedure_norms, mine_procedure_cost_pctiles, _percentile,
)
from evidence.rules_baseline import build_indexes, evaluate_claim


def _corpus():
    """Small population: A09 goes with BED and ICU; E11 goes with LAB."""
    rows = []
    for i in range(30):
        rows += [
            {"claim_id": f"C{i}", "line_no": 1, "icd10_primary": "A09",
             "hbp_code": "HBP-BED-001", "provider_state": "MH", "billed_inr": 2000.0,
             "quantity": 1.0, "los_days": 2, "hbp_package_rate_inr": 2000.0},
            {"claim_id": f"C{i}", "line_no": 2, "icd10_primary": "E11",
             "hbp_code": "HBP-LAB-014", "provider_state": "MH", "billed_inr": 180.0,
             "quantity": 1.0, "los_days": 2, "hbp_package_rate_inr": 180.0},
        ]
    return rows


def _idx():
    rows = _corpus()
    return build_indexes(mine_diag_procedure_norms(rows),
                         mine_procedure_cost_pctiles(rows))


def test_percentile_basic():
    assert _percentile([10, 20, 30, 40], 50) == 25.0
    assert _percentile([5], 95) == 5.0


def test_norms_capture_seen_pairs():
    norms = mine_diag_procedure_norms(_corpus())
    pairs = {(n["icd10_primary"], n["hbp_code"]) for n in norms}
    assert ("A09", "HBP-BED-001") in pairs
    assert ("E11", "HBP-LAB-014") in pairs
    assert ("A09", "HBP-LAB-014") not in pairs      # this pair never co-occurs


def test_r1_overbilling_detected():
    line = {"line_no": 1, "hbp_code": "HBP-BED-001", "icd10_primary": "A09",
            "provider_state": "MH", "quantity": 2.0, "hbp_package_rate_inr": 2000.0,
            "billed_inr": 9000.0, "los_days": 2}          # allowed 4000, excess 5000
    res = evaluate_claim([line], _idx())
    assert "R1_overbilling" in res["rule_hits"]
    assert res["total_excess_inr"] == 5000.0


def test_r2_dx_procedure_mismatch_detected():
    line = {"line_no": 1, "hbp_code": "HBP-LAB-014", "icd10_primary": "A09",
            "provider_state": "MH", "quantity": 1.0, "hbp_package_rate_inr": 180.0,
            "billed_inr": 180.0, "los_days": 2}           # A09+LAB never seen
    res = evaluate_claim([line], _idx())
    assert "R2_dx_procedure_fit" in res["rule_hits"]


def test_r3_stay_logic_detected():
    line = {"line_no": 1, "hbp_code": "HBP-ICU-002", "icd10_primary": "A09",
            "provider_state": "MH", "quantity": 6.0, "hbp_package_rate_inr": 4500.0,
            "billed_inr": 27000.0, "los_days": 2}         # 6 ICU days, LOS 2
    res = evaluate_claim([line], _idx())
    assert "R3_stay_logic" in res["rule_hits"]


def test_clean_line_has_no_findings():
    line = {"line_no": 1, "hbp_code": "HBP-BED-001", "icd10_primary": "A09",
            "provider_state": "MH", "quantity": 1.0, "hbp_package_rate_inr": 2000.0,
            "billed_inr": 2000.0, "los_days": 2}          # exactly at rate, valid pair
    res = evaluate_claim([line], _idx())
    assert res["findings"] == [] and res["total_excess_inr"] == 0.0
