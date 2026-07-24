"""Tests for the ingestion gate. Run: pytest tests/test_ingestion.py -v"""
from data_quality.ingestion_gate import run_gate
from pathlib import Path

MAPPING = str(Path(__file__).parent.parent / "data_quality" / "mappings" / "source_hospital_a.yaml")


def _base_raw(**over):
    row = {
        "BILL_NO": "CLM-T-1", "LN": 1, "PT_NAME": "Test Patient", "PT_UID": "UID99999",
        "HOSP_ID": "H01", "HOSP_NAME": "Hospital H01", "ST": "MH",
        "ADMIT": "2026-07-10", "DISCH": "2026-07-13", "SVC_DT": "2026-07-11",
        "DX_SNOMED": "25374005", "PROC_CPT": "85025", "ITEM_DESC": "CBC test",
        "DEPT": "General Medicine", "QTY": 2, "RATE_INR": 180, "AMOUNT_INR": 360,
    }
    row.update(over)
    return row


def test_pii_is_stripped_and_patient_hashed():
    out = run_gate([_base_raw()], MAPPING)
    row = out["clean"][0]
    assert "PT_NAME" not in row and "provider_name" in row  # name gone, structure intact
    assert row["patient_hash"].startswith("h")
    assert "Test Patient" not in str(row)                   # name nowhere in output


def test_patient_hash_is_deterministic():
    a = run_gate([_base_raw()], MAPPING)["clean"][0]["patient_hash"]
    b = run_gate([_base_raw()], MAPPING)["clean"][0]["patient_hash"]
    assert a == b                                           # same id -> same hash


def test_foreign_codes_translated_to_indian():
    row = run_gate([_base_raw()], MAPPING)["clean"][0]
    assert row["icd10_primary"] == "A09"                   # SNOMED 25374005 -> A09
    assert row["hbp_code"] == "HBP-LAB-014"                # CPT 85025 -> HBP-LAB-014
    assert row["hbp_package_rate_inr"] == 180.0            # official rate attached
    assert row["snomed_src"] == "25374005"                 # provenance kept


def test_discharge_before_admission_quarantined():
    out = run_gate([_base_raw(ADMIT="2026-07-13", DISCH="2026-07-10")], MAPPING)
    assert out["clean"] == [] and out["quarantined"][0]["reason"].startswith("schema_fail")


def test_negative_quantity_quarantined():
    out = run_gate([_base_raw(QTY=-3)], MAPPING)
    assert out["clean"] == [] and len(out["quarantined"]) == 1


def test_bad_date_quarantined():
    out = run_gate([_base_raw(ADMIT="not-a-date")], MAPPING)
    assert out["quarantined"][0]["reason"] == "bad_or_missing_date"


def test_unmapped_code_flagged_not_rejected():
    out = run_gate([_base_raw(DX_SNOMED="00000000", PROC_CPT="00000")], MAPPING)
    assert len(out["clean"]) == 1                           # kept, not rejected
    assert "icd10_unmapped" in out["clean"][0]["quality_flags"]
    assert "hbp_unmapped" in out["clean"][0]["quality_flags"]


def test_report_counts_add_up():
    rows = [_base_raw(), _base_raw(QTY=-1), _base_raw(ADMIT="bad")]
    rep = run_gate(rows, MAPPING)["report"]
    assert rep["total_in"] == 3
    assert rep["clean_out"] + rep["quarantined"] == 3
