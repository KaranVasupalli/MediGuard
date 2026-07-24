"""Tests for patient history. Run: pytest tests/test_history.py -v

The critical test is no-lookahead: a claim must be scored using only what came BEFORE
it. If history from later claims leaks backwards, the component looks brilliant in
testing and is useless in production, where the future has not happened yet.
"""
from datetime import date, timedelta

from batch.patient_history import build_patient_history, summarise


def _line(claim, patient, provider, adm, los, hbp, svc_offset=0, billed=4500.0, ln=1):
    a = adm
    return {"claim_id": claim, "line_no": ln, "patient_hash": patient,
            "provider_id": provider, "admission_date": a,
            "discharge_date": a + timedelta(days=los),
            "service_date": a + timedelta(days=svc_offset),
            "hbp_code": hbp, "billed_inr": billed, "quantity": 1.0, "los_days": los}


def test_single_claim_patient_has_no_history_flags():
    rows = [_line("C1", "p1", "H1", date(2026, 3, 1), 2, "HBP-ICU-002")]
    h = build_patient_history(rows)
    assert len(h) == 1
    assert h[0]["prior_claims"] == 0 and h[0]["history_flags"] == ""


def test_first_claim_is_never_flagged_no_lookahead():
    """A patient's first claim must be clean — the future must not leak backwards."""
    d = date(2026, 3, 1)
    rows = []
    for i in range(4):                       # four rapid repeat ICU admissions
        rows.append(_line(f"C{i}", "p1", "H1", d + timedelta(days=i * 2), 1,
                          "HBP-ICU-002"))
    h = sorted(build_patient_history(rows), key=lambda x: x["prior_claims"])
    assert h[0]["prior_claims"] == 0
    assert h[0]["history_flags"] == "", "first claim was flagged using future data"


def test_prior_claims_counts_increase_in_order():
    d = date(2026, 3, 1)
    rows = [_line(f"C{i}", "p1", "H1", d + timedelta(days=i * 10), 1, "HBP-BED-001")
            for i in range(3)]
    h = sorted(build_patient_history(rows), key=lambda x: x["prior_claims"])
    assert [x["prior_claims"] for x in h] == [0, 1, 2]


def test_rapid_readmission_detected():
    d = date(2026, 3, 1)
    rows = [_line("C1", "p1", "H1", d, 2, "HBP-BED-001"),
            _line("C2", "p1", "H1", d + timedelta(days=3), 2, "HBP-BED-001")]
    h = sorted(build_patient_history(rows), key=lambda x: x["prior_claims"])
    assert "rapid_readmission" in h[1]["history_flags"]


def test_normal_gap_is_not_flagged_as_readmission():
    d = date(2026, 3, 1)
    rows = [_line("C1", "p1", "H1", d, 2, "HBP-BED-001"),
            _line("C2", "p1", "H1", d + timedelta(days=90), 2, "HBP-BED-001")]
    h = sorted(build_patient_history(rows), key=lambda x: x["prior_claims"])
    assert "rapid_readmission" not in h[1]["history_flags"]


def test_bed_charges_do_not_trigger_repeat_flag():
    """A ward bed recurs on every admission — flagging it would flag everyone."""
    d = date(2026, 3, 1)
    rows = [_line(f"C{i}", "p1", "H1", d + timedelta(days=i * 5), 1, "HBP-BED-001")
            for i in range(4)]
    h = build_patient_history(rows)
    assert all("repeat_costly_procedure" not in x["history_flags"] for x in h)


def test_repeated_icu_is_flagged_after_a_pattern_forms():
    d = date(2026, 3, 1)
    rows = [_line(f"C{i}", "p1", "H1", d + timedelta(days=i * 4), 1, "HBP-ICU-002")
            for i in range(4)]
    h = sorted(build_patient_history(rows), key=lambda x: x["prior_claims"])
    assert any("repeat_costly_procedure" in x["history_flags"] for x in h)


def test_duplicate_service_date_detected():
    d = date(2026, 3, 1)
    rows = [_line("C1", "p1", "H1", d, 1, "HBP-ICU-002", svc_offset=0),
            _line("C2", "p1", "H2", d + timedelta(days=5), 1, "HBP-ICU-002",
                  svc_offset=-5)]                      # same service date re-billed
    h = sorted(build_patient_history(rows), key=lambda x: x["prior_claims"])
    assert "duplicate_service_date" in h[1]["history_flags"]


def test_provider_shuttling_detected():
    d = date(2026, 3, 1)
    rows = [_line("C1", "p1", "H1", d, 1, "HBP-BED-001"),
            _line("C2", "p1", "H2", d + timedelta(days=10), 1, "HBP-BED-001"),
            _line("C3", "p1", "H3", d + timedelta(days=20), 1, "HBP-BED-001")]
    h = sorted(build_patient_history(rows), key=lambda x: x["prior_claims"])
    assert "provider_shuttling" in h[2]["history_flags"]


def test_patients_do_not_contaminate_each_other():
    d = date(2026, 3, 1)
    rows = [_line("C1", "p1", "H1", d, 1, "HBP-ICU-002"),
            _line("C2", "p2", "H1", d + timedelta(days=1), 1, "HBP-ICU-002")]
    h = build_patient_history(rows)
    assert all(x["prior_claims"] == 0 for x in h)      # different patients
    assert all(x["history_flags"] == "" for x in h)


def test_summarise_counts_flags():
    d = date(2026, 3, 1)
    rows = [_line("C1", "p1", "H1", d, 2, "HBP-BED-001"),
            _line("C2", "p1", "H1", d + timedelta(days=2), 2, "HBP-BED-001")]
    s = summarise(build_patient_history(rows))
    assert s["total"] == 2 and s["flagged"] >= 1
    assert "rapid_readmission" in s["by_flag"]
