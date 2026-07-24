"""The ingestion gate: raw hospital bills -> clean canonical claim-lines.

Five jobs, in order, per row:
  1. map    - rename the hospital's columns to canonical fields
  2. deidentify - drop names, salt-hash the patient id (PII never stored raw)
  3. derive  - compute los_days, claim_year, claim_month from dates
  4. translate - foreign codes -> Indian (SNOMED->ICD-10, CPT->HBP + rate lookup)
  5. validate - check against the ClaimLine contract; bad rows are quarantined

Returns clean rows, quarantined rows (with reasons), and a quality report.
The corpus writer only ever receives rows that passed here.
"""
import hashlib
import os
from datetime import datetime

import yaml

from data_quality.schema_validator import ClaimLine
from data_quality import code_translation as ct

# Salt for hashing patient ids. In production this MUST come from a secret env var;
# a dev default keeps the demo runnable but is clearly not for real data.
_SALT = os.environ.get("PATIENT_SALT", "dev-only-salt-change-me")


def _hash_patient(raw_id: str) -> str:
    return "h" + hashlib.sha256((_SALT + str(raw_id)).encode()).hexdigest()[:12]


def load_mapping(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _process_row(raw: dict, mapping: dict) -> tuple[dict | None, dict | None]:
    """Return (clean_row, None) if valid, or (None, quarantine_row) if not."""
    cols = mapping["columns"]
    fmt = mapping.get("date_format", "%Y-%m-%d")
    src = mapping["source_system"]

    # 1. map + 2. deidentify
    c: dict = {}
    raw_patient = None
    for raw_col, canon in cols.items():
        if canon == "_pii_drop":
            continue                       # name etc. dropped at the door
        if canon == "_patient_raw_id":
            raw_patient = raw.get(raw_col)
            continue
        c[canon] = raw.get(raw_col)
    c["patient_hash"] = _hash_patient(raw_patient)
    c["source_system"] = src
    c["ingest_ts"] = datetime.now()

    flags: list[str] = []

    def quarantine(reason: str):
        return None, {"claim_id": c.get("claim_id"), "line_no": c.get("line_no"),
                      "source_system": src, "reason": reason}

    # 3. derive
    try:
        adm = datetime.strptime(str(c["admission_date"]), fmt).date()
        dis = datetime.strptime(str(c["discharge_date"]), fmt).date()
        svc = datetime.strptime(str(c["service_date"]), fmt).date()
    except (ValueError, TypeError, KeyError):
        return quarantine("bad_or_missing_date")
    c["admission_date"], c["discharge_date"], c["service_date"] = adm, dis, svc
    c["los_days"] = (dis - adm).days
    c["claim_year"], c["claim_month"] = adm.year, adm.month

    # 4. translate foreign -> Indian (keep originals as provenance)
    icd10, icd10_desc, ok_dx = ct.snomed_to_icd10(c.get("snomed_src"))
    if ok_dx:
        c["icd10_primary"], c["icd10_desc"] = icd10, icd10_desc
    else:
        flags.append("icd10_unmapped")
        c["icd10_primary"], c["icd10_desc"] = "UNKNOWN", None

    hbp, hbp_desc, ok_pr = ct.cpt_to_hbp(c.get("cpt_code"))
    if ok_pr:
        c["hbp_code"], c["hbp_desc"] = hbp, hbp_desc
        c["hbp_package_rate_inr"] = ct.hbp_rate(hbp)
    else:
        flags.append("hbp_unmapped")
        c["hbp_code"], c["hbp_desc"], c["hbp_package_rate_inr"] = None, None, None

    # numeric coercion
    try:
        c["quantity"] = float(c["quantity"])
        c["unit_price_inr"] = float(c["unit_price_inr"])
        c["billed_inr"] = float(c["billed_inr"])
        c["line_no"] = int(c["line_no"])
    except (ValueError, TypeError, KeyError):
        return quarantine("bad_numeric_value")

    c["quality_flags"] = ",".join(flags)

    # 5. validate against the frozen contract
    try:
        valid = ClaimLine(**c)
    except Exception as e:
        first = str(e).splitlines()[-1][:80]
        return quarantine(f"schema_fail: {first}")
    return valid.model_dump(), None


def run_gate(raw_rows: list[dict], mapping_path: str) -> dict:
    """Process a batch. Returns clean, quarantined, and a quality report."""
    mapping = load_mapping(mapping_path)
    clean, quarantined = [], []
    for raw in raw_rows:
        c, q = _process_row(raw, mapping)
        (clean if c else quarantined).append(c or q)

    reasons: dict[str, int] = {}
    for q in quarantined:
        reasons[q["reason"].split(":")[0]] = reasons.get(q["reason"].split(":")[0], 0) + 1
    flagged = sum(1 for c in clean if c["quality_flags"])

    report = {
        "run_id": datetime.now().strftime("run-%Y%m%d-%H%M%S"),
        "total_in": len(raw_rows),
        "clean_out": len(clean),
        "quarantined": len(quarantined),
        "clean_with_flags": flagged,
        "quarantine_reasons": reasons,
        "run_ts": datetime.now(),
    }
    return {"clean": clean, "quarantined": quarantined, "report": report}
