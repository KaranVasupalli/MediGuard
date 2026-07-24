"""Foreign -> Indian code translation.

Loads the crosswalk tables (SNOMED->ICD-10, CPT->HBP) and the HBP rate table.
The sample tables here are illustrative; the real full crosswalks from WHO (ICD-10)
and NHA (HBP 2.2) drop into the same CSVs with no code change.

We translate CODES only. We never convert currency: incoming bills are already in
INR, and HBP rates are official Indian rupee package rates.
"""
import csv
from pathlib import Path

_MAP_DIR = Path(__file__).parent / "mappings"


def _load_csv(name: str, key: str) -> dict:
    out: dict[str, dict] = {}
    with open(_MAP_DIR / name, newline="") as f:
        for row in csv.DictReader(f):
            out[row[key].strip()] = row
    return out


# loaded once at import
_SNOMED = _load_csv("snomed_to_icd10.csv", "snomed_src")
_CPT = _load_csv("cpt_to_hbp.csv", "cpt_code")
_HBP_RATES = _load_csv("hbp_rates.csv", "hbp_code")


def snomed_to_icd10(snomed: str | None):
    """Return (icd10_primary, icd10_desc, ok). ok=False means unmapped."""
    if snomed and snomed in _SNOMED:
        r = _SNOMED[snomed]
        return r["icd10_primary"], r["icd10_desc"], True
    return None, None, False


def cpt_to_hbp(cpt: str | None):
    """Return (hbp_code, hbp_desc, ok). ok=False means unmapped."""
    if cpt and cpt in _CPT:
        r = _CPT[cpt]
        return r["hbp_code"], r["hbp_desc"], True
    return None, None, False


def hbp_rate(hbp_code: str | None):
    """Return official INR package rate for an HBP code, or None if unknown."""
    if hbp_code and hbp_code in _HBP_RATES:
        return float(_HBP_RATES[hbp_code]["package_rate_inr"])
    return None


def hbp_rates_table() -> list[dict]:
    """The HBP rate reference rows (written to the hbp_rates Delta table)."""
    return [
        {"hbp_code": k, "hbp_desc": v["hbp_desc"],
         "package_rate_inr": float(v["package_rate_inr"])}
        for k, v in _HBP_RATES.items()
    ]
