"""
THE single schema. Imported by batch/build_corpus.py and streaming/stream_job.py.
If these two ever parse a claim differently, reconciliation fails and looks like
a Delta bug. One module, imported twice. Do not fork it.
"""
from datetime import date, datetime
from typing import Optional
from pydantic import BaseModel, Field, field_validator
import pandera as pa
from pandera.typing import Series

# 27 fields (Appendix 24.1). Kept in full per project decision.
CANONICAL_COLUMNS = [
    "claim_id", "line_no", "patient_hash", "provider_id", "provider_name",
    "provider_state", "admission_date", "discharge_date", "service_date", "los_days",
    "icd10_primary", "icd10_desc", "snomed_src", "cpt_code", "hbp_code", "hbp_desc",
    "line_desc", "department", "quantity", "unit_price_inr", "billed_inr",
    "hbp_package_rate_inr", "source_system", "ingest_ts", "quality_flags",
    "claim_year", "claim_month",
]


class ClaimLine(BaseModel):
    """Row-level contract for a single canonical claim-line (boundary validation)."""
    claim_id: str
    line_no: int = Field(ge=1)
    patient_hash: str                       # salted hash — never a raw identifier
    provider_id: str
    provider_name: Optional[str] = None
    provider_state: str
    admission_date: date
    discharge_date: date
    service_date: date
    los_days: int = Field(ge=0)
    icd10_primary: str
    icd10_desc: Optional[str] = None
    snomed_src: Optional[str] = None        # provenance retained for audit
    cpt_code: Optional[str] = None
    hbp_code: Optional[str] = None
    hbp_desc: Optional[str] = None
    line_desc: str
    department: Optional[str] = None
    quantity: float = Field(ge=0)
    unit_price_inr: float = Field(ge=0)
    billed_inr: float = Field(ge=0)
    hbp_package_rate_inr: Optional[float] = Field(default=None, ge=0)
    source_system: str
    ingest_ts: datetime
    quality_flags: str = ""                 # comma-joined flags, "" = clean
    claim_year: int
    claim_month: int = Field(ge=1, le=12)

    @field_validator("discharge_date")
    @classmethod
    def discharge_after_admission(cls, v, info):
        adm = info.data.get("admission_date")
        if adm and v < adm:
            raise ValueError("discharge_date before admission_date")
        return v


class ClaimLineFrame(pa.DataFrameModel):
    """DataFrame-level contract for pandera validation at the ingestion gate."""
    claim_id: Series[str]
    line_no: Series[int] = pa.Field(ge=1)
    patient_hash: Series[str]
    provider_id: Series[str]
    provider_state: Series[str]
    los_days: Series[int] = pa.Field(ge=0)
    icd10_primary: Series[str]
    hbp_code: Series[str] = pa.Field(nullable=True)
    quantity: Series[float] = pa.Field(ge=0)
    unit_price_inr: Series[float] = pa.Field(ge=0)
    billed_inr: Series[float] = pa.Field(ge=0)
    hbp_package_rate_inr: Series[float] = pa.Field(ge=0, nullable=True)
    claim_year: Series[int]
    claim_month: Series[int] = pa.Field(ge=1, le=12)

    class Config:
        strict = False   # allow the full 27-col superset; enforce the critical subset
        coerce = True
