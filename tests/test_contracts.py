"""Green here = contracts frozen. Run: pytest tests/test_contracts.py"""
from datetime import date, datetime
import pytest

from data_quality.schema_validator import ClaimLine, CANONICAL_COLUMNS
from serving.schema import SCHEMAS, WRITERS, SCHEMA_VERSION
from agents.schemas import Verdict, EvidenceBundle, Audit


def test_canonical_has_expected_column_count():
    assert len(CANONICAL_COLUMNS) == 27
    assert len(set(CANONICAL_COLUMNS)) == 27          # no dupes


def test_claimline_pydantic_matches_canonical_columns():
    assert set(ClaimLine.model_fields) == set(CANONICAL_COLUMNS)


def test_corpus_schema_matches_canonical_columns():
    corpus_cols = [f.name for f in SCHEMAS["corpus"].fields]
    assert set(corpus_cols) == set(CANONICAL_COLUMNS)  # corpus == canonical, no drift


def test_every_table_has_exactly_one_writer():
    assert set(WRITERS) == set(SCHEMAS)                # 9 tables, 9 registered writers
    for t, w in WRITERS.items():
        assert w["job"], f"{t} has no writer"
    shared = [t for t, w in WRITERS.items() if "+" in w["job"]]
    assert shared == ["verdicts"]                      # only intentional shared table


def test_streaming_counters_is_idempotent_merge():
    assert WRITERS["streaming_counters"]["mode"] == "merge_idem"


def test_valid_claimline_parses():
    ClaimLine(
        claim_id="CLM-1", line_no=1, patient_hash="h", provider_id="P1",
        provider_state="MH", admission_date=date(2026, 7, 10),
        discharge_date=date(2026, 7, 13), service_date=date(2026, 7, 11),
        los_days=3, icd10_primary="A09", line_desc="ICU care",
        quantity=2, unit_price_inr=9500, billed_inr=19000,
        source_system="synthea", ingest_ts=datetime.now(),
        claim_year=2026, claim_month=7,
    )


def test_discharge_before_admission_rejected():
    with pytest.raises(ValueError):
        ClaimLine(
            claim_id="CLM-2", line_no=1, patient_hash="h", provider_id="P1",
            provider_state="MH", admission_date=date(2026, 7, 13),
            discharge_date=date(2026, 7, 10), service_date=date(2026, 7, 11),
            los_days=0, icd10_primary="A09", line_desc="x", quantity=1,
            unit_price_inr=1, billed_inr=1, source_system="s",
            ingest_ts=datetime.now(), claim_year=2026, claim_month=7,
        )


def test_verdict_contract_roundtrips():
    v = Verdict(
        claim_id="CLM-2026-0447821", verdict="FLAG_FOR_AUDIT", fraud_score=0.87,
        billed_total_inr=55100, justified_total_inr=19230, estimated_excess_inr=35870,
        recommended_action="HOLD_AND_ROUTE_TO_SIU",
        evidence=EvidenceBundle(), audit=Audit(human_review_required=True),
    )
    assert Verdict.model_validate_json(v.model_dump_json()).fraud_score == 0.87


def test_schema_version_pinned():
    assert SCHEMA_VERSION == "0.1.0"
