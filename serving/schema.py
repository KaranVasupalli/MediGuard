"""
Delta table DDL + version registry + the single-writer contract.
Each table has exactly one writer (verdicts is the one intentional shared table).
"""
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType, LongType,
    DoubleType, TimestampType, DateType,
)

# ── the ONE writer allowed per table ───────────────────────────────────────────
WRITERS = {
    "corpus":                  {"job": "build_corpus",     "mode": "overwrite"},
    "diag_procedure_norms":    {"job": "mine_baselines",   "mode": "overwrite"},
    "procedure_cost_pctiles":  {"job": "mine_baselines",   "mode": "overwrite"},
    "provider_risk":           {"job": "provider_graph",   "mode": "overwrite"},
    "patient_history":         {"job": "patient_history",  "mode": "overwrite"},
    "hbp_rates":               {"job": "ingestion_gate",   "mode": "overwrite"},
    "streaming_counters":      {"job": "stream_job",       "mode": "merge_idem"},
    "verdicts":                {"job": "dashboard+agents", "mode": "merge"},
    "quality_report":          {"job": "ingestion_gate",   "mode": "append"},
}

# ── schemas ─────────────────────────────────────────────────────────────────────
CORPUS = StructType([
    StructField("claim_id", StringType(), False),
    StructField("line_no", IntegerType(), False),
    StructField("patient_hash", StringType(), False),
    StructField("provider_id", StringType(), False),
    StructField("provider_name", StringType(), True),
    StructField("provider_state", StringType(), False),
    StructField("admission_date", DateType(), False),
    StructField("discharge_date", DateType(), False),
    StructField("service_date", DateType(), False),
    StructField("los_days", IntegerType(), False),
    StructField("icd10_primary", StringType(), False),
    StructField("icd10_desc", StringType(), True),
    StructField("snomed_src", StringType(), True),
    StructField("cpt_code", StringType(), True),
    StructField("hbp_code", StringType(), True),
    StructField("hbp_desc", StringType(), True),
    StructField("line_desc", StringType(), False),
    StructField("department", StringType(), True),
    StructField("quantity", DoubleType(), False),
    StructField("unit_price_inr", DoubleType(), False),
    StructField("billed_inr", DoubleType(), False),
    StructField("hbp_package_rate_inr", DoubleType(), True),
    StructField("source_system", StringType(), False),
    StructField("ingest_ts", TimestampType(), False),
    StructField("quality_flags", StringType(), True),
    StructField("claim_year", IntegerType(), False),    # partition col
    StructField("claim_month", IntegerType(), False),   # partition col
])

DIAG_PROCEDURE_NORMS = StructType([          # grain: (icd10, hbp_code)
    StructField("icd10_primary", StringType(), False),
    StructField("hbp_code", StringType(), False),
    StructField("cooccurrence", DoubleType(), False),
    StructField("support_n", LongType(), False),
    StructField("support_band", StringType(), True),    # high / low / none
])

PROCEDURE_COST_PCTILES = StructType([        # grain: (hbp, state, tier)
    StructField("hbp_code", StringType(), False),
    StructField("provider_state", StringType(), False),
    StructField("hospital_tier", StringType(), False),
    StructField("p25_inr", DoubleType(), False),
    StructField("p50_inr", DoubleType(), False),
    StructField("p95_inr", DoubleType(), False),
    StructField("n", LongType(), False),
])

PROVIDER_RISK = StructType([                 # grain: provider_id
    StructField("provider_id", StringType(), False),
    StructField("pagerank", DoubleType(), False),
    StructField("pagerank_pctile", DoubleType(), False),
    StructField("community_id", StringType(), True),
    StructField("community_size", IntegerType(), True),
    StructField("shared_patient_edges", LongType(), True),
])

PATIENT_HISTORY = StructType([               # grain: (patient, claim)
    StructField("patient_hash", StringType(), False),
    StructField("claim_id", StringType(), False),
    StructField("prior_claims", IntegerType(), True),
    StructField("prior_mri", IntegerType(), True),
    StructField("repeat_test_flag", IntegerType(), True),
])

HBP_RATES = StructType([                     # grain: hbp_code
    StructField("hbp_code", StringType(), False),
    StructField("hbp_desc", StringType(), True),
    StructField("package_rate_inr", DoubleType(), False),
])

STREAMING_COUNTERS = StructType([            # grain: (provider, window)
    StructField("provider_id", StringType(), False),
    StructField("window_start", TimestampType(), False),
    StructField("window_end", TimestampType(), True),
    StructField("icu_rate", DoubleType(), True),
    StructField("procedure_rate", DoubleType(), True),
    StructField("cost_drift", DoubleType(), True),
])

VERDICTS = StructType([                      # grain: claim_id
    StructField("claim_id", StringType(), False),
    StructField("verdict", StringType(), True),
    StructField("fraud_score", DoubleType(), True),
    StructField("estimated_excess_inr", DoubleType(), True),
    StructField("recommended_action", StringType(), True),
    StructField("adjudicator_decision", StringType(), True),   # accept / reject / escalate
    StructField("decided_by", StringType(), True),
    StructField("decided_ts", TimestampType(), True),
    StructField("verdict_json", StringType(), True),           # full contract blob
])

QUALITY_REPORT = StructType([                # grain: run_id
    StructField("run_id", StringType(), False),
    StructField("table", StringType(), True),
    StructField("metric", StringType(), True),
    StructField("value", DoubleType(), True),
    StructField("run_ts", TimestampType(), True),
])

SCHEMAS = {
    "corpus": CORPUS, "diag_procedure_norms": DIAG_PROCEDURE_NORMS,
    "procedure_cost_pctiles": PROCEDURE_COST_PCTILES, "provider_risk": PROVIDER_RISK,
    "patient_history": PATIENT_HISTORY, "hbp_rates": HBP_RATES,
    "streaming_counters": STREAMING_COUNTERS, "verdicts": VERDICTS,
    "quality_report": QUALITY_REPORT,
}
SCHEMA_VERSION = "0.1.0"   # bump on any frozen-contract change; changes need owner sign-off
