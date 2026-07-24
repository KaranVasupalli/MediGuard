"""Tests for the Spark batch jobs. Run: pytest tests/test_spark_jobs.py -v

THE POINT OF THESE TESTS
The Python implementations run on a laptop; the Spark ones run on a cluster. If they
ever disagree about what fraud is, the system has two different truths and neither can
be defended. So each test computes the same thing both ways and asserts equality.

Spark is slow to start, so the session is shared across the module.
"""
import pytest

pyspark = pytest.importorskip("pyspark")

from datetime import date

from pyspark.sql import SparkSession

from batch.spark_jobs import (
    mine_diag_procedure_norms, mine_procedure_cost_pctiles,
    build_provider_edges, build_patient_history, corpus_quality_report,
)
from batch.mine_baselines import (
    mine_diag_procedure_norms as py_norms,
    mine_procedure_cost_pctiles as py_pctiles,
)
from batch.provider_graph import build_provider_graph


@pytest.fixture(scope="module")
def spark():
    s = (SparkSession.builder.appName("mediguard-tests").master("local[1]")
         .config("spark.ui.enabled", "false")
         .config("spark.sql.shuffle.partitions", "2")
         .config("spark.driver.memory", "1g")
         .getOrCreate())
    s.sparkContext.setLogLevel("ERROR")
    yield s
    s.stop()


def _rows():
    """Small corpus with known structure, as plain dicts (the Python side's input)."""
    out = []
    for i in range(30):
        out.append({"claim_id": f"C{i}", "line_no": 1, "patient_hash": f"p{i % 10}",
                    "provider_id": "H01", "provider_state": "MH",
                    "icd10_primary": "A09", "hbp_code": "HBP-BED-001",
                    "billed_inr": 2000.0 + i, "quantity": 1.0, "los_days": 2,
                    "admission_date": date(2026, 3, 1 + (i % 20)),
                    "discharge_date": date(2026, 3, 2 + (i % 20))})
    for i in range(30, 45):
        out.append({"claim_id": f"C{i}", "line_no": 1, "patient_hash": f"p{i % 10}",
                    "provider_id": "H02", "provider_state": "MH",
                    "icd10_primary": "E11", "hbp_code": "HBP-LAB-014",
                    "billed_inr": 180.0 + i, "quantity": 1.0, "los_days": 1,
                    "admission_date": date(2026, 3, 1 + (i % 20)),
                    "discharge_date": date(2026, 3, 2 + (i % 20))})
    return out


@pytest.fixture(scope="module")
def corpus(spark):
    return spark.createDataFrame(_rows())


# ---------- Spark result == Python result ----------

def test_norms_match_python_implementation(corpus):
    spark_out = {(r["icd10_primary"], r["hbp_code"]): r
                 for r in mine_diag_procedure_norms(corpus).collect()}
    py_out = {(r["icd10_primary"], r["hbp_code"]): r for r in py_norms(_rows())}

    assert set(spark_out) == set(py_out), "different (diagnosis, procedure) pairs"
    for key, py_row in py_out.items():
        s = spark_out[key]
        assert s["support_n"] == py_row["support_n"]
        assert abs(s["cooccurrence"] - py_row["cooccurrence"]) < 1e-4
        assert s["support_band"] == py_row["support_band"]


def test_cost_percentiles_match_python_implementation(corpus):
    spark_out = {(r["hbp_code"], r["provider_state"]): r
                 for r in mine_procedure_cost_pctiles(corpus).collect()}
    py_out = {(r["hbp_code"], r["provider_state"]): r for r in py_pctiles(_rows())}

    assert set(spark_out) == set(py_out)
    for key, py_row in py_out.items():
        s = spark_out[key]
        assert s["n"] == py_row["n"]
        for field in ("p25_inr", "p50_inr", "p95_inr"):
            assert abs(s[field] - py_row[field]) < 0.5, f"{field} differs for {key}"


def test_provider_edges_match_python_graph(corpus):
    edges = {(r["provider_a"], r["provider_b"]): r["shared_patients"]
             for r in build_provider_edges(corpus, min_shared=1).collect()}
    G = build_provider_graph(_rows(), min_shared=1)
    py_edges = {tuple(sorted((a, b))): d["weight"] for a, b, d in G.edges(data=True)}
    assert edges == py_edges


# ---------- Spark-specific correctness ----------

def test_self_join_does_not_link_a_provider_to_itself(corpus):
    edges = build_provider_edges(corpus, min_shared=1).collect()
    assert all(r["provider_a"] != r["provider_b"] for r in edges)


def test_each_provider_pair_appears_once(corpus):
    edges = [(r["provider_a"], r["provider_b"])
             for r in build_provider_edges(corpus, min_shared=1).collect()]
    assert len(edges) == len(set(edges))
    assert all(a < b for a, b in edges)          # canonical ordering


def test_min_shared_filter_applies(corpus):
    loose = build_provider_edges(corpus, min_shared=1).count()
    strict = build_provider_edges(corpus, min_shared=999).count()
    assert strict == 0 and loose > 0


def test_patient_history_first_claim_has_no_prior(corpus):
    """No-lookahead, enforced in the Spark version too."""
    rows = build_patient_history(corpus).collect()
    firsts = [r for r in rows if r["prior_claims"] == 0]
    assert firsts
    for r in firsts:
        assert r["days_since_last_discharge"] == -1
        assert r["rapid_readmission"] == 0
        assert r["repeat_icu"] == 0


def test_patient_history_counts_increase_within_a_patient(corpus):
    rows = build_patient_history(corpus).collect()
    by_patient = {}
    for r in rows:
        by_patient.setdefault(r["patient_hash"], []).append(r["prior_claims"])
    for counts in by_patient.values():
        assert sorted(counts) == list(range(len(counts)))   # 0,1,2,... no gaps


def test_patients_are_not_mixed_together(corpus):
    rows = build_patient_history(corpus).collect()
    assert len({r["patient_hash"] for r in rows}) == 10


def test_quality_report_counts_are_right(corpus):
    rep = {r["metric"]: r["value"] for r in corpus_quality_report(corpus, "run-1").collect()}
    assert rep["total_lines"] == 45
    assert rep["distinct_claims"] == 45
    assert rep["distinct_providers"] == 2
    assert rep["unmapped_icd10"] == 0
