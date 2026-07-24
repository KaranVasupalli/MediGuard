"""Real PySpark implementations of the batch layer.

WHY THESE AND NOT THE OTHERS
Spark earns its place where the work is genuinely large and parallel:
  * mining baselines  - group and aggregate over every claim line ever received
  * patient history   - window functions over each patient's ordered visit history
  * provider edges    - a self-join over patients to find shared-patient links

The graph ALGORITHMS (PageRank, community detection) deliberately stay in networkx.
Forty providers is a tiny graph; distributing it would add cost and no speed. Spark
builds the edges from millions of rows, then the small graph is analysed in memory.
Claiming to "use Spark for the graph" when the graph has 40 nodes would be dishonest
engineering.

Every function here mirrors a pure-Python equivalent, and a test asserts the two agree.
That matters: the Python versions run locally on a laptop, the Spark versions run on a
cluster, and they must never disagree about what fraud is.
"""
from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F


# ---------------------------------------------------------------- baselines
def mine_diag_procedure_norms(corpus: DataFrame, min_support: int = 20) -> DataFrame:
    """Co-occurrence of (diagnosis, procedure) as a share of that diagnosis."""
    valid = corpus.filter(
        F.col("icd10_primary").isNotNull()
        & (F.col("icd10_primary") != "UNKNOWN")
        & F.col("hbp_code").isNotNull()
    )
    pair = valid.groupBy("icd10_primary", "hbp_code").agg(
        F.count("*").alias("support_n"))
    dx = valid.groupBy("icd10_primary").agg(F.count("*").alias("dx_total"))

    return (pair.join(dx, on="icd10_primary", how="inner")
            .withColumn("cooccurrence",
                        F.round(F.col("support_n") / F.col("dx_total"), 4))
            .withColumn("support_band",
                        F.when(F.col("support_n") >= min_support, F.lit("high"))
                         .otherwise(F.lit("low")))
            .select("icd10_primary", "hbp_code", "cooccurrence",
                    "support_n", "support_band"))


def mine_procedure_cost_pctiles(corpus: DataFrame) -> DataFrame:
    """Exact p25/p50/p95 of billed amount per (procedure, state).

    Uses `percentile` (exact) rather than `percentile_approx`: the cost model accuses
    hospitals of overcharging, and an approximate boundary is not something to defend
    in an audit. At this data size the exact version is affordable.
    """
    return (corpus.filter(F.col("hbp_code").isNotNull())
            .groupBy("hbp_code", "provider_state")
            .agg(
                F.expr("percentile(billed_inr, 0.25)").alias("p25_inr"),
                F.expr("percentile(billed_inr, 0.50)").alias("p50_inr"),
                F.expr("percentile(billed_inr, 0.95)").alias("p95_inr"),
                F.count("*").alias("n"))
            .withColumn("hospital_tier", F.lit("ALL"))
            .withColumn("p25_inr", F.round("p25_inr", 2))
            .withColumn("p50_inr", F.round("p50_inr", 2))
            .withColumn("p95_inr", F.round("p95_inr", 2))
            .select("hbp_code", "provider_state", "hospital_tier",
                    "p25_inr", "p50_inr", "p95_inr", "n"))


# ---------------------------------------------------------------- provider edges
def build_provider_edges(corpus: DataFrame, min_shared: int = 2) -> DataFrame:
    """Providers linked by shared patients, with the count of distinct patients.

    A self-join on patient_hash. The `a < b` predicate keeps one row per unordered
    pair and, importantly, drops the self-match — without it every provider would
    appear linked to itself with a huge weight.
    """
    pp = corpus.select("patient_hash", "provider_id").distinct()
    a = pp.withColumnRenamed("provider_id", "provider_a")
    b = pp.withColumnRenamed("provider_id", "provider_b")

    return (a.join(b, on="patient_hash", how="inner")
            .filter(F.col("provider_a") < F.col("provider_b"))
            .groupBy("provider_a", "provider_b")
            .agg(F.countDistinct("patient_hash").alias("shared_patients"))
            .filter(F.col("shared_patients") >= min_shared))


# ---------------------------------------------------------------- patient history
def build_patient_claim_facts(corpus: DataFrame) -> DataFrame:
    """Collapse claim lines to one row per claim (the grain history works on)."""
    return (corpus.groupBy("claim_id", "patient_hash", "provider_id")
            .agg(F.min("admission_date").alias("admission_date"),
                 F.max("discharge_date").alias("discharge_date"),
                 F.sum("billed_inr").alias("billed_inr"),
                 F.max(F.when(F.col("hbp_code") == "HBP-ICU-002", 1)
                       .otherwise(0)).alias("has_icu")))


def build_patient_history(corpus: DataFrame, rapid_readmit_days: int = 3) -> DataFrame:
    """Cross-visit context per (patient, claim), using window functions.

    STRICTLY NO LOOKAHEAD. Every window is bounded by `rowsBetween(
    Window.unboundedPreceding, -1)` — that is, prior claims only. A window that
    included the current or later rows would leak the future into the score, which
    looks excellent in testing and is worthless in production.
    """
    claims = build_patient_claim_facts(corpus)

    ordered = Window.partitionBy("patient_hash").orderBy("admission_date", "claim_id")
    prior_only = ordered.rowsBetween(Window.unboundedPreceding, -1)

    return (claims
            .withColumn("prior_claims", F.count("*").over(prior_only))
            .withColumn("prev_discharge", F.lag("discharge_date").over(ordered))
            .withColumn("prior_icu", F.sum("has_icu").over(prior_only))
            .withColumn("distinct_providers_prior",
                        F.size(F.collect_set("provider_id").over(prior_only)))
            .withColumn("days_since_last_discharge",
                        F.when(F.col("prev_discharge").isNull(), F.lit(-1))
                         .otherwise(F.datediff("admission_date", "prev_discharge")))
            .withColumn("rapid_readmission",
                        F.when((F.col("days_since_last_discharge") >= 0)
                               & (F.col("days_since_last_discharge") <= rapid_readmit_days),
                               F.lit(1)).otherwise(F.lit(0)))
            .withColumn("repeat_icu",
                        F.when((F.col("has_icu") == 1) & (F.col("prior_icu") >= 1),
                               F.lit(1)).otherwise(F.lit(0)))
            .select("patient_hash", "claim_id", "provider_id", "prior_claims",
                    "days_since_last_discharge", "distinct_providers_prior",
                    "rapid_readmission", "repeat_icu"))


# ---------------------------------------------------------------- quality
def corpus_quality_report(corpus: DataFrame, run_id: str) -> DataFrame:
    """Row counts and null rates — the health record of a batch run."""
    total = corpus.count()
    unmapped = corpus.filter(F.col("icd10_primary") == "UNKNOWN").count()
    no_hbp = corpus.filter(F.col("hbp_code").isNull()).count()
    spark = corpus.sparkSession
    return spark.createDataFrame([
        (run_id, "corpus", "total_lines", float(total)),
        (run_id, "corpus", "unmapped_icd10", float(unmapped)),
        (run_id, "corpus", "missing_hbp_code", float(no_hbp)),
        (run_id, "corpus", "distinct_claims", float(corpus.select("claim_id").distinct().count())),
        (run_id, "corpus", "distinct_providers", float(corpus.select("provider_id").distinct().count())),
    ], ["run_id", "table", "metric", "value"])
