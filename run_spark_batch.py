"""Run the batch layer on REAL Spark.

  docker compose up -d
  docker compose exec spark spark-submit \
      --packages io.delta:delta-spark_2.12:3.2.0 \
      --conf spark.sql.extensions=io.delta.sql.DeltaSparkSessionExtension \
      --conf spark.sql.catalog.spark_catalog=org.apache.spark.sql.delta.catalog.DeltaCatalog \
      /workspace/run_spark_batch.py

The same file runs unchanged on Azure Databricks against ADLS Gen2 — only
`storage.backend` in config.yaml and the credential environment variables differ.

`--smoke` writes Parquet instead of Delta so the transformations can be exercised
without downloading the Delta jars.
"""
import argparse
import sys
from datetime import datetime

from pyspark.sql import SparkSession

from config.spark_config import load_config
from config import storage as stg
from batch.spark_jobs import (
    mine_diag_procedure_norms, mine_procedure_cost_pctiles,
    build_provider_edges, build_patient_history, corpus_quality_report,
)


def build_session(app: str = "mediguard-batch", delta: bool = True) -> SparkSession:
    cfg = load_config()["spark"]
    b = (SparkSession.builder.appName(app)
         .master(cfg.get("master", "local[*]"))
         .config("spark.sql.shuffle.partitions", cfg.get("shuffle_partitions", 64))
         .config("spark.driver.memory", cfg.get("driver_memory", "4g")))

    if delta:
        b = (b.config("spark.sql.extensions",
                      "io.delta.sql.DeltaSparkSessionExtension")
              .config("spark.sql.catalog.spark_catalog",
                      "org.apache.spark.sql.delta.catalog.DeltaCatalog"))

    for k, v in stg.spark_storage_options().items():
        if not k.startswith("_"):
            b = b.config(k, v)

    s = b.getOrCreate()
    s.sparkContext.setLogLevel("ERROR")
    return s


def main(smoke: bool = False, limit: int | None = None):
    fmt = "parquet" if smoke else "delta"
    cfg = load_config()
    run_id = datetime.now().strftime("run-%Y%m%d-%H%M%S")

    print("=" * 64)
    print("MEDIGUARD — SPARK BATCH LAYER")
    print("=" * 64)
    print(f"  storage : {stg.describe()}")
    print(f"  format  : {fmt}")
    print(f"  run id  : {run_id}")

    spark = build_session(delta=not smoke)
    print(f"  spark   : {spark.version}  master={spark.sparkContext.master}")

    corpus_path = cfg["paths"]["corpus"] if stg.backend() == "local" \
        else stg.table_path("corpus")
    print(f"\n1) reading corpus from {corpus_path} ...")
    if smoke:
        # A Delta table is Parquet files plus a transaction log, so the Parquet can be
        # read directly when the Delta jars are unavailable. Fine for a smoke test;
        # NOT correct in general, because it ignores the log and would include files
        # deleted by later versions.
        corpus = spark.read.option("recursiveFileLookup", "true") \
            .parquet(corpus_path)
    else:
        corpus = spark.read.format("delta").load(corpus_path)
    if limit:
        corpus = corpus.limit(limit)
    corpus.cache()
    n_lines = corpus.count()
    n_claims = corpus.select("claim_id").distinct().count()
    print(f"   {n_lines:,} lines across {n_claims:,} claims")

    def write(df, name, partition_by=None):
        path = stg.table_path(name)
        w = df.write.format(fmt).mode("overwrite").option("overwriteSchema", "true")
        if partition_by:
            w = w.partitionBy(*partition_by)
        w.save(path)
        return path

    print("\n2) mining diagnosis-procedure norms ...")
    norms = mine_diag_procedure_norms(corpus)
    print(f"   {norms.count()} pairs -> {write(norms, 'diag_procedure_norms')}")

    print("\n3) mining cost percentiles ...")
    pct = mine_procedure_cost_pctiles(corpus)
    print(f"   {pct.count()} price bands -> {write(pct, 'procedure_cost_pctiles')}")

    print("\n4) building provider shared-patient edges ...")
    edges = build_provider_edges(corpus, min_shared=2)
    print(f"   {edges.count()} edges -> {write(edges, 'provider_edges')}")

    print("\n5) building patient history (window functions) ...")
    hist = build_patient_history(corpus)
    print(f"   {hist.count()} rows -> {write(hist, 'patient_history')}")

    print("\n6) quality report ...")
    qr = corpus_quality_report(corpus, run_id)
    write(qr, "quality_report")
    for r in qr.collect():
        print(f"   {r['metric']:<22} {r['value']:,.0f}")

    print("\n" + "=" * 64)
    print("Spark batch complete. Graph algorithms run separately in networkx —")
    print("Spark builds the edges from every claim line; a 40-node graph does not")
    print("need distributing.")
    print("=" * 64)
    spark.stop()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="write Parquet instead of Delta (no Maven download needed)")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    sys.exit(main(smoke=args.smoke, limit=args.limit))
