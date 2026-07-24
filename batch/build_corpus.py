"""STUB build_corpus for the walking skeleton.
Takes fake rows and writes them as a REAL Delta table, partitioned as the real
job will be. The real version adds ingestion-gate cleaning; the WRITE is genuine."""
from datetime import date, datetime

from pyspark.sql import SparkSession
from serving.schema import CORPUS


def build(spark: SparkSession, rows: list[dict], corpus_path: str) -> int:
    # Pydantic emits python date/datetime; Spark maps these to DateType/TimestampType.
    for r in rows:
        for k in ("admission_date", "discharge_date", "service_date"):
            if isinstance(r[k], datetime):
                r[k] = r[k].date()
    df = spark.createDataFrame(rows, schema=CORPUS)
    (df.write.format("delta")
        .mode("overwrite")
        .partitionBy("claim_year", "claim_month")
        .save(corpus_path))
    return df.count()
