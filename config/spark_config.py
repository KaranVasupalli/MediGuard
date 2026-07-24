"""Builds a Delta-enabled SparkSession from config.yaml. One place, so no job
ever hardcodes Spark settings."""
from pathlib import Path
import yaml
from pyspark.sql import SparkSession
from delta import configure_spark_with_delta_pip

_CONFIG_PATH = Path(__file__).parent / "config.yaml"


def load_config(path: Path = _CONFIG_PATH) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def get_spark(app_suffix: str = "") -> SparkSession:
    cfg = load_config()["spark"]
    name = cfg["app_name"] + (f"-{app_suffix}" if app_suffix else "")
    builder = (
        SparkSession.builder
        .appName(name)
        .master(cfg["master"])
        .config("spark.driver.memory", cfg["driver_memory"])
        .config("spark.sql.shuffle.partitions", cfg["shuffle_partitions"])
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog",
                "org.apache.spark.sql.delta.catalog.DeltaCatalog")
    )
    spark = configure_spark_with_delta_pip(builder).getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark
