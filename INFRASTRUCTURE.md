# Infrastructure runbook

Two deployments of the **same code**. Only `config.yaml` and environment variables
differ — no application file changes between them.

| | LOCAL | CLOUD |
|---|---|---|
| Broker | Redpanda (Docker) | Azure Event Hubs (Kafka endpoint) |
| Compute | Apache Spark (Docker) | Azure Databricks |
| Storage | Azurite → Azure Storage API | ADLS Gen2 |
| Table format | Delta Lake | Delta Lake |

---

## LOCAL

### Start the stack
```bash
docker compose up -d --build   # --build is needed the first time
docker compose ps              # redpanda, azurite, spark should be Up
```
The first run builds the Spark image (a few minutes, a few GB). Later runs start in
seconds. We build our own Spark image rather than using a vendor one because vendor
tags move or go paid — `bitnami/spark` was withdrawn from the free catalog in 2025.

Useful endpoints while it's up:
- Spark job UI (during a run) — http://localhost:4040
- Redpanda broker — localhost:19092
- Azurite blob — localhost:10000

### Run the Spark batch layer
```bash
docker compose exec spark spark-submit \
  --packages io.delta:delta-spark_2.12:3.2.0 \
  --conf spark.sql.extensions=io.delta.sql.DeltaSparkSessionExtension \
  --conf spark.sql.catalog.spark_catalog=org.apache.spark.sql.delta.catalog.DeltaCatalog \
  /workspace/run_spark_batch.py
```
The Delta jars download once and are cached in the `spark-ivy` volume.

**Why Docker and not native Windows Spark:** inside the container Spark runs on Linux,
so the winutils/Hadoop/JVM-mismatch problems that crash Spark on Windows do not exist.

### Smoke test without Delta jars
```bash
python run_spark_batch.py --smoke     # writes Parquet, no Maven download
```

### Use Azurite instead of plain folders
In `config.yaml` set `storage.backend: azurite`. Nothing else changes.
Create the container once:
```bash
docker compose exec azurite \
  az storage container create -n mediguard --connection-string \
  "DefaultEndpointsProtocol=http;AccountName=devstoreaccount1;AccountKey=Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSRZ6IFsuFq2UVErCz4I6tq/K1SZFPTOtr/KBHBeksoGMGw==;BlobEndpoint=http://127.0.0.1:10000/devstoreaccount1;"
```
Azurite's key is a **published Microsoft constant**, identical on every machine. It is
not a secret and is safe to commit — real credentials never are.

### Stop
```bash
docker compose down          # keeps data volumes
docker compose down -v       # deletes them too
```

---

## CLOUD (Azure)

Activate the $200 credit **only when ready to run and record** — it expires 30 days
after activation regardless of use.

### 1. Storage — ADLS Gen2
Create a storage account with **hierarchical namespace enabled** (that is what makes it
ADLS Gen2 rather than plain Blob), and a container named `mediguard`.

Then, in your shell — **never in config.yaml, never committed**:
```bash
export AZURE_STORAGE_ACCOUNT="youraccountname"
export AZURE_STORAGE_KEY="your-access-key"
```
Set `storage.backend: adls` in `config.yaml`.

### 2. Compute — Azure Databricks
Create a workspace (trial tier), a small single-node cluster with Spark 3.5, then
upload the repo and run `run_spark_batch.py` as a job.

**Cost discipline:** set the cluster to auto-terminate after 15–20 minutes idle. An
idle cluster is the single most common way a student credit disappears.

### 3. Streaming — Event Hubs
Create an Event Hubs namespace (Basic tier) with an event hub named `claims`. It
exposes a **Kafka-compatible endpoint**, so the existing client works unchanged:
```yaml
streaming:
  source: kafka
  kafka:
    bootstrap_servers: "<namespace>.servicebus.windows.net:9093"
    topic: claims
```
Event Hubs requires SASL/TLS; the connection string goes in an environment variable.

### 4. Shut everything down
Delete the Databricks cluster, pause or delete the Event Hubs namespace, and keep only
the storage account (a few GB costs pennies). Then verify in **Cost Management** that
daily spend has dropped to near zero.

---

## Cost expectations

| Resource | Rough cost | Note |
|---|---|---|
| ADLS Gen2, a few GB | ~free | Within the 12-month free allowance |
| Databricks single node | ~₹80–150/hour | Auto-terminate; only run when recording |
| Event Hubs Basic | ~₹900/month | Delete the namespace when finished |

A disciplined end-to-end cloud run — set up, execute, screenshot, tear down — should
cost well under $30 of the $200.

---

## What to claim honestly

> Developed against a containerised Kafka-compatible broker, Apache Spark, and the
> Azure Storage API (Azurite), with the batch layer validated on Azure Databricks
> writing Delta tables to ADLS Gen2.

Every component is the real tool or Microsoft's own emulator of it. The Spark
transformations are tested to produce results identical to the reference Python
implementations (`tests/test_spark_jobs.py`).
