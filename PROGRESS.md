# MediGuard AI — Progress So Far

_A plain-words record of where the project stands, so I can pick up cleanly later._

_Last updated: 24 July 2026. Working alone from here on._

## What the project is, in one line
A system that reads hospital bills, checks them for fraud and overcharging, and
produces a clear, explainable verdict for a human reviewer — built as a big-data
platform first, with machine learning and AI added on top.

---

## The big picture: how the whole system works
Think of an assembly line for a hospital bill:

1. A messy bill comes in →
2. It gets cleaned and standardised (front door) →
3. It's checked against rules and normal patterns (fraud checks) →
4. Machine learning scores how suspicious it is →
5. An AI writes a plain-language explanation with evidence →
6. A human sees it all on a dashboard and decides.

Every station is now built. What remains is running it on cloud.

---

## What is DONE

**Step 1 — Contracts.** The exact shape of the data was frozen before any logic was
written: 27 fields per bill line, 9 storage tables, one verdict shape. A test proves
they all agree.

**Step 2 — Walking skeleton.** A stub version of the whole line, with fake bills going
in one end and verdicts coming out the other. Proved the stations connect before
filling them in.

**Step 3 — Ingestion gate.** Turns messy hospital bills into clean standard rows.
Renames columns, removes patient names and hashes patient IDs, translates foreign
codes to Indian ones (SNOMED→ICD-10, CPT→PM-JAY HBP) and attaches the official rate,
validates every row, and quarantines bad rows with a reason.

**Step 4 — Rules and baselines.** Learns what normal looks like from the corpus, then
runs four checks: charged above the official rate, treatment doesn't match the
illness, more days billed than the patient stayed, price far above normal. Each
finding has a plain reason and a rupee figure.

**Step 5 — Cost model.** Estimates a fair price range per line and flags bills far
above it even when they stay under the official cap.

**Step 6 — Feature engineering.** Turns all the signals above into one row of numbers
per claim, in a fixed order, ready for the model.

**Step 7 — Machine learning.** A LightGBM model gives one fraud score per claim. SHAP
explains which features drove each score. The model beats the rules-only baseline —
this is the number that justifies building it.

**Step 8 — Anomaly detection.** Flags strange bills that break no specific rule.
Catches unbundling, which the four rules structurally cannot see.

**Step 9 — Provider fraud-ring graph.** Builds a network of hospitals linked by shared
patients and finds tight clusters. Uses networkx. The generator plants a real ring of
3 hospitals shuttling 45 patients, and records who is in it, so the graph can be
scored honestly.

**Step 10 — Patient history checks.** Repeated expensive tests, rapid readmissions,
duplicate service dates, patients shuttled between providers.

**Step 11 — Streaming layer.** Redpanda in Docker, windowed counters for sudden spikes
in a hospital's billing.

**Step 12 — The two agents.** Reader pulls facts from the discharge note. Reasoner
writes the final explanation with citations. Both run on local Ollama. A numeric guard
throws away any output where the model invented a number that was not in the evidence.

**Step 13 — Dashboard.** The screen a reviewer uses: verdict, evidence, money at
stake, accept/reject/escalate.

**Step 14 — Spark batch layer.** Runs in Docker. Tested to produce the same results as
the plain Python versions.

**Step 15 — Storage abstraction.** `config/storage.py` supports three backends —
local folders, Azurite (Azure's emulator, runs on the laptop), and real ADLS Gen2.
Switching is one line in `config.yaml`. Credentials come from environment variables
only.

**State right now:** runs on the Windows laptop. **133 tests**, all passing when the
Spark tests are skipped. Pushed to a private GitHub repo.

---

## What is REMAINING

**1. Make the scripts use the storage switch.**
Only `run_spark_batch.py` currently calls `table_path()`. The other seventeen scripts
still read `cfg["paths"]` directly, so they write to the laptop no matter what
`storage.backend` says. Each one needs changing. Start with `rebuild_data.py`.

**2. Remove `shutil.rmtree` from `rebuild_data.py`.**
It deletes a local folder, which will not work against Azure. Delta's own
`mode="overwrite"` already does the job and keeps version history.

**3. Test on Azurite.**
Set `storage.backend: azurite`, start Docker, create the container, run the pipeline.
Free, offline, and uses the real Azure Storage API — so it proves the cloud code path
before spending anything.

**4. Run on real Azure.**
Storage account with hierarchical namespace enabled, container named `mediguard`,
export `AZURE_STORAGE_ACCOUNT` and `AZURE_STORAGE_KEY`, set
`storage.backend: adls`. Set a budget alert first. Full instructions in
`INFRASTRUCTURE.md`.

**5. Run the Spark tests in Docker.**
`tests/test_spark_jobs.py` hangs on Windows because Spark needs Java and Linux. It
must run inside the container:
`docker compose exec spark python -m pytest tests/test_spark_jobs.py -v`

**6. Scale up and evaluate.**
Generate a larger corpus, run the batch jobs on cloud, record the results, then tear
the cloud resources down.

---

## How much is done?

**Roughly 80%.**

Every component is built and tested. What remains is plumbing the storage switch
through the remaining scripts, and proving the whole thing runs on real Azure. That is
real work, but it is not new components.

The honest caveat for the write-up: all results are on synthetic data. The numbers show
the pipeline works and each component does its job. They are not production accuracy
figures and should not be presented as such.

---

## Key decisions locked in
- **Python 3.12** (not 3.14 — too new for these tools).
- **Cloud is Azure**, not AWS. Storage = ADLS Gen2, compute = Databricks, broker =
  Event Hubs. Earlier notes mentioning Colab and S3 are out of date; ignore them.
- **Azurite is tested before real Azure.** It speaks the real Azure API and costs
  nothing.
- **LLM is local Ollama** (`qwen2.5:3b`, sized for a 4GB GPU) with an optional cloud
  fallback switch, off by default. No API keys needed.
- **Spark runs in Docker**, not natively on Windows. Windows Spark breaks on
  winutils and JVM mismatches.
- **The graph uses networkx**, not Spark. Fine at this corpus size.
- **GitHub: private repo, one commit per step.**
- Secrets (patient salt, Azure keys) live in environment variables, never in the repo.
  `.gitignore` covers `.venv`, `/data`, `.env`, `__pycache__`.

---

## Cost discipline (Azure)
- Storage is nearly free. Databricks and Event Hubs are not.
- Set the Databricks cluster to auto-terminate after 15–20 minutes idle.
- Delete the Event Hubs namespace when finished.
- Set a budget alert in Cost Management before creating anything.
- Activate the $200 credit only when ready to run and record — it expires 30 days
  after activation whether used or not.

---

## How to resume
1. Open `D:\bigdata\mediguard-ai`, run `source .venv/Scripts/activate`.
2. Check health: `python -m pytest tests/ -v --ignore=tests/test_spark_jobs.py`
   → expect **128 passed**. The `--ignore` is required; the Spark tests hang on
   Windows.
3. If the ML tests skip, run `python rebuild_data.py` then `python run_ml.py`.
4. Next step to build: make `rebuild_data.py` use `table_path()` from
   `config/storage.py`.
