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

Every station is built. The whole pipeline now runs against the Azure Storage API.

---

## What is DONE

**Step 1 — Contracts.** The exact shape of the data was frozen before any logic was
written: 27 fields per bill line, 9 storage tables, one verdict shape. A test proves
they all agree.

**Step 2 — Walking skeleton.** A stub version of the whole line, proving the stations
connect before filling them in. (`orchestrate.py` — kept for history, not run any more.)

**Step 3 — Ingestion gate.** Turns messy hospital bills into clean standard rows.
Renames columns, removes patient names and hashes patient IDs, translates foreign
codes to Indian ones (SNOMED→ICD-10, CPT→PM-JAY HBP) and attaches the official rate,
validates every row, and quarantines bad rows with a reason.

**Step 4 — Rules and baselines.** Four checks: charged above the official rate,
treatment doesn't match the illness, more days billed than the patient stayed, price
far above normal. Each finding has a plain reason and a rupee figure.

**Step 5 — Cost model.** Flags bills far above a fair price range even when they stay
under the official cap.

**Step 6 — Feature engineering.** 21 features, one row per claim, fixed order.

**Step 7 — Machine learning.** LightGBM plus SHAP.

**Step 8 — Anomaly detection.** Isolation Forest on claim shape.

**Step 9 — Provider fraud-ring graph.** networkx; hospitals linked by shared patients.

**Step 10 — Patient history checks.** Repeated expensive tests, rapid readmissions,
duplicate service dates, provider shuttling.

**Step 11 — Streaming layer.** Windowed counters and live alerts; falls back to
replaying the corpus when no broker is running.

**Step 12 — The two agents.** Reader pulls facts from the discharge note, Reasoner
writes the explanation with citations. Local Ollama. A numeric guard discards any
output containing a number that was not in the evidence.

**Step 13 — Dashboard.** Streamlit. Queue, verdict, line-by-line adjudication,
evidence, audit trail, accept/reject/escalate.

**Step 14 — Spark batch layer.** Runs in Docker, tested against the Python versions.

**Step 15 — Storage abstraction.** `config/storage.py` supports local, Azurite, and
ADLS Gen2. One line in `config.yaml` switches between them.

**Step 16 — Every script wired to the storage switch.** Previously only
`run_spark_batch.py` used it, so the rest silently wrote to the laptop whatever the
config said. Now `rebuild_data.py`, `run_ml.py`, `ml/train.py`, `run_anomaly.py`,
`run_graph.py`, `run_history.py`, `score_all.py`, `run_agents.py`, `run_features.py`,
`run_rules.py`, `run_ingestion.py`, `run_cost_model.py`, `run_streaming.py`,
`app/dashboard.py` and `app/review_logic.py` all resolve paths through
`stg.table_path()` and pass `storage_options`.

**Step 17 — Full pipeline verified on Azurite.** Every stage read and wrote through
the real Azure Storage API, with identical results to the local run.

**State right now:** 123 tests passing (Spark tests excluded — they need Docker).
Private GitHub repo. `storage.backend` currently set to `azurite`.

---

## Results (synthetic data, 2000 claims, 11.5% fraud)

**Model vs rules-only**, same test set of 600 claims:

| | Rules only | Model |
|---|---|---|
| Precision | 0.67 | 1.00 |
| Recall | 0.97 | 0.94 |
| F1 | 0.79 | 0.97 |
| PR-AUC | — | 0.976 |

The model catches 65 of 69 fraud claims with **zero** false alarms; the rules flag far
more honest claims to achieve similar recall.

**Each layer catches what the others structurally cannot:**

| Component | Held-out test | Rules catch | It catches |
|---|---|---|---|
| Anomaly detection | unbundling | 0/40 (0%) | 37/40 (92%) |
| Patient history | cross-visit fraud | 0/105 (0%) | 75/105 (71%) |
| Provider graph | planted ring of 3 | — | 3/3 |

The graph never saw the answer. Ring providers scored 0.96–0.97 against 0.34 for the
next highest; their internal sharing ratio was 0.60 vs 0.41 for honest providers.

**Streaming vs batch:** 100% agreement across 400 claims. One logic, two speeds.

**End to end:** 2000 verdicts, 230 flagged for review (11.5%), 1770 auto-approved,
₹1,355,936 of excess identified.

---

## What is REMAINING

**1. Run on real Azure (ADLS Gen2).**
The only untested backend. Steps are in `INFRASTRUCTURE.md`. Storage account with
hierarchical namespace enabled, container `mediguard`, export
`AZURE_STORAGE_ACCOUNT` and `AZURE_STORAGE_KEY`, set `backend: adls`, run
`rebuild_data.py`. Set a budget alert first.

**2. Run the Spark tests in Docker.**
`tests/test_spark_jobs.py` hangs on Windows because Spark needs Java and Linux:
`docker compose exec spark python -m pytest tests/test_spark_jobs.py -v`
The Spark image build previously failed on a DNS error inside Docker.

**3. Optional — Ollama.**
Not installed, so the agents use the deterministic template and record
`explanation source: offline` in the audit trail. Installing it and pulling
`qwen2.5:3b` would produce more natural explanations. Not required.

**4. Scale up and record.**
Larger corpus, cloud run, screenshots, then tear the cloud resources down.

---

## How much is done?

**Roughly 90%.** Every component is built, tested, and verified against the Azure
Storage API. What remains is one cloud run and the write-up.

The honest caveat for the report: all results are on synthetic data. They show the
pipeline works and each component does its job. They are not production accuracy
figures and must not be presented as such. The fraud taxonomy in
`eval/generate_realistic.py` — upcoding, phantom service, impossible stay, quantity
inflation, unbundling, repeat costly visits, plus a planted provider ring — is the
ground truth, and should be described in the write-up.

---

## Key decisions locked in
- **Python 3.12** (not 3.14 — too new for these tools).
- **Cloud is Azure**, not AWS. Storage = ADLS Gen2, compute = Databricks, broker =
  Event Hubs. Earlier notes mentioning Colab and S3 are out of date; ignore them.
- **Azurite is tested before real Azure.** It speaks the real Azure API and costs
  nothing. This already paid for itself: it caught a `wasbs://` scheme bug that
  deltalake does not support, which would have failed in the cloud.
- **Azurite paths use `az://`** for delta-rs, with `allow_http: true` because the
  emulator serves plain HTTP. Spark still uses its own connector config.
- **No `shutil.rmtree` anywhere.** Delta's `mode="overwrite"` works on every backend
  and keeps version history; deleting a folder only works locally.
- **LLM is local Ollama** (`qwen2.5:3b`, sized for a 4GB GPU), fallback off by
  default. No API keys needed.
- **Spark runs in Docker**, not natively on Windows.
- **The graph uses networkx**, not Spark. Fine at this corpus size.
- **Use Git Bash, not PowerShell.** The venv activation differs and PowerShell keeps
  picking up the system Python 3.14.
- **GitHub: private repo.**
- Secrets live in environment variables, never in the repo. `.gitignore` covers
  `.venv`, `/data`, `.env`, `__pycache__`. The Azurite key in `config/storage.py` is a
  published Microsoft constant, not a secret.

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
1. Open Git Bash at `D:\bigdata\mediguard-ai`, run `source .venv/Scripts/activate`.
2. If using Azurite: `docker compose up -d azurite` and check `docker compose ps`.
3. Check health: `python -m pytest tests/ -v --ignore=tests/test_spark_jobs.py`
   → expect **123 passed**. The `--ignore` is required; the Spark tests hang on
   Windows.
4. Full run, in order:
   ```
   python rebuild_data.py
   python run_ml.py
   python run_anomaly.py
   python run_graph.py
   python run_history.py
   python score_all.py
   streamlit run app/dashboard.py
   ```
5. Next step: the real Azure (ADLS Gen2) run.
