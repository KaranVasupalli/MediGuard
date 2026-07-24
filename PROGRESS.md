# MediGuard AI — Progress So Far

_A plain-words record of where the project stands, so we can pick up cleanly later._

## What the project is, in one line
A system that reads hospital bills, checks them for fraud and overcharging, and
produces a clear, explainable verdict for a human reviewer — built as a big-data
platform first, with machine learning and AI added on top.

---

## The big picture: how the whole system will work
Think of an assembly line for a hospital bill:

1. A messy bill comes in →
2. It gets cleaned and standardised (front door) →
3. It's checked against rules and normal patterns (fraud checks) →
4. Machine learning scores how suspicious it is →
5. An AI writes a plain-language explanation with evidence →
6. A human sees it all on a dashboard and decides.

We are building this line one station at a time. Each station is finished, tested,
and saved to GitHub before moving to the next.

---

## What is DONE (Steps 1–5)

**Step 1 — The foundation (contracts).**
We agreed on the exact "shape" of the data before building anything: what a bill line
looks like (27 fields), the shape of the 9 storage tables, and the shape of the final
verdict. A test proves all these shapes agree. This is the ground everything stands on.

**Step 2 — The walking skeleton.**
We built a tiny version of the *whole* line and pushed fake bills through it end to
end — in one side, out the other as a verdict. No real fraud logic yet; the point was
to prove every station connects to the next before filling them in.

**Step 3 — The ingestion gate (the front door).**
The first real component. It takes messy hospital bills and turns each one into a
clean, standard row. It renames each hospital's columns to our standard, removes
patient names and scrambles patient IDs (privacy), translates foreign medical codes
into Indian ones (SNOMED→ICD-10, CPT→PM-JAY HBP codes) and attaches the official
government rupee rate, checks every row against the rulebook, and sends bad rows to a
"quarantine" pile with a reason. Messy in, clean and private out.

**Step 4 — The rules & baseline engine (first fraud detection).**
The first component that actually looks for fraud. It first learns what "normal" looks
like from all the bills (which treatments go with which illnesses, and typical price
ranges), then runs four checks on every bill: charged above the official rate,
treatment doesn't match the illness, more days billed than the patient stayed, and
price far above the population norm. Every finding comes with a plain reason and an
exact rupee figure.

**Step 5 — The cost model (smarter money check).**
The last of the hand-built checks. Instead of only "did they break the fixed rate," it
estimates a fair price range for each line and flags bills sitting far above it — even
when they stay under the official cap. This catches inflated bills the hard rules would
miss.

**State right now:** everything runs on your Windows laptop, no cloud needed yet.
**29 automated tests pass.** Every step is a clean commit in GitHub.

---

## What is REMAINING

**Feature engineering (next step).**
Turn all the signals from the checks above into one tidy row of numbers per claim —
the format machine learning needs.

**Machine learning model.**
Train a model that combines all the signals into a single fraud score, plus a tool
(SHAP) that explains which signals drove each score.

**Anomaly detection.**
A method that flags unusual bills even when they break no specific rule.

**The "reader" AI (semantic matching).**
Matches the doctor's discharge summary against the billed items to spot charges with no
medical justification in the notes.

**Provider fraud-ring graph.**
Builds a network of hospitals and shared patients to spot coordinated fraud rings.
(This is the genuine "big data" step that will run on Spark, on Google Colab, writing
to your S3 bucket.)

**Patient history checks.**
Spot things like the same expensive test billed repeatedly across visits.

**The live/streaming layer.**
Process bills in real time as they arrive (Redpanda locally, Kafka on cloud), catching
sudden spikes in a hospital's billing.

**The two AI agents (with local Ollama).**
The Reader (pulls facts from documents) and the Reasoner (writes the final
explanation with citations). Runs locally on your machine, private, no API keys.

**The serving layer + dashboard.**
The screen a human reviewer actually uses: the verdict, the evidence, the money at
stake, and accept/reject buttons.

**Real data at scale + cloud run.**
Generate the full synthetic corpus and run the heavy batch jobs on Colab + S3.

**Final evaluation.**
Measure how well the system catches fraud and write up the results.

---

## How much is done?

**Roughly 25–30% of the full project.**

Why that number, honestly:
- What's done is the **foundation and the entire hand-built fraud-checking layer** —
  the hardest part to get *right*, and the part everything else depends on. Getting
  this solid removes most of the risk from the rest.
- But by sheer volume, a lot remains: machine learning, the AI agents, the streaming
  layer, the fraud-ring graph, the dashboard, and running it all at scale in the cloud.

So: **the skeleton and the muscle of the "rules" half are complete; the "learning" and
"AI" halves, the live layer, and the cloud/scale work are still ahead.** The early
foundation counts for more than its raw percentage suggests, because it makes each
remaining step faster and safer to build.

---

## Key decisions locked in (so we don't re-litigate them later)
- **Python 3.12** (not 3.14 — too new for these tools).
- **Local development uses delta-rs**; real Spark big-data jobs run on **Google Colab**,
  writing to an **AWS S3 bucket** (EC2 dropped to stay near-free; region Mumbai).
- **LLM is local Ollama** (`qwen2.5:3b`, sized for your 4GB GPU) with an optional
  cloud fallback switch. No Groq.
- **GitHub: one commit per step.**
- Secrets (patient salt, any API/AWS keys) live in environment variables, never in the
  repo.

## How to resume
1. Open `C:\mediguard-ai`, run `source .venv/Scripts/activate`.
2. Check health: `python -m pytest tests/ -v` → expect **29 passed**.
3. Next step to build: **feature engineering** (turn the check signals into a numeric
   row per claim for the ML model).
