"""Realistic RAW hospital bills, in 'Hospital A' format, with RARE labelled fraud.

Why this replaces the old generator: the first version billed everything at
0.9-1.7x the official rate, so ~97% of claims looked overbilled. A model trained on
that learns "everything is fraud" and is useless. Real fraud is rare.

What this produces instead:
  * ~92% honest claims  - procedures that fit the diagnosis, billed close to the
    official HBP rate with small legitimate variation. A few honest claims drift
    slightly over rate on purpose, so the rules produce natural FALSE POSITIVES.
  * ~8% fraudulent claims - one or two injected patterns, some blatant and some
    deliberately SUBTLE, so no single rule catches them all.

Labels are returned SEPARATELY, never inside the bill. Real labels come from audit
outcomes, not from the claim, and the canonical 27-column schema is frozen with no
label field. Keeping them apart also prevents the label leaking into features.
"""
import random
from datetime import date, timedelta

# Clinical map: each diagnosis has procedures that legitimately go with it.
# (SNOMED dx code) -> list of (item desc, CPT proc code, base INR rate)
CLINICAL = {
    "25374005": [  # infectious gastroenteritis
        ("General ward bed", "99221", 2000),
        ("IV fluids", "96360", 280),
        ("CBC test", "85025", 180),
        ("Doctor visit", "99231", 900),
    ],
    "44054006": [  # type 2 diabetes
        ("General ward bed", "99221", 2000),
        ("CBC test", "85025", 180),
        ("Doctor visit", "99231", 900),
    ],
    "13645005": [  # COPD
        ("General ward bed", "99221", 2000),
        ("ICU care", "99291", 4500),
        ("Doctor visit", "99231", 900),
    ],
    "22298006": [  # acute myocardial infarction
        ("ICU care", "99291", 4500),
        ("General ward bed", "99221", 2000),
        ("Doctor visit", "99231", 900),
        ("CBC test", "85025", 180),
    ],
}
ALL_PROCS = {p[1]: p for procs in CLINICAL.values() for p in procs}

STATES = ["MH", "KA", "TN", "UP", "GJ"]
NAMES = ["Rahul Sharma", "Priya Nair", "Amit Patel", "Sana Khan", "Vikram Rao"]
PER_DAY_CPT = {"99291", "99221"}          # billed per day of stay

FRAUD_PATTERNS = ["upcoding", "phantom_service", "impossible_stay", "quantity_inflation"]


def _line(claim_id, ln, pt_name, pt_uid, hosp, state, adm, dis, svc, dx,
          desc, cpt, qty, rate, billed):
    return {
        "BILL_NO": claim_id, "LN": ln, "PT_NAME": pt_name, "PT_UID": pt_uid,
        "HOSP_ID": hosp, "HOSP_NAME": f"Hospital {hosp}", "ST": state,
        "ADMIT": adm.isoformat(), "DISCH": dis.isoformat(), "SVC_DT": svc.isoformat(),
        "DX_SNOMED": dx, "PROC_CPT": cpt, "ITEM_DESC": desc,
        "DEPT": "General Medicine", "QTY": qty, "RATE_INR": rate,
        "AMOUNT_INR": round(billed, 2),
    }


def generate(n_claims: int = 2000, fraud_rate: float = 0.08, seed: int = 42,
             n_patients: int | None = None, ring_providers: int = 3,
             ring_patients: int = 45, ring_fraud_rate: float = 0.45,
             n_hospitals: int = 40):
    """Return (raw_rows, labels, truth).

    Patients come from a POOL so they recur across claims — without recurrence there
    are no shared-patient links and no provider graph to analyse.

    Provider count and pool size matter: with too few hospitals and too small a pool,
    EVERY pair of hospitals shares many patients by chance and an injected ring cannot
    stand out. Real provider networks are sparse, so the defaults keep honest sharing
    rare and a ring genuinely distinctive.

    A small COLLUDING RING is injected: `ring_providers` hospitals that shuttle the
    same `ring_patients` between them and bill fraudulently far more often. `truth`
    records who is in the ring so the graph can be evaluated honestly.
    """
    rng = random.Random(seed)
    rows, labels = [], []

    all_hosp = [f"H{i:02d}" for i in range(1, n_hospitals + 1)]
    ring_hosp = all_hosp[:ring_providers]              # the colluding cluster
    honest_hosp = all_hosp[ring_providers:]

    n_patients = n_patients or max(200, n_claims // 2)
    pool = [f"UID{10000 + i}" for i in range(n_patients)]
    ring_pool = pool[:ring_patients]                   # patients shuttled around

    for c in range(n_claims):
        cid = f"CLM-2026-{c:06d}"

        # ~12% of claims belong to the ring
        in_ring = rng.random() < 0.12
        if in_ring:
            hosp = rng.choice(ring_hosp)
            pt_uid = rng.choice(ring_pool)
            claim_fraud_rate = ring_fraud_rate
        else:
            hosp = rng.choice(honest_hosp)
            pt_uid = rng.choice(pool)
            claim_fraud_rate = fraud_rate

        dx = rng.choice(list(CLINICAL))
        adm = date(2026, rng.randint(1, 7), rng.randint(1, 28))
        los = rng.randint(1, 7)
        dis = adm + timedelta(days=los)
        state = rng.choice(STATES)
        pt_name = rng.choice(NAMES)

        is_fraud = rng.random() < claim_fraud_rate
        patterns = rng.sample(FRAUD_PATTERNS, k=rng.randint(1, 2)) if is_fraud else []
        applied: set[str] = set()      # patterns that ACTUALLY landed on a line

        legit = CLINICAL[dx]
        chosen = rng.sample(legit, k=min(len(legit), rng.randint(2, 4)))
        claim_lines, ln = [], 1

        for desc, cpt, rate in chosen:
            # quantity: per-day items track the stay, others are small counts
            qty = los if cpt in PER_DAY_CPT else rng.randint(1, 3)

            if is_fraud and "quantity_inflation" in patterns and rng.random() < 0.5:
                qty = int(qty * rng.uniform(2.0, 3.5)) + 1
                applied.add("quantity_inflation")
            if is_fraud and "impossible_stay" in patterns and cpt in PER_DAY_CPT:
                qty = los + rng.randint(2, 5)          # more days than the stay
                applied.add("impossible_stay")

            # price: honest bills sit at/near the rate; a few drift slightly over
            if is_fraud and "upcoding" in patterns and rng.random() < 0.6:
                # subtle half the time, blatant the other half
                mult = rng.uniform(1.25, 1.6) if rng.random() < 0.5 else rng.uniform(1.8, 3.0)
                applied.add("upcoding")
            else:
                mult = rng.uniform(0.97, 1.02)          # legitimate variation
                if rng.random() < 0.06:                 # honest but slightly over -> FP
                    mult = rng.uniform(1.03, 1.12)

            billed = rate * qty * mult
            svc = adm + timedelta(days=rng.randint(0, los))
            claim_lines.append(_line(cid, ln, pt_name, pt_uid, hosp, state,
                                     adm, dis, svc, dx, desc, cpt, qty, rate, billed))
            ln += 1

        # phantom service: a procedure that does NOT belong to this diagnosis
        if is_fraud and "phantom_service" in patterns:
            legit_cpts = {p[1] for p in legit}
            outside = [p for cpt, p in ALL_PROCS.items() if cpt not in legit_cpts]
            if outside:
                desc, cpt, rate = rng.choice(outside)
                qty = rng.randint(1, 2)
                svc = adm + timedelta(days=rng.randint(0, los))
                claim_lines.append(_line(cid, ln, pt_name, pt_uid, hosp, state, adm, dis,
                                         svc, dx, desc, cpt, qty, rate,
                                         rate * qty * rng.uniform(0.98, 1.05)))
                applied.add("phantom_service")

        rows.extend(claim_lines)
        # A claim is fraud ONLY if a pattern actually landed on a line. Without this,
        # a claim can be labelled fraud while carrying no fraudulent line at all
        # (e.g. 'impossible_stay' selected for a claim with no per-day items), which
        # is label noise and makes every downstream metric meaningless.
        truly_fraud = 1 if applied else 0
        labels.append({"claim_id": cid, "is_fraud": truly_fraud,
                       "fraud_patterns": ",".join(sorted(applied))})

    truth = {"ring_providers": ring_hosp, "honest_providers": honest_hosp,
             "ring_patient_ids": ring_pool, "n_patients": n_patients}
    return rows, labels, truth


if __name__ == "__main__":
    r, lab, truth = generate()
    n_f = sum(l["is_fraud"] for l in lab)
    print(f"{len(r)} raw lines, {len(lab)} claims, {n_f} fraudulent "
          f"({n_f/len(lab)*100:.1f}%)")
    print("ring providers:", truth["ring_providers"])


def generate_history_fraud(n_patients: int = 30, seed: int = 77):
    """Cross-visit fraud: each claim is INDIVIDUALLY clean, the SEQUENCE is not.

    One patient is admitted repeatedly in a short window, each stay billing the same
    costly per-day item at the correct rate for a plausible number of days, and some
    service dates are billed twice across different claims.

    Every single claim passes every per-claim check: codes valid, matching diagnosis,
    billed at rate, days within stay. Only the patient's HISTORY reveals it.

    Used to test whether the patient-history component earns its place.
    """
    rng = random.Random(seed)
    rows, labels = [], []
    dx = "13645005"                                    # COPD -> ICU is legitimate
    desc, cpt, rate = ("ICU care", "99291", 4500)

    for p in range(n_patients):
        pt_uid = f"UIDHX{p:04d}"
        pt_name = rng.choice(NAMES)
        hosp = f"H{rng.randint(1, 40):02d}"
        state = rng.choice(STATES)
        base = date(2026, rng.randint(2, 5), rng.randint(1, 20))

        for visit in range(rng.randint(3, 4)):         # repeated admissions, days apart
            adm = base + timedelta(days=visit * rng.randint(2, 6))
            los = rng.randint(1, 3)
            dis = adm + timedelta(days=los)
            cid = f"CLM-HIST-{p:04d}-{visit}"
            for ln in range(1, 3):
                # duplicate a service date across visits on purpose
                svc = adm if ln == 1 else base
                rows.append(_line(cid, ln, pt_name, pt_uid, hosp, state, adm, dis, svc,
                                  dx, desc, cpt, los, rate, rate * los * rng.uniform(0.98, 1.01)))
            labels.append({"claim_id": cid, "is_fraud": 1,
                           "fraud_patterns": "repeat_costly_visits"})
    return rows, labels


def generate_novel_fraud(n_claims: int = 40, seed: int = 99):
    """'Unbundling': a fraud type NO rule in this system checks for.

    One treatment episode is split into many small legitimate-looking lines. Every
    line uses a valid code, matches the diagnosis, and is billed AT the official rate,
    so R1 (overbilling), R2 (diagnosis fit), R3 (stay logic) and R4 (cost outlier) all
    stay silent. Only the claim's SHAPE is abnormal — many more lines than normal,
    each unusually small.

    Used to test whether anomaly detection earns its place: can it catch fraud that
    was never written into a rule?
    """
    rng = random.Random(seed)
    rows, labels = [], []
    for c in range(n_claims):
        cid = f"CLM-NOVEL-{c:04d}"
        dx = rng.choice(list(CLINICAL))
        adm = date(2026, rng.randint(1, 7), rng.randint(1, 28))
        los = rng.randint(1, 4)
        dis = adm + timedelta(days=los)
        hosp = f"H{rng.randint(1, 12):02d}"
        state = rng.choice(STATES)
        pt_name, pt_uid = rng.choice(NAMES), f"UID{rng.randint(10000, 99999)}"

        legit = [p for p in CLINICAL[dx] if p[1] not in PER_DAY_CPT] or CLINICAL[dx]
        ln = 1
        for _ in range(rng.randint(12, 20)):          # abnormally many small lines
            desc, cpt, rate = rng.choice(legit)
            svc = adm + timedelta(days=rng.randint(0, los))
            rows.append(_line(cid, ln, pt_name, pt_uid, hosp, state, adm, dis, svc,
                              dx, desc, cpt, 1, rate, rate * rng.uniform(0.98, 1.01)))
            ln += 1
        labels.append({"claim_id": cid, "is_fraud": 1, "fraud_patterns": "unbundling"})
    return rows, labels
