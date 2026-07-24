"""Fake RAW bills in 'Hospital A' format (NOT canonical): different column names,
foreign SNOMED/CPT codes, patient names present, INR amounts, and some broken rows
so we can watch the ingestion gate quarantine them.

This is the messy input the gate must clean. Real hospital exports replace it later.
"""
import random
from datetime import date, timedelta

random.seed(11)

STATES = ["MH", "KA", "TN", "UP", "GJ"]
NAMES = ["Rahul Sharma", "Priya Nair", "Amit Patel", "Sana Khan", "Vikram Rao"]
# (item desc, SNOMED dx, CPT proc, base rate INR)
CATALOG = [
    ("General ward bed", "25374005", "99221", 2000),
    ("ICU care",         "25374005", "99291", 4500),
    ("CBC test",         "44054006", "85025", 180),
    ("IV fluids",        "13645005", "96360", 280),
    ("Doctor visit",     "22298006", "99231", 900),
]


def generate_raw(n_claims: int = 30) -> list[dict]:
    rows: list[dict] = []
    for c in range(n_claims):
        adm = date(2026, 7, random.randint(1, 20))
        los = random.randint(1, 5)
        dis = adm + timedelta(days=los)
        hosp = f"H{random.randint(1, 6):02d}"
        state = random.choice(STATES)
        pt_name = random.choice(NAMES)
        pt_uid = f"UID{random.randint(10000, 99999)}"
        for ln, (desc, dx, cpt, rate) in enumerate(
            random.sample(CATALOG, k=random.randint(3, 5)), start=1
        ):
            qty = random.randint(1, los + 2)
            amount = round(rate * qty * random.uniform(0.9, 1.7), 2)
            row = {
                "BILL_NO": f"CLM-2026-{c:06d}", "LN": ln,
                "PT_NAME": pt_name, "PT_UID": pt_uid,
                "HOSP_ID": hosp, "HOSP_NAME": f"Hospital {hosp}", "ST": state,
                "ADMIT": adm.isoformat(), "DISCH": dis.isoformat(),
                "SVC_DT": (adm + timedelta(days=random.randint(0, los))).isoformat(),
                "DX_SNOMED": dx, "PROC_CPT": cpt, "ITEM_DESC": desc,
                "DEPT": "General Medicine", "QTY": qty,
                "RATE_INR": rate, "AMOUNT_INR": amount,
            }
            rows.append(row)

    # --- inject deliberately broken rows to exercise the gate ---
    rows.append({**rows[0], "BILL_NO": "CLM-2026-BAD01", "DISCH": "2026-07-01",
                 "ADMIT": "2026-07-10"})                       # discharge before admit
    rows.append({**rows[1], "BILL_NO": "CLM-2026-BAD02", "QTY": -3})   # negative qty
    rows.append({**rows[2], "BILL_NO": "CLM-2026-BAD03",
                 "DX_SNOMED": "00000000", "PROC_CPT": "00000"})   # unmappable codes
    rows.append({**rows[3], "BILL_NO": "CLM-2026-BAD04", "ADMIT": "not-a-date"})  # bad date
    random.shuffle(rows)
    return rows


if __name__ == "__main__":
    r = generate_raw()
    print(f"generated {len(r)} raw lines; sample columns: {list(r[0].keys())}")
