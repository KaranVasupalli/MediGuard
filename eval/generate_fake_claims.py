"""Walking-skeleton data: 100 fake but schema-valid claim-lines.
No fraud logic — just enough real rows to push through the pipe and prove wiring.
Real corpus generation (Synthea + SynPUF) replaces this later."""
import random
from datetime import date, datetime, timedelta

from data_quality.schema_validator import ClaimLine

random.seed(7)

STATES = ["MH", "KA", "TN", "UP", "GJ"]
PROC = [
    # (line_desc, hbp_code, hbp_desc, package_rate)
    ("General ward bed charge", "HBP-BED-001", "General ward per day", 2000),
    ("ICU care per day",        "HBP-ICU-002", "Critical care ICU/day", 4500),
    ("CBC with differential",   "HBP-LAB-014", "Complete Blood Count",  180),
    ("IV fluid therapy",        "HBP-MED-030", "IV fluid therapy",      280),
    ("Physician daily visit",   "HBP-CON-005", "Physician consult",     900),
]


def make_claims(n_claims: int = 20) -> list[dict]:
    """Each claim has a few lines; ~100 lines total by default."""
    rows: list[dict] = []
    for c in range(n_claims):
        adm = date(2026, 7, random.randint(1, 20))
        los = random.randint(1, 5)
        dis = adm + timedelta(days=los)
        pid = f"P{random.randint(1, 8):03d}"
        state = random.choice(STATES)
        for line_no, (desc, hbp, hbp_desc, rate) in enumerate(
            random.sample(PROC, k=random.randint(3, 5)), start=1
        ):
            qty = random.randint(1, los + 2)
            billed = round(rate * qty * random.uniform(0.9, 1.6), 2)
            row = ClaimLine(
                claim_id=f"CLM-2026-{c:06d}",
                line_no=line_no,
                patient_hash=f"h{random.randint(1000, 9999)}",
                provider_id=pid,
                provider_name=f"Hospital {pid}",
                provider_state=state,
                admission_date=adm,
                discharge_date=dis,
                service_date=adm + timedelta(days=random.randint(0, los)),
                los_days=los,
                icd10_primary="A09",
                icd10_desc="Infectious gastroenteritis",
                snomed_src="25374005",
                cpt_code=None,
                hbp_code=hbp,
                hbp_desc=hbp_desc,
                line_desc=desc,
                department="General Medicine",
                quantity=float(qty),
                unit_price_inr=float(rate),
                billed_inr=billed,
                hbp_package_rate_inr=float(rate),
                source_system="fake_skeleton",
                ingest_ts=datetime.now(),
                quality_flags="",
                claim_year=2026,
                claim_month=7,
            )
            rows.append(row.model_dump())
    return rows


if __name__ == "__main__":
    data = make_claims()
    print(f"generated {len(data)} claim-lines across "
          f"{len({r['claim_id'] for r in data})} claims")
