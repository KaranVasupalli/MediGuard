"""Patient history: fraud that only appears ACROSS a patient's visits.

Every check so far sees one claim alone. Some fraud is invisible that way because each
individual claim is fine — it is the SEQUENCE that is wrong:

  * repeat testing      - the same expensive test billed again days later with no
                          clinical reason
  * duplicate billing   - the same procedure on the same service date appearing on two
                          different claims
  * provider shuttling  - one patient bouncing between several hospitals in a short
                          window (the patient-level shadow of a provider ring)
  * rapid readmission   - discharged and readmitted almost immediately, repeatedly

Each output is a plain count or gap in days, so a reviewer can check it by hand.
Like the graph, this needs the whole history at once, not a single claim.
"""
from collections import defaultdict
from datetime import date

# Procedures worth re-testing scrutiny. A general ward bed obviously recurs on every
# admission, so watching it would flag every legitimate readmission. Only items that
# are costly AND should not normally repeat within the window belong here.
EXPENSIVE_REPEAT_WATCH = {"HBP-ICU-002"}
REPEAT_WINDOW_DAYS = 30        # a repeat of the same costly item inside this window
MIN_REPEATS_TO_FLAG = 2        # one repeat can be clinically legitimate; two is odd
RAPID_READMIT_DAYS = 3         # discharged and back again this fast
SHUTTLE_WINDOW_DAYS = 45
SHUTTLE_MIN_PROVIDERS = 3


def _as_date(v) -> date | None:
    if isinstance(v, date):
        return v
    try:
        return date.fromisoformat(str(v)[:10])
    except (ValueError, TypeError):
        return None


def build_patient_history(rows: list[dict]) -> list[dict]:
    """One row per (patient, claim) with cross-claim context. Chronological."""
    # group lines -> claims, claims -> patients
    claims: dict[str, dict] = {}
    for r in rows:
        c = claims.setdefault(r["claim_id"], {
            "claim_id": r["claim_id"], "patient_hash": r["patient_hash"],
            "provider_id": r["provider_id"],
            "admission_date": _as_date(r.get("admission_date")),
            "discharge_date": _as_date(r.get("discharge_date")),
            "procedures": [], "billed_inr": 0.0,
        })
        c["procedures"].append((r.get("hbp_code"), _as_date(r.get("service_date"))))
        c["billed_inr"] += float(r.get("billed_inr") or 0.0)

    by_patient: dict[str, list[dict]] = defaultdict(list)
    for c in claims.values():
        by_patient[c["patient_hash"]].append(c)

    out = []
    repeat_running: dict[str, int] = {}
    for patient, cl in by_patient.items():
        cl.sort(key=lambda c: (c["admission_date"] or date.min, c["claim_id"]))
        seen_procs: list[tuple[str, date]] = []      # (hbp_code, service_date)
        seen_pairs: set[tuple] = set()               # (hbp_code, service_date)
        prev_discharge: date | None = None
        providers_seen: list[tuple[str, date]] = []

        for i, c in enumerate(cl):
            adm = c["admission_date"]
            flags: list[str] = []

            # repeat testing: same watched procedure again inside the window.
            # repeat_count accumulates across the patient's history, so a single
            # clinically plausible repeat does not flag; a pattern does.
            repeat_hits = 0
            for hbp, svc in c["procedures"]:
                if hbp not in EXPENSIVE_REPEAT_WATCH or svc is None:
                    continue
                for phbp, psvc in seen_procs:
                    if phbp == hbp and psvc and 0 <= (svc - psvc).days <= REPEAT_WINDOW_DAYS:
                        repeat_hits += 1
                        break
            prior_repeats = repeat_running.get(patient, 0) + repeat_hits
            repeat_running[patient] = prior_repeats
            if prior_repeats >= MIN_REPEATS_TO_FLAG:
                flags.append("repeat_costly_procedure")

            # duplicate billing: same procedure + same service date already billed
            dup_hits = sum(1 for hbp, svc in c["procedures"]
                           if hbp and svc and (hbp, svc) in seen_pairs)
            if dup_hits:
                flags.append("duplicate_service_date")

            # rapid readmission
            gap = (adm - prev_discharge).days if (adm and prev_discharge) else None
            if gap is not None and 0 <= gap <= RAPID_READMIT_DAYS:
                flags.append("rapid_readmission")

            # provider shuttling inside a window
            recent = [p for p, d in providers_seen
                      if adm and d and 0 <= (adm - d).days <= SHUTTLE_WINDOW_DAYS]
            distinct_recent = len(set(recent) | {c["provider_id"]})
            if distinct_recent >= SHUTTLE_MIN_PROVIDERS:
                flags.append("provider_shuttling")

            out.append({
                "patient_hash": patient,
                "claim_id": c["claim_id"],
                "provider_id": c["provider_id"],
                "prior_claims": i,
                "days_since_last_discharge": gap if gap is not None else -1,
                "repeat_costly_count": repeat_hits,
                "duplicate_service_count": dup_hits,
                "distinct_providers_recent": distinct_recent,
                "history_flags": ",".join(flags),
                "history_risk": round(min(
                    0.35 * min(repeat_hits, 2) / 2
                    + 0.35 * min(dup_hits, 2) / 2
                    + 0.15 * (1.0 if "rapid_readmission" in flags else 0.0)
                    + 0.15 * (1.0 if "provider_shuttling" in flags else 0.0), 1.0), 4),
            })

            # update history AFTER scoring this claim (no peeking at the future)
            seen_procs.extend([(h, s) for h, s in c["procedures"] if h and s])
            seen_pairs.update({(h, s) for h, s in c["procedures"] if h and s})
            providers_seen.append((c["provider_id"], adm))
            if c["discharge_date"]:
                prev_discharge = c["discharge_date"]

    return out


def summarise(history: list[dict]) -> dict:
    flagged = [h for h in history if h["history_flags"]]
    counts: dict[str, int] = defaultdict(int)
    for h in flagged:
        for f in h["history_flags"].split(","):
            counts[f] += 1
    return {"total": len(history), "flagged": len(flagged), "by_flag": dict(counts)}
