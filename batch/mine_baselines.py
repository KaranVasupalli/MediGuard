"""Mine population baselines from the clean corpus.

Produces two reference tables the rules engine reads:
  1. diag_procedure_norms   - how often each (diagnosis, procedure) pair co-occurs
  2. procedure_cost_pctiles - typical cost range (p25/p50/p95) per procedure & state

Pure-python here so it runs locally and is easy to test. The SAME grouping logic
ports to Spark on EC2 for the full-population run — only the engine changes, not the
definitions.
"""
from collections import defaultdict


def _percentile(values: list[float], pct: float) -> float:
    """Linear-interpolation percentile. pct in 0..100."""
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * (pct / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def mine_diag_procedure_norms(rows: list[dict], min_support: int = 20) -> list[dict]:
    """For each (icd10, hbp) pair: co-occurrence share within the diagnosis + support."""
    pair_count: dict[tuple, int] = defaultdict(int)
    dx_count: dict[str, int] = defaultdict(int)
    for r in rows:
        dx, hbp = r.get("icd10_primary"), r.get("hbp_code")
        if not dx or not hbp or dx == "UNKNOWN":
            continue
        pair_count[(dx, hbp)] += 1
        dx_count[dx] += 1

    out = []
    for (dx, hbp), n in pair_count.items():
        share = n / dx_count[dx] if dx_count[dx] else 0.0
        band = "high" if n >= min_support else "low"
        out.append({
            "icd10_primary": dx, "hbp_code": hbp,
            "cooccurrence": round(share, 4), "support_n": n, "support_band": band,
        })
    return out


def mine_procedure_cost_pctiles(rows: list[dict]) -> list[dict]:
    """For each (hbp, state): p25/p50/p95 of billed amount. tier='ALL' (no tier data
    in the demo corpus; the grain widens to include tier when that field exists)."""
    buckets: dict[tuple, list[float]] = defaultdict(list)
    for r in rows:
        hbp, st = r.get("hbp_code"), r.get("provider_state")
        if not hbp:
            continue
        buckets[(hbp, st)].append(float(r["billed_inr"]))

    out = []
    for (hbp, st), vals in buckets.items():
        out.append({
            "hbp_code": hbp, "provider_state": st, "hospital_tier": "ALL",
            "p25_inr": round(_percentile(vals, 25), 2),
            "p50_inr": round(_percentile(vals, 50), 2),
            "p95_inr": round(_percentile(vals, 95), 2),
            "n": len(vals),
        })
    return out


def mine_all(rows: list[dict]) -> dict:
    return {
        "diag_procedure_norms": mine_diag_procedure_norms(rows),
        "procedure_cost_pctiles": mine_procedure_cost_pctiles(rows),
    }
