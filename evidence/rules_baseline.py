"""Rules-and-baseline engine: the first component that hunts for fraud.

Runs four deterministic checks on a claim's lines. Every finding carries a plain
reason and, where money is involved, an exact rupee figure. These numbers are
computed BEFORE any ML/LLM — they are the ground truth the later layers explain.

Checks:
  R1 overbilling            billed above the official HBP package rate
  R2 diagnosis-procedure fit procedure never/rarely seen with this diagnosis
  R3 stay logic             per-day charges (ICU/bed) exceed the length of stay
  R4 cost outlier           billed above the population p95 for this procedure+state
"""

# HBP codes charged per day of stay (days billed should not exceed length of stay).
PER_DAY_CODES = {"HBP-ICU-002", "HBP-BED-001"}
TOLERANCE_INR = 1.0        # ignore sub-rupee rounding noise
# Materiality: a real auditor does not chase a 2% overage. Only flag an overcharge
# that is both meaningful in rupees AND a meaningful share of the allowed amount.
MATERIAL_PCT = 0.10        # 10% over the allowed amount
MATERIAL_INR = 500.0       # or at least this many rupees
# Cost outliers: p95 is exceeded by 5% of bills BY DEFINITION, so require a margin
# above it before calling something an outlier.
OUTLIER_MARGIN = 1.15      # 15% above p95


def build_indexes(norms: list[dict], pctiles: list[dict]):
    norms_idx = {(r["icd10_primary"], r["hbp_code"]): r for r in norms}
    seen_dx = {r["icd10_primary"] for r in norms}
    cost_idx = {(r["hbp_code"], r["provider_state"]): r for r in pctiles}
    return {"norms": norms_idx, "seen_dx": seen_dx, "cost": cost_idx}


def evaluate_claim(lines: list[dict], idx: dict) -> dict:
    findings: list[dict] = []
    total_excess = 0.0

    for ln in lines:
        line_no = ln["line_no"]
        hbp = ln.get("hbp_code")
        billed = float(ln["billed_inr"])
        qty = float(ln["quantity"])
        rate = ln.get("hbp_package_rate_inr")
        dx = ln.get("icd10_primary")

        # R1 overbilling (only if material — see MATERIAL_* above)
        if rate is not None:
            allowed = rate * qty
            excess = billed - allowed
            material = excess > TOLERANCE_INR and (
                excess >= MATERIAL_INR or (allowed > 0 and excess / allowed >= MATERIAL_PCT)
            )
            if material:
                total_excess += excess
                findings.append({
                    "rule": "R1_overbilling", "line_no": line_no, "hbp_code": hbp,
                    "billed_inr": round(billed, 2), "allowed_inr": round(allowed, 2),
                    "excess_inr": round(excess, 2),
                    "severity": "high" if excess > allowed else "medium",
                    "reason": f"billed INR {billed:,.0f} vs allowed INR {allowed:,.0f} "
                              f"(HBP rate {rate:,.0f} x qty {qty:g})",
                })

        # R2 diagnosis-procedure fit (needs baseline)
        if dx and hbp and dx in idx["seen_dx"]:
            pair = idx["norms"].get((dx, hbp))
            if pair is None:
                findings.append({
                    "rule": "R2_dx_procedure_fit", "line_no": line_no, "hbp_code": hbp,
                    "severity": "high",
                    "reason": f"procedure {hbp} never seen with diagnosis {dx} "
                              f"in the population",
                })
            elif pair["support_band"] == "low":
                findings.append({
                    "rule": "R2_dx_procedure_fit", "line_no": line_no, "hbp_code": hbp,
                    "severity": "low",
                    "reason": f"procedure {hbp} rarely seen with diagnosis {dx} "
                              f"(support n={pair['support_n']})",
                })

        # R3 stay logic
        if hbp in PER_DAY_CODES and qty > ln.get("los_days", qty):
            findings.append({
                "rule": "R3_stay_logic", "line_no": line_no, "hbp_code": hbp,
                "severity": "high",
                "reason": f"{qty:g} days billed but length of stay is "
                          f"{ln.get('los_days')} days",
            })

        # R4 cost outlier vs population p95 (needs baseline)
        cost = idx["cost"].get((hbp, ln.get("provider_state")))
        if cost and cost["n"] >= 5 and cost["p95_inr"] > 0 \
                and billed > cost["p95_inr"] * OUTLIER_MARGIN:
            findings.append({
                "rule": "R4_cost_outlier", "line_no": line_no, "hbp_code": hbp,
                "billed_inr": round(billed, 2), "p95_inr": cost["p95_inr"],
                "severity": "medium",
                "reason": f"billed INR {billed:,.0f} above population p95 "
                          f"INR {cost['p95_inr']:,.0f}",
            })

    return {"findings": findings, "total_excess_inr": round(total_excess, 2),
            "rule_hits": sorted({f['rule'] for f in findings})}
