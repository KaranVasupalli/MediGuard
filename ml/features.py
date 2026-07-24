"""Feature engineering: the bridge from deterministic evidence into ML.

Takes a claim's lines plus the findings from the rules engine and cost model, and
packs them into ONE fixed row of numbers per claim.

Three design rules, all deliberate:
  1. FEATURE_NAMES is a frozen, ordered contract. A model trained on this order must
     be scored on this order, so the order lives in one place and never varies.
  2. Ratios matter more than raw rupees. INR 5,000 excess on a INR 6,000 bill is far
     worse than the same excess on a INR 500,000 bill, so we include normalised
     features, not just totals.
  3. Every feature traces back to a number some deterministic computer already
     produced. Nothing here is invented, so SHAP explanations later stay readable
     and every model input can be justified to a human reviewer.
"""

# Frozen, ordered feature contract. Adding a feature = append to the END and bump
# FEATURE_VERSION, so previously trained models stay interpretable.
FEATURE_NAMES = [
    # --- claim shape ---
    "n_lines",
    "los_days",
    "total_billed_inr",
    "avg_line_inr",
    "max_line_inr",
    "distinct_procedures",
    # --- rules engine ---
    "r1_overbilling_count",
    "r2_dx_mismatch_count",
    "r3_stay_logic_count",
    "r4_cost_outlier_count",
    "n_findings",
    "n_high_severity",
    "total_excess_inr",
    "excess_ratio",              # excess / billed  (normalised)
    "share_lines_flagged",       # flagged lines / all lines
    # --- cost model ---
    "cost_lines_above_band",
    "cost_gap_over_p95_inr",
    "cost_gap_ratio",            # gap / billed  (normalised)
    "max_pct_over_median",
    "worst_cost_severity",       # 0 none, 1 low, 2 medium, 3 high
    # --- data quality ---
    "n_unmapped_codes",
]
FEATURE_VERSION = "1.0.0"

_SEV_ORD = {"none": 0, "low": 1, "medium": 2, "high": 3}


def _safe_ratio(num: float, den: float) -> float:
    return round(num / den, 4) if den else 0.0


def build_features(lines: list[dict], rules_result: dict, cost_result: dict) -> dict:
    """Return {feature_name: value} for one claim. Keys always match FEATURE_NAMES."""
    n_lines = len(lines)
    billed = [float(l["billed_inr"]) for l in lines]
    total_billed = sum(billed)

    findings = rules_result.get("findings", [])
    by_rule: dict[str, int] = {}
    for f in findings:
        by_rule[f["rule"]] = by_rule.get(f["rule"], 0) + 1

    flagged_lines = {f["line_no"] for f in findings}
    cost_findings = cost_result.get("cost_findings", [])

    n_unmapped = sum(
        1 for l in lines
        if "unmapped" in str(l.get("quality_flags") or "")
    )

    feats = {
        # claim shape
        "n_lines": float(n_lines),
        "los_days": float(lines[0].get("los_days") or 0) if lines else 0.0,
        "total_billed_inr": round(total_billed, 2),
        "avg_line_inr": round(total_billed / n_lines, 2) if n_lines else 0.0,
        "max_line_inr": round(max(billed), 2) if billed else 0.0,
        "distinct_procedures": float(len({l.get("hbp_code") for l in lines if l.get("hbp_code")})),
        # rules
        "r1_overbilling_count": float(by_rule.get("R1_overbilling", 0)),
        "r2_dx_mismatch_count": float(by_rule.get("R2_dx_procedure_fit", 0)),
        "r3_stay_logic_count": float(by_rule.get("R3_stay_logic", 0)),
        "r4_cost_outlier_count": float(by_rule.get("R4_cost_outlier", 0)),
        "n_findings": float(len(findings)),
        "n_high_severity": float(sum(1 for f in findings if f.get("severity") == "high")),
        "total_excess_inr": float(rules_result.get("total_excess_inr", 0.0)),
        "excess_ratio": _safe_ratio(float(rules_result.get("total_excess_inr", 0.0)),
                                    total_billed),
        "share_lines_flagged": _safe_ratio(len(flagged_lines), n_lines),
        # cost model
        "cost_lines_above_band": float(cost_result.get("lines_above_fair_range", 0)),
        "cost_gap_over_p95_inr": float(cost_result.get("total_gap_over_p95_inr", 0.0)),
        "cost_gap_ratio": _safe_ratio(float(cost_result.get("total_gap_over_p95_inr", 0.0)),
                                      total_billed),
        "max_pct_over_median": round(
            max((f.get("pct_over_median", 0.0) for f in cost_findings), default=0.0), 1),
        "worst_cost_severity": float(_SEV_ORD.get(cost_result.get("worst_severity", "none"), 0)),
        # data quality
        "n_unmapped_codes": float(n_unmapped),
    }

    # guarantee the contract holds
    assert set(feats) == set(FEATURE_NAMES), "feature set drifted from FEATURE_NAMES"
    return feats


def to_vector(feats: dict) -> list[float]:
    """Features as a plain ordered list — the exact order a model expects."""
    return [float(feats[name]) for name in FEATURE_NAMES]


def build_feature_row(claim_id: str, lines: list[dict],
                      rules_result: dict, cost_result: dict) -> dict:
    """A storable row: claim id + all features + the version that produced them."""
    feats = build_features(lines, rules_result, cost_result)
    return {"claim_id": claim_id, "feature_version": FEATURE_VERSION, **feats}
