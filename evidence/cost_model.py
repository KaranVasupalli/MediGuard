"""Cost model: the last deterministic evidence computer before ML.

The rules engine asks "did they break the fixed HBP rate?" The cost model asks the
subtler question "is this price believable given everything similar we've seen?" —
so it can flag bills that stay under the cap but sit far above where comparable real
bills land.

For each line it estimates a fair price BAND from the mined cost percentiles
(procedure + state), then measures how far the billed amount sits above that band.
Every output is a plain number; no ML, no black box.
"""

# how far above the fair band counts as mild / strong (as a multiple of the band's p50)
_MILD = 0.15      # 15% above the upper fair bound
_STRONG = 0.50    # 50% above


def _fair_band(cost_row: dict) -> tuple[float, float, float]:
    """Fair band = (p25, p50, p95) of comparable real bills."""
    return cost_row["p25_inr"], cost_row["p50_inr"], cost_row["p95_inr"]


def score_line(line: dict, cost_idx: dict) -> dict | None:
    """Return a cost finding for one line, or None if there's nothing to say / no data."""
    hbp = line.get("hbp_code")
    state = line.get("provider_state")
    billed = float(line["billed_inr"])
    cost = cost_idx.get((hbp, state))
    if not cost or cost["n"] < 5:          # not enough comparable bills to judge
        return None

    p25, p50, p95 = _fair_band(cost)
    if p50 <= 0:
        return None

    # position of the bill relative to the fair band
    gap_over_p95 = billed - p95
    ratio_over_p50 = (billed - p50) / p50

    if billed <= p95:
        band_status = "within_fair_range"
        severity = "none"
    elif ratio_over_p50 >= _STRONG:
        band_status = "far_above_fair_range"
        severity = "high"
    elif ratio_over_p50 >= _MILD:
        band_status = "above_fair_range"
        severity = "medium"
    else:
        band_status = "slightly_above_fair_range"
        severity = "low"

    return {
        "line_no": line["line_no"], "hbp_code": hbp,
        "billed_inr": round(billed, 2),
        "fair_p25_inr": round(p25, 2), "fair_p50_inr": round(p50, 2),
        "fair_p95_inr": round(p95, 2),
        "gap_over_p95_inr": round(max(gap_over_p95, 0.0), 2),
        "pct_over_median": round(ratio_over_p50 * 100, 1),
        "band_status": band_status, "severity": severity,
        "reason": (f"billed INR {billed:,.0f}; fair range INR {p25:,.0f}-{p95:,.0f} "
                   f"(typical {p50:,.0f}); {ratio_over_p50*100:.0f}% over typical"),
    }


def score_claim(lines: list[dict], cost_idx: dict) -> dict:
    """Cost evidence for a whole claim: per-line findings + a summary."""
    findings = []
    total_gap = 0.0
    for ln in lines:
        f = score_line(ln, cost_idx)
        if f and f["severity"] != "none":
            findings.append(f)
            total_gap += f["gap_over_p95_inr"]

    worst = max((f["severity"] for f in findings),
                key=lambda s: {"low": 1, "medium": 2, "high": 3}.get(s, 0),
                default="none")
    return {
        "cost_findings": findings,
        "total_gap_over_p95_inr": round(total_gap, 2),
        "worst_severity": worst,
        "lines_above_fair_range": len(findings),
    }
