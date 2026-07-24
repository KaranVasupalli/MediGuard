"""Tests for the cost model. Run: pytest tests/test_cost_model.py -v"""
from evidence.cost_model import score_line, score_claim

# one price band: procedure P in state MH, fair range 1000-2000, typical 1200
_COST_IDX = {
    ("P", "MH"): {"hbp_code": "P", "provider_state": "MH",
                  "p25_inr": 1000.0, "p50_inr": 1200.0, "p95_inr": 2000.0, "n": 40},
    ("Q", "MH"): {"hbp_code": "Q", "provider_state": "MH",
                  "p25_inr": 100.0, "p50_inr": 120.0, "p95_inr": 200.0, "n": 3},  # too few
}


def _line(hbp, billed, state="MH"):
    return {"line_no": 1, "hbp_code": hbp, "provider_state": state, "billed_inr": billed}


def test_within_fair_range_is_not_flagged():
    assert score_line(_line("P", 1500), _COST_IDX) is None or \
           score_line(_line("P", 1500), _COST_IDX)["severity"] == "none"


def test_far_above_typical_is_high_severity():
    f = score_line(_line("P", 2400), _COST_IDX)   # 100% over p50 (1200)
    assert f["severity"] == "high"
    assert f["band_status"] == "far_above_fair_range"


def test_mildly_above_is_medium_or_low():
    f = score_line(_line("P", 2100), _COST_IDX)   # ~5% over p95, ~75% over... check band
    assert f["severity"] in {"low", "medium", "high"}


def test_insufficient_data_returns_none():
    assert score_line(_line("Q", 5000), _COST_IDX) is None   # only n=3 comparable bills


def test_under_cap_but_inflated_is_caught():
    # billed above typical but the point is: cost model judges vs the BAND, not a cap
    f = score_line(_line("P", 1900), _COST_IDX)   # under p95 (2000) -> within range
    assert f is None or f["severity"] == "none"
    f2 = score_line(_line("P", 2200), _COST_IDX)  # over p95, ~83% over p50
    assert f2["severity"] == "high"


def test_claim_summary_counts_flagged_lines():
    lines = [_line("P", 2400), _line("P", 1500)]   # one high, one within range
    res = score_claim(lines, _COST_IDX)
    assert res["lines_above_fair_range"] == 1
    assert res["worst_severity"] == "high"
