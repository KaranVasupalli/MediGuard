"""Tests for anomaly detection. Run: pytest tests/test_anomaly.py -v

The important test here is the last one: does the detector catch a fraud type that no
rule checks? An anomaly model that only re-finds what the rules already found is dead
weight, and the test suite should say so.
"""
import numpy as np
import pytest

from ml.anomaly import AnomalyDetector, build_profile, to_frame, PROFILE_FEATURES


def _lines(n=3, billed=2000.0, los=3, qty=1.0, code="HBP-BED-001"):
    return [{"line_no": i + 1, "billed_inr": billed, "hbp_code": f"{code}-{i}",
             "los_days": los, "quantity": qty} for i in range(n)]


def _cost(pct=0.0):
    return {"cost_findings": [{"pct_over_median": pct}] if pct else []}


def test_profile_has_all_expected_fields():
    p = build_profile(_lines(), _cost())
    assert set(p) == set(PROFILE_FEATURES)


def test_profile_excludes_rule_verdicts():
    """The whole point: anomaly must NOT look at rule findings, or it is redundant."""
    p = build_profile(_lines(), _cost())
    banned = {"n_findings", "r1_overbilling_count", "r2_dx_mismatch_count",
              "r3_stay_logic_count", "r4_cost_outlier_count", "total_excess_inr"}
    assert not (set(p) & banned)


def test_billed_per_day_handles_zero_stay():
    p = build_profile(_lines(los=0), _cost())
    assert p["billed_per_day"] == p["total_billed_inr"]      # no divide-by-zero


def test_max_line_share_is_a_fraction():
    lines = _lines(2)
    lines[0]["billed_inr"] = 9000.0
    lines[1]["billed_inr"] = 1000.0
    p = build_profile(lines, _cost())
    assert p["max_line_share"] == pytest.approx(0.9)


def test_score_requires_fit():
    with pytest.raises(RuntimeError):
        AnomalyDetector().score([build_profile(_lines(), _cost())])


def test_scores_are_normalised_0_to_1():
    profiles = [build_profile(_lines(n=3, billed=2000.0 + i * 10), _cost())
                for i in range(60)]
    det = AnomalyDetector(contamination=0.05).fit(profiles)
    s = det.score(profiles)
    assert s.min() >= 0.0 and s.max() <= 1.0


def test_outlier_scores_higher_than_normal_claims():
    normal = [build_profile(_lines(n=3, billed=2000.0), _cost()) for _ in range(80)]
    weird = build_profile(_lines(n=25, billed=50.0), _cost())     # many tiny lines
    det = AnomalyDetector(contamination=0.05).fit(normal + [weird])
    s = det.score(normal + [weird])
    assert s[-1] > np.median(s[:-1])


def test_explain_returns_z_scores():
    profiles = [build_profile(_lines(n=3, billed=2000.0 + i), _cost()) for i in range(50)]
    det = AnomalyDetector().fit(profiles)
    drivers = AnomalyDetector.explain(profiles[0], to_frame(profiles), top_k=3)
    assert len(drivers) == 3
    assert all("z_score" in d for d in drivers)


def test_catches_unbundling_that_rules_cannot_see():
    """Unbundling: many small valid lines. No rule fires; the SHAPE is the giveaway."""
    rng = np.random.default_rng(0)
    # a normal population must VARY — a zero-variance population teaches the model
    # nothing and is not a fair test
    normal = [
        build_profile(
            _lines(n=int(rng.integers(2, 5)),
                   billed=float(rng.normal(2000, 300)),
                   los=int(rng.integers(1, 7))),
            _cost())
        for _ in range(300)
    ]
    unbundled = [build_profile(_lines(n=16, billed=180.0, los=3), _cost())
                 for _ in range(20)]
    det = AnomalyDetector(contamination=0.05).fit(normal)
    caught = det.is_anomaly(unbundled).mean()
    assert caught >= 0.5, f"only caught {caught:.0%} of unbundled claims"
