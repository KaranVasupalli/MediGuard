"""Unsupervised anomaly detection: catch odd claims that break no written rule.

CRITICAL DESIGN POINT
The rules engine already flags what we knew to look for. If we fed the anomaly model
the SAME rule outputs, it would only re-find the same claims and add nothing. So this
model deliberately looks at a DIFFERENT view of the claim: its behavioural shape —
billing intensity, how concentrated the money is, procedure mix, price position — none
of which is a rule verdict.

That way it can flag a claim that breaks no rule but simply does not look like any
normal claim. Those are the patterns nobody wrote a rule for yet.

It is unsupervised: it never sees the fraud labels during training, exactly as in
production where labels do not exist at scoring time.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

# Behavioural descriptors — deliberately NOT rule findings.
PROFILE_FEATURES = [
    "n_lines",
    "los_days",
    "total_billed_inr",
    "billed_per_day",          # spend intensity of the stay
    "avg_line_inr",
    "max_line_share",          # how concentrated the money is in one line
    "procedure_diversity",     # distinct procedures per line
    "avg_qty_per_line",
    "max_pct_over_median",     # price position vs peers (continuous, not a verdict)
]


def build_profile(lines: list[dict], cost_result: dict) -> dict:
    """Behavioural profile of one claim."""
    n = len(lines)
    billed = [float(l["billed_inr"]) for l in lines]
    total = sum(billed)
    los = float(lines[0].get("los_days") or 0) if lines else 0.0
    qtys = [float(l.get("quantity") or 0) for l in lines]
    procs = {l.get("hbp_code") for l in lines if l.get("hbp_code")}
    cost_findings = cost_result.get("cost_findings", [])

    return {
        "n_lines": float(n),
        "los_days": los,
        "total_billed_inr": round(total, 2),
        "billed_per_day": round(total / los, 2) if los > 0 else round(total, 2),
        "avg_line_inr": round(total / n, 2) if n else 0.0,
        "max_line_share": round(max(billed) / total, 4) if total > 0 else 0.0,
        "procedure_diversity": round(len(procs) / n, 4) if n else 0.0,
        "avg_qty_per_line": round(sum(qtys) / n, 2) if n else 0.0,
        "max_pct_over_median": round(
            max((f.get("pct_over_median", 0.0) for f in cost_findings), default=0.0), 1),
    }


def to_frame(profiles: list[dict]) -> pd.DataFrame:
    return pd.DataFrame([[p[f] for f in PROFILE_FEATURES] for p in profiles],
                        columns=PROFILE_FEATURES)


class AnomalyDetector:
    """Isolation Forest over behavioural profiles."""

    def __init__(self, contamination: float = 0.05, random_state: int = 42):
        self.model = IsolationForest(
            n_estimators=200, contamination=contamination,
            random_state=random_state, n_jobs=-1,
        )
        self.fitted = False

    def fit(self, profiles: list[dict]):
        """Unsupervised — labels are never passed in."""
        self.model.fit(to_frame(profiles))
        self.fitted = True
        return self

    def score(self, profiles: list[dict]) -> np.ndarray:
        """Anomaly score in 0-1; higher = more unusual."""
        if not self.fitted:
            raise RuntimeError("call fit() first")
        raw = self.model.score_samples(to_frame(profiles))   # higher = more normal
        lo, hi = raw.min(), raw.max()
        if hi - lo < 1e-12:
            return np.zeros(len(raw))
        return (hi - raw) / (hi - lo)                        # invert + normalise

    def is_anomaly(self, profiles: list[dict]) -> np.ndarray:
        return (self.model.predict(to_frame(profiles)) == -1).astype(int)

    @staticmethod
    def explain(profile: dict, population: pd.DataFrame, top_k: int = 3) -> list[dict]:
        """Which descriptors are most unusual vs the population (in std deviations)."""
        out = []
        for f in PROFILE_FEATURES:
            col = population[f]
            sd = col.std()
            if sd and sd > 1e-9:
                z = (profile[f] - col.mean()) / sd
                out.append({"feature": f, "value": profile[f], "z_score": round(float(z), 2)})
        out.sort(key=lambda d: -abs(d["z_score"]))
        return out[:top_k]
