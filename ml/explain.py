"""Explain a model score with SHAP.

The score alone is not usable by a reviewer — "0.87" is not a reason. SHAP gives the
contribution of each feature to THIS claim's score, so the verdict can say
"driven by: total excess 3,200 (+0.31), impossible stay (+0.22)".

Because every feature was produced by a deterministic computer, each SHAP line traces
back to a real, checkable number — not a black-box abstraction.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
import shap

from ml.features import FEATURE_NAMES

MODEL_DIR = Path("./data/models")

# plain-language names so a reviewer never sees a raw variable name
READABLE = {
    "n_lines": "number of billed lines",
    "los_days": "length of stay",
    "total_billed_inr": "total billed",
    "avg_line_inr": "average line amount",
    "max_line_inr": "largest single line",
    "distinct_procedures": "distinct procedures",
    "r1_overbilling_count": "lines billed over the official rate",
    "r2_dx_mismatch_count": "treatments not matching the diagnosis",
    "r3_stay_logic_count": "more days billed than the stay",
    "r4_cost_outlier_count": "prices far above the population",
    "n_findings": "total rule findings",
    "n_high_severity": "high-severity findings",
    "total_excess_inr": "total excess over allowed",
    "excess_ratio": "share of the bill that is excess",
    "share_lines_flagged": "share of lines flagged",
    "cost_lines_above_band": "lines above the fair price range",
    "cost_gap_over_p95_inr": "amount above the fair range",
    "cost_gap_ratio": "share of bill above fair range",
    "max_pct_over_median": "worst % over typical price",
    "worst_cost_severity": "worst price severity",
    "n_unmapped_codes": "lines with unrecognised codes",
}


class FraudExplainer:
    def __init__(self, model_path: Path = MODEL_DIR / "fraud_model.txt"):
        self.booster = lgb.Booster(model_file=str(model_path))
        self.explainer = shap.TreeExplainer(self.booster)

    def score(self, feats: dict) -> float:
        """Fraud score 0-1 for one claim's feature dict."""
        X = pd.DataFrame([[float(feats[n]) for n in FEATURE_NAMES]], columns=FEATURE_NAMES)
        raw = self.booster.predict(X)[0]
        return float(raw)

    def explain(self, feats: dict, top_k: int = 5) -> dict:
        """Score + the top drivers, in plain language."""
        X = pd.DataFrame([[float(feats[n]) for n in FEATURE_NAMES]], columns=FEATURE_NAMES)
        sv = self.explainer.shap_values(X)
        vals = np.array(sv[1] if isinstance(sv, list) else sv).reshape(-1)

        order = np.argsort(-np.abs(vals))[:top_k]
        drivers = []
        for i in order:
            name = FEATURE_NAMES[i]
            drivers.append({
                "feature": name,
                "label": READABLE.get(name, name),
                "value": round(float(X.iloc[0, i]), 2),
                "contribution": round(float(vals[i]), 4),
                "direction": "increases" if vals[i] > 0 else "decreases",
            })
        return {"fraud_score": round(self.score(feats), 4), "top_drivers": drivers}

    @staticmethod
    def to_sentence(explanation: dict) -> str:
        up = [d for d in explanation["top_drivers"] if d["contribution"] > 0][:3]
        if not up:
            return "No signal materially increased the risk score."
        parts = [f"{d['label']} ({d['value']:g})" for d in up]
        return "Score driven mainly by " + ", ".join(parts) + "."
