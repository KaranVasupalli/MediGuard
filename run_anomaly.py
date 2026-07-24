"""Anomaly-detection step.

The evaluation that matters is NOT "does it find fraud" — the rules already find most
fraud. It is: does it find fraud the RULES MISSED? That is the only reason to add an
unsupervised model on top of a rules engine. This script measures exactly that, and
reports honestly if the answer is no.
"""
import numpy as np
import pandas as pd
from deltalake import DeltaTable

from config.spark_config import load_config
from batch.mine_baselines import mine_all
from evidence.rules_baseline import build_indexes, evaluate_claim
from evidence.cost_model import score_claim
from ml.anomaly import AnomalyDetector, build_profile, to_frame, PROFILE_FEATURES


def main():
    cfg = load_config()
    ref = cfg["paths"]["reference"]

    print("1) loading corpus + labels ...")
    rows = DeltaTable(cfg["paths"]["corpus"]).to_pyarrow_table().to_pylist()
    labels = {l["claim_id"]: l for l in
              DeltaTable(f"{ref}/claim_labels").to_pyarrow_table().to_pylist()}

    by_claim: dict[str, list[dict]] = {}
    for r in rows:
        by_claim.setdefault(r["claim_id"], []).append(r)
    print(f"   {len(by_claim)} claims")

    print("2) mining baselines + computing profiles ...")
    b = mine_all(rows)
    idx = build_indexes(b["diag_procedure_norms"], b["procedure_cost_pctiles"])
    cost_idx = {(r["hbp_code"], r["provider_state"]): r
                for r in b["procedure_cost_pctiles"]}

    ids, profiles, rule_flagged, is_fraud = [], [], [], []
    for cid, lines in by_claim.items():
        lines = sorted(lines, key=lambda r: r["line_no"])
        cost_res = score_claim(lines, cost_idx)
        rules_res = evaluate_claim(lines, idx)
        ids.append(cid)
        profiles.append(build_profile(lines, cost_res))
        rule_flagged.append(1 if rules_res["findings"] else 0)
        is_fraud.append(labels[cid]["is_fraud"])

    rule_flagged = np.array(rule_flagged)
    is_fraud = np.array(is_fraud)

    print("3) fitting Isolation Forest (unsupervised — no labels used) ...")
    det = AnomalyDetector(contamination=0.05).fit(profiles)
    anom = det.is_anomaly(profiles)
    scores = det.score(profiles)
    print(f"   flagged {anom.sum()} claims as unusual ({anom.mean()*100:.1f}%)")

    # ---- the evaluation that matters ----
    missed = (is_fraud == 1) & (rule_flagged == 0)          # fraud the rules missed
    caught_by_anom = missed & (anom == 1)
    print("\n" + "=" * 62)
    print("DOES IT CATCH WHAT THE RULES MISS?")
    print("=" * 62)
    print(f"  total fraud                 : {int(is_fraud.sum())}")
    print(f"  fraud the rules already flag: {int(((is_fraud==1)&(rule_flagged==1)).sum())}")
    print(f"  fraud the rules MISSED      : {int(missed.sum())}")
    print(f"  ...of those, anomaly caught : {int(caught_by_anom.sum())}")

    honest_flagged = int(((is_fraud == 0) & (anom == 1)).sum())
    print(f"  new false alarms introduced : {honest_flagged}")

    if missed.sum() == 0:
        verdict = "rules missed nothing here — anomaly adds no recall on this data"
    elif caught_by_anom.sum() == 0:
        verdict = "anomaly caught NONE of the rule-missed fraud — not earning its place"
    else:
        verdict = (f"anomaly recovered {int(caught_by_anom.sum())}/{int(missed.sum())} "
                   f"rule-missed fraud, at the cost of {honest_flagged} false alarms")
    print(f"\n  VERDICT: {verdict}")

    # fraud vs honest separation, as a sanity check
    print(f"\n  mean anomaly score  fraud: {scores[is_fraud==1].mean():.3f}   "
          f"honest: {scores[is_fraud==0].mean():.3f}")

    print("\n4) example — most unusual claim and WHY:")
    pop = to_frame(profiles)
    top = int(np.argmax(scores))
    print(f"   {ids[top]}  (actually fraud: {bool(is_fraud[top])}, "
          f"rules flagged: {bool(rule_flagged[top])})")
    for d in AnomalyDetector.explain(profiles[top], pop):
        direction = "above" if d["z_score"] > 0 else "below"
        print(f"     {d['feature']:<22} {d['value']:<12g} "
              f"{abs(d['z_score']):.1f} sd {direction} normal")

    # ---- the real test: fraud NO rule was written for ----
    print("\n" + "=" * 62)
    print("HELD-OUT TEST: 'UNBUNDLING' — A FRAUD TYPE NO RULE CHECKS")
    print("=" * 62)
    from eval.generate_realistic import generate_novel_fraud
    from data_quality.ingestion_gate import run_gate
    from pathlib import Path

    mapping = str(Path(__file__).parent / "data_quality" / "mappings" / "source_hospital_a.yaml")
    novel_raw, _ = generate_novel_fraud(n_claims=40)
    novel_clean = run_gate(novel_raw, mapping)["clean"]

    novel_by_claim: dict[str, list[dict]] = {}
    for r in novel_clean:
        novel_by_claim.setdefault(r["claim_id"], []).append(r)

    novel_profiles, novel_rule_hits = [], 0
    for cid, lines in novel_by_claim.items():
        lines = sorted(lines, key=lambda r: r["line_no"])
        if evaluate_claim(lines, idx)["findings"]:
            novel_rule_hits += 1
        novel_profiles.append(build_profile(lines, score_claim(lines, cost_idx)))

    novel_anom = det.is_anomaly(novel_profiles)      # scored by the model fit ABOVE
    n = len(novel_profiles)
    print(f"  {n} unbundled claims (every line valid, at rate, matching diagnosis)")
    print(f"  caught by RULES  : {novel_rule_hits}/{n} "
          f"({novel_rule_hits/n*100:.0f}%)")
    print(f"  caught by ANOMALY: {int(novel_anom.sum())}/{n} "
          f"({novel_anom.mean()*100:.0f}%)")
    if novel_anom.sum() > novel_rule_hits:
        print("\n  -> anomaly detection EARNS ITS PLACE: it catches fraud the rules "
              "cannot see, because it judges the claim's shape, not its rule breaches.")
    else:
        print("\n  -> anomaly detection did not beat the rules even here.")

    print("\n=== anomaly detection evaluated honestly ===")


if __name__ == "__main__":
    main()
