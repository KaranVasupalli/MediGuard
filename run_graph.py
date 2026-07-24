"""Provider-graph step: can we find a colluding ring we never told the model about?

Evaluation is against injected ground truth: a known set of ring providers exists in
the data, the graph never sees that list, and we measure whether it surfaces them at
the top of the ring-risk ranking.
"""
import shutil

import pyarrow as pa
from deltalake import write_deltalake, DeltaTable

from config.spark_config import load_config
from batch.provider_graph import build_provider_graph, analyse, ring_risk_score


def write_delta(path: str, rows: list[dict]):
    shutil.rmtree(path, ignore_errors=True)
    if not rows:
        return
    t = pa.Table.from_pylist(rows)
    for i, f in enumerate(t.schema):
        if pa.types.is_null(f.type):
            t = t.set_column(i, f.name, t.column(i).cast(pa.string()))
    write_deltalake(path, t, mode="overwrite")


def main():
    cfg = load_config()
    ref = cfg["paths"]["reference"]

    print("1) loading corpus ...")
    rows = DeltaTable(cfg["paths"]["corpus"]).to_pyarrow_table().to_pylist()
    providers = {r["provider_id"] for r in rows}
    patients = {r["patient_hash"] for r in rows}
    print(f"   {len(rows)} lines | {len(providers)} providers | {len(patients)} patients")

    print("2) building shared-patient graph ...")
    G = build_provider_graph(rows, min_shared=2)
    print(f"   {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    if G.number_of_edges() == 0:
        print("   no shared patients — graph analysis impossible on this data")
        return
    del rows                    # the graph is all we need from here; free the corpus

    print("3) computing PageRank + communities + ring risk ...")
    stats = analyse(G)                      # reuse the graph already built
    edges = sorted(s["shared_patient_edges"] for s in stats)
    median_edges = edges[len(edges) // 2] if edges else 0
    for s in stats:
        s["ring_risk"] = ring_risk_score(s, median_edges)
    stats.sort(key=lambda s: -s["ring_risk"])
    write_delta(f"{ref}/provider_risk", stats)

    print("\n   provider ranking by ring risk:")
    print(f"   {'provider':<10}{'ring_risk':>10}{'internal':>10}{'comm':>7}"
          f"{'size':>6}{'edges':>7}{'pagerank':>10}")
    for s in stats:
        print(f"   {s['provider_id']:<10}{s['ring_risk']:>10.3f}"
              f"{s['internal_ratio']:>10.2f}{s['community_id']:>7}"
              f"{s['community_size']:>6}{s['shared_patient_edges']:>7}"
              f"{s['pagerank']:>10.4f}")

    # ---- honest evaluation against injected truth ----
    try:
        truth = {t["provider_id"]: t["in_ring"] for t in
                 DeltaTable(f"{ref}/ring_truth").to_pyarrow_table().to_pylist()}
    except Exception:
        print("\n   (no ring ground truth found — run rebuild_data.py first)")
        return

    n_ring = sum(truth.values())
    top = [s["provider_id"] for s in stats[:n_ring]]
    hits = sum(1 for p in top if truth.get(p) == 1)

    print("\n" + "=" * 62)
    print("DID THE GRAPH FIND THE RING IT WAS NEVER TOLD ABOUT?")
    print("=" * 62)
    print(f"  actual ring providers : {sorted(p for p, v in truth.items() if v == 1)}")
    print(f"  top-{n_ring} by ring risk : {sorted(top)}")
    print(f"  correctly identified  : {hits}/{n_ring}")

    ring_ir = [s["internal_ratio"] for s in stats if truth.get(s["provider_id"]) == 1]
    honest_ir = [s["internal_ratio"] for s in stats if truth.get(s["provider_id"]) == 0]
    if ring_ir and honest_ir:
        print(f"  mean internal_ratio   : ring {sum(ring_ir)/len(ring_ir):.2f}  "
              f"vs honest {sum(honest_ir)/len(honest_ir):.2f}")

    if hits == n_ring:
        print("\n  VERDICT: ring fully identified from structure alone.")
    elif hits > 0:
        print(f"\n  VERDICT: partial — {hits} of {n_ring} found; the rest look "
              f"structurally normal.")
    else:
        print("\n  VERDICT: graph did NOT find the ring — signal too weak here.")

    print("\n=== provider graph evaluated honestly ===")


if __name__ == "__main__":
    main()
