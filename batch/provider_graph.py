"""Provider fraud-ring graph: find COORDINATED fraud across hospitals.

Every component so far judges one claim at a time. A ring is invisible that way — each
individual claim can look ordinary while the pattern only appears across providers.

The graph:
  nodes  = providers (hospitals)
  edges  = shared patients; weight = how many distinct patients two providers share

Real patients occasionally visit two hospitals, so a few shared patients are normal.
A ring shows up as an unusually DENSE, TIGHT cluster: a small group of providers
sharing far more patients with each other than with anyone outside.

Three signals are computed per provider:
  * pagerank           - influence in the shared-patient network
  * community          - which cluster it belongs to (Louvain modularity)
  * internal_ratio     - share of its patient links that stay inside its own cluster
                         (the strongest ring signal: rings are inward-facing)

This is the genuine big-data step: it needs the whole population at once, not one
claim. The same logic ports to Spark/GraphFrames for full-scale runs.
"""
from collections import defaultdict
from itertools import combinations

import networkx as nx


def build_provider_graph(rows: list[dict], min_shared: int = 2) -> nx.Graph:
    """Providers linked when they share patients. min_shared filters coincidences."""
    patients: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        patients[r["patient_hash"]].add(r["provider_id"])

    pair_counts: dict[tuple[str, str], int] = defaultdict(int)
    for _, provs in patients.items():
        if len(provs) < 2:
            continue                       # patient stayed with one provider
        for a, b in combinations(sorted(provs), 2):
            pair_counts[(a, b)] += 1

    G = nx.Graph()
    G.add_nodes_from({r["provider_id"] for r in rows})
    for (a, b), n in pair_counts.items():
        if n >= min_shared:
            G.add_edge(a, b, weight=n)
    return G


def analyse(G: nx.Graph) -> list[dict]:
    """Per-provider risk signals. Returns rows for the provider_risk table."""
    if G.number_of_nodes() == 0:
        return []

    pagerank = nx.pagerank(G, weight="weight") if G.number_of_edges() else \
        {n: 1 / G.number_of_nodes() for n in G}

    # communities (Louvain modularity); isolated nodes each form their own
    if G.number_of_edges():
        communities = nx.community.louvain_communities(G, weight="weight", seed=42)
    else:
        communities = [{n} for n in G]
    comm_of = {n: i for i, c in enumerate(communities) for n in c}
    comm_size = {i: len(c) for i, c in enumerate(communities)}

    ranked = sorted(pagerank.values())

    out = []
    for node in G.nodes():
        nbrs = list(G[node])
        total_w = sum(G[node][v]["weight"] for v in nbrs)
        inside = [G[node][v]["weight"] for v in nbrs if comm_of.get(v) == comm_of.get(node)]
        outside = [G[node][v]["weight"] for v in nbrs if comm_of.get(v) != comm_of.get(node)]
        internal_w = sum(inside)
        # concentration: how much more this provider shares with its own cluster than
        # with everyone else. A ring shuttles patients internally -> high ratio.
        avg_in = sum(inside) / len(inside) if inside else 0.0
        avg_out = sum(outside) / len(outside) if outside else 0.0
        concentration = (avg_in / avg_out) if avg_out > 0 else (avg_in if avg_in else 0.0)

        pr = pagerank.get(node, 0.0)
        pct = sum(1 for v in ranked if v <= pr) / len(ranked) if ranked else 0.0

        out.append({
            "provider_id": node,
            "pagerank": round(float(pr), 6),
            "pagerank_pctile": round(float(pct), 4),
            "community_id": str(comm_of.get(node, -1)),
            "community_size": int(comm_size.get(comm_of.get(node, -1), 1)),
            "shared_patient_edges": int(total_w),
            "internal_ratio": round(internal_w / total_w, 4) if total_w else 0.0,
            "concentration": round(float(concentration), 3),
            "degree": len(nbrs),
        })
    return out


def ring_risk_score(row: dict, median_edges: float) -> float:
    """Combine graph signals into one 0-1 ring-risk score.

    NOTE ON DESIGN: an earlier version scored 'high internal_ratio' as suspicious.
    That was wrong — in a healthy network the large honest community is ALSO
    internally connected, so internal_ratio alone flags everyone. What actually
    separates a ring is:
      * it is a SMALL cluster (a handful of providers, not the bulk of the market)
      * its members' patient sharing is CONCENTRATED on each other far more than on
        the outside world (concentration = inside avg weight / outside avg weight)
    Both are plain, checkable quantities.
    """
    conc = min(row.get("concentration", 0.0) / 5.0, 1.0)     # 5x inside:outside = max
    size = row.get("community_size", 1)
    small_cluster = 1.0 if 2 <= size <= 6 else (0.4 if size <= 10 else 0.0)
    busy = min(row["shared_patient_edges"] / max(median_edges * 2, 1), 1.0)
    return round(0.55 * conc + 0.35 * small_cluster + 0.10 * busy, 4)


def score_providers(rows: list[dict], min_shared: int = 2) -> list[dict]:
    G = build_provider_graph(rows, min_shared=min_shared)
    stats = analyse(G)
    if not stats:
        return []
    edges = sorted(s["shared_patient_edges"] for s in stats)
    median_edges = edges[len(edges) // 2] if edges else 0
    for s in stats:
        s["ring_risk"] = ring_risk_score(s, median_edges)
    return sorted(stats, key=lambda s: -s["ring_risk"])
