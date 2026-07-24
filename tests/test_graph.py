"""Tests for the provider fraud-ring graph. Run: pytest tests/test_graph.py -v

The decisive test is the last one: a planted ring must rank above honest providers.
A graph component that cannot separate a ring from normal referral traffic is
decoration, and the suite should fail if that happens.
"""
import pytest

from batch.provider_graph import (
    build_provider_graph, analyse, ring_risk_score, score_providers,
)


def _rows(pairs):
    """pairs: list of (patient_hash, provider_id)."""
    return [{"patient_hash": p, "provider_id": h, "claim_id": f"C{i}", "line_no": 1}
            for i, (p, h) in enumerate(pairs)]


def test_no_shared_patients_gives_no_edges():
    rows = _rows([("p1", "H1"), ("p2", "H2"), ("p3", "H3")])
    G = build_provider_graph(rows, min_shared=1)
    assert G.number_of_nodes() == 3 and G.number_of_edges() == 0


def test_shared_patient_creates_weighted_edge():
    rows = _rows([("p1", "H1"), ("p1", "H2"), ("p2", "H1"), ("p2", "H2")])
    G = build_provider_graph(rows, min_shared=1)
    assert G.has_edge("H1", "H2")
    assert G["H1"]["H2"]["weight"] == 2         # two distinct shared patients


def test_min_shared_filters_coincidences():
    rows = _rows([("p1", "H1"), ("p1", "H2")])   # only ONE shared patient
    assert build_provider_graph(rows, min_shared=2).number_of_edges() == 0
    assert build_provider_graph(rows, min_shared=1).number_of_edges() == 1


def test_analyse_returns_one_row_per_provider():
    rows = _rows([("p1", "H1"), ("p1", "H2"), ("p2", "H1"), ("p2", "H2")])
    stats = analyse(build_provider_graph(rows, min_shared=1))
    assert {s["provider_id"] for s in stats} == {"H1", "H2"}
    for s in stats:
        assert 0.0 <= s["pagerank_pctile"] <= 1.0
        assert 0.0 <= s["internal_ratio"] <= 1.0


def test_empty_graph_is_handled():
    assert analyse(build_provider_graph([], min_shared=1)) == []


def test_small_tight_cluster_scores_higher_than_large_loose_one():
    tight = {"concentration": 6.0, "community_size": 3, "shared_patient_edges": 100}
    loose = {"concentration": 1.1, "community_size": 20, "shared_patient_edges": 100}
    assert ring_risk_score(tight, 50) > ring_risk_score(loose, 50)


def test_planted_ring_ranks_above_honest_providers():
    """3 colluding hospitals shuttling 30 patients, vs 20 hospitals with sparse
    normal cross-referrals. The ring must come out on top."""
    import random
    rng = random.Random(0)
    rows = []

    ring = ["R1", "R2", "R3"]
    for p in range(30):                       # ring patients seen at all ring hospitals
        for h in ring:
            rows += _rows([(f"rp{p}", h)])

    honest = [f"H{i:02d}" for i in range(20)]
    for p in range(600):                      # normal patients, mostly one hospital
        hs = rng.sample(honest, k=1 if rng.random() < 0.85 else 2)
        for h in hs:
            rows += _rows([(f"hp{p}", h)])

    stats = score_providers(rows, min_shared=2)
    top3 = {s["provider_id"] for s in stats[:3]}
    assert top3 == set(ring), f"ring not found; top3 was {top3}"
