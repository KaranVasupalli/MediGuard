"""Serving layer: DuckDB reads Delta tables directly via delta_scan.
This is the fast read path the online verdict pipeline uses."""
import duckdb


def _conn():
    con = duckdb.connect()
    con.execute("INSTALL delta; LOAD delta;")
    return con


def read_corpus(corpus_path: str):
    """Return the corpus as a list of dict rows (small in the skeleton)."""
    con = _conn()
    rel = con.execute(f"SELECT * FROM delta_scan('{corpus_path}')")
    cols = [d[0] for d in rel.description]
    return [dict(zip(cols, row)) for row in rel.fetchall()]


def read_claim(corpus_path: str, claim_id: str):
    con = _conn()
    rel = con.execute(
        f"SELECT * FROM delta_scan('{corpus_path}') WHERE claim_id = ? ORDER BY line_no",
        [claim_id],
    )
    cols = [d[0] for d in rel.description]
    return [dict(zip(cols, row)) for row in rel.fetchall()]
