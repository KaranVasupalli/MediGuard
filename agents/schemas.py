"""Typed evidence bundle and verdict output. The verdict is the pipeline's contract
with the outside world; every number in it is computed BEFORE the LLM is called."""
from typing import Optional
from pydantic import BaseModel


class LineAdjudication(BaseModel):
    line: int
    billed: float
    allowed: float
    status: str            # REJECTED / REDUCED / ALLOWED
    reason: str


class Citation(BaseModel):
    finding: str
    source: str
    span: str              # verbatim quote — the prompt-injection boundary & audit anchor


class EvidenceBundle(BaseModel):
    """Produced by the deterministic evidence computers. LLM only reads this."""
    rules_baseline: list[dict] = []
    semantic_similarity: dict = {}
    anomaly: dict = {}
    cost_model: dict = {}
    supervised: dict = {}
    shap_top_features: list = []
    provider_context: dict = {}


class Audit(BaseModel):
    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None
    temperature: float = 0
    prompt_hash: Optional[str] = None
    reference_delta_versions: dict = {}
    human_review_required: bool = True


class Verdict(BaseModel):
    claim_id: str
    verdict: str                       # FLAG_FOR_AUDIT / APPROVE / ...
    fraud_score: float
    billed_total_inr: float
    justified_total_inr: float
    estimated_excess_inr: float
    recommended_action: str
    latency_ms: dict = {}
    evidence: EvidenceBundle
    line_adjudication: list[LineAdjudication] = []
    explanation: str = ""
    citations: list[Citation] = []
    audit: Audit
