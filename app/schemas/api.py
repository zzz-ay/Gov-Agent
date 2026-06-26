from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str
    debug_mode: bool = False
    auto_route: bool = True
    source_ids: List[str] = Field(default_factory=list)
    session_id: Optional[str] = None
    business_flow_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ChatResponse(BaseModel):
    log_id: Optional[str] = None
    answer: str
    sources: List[Dict[str, Any]] = Field(default_factory=list)
    citations: List[Dict[str, Any]] = Field(default_factory=list)
    business_flow: Optional[Dict[str, Any]] = None
    source_router_result: Optional[Dict[str, Any]] = None
    knowledge_result: Optional[Dict[str, Any]] = None
    evidence_result: Optional[Dict[str, Any]] = None
    risk_level: Optional[str] = None
    verification_score: Optional[int] = None
    latency_ms: Optional[float] = None
    debug_result: Optional[Dict[str, Any]] = None


class RetrievalRequest(BaseModel):
    query: str
    top_k: int = 5
    mode: str = "hybrid"
    source_ids: List[str] = Field(default_factory=list)


class RetrievalHit(BaseModel):
    rank: int
    title: str = ""
    knowledge_source_id: str = ""
    knowledge_source_name: str = ""
    knowledge_source_type: str = ""
    source_file: str = ""
    source_url: str = ""
    region: str = ""
    topic: str = ""
    score: Optional[float] = None
    rerank_score: Optional[float] = None
    rerank_provider: Optional[str] = None
    rerank_features: Dict[str, Any] = Field(default_factory=dict)
    retriever: str = ""
    content_preview: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RetrievalResponse(BaseModel):
    query: str
    mode: str
    top_k: int
    source_ids: List[str] = Field(default_factory=list)
    hits: List[RetrievalHit]
    citations: List[Dict[str, Any]] = Field(default_factory=list)
    rerank_enabled: bool = True


class SourceRouteRequest(BaseModel):
    query: str
    source_ids: List[str] = Field(default_factory=list)


class SourceRouteResponse(BaseModel):
    query: str
    source_router_result: Dict[str, Any]


class FeedbackRequest(BaseModel):
    question: str
    answer: str
    rating: str
    issue_type: Optional[str] = None
    comment: Optional[str] = None
    session_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class FeedbackResponse(BaseModel):
    success: bool
    feedback_id: str
    saved_path: str


class EvalCase(BaseModel):
    question: str
    gold_source_file: Optional[str] = None
    expected_keywords: List[str] = Field(default_factory=list)
    expected_source_ids: List[str] = Field(default_factory=list)
    expected_task_type: Optional[str] = None


class EvalRunRequest(BaseModel):
    cases: List[EvalCase]
    top_k: int = 5
    mode: str = "hybrid"
    source_ids: List[str] = Field(default_factory=list)


class AnswerEvalRunRequest(BaseModel):
    cases: List[EvalCase]
    debug_mode: bool = False


class EvalCaseResult(BaseModel):
    question: str
    source_ids: List[str] = Field(default_factory=list)
    hit: bool
    reciprocal_rank: float
    matched_rank: Optional[int] = None
    matched_source_file: Optional[str] = None
    matched_rerank_score: Optional[float] = None
    latency_ms: float


class EvalRunResponse(BaseModel):
    eval_id: Optional[str] = None
    total: int
    hit_rate: float
    mrr: float
    avg_latency_ms: float
    results: List[EvalCaseResult]


class AnswerEvalCaseResult(BaseModel):
    question: str
    answer_length: int
    source_count: int
    keyword_coverage: float
    covered_keywords: List[str] = Field(default_factory=list)
    missing_keywords: List[str] = Field(default_factory=list)
    uncertainty_count: int
    unsupported_risk_count: int
    latency_ms: float
    passed: bool


class AnswerEvalRunResponse(BaseModel):
    eval_id: Optional[str] = None
    total: int
    pass_rate: float
    avg_keyword_coverage: float
    avg_source_count: float
    avg_latency_ms: float
    results: List[AnswerEvalCaseResult]


class DocumentStatusResponse(BaseModel):
    status: Dict[str, Any]


class RebuildVectorstoreResponse(BaseModel):
    success: bool
    returncode: int
    stdout: str = ""
    stderr: str = ""
    command: str = ""


class ChatLogListResponse(BaseModel):
    logs: List[Dict[str, Any]]


class FeedbackLogListResponse(BaseModel):
    feedback: List[Dict[str, Any]]


class EvalRunListResponse(BaseModel):
    eval_runs: List[Dict[str, Any]]


class SourceRoutingEvalCaseResult(BaseModel):
    question: str
    expected_source_ids: List[str] = Field(default_factory=list)
    selected_source_ids: List[str] = Field(default_factory=list)
    expected_task_type: Optional[str] = None
    selected_task_type: str = ""
    source_match: bool
    task_match: bool
    matched: bool
    latency_ms: float
    routing_reasons: List[str] = Field(default_factory=list)


class SourceRoutingEvalRunRequest(BaseModel):
    cases: List[EvalCase]


class SourceRoutingEvalRunResponse(BaseModel):
    eval_id: Optional[str] = None
    total: int
    accuracy: float
    source_accuracy: float
    task_accuracy: float
    avg_latency_ms: float
    results: List[SourceRoutingEvalCaseResult]


class ModelRuntimeInfoResponse(BaseModel):
    provider: str
    chat_model: str
    base_url: Optional[str] = None
    api_key_masked: Optional[str] = None
    provider_env: Dict[str, Any] = Field(default_factory=dict)
    supported_providers: List[str] = Field(default_factory=list)
    reranker: Dict[str, Any] = Field(default_factory=dict)


class KnowledgeSourceInfo(BaseModel):
    id: str
    name: str
    source_type: str
    description: str
    path: str
    authority_level: str
    enabled: bool
    exists: bool


class KnowledgeSourceListResponse(BaseModel):
    sources: List[KnowledgeSourceInfo]


class BusinessFlowInfo(BaseModel):
    flow_id: str
    flow_name: str
    description: str
    source_ids: List[str] = Field(default_factory=list)
    steps: List[str] = Field(default_factory=list)
    output_focus: List[str] = Field(default_factory=list)


class BusinessFlowListResponse(BaseModel):
    flows: List[BusinessFlowInfo]


class BusinessFlowClassifyRequest(BaseModel):
    question: str


class BusinessFlowClassifyResponse(BaseModel):
    business_flow: Dict[str, Any]
