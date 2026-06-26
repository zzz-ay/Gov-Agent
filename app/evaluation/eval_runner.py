import time
from typing import List, Optional, Tuple

from app.evaluation.metrics import reciprocal_rank, safe_average
from app.rag.hybrid_retriever import hybrid_search
from app.schemas.api import EvalCase, EvalCaseResult, EvalRunResponse


def find_matched_rank(case: EvalCase, hits: list[dict]) -> Tuple[bool, Optional[int], Optional[str], Optional[float]]:
    if not case.gold_source_file:
        return False, None, None, None

    for rank, hit in enumerate(hits, start=1):
        doc = hit["doc"]
        source_file = str((doc.metadata or {}).get("source", ""))
        if case.gold_source_file in source_file:
            return True, rank, source_file, hit.get("rerank_score")

    return False, None, None, None


def run_retrieval_eval(
    cases: List[EvalCase],
    top_k: int,
    mode: str,
    source_ids: List[str] = None,
) -> EvalRunResponse:
    results: List[EvalCaseResult] = []

    for case in cases:
        case_source_ids = source_ids or case.expected_source_ids
        started = time.perf_counter()
        hits = hybrid_search(case.question, top_k=top_k, mode=mode, source_ids=case_source_ids)
        latency_ms = (time.perf_counter() - started) * 1000
        hit, matched_rank, matched_source, matched_rerank_score = find_matched_rank(case, hits)

        results.append(
            EvalCaseResult(
                question=case.question,
                source_ids=case_source_ids,
                hit=hit,
                reciprocal_rank=reciprocal_rank(matched_rank),
                matched_rank=matched_rank,
                matched_source_file=matched_source,
                matched_rerank_score=matched_rerank_score,
                latency_ms=latency_ms,
            )
        )

    total = len(results)
    hit_rate = sum(1 for item in results if item.hit) / total if total else 0.0
    mrr = safe_average([item.reciprocal_rank for item in results])
    avg_latency_ms = safe_average([item.latency_ms for item in results])

    return EvalRunResponse(
        total=total,
        hit_rate=hit_rate,
        mrr=mrr,
        avg_latency_ms=avg_latency_ms,
        results=results,
    )
