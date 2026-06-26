import os
from typing import Any, Dict, List, Set

from app.rag.hybrid_retriever import tokenize


SUPPORTED_RERANKERS = {
    "local_rule",
    "bge_cross_encoder",
    "dashscope_rerank",
}


def get_reranker_provider() -> str:
    provider = os.getenv("RERANKER_PROVIDER", "local_rule").strip().lower()
    if provider not in SUPPORTED_RERANKERS:
        return "local_rule"
    return provider


def _term_set(text: str) -> Set[str]:
    return set(tokenize(text))


def _coverage_score(query_terms: Set[str], content_terms: Set[str]) -> float:
    if not query_terms:
        return 0.0
    return len(query_terms & content_terms) / len(query_terms)


def _phrase_score(query: str, content: str) -> float:
    query = query.strip()
    content = content.strip()

    if not query or not content:
        return 0.0

    if query in content:
        return 1.0

    query_terms = tokenize(query)
    if len(query_terms) < 2:
        return 0.0

    bigram_hits = 0
    bigram_count = 0

    for idx in range(len(query_terms) - 1):
        phrase = query_terms[idx] + query_terms[idx + 1]
        if len(phrase) < 2:
            continue
        bigram_count += 1
        if phrase in content:
            bigram_hits += 1

    if bigram_count == 0:
        return 0.0

    return bigram_hits / bigram_count


def local_rule_rerank(query: str, hits: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
    """
    Lightweight local reranker.

    This is intentionally dependency-free so the enterprise API can run without
    downloading a cross-encoder model. A later iteration can replace this with
    bge-reranker or another CrossEncoder while keeping the same hit contract.
    """
    query_terms = _term_set(query)
    reranked = []

    for hit in hits:
        doc = hit["doc"]
        content = doc.page_content or ""
        metadata = doc.metadata or {}
        content_terms = _term_set(content)

        coverage = _coverage_score(query_terms, content_terms)
        phrase = _phrase_score(query, content)
        retrieval_score = float(hit.get("score") or 0.0)
        retriever_bonus = 0.15 if len(hit.get("retrievers", [])) > 1 else 0.0

        metadata_text = " ".join(
            str(metadata.get(key, ""))
            for key in ["title", "topic", "region", "source_org"]
        )
        metadata_coverage = _coverage_score(query_terms, _term_set(metadata_text))

        rerank_score = (
            coverage * 0.45
            + phrase * 0.2
            + metadata_coverage * 0.2
            + retriever_bonus
            + min(retrieval_score / 10.0, 1.0) * 0.15
        )

        enriched = dict(hit)
        enriched["rerank_provider"] = "local_rule"
        enriched["rerank_score"] = rerank_score
        enriched["rerank_features"] = {
            "coverage": coverage,
            "phrase": phrase,
            "metadata_coverage": metadata_coverage,
            "retriever_bonus": retriever_bonus,
            "retrieval_score": retrieval_score,
        }
        reranked.append(enriched)

    reranked.sort(key=lambda item: item.get("rerank_score", 0.0), reverse=True)
    return reranked[:top_k]


def rerank_hits(query: str, hits: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
    provider = get_reranker_provider()

    if provider == "local_rule":
        return local_rule_rerank(query, hits, top_k)

    # Placeholders for future production rerankers. Falling back keeps the
    # service usable when optional model dependencies or API keys are missing.
    fallback_hits = local_rule_rerank(query, hits, top_k)
    for hit in fallback_hits:
        hit["rerank_provider"] = f"{provider}:fallback_local_rule"
    return fallback_hits


def get_reranker_runtime_info() -> Dict[str, Any]:
    provider = get_reranker_provider()
    return {
        "provider": provider,
        "supported_providers": sorted(SUPPORTED_RERANKERS),
        "fallback_provider": "local_rule",
        "env": {
            "provider_env": "RERANKER_PROVIDER",
            "bge_model_env": "BGE_RERANKER_MODEL",
            "dashscope_model_env": "DASHSCOPE_RERANKER_MODEL",
        },
        "status": (
            "active"
            if provider == "local_rule"
            else "placeholder_with_local_rule_fallback"
        ),
    }
