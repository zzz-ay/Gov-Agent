from typing import Any, Dict, Iterable, List, Optional

from langchain_core.documents import Document


def _preview(text: str, limit: int = 300) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


def citation_from_source(source: Dict[str, Any], rank: int) -> Dict[str, Any]:
    content = source.get("content") or source.get("content_preview") or ""

    return {
        "rank": rank,
        "title": source.get("title", ""),
        "source_org": source.get("source_org", ""),
        "knowledge_source_id": source.get("knowledge_source_id", ""),
        "knowledge_source_name": source.get("knowledge_source_name", ""),
        "knowledge_source_type": source.get("knowledge_source_type", ""),
        "source_file": source.get("source_file") or source.get("source", ""),
        "source_url": source.get("source_url", ""),
        "region": source.get("region", ""),
        "topic": source.get("topic", ""),
        "page": source.get("page", ""),
        "chunk_id": source.get("chunk_id", ""),
        "authority_level": source.get("authority_level", ""),
        "content_preview": _preview(content),
    }


def citation_from_document(doc: Document, rank: int, score: Optional[float] = None) -> Dict[str, Any]:
    metadata = doc.metadata or {}

    citation = {
        "rank": rank,
        "title": metadata.get("title", ""),
        "source_org": metadata.get("source_org", ""),
        "knowledge_source_id": metadata.get("knowledge_source_id", ""),
        "knowledge_source_name": metadata.get("knowledge_source_name", ""),
        "knowledge_source_type": metadata.get("knowledge_source_type", ""),
        "source_file": metadata.get("source", ""),
        "source_url": metadata.get("source_url", ""),
        "region": metadata.get("region", ""),
        "topic": metadata.get("topic", ""),
        "page": metadata.get("page", ""),
        "chunk_id": metadata.get("chunk_id", ""),
        "authority_level": metadata.get("authority_level", ""),
        "content_preview": _preview(doc.page_content),
    }

    if score is not None:
        citation["score"] = score

    return citation


def citations_from_sources(sources: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        citation_from_source(source, rank=idx)
        for idx, source in enumerate(sources or [], start=1)
    ]


def citations_from_hits(hits: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    citations = []

    for idx, hit in enumerate(hits or [], start=1):
        doc = hit["doc"]
        score = hit.get("rerank_score", hit.get("score"))
        citation = citation_from_document(doc, rank=idx, score=score)
        citation["retriever"] = ",".join(sorted(hit.get("retrievers", [])))
        citation["rerank_provider"] = hit.get("rerank_provider", "")
        citation["rerank_features"] = hit.get("rerank_features", {})
        citations.append(citation)

    return citations
