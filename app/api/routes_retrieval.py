from fastapi import APIRouter

from app.agents.source_router_agent import source_router_agent
from app.rag.hybrid_retriever import hybrid_search
from app.rag.citation import citations_from_hits
from app.schemas.api import (
    RetrievalHit,
    RetrievalRequest,
    RetrievalResponse,
    SourceRouteRequest,
    SourceRouteResponse,
)


router = APIRouter(prefix="/api/retrieval", tags=["retrieval"])


@router.post("/route", response_model=SourceRouteResponse)
def route(request: SourceRouteRequest) -> SourceRouteResponse:
    return SourceRouteResponse(
        query=request.query,
        source_router_result=source_router_agent(
            question=request.query,
            requested_source_ids=request.source_ids,
        ),
    )


@router.post("/search", response_model=RetrievalResponse)
def search(request: RetrievalRequest) -> RetrievalResponse:
    hits = hybrid_search(
        query=request.query,
        top_k=request.top_k,
        mode=request.mode,
        source_ids=request.source_ids,
    )

    response_hits = []
    for rank, hit in enumerate(hits, start=1):
        doc = hit["doc"]
        metadata = doc.metadata or {}
        response_hits.append(
            RetrievalHit(
                rank=rank,
                title=metadata.get("title", ""),
                knowledge_source_id=metadata.get("knowledge_source_id", ""),
                knowledge_source_name=metadata.get("knowledge_source_name", ""),
                knowledge_source_type=metadata.get("knowledge_source_type", ""),
                source_file=metadata.get("source", ""),
                source_url=metadata.get("source_url", ""),
                region=metadata.get("region", ""),
                topic=metadata.get("topic", ""),
                score=hit.get("score"),
                rerank_score=hit.get("rerank_score"),
                rerank_provider=hit.get("rerank_provider"),
                rerank_features=hit.get("rerank_features", {}),
                retriever=",".join(sorted(hit.get("retrievers", []))),
                content_preview=doc.page_content[:500],
                metadata=metadata,
            )
        )

    return RetrievalResponse(
        query=request.query,
        mode=request.mode,
        top_k=request.top_k,
        source_ids=request.source_ids,
        hits=response_hits,
        citations=citations_from_hits(hits),
        rerank_enabled=True,
    )
