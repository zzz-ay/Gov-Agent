import time
import json
from typing import List

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.agents.business_flow import classify_business_flow
from app.graph.gov_graph import run_gov_graph
from app.knowledge_sources import get_enabled_knowledge_sources
from app.rag.citation import citations_from_sources
from app.rag.streaming_answer import build_streaming_context, stream_answer_from_context
from app.schemas.api import ChatRequest, ChatResponse
from app.storage.jsonl_store import save_chat_log


router = APIRouter(prefix="/api", tags=["chat"])


def jsonl_event(event: str, payload: dict) -> str:
    return json.dumps(
        {
            "event": event,
            "data": payload,
        },
        ensure_ascii=False,
    ) + "\n"


def get_business_flow(question: str) -> dict:
    enabled_source_ids = [source.id for source in get_enabled_knowledge_sources()]
    return classify_business_flow(question=question, enabled_source_ids=enabled_source_ids)


def summarize_knowledge_result(result: dict) -> dict:
    knowledge_result = result.get("knowledge_result") or result.get("policy_result") or {}
    return {
        "agent_name": knowledge_result.get("agent_name", ""),
        "task_type": knowledge_result.get("task_type", ""),
        "source_ids": knowledge_result.get("source_ids", []),
        "source_distribution": knowledge_result.get("source_distribution", {}),
        "evidence_groups": knowledge_result.get("evidence_groups", []),
        "source_count": len(knowledge_result.get("knowledge_sources") or knowledge_result.get("policy_sources", [])),
    }


def resolve_requested_source_ids(request: ChatRequest) -> List[str]:
    if request.source_ids:
        return request.source_ids
    if not request.auto_route:
        return [source.id for source in get_enabled_knowledge_sources()]
    return []


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    started = time.perf_counter()
    business_flow = get_business_flow(request.question)
    requested_source_ids = resolve_requested_source_ids(request)
    result = run_gov_graph(
        question=request.question,
        debug_mode=request.debug_mode,
        source_ids=requested_source_ids,
    )
    latency_ms = (time.perf_counter() - started) * 1000
    sources = result.get("sources", [])
    citations = citations_from_sources(sources)
    answer = result.get("final_answer", "")
    risk_level = result.get("risk_level")
    verification_score = result.get("verification_score")
    source_router_result = result.get("source_router_result", {})
    evidence_result = result.get("evidence_result", {})
    knowledge_result = summarize_knowledge_result(result)

    saved = save_chat_log(
        {
            "session_id": request.session_id,
            "question": request.question,
            "answer": answer,
            "source_count": len(sources),
            "sources": [
                {
                    "title": source.get("title", ""),
                    "knowledge_source_id": source.get("knowledge_source_id", ""),
                    "knowledge_source_name": source.get("knowledge_source_name", ""),
                    "knowledge_source_type": source.get("knowledge_source_type", ""),
                    "source_file": source.get("source_file", ""),
                    "source_url": source.get("source_url", ""),
                    "region": source.get("region", ""),
                    "topic": source.get("topic", ""),
                }
                for source in sources
            ],
            "citations": citations,
            "risk_level": risk_level,
            "verification_score": verification_score,
            "debug_mode": request.debug_mode,
            "latency_ms": latency_ms,
            "metadata": request.metadata,
            "business_flow": business_flow,
            "source_router_result": source_router_result,
            "knowledge_result": knowledge_result,
            "evidence_result": evidence_result,
            "source_distribution": evidence_result.get("source_distribution", {}),
            "selected_source_ids": source_router_result.get("source_ids", []),
            "requested_source_ids": requested_source_ids,
            "auto_route": request.auto_route,
        }
    )

    return ChatResponse(
        log_id=saved["log_id"],
        answer=answer,
        sources=sources,
        citations=citations,
        business_flow=business_flow,
        source_router_result=source_router_result,
        knowledge_result=knowledge_result,
        evidence_result=evidence_result,
        risk_level=risk_level,
        verification_score=verification_score,
        latency_ms=latency_ms,
        debug_result=result if request.debug_mode else None,
    )


@router.post("/chat/stream")
def chat_stream(request: ChatRequest):
    def generate():
        started = time.perf_counter()
        business_flow = get_business_flow(request.question)
        requested_source_ids = resolve_requested_source_ids(request)
        yield jsonl_event(
            "started",
            {
                "question": request.question,
                "business_flow": business_flow,
                "requested_source_ids": requested_source_ids,
                "auto_route": request.auto_route,
            },
        )

        context = build_streaming_context(
            request.question,
            source_ids=requested_source_ids,
        )
        policy_sources = context.get("policy_result", {}).get("policy_sources", [])
        citations = context.get("citations", [])
        source_router_result = context.get("source_router_result", {})
        evidence_result = context.get("policy_result", {}).get("evidence_result", {})
        knowledge_result = {
            "agent_name": context.get("policy_result", {}).get("agent_name", ""),
            "task_type": context.get("policy_result", {}).get("task_type", ""),
            "source_ids": context.get("policy_result", {}).get("source_ids", []),
            "source_distribution": context.get("policy_result", {}).get("source_distribution", {}),
            "evidence_groups": context.get("policy_result", {}).get("evidence_groups", []),
            "source_count": len(policy_sources),
        }
        answer_parts = []

        for chunk in stream_answer_from_context(context):
            answer_parts.append(chunk)
            yield jsonl_event("chunk", {"text": chunk})

        answer = "".join(answer_parts)
        latency_ms = (time.perf_counter() - started) * 1000

        saved = save_chat_log(
            {
                "session_id": request.session_id,
                "question": request.question,
                "answer": answer,
                "source_count": len(policy_sources),
                "sources": [
                    {
                        "title": source.get("title", ""),
                        "knowledge_source_id": source.get("knowledge_source_id", ""),
                        "knowledge_source_name": source.get("knowledge_source_name", ""),
                        "knowledge_source_type": source.get("knowledge_source_type", ""),
                        "source_file": source.get("source_file", ""),
                        "source_url": source.get("source_url", ""),
                        "region": source.get("region", ""),
                        "topic": source.get("topic", ""),
                    }
                    for source in policy_sources
                ],
                "citations": citations,
                "risk_level": "not_checked",
                "verification_score": 0,
                "debug_mode": request.debug_mode,
                "latency_ms": latency_ms,
                "metadata": request.metadata,
                "business_flow": business_flow,
                "source_router_result": source_router_result,
                "knowledge_result": knowledge_result,
                "evidence_result": evidence_result,
                "source_distribution": evidence_result.get("source_distribution", {}),
                "selected_source_ids": source_router_result.get("source_ids", []),
                "requested_source_ids": requested_source_ids,
                "auto_route": request.auto_route,
            }
        )

        yield jsonl_event(
            "done",
            {
                "log_id": saved["log_id"],
                "latency_ms": latency_ms,
                "sources": policy_sources,
                "citations": citations,
                "business_flow": business_flow,
                "source_router_result": source_router_result,
                "knowledge_result": knowledge_result,
                "evidence_result": evidence_result,
                "risk_level": "not_checked",
                "verification_score": 0,
                "streaming_mode": "fast_policy_answer",
            },
        )

    return StreamingResponse(
        generate(),
        media_type="application/x-ndjson; charset=utf-8",
    )
