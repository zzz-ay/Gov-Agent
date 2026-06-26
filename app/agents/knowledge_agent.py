from typing import Any, Dict

from app.agents.policy_agent import policy_agent
from app.config import RETRIEVER_TOP_K


def knowledge_agent(
    question: str,
    intent_result: Dict[str, Any],
    source_router_result: Dict[str, Any] = None,
    top_k: int = RETRIEVER_TOP_K,
) -> Dict[str, Any]:
    """
    General multi-source knowledge retrieval agent.

    This is the platform-facing wrapper around the historical policy_agent.
    It keeps legacy keys such as policy_sources for compatibility while adding
    knowledge_* aliases for the policy/law/case/notice platform model.
    """
    result = policy_agent(
        question=question,
        intent_result=intent_result,
        source_router_result=source_router_result,
        top_k=top_k,
    )

    knowledge_sources = result.get("policy_sources", [])
    result.update(
        {
            "agent_name": "knowledge_agent",
            "knowledge_docs": result.get("policy_docs", []),
            "knowledge_sources": knowledge_sources,
            "knowledge_context_type": "multi_source_evidence",
            "task_type": (source_router_result or {}).get("task_type", ""),
        }
    )

    return result


# Compatibility alias for future imports that prefer a noun phrase.
retrieve_knowledge = knowledge_agent
