from typing import Any, Dict, Generator, Iterable, List

from langchain_core.prompts import ChatPromptTemplate

from app.agents.intent_agent import classify_intent
from app.agents.knowledge_agent import knowledge_agent
from app.agents.source_router_agent import source_router_agent
from app.llm.provider import get_chat_llm
from app.rag.citation import citations_from_sources


PLATFORM_TASK_TYPES = {
    "legal_basis",
    "case_reference",
    "risk_compliance",
    "public_notice",
}


STREAM_ANSWER_SYSTEM_PROMPT = """
你是政策法规 RAG Agent 平台中的回答生成助手。

你必须只依据给定的检索资料回答用户问题，不要编造政策名称、法条内容、案例结论、金额、条件、材料、流程、机构、电话或网址。
如果资料没有明确说明，请说“当前检索资料未明确说明”。
涉及地区差异、执法口径或司法判断时，请提醒以主管部门、正式法律文书或专业法律意见为准。

回答要求：
- 先给简要结论，再给依据、风险点和建议动作。
- 如果检索资料来自法律法规或案例，需要说明其作用是“法律依据”或“案例参考”。
- 语言简洁、保守，避免过度承诺。
- 不要输出内部 Agent 名称。

用户问题：
{question}

意图识别：
{intent_info}

检索资料：
{policy_context}
"""


def format_policy_context(policy_sources: List[Dict[str, Any]]) -> str:
    if not policy_sources:
        return "当前未检索到可用政策资料。"

    blocks = []
    for source in policy_sources:
        blocks.append(
            f"""
【政策资料 {source.get('index', '')}】
标题：{source.get('title', '')}
发布机构：{source.get('source_org', '')}
发布时间：{source.get('publish_date', '')}
适用地区：{source.get('region', '')}
主题：{source.get('topic', '')}
来源文件：{source.get('source_file', '')}
来源链接：{source.get('source_url', '')}
页码：{source.get('page', '')}

正文片段：
{source.get('content', '')}
""".strip()
        )

    return "\n\n".join(blocks)


def format_evidence_context(policy_sources: List[Dict[str, Any]]) -> str:
    if not policy_sources:
        return "当前未检索到可用证据。"

    label_by_source = {
        "gov_policy": "政策依据",
        "laws": "法律法规依据",
        "legal_cases": "案例参考",
        "public_notices": "公告通知",
    }
    blocks = []

    for source in policy_sources:
        source_id = source.get("knowledge_source_id", "")
        label = label_by_source.get(source_id, "检索证据")
        blocks.append(
            f"""
【{label}{source.get("index", "")}】
Title: {source.get("title", "")}
Source Org: {source.get("source_org", "")}
Publish Date: {source.get("publish_date", "")}
Region: {source.get("region", "")}
Topic: {source.get("topic", "")}
Knowledge Source: {source_id}
Source File: {source.get("source_file", "")}
Source URL: {source.get("source_url", "")}
Page: {source.get("page", "")}

Content:
{source.get("content", "")}
""".strip()
        )

    return "\n\n".join(blocks)


def build_streaming_context(question: str, source_ids: List[str] = None) -> Dict[str, Any]:
    intent_result = classify_intent(question)
    source_router_result = source_router_agent(
        question=question,
        intent_result=intent_result,
        requested_source_ids=source_ids or [],
    )
    task_type = source_router_result.get("task_type", "")

    if task_type in PLATFORM_TASK_TYPES:
        intent_result = {
            "intent": task_type,
            "intent_name": {
                "legal_basis": "法规依据",
                "case_reference": "类案参考",
                "risk_compliance": "风险合规",
                "public_notice": "公告通知",
            }.get(task_type, "综合问答"),
            "region": "未明确",
            "need_clarification": False,
            "clarification_question": "",
            "reason": "Source Router 已识别为政策法规平台任务，不使用旧毕业生政务范围限制。",
        }

    if intent_result.get("need_clarification", False):
        clarification = intent_result.get(
            "clarification_question",
            "请补充所在地区或具体情况，以便查询对应政策。",
        )
        return {
            "question": question,
            "intent_result": intent_result,
            "source_router_result": source_router_result,
            "policy_result": {"policy_sources": []},
            "citations": [],
            "direct_answer": clarification,
        }

    policy_result = knowledge_agent(
        question=question,
        intent_result=intent_result,
        source_router_result=source_router_result,
    )
    policy_sources = policy_result.get("policy_sources", [])

    return {
        "question": question,
        "intent_result": intent_result,
        "source_router_result": policy_result.get("source_router_result", {}),
        "policy_result": policy_result,
        "citations": citations_from_sources(policy_sources),
        "direct_answer": "",
    }


def stream_answer_from_context(context: Dict[str, Any]) -> Iterable[str]:
    direct_answer = context.get("direct_answer", "")
    if direct_answer:
        yield direct_answer
        return

    question = context["question"]
    intent_result = context.get("intent_result", {})
    policy_sources = context.get("policy_result", {}).get("policy_sources", [])

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", STREAM_ANSWER_SYSTEM_PROMPT),
            ("human", "请根据以上资料生成回答。"),
        ]
    )

    chain = prompt | get_chat_llm(temperature=0.1)
    stream = chain.stream(
        {
            "question": question,
            "intent_info": str(intent_result),
            "policy_context": format_evidence_context(policy_sources),
        }
    )

    for chunk in stream:
        content = getattr(chunk, "content", "")
        if content:
            yield content


def collect_stream(chunks: Generator[str, None, None]) -> str:
    return "".join(chunks)
