from typing import Any, Dict, List, Optional

from app.knowledge_sources import get_enabled_knowledge_sources


SOURCE_PRIORITY = ["gov_policy", "laws", "legal_cases", "public_notices"]

SOURCE_KEYWORDS: Dict[str, List[str]] = {
    "gov_policy": [
        "政策",
        "补贴",
        "申请",
        "办理",
        "流程",
        "材料",
        "条件",
        "资格",
        "社保",
        "医保",
        "就业",
        "创业",
        "档案",
        "户口",
        "户籍",
        "居住证",
        "公积金",
        "政务",
        "人才",
        "落户",
    ],
    "laws": [
        "法律",
        "法规",
        "法条",
        "规定",
        "依据",
        "劳动合同",
        "民法典",
        "行政处罚",
        "行政许可",
        "个人信息",
        "数据安全",
        "网络安全",
        "违法",
        "处罚",
        "责任",
        "合规",
    ],
    "legal_cases": [
        "案例",
        "类案",
        "判决",
        "法院",
        "裁判",
        "怎么判",
        "类似情况",
        "相似情况",
        "罪",
        "刑事",
        "诈骗",
        "风险",
        "虚假",
        "骗取",
        "纠纷",
    ],
    "public_notices": [
        "公告",
        "公示",
        "通知",
        "名单",
        "申报时间",
        "截止",
        "最新",
        "批次",
        "结果公示",
    ],
}

RISK_KEYWORDS = ["风险", "后果", "责任", "处罚", "违法", "虚假", "骗取"]
SERVICE_KEYWORDS = ["申请", "办理", "补贴", "材料", "流程", "条件", "政务"]
CASE_REFERENCE_KEYWORDS = ["案例", "类案", "判决", "法院", "裁判", "怎么判", "类似"]


def infer_task_type(text: str, source_ids: List[str]) -> str:
    if contains_any(text, CASE_REFERENCE_KEYWORDS):
        return "case_reference"

    if contains_any(text, RISK_KEYWORDS):
        return "risk_compliance"

    if "legal_cases" in source_ids and "gov_policy" not in source_ids:
        return "case_reference"

    if "laws" in source_ids and "gov_policy" not in source_ids:
        return "legal_basis"

    if "public_notices" in source_ids and "gov_policy" not in source_ids:
        return "public_notice"

    if "gov_policy" in source_ids and contains_any(text, SERVICE_KEYWORDS):
        return "policy_service"

    return "general_qa"


def contains_any(text: str, keywords: List[str]) -> bool:
    return any(keyword in text for keyword in keywords)


def unique_sources(source_ids: List[str]) -> List[str]:
    ordered = []
    for source_id in source_ids:
        if source_id and source_id not in ordered:
            ordered.append(source_id)
    return ordered


def enabled_source_ids() -> List[str]:
    return [source.id for source in get_enabled_knowledge_sources()]


def sort_sources(source_ids: List[str]) -> List[str]:
    priority = {source_id: index for index, source_id in enumerate(SOURCE_PRIORITY)}
    return sorted(unique_sources(source_ids), key=lambda item: priority.get(item, 99))


def keep_enabled_sources(source_ids: List[str], reasons: List[str]) -> List[str]:
    enabled = enabled_source_ids()
    enabled_set = set(enabled)
    requested = sort_sources(source_ids)
    filtered = [source_id for source_id in requested if source_id in enabled_set]
    missing = [source_id for source_id in requested if source_id not in enabled_set]

    if missing:
        reasons.append(f"Skipped disabled or unavailable sources: {', '.join(missing)}.")

    if filtered:
        return filtered

    for fallback in SOURCE_PRIORITY:
        if fallback in enabled_set:
            reasons.append(f"Fallback to enabled source: {fallback}.")
            return [fallback]

    return enabled[:1]


def route_knowledge_sources(
    question: str,
    intent_result: Optional[Dict[str, Any]] = None,
    requested_source_ids: Optional[List[str]] = None,
) -> Dict[str, Any]:
    text = question or ""
    reasons: List[str] = []

    if requested_source_ids:
        selected = keep_enabled_sources(requested_source_ids, reasons)
        task_type = infer_task_type(text, selected)
        return {
            "source_ids": selected,
            "requested_source_ids": unique_sources(requested_source_ids),
            "enabled_source_ids": enabled_source_ids(),
            "routing_reasons": reasons + ["Manual source_ids override was provided."],
            "strategy": "manual_override",
            "task_type": task_type,
            "should_run_service_agents": task_type == "policy_service",
        }

    source_ids: List[str] = []

    for source_id, keywords in SOURCE_KEYWORDS.items():
        if contains_any(text, keywords):
            source_ids.append(source_id)
            reasons.append(f"Matched {source_id} keywords.")

    if contains_any(text, RISK_KEYWORDS):
        source_ids.extend(["gov_policy", "laws", "legal_cases"])
        reasons.append("Risk or compliance question benefits from policy, law, and case evidence.")

    if contains_any(text, CASE_REFERENCE_KEYWORDS) and "laws" not in source_ids:
        source_ids.append("laws")
        reasons.append("Case-reference questions also need legal basis.")

    if contains_any(text, SERVICE_KEYWORDS) and "gov_policy" not in source_ids:
        source_ids.append("gov_policy")
        reasons.append("Public-service questions should include policy documents.")

    if not source_ids:
        source_ids.append("gov_policy")
        reasons.append("Defaulted to gov_policy for general public-service question.")

    selected = keep_enabled_sources(source_ids, reasons)
    task_type = infer_task_type(text, selected)
    return {
        "source_ids": selected,
        "requested_source_ids": sort_sources(source_ids),
        "enabled_source_ids": enabled_source_ids(),
        "routing_reasons": reasons,
        "strategy": "rule_based",
        "task_type": task_type,
        "should_run_service_agents": task_type == "policy_service",
    }


def source_router_agent(
    question: str,
    intent_result: Optional[Dict[str, Any]] = None,
    requested_source_ids: Optional[List[str]] = None,
) -> Dict[str, Any]:
    return route_knowledge_sources(
        question=question,
        intent_result=intent_result or {},
        requested_source_ids=requested_source_ids,
    )
