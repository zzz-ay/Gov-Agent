from typing import TypedDict, Dict, Any, List

from langgraph.graph import StateGraph, START, END

from app.agents.intent_agent import classify_intent
from app.agents.source_router_agent import source_router_agent
from app.agents.service_recommendation_agent import service_recommendation_agent
from app.agents.knowledge_agent import knowledge_agent
from app.agents.policy_validity_agent import policy_validity_agent
from app.agents.agency_agent import agency_agent, should_run_agency_agent
from app.agents.eligibility_agent import eligibility_agent, should_run_eligibility
from app.agents.procedure_agent import procedure_agent, should_run_procedure
from app.agents.material_agent import material_agent, should_run_material
from app.agents.answer_agent import answer_agent
from app.agents.verify_agent import verify_agent


PLATFORM_TASK_TYPES = {
    "legal_basis",
    "case_reference",
    "risk_compliance",
    "public_notice",
}


class GovRAGState(TypedDict, total=False):
    question: str
    debug_mode: bool
    requested_source_ids: List[str]

    intent_result: Dict[str, Any]
    source_router_result: Dict[str, Any]
    service_recommendation_result: Dict[str, Any]
    knowledge_result: Dict[str, Any]
    policy_result: Dict[str, Any]
    policy_validity_result: Dict[str, Any]
    agency_result: Dict[str, Any]
    eligibility_result: Dict[str, Any]
    procedure_result: Dict[str, Any]
    material_result: Dict[str, Any]
    answer_result: Dict[str, Any]
    verify_result: Dict[str, Any]

    final_answer: str
    sources: List[Dict[str, Any]]
    evidence_result: Dict[str, Any]
    need_clarification: bool
    clarification_question: str
    verification_score: int
    risk_level: str


def intent_node(state: GovRAGState) -> GovRAGState:
    question = state["question"]
    intent_result = classify_intent(question)

    return {
        **state,
        "intent_result": intent_result,
        "need_clarification": intent_result.get("need_clarification", False),
        "clarification_question": intent_result.get("clarification_question", ""),
    }


def service_recommendation_node(state: GovRAGState) -> GovRAGState:
    question = state["question"]
    intent_result = state.get("intent_result", {})

    service_recommendation_result = service_recommendation_agent(
        question=question,
        intent_result=intent_result,
    )

    return {
        **state,
        "service_recommendation_result": service_recommendation_result,
    }


def source_router_node(state: GovRAGState) -> GovRAGState:
    question = state["question"]
    intent_result = state.get("intent_result", {})

    source_router_result = source_router_agent(
        question=question,
        intent_result=intent_result,
        requested_source_ids=state.get("requested_source_ids", []),
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

    return {
        **state,
        "intent_result": intent_result,
        "need_clarification": intent_result.get("need_clarification", False),
        "clarification_question": intent_result.get("clarification_question", ""),
        "source_router_result": source_router_result,
    }


def knowledge_node(state: GovRAGState) -> GovRAGState:
    question = state["question"]
    intent_result = state["intent_result"]
    source_router_result = state.get("source_router_result", {})

    policy_result = knowledge_agent(
        question=question,
        intent_result=intent_result,
        source_router_result=source_router_result,
    )

    return {
        **state,
        "knowledge_result": policy_result,
        "policy_result": policy_result,
    }


def policy_validity_node(state: GovRAGState) -> GovRAGState:
    """
    普通模式下跳过 Policy Validity Agent；
    开发者调试模式下完整执行。
    """
    debug_mode = state.get("debug_mode", False)

    if not debug_mode:
        policy_validity_result = {
            "skipped": True,
            "overall_risk_level": "not_checked",
            "overall_score": None,
            "summary": "快速模式下未执行政策时效与来源可靠性判断。",
            "warnings": [],
            "suggestions": [],
            "source_checks": [],
        }

        return {
            **state,
            "policy_validity_result": policy_validity_result,
        }

    question = state["question"]
    intent_result = state["intent_result"]
    policy_result = state.get("policy_result", {})

    policy_validity_result = policy_validity_agent(
        question=question,
        intent_result=intent_result,
        policy_result=policy_result,
    )

    return {
        **state,
        "policy_validity_result": policy_validity_result,
    }


def should_run_service_agents(state: GovRAGState) -> bool:
    router_result = state.get("source_router_result", {})
    return bool(router_result.get("should_run_service_agents", False))


def agency_node(state: GovRAGState) -> GovRAGState:
    question = state["question"]
    intent_result = state["intent_result"]

    agency_result = agency_agent(
        question=question,
        intent_result=intent_result,
    )

    return {
        **state,
        "agency_result": agency_result,
    }


def skip_agency_node(state: GovRAGState) -> GovRAGState:
    return {
        **state,
        "agency_result": {
            "should_run": False,
            "skipped": True,
            "agencies": [],
            "agency_info": "动态路由判断当前问题不需要办理机构查询。",
            "summary": "已跳过办理机构查询。",
        },
    }


def eligibility_node(state: GovRAGState) -> GovRAGState:
    question = state["question"]
    intent_result = state["intent_result"]
    policy_result = state.get("policy_result", {})

    eligibility_result = eligibility_agent(
        question=question,
        intent_result=intent_result,
        policy_result=policy_result,
    )

    return {
        **state,
        "eligibility_result": eligibility_result,
    }


def skip_eligibility_node(state: GovRAGState) -> GovRAGState:
    return {
        **state,
        "eligibility_result": {
            "should_run": False,
            "skipped": True,
            "eligibility": "动态路由判断当前问题不需要资格条件判断。",
            "judgement": "已跳过资格判断",
        },
    }


def procedure_node(state: GovRAGState) -> GovRAGState:
    question = state["question"]
    intent_result = state["intent_result"]
    policy_result = state.get("policy_result", {})

    procedure_result = procedure_agent(
        question=question,
        intent_result=intent_result,
        policy_result=policy_result,
    )

    return {
        **state,
        "procedure_result": procedure_result,
    }


def skip_procedure_node(state: GovRAGState) -> GovRAGState:
    return {
        **state,
        "procedure_result": {
            "procedure": "动态路由判断当前问题不需要办理流程整理。",
            "has_procedure": False,
            "skipped": True,
        },
    }


def material_node(state: GovRAGState) -> GovRAGState:
    question = state["question"]
    intent_result = state["intent_result"]
    policy_result = state.get("policy_result", {})

    material_result = material_agent(
        question=question,
        intent_result=intent_result,
        policy_result=policy_result,
    )

    return {
        **state,
        "material_result": material_result,
    }


def skip_material_node(state: GovRAGState) -> GovRAGState:
    return {
        **state,
        "material_result": {
            "materials": "动态路由判断当前问题不需要材料清单整理。",
            "has_materials": False,
            "skipped": True,
        },
    }


def answer_node(state: GovRAGState) -> GovRAGState:
    question = state["question"]
    intent_result = state["intent_result"]
    service_recommendation_result = state.get("service_recommendation_result", {})
    policy_result = state.get("policy_result", {})
    policy_validity_result = state.get("policy_validity_result", {})
    agency_result = state.get("agency_result", {})
    eligibility_result = state.get("eligibility_result", {})
    procedure_result = state.get("procedure_result", {})
    material_result = state.get("material_result", {})

    answer_result = answer_agent(
        question=question,
        intent_result=intent_result,
        policy_result=policy_result,
        service_recommendation_result=service_recommendation_result,
        policy_validity_result=policy_validity_result,
        agency_result=agency_result,
        eligibility_result=eligibility_result,
        procedure_result=procedure_result,
        material_result=material_result,
    )

    return {
        **state,
        "answer_result": answer_result,
        "final_answer": answer_result.get("answer", ""),
        "sources": answer_result.get("used_sources", []),
        "evidence_result": answer_result.get("evidence_result", {}),
    }


def verify_node(state: GovRAGState) -> GovRAGState:
    """
    普通模式下跳过 Verify Agent；
    开发者调试模式下完整执行。
    """
    debug_mode = state.get("debug_mode", False)

    if not debug_mode:
        verify_result = {
            "skipped": True,
            "passed": True,
            "score": None,
            "risk_level": "not_checked",
            "issues": [],
            "summary": "快速模式下未执行回答校验。",
            "need_rewrite": False,
            "rewrite_instruction": "",
        }

        return {
            **state,
            "verify_result": verify_result,
            "verification_score": 0,
            "risk_level": "not_checked",
        }

    question = state["question"]
    intent_result = state.get("intent_result", {})
    service_recommendation_result = state.get("service_recommendation_result", {})
    policy_result = state.get("policy_result", {})
    policy_validity_result = state.get("policy_validity_result", {})
    agency_result = state.get("agency_result", {})
    eligibility_result = state.get("eligibility_result", {})
    procedure_result = state.get("procedure_result", {})
    material_result = state.get("material_result", {})
    answer_result = state.get("answer_result", {})

    if state.get("need_clarification", False):
        verify_result = {
            "passed": True,
            "score": 100,
            "risk_level": "low",
            "issues": [],
            "summary": "当前问题需要用户补充信息，系统已生成追问或事项推荐，不涉及严格政策结论校验。",
            "need_rewrite": False,
            "rewrite_instruction": "",
        }
    else:
        verify_result = verify_agent(
            question=question,
            intent_result=intent_result,
            service_recommendation_result=service_recommendation_result,
            policy_result=policy_result,
            policy_validity_result=policy_validity_result,
            agency_result=agency_result,
            eligibility_result=eligibility_result,
            procedure_result=procedure_result,
            material_result=material_result,
            answer_result=answer_result,
        )

    return {
        **state,
        "verify_result": verify_result,
        "verification_score": verify_result.get("score", 0) or 0,
        "risk_level": verify_result.get("risk_level", "medium"),
    }


def should_continue_after_recommendation(state: GovRAGState) -> str:
    """
    信息不足时，直接进入 Answer；
    信息明确时，继续政策检索。
    """
    if state.get("need_clarification", False):
        return "answer"

    return "policy"


def route_after_source_router(state: GovRAGState) -> str:
    if should_run_service_agents(state):
        return "service_recommendation"

    return "policy"


def route_agency(state: GovRAGState) -> str:
    if not should_run_service_agents(state):
        return "skip_agency"

    if state.get("debug_mode", False):
        return "agency"

    if should_run_agency_agent(state.get("intent_result", {})):
        return "agency"

    return "skip_agency"


def route_eligibility(state: GovRAGState) -> str:
    if not should_run_service_agents(state):
        return "skip_eligibility"

    if state.get("debug_mode", False):
        return "eligibility"

    if should_run_eligibility(
        question=state.get("question", ""),
        intent_result=state.get("intent_result", {}),
    ):
        return "eligibility"

    return "skip_eligibility"


def route_procedure(state: GovRAGState) -> str:
    if not should_run_service_agents(state):
        return "skip_procedure"

    if state.get("debug_mode", False):
        return "procedure"

    if should_run_procedure(
        question=state.get("question", ""),
        intent_result=state.get("intent_result", {}),
    ):
        return "procedure"

    return "skip_procedure"


def route_material(state: GovRAGState) -> str:
    if not should_run_service_agents(state):
        return "skip_material"

    if state.get("debug_mode", False):
        return "material"

    if should_run_material(
        question=state.get("question", ""),
        intent_result=state.get("intent_result", {}),
    ):
        return "material"

    return "skip_material"


def build_gov_graph():
    graph = StateGraph(GovRAGState)

    graph.add_node("intent", intent_node)
    graph.add_node("source_router", source_router_node)
    graph.add_node("service_recommendation", service_recommendation_node)
    graph.add_node("policy", knowledge_node)
    graph.add_node("policy_validity", policy_validity_node)
    graph.add_node("agency", agency_node)
    graph.add_node("skip_agency", skip_agency_node)
    graph.add_node("eligibility", eligibility_node)
    graph.add_node("skip_eligibility", skip_eligibility_node)
    graph.add_node("procedure", procedure_node)
    graph.add_node("skip_procedure", skip_procedure_node)
    graph.add_node("material", material_node)
    graph.add_node("skip_material", skip_material_node)
    graph.add_node("answer", answer_node)
    graph.add_node("verify", verify_node)

    graph.add_edge(START, "intent")
    graph.add_edge("intent", "source_router")
    graph.add_conditional_edges(
        "source_router",
        route_after_source_router,
        {
            "service_recommendation": "service_recommendation",
            "policy": "policy",
        },
    )

    graph.add_conditional_edges(
        "service_recommendation",
        should_continue_after_recommendation,
        {
            "policy": "policy",
            "answer": "answer",
        },
    )

    graph.add_edge("policy", "policy_validity")

    graph.add_conditional_edges(
        "policy_validity",
        route_agency,
        {
            "agency": "agency",
            "skip_agency": "skip_agency",
        },
    )
    graph.add_conditional_edges(
        "agency",
        route_eligibility,
        {
            "eligibility": "eligibility",
            "skip_eligibility": "skip_eligibility",
        },
    )
    graph.add_conditional_edges(
        "skip_agency",
        route_eligibility,
        {
            "eligibility": "eligibility",
            "skip_eligibility": "skip_eligibility",
        },
    )
    graph.add_conditional_edges(
        "eligibility",
        route_procedure,
        {
            "procedure": "procedure",
            "skip_procedure": "skip_procedure",
        },
    )
    graph.add_conditional_edges(
        "skip_eligibility",
        route_procedure,
        {
            "procedure": "procedure",
            "skip_procedure": "skip_procedure",
        },
    )
    graph.add_conditional_edges(
        "procedure",
        route_material,
        {
            "material": "material",
            "skip_material": "skip_material",
        },
    )
    graph.add_conditional_edges(
        "skip_procedure",
        route_material,
        {
            "material": "material",
            "skip_material": "skip_material",
        },
    )
    graph.add_edge("material", "answer")
    graph.add_edge("skip_material", "answer")
    graph.add_edge("answer", "verify")
    graph.add_edge("verify", END)

    return graph.compile()


gov_graph = build_gov_graph()


def run_gov_graph(
    question: str,
    debug_mode: bool = False,
    source_ids: List[str] = None,
) -> GovRAGState:
    initial_state: GovRAGState = {
        "question": question,
        "debug_mode": debug_mode,
        "requested_source_ids": source_ids or [],
    }

    result = gov_graph.invoke(initial_state)

    return result


def print_graph_result(result: GovRAGState):
    print("\n========== 用户问题 ==========")
    print(result.get("question", ""))

    print("\n========== Debug Mode ==========")
    print(result.get("debug_mode", False))

    print("\n========== Intent Agent ==========")
    print(result.get("intent_result", {}))

    print("\n========== Service Recommendation Agent ==========")
    recommendation = result.get("service_recommendation_result", {})
    if recommendation:
        print(f"是否触发：{recommendation.get('should_run')}")
        print(f"摘要：{recommendation.get('summary')}")
        recommendations = recommendation.get("recommendations", [])
        for item in recommendations:
            print(f"- {item.get('service_name')} | {item.get('reason')}")
    else:
        print("未执行事项推荐。")

    print("\n========== Knowledge Agent ==========")
    policy_result = result.get("policy_result", {})
    if policy_result:
        print("增强检索 Query:")
        print(policy_result.get("enhanced_query", ""))
        print("Source distribution:")
        print(policy_result.get("source_distribution", {}))

        print("\n检索来源:")
        for source in policy_result.get("policy_sources", []):
            print(
                f"- 资料{source.get('index')}: "
                f"{source.get('title')} | "
                f"{source.get('region')} | "
                f"{source.get('topic')}"
            )
    else:
        print("未执行政策检索。")

    print("\n========== Policy Validity Agent ==========")
    print(result.get("policy_validity_result", {}))

    print("\n========== Agency Agent ==========")
    agency_result = result.get("agency_result", {})
    if agency_result:
        print(f"是否触发：{agency_result.get('should_run')}")
        print(f"摘要：{agency_result.get('summary')}")
        if agency_result.get("should_run"):
            print(agency_result.get("agency_info", ""))
    else:
        print("未执行办理机构查询。")

    print("\n========== Eligibility Agent ==========")
    print(result.get("eligibility_result", {}))

    print("\n========== Procedure Agent ==========")
    print(result.get("procedure_result", {}))

    print("\n========== Material Agent ==========")
    print(result.get("material_result", {}))

    print("\n========== Final Answer ==========")
    print(result.get("final_answer", ""))

    print("\n========== Verification Agent ==========")
    print(result.get("verify_result", {}))


if __name__ == "__main__":
    test_questions = [
        "我是河北应届毕业生，还没找到工作，档案和户口不知道怎么办。",
        "我毕业后准备回石家庄工作，单位不能接收档案，还想看看有没有补贴可以申请。",
        "我是2025届毕业生，暂时没就业，社保和档案需要怎么处理？",
        "石家庄毕业生档案接收怎么办理？",
        "我是2025届本科毕业生，在石家庄企业工作，社保交了6个月，可以申请安家补贴吗？",
        "高校毕业生档案可以自己携带吗？",
        "现在还需要就业报到证吗？",
        "创业担保贷款有什么条件？",
    ]

    for question in test_questions:
        print("\n\n========================================")
        print(f"测试问题：{question}")
        print("========================================")

        result = run_gov_graph(question, debug_mode=True)
        print_graph_result(result)
