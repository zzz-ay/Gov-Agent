from typing import Dict, Any, List

from app.mcp.agency_tool import (
    agency_tool,
    format_agencies_for_answer,
)


def should_run_agency_agent(intent_result: Dict[str, Any]) -> bool:
    """
    判断当前问题是否适合查询办理机构。

    这些事项通常需要给出办理机构：
    - 档案接收/档案转递
    - 就业补贴
    - 社保补贴
    - 创业担保贷款
    - 户籍迁移
    """
    intent = intent_result.get("intent", "general")

    service_related_intents = {
        "archive_transfer",
        "employment_policy",
        "employment_subsidy",
        "social_security",
        "entrepreneurship",
        "hukou",
    }

    return intent in service_related_intents


def agency_agent(
    question: str,
    intent_result: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Agency Agent 主函数。

    输入：
    - question: 用户原始问题
    - intent_result: Intent Agent 输出

    输出：
    {
        "should_run": True / False,
        "agencies": [...],
        "agency_info": "...",
        "summary": "..."
    }
    """

    if not should_run_agency_agent(intent_result):
        return {
            "should_run": False,
            "agencies": [],
            "agency_info": "当前问题不属于明显需要查询办理机构的事项。",
            "summary": "未触发办理机构查询。",
        }

    result = agency_tool(
        intent_result=intent_result,
        top_k=3,
    )

    agencies = result.get("agencies", [])

    agency_info = format_agencies_for_answer(agencies)

    return {
        "should_run": result.get("should_use", False),
        "agencies": agencies,
        "agency_info": agency_info,
        "summary": result.get("summary", ""),
    }


if __name__ == "__main__":
    from app.agents.intent_agent import classify_intent

    test_questions = [
        "石家庄毕业生档案接收怎么办理？",
        "我是2025届本科毕业生，在石家庄企业工作，社保交了6个月，可以申请安家补贴吗？",
        "河北高校毕业生可以申请哪些就业补贴？",
        "灵活就业社保补贴怎么申请？",
        "创业担保贷款有什么条件？",
        "河北高校毕业生户口迁移怎么办？",
        "现在还需要就业报到证吗？",
    ]

    for question in test_questions:
        print("\n\n========================================")
        print(f"测试问题：{question}")
        print("========================================")

        intent_result = classify_intent(question)

        result = agency_agent(
            question=question,
            intent_result=intent_result,
        )

        print("\n========== Intent Agent ==========")
        print(intent_result)

        print("\n========== Agency Agent ==========")
        print(f"是否触发办理机构查询：{result['should_run']}")
        print(f"摘要：{result['summary']}")
        print("\n办理机构信息：")
        print(result["agency_info"])