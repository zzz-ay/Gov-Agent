from typing import Dict, Any, List
import json

from langchain_core.prompts import ChatPromptTemplate

from app.llm.provider import get_chat_llm

from app.agents.intent_agent import classify_intent


SERVICE_CATALOG = {
    "graduation_registration": {
        "name": "毕业去向登记",
        "description": "适用于毕业生在全国高校毕业生毕业去向登记系统中登记就业、升学、未就业、档案、户口等信息。",
    },
    "archive_transfer": {
        "name": "档案转递 / 档案接收",
        "description": "适用于高校毕业生档案转回户籍地、就业地、人才中心或单位接收等事项。",
    },
    "hukou": {
        "name": "户籍迁移 / 落户咨询",
        "description": "适用于毕业生户口迁回原籍、迁入就业地、人才市场集体户或办理落户等事项。",
    },
    "employment_subsidy": {
        "name": "就业 / 人才补贴资格咨询",
        "description": "适用于求职创业补贴、一次性安家补贴、就业见习补贴、人才补贴等事项。",
    },
    "social_security": {
        "name": "灵活就业社保补贴",
        "description": "适用于毕业生灵活就业、自行缴纳社保、申请社会保险补贴等事项。",
    },
    "entrepreneurship": {
        "name": "创业担保贷款 / 创业扶持",
        "description": "适用于毕业生自主创业、创业担保贷款、创业补贴、创业扶持政策等事项。",
    },
    "employment_policy": {
        "name": "未就业毕业生就业服务",
        "description": "适用于未就业毕业生就业服务登记、就业见习、职业指导、公共就业服务等事项。",
    },
    "registration_certificate": {
        "name": "就业报到证取消衔接",
        "description": "适用于咨询就业报到证是否还需要、取消报到证后档案、落户、就业手续如何衔接等问题。",
    },
}


SERVICE_RECOMMENDATION_PROMPT = """
你是 GOV-RAG 政务服务系统中的“政务事项推荐智能体”。

你的任务：
根据用户问题和意图识别结果，判断用户可能涉及哪些高校毕业生政务服务事项，并给出推荐事项列表。

适用场景：
- 用户没有问一个单一明确的问题，而是在描述自己的毕业、就业、档案、户口、补贴、社保、创业等综合情况。
- 用户说“不知道怎么办”“需要办哪些”“接下来该做什么”“帮我规划一下”“有哪些事项要处理”。
- 用户同时提到多个主题，如档案、户口、就业、补贴、社保、创业等。

可推荐事项范围：
1. graduation_registration：毕业去向登记
2. archive_transfer：档案转递 / 档案接收
3. hukou：户籍迁移 / 落户咨询
4. employment_subsidy：就业 / 人才补贴资格咨询
5. social_security：灵活就业社保补贴
6. entrepreneurship：创业担保贷款 / 创业扶持
7. employment_policy：未就业毕业生就业服务
8. registration_certificate：就业报到证取消衔接

重要原则：
1. 只推荐和用户描述明显相关的事项，不要为了凑数量而推荐无关事项。
2. 如果用户问题非常具体，例如“现在还需要就业报到证吗？”，可以只推荐 0-1 个事项。
3. 如果用户描述较宽泛，可以推荐 2-5 个事项。
4. 每个推荐事项要说明推荐原因。
5. 如果需要用户补充信息，要列出下一步建议追问的问题。
6. 不要编造具体政策金额、办理材料和办理流程。

请严格输出 JSON，不要输出 Markdown，不要输出解释性段落。

输出格式示例：
{{
  "should_run": true,
  "recommendations": [
    {{
      "service_type": "archive_transfer",
      "service_name": "档案转递 / 档案接收",
      "priority": "high",
      "reason": "用户提到毕业后档案不知道如何处理，需先确认档案接收地和转递方式。",
      "suggested_question": "你的户籍地和就业地分别在哪里？单位是否具有人事档案管理权限？"
    }}
  ],
  "summary": "用户描述涉及多个毕业后政务事项，建议优先处理档案和毕业去向登记。",
  "need_more_info": true,
  "follow_up_questions": [
    "你的户籍地是哪里？",
    "你是否已经签约就业单位？"
  ]
}}
"""


def get_recommendation_llm():
    """
    创建事项推荐用 LLM。
    """
    return get_chat_llm(temperature=0.1)


def should_run_service_recommendation(
    question: str,
    intent_result: Dict[str, Any],
) -> bool:
    """
    判断是否需要运行事项推荐。

    如果用户问题很宽泛、同时涉及多个事项、或表达“不知道怎么办”，则触发。
    """
    q = question.strip()

    broad_keywords = [
        "不知道怎么办",
        "不知道怎么处理",
        "怎么处理",
        "接下来该怎么办",
        "接下来怎么做",
        "需要办哪些",
        "要办哪些",
        "有哪些事项",
        "帮我规划",
        "帮我看看",
        "给我一个方案",
        "毕业后怎么办",
        "毕业以后怎么办",
        "还没找到工作",
        "暂未就业",
        "未就业",
    ]

    multi_topic_keywords = [
        "档案",
        "户口",
        "户籍",
        "落户",
        "就业",
        "补贴",
        "社保",
        "创业",
        "报到证",
        "毕业去向",
    ]

    if any(kw in q for kw in broad_keywords):
        return True

    hit_count = sum(1 for kw in multi_topic_keywords if kw in q)

    if hit_count >= 2:
        return True

    # 如果 Intent Agent 已经判断需要追问，也可以推荐可能涉及的事项
    if intent_result.get("need_clarification", False):
        intent = intent_result.get("intent", "")
        if intent in {
            "archive_transfer",
            "employment_subsidy",
            "social_security",
            "entrepreneurship",
            "hukou",
            "employment_policy",
        }:
            return True

    return False


def parse_json_response(raw_text: str) -> Dict[str, Any]:
    """
    解析模型输出 JSON。
    """
    text = raw_text.strip()

    if text.startswith("```json"):
        text = text.replace("```json", "").replace("```", "").strip()
    elif text.startswith("```"):
        text = text.replace("```", "").strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {
            "should_run": False,
            "recommendations": [],
            "summary": "事项推荐结果解析失败，未生成推荐事项。",
            "need_more_info": False,
            "follow_up_questions": [],
            "raw_output": raw_text,
        }


def normalize_recommendations(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    规范化推荐结果，避免模型输出未知 service_type。
    """
    recommendations = result.get("recommendations", [])

    if not isinstance(recommendations, list):
        recommendations = []

    normalized = []

    for item in recommendations:
        if not isinstance(item, dict):
            continue

        service_type = item.get("service_type", "")

        if service_type not in SERVICE_CATALOG:
            continue

        catalog_item = SERVICE_CATALOG[service_type]

        normalized.append(
            {
                "service_type": service_type,
                "service_name": item.get("service_name") or catalog_item["name"],
                "priority": item.get("priority", "medium"),
                "reason": item.get("reason", ""),
                "suggested_question": item.get("suggested_question", ""),
            }
        )

    result["recommendations"] = normalized
    result["should_run"] = bool(result.get("should_run", False)) and len(normalized) > 0
    result.setdefault("summary", "")
    result.setdefault("need_more_info", False)
    result.setdefault("follow_up_questions", [])

    return result


def format_recommendations_for_answer(recommendation_result: Dict[str, Any]) -> str:
    """
    将推荐事项格式化为 Answer Agent 可读文本。
    前端展示时不显示 priority，只保留事项名称、推荐原因和建议补充信息。
    """
    if not recommendation_result or not recommendation_result.get("should_run"):
        return "当前问题未触发政务事项推荐。"

    recommendations = recommendation_result.get("recommendations", [])

    if not recommendations:
        return "当前未生成推荐事项。"

    lines = [
        f"推荐摘要：{recommendation_result.get('summary', '')}",
        "",
        "推荐事项：",
    ]

    for idx, item in enumerate(recommendations, start=1):
        service_name = item.get("service_name", "")
        reason = item.get("reason", "")
        suggested_question = item.get("suggested_question", "")

        lines.append(f"{idx}. {service_name}")

        if reason:
            lines.append(f"   推荐原因：{reason}")

        if suggested_question:
            lines.append(f"   建议补充：{suggested_question}")

    follow_up_questions = recommendation_result.get("follow_up_questions", [])

    if follow_up_questions:
        lines.append("")
        lines.append("建议补充信息：")
        for q in follow_up_questions:
            lines.append(f"- {q}")

    return "\n".join(lines)


def service_recommendation_agent(
    question: str,
    intent_result: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Service Recommendation Agent 主函数。

    输出：
    {
        "should_run": True / False,
        "recommendations": [...],
        "summary": "...",
        "need_more_info": True / False,
        "follow_up_questions": [...]
    }
    """
    run_flag = should_run_service_recommendation(
        question=question,
        intent_result=intent_result,
    )

    if not run_flag:
        return {
            "should_run": False,
            "recommendations": [],
            "summary": "当前问题较为具体，未触发政务事项推荐。",
            "need_more_info": False,
            "follow_up_questions": [],
        }

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SERVICE_RECOMMENDATION_PROMPT),
            (
                "human",
                """
用户问题：
{question}

意图识别结果：
{intent_result}

可推荐事项说明：
{service_catalog}

请判断用户可能涉及哪些政务事项，并输出 JSON。
""",
            ),
        ]
    )

    llm = get_recommendation_llm()
    chain = prompt | llm

    response = chain.invoke(
        {
            "question": question,
            "intent_result": str(intent_result),
            "service_catalog": json.dumps(SERVICE_CATALOG, ensure_ascii=False, indent=2),
        }
    )

    result = parse_json_response(response.content)
    result = normalize_recommendations(result)

    return result


if __name__ == "__main__":
    test_questions = [
        "我是河北应届毕业生，还没找到工作，档案和户口不知道怎么办。",
        "我毕业后准备回石家庄工作，单位不能接收档案，还想看看有没有补贴可以申请。",
        "我是2025届毕业生，暂时没就业，社保和档案需要怎么处理？",
        "石家庄毕业生档案接收怎么办理？",
        "现在还需要就业报到证吗？",
        "创业担保贷款有什么条件？",
    ]

    for question in test_questions:
        print("\n\n========================================")
        print(f"测试问题：{question}")
        print("========================================")

        intent_result = classify_intent(question)

        result = service_recommendation_agent(
            question=question,
            intent_result=intent_result,
        )

        print("\n========== Intent Agent ==========")
        print(intent_result)

        print("\n========== Service Recommendation Agent ==========")
        print(json.dumps(result, ensure_ascii=False, indent=2))

        print("\n========== 格式化输出 ==========")
        print(format_recommendations_for_answer(result))
