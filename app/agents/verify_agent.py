from typing import Dict, Any, List
import json

from langchain_core.prompts import ChatPromptTemplate

from app.llm.provider import get_chat_llm


VERIFY_SYSTEM_PROMPT = """
你是 GOV-RAG 政务问答系统中的“回答校验智能体”。

你的任务：
根据用户问题、意图识别结果、政务事项推荐结果、政策检索资料、政策时效与来源可靠性结果、办理机构结果、资格判断结果、办理流程结果、材料清单结果和最终回答，
检查最终回答是否存在不可靠、无依据、过度扩展或不适合政务场景的问题。

校验重点：
1. 是否出现政策资料中没有明确说明的政策名称、适用对象、金额、材料、流程、办理时限。
2. 是否出现办理机构结果中没有明确说明的机构名称、地址、电话、官网、办公时间。
3. 如果最终回答包含政务事项推荐，是否与政务事项推荐结果一致。
4. 是否出现政策资料中没有明确说明的法律后果、风险后果、责任后果或办理后果。
5. 如果最终回答涉及资格判断，是否与资格判断结果一致。
6. 是否把“可能符合”说成了“一定符合”。
7. 是否遗漏资格判断中的“仍需确认的信息”。
8. 如果政策时效与来源可靠性结果提示中高风险，最终回答是否做了必要提醒。
9. 涉及地区差异时，是否提醒“以当地主管部门或政务服务窗口最新要求为准”。
10. 涉及金额、材料、流程时，是否有政策检索资料支撑。
11. 参考来源是否与回答内容相关。
12. 是否存在回答过度扩展、答非所问或遗漏核心问题。
13. 是否适合高校毕业生阅读，是否清晰、准确、可追溯。

请严格输出 JSON，不要输出 Markdown，不要输出解释性段落。

输出格式示例：
{{
  "passed": true,
  "score": 90,
  "risk_level": "low",
  "issues": [
    {{
      "type": "unsupported_claim",
      "description": "回答中提到了资料未明确说明的内容",
      "suggestion": "删除该表述或改为当前检索资料未明确说明"
    }}
  ],
  "summary": "回答整体可靠，仅有少量可优化表述",
  "need_rewrite": false,
  "rewrite_instruction": ""
}}
"""


def get_verify_llm():
    return get_chat_llm(temperature=0)


def format_sources_for_verify(policy_sources: List[Dict[str, Any]]) -> str:
    formatted_docs = []

    for source in policy_sources:
        index = source.get("index", "")
        title = source.get("title", "")
        source_org = source.get("source_org", "")
        publish_date = source.get("publish_date", "")
        region = source.get("region", "")
        topic = source.get("topic", "")
        source_url = source.get("source_url", "")
        page = source.get("page", "")
        content = source.get("content", "")

        page_info = f"页码：{page}" if page != "" else "页码：无"

        formatted = f"""
【政策资料{index}】
标题：{title}
发布机构：{source_org}
发布时间：{publish_date}
适用地区：{region}
主题：{topic}
来源链接：{source_url}
{page_info}

正文片段：
{content}
"""
        formatted_docs.append(formatted.strip())

    return "\n\n".join(formatted_docs)


def parse_json_response(raw_text: str) -> Dict[str, Any]:
    text = raw_text.strip()

    if text.startswith("```json"):
        text = text.replace("```json", "").replace("```", "").strip()
    elif text.startswith("```"):
        text = text.replace("```", "").strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {
            "passed": False,
            "score": 60,
            "risk_level": "medium",
            "issues": [
                {
                    "type": "verify_parse_error",
                    "description": "校验模型输出不是合法 JSON",
                    "suggestion": "检查 Verify Agent Prompt 或重新运行校验",
                }
            ],
            "summary": "校验结果解析失败，建议人工查看最终回答。",
            "need_rewrite": False,
            "rewrite_instruction": "",
            "raw_output": raw_text,
        }


def verify_agent(
    question: str,
    intent_result: Dict[str, Any],
    service_recommendation_result: Dict[str, Any],
    policy_result: Dict[str, Any],
    policy_validity_result: Dict[str, Any],
    agency_result: Dict[str, Any],
    eligibility_result: Dict[str, Any],
    procedure_result: Dict[str, Any],
    material_result: Dict[str, Any],
    answer_result: Dict[str, Any],
) -> Dict[str, Any]:
    policy_sources = policy_result.get("policy_sources", [])
    final_answer = answer_result.get("answer", "")

    if not final_answer:
        return {
            "passed": False,
            "score": 0,
            "risk_level": "high",
            "issues": [
                {
                    "type": "empty_answer",
                    "description": "最终回答为空",
                    "suggestion": "检查 Answer Agent 是否正常生成回答",
                }
            ],
            "summary": "最终回答为空，无法通过校验。",
            "need_rewrite": True,
            "rewrite_instruction": "请重新生成回答。",
        }

    policy_context = format_sources_for_verify(policy_sources)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", VERIFY_SYSTEM_PROMPT),
            (
                "human",
                """
用户问题：
{question}

意图识别结果：
{intent_result}

政务事项推荐结果：
{service_recommendation_result}

政策检索资料：
{policy_context}

政策时效与来源可靠性结果：
{policy_validity_result}

办理机构结果：
{agency_result}

资格判断结果：
{eligibility_result}

办理流程结果：
{procedure_result}

材料清单结果：
{material_result}

最终回答：
{final_answer}

请对最终回答进行可靠性校验。
""",
            ),
        ]
    )

    llm = get_verify_llm()
    chain = prompt | llm

    response = chain.invoke(
        {
            "question": question,
            "intent_result": str(intent_result),
            "service_recommendation_result": str(service_recommendation_result),
            "policy_context": policy_context,
            "policy_validity_result": str(policy_validity_result),
            "agency_result": str(agency_result),
            "eligibility_result": str(eligibility_result),
            "procedure_result": str(procedure_result),
            "material_result": str(material_result),
            "final_answer": final_answer,
        }
    )

    result = parse_json_response(response.content)

    result.setdefault("passed", False)
    result.setdefault("score", 0)
    result.setdefault("risk_level", "medium")
    result.setdefault("issues", [])
    result.setdefault("summary", "")
    result.setdefault("need_rewrite", False)
    result.setdefault("rewrite_instruction", "")

    return result


if __name__ == "__main__":
    from app.agents.intent_agent import classify_intent
    from app.agents.service_recommendation_agent import service_recommendation_agent
    from app.agents.policy_agent import policy_agent
    from app.agents.policy_validity_agent import policy_validity_agent
    from app.agents.agency_agent import agency_agent
    from app.agents.eligibility_agent import eligibility_agent
    from app.agents.procedure_agent import procedure_agent
    from app.agents.material_agent import material_agent
    from app.agents.answer_agent import answer_agent

    test_questions = [
        "我是河北应届毕业生，还没找到工作，档案和户口不知道怎么办。",
        "石家庄毕业生档案接收怎么办理？",
        "我是2025届本科毕业生，在石家庄企业工作，社保交了6个月，可以申请安家补贴吗？",
        "高校毕业生档案可以自己携带吗？",
        "河北高校毕业生可以申请哪些就业补贴？",
    ]

    for question in test_questions:
        print("\n\n========================================")
        print(f"测试问题：{question}")
        print("========================================")

        intent_result = classify_intent(question)
        service_recommendation_result = service_recommendation_agent(question, intent_result)
        policy_result = policy_agent(question, intent_result)
        policy_validity_result = policy_validity_agent(question, intent_result, policy_result)
        agency_result = agency_agent(question, intent_result)
        eligibility_result = eligibility_agent(question, intent_result, policy_result)
        procedure_result = procedure_agent(question, intent_result, policy_result)
        material_result = material_agent(question, intent_result, policy_result)

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

        print("\n========== 最终回答 ==========")
        print(answer_result.get("answer", ""))

        print("\n========== Verify Agent 校验结果 ==========")
        print(json.dumps(verify_result, ensure_ascii=False, indent=2))
