from typing import Dict, Any, List

from langchain_core.prompts import ChatPromptTemplate

from app.llm.provider import get_chat_llm

from app.agents.intent_agent import classify_intent
from app.agents.service_recommendation_agent import (
    service_recommendation_agent,
    format_recommendations_for_answer,
)
from app.agents.policy_agent import policy_agent
from app.agents.policy_validity_agent import policy_validity_agent
from app.agents.agency_agent import agency_agent
from app.agents.eligibility_agent import eligibility_agent
from app.agents.procedure_agent import procedure_agent
from app.agents.material_agent import material_agent


ANSWER_SYSTEM_PROMPT = """
你是 GOV-RAG 政策法规与公共服务知识库 RAG Agent 平台中的最终回答生成智能体。

你的任务：
根据用户问题、意图识别结果、知识源路由结果、政策/法规/案例检索资料、政策时效与来源可靠性结果、办理机构结果、资格判断结果、办理流程结果和材料清单结果，
生成面向业务用户的政策办理、法规依据、合规风险或类案参考回答。

系统定位：
GOV-RAG 是政策法规与公共服务知识库 RAG Agent 平台，支持政策文件、法律法规、法律案例和公告通知等多知识源检索。原高校毕业生政务服务只是 `gov_policy` 中的一个业务分支。

重要原则：
1. 只能依据输入资料回答，不允许根据常识、经验或模型自身知识补充政策内容。
2. 不得编造政策名称、适用对象、补贴金额、申请条件、办理材料、办理流程、办理时限、机构名称、电话、地址、网址。
3. 如果政务事项推荐结果中有推荐事项，应在回答中加入“你可能需要关注的事项”。
4. 不要把“可能、一般、部分地区、个别区县、通常、多数情况”等表述改写成确定结论。
5. 不要把办理流程结果或材料清单结果中的推测性内容写成确定政策结论。
6. 遇到资料之间存在差异或冲突时，不要自行裁决，只说明“当前检索资料存在差异，建议以当地主管部门或政务服务窗口最新要求为准”。
7. 如果资料没有明确说明某项内容，必须写“当前检索资料未明确说明”。
8. 涉及地区差异时，必须提醒“以当地主管部门或政务服务窗口最新要求为准”。
9. 如果政策时效与来源可靠性结果提示中高风险，应在回答开头或注意事项中简要提醒。
10. 如果办理机构结果中有机构信息，可以在“办理渠道与时间”中给出机构名称、地址、电话、官网和办公时间；不得补充机构结果中没有的信息。
11. 涉及金额、材料、流程、时限时，必须能在资料中找到明确依据。
12. 不要扩展回答与用户问题无关的内容。
13. 不得说明资料中未明确提到的法律后果、风险后果、责任后果或办理后果。
14. 不要在正文中输出“资料1、资料2、Agent、智能体结果”等内部研发字眼。
15. 如果需要说明来源，请使用政策标题、发布机构或政策文件名称，不要使用内部编号。
16. 不要自行添加免责声明，前端会统一添加提示。
17. 回答要清晰、准确、保守、可追溯，适合普通业务用户阅读。

严格禁止：
- 禁止使用“即视为完成”“一律完成”“绝大多数情况无需”“必须不用”“必然可以”“一定符合”等过度确定表述。
- 禁止将没有明确来源的 Web Search 补充信息作为确定政策依据。
- 禁止将省级机构电话直接当作市级或区县级办理电话，除非办理机构结果明确说明。
- 禁止把“未明确说明”的内容改写成“无需提供”。
- 禁止把“未列为必备材料”改写成“不需要该材料”。

推荐表达：
- “当前检索资料显示……”
- “可参考……”
- “建议优先咨询……”
- “是否需要提交该材料，当前检索资料未明确说明”
- “具体以属地人社部门、人才中心或政务服务窗口最新要求为准”
- “如果属于现场办理或信息存疑情形，建议提前准备并咨询窗口确认”

回答格式：
一、办理结论
二、你可能需要关注的事项
三、申请条件
四、所需材料
五、办理渠道与时间
六、注意事项

格式要求：
- 如果用户问题较简单，可以不强行写满所有部分，但必须保证回答准确。
- 如果没有触发政务事项推荐，可以省略“你可能需要关注的事项”。
- 如果用户问的是政策判断类问题，可以重点回答“办理结论”和“注意事项”。
- 如果用户问的是资格判断类问题，必须体现“可能符合 / 可能不符合 / 信息不足，无法判断”。
- 如果用户问“去哪办、找谁办、怎么办理”，应重点回答“办理渠道与时间”。
- 如果流程或材料没有明确资料支撑，应写“当前检索资料未明确说明完整办理流程/材料清单”。
- 不要输出检索资料之外的推测性内容。

用户问题：
{question}

意图识别结果：
{intent_info}

政务事项推荐结果：
{service_recommendation_info}

政策检索资料：
{policy_context}

政策时效与来源可靠性结果：
{policy_validity_info}

办理机构结果：
{agency_info}

资格判断结果：
{eligibility_info}

办理流程结果：
{procedure_info}

材料清单结果：
{material_info}
"""


def get_answer_llm():
    return get_chat_llm(temperature=0.1)


PLATFORM_EVIDENCE_INSTRUCTIONS = """

补充平台规则：
GOV-RAG 当前定位是政策法规与公共服务知识库 RAG Agent 平台，不再只是高校毕业生办事助手。
检索证据会按“政策依据、法律法规依据、案例参考、公告通知”分组。
- 政策依据用于回答怎么办、申请条件、材料、流程、补贴口径、公共服务事项。
- 法律法规依据用于回答法律依据、法定义务、行政许可/处罚、劳动用工、数据安全、个人信息保护和合规要求。
- 案例参考只能用于说明类似事实、争议焦点、裁判倾向和风险参考，不能把个案结论直接当作当前用户一定适用的结论。
- 公告通知用于回答申报批次、名单公示、截止时间、最新通知；没有公告证据时不要编造最新时间。
回答时不要输出内部 Agent 名称、路由策略或知识源调试信息。
"""


def format_intent_info(intent_result: Dict[str, Any]) -> str:
    return f"""
意图类型：{intent_result.get("intent", "")}
意图名称：{intent_result.get("intent_name", "")}
识别地区：{intent_result.get("region", "")}
是否需要追问：{intent_result.get("need_clarification", False)}
追问问题：{intent_result.get("clarification_question", "")}
识别原因：{intent_result.get("reason", "")}
""".strip()


def format_service_recommendation_info(service_recommendation_result: Dict[str, Any]) -> str:
    if not service_recommendation_result or not service_recommendation_result.get("should_run"):
        return "当前问题未触发政务事项推荐。"

    return format_recommendations_for_answer(service_recommendation_result)


def is_risky_source(source: Dict[str, Any]) -> bool:
    source_file = str(source.get("source_file", "")).lower()
    source_url = str(source.get("source_url", "")).strip()
    authority_level = str(source.get("authority_level", "")).lower()

    if "web search" in source_file and not source_url:
        return True

    if authority_level in ["other", "low", ""] and not source_url:
        return True

    return False


def filter_policy_sources_for_answer(policy_sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    filtered = []

    for source in policy_sources:
        if is_risky_source(source):
            continue
        filtered.append(source)

    return filtered if filtered else policy_sources


def format_policy_context(policy_sources: List[Dict[str, Any]]) -> str:
    formatted_docs = []
    safe_sources = filter_policy_sources_for_answer(policy_sources)

    for source in safe_sources:
        index = source.get("index", "")
        title = source.get("title", "")
        source_org = source.get("source_org", "")
        publish_date = source.get("publish_date", "")
        region = source.get("region", "")
        topic = source.get("topic", "")
        doc_type = source.get("doc_type", "")
        authority_level = source.get("authority_level", "")
        source_file = source.get("source_file", "")
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
文件类型：{doc_type}
权威级别：{authority_level}
来源文件：{source_file}
来源链接：{source_url}
{page_info}

正文片段：
{content}
"""
        formatted_docs.append(formatted.strip())

    return "\n\n".join(formatted_docs)


def format_evidence_context(policy_sources: List[Dict[str, Any]]) -> str:
    safe_sources = filter_policy_sources_for_answer(policy_sources)
    if not safe_sources:
        return "当前未检索到可用证据。"

    label_by_source = {
        "gov_policy": "政策依据",
        "laws": "法律法规依据",
        "legal_cases": "案例参考",
        "public_notices": "公告通知",
    }
    formatted_docs = []

    for source in safe_sources:
        source_id = source.get("knowledge_source_id", "")
        evidence_label = label_by_source.get(source_id, "检索证据")
        page = source.get("page", "")
        page_info = f"Page: {page}" if page != "" else "Page: none"
        formatted_docs.append(
            f"""
【{evidence_label}{source.get("index", "")}】
Title: {source.get("title", "")}
Source Org: {source.get("source_org", "")}
Publish Date: {source.get("publish_date", "")}
Region: {source.get("region", "")}
Topic: {source.get("topic", "")}
Knowledge Source: {source_id}
Document Type: {source.get("doc_type", "")}
Authority Level: {source.get("authority_level", "")}
Source File: {source.get("source_file", "")}
Source URL: {source.get("source_url", "")}
{page_info}

Content:
{source.get("content", "")}
""".strip()
        )

    return "\n\n".join(formatted_docs)


def format_policy_validity_info(policy_validity_result: Dict[str, Any]) -> str:
    if not policy_validity_result:
        return "当前未生成政策时效与来源可靠性判断。"

    return f"""
综合风险等级：{policy_validity_result.get("overall_risk_level", "")}
综合可靠性分数：{policy_validity_result.get("overall_score", "")}
摘要：{policy_validity_result.get("summary", "")}

风险提示：
{policy_validity_result.get("warnings", [])}

建议：
{policy_validity_result.get("suggestions", [])}
""".strip()


def format_agency_info(agency_result: Dict[str, Any]) -> str:
    if not agency_result or not agency_result.get("should_run"):
        return "当前问题未触发办理机构查询，或未匹配到办理机构信息。"

    return f"""
是否触发办理机构查询：{agency_result.get("should_run")}
摘要：{agency_result.get("summary", "")}

办理机构信息：
{agency_result.get("agency_info", "")}
""".strip()


def format_eligibility_info(eligibility_result: Dict[str, Any]) -> str:
    if not eligibility_result or not eligibility_result.get("should_run"):
        return "当前问题未触发资格判断，或无需进行资格判断。"

    return f"""
是否执行资格判断：{eligibility_result.get("should_run")}
系统初判结论：{eligibility_result.get("judgement")}

资格判断详细输出：
{eligibility_result.get("eligibility")}
""".strip()


def format_procedure_info(procedure_result: Dict[str, Any]) -> str:
    if not procedure_result:
        return "当前未生成办理流程信息。"

    return f"""
是否提取到明确流程：{procedure_result.get("has_procedure", False)}

办理流程输出：
{procedure_result.get("procedure", "")}
""".strip()


def format_material_info(material_result: Dict[str, Any]) -> str:
    if not material_result:
        return "当前未生成材料清单信息。"

    return f"""
是否提取到明确材料清单：{material_result.get("has_materials", False)}

材料清单输出：
{material_result.get("materials", "")}
""".strip()


def answer_agent(
    question: str,
    intent_result: Dict[str, Any],
    policy_result: Dict[str, Any],
    service_recommendation_result: Dict[str, Any] = None,
    policy_validity_result: Dict[str, Any] = None,
    agency_result: Dict[str, Any] = None,
    eligibility_result: Dict[str, Any] = None,
    procedure_result: Dict[str, Any] = None,
    material_result: Dict[str, Any] = None,
) -> Dict[str, Any]:
    if intent_result.get("need_clarification", False):
        service_recommendation_info = format_service_recommendation_info(
            service_recommendation_result or {}
        )

        clarification_question = intent_result.get(
            "clarification_question",
            "请补充你的所在地区或具体情况，以便查询对应政策。"
        )

        if service_recommendation_result and service_recommendation_result.get("should_run"):
            answer = f"""
一、办理结论

当前问题还需要补充信息后才能给出更准确的办理建议。

二、你可能需要关注的事项

{service_recommendation_info}

三、需要补充的信息

{clarification_question}
""".strip()
        else:
            answer = clarification_question

        return {
            "answer": answer,
            "used_sources": [],
        }

    policy_sources = policy_result.get("policy_sources", [])

    if not policy_sources:
        service_recommendation_info = format_service_recommendation_info(
            service_recommendation_result or {}
        )

        if service_recommendation_result and service_recommendation_result.get("should_run"):
            answer = f"""
一、办理结论

当前知识库没有检索到足够明确的政策资料，暂不能生成完整政策依据。

二、你可能需要关注的事项

{service_recommendation_info}

三、注意事项

建议补充具体地区、毕业年份、就业状态、户籍地、单位是否具备档案管理权限等信息后，再查询对应政策。
""".strip()
        else:
            answer = "当前知识库没有检索到相关政策资料，建议咨询当地主管部门或政务服务窗口。"

        return {
            "answer": answer,
            "used_sources": [],
        }

    intent_info = format_intent_info(intent_result)
    service_recommendation_info = format_service_recommendation_info(
        service_recommendation_result or {}
    )
    policy_context = format_evidence_context(policy_sources)
    policy_validity_info = format_policy_validity_info(policy_validity_result or {})
    agency_info = format_agency_info(agency_result or {})
    eligibility_info = format_eligibility_info(eligibility_result or {})
    procedure_info = format_procedure_info(procedure_result or {})
    material_info = format_material_info(material_result or {})

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", ANSWER_SYSTEM_PROMPT + PLATFORM_EVIDENCE_INSTRUCTIONS),
            ("human", "请根据以上资料生成保守、准确、可追溯的政务回答。"),
        ]
    )

    llm = get_answer_llm()
    chain = prompt | llm

    response = chain.invoke(
        {
            "question": question,
            "intent_info": intent_info,
            "service_recommendation_info": service_recommendation_info,
            "policy_context": policy_context,
            "policy_validity_info": policy_validity_info,
            "agency_info": agency_info,
            "eligibility_info": eligibility_info,
            "procedure_info": procedure_info,
            "material_info": material_info,
        }
    )

    return {
        "answer": response.content,
        "used_sources": policy_sources,
        "evidence_result": policy_result.get("evidence_result", {}),
        "evidence_groups": policy_result.get("evidence_groups", []),
        "source_distribution": policy_result.get("source_distribution", {}),
    }


def run_multi_agent_once(question: str) -> Dict[str, Any]:
    intent_result = classify_intent(question)

    service_recommendation_result = service_recommendation_agent(
        question=question,
        intent_result=intent_result,
    )

    policy_result = policy_agent(
        question=question,
        intent_result=intent_result,
    )

    policy_validity_result = policy_validity_agent(
        question=question,
        intent_result=intent_result,
        policy_result=policy_result,
    )

    agency_result = agency_agent(
        question=question,
        intent_result=intent_result,
    )

    eligibility_result = eligibility_agent(
        question=question,
        intent_result=intent_result,
        policy_result=policy_result,
    )

    procedure_result = procedure_agent(
        question=question,
        intent_result=intent_result,
        policy_result=policy_result,
    )

    material_result = material_agent(
        question=question,
        intent_result=intent_result,
        policy_result=policy_result,
    )

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
        "question": question,
        "intent_result": intent_result,
        "service_recommendation_result": service_recommendation_result,
        "policy_result": policy_result,
        "policy_validity_result": policy_validity_result,
        "agency_result": agency_result,
        "eligibility_result": eligibility_result,
        "procedure_result": procedure_result,
        "material_result": material_result,
        "answer_result": answer_result,
    }


def print_answer_result(result: Dict[str, Any]):
    print("\n========== Intent Agent ==========")
    print(result["intent_result"])

    print("\n========== Service Recommendation Agent ==========")
    recommendation = result["service_recommendation_result"]
    print(f"是否触发事项推荐: {recommendation.get('should_run')}")
    print(f"摘要: {recommendation.get('summary')}")

    print("\n========== Policy Agent ==========")
    print("增强检索 Query:")
    print(result["policy_result"].get("enhanced_query", ""))

    print("\n========== Policy Validity Agent ==========")
    validity = result["policy_validity_result"]
    print(f"综合风险等级: {validity.get('overall_risk_level')}")
    print(f"综合可靠性分数: {validity.get('overall_score')}")
    print(f"摘要: {validity.get('summary')}")

    print("\n========== Agency Agent ==========")
    agency = result["agency_result"]
    print(f"是否触发办理机构查询: {agency.get('should_run')}")
    print(f"摘要: {agency.get('summary')}")
    print(agency.get("agency_info", ""))

    print("\n========== Eligibility Agent ==========")
    eligibility = result["eligibility_result"]
    print(f"是否触发资格判断: {eligibility.get('should_run', False)}")
    print(f"初判结论: {eligibility.get('judgement', '无')}")
    if eligibility.get("should_run"):
        print(f"详细推导:\n{eligibility.get('eligibility', '')}")

    print("\n========== Procedure Agent ==========")
    print(result["procedure_result"].get("procedure", ""))

    print("\n========== Material Agent ==========")
    print(result["material_result"].get("materials", ""))

    print("\n========== Answer Agent 最终回答 ==========")
    print(result["answer_result"]["answer"])

    print("\n========== 参考来源 ==========")
    for source in result["answer_result"].get("used_sources", []):
        print("\n------------------------------")
        print(f"资料 {source.get('index', '')}")
        print(f"标题：{source.get('title', '')}")
        print(f"发布机构：{source.get('source_org', '')}")
        print(f"地区：{source.get('region', '')}")
        print(f"主题：{source.get('topic', '')}")
        print(f"来源文件：{source.get('source_file', '')}")
        print(f"来源链接：{source.get('source_url', '')}")
        if source.get("page", "") != "":
            print(f"页码：{source.get('page', '')}")


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

        result = run_multi_agent_once(question)
        print_answer_result(result)
