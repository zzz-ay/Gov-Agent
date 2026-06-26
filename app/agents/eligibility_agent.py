from typing import Dict, Any, List

from langchain_core.prompts import ChatPromptTemplate

from app.llm.provider import get_chat_llm


ELIGIBILITY_SYSTEM_PROMPT = """
你是 GOV-RAG 政务问答系统中的“资格判断智能体”。

你的任务：
根据用户问题、意图识别结果和政策检索资料，对用户是否可能符合某项政策、补贴、贷款、档案接收或户籍办理条件进行初步判断。

适用场景：
- 用户询问“我能不能申请？”
- 用户询问“是否符合条件？”
- 用户提供了个人情况，如毕业年份、学历、就业地、户籍地、社保缴纳、劳动合同、创业情况等
- 用户咨询补贴、社保补贴、创业担保贷款、安家补贴、档案接收、户籍迁移等事项

重要原则：
1. 只能依据“政策检索资料”进行判断，不允许根据常识或模型自身知识补充条件。
2. 不得直接给出绝对结论，如“你一定符合”或“你一定不符合”，除非资料和用户信息都非常明确。
3. 推荐使用“可能符合”“可能不符合”“信息不足，无法判断”这三类结论。
4. 如果用户缺少关键信息，必须列出“仍需确认的信息”。
5. 涉及地区差异时，必须提醒“以当地主管部门或政务服务窗口最新要求为准”。
6. 不要编造政策条件、金额、材料、流程、时间。
7. 如果政策资料中没有资格条件，请说明“当前检索资料未明确说明申请条件”。

输出格式：

初步判断：
- 可能符合 / 可能不符合 / 信息不足，无法判断

判断依据：
- ...

用户已提供的信息：
- ...

政策要求：
- ...

已匹配条件：
- ...

未匹配或可能不符合条件：
- ...

仍需确认的信息：
- ...

提醒：
- ...
"""


def get_eligibility_llm():
    """
    创建资格判断用 LLM。
    """
    return get_chat_llm(temperature=0.1)


def format_policy_context(policy_sources: List[Dict[str, Any]]) -> str:
    """
    将政策资料格式化为资格判断智能体上下文。
    """
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
【资料{index}】
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


def should_run_eligibility(question: str, intent_result: Dict[str, Any]) -> bool:
    """
    判断是否需要运行资格判断。

    主要针对：
    - 补贴
    - 社保补贴
    - 创业担保贷款
    - 户籍迁移
    - 档案接收中包含“我是否可以/能否”等条件判断问题
    """
    question_lower = question.lower()

    keywords = [
        "能不能",
        "可以申请",
        "能申请",
        "是否符合",
        "符合条件",
        "有没有资格",
        "有资格",
        "能否",
        "可不可以",
        "可以办理",
        "能办理",
        "我可以",
        "我能",
        "申请条件",
        "什么条件",
        "条件是什么",
    ]

    intent = intent_result.get("intent", "")

    eligibility_intents = {
        "employment_subsidy",
        "social_security",
        "entrepreneurship",
        "hukou",
        "archive_transfer",
        "employment_policy",
    }

    if intent in eligibility_intents:
        for kw in keywords:
            if kw in question:
                return True

    # 如果用户问题明显包含个人情况，也运行资格判断
    personal_keywords = [
        "我是",
        "我在",
        "我已经",
        "我目前",
        "我毕业",
        "社保",
        "劳动合同",
        "企业工作",
        "自主创业",
        "户籍",
        "本科",
        "硕士",
        "博士",
        "毕业生",
    ]

    if intent in eligibility_intents:
        hit_count = sum(1 for kw in personal_keywords if kw in question)
        if hit_count >= 2:
            return True

    return False


def eligibility_agent(
    question: str,
    intent_result: Dict[str, Any],
    policy_result: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Eligibility Agent 主函数。

    输入：
    - question: 用户问题
    - intent_result: Intent Agent 输出
    - policy_result: Policy Agent 输出

    输出：
    {
        "should_run": True / False,
        "eligibility": "...",
        "judgement": "可能符合 / 可能不符合 / 信息不足，无法判断 / 未触发资格判断"
    }
    """
    run_flag = should_run_eligibility(question, intent_result)

    if not run_flag:
        return {
            "should_run": False,
            "eligibility": "当前问题不属于明显的资格判断类问题，未执行资格判断。",
            "judgement": "未触发资格判断",
        }

    policy_sources = policy_result.get("policy_sources", [])

    if not policy_sources:
        return {
            "should_run": True,
            "eligibility": "当前知识库没有检索到相关政策资料，无法进行资格判断。",
            "judgement": "信息不足，无法判断",
        }

    policy_context = format_policy_context(policy_sources)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", ELIGIBILITY_SYSTEM_PROMPT),
            (
                "human",
                """
用户问题：
{question}

意图识别结果：
{intent_result}

政策检索资料：
{policy_context}

请根据用户问题和政策资料进行资格条件初步判断。
""",
            ),
        ]
    )

    llm = get_eligibility_llm()
    chain = prompt | llm

    response = chain.invoke(
        {
            "question": question,
            "intent_result": str(intent_result),
            "policy_context": policy_context,
        }
    )

    eligibility_text = response.content

    judgement = "信息不足，无法判断"
    if "可能符合" in eligibility_text:
        judgement = "可能符合"
    elif "可能不符合" in eligibility_text:
        judgement = "可能不符合"
    elif "信息不足" in eligibility_text or "无法判断" in eligibility_text:
        judgement = "信息不足，无法判断"

    return {
        "should_run": True,
        "eligibility": eligibility_text,
        "judgement": judgement,
    }


if __name__ == "__main__":
    from app.agents.intent_agent import classify_intent
    from app.agents.policy_agent import policy_agent

    test_questions = [
        "我是2025届本科毕业生，在石家庄企业工作，社保交了6个月，可以申请安家补贴吗？",
        "我是毕业年度学生，家庭比较困难，可以申请求职创业补贴吗？",
        "我毕业2年内还没就业，现在灵活就业并自己交社保，可以申请社保补贴吗？",
        "创业担保贷款有什么条件？",
        "高校毕业生档案可以自己携带吗？",
        "现在还需要就业报到证吗？",
    ]

    for question in test_questions:
        print("\n\n========================================")
        print(f"测试问题：{question}")
        print("========================================")

        intent_result = classify_intent(question)
        policy_result = policy_agent(question, intent_result)

        result = eligibility_agent(
            question=question,
            intent_result=intent_result,
            policy_result=policy_result,
        )

        print("\n========== Intent Agent ==========")
        print(intent_result)

        print("\n========== Eligibility Agent 输出 ==========")
        print(result["eligibility"])
        print("\n是否执行资格判断：", result["should_run"])
        print("判断结果：", result["judgement"])
