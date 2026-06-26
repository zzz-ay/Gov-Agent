from typing import Dict, Any, List

from langchain_core.prompts import ChatPromptTemplate

from app.llm.provider import get_chat_llm


MATERIAL_SYSTEM_PROMPT = """
你是 GOV-RAG 政务问答系统中的“材料清单智能体”。

你的任务：
根据用户问题、意图识别结果和政策检索资料，提取或整理办理该事项可能需要准备的材料清单。

重要原则：
1. 只能依据“政策检索资料”提取材料，不允许自行编造材料。
2. 如果资料中没有明确材料清单，请说明“当前检索资料未明确说明完整材料清单”。
3. 不得补充资料中没有出现的材料、证明、表格、费用、办理地点或办理时限。
4. 如果不同地区材料可能不同，必须提醒“以当地主管部门或政务服务窗口最新要求为准”。
5. 如果资料只提到部分材料，需要明确说明“以下仅为当前资料中提到的材料”。
6. 输出要清晰、简洁，适合高校毕业生直接查看。

输出格式：

材料清单：
1. ...
2. ...
3. ...

材料说明：
- ...

缺失信息：
- 当前检索资料未明确说明...
"""


def get_material_llm():
    """
    创建材料清单生成用 LLM。
    """
    return get_chat_llm(temperature=0.1)


def format_policy_context(policy_sources: List[Dict[str, Any]]) -> str:
    """
    将政策资料格式化为材料清单智能体上下文。
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


def should_run_material(question: str, intent_result: Dict[str, Any]) -> bool:
    keywords = [
        "材料",
        "资料",
        "证明",
        "证件",
        "申请表",
        "准备什么",
        "需要准备",
        "带什么",
        "提交什么",
    ]

    if any(keyword in question for keyword in keywords):
        return True

    intent = intent_result.get("intent", "")
    return intent in {"employment_subsidy", "social_security", "entrepreneurship"}


def material_agent(
    question: str,
    intent_result: Dict[str, Any],
    policy_result: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Material Agent 主函数。

    输入：
    - question: 用户问题
    - intent_result: Intent Agent 输出
    - policy_result: Policy Agent 输出

    输出：
    {
        "materials": "...",
        "has_materials": True / False
    }
    """
    if not should_run_material(question, intent_result):
        return {
            "materials": "当前问题未触发材料清单整理，以减少不必要的模型调用。",
            "has_materials": False,
            "skipped": True,
        }

    policy_sources = policy_result.get("policy_sources", [])

    if not policy_sources:
        return {
            "materials": "当前知识库没有检索到相关政策资料，无法整理材料清单。",
            "has_materials": False,
        }

    policy_context = format_policy_context(policy_sources)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", MATERIAL_SYSTEM_PROMPT),
            (
                "human",
                """
用户问题：
{question}

意图识别结果：
{intent_result}

政策检索资料：
{policy_context}

请整理与该问题相关的办理材料清单。
""",
            ),
        ]
    )

    llm = get_material_llm()
    chain = prompt | llm

    response = chain.invoke(
        {
            "question": question,
            "intent_result": intent_result,
            "policy_context": policy_context,
        }
    )

    materials_text = response.content

    has_materials = (
        "当前检索资料未明确说明完整材料清单" not in materials_text
        and "无法整理材料清单" not in materials_text
    )

    return {
        "materials": materials_text,
        "has_materials": has_materials,
    }


if __name__ == "__main__":
    from app.agents.intent_agent import classify_intent
    from app.agents.policy_agent import policy_agent

    test_questions = [
        "石家庄毕业生档案接收怎么办理？",
        "河北高校毕业生可以申请哪些就业补贴？",
        "毕业去向登记系统怎么填写档案转递信息？",
        "高校毕业生档案可以自己携带吗？",
        "创业担保贷款有什么条件？",
    ]

    for question in test_questions:
        print("\n\n========================================")
        print(f"测试问题：{question}")
        print("========================================")

        intent_result = classify_intent(question)
        policy_result = policy_agent(question, intent_result)

        result = material_agent(
            question=question,
            intent_result=intent_result,
            policy_result=policy_result,
        )

        print("\n========== Material Agent 输出 ==========")
        print(result["materials"])
        print("\n是否提取到材料清单：", result["has_materials"])
