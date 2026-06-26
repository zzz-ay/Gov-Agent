from typing import Dict, Any, List

from langchain_core.prompts import ChatPromptTemplate

from app.llm.provider import get_chat_llm


PROCEDURE_SYSTEM_PROMPT = """
你是 GOV-RAG 政务问答系统中的“办理流程智能体”。

你的任务：
根据用户问题、意图识别结果和政策检索资料，提取或整理与该问题相关的办理流程。

重要原则：
1. 只能依据“政策检索资料”生成办理流程，不允许自行编造流程。
2. 如果资料中没有明确流程，请说明“当前检索资料未明确说明完整办理流程”。
3. 不要补充资料中没有出现的材料、时间、地点、费用或办理机构。
4. 如果流程涉及地区差异，必须提醒“以当地主管部门或政务服务窗口最新要求为准”。
5. 输出要简洁、清晰，适合高校毕业生理解。

输出格式：

办理流程：
1. ...
2. ...
3. ...

流程说明：
- ...

缺失信息：
- 当前检索资料未明确说明...
"""


def get_procedure_llm():
    """
    创建办理流程生成用 LLM。
    """
    return get_chat_llm(temperature=0.1)


def format_policy_context(policy_sources: List[Dict[str, Any]]) -> str:
    """
    将政策资料格式化为流程智能体上下文。
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


def should_run_procedure(question: str, intent_result: Dict[str, Any]) -> bool:
    keywords = [
        "流程",
        "步骤",
        "怎么办",
        "如何办理",
        "怎么办理",
        "办理",
        "操作",
        "去哪",
        "去哪里",
        "渠道",
    ]

    if any(keyword in question for keyword in keywords):
        return True

    intent = intent_result.get("intent", "")
    return intent in {"archive_transfer", "graduation_registration", "hukou"}


def procedure_agent(
    question: str,
    intent_result: Dict[str, Any],
    policy_result: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Procedure Agent 主函数。

    输入：
    - question: 用户问题
    - intent_result: Intent Agent 输出
    - policy_result: Policy Agent 输出

    输出：
    {
        "procedure": "...",
        "has_procedure": True / False
    }
    """
    if not should_run_procedure(question, intent_result):
        return {
            "procedure": "当前问题未触发办理流程整理，以减少不必要的模型调用。",
            "has_procedure": False,
            "skipped": True,
        }

    policy_sources = policy_result.get("policy_sources", [])

    if not policy_sources:
        return {
            "procedure": "当前知识库没有检索到相关政策资料，无法整理办理流程。",
            "has_procedure": False,
        }

    policy_context = format_policy_context(policy_sources)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", PROCEDURE_SYSTEM_PROMPT),
            (
                "human",
                """
用户问题：
{question}

意图识别结果：
{intent_result}

政策检索资料：
{policy_context}

请整理与该问题相关的办理流程。
""",
            ),
        ]
    )

    llm = get_procedure_llm()
    chain = prompt | llm

    response = chain.invoke(
        {
            "question": question,
            "intent_result": intent_result,
            "policy_context": policy_context,
        }
    )

    procedure_text = response.content

    has_procedure = "当前检索资料未明确说明完整办理流程" not in procedure_text

    return {
        "procedure": procedure_text,
        "has_procedure": has_procedure,
    }


if __name__ == "__main__":
    from app.agents.intent_agent import classify_intent
    from app.agents.policy_agent import policy_agent

    test_questions = [
        "毕业去向登记系统怎么填写档案转递信息？",
        "石家庄毕业生档案接收怎么办理？",
        "河北高校毕业生可以申请哪些就业补贴？",
        "高校毕业生档案可以自己携带吗？",
    ]

    for question in test_questions:
        print("\n\n========================================")
        print(f"测试问题：{question}")
        print("========================================")

        intent_result = classify_intent(question)
        policy_result = policy_agent(question, intent_result)

        result = procedure_agent(
            question=question,
            intent_result=intent_result,
            policy_result=policy_result,
        )

        print("\n========== Procedure Agent 输出 ==========")
        print(result["procedure"])
        print("\n是否提取到流程：", result["has_procedure"])
