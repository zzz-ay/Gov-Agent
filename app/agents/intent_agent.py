import json
from typing import Dict, Any

from langchain_core.prompts import ChatPromptTemplate

from app.llm.provider import get_chat_llm


INTENT_SYSTEM_PROMPT = """
你是 GOV-RAG 政务问答系统中的意图识别智能体。

你的任务：
根据用户输入的问题，识别其政务服务意图、涉及地区，以及是否需要进一步追问。

系统面向场景：
高校毕业生政务服务，包括就业、档案转递、毕业去向登记、就业报到证、户籍迁移、社保、就业/人才/创业补贴等。

请从以下 intent 中选择一个最合适的：

1. archive_transfer
含义：档案转递、档案接收、档案查询、档案存放、档案回原籍、人才中心接收档案等

2. graduation_registration
含义：毕业去向登记、全国高校毕业生毕业去向登记系统、档案信息登记、户口迁移信息登记等

3. registration_certificate
含义：就业报到证、报到证取消、报到证补办、改派、报到证是否还需要等

4. employment_policy
含义：高校毕业生就业政策、就业服务、就业见习、就业帮扶、公共就业服务等

5. employment_subsidy
含义：求职创业补贴、就业补贴、安家补贴、人才补贴、见习补贴等

6. social_security
含义：社保、灵活就业社保补贴、社保缴纳、社会保险补贴等

7. entrepreneurship
含义：创业补贴、创业担保贷款、创业扶持、创业政策等

8. hukou
含义：户籍迁移、户口迁移、落户、迁户口等

9. general
含义：其他无法明确归类的高校毕业生政务服务问题

地区识别规则：
- 如果用户明确提到“河北”“河北省”，region 写“河北省”
- 如果用户明确提到“石家庄”，region 写“河北省石家庄市”
- 如果用户明确提到“保定”，region 写“河北省保定市”
- 如果用户只问全国性政策，region 写“全国”
- 如果无法判断地区，region 写“未明确”

是否需要追问 need_clarification：
- 如果问题涉及明显有地区差异的事项，如补贴、户籍迁移、地方档案接收、社保补贴，但用户没有说明地区，则 need_clarification=true
- 如果问题是全国统一政策，如就业报到证是否取消、档案能否个人携带，则 need_clarification=false
- 如果问题表述过于模糊，也可以 need_clarification=true

请严格输出 JSON，不要输出任何解释文字。

输出格式：
{{
  "intent": "archive_transfer",
  "intent_name": "档案转递",
  "region": "河北省石家庄市",
  "need_clarification": false,
  "clarification_question": "",
  "reason": "用户询问档案接收办理，属于档案转递问题"
}}
"""


INTENT_NAME_MAP = {
    "archive_transfer": "档案转递",
    "graduation_registration": "毕业去向登记",
    "registration_certificate": "就业报到证",
    "employment_policy": "就业政策",
    "employment_subsidy": "就业/人才补贴",
    "social_security": "社保/灵活就业社保补贴",
    "entrepreneurship": "创业政策",
    "hukou": "户籍迁移",
    "general": "其他政务问题",
}


def get_intent_llm():
    """
    创建意图识别用 LLM。
    """
    return get_chat_llm(temperature=0)


def parse_json_response(text: str) -> Dict[str, Any]:
    """
    尽量稳健地解析模型输出的 JSON。
    """
    text = text.strip()

    if text.startswith("```json"):
        text = text.replace("```json", "").replace("```", "").strip()
    elif text.startswith("```"):
        text = text.replace("```", "").strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {
            "intent": "general",
            "intent_name": "其他政务问题",
            "region": "未明确",
            "need_clarification": True,
            "clarification_question": "请补充你的具体问题和所在地区，以便查询对应政策。",
            "reason": "模型输出 JSON 解析失败，已回退为 general",
        }


def classify_intent(question: str) -> Dict[str, Any]:
    """
    识别用户问题意图。
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", INTENT_SYSTEM_PROMPT),
            ("human", "用户问题：{question}"),
        ]
    )

    llm = get_intent_llm()
    chain = prompt | llm

    response = chain.invoke({"question": question})
    result = parse_json_response(response.content)

    intent = result.get("intent", "general")

    if intent not in INTENT_NAME_MAP:
        intent = "general"
        result["intent"] = "general"

    result["intent_name"] = INTENT_NAME_MAP.get(intent, "其他政务问题")

    result.setdefault("region", "未明确")
    result.setdefault("need_clarification", False)
    result.setdefault("clarification_question", "")
    result.setdefault("reason", "")

    return result


if __name__ == "__main__":
    test_questions = [
        "高校毕业生档案可以自己携带吗？",
        "现在还需要就业报到证吗？",
        "毕业去向登记系统怎么填写档案转递信息？",
        "河北高校毕业生可以申请哪些就业补贴？",
        "石家庄毕业生档案接收怎么办理？",
        "我毕业后想把户口迁回河北，需要怎么办？",
        "灵活就业社保补贴怎么申请？",
        "创业担保贷款有什么条件？",
    ]

    for q in test_questions:
        print("\n====================================")
        print(f"问题：{q}")
        result = classify_intent(q)
        print(json.dumps(result, ensure_ascii=False, indent=2))
