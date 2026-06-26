from typing import Dict, Any, Optional
import json

from langchain_core.prompts import ChatPromptTemplate

from app.llm.provider import get_chat_llm


SLOT_SYSTEM_PROMPT = """
你是 GOV-RAG 政务服务系统中的“用户信息槽位识别智能体”。

你的任务：
从用户当前问题中提取高校毕业生政务服务相关信息，并与已有用户信息进行合并。

系统场景：
高校毕业生政务服务，包括档案转递、毕业去向登记、户籍迁移、就业服务、就业补贴、社保补贴、创业担保贷款等。

你需要识别以下槽位：

1. region
用户当前咨询地区，例如：全国、河北省、河北省石家庄市、河北省保定市、未明确

2. hukou_region
用户户籍地 / 生源地 / 原籍，例如：河北省石家庄市、河北省保定市、未明确

3. current_city
用户当前所在城市或就业城市，例如：河北省石家庄市、未明确

4. graduation_year
毕业年份，例如：2025、2026、未明确

5. education_level
学历，例如：专科、本科、硕士、博士、未明确

6. is_fresh_graduate
是否应届毕业生：true / false / null

7. employment_status
就业状态：
- employed：已就业
- unemployed：未就业
- flexible_employment：灵活就业
- entrepreneurship：创业
- further_study：升学
- unknown：未明确

8. employer_has_archive_authority
单位是否有人事档案管理权限：
- true：有
- false：没有
- null：未明确

9. archive_need
档案诉求：
- transfer_to_hukou：转回户籍地
- transfer_to_employment_city：转到就业地
- transfer_to_employer：转到单位
- query_archive：查询档案
- unknown：未明确

10. hukou_need
户口诉求：
- move_back_home：迁回原籍
- move_to_work_city：迁入就业地
- keep_at_school：保留学校集体户
- move_to_talent_center：迁入人才中心
- unknown：未明确

11. subsidy_interest
是否关注补贴：true / false / null

12. social_security_interest
是否关注社保 / 社保补贴：true / false / null

13. entrepreneurship_interest
是否关注创业 / 创业担保贷款：true / false / null

合并规则：
- 如果当前用户问题提供了新信息，应覆盖已有信息。
- 如果当前问题没有提到某个槽位，应保留 existing_slots 中已有值。
- 不要臆测用户没有明确说出的信息。
- “河北应届毕业生”只能推断 region=河北省、is_fresh_graduate=true，但不能推断具体城市。
- “石家庄企业工作”可以推断 current_city=河北省石家庄市、employment_status=employed。
- “还没找到工作”“暂未就业”可以推断 employment_status=unemployed。
- “单位不能接收档案”可以推断 employer_has_archive_authority=false。
- “档案转回老家/原籍/户籍地”可以推断 archive_need=transfer_to_hukou。
- “户口迁回原籍”可以推断 hukou_need=move_back_home。

请严格输出 JSON，不要输出 Markdown，不要输出解释性文字。

输出格式：
{{
  "slots": {{
    "region": "河北省",
    "hukou_region": "未明确",
    "current_city": "未明确",
    "graduation_year": "2025",
    "education_level": "本科",
    "is_fresh_graduate": true,
    "employment_status": "unemployed",
    "employer_has_archive_authority": null,
    "archive_need": "unknown",
    "hukou_need": "unknown",
    "subsidy_interest": false,
    "social_security_interest": false,
    "entrepreneurship_interest": false
  }},
  "updated_fields": ["region", "graduation_year", "employment_status"],
  "missing_fields": ["hukou_region", "current_city"],
  "summary": "用户是河北应届本科毕业生，目前未就业，但未说明具体户籍地和档案去向。"
}}
"""


DEFAULT_SLOTS = {
    "region": "未明确",
    "hukou_region": "未明确",
    "current_city": "未明确",
    "graduation_year": "未明确",
    "education_level": "未明确",
    "is_fresh_graduate": None,
    "employment_status": "unknown",
    "employer_has_archive_authority": None,
    "archive_need": "unknown",
    "hukou_need": "unknown",
    "subsidy_interest": None,
    "social_security_interest": None,
    "entrepreneurship_interest": None,
}


def get_slot_llm():
    """
    创建槽位识别用 LLM。
    """
    return get_chat_llm(temperature=0)


def normalize_slots(slots: Dict[str, Any]) -> Dict[str, Any]:
    """
    规范化 slots，保证字段完整。
    """
    normalized = dict(DEFAULT_SLOTS)

    if isinstance(slots, dict):
        for key in DEFAULT_SLOTS:
            if key in slots:
                normalized[key] = slots[key]

    return normalized


def parse_json_response(raw_text: str) -> Dict[str, Any]:
    """
    稳健解析模型 JSON 输出。
    """
    text = raw_text.strip()

    if text.startswith("```json"):
        text = text.replace("```json", "").replace("```", "").strip()
    elif text.startswith("```"):
        text = text.replace("```", "").strip()

    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        return {
            "slots": dict(DEFAULT_SLOTS),
            "updated_fields": [],
            "missing_fields": [],
            "summary": "槽位识别结果解析失败，已使用默认空槽位。",
            "raw_output": raw_text,
        }

    slots = normalize_slots(result.get("slots", {}))

    return {
        "slots": slots,
        "updated_fields": result.get("updated_fields", []),
        "missing_fields": result.get("missing_fields", []),
        "summary": result.get("summary", ""),
    }


def slot_agent(
    question: str,
    existing_slots: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Slot Agent 主函数。

    输入：
    - question: 用户当前问题
    - existing_slots: 历史槽位信息

    输出：
    {
        "slots": {...},
        "updated_fields": [...],
        "missing_fields": [...],
        "summary": "..."
    }
    """
    existing_slots = normalize_slots(existing_slots or {})

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SLOT_SYSTEM_PROMPT),
            (
                "human",
                """
已有用户信息：
{existing_slots}

用户当前输入：
{question}

请提取并合并用户信息槽位，输出 JSON。
""",
            ),
        ]
    )

    llm = get_slot_llm()
    chain = prompt | llm

    response = chain.invoke(
        {
            "existing_slots": json.dumps(existing_slots, ensure_ascii=False, indent=2),
            "question": question,
        }
    )

    result = parse_json_response(response.content)

    return result


def build_context_from_slots(slots: Dict[str, Any]) -> str:
    """
    将槽位信息转换为可拼接到用户问题前的上下文。
    后续 main.py / gov_graph.py 会用到。
    """
    slots = normalize_slots(slots)

    lines = []

    if slots.get("region") != "未明确":
        lines.append(f"咨询地区：{slots.get('region')}")

    if slots.get("hukou_region") != "未明确":
        lines.append(f"户籍地/生源地：{slots.get('hukou_region')}")

    if slots.get("current_city") != "未明确":
        lines.append(f"当前所在或就业城市：{slots.get('current_city')}")

    if slots.get("graduation_year") != "未明确":
        lines.append(f"毕业年份：{slots.get('graduation_year')}")

    if slots.get("education_level") != "未明确":
        lines.append(f"学历：{slots.get('education_level')}")

    if slots.get("is_fresh_graduate") is True:
        lines.append("是否应届：是")
    elif slots.get("is_fresh_graduate") is False:
        lines.append("是否应届：否")

    employment_status_map = {
        "employed": "已就业",
        "unemployed": "未就业",
        "flexible_employment": "灵活就业",
        "entrepreneurship": "创业",
        "further_study": "升学",
        "unknown": "未明确",
    }

    if slots.get("employment_status") != "unknown":
        lines.append(f"就业状态：{employment_status_map.get(slots.get('employment_status'), slots.get('employment_status'))}")

    if slots.get("employer_has_archive_authority") is True:
        lines.append("单位是否具有人事档案管理权限：有")
    elif slots.get("employer_has_archive_authority") is False:
        lines.append("单位是否具有人事档案管理权限：没有")

    archive_need_map = {
        "transfer_to_hukou": "档案转回户籍地",
        "transfer_to_employment_city": "档案转到就业地",
        "transfer_to_employer": "档案转到单位",
        "query_archive": "查询档案",
        "unknown": "未明确",
    }

    if slots.get("archive_need") != "unknown":
        lines.append(f"档案诉求：{archive_need_map.get(slots.get('archive_need'), slots.get('archive_need'))}")

    hukou_need_map = {
        "move_back_home": "户口迁回原籍",
        "move_to_work_city": "户口迁入就业地",
        "keep_at_school": "户口保留学校集体户",
        "move_to_talent_center": "户口迁入人才中心",
        "unknown": "未明确",
    }

    if slots.get("hukou_need") != "unknown":
        lines.append(f"户口诉求：{hukou_need_map.get(slots.get('hukou_need'), slots.get('hukou_need'))}")

    if slots.get("subsidy_interest") is True:
        lines.append("关注补贴：是")

    if slots.get("social_security_interest") is True:
        lines.append("关注社保/社保补贴：是")

    if slots.get("entrepreneurship_interest") is True:
        lines.append("关注创业/创业担保贷款：是")

    if not lines:
        return ""

    return "用户已补充信息：\n" + "\n".join(f"- {line}" for line in lines)


if __name__ == "__main__":
    test_turns = [
        "我是河北应届毕业生，还没找到工作，档案和户口不知道怎么办。",
        "我是石家庄的，本科，2025届。",
        "我想把档案转回石家庄，户口也迁回原籍。",
        "单位目前还没找到，也想看看有没有补贴。",
    ]

    slots = dict(DEFAULT_SLOTS)

    for turn in test_turns:
        print("\n\n========================================")
        print(f"用户输入：{turn}")
        print("========================================")

        result = slot_agent(
            question=turn,
            existing_slots=slots,
        )

        slots = result["slots"]

        print("\n========== Slot Agent 输出 ==========")
        print(json.dumps(result, ensure_ascii=False, indent=2))

        print("\n========== 当前累计用户信息 ==========")
        print(build_context_from_slots(slots))
