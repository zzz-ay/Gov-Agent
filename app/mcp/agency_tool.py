import json
from pathlib import Path
from typing import Dict, Any, List


BASE_DIR = Path(__file__).resolve().parent.parent.parent
AGENCY_DB_PATH = BASE_DIR / "data" / "service_db" / "agencies.json"


INTENT_TO_SERVICE_TYPE = {
    "archive_transfer": "archive_transfer",
    "graduation_registration": "graduation_registration",
    "registration_certificate": "registration_certificate",
    "employment_policy": "employment_policy",
    "employment_subsidy": "employment_subsidy",
    "social_security": "social_security",
    "entrepreneurship": "entrepreneurship",
    "hukou": "hukou",
    "general": "general",
}


def load_agency_db() -> List[Dict[str, Any]]:
    """
    加载本地办理机构数据库。
    """
    if not AGENCY_DB_PATH.exists():
        raise FileNotFoundError(
            f"办理机构数据库不存在: {AGENCY_DB_PATH}"
        )

    with open(AGENCY_DB_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("agencies.json 格式错误，顶层应为列表。")

    return data


def normalize_region(region: str) -> str:
    """
    简单规范化地区字段。
    """
    if not region:
        return "未明确"

    region = region.strip()

    if region in ["石家庄", "石家庄市"]:
        return "河北省石家庄市"

    if region in ["河北", "河北省"]:
        return "河北省"

    return region


def region_match(user_region: str, agency_region: str) -> int:
    """
    地区匹配分数。

    返回：
    - 3：完全匹配
    - 2：省级匹配
    - 1：全国或用户地区未明确
    - 0：不匹配
    """
    user_region = normalize_region(user_region)
    agency_region = normalize_region(agency_region)

    if user_region == "未明确":
        return 1

    if user_region == agency_region:
        return 3

    if user_region in agency_region or agency_region in user_region:
        return 2

    if "河北" in user_region and "河北" in agency_region:
        return 2

    if agency_region in ["全国", "未明确"]:
        return 1

    return 0


def service_match(intent: str, service_type: str) -> int:
    """
    服务类型匹配分数。

    返回：
    - 3：完全匹配
    - 2：同一大类相关
    - 0：不相关
    """
    target_service_type = INTENT_TO_SERVICE_TYPE.get(intent, "general")

    if target_service_type == service_type:
        return 3

    # 就业政策可以泛化匹配就业补贴、社保补贴、创业政策
    if target_service_type == "employment_policy" and service_type in [
        "employment_subsidy",
        "social_security",
        "entrepreneurship",
    ]:
        return 2

    # general 不主动泛化到所有机构，避免无关机构乱入
    return 0


def search_agencies(
    intent: str,
    region: str,
    top_k: int = 3,
) -> List[Dict[str, Any]]:
    """
    根据意图和地区查询办理机构。

    核心规则：
    - 服务类型必须匹配
    - 地区只用于排序
    - 避免因为同省地区匹配而返回无关机构
    """
    agencies = load_agency_db()

    scored_results = []

    for agency in agencies:
        agency_region = agency.get("region", "")
        service_type = agency.get("service_type", "")

        s_score = service_match(intent, service_type)

        # 关键修正：
        # 服务类型不匹配，直接跳过
        if s_score <= 0:
            continue

        r_score = region_match(region, agency_region)

        # 如果用户明确了地区，但机构地区完全不匹配，也跳过
        if normalize_region(region) != "未明确" and r_score <= 0:
            continue

        total_score = s_score * 10 + r_score * 3

        result = dict(agency)
        result["match_score"] = total_score
        result["region_match_score"] = r_score
        result["service_match_score"] = s_score
        scored_results.append(result)

    scored_results.sort(
        key=lambda x: x.get("match_score", 0),
        reverse=True,
    )

    return scored_results[:top_k]


def agency_tool(
    intent_result: Dict[str, Any],
    top_k: int = 3,
) -> Dict[str, Any]:
    """
    办理机构查询工具主函数。
    """
    intent = intent_result.get("intent", "general")
    region = intent_result.get("region", "未明确")

    agencies = search_agencies(
        intent=intent,
        region=region,
        top_k=top_k,
    )

    if not agencies:
        return {
            "should_use": False,
            "agencies": [],
            "summary": "当前办理机构数据库中未匹配到相关机构信息。",
        }

    return {
        "should_use": True,
        "agencies": agencies,
        "summary": f"已根据问题类型“{intent}”和地区“{region}”匹配到 {len(agencies)} 条办理机构信息。",
    }


def format_agencies_for_answer(agencies: List[Dict[str, Any]]) -> str:
    """
    将机构信息格式化为可供 Answer Agent 使用的文本。
    """
    if not agencies:
        return "当前未匹配到办理机构信息。"

    lines = []

    for idx, agency in enumerate(agencies, start=1):
        text = f"""
【办理机构 {idx}】
服务事项：{agency.get("service_name", "")}
适用地区：{agency.get("region", "")}
服务类型：{agency.get("service_type", "")}
机构名称：{agency.get("agency_name", "")}
地址：{agency.get("address", "")}
联系电话：{agency.get("phone", "")}
官网链接：{agency.get("website", "")}
办公时间：{agency.get("office_hours", "")}
信息来源：{agency.get("source", "")}
说明：{agency.get("notes", "")}
"""
        lines.append(text.strip())

    return "\n\n".join(lines)


if __name__ == "__main__":
    test_intent_results = [
        {
            "intent": "archive_transfer",
            "intent_name": "档案转递",
            "region": "河北省石家庄市",
        },
        {
            "intent": "employment_subsidy",
            "intent_name": "就业/人才补贴",
            "region": "河北省",
        },
        {
            "intent": "social_security",
            "intent_name": "社保/灵活就业社保补贴",
            "region": "未明确",
        },
        {
            "intent": "hukou",
            "intent_name": "户籍迁移",
            "region": "河北省",
        },
        {
            "intent": "registration_certificate",
            "intent_name": "就业报到证",
            "region": "全国",
        },
    ]

    for item in test_intent_results:
        print("\n\n========================================")
        print(f"测试意图：{item}")
        print("========================================")

        result = agency_tool(item)

        print(result["summary"])
        print("\n匹配机构：")
        print(format_agencies_for_answer(result["agencies"]))