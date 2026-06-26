from typing import Dict, Any, List, Optional
from datetime import datetime
import re


AUTHORITY_SCORE_MAP = {
    "national": 5,
    "provincial": 4,
    "city": 3,
    "municipal": 3,
    "university": 2,
    "school": 2,
    "other": 1,
    "": 1,
}


OFFICIAL_DOMAINS = [
    "gov.cn",
    "moe.gov.cn",
    "mohrss.gov.cn",
    "hebei.gov.cn",
    "hebhrss.gov.cn",
    "rsj.sjz.gov.cn",
    "sjz.gov.cn",
    "baoding.gov.cn",
    "ncss.cn",
    "chsi.com.cn",
]


def parse_year_from_date(date_text: str) -> Optional[int]:
    """
    从发布时间字段中提取年份。

    支持：
    - 2023-06-13
    - 2023-05
    - 2024
    - 2025年6月
    """
    if not date_text:
        return None

    match = re.search(r"(20\d{2})", str(date_text))
    if not match:
        return None

    try:
        return int(match.group(1))
    except ValueError:
        return None


def get_authority_score(authority_level: str) -> int:
    """
    根据 authority_level 计算来源权威分。
    """
    if not authority_level:
        return 1

    normalized = str(authority_level).strip().lower()

    return AUTHORITY_SCORE_MAP.get(normalized, 1)


def is_official_source(source_url: str) -> bool:
    """
    判断来源链接是否疑似官方来源。
    """
    if not source_url:
        return False

    url = source_url.lower()

    return any(domain in url for domain in OFFICIAL_DOMAINS)


def check_region_match(
    user_region: str,
    policy_region: str,
) -> Dict[str, Any]:
    """
    判断政策地区与用户问题地区是否匹配。
    """
    user_region = user_region or "未明确"
    policy_region = policy_region or "未明确"

    if user_region == "未明确":
        return {
            "matched": True,
            "level": "unknown_user_region",
            "message": "用户未明确地区，暂不判断地区冲突。",
        }

    if policy_region == "全国":
        return {
            "matched": True,
            "level": "national_policy",
            "message": "政策为全国性政策，可作为通用依据。",
        }

    if user_region in policy_region or policy_region in user_region:
        return {
            "matched": True,
            "level": "direct_match",
            "message": "政策地区与用户问题地区匹配。",
        }

    if "河北" in user_region and "河北" in policy_region:
        return {
            "matched": True,
            "level": "province_match",
            "message": "政策地区与用户问题同属河北省，基本匹配。",
        }

    return {
        "matched": False,
        "level": "mismatch",
        "message": f"用户地区为“{user_region}”，政策地区为“{policy_region}”，存在地区不完全匹配。",
    }


def analyze_single_source(
    source: Dict[str, Any],
    user_region: str,
    current_year: int,
) -> Dict[str, Any]:
    """
    分析单条政策来源的时效性、权威性和地区匹配情况。
    """
    title = source.get("title", "")
    source_org = source.get("source_org", "")
    publish_date = source.get("publish_date", "")
    region = source.get("region", "")
    topic = source.get("topic", "")
    authority_level = source.get("authority_level", "")
    source_url = source.get("source_url", "")
    source_file = source.get("source_file", "")

    publish_year = parse_year_from_date(publish_date)

    issues = []
    suggestions = []

    # 1. 时效性判断
    if publish_year is None:
        freshness_level = "unknown"
        freshness_message = "未识别到明确发布时间。"
        issues.append("缺少明确发布时间")
        suggestions.append("建议补充政策发布时间，或办理前核验最新政策。")
    else:
        age = current_year - publish_year

        if age <= 1:
            freshness_level = "fresh"
            freshness_message = f"政策发布时间为 {publish_year} 年，时效性较好。"
        elif age <= 3:
            freshness_level = "acceptable"
            freshness_message = f"政策发布时间为 {publish_year} 年，仍可参考，但建议办理前核验最新要求。"
            suggestions.append("建议办理前以当地主管部门或政务服务窗口最新要求为准。")
        else:
            freshness_level = "old"
            freshness_message = f"政策发布时间为 {publish_year} 年，距今较久，可能存在更新。"
            issues.append("政策发布时间较早")
            suggestions.append("建议查询主管部门最新公告或通过政务服务窗口确认。")

    # 2. 来源权威性判断
    authority_score = get_authority_score(authority_level)
    official = is_official_source(source_url)

    if authority_score >= 4:
        authority_level_desc = "high"
        authority_message = "来源权威级别较高。"
    elif authority_score == 3:
        authority_level_desc = "medium"
        authority_message = "来源权威级别中等，适合作为地方办事参考。"
    else:
        authority_level_desc = "low"
        authority_message = "来源权威级别较低，建议优先核验官方来源。"
        issues.append("来源权威级别较低")
        suggestions.append("建议优先使用政府官网、人社部门、教育部门或政务服务网来源。")

    if not official:
        issues.append("来源链接未识别为常见官方域名")
        suggestions.append("建议核验该来源是否为官方发布渠道。")

    # 3. 地区匹配判断
    region_check = check_region_match(user_region, region)

    if not region_check["matched"]:
        issues.append("政策地区与用户地区不完全匹配")
        suggestions.append("建议优先检索用户所在地区的地方政策或办事指南。")

    # 4. 综合风险
    risk_score = 0

    if freshness_level == "old":
        risk_score += 35
    elif freshness_level == "unknown":
        risk_score += 20
    elif freshness_level == "acceptable":
        risk_score += 10

    if authority_level_desc == "low":
        risk_score += 25
    elif authority_level_desc == "medium":
        risk_score += 10

    if not official:
        risk_score += 15

    if not region_check["matched"]:
        risk_score += 25

    if risk_score >= 50:
        risk_level = "high"
    elif risk_score >= 25:
        risk_level = "medium"
    else:
        risk_level = "low"

    return {
        "title": title,
        "source_org": source_org,
        "publish_date": publish_date,
        "publish_year": publish_year,
        "region": region,
        "topic": topic,
        "authority_level": authority_level,
        "authority_score": authority_score,
        "authority_message": authority_message,
        "source_url": source_url,
        "source_file": source_file,
        "is_official_source": official,
        "freshness_level": freshness_level,
        "freshness_message": freshness_message,
        "region_check": region_check,
        "risk_level": risk_level,
        "risk_score": risk_score,
        "issues": list(dict.fromkeys(issues)),
        "suggestions": list(dict.fromkeys(suggestions)),
    }


def summarize_validity_results(
    source_checks: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    汇总所有政策来源的可靠性判断。
    """
    if not source_checks:
        return {
            "overall_risk_level": "high",
            "overall_score": 0,
            "summary": "未检索到政策来源，无法进行时效与来源可靠性判断。",
            "warnings": ["未检索到政策来源"],
            "suggestions": ["建议补充政策资料后再回答。"],
        }

    risk_values = {
        "low": 1,
        "medium": 2,
        "high": 3,
    }

    max_risk_value = max(
        risk_values.get(item.get("risk_level", "medium"), 2)
        for item in source_checks
    )

    if max_risk_value == 3:
        overall_risk_level = "high"
    elif max_risk_value == 2:
        overall_risk_level = "medium"
    else:
        overall_risk_level = "low"

    avg_risk_score = sum(item.get("risk_score", 0) for item in source_checks) / len(source_checks)
    overall_score = max(0, int(100 - avg_risk_score))

    warnings = []
    suggestions = []

    for item in source_checks:
        for issue in item.get("issues", []):
            warnings.append(issue)
        for suggestion in item.get("suggestions", []):
            suggestions.append(suggestion)

    warnings = list(dict.fromkeys(warnings))
    suggestions = list(dict.fromkeys(suggestions))

    if overall_risk_level == "low":
        summary = "当前检索到的政策资料整体时效性和来源可靠性较好。"
    elif overall_risk_level == "medium":
        summary = "当前检索到的政策资料整体可参考，但存在部分时效、地区或来源可靠性提示。"
    else:
        summary = "当前检索到的政策资料存在较高时效、地区或来源可靠性风险，建议进一步核验。"

    return {
        "overall_risk_level": overall_risk_level,
        "overall_score": overall_score,
        "summary": summary,
        "warnings": warnings,
        "suggestions": suggestions,
    }


def policy_validity_agent(
    question: str,
    intent_result: Dict[str, Any],
    policy_result: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Policy Validity Agent 主函数。

    输入：
    - question: 用户问题
    - intent_result: Intent Agent 输出
    - policy_result: Policy Agent 输出

    输出：
    {
        "overall_risk_level": "low / medium / high",
        "overall_score": 90,
        "summary": "...",
        "warnings": [...],
        "suggestions": [...],
        "source_checks": [...]
    }
    """
    user_region = intent_result.get("region", "未明确")
    policy_sources = policy_result.get("policy_sources", [])
    current_year = datetime.now().year

    source_checks = [
        analyze_single_source(
            source=source,
            user_region=user_region,
            current_year=current_year,
        )
        for source in policy_sources
    ]

    summary = summarize_validity_results(source_checks)

    return {
        "question": question,
        "user_region": user_region,
        "current_year": current_year,
        "overall_risk_level": summary["overall_risk_level"],
        "overall_score": summary["overall_score"],
        "summary": summary["summary"],
        "warnings": summary["warnings"],
        "suggestions": summary["suggestions"],
        "source_checks": source_checks,
    }


if __name__ == "__main__":
    from app.agents.intent_agent import classify_intent
    from app.agents.policy_agent import policy_agent

    test_questions = [
        "高校毕业生档案可以自己携带吗？",
        "现在还需要就业报到证吗？",
        "河北高校毕业生可以申请哪些就业补贴？",
        "石家庄毕业生档案接收怎么办理？",
        "我是2025届本科毕业生，在石家庄企业工作，社保交了6个月，可以申请安家补贴吗？",
    ]

    for question in test_questions:
        print("\n\n========================================")
        print(f"测试问题：{question}")
        print("========================================")

        intent_result = classify_intent(question)
        policy_result = policy_agent(question, intent_result)

        validity_result = policy_validity_agent(
            question=question,
            intent_result=intent_result,
            policy_result=policy_result,
        )

        print("\n========== Policy Validity Agent ==========")
        print(f"用户地区：{validity_result['user_region']}")
        print(f"综合风险等级：{validity_result['overall_risk_level']}")
        print(f"综合可靠性分数：{validity_result['overall_score']}")
        print(f"摘要：{validity_result['summary']}")

        print("\n风险提示：")
        for warning in validity_result["warnings"]:
            print(f"- {warning}")

        print("\n建议：")
        for suggestion in validity_result["suggestions"]:
            print(f"- {suggestion}")

        print("\n逐条来源检查：")
        for idx, check in enumerate(validity_result["source_checks"], start=1):
            print("\n------------------------------")
            print(f"来源 {idx}: {check['title']}")
            print(f"发布时间：{check['publish_date']}")
            print(f"地区：{check['region']}")
            print(f"权威级别：{check['authority_level']}")
            print(f"是否官方来源：{check['is_official_source']}")
            print(f"时效判断：{check['freshness_message']}")
            print(f"地区判断：{check['region_check']['message']}")
            print(f"风险等级：{check['risk_level']}")
