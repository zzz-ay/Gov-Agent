from typing import Any, Dict, List


FLOW_DEFINITIONS: Dict[str, Dict[str, Any]] = {
    "policy_service": {
        "name": "政策服务咨询",
        "description": "面向就业创业、社保医保、户籍居住证、住房公积金等政策的条件、材料、流程问答。",
        "source_ids": ["gov_policy", "laws"],
        "steps": [
            "识别用户咨询主题和地区约束",
            "检索政策文件和必要法律依据",
            "整理适用条件、办理材料、办理流程和注意事项",
            "返回引用来源，提示以主管部门最新口径为准",
        ],
        "output_focus": ["适用条件", "材料清单", "办理流程", "政策依据"],
    },
    "legal_basis": {
        "name": "法律法规依据分析",
        "description": "围绕劳动用工、行政许可/处罚、数据安全、个人信息保护等问题查找法条依据。",
        "source_ids": ["laws", "gov_policy"],
        "steps": [
            "识别法律主题和争议焦点",
            "优先检索法律法规库",
            "补充检索相关政策或监管文件",
            "按法条依据、风险点、建议动作组织回答",
        ],
        "output_focus": ["法条依据", "合规风险", "建议动作", "引用来源"],
    },
    "case_reference": {
        "name": "案例参考检索",
        "description": "从法律案例数据中查找相似事实，用于辅助判断争议风险和裁判倾向。",
        "source_ids": ["legal_cases", "laws"],
        "steps": [
            "提取事实要素、主体关系和争议类型",
            "检索相似法律案例",
            "补充匹配相关法律法规",
            "总结相似点、差异点和风险提示",
        ],
        "output_focus": ["相似案例", "争议焦点", "裁判倾向", "法律依据"],
    },
    "enterprise_compliance": {
        "name": "企业合规问答",
        "description": "面向企业登记、劳动用工、数据合规、市场监管等综合合规问题。",
        "source_ids": ["gov_policy", "laws", "legal_cases"],
        "steps": [
            "识别企业场景和合规主题",
            "同时检索政策、法律法规和案例库",
            "拆分为义务要求、风险后果、整改建议",
            "返回可执行的合规检查清单",
        ],
        "output_focus": ["合规义务", "风险后果", "整改建议", "检查清单"],
    },
}


KEYWORD_RULES = [
    (
        "case_reference",
        [
            "案例",
            "判决",
            "法院",
            "裁判",
            "类似情况",
            "纠纷",
            "争议",
            "诈骗",
            "刑事",
            "民事",
        ],
    ),
    (
        "enterprise_compliance",
        [
            "企业",
            "公司",
            "合规",
            "监管",
            "市场主体",
            "登记",
            "处罚风险",
            "数据合规",
            "用工风险",
        ],
    ),
    (
        "legal_basis",
        [
            "法律",
            "法规",
            "法条",
            "依据",
            "劳动合同",
            "行政许可",
            "行政处罚",
            "个人信息",
            "数据安全",
            "网络安全",
            "违法",
            "处罚",
        ],
    ),
    (
        "policy_service",
        [
            "政策",
            "补贴",
            "申请",
            "办理",
            "流程",
            "材料",
            "条件",
            "社保",
            "医保",
            "就业",
            "创业",
            "户籍",
            "居住证",
            "公积金",
        ],
    ),
]


def _contains_any(text: str, keywords: List[str]) -> bool:
    return any(keyword in text for keyword in keywords)


def classify_business_flow(question: str, enabled_source_ids: List[str] = None) -> Dict[str, Any]:
    text = question or ""
    enabled = set(enabled_source_ids or [])

    flow_id = "policy_service"
    matched_keywords: List[str] = []

    for candidate_flow_id, keywords in KEYWORD_RULES:
        hits = [keyword for keyword in keywords if keyword in text]
        if hits:
            flow_id = candidate_flow_id
            matched_keywords = hits[:8]
            break

    definition = FLOW_DEFINITIONS[flow_id]
    requested_source_ids = list(definition["source_ids"])

    if enabled:
        source_ids = [source_id for source_id in requested_source_ids if source_id in enabled]
        if not source_ids:
            source_ids = [next(iter(enabled))]
    else:
        source_ids = requested_source_ids

    return {
        "flow_id": flow_id,
        "flow_name": definition["name"],
        "description": definition["description"],
        "steps": definition["steps"],
        "output_focus": definition["output_focus"],
        "source_ids": source_ids,
        "requested_source_ids": requested_source_ids,
        "matched_keywords": matched_keywords,
        "confidence": 0.72 if matched_keywords else 0.45,
    }


def list_business_flows() -> List[Dict[str, Any]]:
    return [
        {
            "flow_id": flow_id,
            "flow_name": definition["name"],
            "description": definition["description"],
            "source_ids": definition["source_ids"],
            "steps": definition["steps"],
            "output_focus": definition["output_focus"],
        }
        for flow_id, definition in FLOW_DEFINITIONS.items()
    ]
