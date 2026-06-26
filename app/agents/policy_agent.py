
# from typing import Dict, Any, List

# from langchain_core.documents import Document

# from app.config import RETRIEVER_TOP_K
# from app.rag.retriever import retrieve_documents


# INTENT_KEYWORDS = {
#     "archive_transfer": "档案转递 档案接收 档案存放 公共就业人才服务机构",
#     "graduation_registration": "毕业去向登记 档案信息登记 全国高校毕业生毕业去向登记系统",
#     "registration_certificate": "就业报到证 取消就业报到证 报到证补办 改派",
#     "employment_policy": "高校毕业生就业政策 就业服务 就业见习 公共就业服务",
#     "employment_subsidy": "高校毕业生就业补贴 求职创业补贴 安家补贴 人才补贴 见习补贴",
#     "social_security": "灵活就业社保补贴 社会保险补贴 社保缴纳",
#     "entrepreneurship": "创业担保贷款 创业补贴 创业扶持 高校毕业生创业",
#     "hukou": "户籍迁移 户口迁移 落户 高校毕业生落户",
#     "general": "高校毕业生 政务服务 政策 办事指南",
# }


# def build_search_query(question: str, intent_result: Dict[str, Any]) -> str:
#     """
#     根据用户问题、意图和地区，构造增强检索 query。

#     这样做的目的：
#     - 原始问题可能太短
#     - 加上 intent 关键词可以提升检索命中率
#     - 加上地区可以优先命中地方政策
#     """
#     intent = intent_result.get("intent", "general")
#     region = intent_result.get("region", "未明确")

#     intent_keywords = INTENT_KEYWORDS.get(
#         intent,
#         INTENT_KEYWORDS["general"],
#     )

#     query_parts = [question, intent_keywords]

#     if region and region != "未明确":
#         query_parts.append(region)

#     enhanced_query = " ".join(query_parts)

#     return enhanced_query


# def retrieve_policy_docs(
#     question: str,
#     intent_result: Dict[str, Any],
#     top_k: int = RETRIEVER_TOP_K,
# ) -> List[Document]:
#     """
#     政策检索 Agent。

#     输入：
#     - question: 用户原始问题
#     - intent_result: Intent Agent 的识别结果

#     输出：
#     - 检索到的政策 Document 列表
#     """
#     enhanced_query = build_search_query(question, intent_result)

#     docs = retrieve_documents(
#         query=enhanced_query,
#         top_k=top_k,
#     )

#     return docs


# def summarize_policy_docs(docs: List[Document]) -> List[Dict[str, Any]]:
#     """
#     将检索结果整理为更适合 LangGraph State 保存的结构。
#     """
#     results = []

#     for idx, doc in enumerate(docs, start=1):
#         metadata = doc.metadata

#         results.append(
#             {
#                 "index": idx,
#                 "title": metadata.get("title", ""),
#                 "source_org": metadata.get("source_org", ""),
#                 "publish_date": metadata.get("publish_date", ""),
#                 "region": metadata.get("region", ""),
#                 "topic": metadata.get("topic", ""),
#                 "doc_type": metadata.get("doc_type", ""),
#                 "authority_level": metadata.get("authority_level", ""),
#                 "source_file": metadata.get("source", ""),
#                 "source_url": metadata.get("source_url", ""),
#                 "page": metadata.get("page", ""),
#                 "content": doc.page_content,
#             }
#         )

#     return results


# def policy_agent(
#     question: str,
#     intent_result: Dict[str, Any],
#     top_k: int = RETRIEVER_TOP_K,
# ) -> Dict[str, Any]:
#     """
#     Policy Agent 主函数。

#     返回结构：
#     {
#         "enhanced_query": "...",
#         "policy_docs": [Document, ...],
#         "policy_sources": [...]
#     }
#     """
#     enhanced_query = build_search_query(question, intent_result)

#     docs = retrieve_documents(
#         query=enhanced_query,
#         top_k=top_k,
#     )

#     sources = summarize_policy_docs(docs)

#     return {
#         "enhanced_query": enhanced_query,
#         "policy_docs": docs,
#         "policy_sources": sources,
#     }


# if __name__ == "__main__":
#     from app.agents.intent_agent import classify_intent

#     test_questions = [
#         "高校毕业生档案可以自己携带吗？",
#         "现在还需要就业报到证吗？",
#         "毕业去向登记系统怎么填写档案转递信息？",
#         "河北高校毕业生可以申请哪些就业补贴？",
#         "石家庄毕业生档案接收怎么办理？",
#         "灵活就业社保补贴怎么申请？",
#     ]

#     for question in test_questions:
#         print("\n\n========================================")
#         print(f"测试问题：{question}")
#         print("========================================")

#         intent_result = classify_intent(question)
#         print("\n[Intent Agent 结果]")
#         print(intent_result)

#         result = policy_agent(
#             question=question,
#             intent_result=intent_result,
#             top_k=4,
#         )

#         print("\n[增强检索 Query]")
#         print(result["enhanced_query"])

#         print("\n[Policy Agent 检索结果]")
#         for source in result["policy_sources"]:
#             print("\n------------------------------")
#             print(f"资料 {source['index']}")
#             print(f"标题：{source['title']}")
#             print(f"发布机构：{source['source_org']}")
#             print(f"地区：{source['region']}")
#             print(f"主题：{source['topic']}")
#             print(f"来源文件：{source['source_file']}")
#             print(f"来源链接：{source['source_url']}")
#             if source["page"] != "":
#                 print(f"页码：{source['page']}")
#             print("内容预览：")
#             print(source["content"][:300])

import json
from typing import Dict, Any, List

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from app.config import ENABLE_WEB_SEARCH, RETRIEVER_TOP_K
from app.llm.provider import get_chat_llm
from app.rag.retriever import retrieve_documents


INTENT_KEYWORDS = {
    "archive_transfer": "档案转递 档案接收 档案存放 公共就业人才服务机构",
    "graduation_registration": "毕业去向登记 档案信息登记 全国高校毕业生毕业去向登记系统",
    "registration_certificate": "就业报到证 取消就业报到证 报到证补办 改派",
    "employment_policy": "高校毕业生就业政策 就业服务 就业见习 公共就业服务",
    "employment_subsidy": "高校毕业生就业补贴 求职创业补贴 安家补贴 人才补贴 见习补贴",
    "social_security": "灵活就业社保补贴 社会保险补贴 社保缴纳",
    "entrepreneurship": "创业担保贷款 创业补贴 创业扶持 高校毕业生创业",
    "hukou": "户籍迁移 户口迁移 落户 高校毕业生落户",
    "general": "高校毕业生 政务服务 政策 办事指南",
}


def build_search_query(question: str, intent_result: Dict[str, Any]) -> str:
    """
    根据用户问题、意图和地区，构造增强检索 query。

    这样做的目的：
    - 原始问题可能太短
    - 加上 intent 关键词可以提升检索命中率
    - 加上地区可以优先命中地方政策
    """
    intent = intent_result.get("intent", "general")
    region = intent_result.get("region", "未明确")

    intent_keywords = INTENT_KEYWORDS.get(
        intent,
        INTENT_KEYWORDS["general"],
    )

    query_parts = [question, intent_keywords]

    if region and region != "未明确":
        query_parts.append(region)

    enhanced_query = " ".join(query_parts)

    return enhanced_query


def retrieve_policy_docs(
    question: str,
    intent_result: Dict[str, Any],
    top_k: int = RETRIEVER_TOP_K,
    source_ids: List[str] = None,
) -> List[Document]:
    """
    政策检索 Agent。

    输入：
    - question: 用户原始问题
    - intent_result: Intent Agent 的识别结果

    输出：
    - 检索到的政策 Document 列表
    """
    enhanced_query = build_search_query(question, intent_result)

    docs = retrieve_documents(
        query=enhanced_query,
        top_k=top_k,
        source_ids=source_ids,
    )

    return docs


def summarize_policy_docs(docs: List[Document]) -> List[Dict[str, Any]]:
    """
    将检索结果整理为更适合 LangGraph State 保存的结构。
    """
    results = []

    for idx, doc in enumerate(docs, start=1):
        metadata = doc.metadata

        results.append(
            {
                "index": idx,
                "title": metadata.get("title", ""),
                "source_org": metadata.get("source_org", ""),
                "knowledge_source_id": metadata.get("knowledge_source_id", ""),
                "knowledge_source_name": metadata.get("knowledge_source_name", ""),
                "knowledge_source_type": metadata.get("knowledge_source_type", ""),
                "publish_date": metadata.get("publish_date", ""),
                "region": metadata.get("region", ""),
                "topic": metadata.get("topic", ""),
                "doc_type": metadata.get("doc_type", ""),
                "authority_level": metadata.get("authority_level", ""),
                "source_file": metadata.get("source", ""),
                "source_url": metadata.get("source_url", ""),
                "page": metadata.get("page", ""),
                "content": doc.page_content,
            }
        )

    return results


EVIDENCE_GROUPS = {
    "gov_policy": {
        "label": "政策依据",
        "description": "用于回答办理条件、材料、流程、补贴口径、公共服务事项。",
    },
    "laws": {
        "label": "法律法规依据",
        "description": "用于回答上位法依据、法定义务、行政许可/处罚、合规要求。",
    },
    "legal_cases": {
        "label": "案例参考",
        "description": "用于回答类案事实、裁判倾向、风险后果和争议参考。",
    },
    "public_notices": {
        "label": "公告通知",
        "description": "用于回答申报批次、名单公示、截止时间、最新通知。",
    },
}


def build_evidence_groups(sources: List[Dict[str, Any]]) -> Dict[str, Any]:
    groups: Dict[str, Dict[str, Any]] = {}

    for source in sources:
        source_id = source.get("knowledge_source_id") or "unknown"
        group_meta = EVIDENCE_GROUPS.get(
            source_id,
            {
                "label": source.get("knowledge_source_name") or source_id,
                "description": "",
            },
        )

        if source_id not in groups:
            groups[source_id] = {
                "source_id": source_id,
                "label": group_meta["label"],
                "description": group_meta["description"],
                "count": 0,
                "sources": [],
            }

        groups[source_id]["sources"].append(source)
        groups[source_id]["count"] += 1

    ordered_groups = [
        groups[source_id]
        for source_id in ["gov_policy", "laws", "legal_cases", "public_notices", "unknown"]
        if source_id in groups
    ]
    ordered_groups.extend(
        group
        for source_id, group in groups.items()
        if source_id not in {"gov_policy", "laws", "legal_cases", "public_notices", "unknown"}
    )

    return {
        "groups": ordered_groups,
        "source_distribution": {
            group["source_id"]: group["count"]
            for group in ordered_groups
        },
        "total_sources": len(sources),
    }


# ====================== 新增：联网搜索能力 ======================
def get_policy_decision_llm():
    return get_chat_llm(temperature=0.1)

def get_web_search_llm():
    return get_chat_llm(
        temperature=0.2,
        extra_body={"enable_search": True},
    )

def check_if_need_web_search(question: str, local_content: str) -> bool:
    llm = get_policy_decision_llm()
    prompt = f"""
请判断本地知识库是否足够完整、准确、最新地回答用户问题。
如果信息不足、过时、缺失区县细则，请返回 {{ "need_search": true }}
如果信息足够，请返回 {{ "need_search": false }}
只返回JSON，不要其他内容。

用户问题：{question}
本地内容：{local_content}
"""
    try:
        resp = llm.invoke(prompt)
        data = json.loads(resp.content.replace("```json", "").replace("```", "").strip())
        return data.get("need_search", False)
    except:
        return True  # 解析失败默认联网更安全

def add_web_search_result(question: str, intent_result: Dict[str, Any], sources: List[Dict]):
    try:
        llm = get_web_search_llm()
        prompt = ChatPromptTemplate.from_messages([
            ("system", "你是权威政务政策助手，请根据问题提供官方、准确的回答。"),
            ("human", question)
        ])
        chain = prompt | llm
        resp = chain.invoke({})

        new_item = {
            "index": len(sources) + 1,
            "title": "🌍 阿里云百炼全网权威政策检索",
            "source_org": "互联网权威政务信息",
            "publish_date": "最新",
            "region": intent_result.get("region", "全网"),
            "topic": "最新政策补充",
            "doc_type": "联网检索",
            "authority_level": "权威参考",
            "source_file": "DashScope Web Search",
            "source_url": "",
            "page": "",
            "content": resp.content
        }
        sources.append(new_item)
        print("\n✅ 已联网补充最新政策")
    except Exception as e:
        print(f"\n❌ 联网失败：{e}")

# ==============================================================


def policy_agent(
    question: str,
    intent_result: Dict[str, Any],
    source_router_result: Dict[str, Any] = None,
    top_k: int = RETRIEVER_TOP_K,
) -> Dict[str, Any]:
    """
    Policy Agent 主函数。

    返回结构：
    {
        "enhanced_query": "...",
        "policy_docs": [Document, ...],
        "policy_sources": [...]
    }
    """
    enhanced_query = build_search_query(question, intent_result)
    source_router_result = source_router_result or {}
    source_ids = source_router_result.get("source_ids") or []

    docs = retrieve_documents(
        query=enhanced_query,
        top_k=top_k,
        source_ids=source_ids,
    )

    sources = summarize_policy_docs(docs)
    evidence_result = build_evidence_groups(sources)

    # ====================== 新增：自动判断 + 联网搜索 ======================
    if ENABLE_WEB_SEARCH:
        local_content = "\n".join([s["content"] for s in sources])
        need_search = check_if_need_web_search(question, local_content)

        if need_search:
            add_web_search_result(question, intent_result, sources)
    # ====================================================================

    return {
        "enhanced_query": enhanced_query,
        "source_ids": source_ids,
        "source_router_result": source_router_result,
        "policy_docs": docs,
        "policy_sources": sources,
        "evidence_result": evidence_result,
        "evidence_groups": evidence_result["groups"],
        "source_distribution": evidence_result["source_distribution"],
    }


if __name__ == "__main__":
    from app.agents.intent_agent import classify_intent

    test_questions = [
        "高校毕业生档案可以自己携带吗？",
        "现在还需要就业报到证吗？",
        "毕业去向登记系统怎么填写档案转递信息？",
        "河北高校毕业生可以申请哪些就业补贴？",
        "石家庄毕业生档案接收怎么办理？",
        "灵活就业社保补贴怎么申请？",
    ]

    for question in test_questions:
        print("\n\n========================================")
        print(f"测试问题：{question}")
        print("========================================")

        intent_result = classify_intent(question)
        print("\n[Intent Agent 结果]")
        print(intent_result)

        result = policy_agent(
            question=question,
            intent_result=intent_result,
            top_k=4,
        )

        print("\n[增强检索 Query]")
        print(result["enhanced_query"])

        print("\n[Policy Agent 检索结果]")
        for source in result["policy_sources"]:
            print("\n------------------------------")
            print(f"资料 {source['index']}")
            print(f"标题：{source['title']}")
            print(f"发布机构：{source['source_org']}")
            print(f"地区：{source['region']}")
            print(f"主题：{source['topic']}")
            print(f"来源文件：{source['source_file']}")
            print(f"来源链接：{source['source_url']}")
            if source["page"] != "":
                print(f"页码：{source['page']}")
            print("内容预览：")
            print(source["content"][:300])
