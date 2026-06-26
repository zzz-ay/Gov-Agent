from typing import List, Tuple

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from app.config import RETRIEVER_TOP_K
from app.llm.provider import get_chat_llm

from app.rag.retriever import retrieve_documents


SYSTEM_PROMPT = """
你是 GOV-RAG 政务问答系统中的政务服务助手。
你的任务是根据检索到的政策资料，回答高校毕业生相关政务服务问题。

重要原则：
1. 只能依据“检索资料”回答，不允许根据常识、经验或模型自身知识补充政策内容。
2. 不得编造政策名称、适用对象、补贴金额、申请条件、办理材料、办理流程、时间限制。
3. 如果检索资料没有明确说明某项内容，必须写“当前检索资料未明确说明”。
4. 涉及地区差异时，必须说明“以当地主管部门或政务服务窗口最新要求为准”。
5. 涉及金额、材料、流程、时限时，必须能在检索资料中找到依据。
6. 不要扩展回答与用户问题无关的内容。
7. 回答应简洁、准确、可追溯，适合高校毕业生阅读。
8. 不得说明检索资料中未明确提到的法律后果、风险后果、责任后果或办理后果。
9. 如果需要提醒风险，只能使用检索资料中已经出现的原文或近义表述。

回答格式：
一、简要结论
二、政策依据
三、办理要求 / 操作说明
四、注意事项
五、参考来源

格式要求：
- 如果用户问题比较简单，可以不强行写满所有部分。
- 每条政策依据后尽量标注对应资料编号，例如“根据资料1”。
- 参考来源只列出实际使用到的资料。
- 不要输出检索资料之外的推测性风险、后果或建议。

检索资料：
{context}
"""


def get_llm():
    """
    创建 Qwen Chat 模型。
    当前通过阿里云百炼 DashScope OpenAI-compatible API 调用。
    """
    return get_chat_llm(temperature=0.2)


def format_docs_for_context(docs: List[Document]) -> str:
    """
    将检索到的 Document 格式化为 LLM 可读的上下文。
    """
    formatted_docs = []

    for idx, doc in enumerate(docs, start=1):
        metadata = doc.metadata

        title = metadata.get("title", "")
        source_org = metadata.get("source_org", "")
        publish_date = metadata.get("publish_date", "")
        region = metadata.get("region", "")
        topic = metadata.get("topic", "")
        doc_type = metadata.get("doc_type", "")
        authority_level = metadata.get("authority_level", "")
        source = metadata.get("source", "")
        source_url = metadata.get("source_url", "")
        page = metadata.get("page", "")

        page_info = f"页码：{page}" if page != "" else "页码：无"

        formatted = f"""
【资料 {idx}】
标题：{title}
发布机构：{source_org}
发布时间：{publish_date}
适用地区：{region}
主题：{topic}
文件类型：{doc_type}
权威级别：{authority_level}
来源文件：{source}
来源链接：{source_url}
{page_info}

正文片段：
{doc.page_content}
"""
        formatted_docs.append(formatted.strip())

    return "\n\n".join(formatted_docs)


def format_sources(docs: List[Document]) -> List[dict]:
    """
    整理来源信息，后续前端展示使用。
    """
    sources = []

    for idx, doc in enumerate(docs, start=1):
        metadata = doc.metadata

        sources.append(
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
                "source_file": metadata.get("source", ""),
                "source_url": metadata.get("source_url", ""),
                "page": metadata.get("page", ""),
                "content_preview": doc.page_content[:300],
            }
        )

    return sources


def answer_question(
    question: str,
    top_k: int = RETRIEVER_TOP_K,
) -> Tuple[str, List[dict], List[Document]]:
    """
    RAG 问答主函数。

    返回：
    - answer: LLM 生成的回答
    - sources: 结构化来源信息
    - docs: 原始检索文档
    """
    docs = retrieve_documents(question, top_k=top_k)

    if not docs:
        return (
            "当前知识库没有检索到相关政策资料，建议咨询当地主管部门或政务服务窗口。",
            [],
            [],
        )

    context = format_docs_for_context(docs)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "用户问题：{question}"),
        ]
    )

    llm = get_llm()
    chain = prompt | llm

    response = chain.invoke(
        {
            "context": context,
            "question": question,
        }
    )

    answer = response.content
    sources = format_sources(docs)

    return answer, sources, docs


def print_answer_with_sources(answer: str, sources: List[dict]):
    """
    命令行测试输出。
    """
    print("\n========== GOV-RAG 回答 ==========\n")
    print(answer)

    print("\n========== 参考来源 ==========")

    for source in sources:
        print(f"\n[{source['index']}] {source['title']}")
        print(f"发布机构：{source['source_org']}")
        print(f"发布时间：{source['publish_date']}")
        print(f"适用地区：{source['region']}")
        print(f"主题：{source['topic']}")
        print(f"来源文件：{source['source_file']}")
        print(f"来源链接：{source['source_url']}")

        if source["page"] != "":
            print(f"页码：{source['page']}")

        print(f"内容预览：{source['content_preview']}")


if __name__ == "__main__":
    test_questions = [
        "高校毕业生档案可以自己携带吗？",
        "现在还需要就业报到证吗？",
        "毕业去向登记系统怎么填写档案转递信息？",
        "河北高校毕业生可以申请哪些就业补贴？",
        "石家庄毕业生档案接收怎么办理？",
    ]

    for question in test_questions:
        print("\n\n========================================")
        print(f"测试问题：{question}")
        print("========================================")

        answer, sources, _ = answer_question(question)
        print_answer_with_sources(answer, sources)
