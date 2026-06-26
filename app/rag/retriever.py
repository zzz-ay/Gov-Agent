from typing import Dict, List, Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document

from app.config import (
    VECTORSTORE_DIR,
    COLLECTION_NAME,
    RETRIEVER_TOP_K,
    check_config,
)

from app.rag.embedder import get_embedding_model


def get_vectorstore() -> Chroma:
    """
    加载已经构建好的 Chroma 向量库。
    """
    check_config()

    embeddings = get_embedding_model()

    vectorstore = Chroma(
        persist_directory=str(VECTORSTORE_DIR),
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )

    return vectorstore


def build_source_filter(source_ids: Optional[List[str]] = None) -> Optional[Dict]:
    if not source_ids:
        return None

    cleaned = [source_id for source_id in source_ids if source_id]
    if not cleaned:
        return None

    if len(cleaned) == 1:
        return {"knowledge_source_id": cleaned[0]}

    return {"knowledge_source_id": {"$in": cleaned}}


def get_retriever(top_k: int = RETRIEVER_TOP_K, source_ids: Optional[List[str]] = None):
    """
    创建检索器。
    """
    vectorstore = get_vectorstore()

    search_kwargs = {"k": top_k}
    source_filter = build_source_filter(source_ids)
    if source_filter:
        search_kwargs["filter"] = source_filter

    retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)

    return retriever


def retrieve_documents(
    query: str,
    top_k: int = RETRIEVER_TOP_K,
    source_ids: Optional[List[str]] = None,
) -> List[Document]:
    """
    根据用户问题检索相关政策片段。
    """
    retriever = get_retriever(top_k=top_k, source_ids=source_ids)
    docs = retriever.invoke(query)
    return docs


def print_retrieved_docs(docs: List[Document]):
    """
    打印检索结果，方便调试。
    """
    print("\n========== 检索结果 ==========")

    if not docs:
        print("没有检索到相关文档。")
        return

    for idx, doc in enumerate(docs, start=1):
        metadata = doc.metadata

        print(f"\n---------- 结果 {idx} ----------")
        print(f"标题: {metadata.get('title', '')}")
        print(f"发布机构: {metadata.get('source_org', '')}")
        print(f"发布时间: {metadata.get('publish_date', '')}")
        print(f"适用地区: {metadata.get('region', '')}")
        print(f"主题: {metadata.get('topic', '')}")
        print(f"文件类型: {metadata.get('doc_type', '')}")
        print(f"权威级别: {metadata.get('authority_level', '')}")
        print(f"来源文件: {metadata.get('source', '')}")
        print(f"来源链接: {metadata.get('source_url', '')}")
        print(f"页码: {metadata.get('page', '')}")
        print("\n内容预览:")
        print(doc.page_content[:500])


if __name__ == "__main__":
    test_questions = [
        "高校毕业生档案可以自己携带吗？",
        "现在还需要就业报到证吗？",
        "河北高校毕业生可以申请哪些就业补贴？",
        "毕业去向登记系统怎么填写档案转递信息？",
        "石家庄毕业生档案接收怎么办理？",
    ]

    for question in test_questions:
        print("\n\n========================================")
        print(f"测试问题: {question}")
        print("========================================")

        results = retrieve_documents(question, top_k=4)
        print_retrieved_docs(results)
