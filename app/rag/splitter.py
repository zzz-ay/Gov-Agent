from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_documents(
    documents: List[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 80,
) -> List[Document]:
    """
    将政策文档切分成适合向量检索的小文本块。

    参数：
    - documents: 原始文档列表
    - chunk_size: 每个文本块最大长度
    - chunk_overlap: 相邻文本块重叠长度

    返回：
    - 切分后的 Document 列表
    """
    if not documents:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n\n",
            "\n",
            "。",
            "；",
            "，",
            " ",
            "",
        ],
    )

    chunks = splitter.split_documents(documents)

    for idx, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = idx

    return chunks


if __name__ == "__main__":
    from app.config import RAW_DOCS_DIR, CHUNK_SIZE, CHUNK_OVERLAP
    from app.rag.loader import load_documents

    docs = load_documents(RAW_DOCS_DIR)

    chunks = split_documents(
        docs,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    print("\n========== 文档切分测试 ==========")
    print(f"原始文档片段数: {len(docs)}")
    print(f"切分后文本块数: {len(chunks)}")

    if chunks:
        print("\n第一个文本块内容预览:")
        print(chunks[0].page_content[:300])

        print("\nmetadata:")
        print(chunks[0].metadata)