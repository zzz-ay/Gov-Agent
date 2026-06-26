import shutil
import os

from langchain_chroma import Chroma

from app.config import (
    BASE_DIR,
    VECTORSTORE_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    COLLECTION_NAME,
    check_config,
    print_config,
)

from app.rag.loader import load_documents_from_enabled_sources
from app.rag.splitter import split_documents
from app.rag.embedder import get_embedding_model


def reset_vectorstore_dir():
    resolved_vectorstore_dir = VECTORSTORE_DIR.resolve()
    resolved_base_dir = BASE_DIR.resolve()

    try:
        resolved_vectorstore_dir.relative_to(resolved_base_dir)
    except ValueError as exc:
        raise RuntimeError(
            f"Refusing to reset vectorstore outside project: {resolved_vectorstore_dir}"
        ) from exc

    if resolved_vectorstore_dir.exists():
        shutil.rmtree(resolved_vectorstore_dir)

    resolved_vectorstore_dir.mkdir(parents=True, exist_ok=True)


def batched(items, batch_size: int):
    for start in range(0, len(items), batch_size):
        yield start, items[start : start + batch_size]


def build_vectorstore():
    """
    构建 GOV-RAG 政策知识向量库。
    """
    print("========== GOV-RAG 向量库构建开始 ==========")

    check_config()
    print_config()

    print("\n[1/4] 正在加载政策文档...")
    documents = load_documents_from_enabled_sources()
    print(f"原始文档片段数量: {len(documents)}")

    if not documents:
        raise ValueError(
            "No documents were loaded. Please check enabled knowledge sources."
        )

    print("\n[2/4] 正在切分政策文档...")
    chunks = split_documents(
        documents=documents,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    print(f"切分后的文本块数量: {len(chunks)}")

    if not chunks:
        raise ValueError("文档切分后为空，请检查原始文档内容。")

    print("\n[3/4] 正在初始化 Embedding 模型...")
    embeddings = get_embedding_model()

    print("\n[4/4] Resetting and writing Chroma vectorstore...")
    reset_vectorstore_dir()

    batch_size = int(os.getenv("VECTORSTORE_BATCH_SIZE", "256"))
    vectorstore = Chroma(
        embedding_function=embeddings,
        persist_directory=str(VECTORSTORE_DIR),
        collection_name=COLLECTION_NAME,
    )

    for start, batch in batched(chunks, batch_size):
        vectorstore.add_documents(batch)
        end = start + len(batch)
        print(f"[vectorstore] written {end}/{len(chunks)} chunks")

    count = vectorstore._collection.count()

    print("\n========== GOV-RAG 向量库构建完成 ==========")
    print(f"向量库保存路径: {VECTORSTORE_DIR}")
    print(f"Collection 名称: {COLLECTION_NAME}")
    print(f"Collection 文档数量: {count}")

    print("\n========== 示例 Chunk Metadata ==========")
    sample = chunks[0]
    for key, value in sample.metadata.items():
        print(f"{key}: {value}")

    print("\n========== 示例 Chunk 内容 ==========")
    print(sample.page_content[:500])

    return vectorstore


if __name__ == "__main__":
    build_vectorstore()
