from langchain_openai import OpenAIEmbeddings

from app.config import (
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    EMBEDDING_MODEL,
    EMBEDDING_DIMENSIONS,
)


def get_embedding_model() -> OpenAIEmbeddings:
    """
    创建 Embedding 模型。

    当前使用：
    - 阿里云百炼 DashScope OpenAI-compatible API
    - text-embedding-v4

    注意：
    - DashScope embedding 接口要求 input 是 str 或 list[str]
    - langchain_openai 默认可能会先转 token
    - 所以这里设置 check_embedding_ctx_length=False
    - DashScope text-embedding-v4 单次 batch size 不能超过 10，所以这里必须设置 chunk_size=10。
    """
    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
        dimensions=EMBEDDING_DIMENSIONS,
        check_embedding_ctx_length=False,
        chunk_size=10,
    )

    return embeddings


if __name__ == "__main__":
    embeddings = get_embedding_model()

    query = "高校毕业生档案转递政策"
    vector = embeddings.embed_query(query)

    print("========== Embedding 测试 ==========")
    print(f"输入文本: {query}")
    print(f"向量维度: {len(vector)}")
    print(f"前 5 个值: {vector[:5]}")