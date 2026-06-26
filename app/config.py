import os
from pathlib import Path
from dotenv import load_dotenv


# =========================
# Load .env
# =========================

load_dotenv()


# =========================
# Project Paths
# =========================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
RAW_DOCS_DIR = DATA_DIR / "raw_docs"
KNOWLEDGE_SOURCES_DIR = DATA_DIR / "knowledge_sources"
PROCESSED_DIR = DATA_DIR / "processed"

VECTORSTORE_DIR = BASE_DIR / "vectorstore"


# =========================
# LLM Provider
# =========================

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "qwen")


# =========================
# DashScope / OpenAI-compatible API
# =========================

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or DASHSCOPE_API_KEY
OPENAI_BASE_URL = os.getenv(
    "OPENAI_BASE_URL",
    "https://dashscope.aliyuncs.com/compatible-mode/v1",
)


# =========================
# Model Settings
# =========================

CHAT_MODEL = os.getenv("CHAT_MODEL", "qwen-plus")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-v4")
EMBEDDING_DIMENSIONS = int(os.getenv("EMBEDDING_DIMENSIONS", "1024"))


# =========================
# Vector Store Settings
# =========================

COLLECTION_NAME = os.getenv("COLLECTION_NAME", "gov_policy")
KNOWLEDGE_SOURCES = os.getenv("KNOWLEDGE_SOURCES", "gov_policy")


# =========================
# RAG Settings
# =========================

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "80"))
RETRIEVER_TOP_K = int(os.getenv("RETRIEVER_TOP_K", "4"))


# =========================
# Runtime Feature Flags
# =========================

ENABLE_WEB_SEARCH = os.getenv("ENABLE_WEB_SEARCH", "false").lower() == "true"


# =========================
# Validation
# =========================

def check_config():
    """
    检查关键配置是否存在。
    """
    missing = []

    if not OPENAI_API_KEY:
        missing.append("OPENAI_API_KEY 或 DASHSCOPE_API_KEY")

    if not OPENAI_BASE_URL:
        missing.append("OPENAI_BASE_URL")

    if not CHAT_MODEL:
        missing.append("CHAT_MODEL")

    if not EMBEDDING_MODEL:
        missing.append("EMBEDDING_MODEL")

    if missing:
        raise ValueError(
            "配置缺失，请检查 .env 文件："
            + ", ".join(missing)
        )

    return True


def print_config():
    """
    打印当前项目配置，注意不会打印完整 API Key。
    """
    masked_key = None
    if OPENAI_API_KEY:
        masked_key = OPENAI_API_KEY[:6] + "****" + OPENAI_API_KEY[-4:]

    print("========== GOV-RAG Config ==========")
    print(f"BASE_DIR: {BASE_DIR}")
    print(f"RAW_DOCS_DIR: {RAW_DOCS_DIR}")
    print(f"KNOWLEDGE_SOURCES_DIR: {KNOWLEDGE_SOURCES_DIR}")
    print(f"VECTORSTORE_DIR: {VECTORSTORE_DIR}")
    print(f"LLM_PROVIDER: {LLM_PROVIDER}")
    print(f"OPENAI_BASE_URL: {OPENAI_BASE_URL}")
    print(f"OPENAI_API_KEY: {masked_key}")
    print(f"CHAT_MODEL: {CHAT_MODEL}")
    print(f"EMBEDDING_MODEL: {EMBEDDING_MODEL}")
    print(f"EMBEDDING_DIMENSIONS: {EMBEDDING_DIMENSIONS}")
    print(f"COLLECTION_NAME: {COLLECTION_NAME}")
    print(f"KNOWLEDGE_SOURCES: {KNOWLEDGE_SOURCES}")
    print(f"CHUNK_SIZE: {CHUNK_SIZE}")
    print(f"CHUNK_OVERLAP: {CHUNK_OVERLAP}")
    print(f"RETRIEVER_TOP_K: {RETRIEVER_TOP_K}")
    print("====================================")
