from pathlib import Path
from typing import Dict, List, Tuple

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader,
)

from app.knowledge_sources import (
    KnowledgeSource,
    get_enabled_knowledge_sources,
    get_knowledge_source,
    get_knowledge_source_for_path,
)


META_KEYS = {
    "标题": "title",
    "发布机构": "source_org",
    "发布时间": "publish_date",
    "来源链接": "source_url",
    "适用地区": "region",
    "主题": "topic",
    "文件类型": "doc_type",
    "权威级别": "authority_level",
}

META_KEYS.update(
    {
        "Title": "title",
        "Source Org": "source_org",
        "Publish Date": "publish_date",
        "Source URL": "source_url",
        "Region": "region",
        "Topic": "topic",
        "Document Type": "doc_type",
        "Authority Level": "authority_level",
        "Case ID": "case_id",
        "Case Type": "case_type",
    }
)


def parse_metadata_from_text(text: str) -> Tuple[Dict[str, str], str]:
    """
    从 txt 或 meta.txt 中解析元数据。

    支持格式：
    标题：xxx
    发布机构：xxx
    发布时间：xxx
    来源链接：xxx
    适用地区：xxx
    主题：xxx
    文件类型：xxx
    权威级别：xxx

    正文：
    ......
    """
    metadata = {}
    body_lines = []
    in_body = False

    for line in text.splitlines():
        stripped = line.strip()

        if stripped in {"Content:", "Body:"}:
            in_body = True
            continue

        if stripped.startswith("正文：") or stripped == "正文:":
            in_body = True
            continue

        if not in_body:
            matched = False

            for cn_key, en_key in META_KEYS.items():
                prefix_cn = f"{cn_key}："
                prefix_en = f"{cn_key}:"

                if stripped.startswith(prefix_cn):
                    metadata[en_key] = stripped[len(prefix_cn):].strip()
                    matched = True
                    break

                if stripped.startswith(prefix_en):
                    metadata[en_key] = stripped[len(prefix_en):].strip()
                    matched = True
                    break

            if matched:
                continue

            if stripped == "":
                continue

            # 没遇到“正文：”但已经出现非元数据内容，则也作为正文处理
            body_lines.append(line)

        else:
            body_lines.append(line)

    body = "\n".join(body_lines).strip()

    return metadata, body


def load_txt_with_metadata(file_path: Path) -> List[Document]:
    """
    加载带有元数据头部的 txt 文件。
    """
    text = file_path.read_text(encoding="utf-8")
    metadata, body = parse_metadata_from_text(text)

    if not body:
        body = text

    metadata["source"] = file_path.name
    metadata["file_path"] = str(file_path)
    metadata["file_type"] = "txt"

    return [Document(page_content=body, metadata=metadata)]


def load_pdf_metadata(file_path: Path) -> Dict[str, str]:
    """
    为 PDF 查找同名 .meta.txt 文件并解析元数据。

    例如：
    xxx.pdf
    xxx.meta.txt
    """
    meta_path = file_path.with_suffix(".meta.txt")

    if not meta_path.exists():
        return {}

    text = meta_path.read_text(encoding="utf-8")
    metadata, _ = parse_metadata_from_text(text)

    return metadata


def resolve_knowledge_source(raw_docs_dir: Path, source_id: str = None) -> KnowledgeSource:
    if source_id:
        return get_knowledge_source(source_id)

    matched_source = get_knowledge_source_for_path(raw_docs_dir)
    if matched_source:
        return matched_source

    return get_knowledge_source("gov_policy")


def apply_knowledge_source_metadata(doc: Document, source: KnowledgeSource) -> None:
    doc.metadata.setdefault("knowledge_source_id", source.id)
    doc.metadata.setdefault("knowledge_source_name", source.name)
    doc.metadata.setdefault("knowledge_source_type", source.source_type)
    doc.metadata.setdefault("authority_level", source.authority_level)


def load_documents(raw_docs_dir: Path, source_id: str = None) -> List[Document]:
    """
    加载 raw_docs_dir 下的正式政策文档。

    支持：
    - .txt：直接解析文件头部元数据
    - .pdf：读取同名 .meta.txt 元数据
    - .docx：普通读取

    注意：
    - .meta.txt 只作为 PDF 元数据文件，不进入正文知识库
    """
    documents: List[Document] = []
    knowledge_source = resolve_knowledge_source(raw_docs_dir, source_id=source_id)

    if not raw_docs_dir.exists():
        raise FileNotFoundError(f"文档目录不存在: {raw_docs_dir}")

    for file_path in raw_docs_dir.rglob("*"):
        if file_path.is_dir():
            continue

        suffix = file_path.suffix.lower()

        # 跳过 PDF 的元数据文件
        if file_path.name.endswith(".meta.txt"):
            continue

        try:
            if suffix in {".txt", ".md"}:
                docs = load_txt_with_metadata(file_path)

            elif suffix == ".pdf":
                loader = PyPDFLoader(str(file_path))
                docs = loader.load()

                pdf_metadata = load_pdf_metadata(file_path)

                for doc in docs:
                    doc.metadata.update(pdf_metadata)
                    doc.metadata["source"] = file_path.name
                    doc.metadata["file_path"] = str(file_path)
                    doc.metadata["file_type"] = "pdf"

            elif suffix == ".docx":
                loader = Docx2txtLoader(str(file_path))
                docs = loader.load()

                for doc in docs:
                    doc.metadata["source"] = file_path.name
                    doc.metadata["file_path"] = str(file_path)
                    doc.metadata["file_type"] = "docx"

            else:
                print(f"[跳过] 暂不支持的文件类型: {file_path.name}")
                continue

            for doc in docs:
                doc.metadata.setdefault("title", file_path.stem)
                doc.metadata.setdefault("source_org", "")
                doc.metadata.setdefault("publish_date", "")
                doc.metadata.setdefault("source_url", "")
                doc.metadata.setdefault("region", "")
                doc.metadata.setdefault("topic", "")
                doc.metadata.setdefault("doc_type", "")
                doc.metadata.setdefault("authority_level", "")
                apply_knowledge_source_metadata(doc, knowledge_source)

            documents.extend(docs)
            print(f"[加载成功] {file_path.name}，片段数量: {len(docs)}")

        except Exception as e:
            print(f"[加载失败] {file_path.name}: {e}")

    return documents


def load_documents_from_enabled_sources() -> List[Document]:
    documents: List[Document] = []

    for source in get_enabled_knowledge_sources():
        if not source.path.exists():
            print(f"[skip knowledge source] {source.id}: directory does not exist: {source.path}")
            continue

        source_docs = load_documents(source.path, source_id=source.id)
        documents.extend(source_docs)
        print(f"[knowledge source loaded] {source.id}: {len(source_docs)} document parts")

    return documents


if __name__ == "__main__":
    from app.config import RAW_DOCS_DIR

    docs = load_documents(RAW_DOCS_DIR)

    print("\n========== 文档加载测试 ==========")
    print(f"共加载文档片段数: {len(docs)}")

    if docs:
        print("\n第一个文档片段内容预览:")
        print(docs[0].page_content[:500])

        print("\nmetadata:")
        for key, value in docs[0].metadata.items():
            print(f"{key}: {value}")
