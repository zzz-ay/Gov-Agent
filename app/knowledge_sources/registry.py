import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from app.config import DATA_DIR


@dataclass(frozen=True)
class KnowledgeSource:
    id: str
    name: str
    source_type: str
    description: str
    path: Path
    authority_level: str


KNOWLEDGE_SOURCES: Dict[str, KnowledgeSource] = {
    "gov_policy": KnowledgeSource(
        id="gov_policy",
        name="Government Policy Documents",
        source_type="policy",
        description="Existing government-service policy documents used by the original GOV-RAG project.",
        path=DATA_DIR / "raw_docs",
        authority_level="policy",
    ),
    "laws": KnowledgeSource(
        id="laws",
        name="National Laws And Regulations",
        source_type="law",
        description="Official law and regulation documents imported from public legal sources.",
        path=DATA_DIR / "knowledge_sources" / "laws",
        authority_level="law",
    ),
    "legal_cases": KnowledgeSource(
        id="legal_cases",
        name="Legal Case Retrieval Corpus",
        source_type="case",
        description="Public legal-case retrieval datasets such as CAIL or LeCaRD-style corpora.",
        path=DATA_DIR / "knowledge_sources" / "legal_cases",
        authority_level="case",
    ),
    "public_notices": KnowledgeSource(
        id="public_notices",
        name="Public Notices And Announcements",
        source_type="notice",
        description="Public notices, announcements, and disclosure documents from authoritative sources.",
        path=DATA_DIR / "knowledge_sources" / "public_notices",
        authority_level="notice",
    ),
}


def list_knowledge_sources() -> List[KnowledgeSource]:
    return list(KNOWLEDGE_SOURCES.values())


def get_knowledge_source(source_id: str) -> KnowledgeSource:
    if source_id not in KNOWLEDGE_SOURCES:
        known = ", ".join(sorted(KNOWLEDGE_SOURCES))
        raise ValueError(f"Unknown knowledge source '{source_id}'. Known sources: {known}")
    return KNOWLEDGE_SOURCES[source_id]


def get_enabled_knowledge_sources() -> List[KnowledgeSource]:
    raw_value = os.getenv("KNOWLEDGE_SOURCES", "gov_policy")
    source_ids = [item.strip() for item in raw_value.split(",") if item.strip()]
    return [get_knowledge_source(source_id) for source_id in source_ids]


def get_knowledge_source_for_path(path: Path) -> Optional[KnowledgeSource]:
    resolved_path = path.resolve()
    for source in list_knowledge_sources():
        try:
            resolved_path.relative_to(source.path.resolve())
            return source
        except ValueError:
            continue
    return None
