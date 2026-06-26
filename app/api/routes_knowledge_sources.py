from fastapi import APIRouter

from app.knowledge_sources import get_enabled_knowledge_sources, list_knowledge_sources
from app.schemas.api import KnowledgeSourceInfo, KnowledgeSourceListResponse


router = APIRouter(prefix="/api/knowledge-sources", tags=["knowledge-sources"])


@router.get("", response_model=KnowledgeSourceListResponse)
def knowledge_sources() -> KnowledgeSourceListResponse:
    enabled_ids = {source.id for source in get_enabled_knowledge_sources()}

    return KnowledgeSourceListResponse(
        sources=[
            KnowledgeSourceInfo(
                id=source.id,
                name=source.name,
                source_type=source.source_type,
                description=source.description,
                path=str(source.path),
                authority_level=source.authority_level,
                enabled=source.id in enabled_ids,
                exists=source.path.exists(),
            )
            for source in list_knowledge_sources()
        ]
    )
