from fastapi import APIRouter

from app.schemas.api import DocumentStatusResponse, RebuildVectorstoreResponse
from app.tools.vectorstore_admin import get_policy_library_status, rebuild_vectorstore


router = APIRouter(prefix="/api/documents", tags=["documents"])


@router.get("/status", response_model=DocumentStatusResponse)
def document_status() -> DocumentStatusResponse:
    return DocumentStatusResponse(status=get_policy_library_status())


@router.post("/rebuild", response_model=RebuildVectorstoreResponse)
def rebuild_documents_vectorstore() -> RebuildVectorstoreResponse:
    result = rebuild_vectorstore()
    return RebuildVectorstoreResponse(**result)
