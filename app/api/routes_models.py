from fastapi import APIRouter

from app.llm.provider import get_model_runtime_info
from app.rag.reranker import get_reranker_runtime_info
from app.schemas.api import ModelRuntimeInfoResponse


router = APIRouter(prefix="/api/models", tags=["models"])


@router.get("/runtime", response_model=ModelRuntimeInfoResponse)
def model_runtime() -> ModelRuntimeInfoResponse:
    runtime = get_model_runtime_info()
    runtime["reranker"] = get_reranker_runtime_info()
    return ModelRuntimeInfoResponse(**runtime)
