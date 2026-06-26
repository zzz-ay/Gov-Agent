from fastapi import APIRouter

from app.agents.business_flow import classify_business_flow, list_business_flows
from app.knowledge_sources import get_enabled_knowledge_sources
from app.schemas.api import (
    BusinessFlowClassifyRequest,
    BusinessFlowClassifyResponse,
    BusinessFlowListResponse,
)


router = APIRouter(prefix="/api/business", tags=["business"])


@router.get("/flows", response_model=BusinessFlowListResponse)
def business_flows() -> BusinessFlowListResponse:
    return BusinessFlowListResponse(flows=list_business_flows())


@router.post("/classify", response_model=BusinessFlowClassifyResponse)
def classify_flow(request: BusinessFlowClassifyRequest) -> BusinessFlowClassifyResponse:
    enabled_source_ids = [source.id for source in get_enabled_knowledge_sources()]
    return BusinessFlowClassifyResponse(
        business_flow=classify_business_flow(
            question=request.question,
            enabled_source_ids=enabled_source_ids,
        )
    )
