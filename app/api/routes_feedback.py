from fastapi import APIRouter

from app.schemas.api import FeedbackRequest, FeedbackResponse
from app.storage.jsonl_store import save_feedback


router = APIRouter(prefix="/api/feedback", tags=["feedback"])


@router.post("", response_model=FeedbackResponse)
def feedback(request: FeedbackRequest) -> FeedbackResponse:
    if hasattr(request, "model_dump"):
        payload = request.model_dump()
    else:
        payload = request.dict()

    saved = save_feedback(payload)
    return FeedbackResponse(
        success=True,
        feedback_id=saved["feedback_id"],
        saved_path=saved["saved_path"],
    )
