from fastapi import APIRouter, Query

from app.schemas.api import ChatLogListResponse, EvalRunListResponse, FeedbackLogListResponse
from app.storage.jsonl_store import (
    read_recent_chat_logs,
    read_recent_eval_runs,
    read_recent_feedback,
)


router = APIRouter(prefix="/api/logs", tags=["logs"])


@router.get("/chat", response_model=ChatLogListResponse)
def chat_logs(limit: int = Query(default=50, ge=1, le=500)) -> ChatLogListResponse:
    return ChatLogListResponse(logs=read_recent_chat_logs(limit=limit))


@router.get("/feedback", response_model=FeedbackLogListResponse)
def feedback_logs(limit: int = Query(default=50, ge=1, le=500)) -> FeedbackLogListResponse:
    return FeedbackLogListResponse(feedback=read_recent_feedback(limit=limit))


@router.get("/eval-runs", response_model=EvalRunListResponse)
def eval_run_logs(limit: int = Query(default=50, ge=1, le=500)) -> EvalRunListResponse:
    return EvalRunListResponse(eval_runs=read_recent_eval_runs(limit=limit))
