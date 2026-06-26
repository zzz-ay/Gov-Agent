from fastapi import APIRouter

from app.evaluation.answer_eval_runner import run_answer_eval
from app.evaluation.eval_runner import run_retrieval_eval
from app.evaluation.source_routing_eval_runner import run_source_routing_eval
from app.schemas.api import (
    AnswerEvalRunRequest,
    AnswerEvalRunResponse,
    EvalRunRequest,
    EvalRunResponse,
    SourceRoutingEvalRunRequest,
    SourceRoutingEvalRunResponse,
)
from app.storage.jsonl_store import save_eval_run


router = APIRouter(prefix="/api/evaluation", tags=["evaluation"])


@router.post("/retrieval", response_model=EvalRunResponse)
def evaluate_retrieval(request: EvalRunRequest) -> EvalRunResponse:
    result = run_retrieval_eval(
        cases=request.cases,
        top_k=request.top_k,
        mode=request.mode,
        source_ids=request.source_ids,
    )

    if hasattr(request, "model_dump"):
        request_payload = request.model_dump()
    else:
        request_payload = request.dict()

    if hasattr(result, "model_dump"):
        result_payload = result.model_dump()
    else:
        result_payload = result.dict()

    saved = save_eval_run(
        {
            "eval_type": "retrieval",
            "request": request_payload,
            "result": result_payload,
        }
    )

    result.eval_id = saved["eval_id"]
    return result


@router.post("/source-routing", response_model=SourceRoutingEvalRunResponse)
def evaluate_source_routing(request: SourceRoutingEvalRunRequest) -> SourceRoutingEvalRunResponse:
    result = run_source_routing_eval(cases=request.cases)

    if hasattr(request, "model_dump"):
        request_payload = request.model_dump()
    else:
        request_payload = request.dict()

    if hasattr(result, "model_dump"):
        result_payload = result.model_dump()
    else:
        result_payload = result.dict()

    saved = save_eval_run(
        {
            "eval_type": "source_routing",
            "request": request_payload,
            "result": result_payload,
        }
    )

    result.eval_id = saved["eval_id"]
    return result


@router.post("/answer", response_model=AnswerEvalRunResponse)
def evaluate_answer(request: AnswerEvalRunRequest) -> AnswerEvalRunResponse:
    result = run_answer_eval(
        cases=request.cases,
        debug_mode=request.debug_mode,
    )

    if hasattr(request, "model_dump"):
        request_payload = request.model_dump()
    else:
        request_payload = request.dict()

    if hasattr(result, "model_dump"):
        result_payload = result.model_dump()
    else:
        result_payload = result.dict()

    saved = save_eval_run(
        {
            "eval_type": "answer",
            "request": request_payload,
            "result": result_payload,
        }
    )

    result.eval_id = saved["eval_id"]
    return result
