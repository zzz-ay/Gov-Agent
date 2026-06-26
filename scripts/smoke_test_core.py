import json
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.evaluation.eval_runner import run_retrieval_eval
from app.llm.provider import get_model_runtime_info
from app.rag.hybrid_retriever import hybrid_search
from app.rag.reranker import get_reranker_runtime_info
from app.schemas.api import EvalCase
from app.storage.jsonl_store import (
    RUNTIME_DB_PATH,
    read_recent_chat_logs,
    read_recent_eval_runs,
    read_recent_feedback,
    save_chat_log,
    save_eval_run,
    save_feedback,
)
from app.tools.runtime_admin import get_runtime_admin_status


QUESTION = "\u9ad8\u6821\u6bd5\u4e1a\u751f\u6863\u6848\u53ef\u4ee5\u81ea\u5df1\u643a\u5e26\u5417"
GOLD_SOURCE = "hebei_archive_receive_transfer_guide_2025.txt"


def dump(name, payload):
    print(f"\n[{name}]")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def assert_true(condition, message):
    if not condition:
        raise AssertionError(message)


def main():
    model_runtime = get_model_runtime_info()
    reranker_runtime = get_reranker_runtime_info()
    dump(
        "runtime",
        {
            "provider": model_runtime.get("provider"),
            "chat_model": model_runtime.get("chat_model"),
            "reranker": reranker_runtime.get("provider"),
        },
    )

    hits = hybrid_search(QUESTION, top_k=3, mode="bm25")
    assert_true(len(hits) > 0, "bm25 retrieval returned no hits")
    dump(
        "retrieval",
        {
            "hit_count": len(hits),
            "top_source": hits[0]["doc"].metadata.get("source"),
            "rerank_provider": hits[0].get("rerank_provider"),
            "rerank_score": round(float(hits[0].get("rerank_score") or 0), 4),
        },
    )

    cases = [EvalCase(question=QUESTION, gold_source_file=GOLD_SOURCE)]
    eval_result = run_retrieval_eval(cases, top_k=3, mode="bm25")
    eval_payload = (
        eval_result.model_dump()
        if hasattr(eval_result, "model_dump")
        else eval_result.dict()
    )
    assert_true(eval_payload["hit_rate"] >= 1.0, "retrieval eval hit_rate is below 1.0")

    saved_eval = save_eval_run(
        {
            "eval_type": "retrieval",
            "request": {
                "top_k": 3,
                "mode": "bm25",
                "cases": [
                    case.model_dump() if hasattr(case, "model_dump") else case.dict()
                    for case in cases
                ],
            },
            "result": eval_payload,
        }
    )
    dump(
        "retrieval_eval",
        {
            "eval_id": saved_eval["eval_id"],
            "hit_rate": eval_payload["hit_rate"],
            "mrr": eval_payload["mrr"],
        },
    )

    saved_chat = save_chat_log(
        {
            "session_id": "smoke-test",
            "question": QUESTION,
            "answer": "smoke test answer",
            "source_count": 1,
            "sources": [{"source_file": GOLD_SOURCE}],
            "citations": [{"source_file": GOLD_SOURCE, "rank": 1}],
            "latency_ms": 1.0,
        }
    )
    saved_feedback = save_feedback(
        {
            "session_id": "smoke-test",
            "question": QUESTION,
            "answer": "smoke test answer",
            "rating": "up",
            "issue_type": "smoke_test",
            "comment": "core smoke test",
        }
    )

    assert_true(RUNTIME_DB_PATH.exists(), "runtime sqlite db was not created")
    assert_true(read_recent_chat_logs(1), "chat log read failed")
    assert_true(read_recent_feedback(1), "feedback read failed")
    assert_true(read_recent_eval_runs(1), "eval run read failed")
    dump(
        "storage",
        {
            "runtime_db": str(RUNTIME_DB_PATH),
            "chat_log_id": saved_chat["log_id"],
            "feedback_id": saved_feedback["feedback_id"],
        },
    )

    admin_status = get_runtime_admin_status(limit=2)
    assert_true(admin_status["runtime_db_exists"], "runtime admin cannot see db")
    dump("runtime_admin", admin_status["counts"])

    print("\nSMOKE TEST PASSED")


if __name__ == "__main__":
    main()
