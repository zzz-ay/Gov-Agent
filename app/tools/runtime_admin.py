from typing import Any, Dict

from app.llm.provider import get_model_runtime_info
from app.storage.jsonl_store import (
    RUNTIME_DB_PATH,
    read_recent_chat_logs,
    read_recent_eval_runs,
    read_recent_feedback,
)


def get_runtime_admin_status(limit: int = 20) -> Dict[str, Any]:
    chat_logs = read_recent_chat_logs(limit=limit)
    feedback = read_recent_feedback(limit=limit)
    eval_runs = read_recent_eval_runs(limit=limit)

    return {
        "runtime_db_path": str(RUNTIME_DB_PATH),
        "runtime_db_exists": RUNTIME_DB_PATH.exists(),
        "model_runtime": get_model_runtime_info(),
        "chat_logs": chat_logs,
        "feedback": feedback,
        "eval_runs": eval_runs,
        "counts": {
            "chat_logs": len(chat_logs),
            "feedback": len(feedback),
            "eval_runs": len(eval_runs),
        },
    }
