import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.config import BASE_DIR


STORAGE_DIR = BASE_DIR / "data" / "runtime"
FEEDBACK_PATH = STORAGE_DIR / "feedback.jsonl"
CHAT_LOG_PATH = STORAGE_DIR / "chat_logs.jsonl"
RUNTIME_DB_PATH = STORAGE_DIR / "gov_rag_runtime.db"
EVAL_RUNS_PATH = STORAGE_DIR / "eval_runs.jsonl"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def init_runtime_db() -> None:
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(RUNTIME_DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_logs (
                id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                session_id TEXT,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                source_count INTEGER DEFAULT 0,
                sources_json TEXT DEFAULT '[]',
                citations_json TEXT DEFAULT '[]',
                risk_level TEXT,
                verification_score INTEGER,
                debug_mode INTEGER DEFAULT 0,
                latency_ms REAL,
                metadata_json TEXT DEFAULT '{}'
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS feedback (
                id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                session_id TEXT,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                rating TEXT NOT NULL,
                issue_type TEXT,
                comment TEXT,
                metadata_json TEXT DEFAULT '{}'
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS eval_runs (
                id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                eval_type TEXT NOT NULL,
                mode TEXT,
                top_k INTEGER,
                total INTEGER DEFAULT 0,
                hit_rate REAL DEFAULT 0,
                mrr REAL DEFAULT 0,
                avg_latency_ms REAL DEFAULT 0,
                request_json TEXT DEFAULT '{}',
                result_json TEXT DEFAULT '{}'
            )
            """
        )
        existing_columns = {
            row[1]
            for row in conn.execute("PRAGMA table_info(chat_logs)").fetchall()
        }
        if "citations_json" not in existing_columns:
            conn.execute(
                "ALTER TABLE chat_logs ADD COLUMN citations_json TEXT DEFAULT '[]'"
            )
        conn.commit()


def append_jsonl(path: Path, payload: Dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    item_id = payload.get("id") or str(uuid.uuid4())
    record = {
        "id": item_id,
        "created_at": utc_now(),
        **payload,
    }

    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return item_id


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def _json_loads(value: Optional[str], default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


def insert_chat_log(payload: Dict[str, Any], log_id: str, created_at: str) -> None:
    init_runtime_db()

    with sqlite3.connect(RUNTIME_DB_PATH) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO chat_logs (
                id,
                created_at,
                session_id,
                question,
                answer,
                source_count,
                sources_json,
                citations_json,
                risk_level,
                verification_score,
                debug_mode,
                latency_ms,
                metadata_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                log_id,
                created_at,
                payload.get("session_id"),
                payload.get("question", ""),
                payload.get("answer", ""),
                int(payload.get("source_count") or 0),
                _json_dumps(payload.get("sources", [])),
                _json_dumps(payload.get("citations", [])),
                payload.get("risk_level"),
                payload.get("verification_score"),
                1 if payload.get("debug_mode") else 0,
                payload.get("latency_ms"),
                _json_dumps(payload.get("metadata", {})),
            ),
        )
        conn.commit()


def insert_feedback(payload: Dict[str, Any], feedback_id: str, created_at: str) -> None:
    init_runtime_db()

    with sqlite3.connect(RUNTIME_DB_PATH) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO feedback (
                id,
                created_at,
                session_id,
                question,
                answer,
                rating,
                issue_type,
                comment,
                metadata_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                feedback_id,
                created_at,
                payload.get("session_id"),
                payload.get("question", ""),
                payload.get("answer", ""),
                payload.get("rating", ""),
                payload.get("issue_type"),
                payload.get("comment"),
                _json_dumps(payload.get("metadata", {})),
            ),
        )
        conn.commit()


def insert_eval_run(payload: Dict[str, Any], eval_id: str, created_at: str) -> None:
    init_runtime_db()

    result = payload.get("result", {})
    request = payload.get("request", {})
    eval_type = payload.get("eval_type", "retrieval")
    primary_rate = result.get("hit_rate", result.get("pass_rate", 0.0))

    with sqlite3.connect(RUNTIME_DB_PATH) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO eval_runs (
                id,
                created_at,
                eval_type,
                mode,
                top_k,
                total,
                hit_rate,
                mrr,
                avg_latency_ms,
                request_json,
                result_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                eval_id,
                created_at,
                eval_type,
                request.get("mode"),
                request.get("top_k"),
                result.get("total", 0),
                primary_rate,
                result.get("mrr", 0.0),
                result.get("avg_latency_ms", 0.0),
                _json_dumps(request),
                _json_dumps(result),
            ),
        )
        conn.commit()


def save_feedback(payload: Dict[str, Any]) -> Dict[str, str]:
    feedback_id = payload.get("id") or str(uuid.uuid4())
    created_at = utc_now()
    record = {"id": feedback_id, "created_at": created_at, **payload}
    append_jsonl(FEEDBACK_PATH, record)
    insert_feedback(payload, feedback_id=feedback_id, created_at=created_at)

    return {
        "feedback_id": feedback_id,
        "saved_path": str(RUNTIME_DB_PATH),
    }


def save_chat_log(payload: Dict[str, Any]) -> Dict[str, str]:
    log_id = payload.get("id") or str(uuid.uuid4())
    created_at = utc_now()
    record = {"id": log_id, "created_at": created_at, **payload}
    append_jsonl(CHAT_LOG_PATH, record)
    insert_chat_log(payload, log_id=log_id, created_at=created_at)

    return {
        "log_id": log_id,
        "saved_path": str(RUNTIME_DB_PATH),
    }


def save_eval_run(payload: Dict[str, Any]) -> Dict[str, str]:
    eval_id = payload.get("id") or str(uuid.uuid4())
    created_at = utc_now()
    record = {"id": eval_id, "created_at": created_at, **payload}
    append_jsonl(EVAL_RUNS_PATH, record)
    insert_eval_run(payload, eval_id=eval_id, created_at=created_at)

    return {
        "eval_id": eval_id,
        "saved_path": str(RUNTIME_DB_PATH),
    }


def read_recent_jsonl(path: Path, limit: int = 50) -> List[Dict[str, Any]]:
    if not path.exists():
        return []

    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if limit <= 0:
        return records

    return records[-limit:]


def read_recent_chat_logs(limit: int = 50) -> List[Dict[str, Any]]:
    init_runtime_db()

    with sqlite3.connect(RUNTIME_DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT *
            FROM chat_logs
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    results = []
    for row in rows:
        results.append(
            {
                "id": row["id"],
                "created_at": row["created_at"],
                "session_id": row["session_id"],
                "question": row["question"],
                "answer": row["answer"],
                "source_count": row["source_count"],
                "sources": _json_loads(row["sources_json"], []),
                "citations": _json_loads(row["citations_json"], []),
                "risk_level": row["risk_level"],
                "verification_score": row["verification_score"],
                "debug_mode": bool(row["debug_mode"]),
                "latency_ms": row["latency_ms"],
                "metadata": _json_loads(row["metadata_json"], {}),
            }
        )

    return results


def read_recent_feedback(limit: int = 50) -> List[Dict[str, Any]]:
    init_runtime_db()

    with sqlite3.connect(RUNTIME_DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT *
            FROM feedback
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    results = []
    for row in rows:
        results.append(
            {
                "id": row["id"],
                "created_at": row["created_at"],
                "session_id": row["session_id"],
                "question": row["question"],
                "answer": row["answer"],
                "rating": row["rating"],
                "issue_type": row["issue_type"],
                "comment": row["comment"],
                "metadata": _json_loads(row["metadata_json"], {}),
            }
        )

    return results


def read_recent_eval_runs(limit: int = 50) -> List[Dict[str, Any]]:
    init_runtime_db()

    with sqlite3.connect(RUNTIME_DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT *
            FROM eval_runs
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    results = []
    for row in rows:
        results.append(
            {
                "id": row["id"],
                "created_at": row["created_at"],
                "eval_type": row["eval_type"],
                "mode": row["mode"],
                "top_k": row["top_k"],
                "total": row["total"],
                "hit_rate": row["hit_rate"],
                "mrr": row["mrr"],
                "avg_latency_ms": row["avg_latency_ms"],
                "request": _json_loads(row["request_json"], {}),
                "result": _json_loads(row["result_json"], {}),
            }
        )

    return results
