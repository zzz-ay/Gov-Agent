import argparse
import json
import sys
from pathlib import Path
from typing import List


BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.evaluation.eval_runner import run_retrieval_eval
from app.evaluation.source_routing_eval_runner import run_source_routing_eval
from app.schemas.api import EvalCase


def load_eval_cases(path: Path, limit: int = 0) -> List[EvalCase]:
    cases: List[EvalCase] = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            cases.append(
                EvalCase(
                    question=item.get("question", ""),
                    gold_source_file=item.get("gold_source_file"),
                    expected_keywords=item.get("expected_keywords", []),
                    expected_source_ids=item.get("expected_source_ids", []),
                    expected_task_type=item.get("expected_task_type"),
                )
            )
            if limit and len(cases) >= limit:
                break

    return cases


def main():
    parser = argparse.ArgumentParser(description="Run platform-level evaluation checks.")
    parser.add_argument(
        "--routing-input",
        default="data/evaluation/source_routing_eval.jsonl",
        help="JSONL file for source-router evaluation.",
    )
    parser.add_argument(
        "--retrieval-input",
        default="",
        help="Optional JSONL file for retrieval evaluation.",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--mode", default="hybrid")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    routing_path = Path(args.routing_input).resolve()
    routing_cases = load_eval_cases(routing_path, limit=args.limit)
    routing_result = run_source_routing_eval(routing_cases)

    print("========== Platform Eval ==========")
    print(f"Routing input: {routing_path}")
    print(f"Routing cases: {routing_result.total}")
    print(f"Routing accuracy: {routing_result.accuracy:.4f}")
    print(f"Source accuracy: {routing_result.source_accuracy:.4f}")
    print(f"Task accuracy: {routing_result.task_accuracy:.4f}")
    print(f"Routing avg latency ms: {routing_result.avg_latency_ms:.2f}")

    if not args.retrieval_input:
        print("")
        print("Retrieval eval: skipped")
        return

    retrieval_path = Path(args.retrieval_input).resolve()
    retrieval_cases = load_eval_cases(retrieval_path, limit=args.limit)
    retrieval_result = run_retrieval_eval(
        cases=retrieval_cases,
        top_k=args.top_k,
        mode=args.mode,
        source_ids=[],
    )

    print("")
    print(f"Retrieval input: {retrieval_path}")
    print(f"Retrieval cases: {retrieval_result.total}")
    print(f"Retrieval mode: {args.mode}")
    print(f"Retrieval hit rate: {retrieval_result.hit_rate:.4f}")
    print(f"Retrieval MRR: {retrieval_result.mrr:.4f}")
    print(f"Retrieval avg latency ms: {retrieval_result.avg_latency_ms:.2f}")


if __name__ == "__main__":
    main()
