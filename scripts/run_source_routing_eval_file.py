import argparse
import json
import sys
from pathlib import Path
from typing import List


BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

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
                    expected_keywords=item.get("expected_keywords", []),
                    expected_source_ids=item.get("expected_source_ids", []),
                    expected_task_type=item.get("expected_task_type"),
                )
            )
            if limit and len(cases) >= limit:
                break

    return cases


def main():
    parser = argparse.ArgumentParser(description="Run source-routing evaluation from a JSONL file.")
    parser.add_argument(
        "--input",
        default="data/evaluation/source_routing_eval.jsonl",
        help="JSONL file with question/expected_source_ids records.",
    )
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    cases = load_eval_cases(input_path, limit=args.limit)
    result = run_source_routing_eval(cases)

    print("========== Source Routing Eval ==========")
    print(f"Input: {input_path}")
    print(f"Cases: {result.total}")
    print(f"Accuracy: {result.accuracy:.4f}")
    print(f"Source accuracy: {result.source_accuracy:.4f}")
    print(f"Task accuracy: {result.task_accuracy:.4f}")
    print(f"Avg latency ms: {result.avg_latency_ms:.2f}")
    print("")

    for item in result.results:
        status = "PASS" if item.matched else "FAIL"
        print(f"[{status}] {item.question}")
        print(f"  expected: {item.expected_source_ids}")
        print(f"  selected: {item.selected_source_ids}")
        print(f"  expected task: {item.expected_task_type or '-'}")
        print(f"  selected task: {item.selected_task_type or '-'}")
        print(f"  source match: {item.source_match}")
        print(f"  task match: {item.task_match}")
        if item.routing_reasons:
            print(f"  reasons: {'; '.join(item.routing_reasons)}")


if __name__ == "__main__":
    main()
