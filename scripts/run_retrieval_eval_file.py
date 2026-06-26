import argparse
import json
import sys
from pathlib import Path
from typing import List


BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.evaluation.eval_runner import run_retrieval_eval
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
                )
            )
            if limit and len(cases) >= limit:
                break

    return cases


def main():
    parser = argparse.ArgumentParser(description="Run retrieval evaluation from a JSONL eval file.")
    parser.add_argument("--input", required=True, help="JSONL file with question/gold_source_file records.")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--modes", default="vector,bm25,hybrid")
    parser.add_argument("--source-ids", default="")
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    cases = load_eval_cases(input_path, limit=args.limit)
    modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    source_ids = [item.strip() for item in args.source_ids.split(",") if item.strip()]

    print("========== Retrieval Eval ==========")
    print(f"Input: {input_path}")
    print(f"Cases: {len(cases)}")
    print(f"Top K: {args.top_k}")
    print(f"Source IDs: {source_ids or 'all'}")
    print("")
    print(f"{'mode':<10} {'hit_rate':>10} {'mrr':>10} {'avg_latency_ms':>16}")

    for mode in modes:
        result = run_retrieval_eval(
            cases=cases,
            top_k=args.top_k,
            mode=mode,
            source_ids=source_ids,
        )
        print(
            f"{mode:<10} "
            f"{result.hit_rate:>10.4f} "
            f"{result.mrr:>10.4f} "
            f"{result.avg_latency_ms:>16.2f}"
        )


if __name__ == "__main__":
    main()
