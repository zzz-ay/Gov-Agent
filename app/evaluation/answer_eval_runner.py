import time
from typing import List

from app.evaluation.metrics import safe_average
from app.graph.gov_graph import run_gov_graph
from app.schemas.api import AnswerEvalCaseResult, AnswerEvalRunResponse, EvalCase


UNCERTAINTY_MARKERS = [
    "当前检索资料未明确说明",
    "无法判断",
    "建议咨询",
    "以当地",
    "以官方",
    "可能",
]

UNSUPPORTED_RISK_MARKERS = [
    "一定",
    "必然",
    "绝对",
    "无需核验",
    "保证",
]


def count_markers(text: str, markers: List[str]) -> int:
    return sum(text.count(marker) for marker in markers)


def keyword_coverage(answer: str, expected_keywords: List[str]) -> tuple[float, List[str], List[str]]:
    if not expected_keywords:
        return 1.0, [], []

    covered = [keyword for keyword in expected_keywords if keyword and keyword in answer]
    missing = [keyword for keyword in expected_keywords if keyword and keyword not in answer]
    return len(covered) / len(expected_keywords), covered, missing


def run_answer_eval(cases: List[EvalCase], debug_mode: bool = False) -> AnswerEvalRunResponse:
    results: List[AnswerEvalCaseResult] = []

    for case in cases:
        started = time.perf_counter()
        graph_result = run_gov_graph(case.question, debug_mode=debug_mode)
        latency_ms = (time.perf_counter() - started) * 1000

        answer = graph_result.get("final_answer", "") or ""
        sources = graph_result.get("sources", []) or []
        coverage, covered, missing = keyword_coverage(answer, case.expected_keywords)
        uncertainty_count = count_markers(answer, UNCERTAINTY_MARKERS)
        unsupported_risk_count = count_markers(answer, UNSUPPORTED_RISK_MARKERS)

        passed = bool(answer.strip()) and coverage >= 0.6 and unsupported_risk_count == 0

        results.append(
            AnswerEvalCaseResult(
                question=case.question,
                answer_length=len(answer),
                source_count=len(sources),
                keyword_coverage=coverage,
                covered_keywords=covered,
                missing_keywords=missing,
                uncertainty_count=uncertainty_count,
                unsupported_risk_count=unsupported_risk_count,
                latency_ms=latency_ms,
                passed=passed,
            )
        )

    total = len(results)
    pass_rate = sum(1 for item in results if item.passed) / total if total else 0.0

    return AnswerEvalRunResponse(
        total=total,
        pass_rate=pass_rate,
        avg_keyword_coverage=safe_average([item.keyword_coverage for item in results]),
        avg_source_count=safe_average([float(item.source_count) for item in results]),
        avg_latency_ms=safe_average([item.latency_ms for item in results]),
        results=results,
    )
