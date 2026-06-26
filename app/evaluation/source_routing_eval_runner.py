import time
from typing import List

from app.agents.source_router_agent import source_router_agent
from app.evaluation.metrics import safe_average
from app.schemas.api import (
    EvalCase,
    SourceRoutingEvalCaseResult,
    SourceRoutingEvalRunResponse,
)


def normalize_source_ids(source_ids: List[str]) -> List[str]:
    return sorted(source_id for source_id in source_ids if source_id)


def run_source_routing_eval(cases: List[EvalCase]) -> SourceRoutingEvalRunResponse:
    results: List[SourceRoutingEvalCaseResult] = []

    for case in cases:
        started = time.perf_counter()
        route = source_router_agent(question=case.question)
        latency_ms = (time.perf_counter() - started) * 1000

        expected = normalize_source_ids(case.expected_source_ids)
        selected = normalize_source_ids(route.get("source_ids", []))
        selected_task_type = route.get("task_type", "")
        source_match = bool(expected) and selected == expected
        task_match = (
            not case.expected_task_type
            or selected_task_type == case.expected_task_type
        )
        matched = source_match and task_match

        results.append(
            SourceRoutingEvalCaseResult(
                question=case.question,
                expected_source_ids=expected,
                selected_source_ids=selected,
                expected_task_type=case.expected_task_type,
                selected_task_type=selected_task_type,
                source_match=source_match,
                task_match=task_match,
                matched=matched,
                latency_ms=latency_ms,
                routing_reasons=route.get("routing_reasons", []),
            )
        )

    total = len(results)
    accuracy = sum(1 for item in results if item.matched) / total if total else 0.0
    source_accuracy = sum(1 for item in results if item.source_match) / total if total else 0.0
    task_accuracy = sum(1 for item in results if item.task_match) / total if total else 0.0

    return SourceRoutingEvalRunResponse(
        total=total,
        accuracy=accuracy,
        source_accuracy=source_accuracy,
        task_accuracy=task_accuracy,
        avg_latency_ms=safe_average([item.latency_ms for item in results]),
        results=results,
    )
