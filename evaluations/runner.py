"""Run a reviewed, fixed evaluation set and record reproducible model metrics."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Iterable, List

from inbox_agent import agent as inbox
from knowledge_brain.answerer import KnowledgeAnswerer
from knowledge_brain.retriever import KnowledgeRetriever
from meeting_to_sow import agent as meetings

TASKS = {"knowledge_brain", "inbox_agent", "meeting_to_sow"}


def load_cases(path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Evaluation cases must be a JSON list")
    return [item for item in payload if isinstance(item, dict)]


def validate_cases(cases: Iterable[Dict[str, Any]], minimum_per_task: int = 20) -> None:
    counts = Counter(case.get("task") for case in cases)
    invalid = sorted(set(counts).difference(TASKS))
    if invalid:
        raise ValueError(f"Unknown task values: {', '.join(invalid)}")
    missing = [task for task in sorted(TASKS) if counts[task] < minimum_per_task]
    if missing:
        raise ValueError(f"At least {minimum_per_task} reviewed cases are required for: {', '.join(missing)}")


def estimate_cost(input_tokens: int | None, output_tokens: int | None, input_cost_per_million: float, output_cost_per_million: float) -> float | None:
    if input_tokens is None or output_tokens is None:
        return None
    return round((input_tokens * input_cost_per_million + output_tokens * output_cost_per_million) / 1_000_000, 8)


def _quality_checks(case: Dict[str, Any], answer: Dict[str, Any]) -> Dict[str, bool]:
    checks: Dict[str, bool] = {}
    if "expected_evidence_found" in case:
        checks["evidence_found"] = answer.get("evidence_found") == case["expected_evidence_found"]
    expected_terms = case.get("required_answer_terms", [])
    if expected_terms:
        answer_text = str(answer.get("answer", "")).lower()
        checks["required_answer_terms"] = all(str(term).lower() in answer_text for term in expected_terms)
    expected_urls = case.get("required_citation_urls", [])
    if expected_urls:
        actual_urls = {citation.get("url") for citation in answer.get("citations", []) if isinstance(citation, dict)}
        checks["required_citation_urls"] = all(url in actual_urls for url in expected_urls)
    return checks


def _answer(answerer: KnowledgeAnswerer, retriever: KnowledgeRetriever, query: str) -> tuple[Dict[str, Any], float]:
    started = perf_counter()
    result = answerer.ask(query, retriever=retriever)
    return result.to_dict(), round((perf_counter() - started) * 1000, 2)


def run_cases(
    cases: Iterable[Dict[str, Any]],
    answerer: KnowledgeAnswerer,
    retriever: KnowledgeRetriever,
    input_cost_per_million: float,
    output_cost_per_million: float,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for case in cases:
        task = case["task"]
        if task == "knowledge_brain":
            answer, latency_ms = _answer(answerer, retriever, str(case["query"]))
            quality = _quality_checks(case, answer)
            records.append({"case_id": case["case_id"], "task": task, "model": answer["model"], "mode": answer["mode"], "latency_ms": latency_ms, "input_tokens": answer["input_tokens"], "output_tokens": answer["output_tokens"], "estimated_cost": estimate_cost(answer["input_tokens"], answer["output_tokens"], input_cost_per_million, output_cost_per_million), "quality_checks": quality, "automatic_quality_pass": all(quality.values()) if quality else None, "failure": None})
        elif task == "inbox_agent":
            email = case["email"]
            answer, latency_ms = _answer(answerer, retriever, str(email.get("subject", "")) + "\n" + str(email.get("body", "")))
            draft = inbox.prepare_draft(email, answer)
            records.append({"case_id": case["case_id"], "task": task, "model": answer["model"], "mode": answer["mode"], "latency_ms": latency_ms, "input_tokens": answer["input_tokens"], "output_tokens": answer["output_tokens"], "estimated_cost": estimate_cost(answer["input_tokens"], answer["output_tokens"], input_cost_per_million, output_cost_per_million), "quality_checks": {"review_required": draft["review_required"], "expected_category": draft["category"] == case.get("expected_category", draft["category"])}, "automatic_quality_pass": None, "failure": None})
        else:
            original_ask = meetings.ask_knowledge_brain
            calls: List[Dict[str, Any]] = []
            try:
                def evaluated_ask(query: str) -> Dict[str, Any]:
                    answer, latency_ms = _answer(answerer, retriever, query)
                    answer["latency_ms"] = latency_ms
                    calls.append(answer)
                    return answer
                meetings.ask_knowledge_brain = evaluated_ask
                scope = meetings.create_scope(case["meeting"])
            finally:
                meetings.ask_knowledge_brain = original_ask
            input_tokens = sum(item["input_tokens"] or 0 for item in calls) if calls and all(item["input_tokens"] is not None for item in calls) else None
            output_tokens = sum(item["output_tokens"] or 0 for item in calls) if calls and all(item["output_tokens"] is not None for item in calls) else None
            latency_ms = round(sum(item["latency_ms"] for item in calls), 2)
            expected_features = set(case.get("required_features", []))
            actual_features = {feature["feature"] for feature in scope["features"]}
            quality = {"required_features": expected_features.issubset(actual_features), "pending_human_approval": scope["approval_status"] == "pending_human_approval", "client_send_blocked": scope["client_send_allowed"] is False}
            records.append({"case_id": case["case_id"], "task": task, "model": answerer.model, "mode": answerer.mode, "latency_ms": latency_ms, "model_calls": len(calls), "input_tokens": input_tokens, "output_tokens": output_tokens, "estimated_cost": estimate_cost(input_tokens, output_tokens, input_cost_per_million, output_cost_per_million), "quality_checks": quality, "automatic_quality_pass": all(quality.values()), "failure": None})
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description="Run fixed, reviewed evaluation cases for one application model.")
    parser.add_argument("--cases", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--data", type=Path, default=Path("knowledge_brain/data/full_articles.json"))
    parser.add_argument("--provider", choices=["claude", "gemini"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--input-cost-per-million", required=True, type=float)
    parser.add_argument("--output-cost-per-million", required=True, type=float)
    args = parser.parse_args()
    cases = load_cases(args.cases)
    validate_cases(cases)
    answerer = KnowledgeAnswerer(use_mock=False, provider=args.provider, model=args.model)
    records = run_cases(cases, answerer, KnowledgeRetriever(articles_path=args.data), args.input_cost_per_million, args.output_cost_per_million)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
