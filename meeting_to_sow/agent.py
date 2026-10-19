"""Meeting-to-SOW workflow with explicit assumptions and human approval gates."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import requests

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
TRANSCRIPTS_PATH = DATA_DIR / "meeting_transcripts.json"
KB_URL = os.getenv("KNOWLEDGE_BRAIN_URL", "http://127.0.0.1:8501/ask")

FEATURE_CATALOG = {
    "member import": ("member import", "member data import"),
    "csv import": ("csv import", "spreadsheet import", "upload a csv"),
    "email notifications": ("email notifications", "email updates"),
    "sms notifications": ("sms notification", "sms notifications"),
    "agent profiles": ("agent profiles",),
    "property listings": ("property listings",),
    "search page": ("search page",),
    "contact form": ("contact form",),
    "instructor profiles": ("instructor profiles",),
    "custom integration": ("custom integration", "custom crm integration"),
    "salesforce": ("salesforce",),
    "crm connection": ("crm connection", "custom connection to our existing crm"),
    "custom dashboard": ("custom dashboard",),
    "admin dashboard": ("admin dashboard",),
}
CUSTOM_FEATURES = {"custom integration", "salesforce", "crm connection", "custom dashboard"}


def load_meetings() -> list[dict[str, Any]]:
    return json.loads(TRANSCRIPTS_PATH.read_text(encoding="utf-8"))


def ask_knowledge_brain(question: str) -> dict[str, Any]:
    try:
        response = requests.post(KB_URL, json={"query": question}, timeout=8)
        data = response.json()
        if isinstance(data, dict):
            if response.status_code >= 400:
                return {"answer": "", "evidence_found": False, "citations": [], "evidence_status": data.get("evidence_status", "provider_unavailable"), "is_mock": bool(data.get("is_mock", False))}
            return data
    except (requests.RequestException, ValueError):
        pass
    return {"answer": "", "evidence_found": False, "citations": [], "evidence_status": "provider_unavailable", "is_mock": False}


def extract_features(transcript: str) -> list[dict[str, str]]:
    text = transcript.lower()
    found = []
    for feature, phrases in FEATURE_CATALOG.items():
        if any(re.search(rf"\b{re.escape(phrase.lower())}\b", text) for phrase in phrases):
            classification = "candidate_custom_or_needs_review" if feature in CUSTOM_FEATURES else "candidate_product_configuration"
            found.append({"feature": feature, "initial_classification": classification})
    return found


def _comparison_key(feature: str) -> str:
    """Normalize alternate import phrasing for scope comparison only."""
    return "data import" if feature in {"member import", "csv import"} else feature


def estimate_tasks(features: list[dict[str, str]]) -> list[dict[str, Any]]:
    tasks = [{"task": "requirements and acceptance criteria", "hours_min": 3, "hours_max": 5, "assumption": "Planning estimate; validate in discovery."}]

    def add_task(task: dict[str, Any]) -> None:
        if not any(existing["task"] == task["task"] for existing in tasks):
            tasks.append(task)

    for item in features:
        feature = item["feature"]
        if feature in {"member import", "csv import"}:
            add_task({"task": "data import and validation", "hours_min": 4, "hours_max": 8, "assumption": "Assumes clean source data and agreed field mapping."})
        elif feature in CUSTOM_FEATURES:
            for task in [
                {"task": "custom integration discovery", "hours_min": 4, "hours_max": 8, "assumption": "API access and documentation must be confirmed."},
                {"task": "custom integration implementation", "hours_min": 12, "hours_max": 24, "assumption": "Depends on API, authentication, and data mapping."},
            ]:
                add_task(task)
        elif "notification" in feature:
            for task in [
                {"task": "notifications discovery", "hours_min": 2, "hours_max": 4, "assumption": "Confirm channels, triggers, consent, and templates."},
                {"task": "notification implementation", "hours_min": 4, "hours_max": 10, "assumption": "Provider configuration and channel support require validation."},
            ]:
                add_task(task)
        else:
            add_task({"task": "configuration and implementation", "hours_min": 8, "hours_max": 16, "assumption": "Depends on exact requirements and platform capability."})
    tasks.extend([
        {"task": "QA and user acceptance testing", "hours_min": 5, "hours_max": 8, "assumption": "Includes agreed acceptance criteria and one review cycle."},
        {"task": "deployment and handover", "hours_min": 3, "hours_max": 5, "assumption": "Assumes timely client access and review."},
    ])
    return tasks


def _feature_has_specific_evidence(feature: str, result: dict[str, Any]) -> bool:
    feature_terms = set(re.findall(r"[a-z0-9]+", feature.lower()))
    evidence_text = str(result.get("answer", "")) + " " + " ".join(
        f"{citation.get('title', '')} {citation.get('url', '')}"
        for citation in result.get("citations", [])
        if isinstance(citation, dict)
    )
    evidence_terms = set(re.findall(r"[a-z0-9]+", evidence_text.lower()))
    return feature_terms.issubset(evidence_terms)


def create_scope(meeting: dict[str, Any]) -> dict[str, Any]:
    transcript = meeting.get("transcript", "")
    features = extract_features(transcript)
    checks = []
    for item in features:
        result = ask_knowledge_brain(f"Is this supported by standard Brilliant Directories product settings or does it require custom work? Feature: {item['feature']}")
        evidence = bool(result.get("evidence_found") and result.get("answer") and result.get("citations")) and _feature_has_specific_evidence(item["feature"], result)
        provider_unavailable = result.get("evidence_status") == "provider_unavailable"
        checks.append({
            "feature": item["feature"], "initial_classification": item["initial_classification"],
            "feasibility": "evidence_found_needs_human_interpretation" if evidence else ("provider_unavailable_needs_human_review" if provider_unavailable else "unknown_needs_human_review"),
            "knowledge_answer": result.get("answer", "") if evidence else "",
            "citations": result.get("citations", []) if evidence else [],
            "evidence_status": result.get("evidence_status", "insufficient_evidence"),
            "is_mock": bool(result.get("is_mock", False)),
        })
    tasks = estimate_tasks(features)
    return {
        "meeting_id": meeting.get("meeting_id"), "client": meeting.get("client"), "scope_version": 1, "source_transcript": transcript,
        "objectives": [f"Validate and deliver the requested {item['feature']}." for item in checks] or ["Run requirements discovery before defining deliverables."],
        "features": checks, "deliverables": [item["feature"] for item in checks], "tasks": tasks,
        "exclusions": ["Fixed pricing, delivery dates, and technical commitments are not approved in this draft.", "Third-party services and custom work remain subject to discovery and written approval."],
        "client_responsibilities": ["Confirm requirements and acceptance criteria.", "Provide timely platform, data, and third-party access where required.", "Review feasibility findings and approve the final proposal."],
        "success_metrics": ["Human-approved acceptance criteria are met.", "Required data, access, and configuration are validated during discovery."],
        "next_steps": ["Review feature feasibility against cited documentation.", "Confirm assumptions, exclusions, acceptance criteria, and change impacts.", "Approve a revised scope before client delivery."],
        "agent_trace": ["requirements analyst extracted candidate features", "feasibility analyst queried Knowledge Brain API", "estimation analyst applied transparent task ranges", "review gate held output for human approval"],
        "hours_min": sum(task["hours_min"] for task in tasks), "hours_max": sum(task["hours_max"] for task in tasks),
        "assumptions": ["Estimates are preliminary planning ranges, not a fixed quote.", "Unknown product feasibility must be verified by a human using cited documentation.", "Custom integrations require API, access, security, and data-mapping discovery."],
        "risks": ["Unverified feature feasibility", "Scope and third-party API uncertainty", "Client access or feedback delays"],
        "approval_status": "pending_human_approval", "client_send_allowed": False,
        "is_mock_or_unverified": any(item["is_mock"] or not item["feasibility"].startswith("evidence_found") for item in checks),
    }


def create_proposal(scope: dict[str, Any]) -> dict[str, Any]:
    """Create a client-readable proposal draft that can never be sent automatically."""
    return {
        "client": scope.get("client"), "meeting_id": scope.get("meeting_id"), "title": f"Draft proposal: {scope.get('client', 'Client')}",
        "objectives": scope.get("objectives", []), "scope": scope.get("deliverables", []), "deliverables": scope.get("deliverables", []),
      "estimated_hours": f"{scope.get('hours_min', 0)}–{scope.get('hours_max', 0)} hours", "assumptions": scope.get("assumptions", []),
        "exclusions": scope.get("exclusions", []), "client_responsibilities": scope.get("client_responsibilities", []), "risks": scope.get("risks", []),
        "success_metrics": scope.get("success_metrics", []), "next_steps": scope.get("next_steps", []),
        "approval_status": "pending_human_approval", "client_send_allowed": False,
        "notice": "Draft only. A human must verify feasibility, commercial terms, and approval before client delivery.",
    }


def compare_follow_up(original_scope: dict[str, Any], followup_meeting: dict[str, Any]) -> dict[str, Any]:
    original = {str(item["feature"]) for item in original_scope.get("features", []) if item.get("feature")}
    follow_up = {item["feature"] for item in extract_features(followup_meeting.get("transcript", ""))}
    text = followup_meeting.get("transcript", "").lower()
    removed = {feature for feature in original if re.search(rf"\b(?:remove|drop)\s+{re.escape(feature)}\b", text)}
    replacements = {feature for feature in original if re.search(rf"\binstead of\s+{re.escape(feature)}\b", text)}
    changes = [{"feature": feature, "change": "removed_or_replaced", "scope_impact": "Remove from planned deliverables; confirm replacement."} for feature in sorted(removed | replacements)]
    original_keys = {_comparison_key(feature) for feature in original}
    additions = {feature for feature in follow_up if _comparison_key(feature) not in original_keys}
    changes.extend({"feature": feature, "change": "added", "scope_impact": "New requirement; estimate and feasibility review needed."} for feature in sorted(additions))
    return {"original_meeting_id": original_scope.get("meeting_id"), "follow_up_meeting_id": followup_meeting.get("meeting_id"), "changes": changes, "requires_reestimate": bool(changes), "approval_status": "pending_human_approval", "client_send_allowed": False}


def load_transcript_by_id(meeting_id: str) -> dict[str, Any]:
    for meeting in load_meetings():
        if meeting["meeting_id"] == meeting_id:
            return meeting
    raise KeyError(f"Unknown meeting id: {meeting_id}")
