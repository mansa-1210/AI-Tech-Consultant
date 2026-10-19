"""Grounded Inbox Agent with auditable human review and simulated autonomy."""
from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import requests

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
EMAILS_PATH = DATA_DIR / "emails.json"
STATE_PATH = DATA_DIR / "review_state.json"
FEEDBACK_PATH = DATA_DIR / "feedback.jsonl"
PATTERNS_PATH = DATA_DIR / "response_patterns.json"
KB_URL = os.getenv("KNOWLEDGE_BRAIN_URL", "http://127.0.0.1:8501/ask")
AUTO_SEND_THRESHOLD = 20
AUTO_SEND_BLOCKED_CATEGORIES = {"billing", "custom_work", "account_access"}
PROMPT_VERSION = "inbox-template-v1"
RULES = {
    "billing": ("invoice", "billing", "payment", "refund", "charge", "subscription", "price"),
    "custom_work": ("custom", "build", "integration", "develop", "new feature", "widget", "crm"),
    "account_access": ("login", "log in", "password", "access", "locked"),
    "support": ("email", "domain", "website", "member", "import", "campaign", "forward"),
}
GENERIC_EVIDENCE_TERMS = {
    "about", "answer", "asking", "based", "contact", "create", "documentation",
    "email", "from", "help", "information", "manage", "need", "please", "question",
    "reaching", "settings", "support", "team", "that", "their", "there", "this", "where",
    "with", "your",
}


def classify_email(email: Dict[str, Any]) -> str:
    text = (email.get("subject", "") + " " + email.get("body", "")).lower()
    scores = {category: sum(word in text for word in words) for category, words in RULES.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] else "general"


def load_emails() -> list[Dict[str, Any]]:
    return json.loads(EMAILS_PATH.read_text(encoding="utf-8"))


def _new_state() -> Dict[str, Any]:
    return {"schema_version": 2, "prompt_version": PROMPT_VERSION, "categories": {}}


def _category_state(state: Dict[str, Any], category: str) -> Dict[str, Any]:
    categories = state.setdefault("categories", {})
    return categories.setdefault(
        category,
        {
            "consecutive_unchanged_sends": 0,
            "autonomy_eligible": False,
            "autonomy_trial_enabled": False,
            "revoked_reason": None,
        },
    )


def load_state() -> Dict[str, Any]:
    try:
        state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return _new_state()
    if not isinstance(state, dict):
        return _new_state()
    if state.get("schema_version") == 2 and isinstance(state.get("categories"), dict):
        return state
    migrated = _new_state()
    category = state.get("last_category")
    if isinstance(category, str) and state.get("consecutive_approvals", 0):
        category_state = _category_state(migrated, category)
        category_state["consecutive_unchanged_sends"] = int(state["consecutive_approvals"])
        category_state["revoked_reason"] = "legacy_state_requires_final_text_audit"
    return migrated


def save_state(state: Dict[str, Any]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_response_patterns() -> Dict[str, Dict[str, str]]:
    try:
        patterns = json.loads(PATTERNS_PATH.read_text(encoding="utf-8"))
        return patterns if isinstance(patterns, dict) else {}
    except (OSError, ValueError):
        return {}


def promote_response_pattern(category: str, greeting: str, closing: str, version: str) -> None:
    """Persist a human-approved, reversible drafting pattern for one category."""
    if not all(isinstance(value, str) and value.strip() for value in (category, greeting, closing, version)):
        raise ValueError("category, greeting, closing, and version must be non-empty strings")
    patterns = load_response_patterns()
    patterns[category] = {"greeting": greeting.strip(), "closing": closing.strip(), "version": version.strip()}
    PATTERNS_PATH.parent.mkdir(parents=True, exist_ok=True)
    PATTERNS_PATH.write_text(json.dumps(patterns, indent=2, ensure_ascii=False), encoding="utf-8")


def feedback_summary() -> Dict[str, Dict[str, float | int]]:
    """Summarize exact-match and review outcomes by category and prompt version."""
    summary: Dict[str, Dict[str, float | int]] = {}
    try:
        lines = FEEDBACK_PATH.read_text(encoding="utf-8").splitlines()
    except OSError:
        return summary
    for line in lines:
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        category = item.get("category")
        if not isinstance(category, str):
            continue
        metrics = summary.setdefault(category, {"reviews": 0, "unchanged_final_text": 0, "edits": 0, "rejections": 0})
        metrics["reviews"] += 1
        metrics["unchanged_final_text"] += int(bool(item.get("unchanged_final_text")))
        metrics["edits"] += int(item.get("decision") == "edit")
        metrics["rejections"] += int(item.get("decision") == "reject")
    for metrics in summary.values():
        reviews = int(metrics["reviews"])
        metrics["unchanged_rate"] = round(int(metrics["unchanged_final_text"]) / reviews, 4) if reviews else 0.0
    return summary


def ask_knowledge_brain(question: str) -> Dict[str, Any]:
    try:
        response = requests.post(KB_URL, json={"query": question}, timeout=8)
        data = response.json()
        if isinstance(data, dict):
            if response.status_code >= 400:
                return {
                    "answer": "",
                    "evidence_found": False,
                    "evidence_status": data.get("evidence_status", "provider_unavailable"),
                    "citations": [],
                    "mode": data.get("mode", "unknown"),
                    "is_mock": bool(data.get("is_mock", False)),
                }
            return data
    except (requests.RequestException, ValueError):
        pass
    return {"answer": "", "evidence_found": False, "evidence_status": "provider_unavailable", "citations": [], "mode": "unknown", "is_mock": False}


def _evidence_terms(text: str) -> set[str]:
    terms = set()
    for word in re.findall(r"[a-z0-9]+", text.lower()):
        normalized = word[:-1] if word.endswith("s") and len(word) > 4 else word
        if len(normalized) >= 3 and normalized not in GENERIC_EVIDENCE_TERMS:
            terms.add(normalized)
    return terms


def evidence_is_relevant_to_email(email: Dict[str, Any], knowledge: Dict[str, Any]) -> bool:
    """Require source metadata and answer text to overlap with email-specific terms."""
    email_terms = _evidence_terms(f"{email.get('subject', '')} {email.get('body', '')}")
    answer_terms = _evidence_terms(str(knowledge.get("answer", "")))
    citation_text = " ".join(
        f"{citation.get('title', '')} {citation.get('url', '')}"
        for citation in knowledge.get("citations", [])
        if isinstance(citation, dict)
    )
    citation_terms = _evidence_terms(citation_text)
    overlapping_terms = email_terms & answer_terms
    if not citation_terms:
        return False
    return len(overlapping_terms) >= 2 or any(len(term) >= 7 for term in overlapping_terms)


def prepare_draft(email: Dict[str, Any], kb_result: Dict[str, Any] | None = None) -> Dict[str, Any]:
    category = classify_email(email)
    pattern = load_response_patterns().get(category, {})
    greeting = str(pattern.get("greeting", "Hello,"))
    closing = str(pattern.get("closing", "Please let us know if you need further help.\n\nBest regards,\nSupport Team"))
    prompt_version = str(pattern.get("version", PROMPT_VERSION))
    knowledge = kb_result if kb_result is not None else ask_knowledge_brain(email.get("subject", "") + "\n" + email.get("body", ""))
    grounded = (
        bool(knowledge.get("evidence_found"))
        and bool(knowledge.get("answer"))
        and bool(knowledge.get("citations"))
        and evidence_is_relevant_to_email(email, knowledge)
    )
    evidence_status = str(knowledge.get("evidence_status", "insufficient_evidence"))
    if grounded:
        answer = str(knowledge["answer"]).replace("[MOCK MODE]", "").strip()
        draft_text = greeting + "\n\nThank you for contacting us. Based on our support documentation:\n\n" + answer + "\n\n" + closing
        status = "grounded_draft"
    elif evidence_status == "provider_unavailable":
        draft_text = greeting + "\n\nThank you for reaching out. Our support drafting service is temporarily unavailable, so Iâ€™m escalating this to a human support team member. Weâ€™ll follow up with a verified answer.\n\n" + closing
        status = "provider_unavailable"
    else:
        draft_text = greeting + "\n\nThank you for reaching out. To avoid giving inaccurate information, I’m escalating your question to our support team for confirmation. We’ll follow up once we have a verified answer.\n\n" + closing
        status = "human_escalation"
    return {
        "email_id": email.get("id"),
        "category": category,
        "draft": draft_text,
        "draft_hash": _text_hash(draft_text),
        "status": status,
        "citations": knowledge.get("citations", []) if grounded else [],
        "evidence_status": evidence_status,
        "mode": knowledge.get("mode", "unknown"),
        "is_mock": bool(knowledge.get("is_mock", False)),
        "prompt_version": prompt_version,
        "review_required": True,
        "auto_send_eligible": False,
    }


def _safety_gates(category: str, draft: Dict[str, Any], unchanged: bool, trial_enabled: bool) -> Dict[str, bool]:
    return {
        "human_trial_enabled": trial_enabled,
        "unchanged_final_text": unchanged,
        "grounded_evidence": draft.get("status") == "grounded_draft" and bool(draft.get("citations")),
        "live_model": not bool(draft.get("is_mock", True)),
        "category_allowed": category not in AUTO_SEND_BLOCKED_CATEGORIES,
    }


def enable_autonomy_trial(category: str, operator_approved: bool) -> Dict[str, Any]:
    """Enable only a human-approved, category-specific simulated autonomy trial."""
    state = load_state()
    category_state = _category_state(state, category)
    if not operator_approved:
        raise ValueError("operator_approved must be true")
    if category in AUTO_SEND_BLOCKED_CATEGORIES:
        raise ValueError("This category is permanently blocked from autonomous sending")
    if not category_state["autonomy_eligible"]:
        raise ValueError("Twenty unchanged audited sends are required before a trial")
    category_state["autonomy_trial_enabled"] = True
    category_state["revoked_reason"] = None
    save_state(state)
    return {"category": category, "state": category_state, "notice": "SIMULATION ONLY: no email was sent."}


def record_review(
    email: Dict[str, Any],
    draft: Dict[str, Any],
    decision: str,
    edited_draft: str | None = None,
    final_sent_draft: str | None = None,
) -> Dict[str, Any]:
    """Record an auditable review; only an exact final-text match can qualify."""
    if decision not in {"approve", "edit", "reject"}:
        raise ValueError("decision must be approve, edit, or reject")
    state = load_state()
    category = str(draft.get("category", classify_email(email)))
    category_state = _category_state(state, category)
    generated_text = str(draft.get("draft", ""))
    final_text = final_sent_draft if final_sent_draft is not None else edited_draft
    unchanged = decision == "approve" and isinstance(final_text, str) and generated_text == final_text
    gates = _safety_gates(category, draft, unchanged, category_state["autonomy_trial_enabled"])
    qualification_gates = {name: passed for name, passed in gates.items() if name != "human_trial_enabled"}
    qualifies = decision == "approve" and all(qualification_gates.values())

    if qualifies:
        category_state["consecutive_unchanged_sends"] += 1
        if category_state["consecutive_unchanged_sends"] >= AUTO_SEND_THRESHOLD:
            category_state["autonomy_eligible"] = True
    else:
        category_state["consecutive_unchanged_sends"] = 0
        category_state["autonomy_eligible"] = False
        category_state["autonomy_trial_enabled"] = False
        category_state["revoked_reason"] = "edited_or_rejected_or_failed_safety_gate"

    auto_send_allowed = qualifies and all(gates.values())
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "email_id": email.get("id"),
        "category": category,
        "decision": decision,
        "draft_status": draft.get("status"),
        "prompt_version": draft.get("prompt_version", PROMPT_VERSION),
        "generated_draft_hash": _text_hash(generated_text),
        "final_sent_draft_hash": _text_hash(final_text) if isinstance(final_text, str) else None,
        "final_text_provided": isinstance(final_text, str),
        "unchanged_final_text": unchanged,
        "safety_gates": gates,
        "simulated_auto_send": auto_send_allowed,
    }
    save_state(state)
    FEEDBACK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with FEEDBACK_PATH.open("a", encoding="utf-8") as feedback_file:
        feedback_file.write(json.dumps(record, ensure_ascii=False) + "\n")
    return {"recorded": True, "state": state, "feedback": record, "notice": "SIMULATION ONLY: no email was sent."}


if __name__ == "__main__":
    data = load_emails()
    print(f"Loaded {len(data)} synthetic emails.")
    for email in data[:5]:
        result = prepare_draft(email)
        print(email["id"], result["category"], result["status"])
        print(result["draft"])
