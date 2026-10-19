import pytest
from inbox_agent import agent
def test_50_synthetic_emails(): assert len(agent.load_emails()) >= 50
def test_billing_classification(): assert agent.classify_email({"subject":"Invoice payment","body":"Need invoice"})=="billing"
def test_custom_classification(): assert agent.classify_email({"subject":"Custom CRM integration","body":"Can you build it"})=="custom_work"
def test_grounded_draft():
    e={"id":"1","subject":"email account","body":"manage email"}
    kb={"answer":"Use Email Accounts settings.","evidence_found":True,"citations":[{"title":"Email Accounts","url":"https://example.com"}],"evidence_status":"found","is_mock":True}
    d=agent.prepare_draft(e,kb); assert d["status"]=="grounded_draft" and d["citations"] and d["review_required"]
def test_irrelevant_citations_are_not_presented_as_email_evidence():
    email={"id":"commerce","subject":"E-commerce AI support", "body":"Can the AI assistant handle our online store orders?"}
    kb={"answer":"Configure domains and mailboxes in Email Accounts.","evidence_found":True,"citations":[{"title":"Domains and Mailboxes","url":"https://example.test/domains-mailboxes"}],"evidence_status":"found","is_mock":True}
    draft=agent.prepare_draft(email,kb)
    assert draft["status"] == "human_escalation"
    assert draft["citations"] == []
    assert draft["review_required"] is True
def test_missing_evidence_escalates():
    d=agent.prepare_draft({"id":"2","subject":"refund","body":"refund?"},{"answer":"","evidence_found":False,"citations":[]})
    assert d["status"]=="human_escalation" and not d["citations"]
def test_provider_failure_is_not_reported_as_insufficient_evidence():
    d=agent.prepare_draft({"id":"2","subject":"mailbox","body":"mailbox?"},{"answer":"","evidence_found":False,"citations":[],"evidence_status":"provider_unavailable","is_mock":False})
    assert d["status"] == "provider_unavailable"
    assert d["evidence_status"] == "provider_unavailable"
    assert d["review_required"] is True
def test_invalid_review_decision():
    with pytest.raises(ValueError): agent.record_review({"id":"3"},{"category":"support"},"send")
def test_threshold_and_revocation(tmp_path,monkeypatch):
    monkeypatch.setattr(agent,"DATA_DIR",tmp_path); monkeypatch.setattr(agent,"STATE_PATH",tmp_path/"state.json"); monkeypatch.setattr(agent,"FEEDBACK_PATH",tmp_path/"feedback.jsonl")
    e={"id":"x","subject":"email","body":"email account"}
    d={"category":"support","status":"grounded_draft","draft":"Verified reply","citations":[{"url":"https://example.test"}],"is_mock":False}
    for i in range(20): r=agent.record_review({"id":str(i)},d,"approve",final_sent_draft="Verified reply")
    category_state=r["state"]["categories"]["support"]
    assert category_state["autonomy_eligible"]
    agent.enable_autonomy_trial("support", operator_approved=True)
    r=agent.record_review(e,d,"reject",final_sent_draft="Verified reply")
    assert not r["state"]["categories"]["support"]["autonomy_trial_enabled"]
    assert "SIMULATION ONLY" in r["notice"]

def test_changed_final_text_never_qualifies_or_enables_autonomy(tmp_path,monkeypatch):
    monkeypatch.setattr(agent,"DATA_DIR",tmp_path); monkeypatch.setattr(agent,"STATE_PATH",tmp_path/"state.json"); monkeypatch.setattr(agent,"FEEDBACK_PATH",tmp_path/"feedback.jsonl")
    draft={"category":"support","status":"grounded_draft","draft":"Original","citations":[{"url":"https://example.test"}],"is_mock":False}
    result=agent.record_review({"id":"x"},draft,"approve",final_sent_draft="Edited by reviewer")
    category_state=result["state"]["categories"]["support"]
    assert not result["feedback"]["unchanged_final_text"]
    assert category_state["consecutive_unchanged_sends"] == 0
    assert not result["feedback"]["simulated_auto_send"]

def test_high_risk_category_cannot_start_autonomy_trial(tmp_path,monkeypatch):
    monkeypatch.setattr(agent,"DATA_DIR",tmp_path); monkeypatch.setattr(agent,"STATE_PATH",tmp_path/"state.json"); monkeypatch.setattr(agent,"FEEDBACK_PATH",tmp_path/"feedback.jsonl")
    with pytest.raises(ValueError): agent.enable_autonomy_trial("billing", operator_approved=True)

def test_feedback_summary_and_approved_pattern(tmp_path,monkeypatch):
    monkeypatch.setattr(agent,"DATA_DIR",tmp_path); monkeypatch.setattr(agent,"STATE_PATH",tmp_path/"state.json"); monkeypatch.setattr(agent,"FEEDBACK_PATH",tmp_path/"feedback.jsonl"); monkeypatch.setattr(agent,"PATTERNS_PATH",tmp_path/"patterns.json")
    agent.promote_response_pattern("support","Hi there,","Regards,\nSupport","support-v2")
    draft=agent.prepare_draft({"id":"x","subject":"domain question","body":"domain"},{"answer":"Use the domain settings.","evidence_found":True,"citations":[{"url":"https://example.test"}],"is_mock":False})
    result=agent.record_review({"id":"x"},draft,"approve",final_sent_draft=draft["draft"])
    summary=agent.feedback_summary()
    assert draft["prompt_version"] == "support-v2"
    assert result["feedback"]["unchanged_final_text"]
    assert summary["support"]["unchanged_rate"] == 1.0

@pytest.mark.parametrize(("subject", "expected"), [
    ("Refund request", "billing"), ("Invoice copy", "billing"), ("Payment failed", "billing"),
    ("Build an integration", "custom_work"), ("New feature request", "custom_work"),
    ("Password reset", "account_access"), ("Locked out", "account_access"),
    ("Domain question", "support"), ("Website issue", "support"), ("Member import", "support"),
    ("Just saying hello", "general"), ("Campaign support", "support"), ("Price question", "billing"),
])
def test_email_category_matrix(subject, expected):
    assert agent.classify_email({"subject": subject, "body": ""}) == expected
