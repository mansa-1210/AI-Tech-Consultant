import pytest
from inbox_agent import agent
def test_50_synthetic_emails(): assert len(agent.load_emails()) >= 50
def test_billing_classification(): assert agent.classify_email({"subject":"Invoice payment","body":"Need invoice"})=="billing"
def test_custom_classification(): assert agent.classify_email({"subject":"Custom CRM integration","body":"Can you build it"})=="custom_work"
def test_grounded_draft():
    e={"id":"1","subject":"email account","body":"manage email"}
    kb={"answer":"Use Email Accounts settings.","evidence_found":True,"citations":[{"title":"Email Accounts","url":"https://example.com"}],"evidence_status":"found","is_mock":True}
    d=agent.prepare_draft(e,kb); assert d["status"]=="grounded_draft" and d["citations"] and d["review_required"]
def test_missing_evidence_escalates():
    d=agent.prepare_draft({"id":"2","subject":"refund","body":"refund?"},{"answer":"","evidence_found":False,"citations":[]})
    assert d["status"]=="human_escalation" and not d["citations"]
def test_invalid_review_decision():
    with pytest.raises(ValueError): agent.record_review({"id":"3"},{"category":"support"},"send")
def test_threshold_and_revocation(tmp_path,monkeypatch):
    monkeypatch.setattr(agent,"DATA_DIR",tmp_path); monkeypatch.setattr(agent,"STATE_PATH",tmp_path/"state.json"); monkeypatch.setattr(agent,"FEEDBACK_PATH",tmp_path/"feedback.jsonl")
    e={"id":"x","subject":"email","body":"email account"}; d={"category":"support","status":"grounded_draft"}
    for i in range(19): r=agent.record_review({"id":str(i)},d,"approve")
    assert not r["state"]["autonomy_enabled"]
    r=agent.record_review(e,d,"approve"); assert r["state"]["autonomy_enabled"]
    r=agent.record_review(e,d,"reject"); assert not r["state"]["autonomy_enabled"]
    assert "SIMULATION ONLY" in r["notice"]

@pytest.mark.parametrize(("subject", "expected"), [
    ("Refund request", "billing"), ("Invoice copy", "billing"), ("Payment failed", "billing"),
    ("Build an integration", "custom_work"), ("New feature request", "custom_work"),
    ("Password reset", "account_access"), ("Locked out", "account_access"),
    ("Domain question", "support"), ("Website issue", "support"), ("Member import", "support"),
    ("Just saying hello", "general"), ("Campaign support", "support"), ("Price question", "billing"),
])
def test_email_category_matrix(subject, expected):
    assert agent.classify_email({"subject": subject, "body": ""}) == expected
