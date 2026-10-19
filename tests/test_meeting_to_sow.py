from meeting_to_sow import agent
def test_has_three_initial_meetings_and_followup():
    meetings=agent.load_meetings()
    assert len([m for m in meetings if m["type"]=="initial"])>=3
    assert any(m["type"]=="follow_up" for m in meetings)
def test_extract_features():
    got=agent.extract_features("Need email notifications, member CSV import and custom CRM integration.")
    names={x["feature"] for x in got}
    assert "email notifications" in names and "member import" not in names and "custom integration" in names
def test_scope_requires_human_approval(monkeypatch):
    monkeypatch.setattr(agent,"ask_knowledge_brain",lambda q: {"answer":"","evidence_found":False,"citations":[],"is_mock":True})
    meeting={"meeting_id":"M1","client":"Demo","transcript":"Need email notifications and member import from CSV."}
    scope=agent.create_scope(meeting)
    assert scope["approval_status"]=="pending_human_approval"
    assert scope["client_send_allowed"] is False
    assert scope["hours_max"]>=scope["hours_min"]
    assert all(x["feasibility"]=="unknown_needs_human_review" for x in scope["features"])
def test_salesforce_requires_salesforce_specific_evidence(monkeypatch):
    monkeypatch.setattr(agent,"ask_knowledge_brain",lambda _: {"answer":"Use generic AI connectors and maps.","evidence_found":True,"citations":[{"title":"AI Connectors","url":"https://example.test/connectors"}],"is_mock":True})
    scope=agent.create_scope({"meeting_id":"M1","client":"Demo","transcript":"We need a Salesforce connection."})
    feature=scope["features"][0]
    assert feature["feature"] == "salesforce"
    assert feature["feasibility"] == "unknown_needs_human_review"
    assert feature["citations"] == []

def test_duplicate_custom_integration_tasks_are_merged_and_hours_use_en_dash(monkeypatch):
    monkeypatch.setattr(agent,"ask_knowledge_brain",lambda _: {"answer":"","evidence_found":False,"citations":[],"is_mock":True})
    scope=agent.create_scope(agent.load_transcript_by_id("MEETING-001"))
    task_names=[task["task"] for task in scope["tasks"]]
    assert task_names.count("custom integration discovery") == 1
    assert task_names.count("custom integration implementation") == 1
    proposal=agent.create_proposal(scope)
    assert proposal["estimated_hours"] == f"{scope['hours_min']}–{scope['hours_max']} hours"
def test_followup_detects_addition_and_replacement():
    original={"meeting_id":"M1","features":[{"feature":"email notifications"},{"feature":"member import"}]}
    follow={"meeting_id":"M1-FU","transcript":"Remove email notifications. Add SMS notifications instead. Keep member import."}
    result=agent.compare_follow_up(original,follow)
    assert result["requires_reestimate"]
    assert result["client_send_allowed"] is False
    assert any(x["feature"]=="sms notifications" for x in result["changes"])

def test_followup_deduplicates_sms_and_keeps_existing_import():
    original={"meeting_id":"M1","features":[{"feature":"email notifications"},{"feature":"member import"},{"feature":"admin dashboard"}]}
    follow={"meeting_id":"M1-FU","transcript":"Remove email notifications. Add SMS notifications instead. Keep member CSV import and the admin dashboard."}
    changes=agent.compare_follow_up(original,follow)["changes"]
    assert sum(item["feature"] == "sms notifications" and item["change"] == "added" for item in changes) == 1
    assert not any(item["feature"] in {"member import", "csv import", "admin dashboard"} and item["change"] == "added" for item in changes)

def test_followup_detects_removals_and_unchanged_scope():
    original={"meeting_id":"M1","features":[{"feature":"email notifications"},{"feature":"member import"}]}
    unchanged=agent.compare_follow_up(original,{"meeting_id":"M2","transcript":"Keep email notifications and member import."})
    removed=agent.compare_follow_up(original,{"meeting_id":"M3","transcript":"Remove member import."})
    assert unchanged["changes"] == []
    assert removed["changes"] == [{"feature":"member import","change":"removed_or_replaced","scope_impact":"Remove from planned deliverables; confirm replacement."}]

import pytest

@pytest.mark.parametrize("transcript,expected", [
    ("Need member import from CSV", "member import"), ("Need CSV import", "csv import"),
    ("We need email notifications", "email notifications"), ("Build agent profiles", "agent profiles"),
    ("Add property listings", "property listings"), ("Create a search page", "search page"),
    ("Need a contact form", "contact form"), ("Need instructor profiles", "instructor profiles"),
    ("Custom integration needed", "custom integration"), ("Salesforce connection", "salesforce"),
    ("CRM connection", "crm connection"), ("Add SMS notifications", "sms notifications"),
    ("Custom dashboard", "custom dashboard"), ("Admin dashboard", "admin dashboard"),
    ("Property listings and search page", "property listings"), ("Email notifications and import", "email notifications"),
])
def test_feature_extraction_matrix(transcript, expected):
    assert expected in {item["feature"] for item in agent.extract_features(transcript)}

def test_proposal_is_never_sendable(monkeypatch):
    monkeypatch.setattr(agent, "ask_knowledge_brain", lambda _: {"evidence_found": False, "is_mock": True})
    proposal=agent.create_proposal(agent.create_scope(agent.load_transcript_by_id("MEETING-001")))
    assert proposal["client_send_allowed"] is False
    assert {"objectives", "scope", "deliverables", "assumptions", "exclusions", "client_responsibilities", "risks", "success_metrics", "next_steps"}.issubset(proposal)
