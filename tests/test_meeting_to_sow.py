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
def test_followup_detects_addition_and_replacement():
    original={"meeting_id":"M1","features":[{"feature":"email notifications"},{"feature":"member import"}]}
    follow={"meeting_id":"M1-FU","transcript":"Remove email notifications. Add SMS notifications instead. Keep member import."}
    result=agent.compare_follow_up(original,follow)
    assert result["requires_reestimate"]
    assert result["client_send_allowed"] is False
    assert any(x["feature"]=="sms notifications" for x in result["changes"])
