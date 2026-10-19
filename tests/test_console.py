from app import create_app


def test_console_has_all_three_workflows(monkeypatch):
    app = create_app()
    client = app.test_client()
    console = client.get("/")
    assert console.status_code == 200
    assert "MOCK / SIMULATED MODE" in console.get_data(as_text=True)
    assert len(client.get("/inbox/emails").get_json()) >= 50
    assert len(client.get("/meetings").get_json()) >= 4


def test_console_draft_and_scope(monkeypatch):
    monkeypatch.setattr("meeting_to_sow.agent.ask_knowledge_brain", lambda _: {"evidence_found": False, "citations": [], "is_mock": True})
    app = create_app()
    client = app.test_client()
    assert client.post("/inbox/draft", json={"email_id": "EMAIL-001"}).status_code == 200
    response = client.post("/meetings/scope", json={"meeting_id": "MEETING-001"})
    assert response.status_code == 200
    assert response.get_json()["proposal"]["client_send_allowed"] is False
    comparison = client.post("/meetings/compare", json={"meeting_id": "MEETING-001", "follow_up_meeting_id": "MEETING-001-FU1"})
    assert comparison.status_code == 200
    assert comparison.get_json()["requires_reestimate"] is True
