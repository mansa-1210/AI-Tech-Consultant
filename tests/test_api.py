"""Unit and integration tests for Knowledge Brain HTTP API.

Tests health check, question answering, citation verification, input validation,
error handling, and CLI runner. Fully offline — zero network calls or paid APIs.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch
import pytest
from werkzeug.test import Client

from knowledge_brain.answerer import (
    INSUFFICIENT_EVIDENCE_PHRASE,
    MOCK_PREFIX,
    AnswerResult,
    Citation,
    KnowledgeAnswerer,
)
from knowledge_brain.api import create_app, parse_api_cli_args
from knowledge_brain.ingestion import ArticleRecord
from knowledge_brain.retriever import KnowledgeRetriever


# ---------------------------------------------------------------------------
# Test Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_records() -> list[ArticleRecord]:
    """Sample documentation article records for testing."""
    return [
        ArticleRecord(
            url="https://support.brilliantdirectories.com/support/solutions/articles/5000558357-can-i-send-a-mass-email-campaign-using-brilliant-directories-",
            title="Can I Send A Mass Email Campaign Using Brilliant Directories?",
            text=(
                "Can I Send A Mass Email Campaign Using Brilliant Directories?\n"
                "The built-in email system is designed for emailing active members who have requested to receive emails from the site.\n"
                "Sending mass unsolicited emails to Claim Listings or an imported contact list may be considered spam.\n"
                "This can violate email compliance laws and damage your domain reputation."
            ),
            retrieval_timestamp="2026-10-17T15:57:53.430752+00:00",
            content_hash="hash1",
            metadata={"char_count": 450, "word_count": 65},
        ),
        ArticleRecord(
            url="https://support.brilliantdirectories.com/support/solutions/articles/5000558363-emails-email-accounts",
            title="Emails » Email Accounts",
            text=(
                "Emails » Email Accounts\n"
                "Email accounts (mailboxes) at the website's own domain, such as support@example.com,\n"
                "are created and managed in Settings > Domain Manager > Mailboxes.\n"
                "The Emails > Email Accounts menu item opens that tab directly.\n"
                "Connect a custom domain to the website. A free directoryup.com address has no mailboxes of its own.\n"
                "On the Mailboxes tab, Add a mailbox creates a new mailbox with its password and storage limit."
            ),
            retrieval_timestamp="2026-10-17T15:57:54.305087+00:00",
            content_hash="hash2",
            metadata={"char_count": 480, "word_count": 75},
        ),
        ArticleRecord(
            url="https://support.brilliantdirectories.com/support/solutions/articles/5000558373-email-forwarders",
            title="Email Forwarders",
            text=(
                "Email Forwarders\n"
                "The Email Forwarders functionality has been removed from our implementation of cPanel as of June, 2024.\n"
                "The recommended alternative is using an email app like Thunderbird, Apple Mail, or Gmail."
            ),
            retrieval_timestamp="2026-10-17T15:57:55.371810+00:00",
            content_hash="hash3",
            metadata={"char_count": 350, "word_count": 50},
        ),
    ]


@pytest.fixture
def test_retriever(sample_records: list[ArticleRecord]) -> KnowledgeRetriever:
    """Fixture providing an in-memory KnowledgeRetriever."""
    return KnowledgeRetriever(articles=sample_records)


@pytest.fixture
def test_answerer() -> KnowledgeAnswerer:
    """Fixture providing a mock-mode KnowledgeAnswerer."""
    return KnowledgeAnswerer(use_mock=True)


@pytest.fixture
def client(test_retriever: KnowledgeRetriever, test_answerer: KnowledgeAnswerer) -> Client:
    """Fixture providing a Flask test client configured with mock components."""
    app = create_app(
        retriever=test_retriever,
        answerer=test_answerer,
        config={"TESTING": True},
    )
    return app.test_client()


# ---------------------------------------------------------------------------
# Health Endpoint Tests
# ---------------------------------------------------------------------------

class TestHealthEndpoint:
    """Tests for GET /health and GET /api/health."""

    def test_health_check_success(self, client: Client) -> None:
        """Health endpoint returns 200 with healthy status and metadata."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.get_json()

        assert data["status"] == "healthy"
        assert data["service"] == "knowledge_brain"
        assert data["mode"] == "mock"
        assert data["articles_loaded"] == 3
        assert "model" in data

    def test_api_health_alias(self, client: Client) -> None:
        """Namespaced /api/health returns identical status."""
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.get_json()
        assert data["status"] == "healthy"

    def test_health_wrong_method(self, client: Client) -> None:
        """POST /health returns 405 Method Not Allowed in structured JSON."""
        response = client.post("/health")
        assert response.status_code == 405
        data = response.get_json()
        assert "error" in data


# ---------------------------------------------------------------------------
# Question-Answering Endpoint Tests (Ask / Query)
# ---------------------------------------------------------------------------

class TestAskEndpoint:
    """Tests for POST /ask and its aliases."""

    def test_ask_supported_mailbox_question(self, client: Client) -> None:
        """Supported question about mailboxes returns grounded answer with citations."""
        response = client.post(
            "/ask",
            json={"query": "Where are email accounts and mailboxes managed?"},
        )
        assert response.status_code == 200
        data = response.get_json()

        assert data["query"] == "Where are email accounts and mailboxes managed?"
        assert data["evidence_found"] is True
        assert data["evidence_status"] == "found"
        assert data["mode"] == "mock"
        assert data["is_mock"] is True

        # Check answer content
        assert MOCK_PREFIX in data["answer"]
        assert "Domain Manager" in data["answer"] or "Mailboxes" in data["answer"]

        # Check citations structure
        assert isinstance(data["citations"], list)
        assert len(data["citations"]) >= 1
        top_citation = data["citations"][0]
        assert "title" in top_citation
        assert "url" in top_citation
        assert "Email Accounts" in top_citation["title"]
        assert "5000558363-emails-email-accounts" in top_citation["url"]

    def test_ask_unsupported_refund_question(self, client: Client) -> None:
        """Unsupported question returns explicit fallback with no fabricated citations."""
        response = client.post(
            "/ask",
            json={"query": "What is the refund policy for directory purchases?"},
        )
        assert response.status_code == 200
        data = response.get_json()

        assert data["evidence_found"] is False
        assert data["evidence_status"] == "insufficient_evidence"
        assert INSUFFICIENT_EVIDENCE_PHRASE in data["answer"]
        assert data["citations"] == []

    def test_ask_mass_email_question(self, client: Client) -> None:
        """Question about mass email returns grounded answer referencing mass email article."""
        response = client.post(
            "/ask",
            json={"query": "Can I send a mass email campaign to imported contacts?"},
        )
        assert response.status_code == 200
        data = response.get_json()

        assert data["evidence_found"] is True
        assert data["evidence_status"] == "found"
        assert len(data["citations"]) >= 1
        assert "Mass Email" in data["citations"][0]["title"]
        assert "5000558357" in data["citations"][0]["url"]

    def test_ask_question_alias(self, client: Client) -> None:
        """Passing 'question' instead of 'query' is supported seamlessly."""
        response = client.post(
            "/ask",
            json={"question": "Where are email accounts managed?"},
        )
        assert response.status_code == 200
        data = response.get_json()
        assert data["evidence_found"] is True
        assert "Domain Manager" in data["answer"] or "Mailboxes" in data["answer"]

    def test_query_route_alias(self, client: Client) -> None:
        """POST /query functions identically to POST /ask."""
        response = client.post(
            "/query",
            json={"query": "Can I send mass emails?"},
        )
        assert response.status_code == 200
        data = response.get_json()
        assert data["evidence_found"] is True

    def test_api_ask_namespaced_alias(self, client: Client) -> None:
        """POST /api/ask functions identically to POST /ask."""
        response = client.post(
            "/api/ask",
            json={"query": "Where are email accounts managed?"},
        )
        assert response.status_code == 200
        data = response.get_json()
        assert data["evidence_found"] is True

    def test_top_k_parameter(self, client: Client) -> None:
        """Explicit top_k parameter is accepted and respected."""
        response = client.post(
            "/ask",
            json={
                "query": "Where are mailboxes managed?",
                "top_k": 1,
            },
        )
        assert response.status_code == 200
        data = response.get_json()
        assert data["evidence_found"] is True
        assert len(data["citations"]) <= 1

    def test_use_mock_explicit_true(self, client: Client) -> None:
        """Explicit use_mock=True works safely."""
        response = client.post(
            "/ask",
            json={
                "query": "Where are mailboxes managed?",
                "use_mock": True,
            },
        )
        assert response.status_code == 200
        data = response.get_json()
        assert data["mode"] == "mock"
        assert data["is_mock"] is True


# ---------------------------------------------------------------------------
# Validation & Error Handling Tests
# ---------------------------------------------------------------------------

class TestValidationAndErrors:
    """Tests for bad inputs, malformed requests, and internal errors."""

    def test_missing_json_content_type(self, client: Client) -> None:
        """Request without application/json Content-Type returns 400."""
        response = client.post(
            "/ask",
            data="query=mailboxes",
            content_type="text/plain",
        )
        assert response.status_code == 400
        data = response.get_json()
        assert "Invalid Content-Type" in data["error"]

    def test_malformed_json_body(self, client: Client) -> None:
        """Malformed JSON payload returns 400."""
        response = client.post(
            "/ask",
            data="{malformed json syntax",
            content_type="application/json",
        )
        assert response.status_code == 400
        data = response.get_json()
        assert "Malformed JSON" in data["error"] or "error" in data

    def test_json_body_not_an_object(self, client: Client) -> None:
        """JSON body that is a list or scalar returns 400."""
        response = client.post(
            "/ask",
            data=json.dumps(["item1", "item2"]),
            content_type="application/json",
        )
        assert response.status_code == 400
        data = response.get_json()
        assert "JSON body must be an object" in data["error"]

    def test_missing_query_and_question_fields(self, client: Client) -> None:
        """Payload without 'query' or 'question' returns 400."""
        response = client.post("/ask", json={"other_field": "val"})
        assert response.status_code == 400
        data = response.get_json()
        assert "Missing required field" in data["error"]

    def test_empty_query_string(self, client: Client) -> None:
        """Empty query string returns 400."""
        response = client.post("/ask", json={"query": ""})
        assert response.status_code == 400
        data = response.get_json()
        assert "cannot be empty" in data["error"]

    def test_whitespace_only_query(self, client: Client) -> None:
        """Whitespace-only query string returns 400."""
        response = client.post("/ask", json={"query": "   \n\t   "})
        assert response.status_code == 400
        data = response.get_json()
        assert "cannot be empty" in data["error"]

    def test_non_string_query(self, client: Client) -> None:
        """Non-string query value returns 400."""
        response = client.post("/ask", json={"query": 12345})
        assert response.status_code == 400
        data = response.get_json()
        assert "must be a string" in data["error"]

    def test_invalid_top_k_values(self, client: Client) -> None:
        """Invalid top_k values (negative, zero, non-int, boolean) return 400."""
        for bad_top_k in [0, -1, "five", True, 2.5]:
            response = client.post(
                "/ask",
                json={"query": "mailboxes", "top_k": bad_top_k},
            )
            assert response.status_code == 400
            data = response.get_json()
            assert "positive integer" in data["error"]

    def test_invalid_use_mock_value(self, client: Client) -> None:
        """Non-boolean use_mock returns 400."""
        response = client.post(
            "/ask",
            json={"query": "mailboxes", "use_mock": "yes"},
        )
        assert response.status_code == 400
        data = response.get_json()
        assert "must be a boolean" in data["error"]

    def test_get_ask_method_not_allowed(self, client: Client) -> None:
        """GET /ask returns 405 Method Not Allowed in JSON."""
        response = client.get("/ask")
        assert response.status_code == 405
        data = response.get_json()
        assert "error" in data

    def test_route_not_found(self, client: Client) -> None:
        """Accessing a non-existent route returns 404 in structured JSON."""
        response = client.get("/nonexistent/endpoint")
        assert response.status_code == 404
        data = response.get_json()
        assert "error" in data

    def test_internal_server_error_safety(self, client: Client, test_answerer: KnowledgeAnswerer) -> None:
        """Unexpected internal exceptions are caught and return 500 JSON without crashing."""
        with patch.object(test_answerer, "ask", side_effect=RuntimeError("Simulated engine crash")):
            response = client.post(
                "/ask",
                json={"query": "Where are mailboxes managed?"},
            )
            assert response.status_code == 500
            data = response.get_json()
            assert "error" in data
            assert "Simulated engine crash" in data["detail"]


# ---------------------------------------------------------------------------
# Default Configuration & CLI Tests
# ---------------------------------------------------------------------------

class TestAppConfigAndCLI:
    """Tests for default app factory and CLI parser."""

    def test_default_create_app(self) -> None:
        """Default create_app initializes with sample disk articles."""
        app = create_app(config={"TESTING": True})
        client = app.test_client()
        response = client.get("/health")
        assert response.status_code == 200
        data = response.get_json()
        assert data["status"] == "healthy"
        assert data["articles_loaded"] >= 3

    def test_explicit_data_path_overrides_default_corpus(self, tmp_path) -> None:
        custom_data = tmp_path / "articles.json"
        custom_data.write_text(json.dumps([{
            "url": "https://example.test/article", "title": "Custom", "text": "Custom facts",
            "retrieval_timestamp": "now", "content_hash": "custom", "metadata": {},
        }]))
        app = create_app(data_path=custom_data, config={"TESTING": True})
        assert app.test_client().get("/health").get_json()["articles_loaded"] == 1

    def test_parse_api_cli_args_defaults(self) -> None:
        """CLI argument parser sets expected defaults."""
        args = parse_api_cli_args([])
        assert args.host in ("127.0.0.1", "localhost")
        assert args.port == 8501
        assert args.mock is False
        assert args.debug is False

    def test_parse_api_cli_args_custom(self) -> None:
        """CLI argument parser properly parses custom flags."""
        args = parse_api_cli_args([
            "--host", "0.0.0.0",
            "--port", "9000",
            "--mock",
            "--debug",
            "--data", "custom_articles.json",
        ])
        assert args.host == "0.0.0.0"
        assert args.port == 9000
        assert args.mock is True
        assert args.debug is True
        assert args.data == "custom_articles.json"


class TestEscalationsAndDemo:
    def test_live_provider_failure_returns_clear_safe_response(self, test_retriever: KnowledgeRetriever) -> None:
        app = create_app(
            retriever=test_retriever,
            answerer=KnowledgeAnswerer(use_mock=False, provider="claude", api_key=""),
        )
        response = app.test_client().post("/ask", json={"query": "Where are mailboxes managed?"})
        assert response.status_code == 503
        data = response.get_json()
        assert data["evidence_status"] == "provider_unavailable"
        assert data["evidence_found"] is False
        assert data["citations"] == []
        assert data["is_mock"] is False

    def test_escalation_create_and_resolve(self, client: Client, tmp_path) -> None:
        client.application.config["ESCALATIONS_PATH"] = str(tmp_path / "escalations.json")
        created = client.post("/escalations", json={"question": "What is the refund policy?"})
        assert created.status_code == 201
        resolved = client.post(f"/escalations/{created.get_json()['id']}/resolve", json={"answer": "Ask billing."})
        assert resolved.status_code == 200
        assert resolved.get_json()["status"] == "resolved"

    def test_health_reports_actual_gemini_provider(self, test_retriever: KnowledgeRetriever, monkeypatch) -> None:
        monkeypatch.setenv("AI_PROVIDER", "gemini")
        app = create_app(
            retriever=test_retriever,
            answerer=KnowledgeAnswerer(use_mock=False, model="gemini-test", api_key="test-key"),
        )
        data = app.test_client().get("/health").get_json()
        assert data["mode"] == "gemini"
        assert data["model"] == "gemini-test"
