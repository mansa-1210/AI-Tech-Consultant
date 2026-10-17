"""Unit and integration tests for Knowledge Brain Answer Generation module.

Tests grounded answer synthesis, explicit fallback on insufficient or missing evidence,
citation preservation, mock mode labeling, absence of hardcoding, Anthropic Claude interface,
and CLI execution. Fully offline — zero network calls or paid APIs required.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from knowledge_brain.answerer import (
    INSUFFICIENT_EVIDENCE_PHRASE,
    MOCK_PREFIX,
    AnswerResult,
    Citation,
    KnowledgeAnswerer,
    answer_question,
    extract_candidate_passages,
    generate_mock_answer,
    is_mock_ai_enabled,
    main,
    normalize_token_root,
)
from knowledge_brain.ingestion import ArticleRecord
from knowledge_brain.retriever import (
    DEFAULT_SAMPLE_DATA_PATH,
    KnowledgeRetriever,
    SearchResult,
)


# ---------------------------------------------------------------------------
# Test Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_records() -> list[ArticleRecord]:
    """Provide realistic sample article records for testing."""
    return [
        ArticleRecord(
            url="https://support.brilliantdirectories.com/support/solutions/articles/5000558357-can-i-send-a-mass-email-campaign-using-brilliant-directories-",
            title="Can I Send A Mass Email Campaign Using Brilliant Directories?",
            text=(
                "Can I Send A Mass Email Campaign Using Brilliant Directories?\n"
                "The built-in email system is designed for emailing active members who have requested to receive emails from the site.\n"
                "A common example would be sending a monthly newsletter to members who have opted in to receive updates.\n"
                "Sending mass unsolicited emails to Claim Listings or an imported contact list may be considered spam.\n"
                "This can violate email compliance laws, damage the sending reputation of your domain, and result in future emails being blocked or blacklisted."
            ),
            retrieval_timestamp="2026-10-17T15:57:53.430752+00:00",
            content_hash="hash1",
            metadata={"char_count": 520, "word_count": 85},
        ),
        ArticleRecord(
            url="https://support.brilliantdirectories.com/support/solutions/articles/5000558363-emails-email-accounts",
            title="Emails » Email Accounts",
            text=(
                "Emails » Email Accounts\n"
                "Email accounts (mailboxes) at the website's own domain, such as\n"
                "support@example.com\n"
                ", are created and managed in\n"
                "Settings > Domain Manager > Mailboxes\n"
                ". The\n"
                "Emails > Email Accounts\n"
                "menu item opens that tab directly.\n"
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
                "Email Forwarders Not Supported\n"
                "IMPORTANT:\n"
                "The Email Forwarders functionality has been removed from our implementation of cPanel as of June, 2024.\n"
                "Due to the way this functionality works in cPanel, it caused several issues for users including Sendgrid blocking emails.\n"
                "The recommended alternative for using email forwarders is to use an email app that can send and receive emails from multiple accounts:\n"
                "Mozilla Thunderbird, Apple Mail, and Gmail."
            ),
            retrieval_timestamp="2026-10-17T15:57:55.371810+00:00",
            content_hash="hash3",
            metadata={"char_count": 510, "word_count": 80},
        ),
    ]


@pytest.fixture
def sample_retriever(sample_records: list[ArticleRecord]) -> KnowledgeRetriever:
    """Fixture providing an initialized KnowledgeRetriever with sample records."""
    return KnowledgeRetriever(articles=sample_records)


# ---------------------------------------------------------------------------
# Test Helpers & Token Processing
# ---------------------------------------------------------------------------

def test_normalize_token_root():
    """Verify stemming/normalization handles common English suffixes."""
    assert normalize_token_root("campaigns") == "campaign"
    assert normalize_token_root("campaign") == "campaign"
    assert normalize_token_root("forwarders") == "forwarder"
    assert normalize_token_root("managing") == "manag"
    assert normalize_token_root("managed") == "manag"
    assert normalize_token_root("mailboxes") == "mailbox"
    assert normalize_token_root("pass") == "pass"  # preserves short/ss words


def test_extract_candidate_passages():
    """Verify passage extraction collapses broken lines while keeping multi-sentence text."""
    broken_text = (
        "Emails » Email Accounts\n"
        "Email accounts (mailboxes) at the website domain, such as\n"
        "support@example.com\n"
        ", are created and managed in Settings > Domain Manager > Mailboxes.\n"
        "The Emails > Email Accounts menu item opens that tab directly."
    )
    passages = extract_candidate_passages(broken_text, title="Emails » Email Accounts")
    assert len(passages) >= 1
    # Punctuation spacing artifacts should be cleaned and sentences should be cohesive
    assert any("support@example.com, are created and managed in" in p for p in passages)
    assert any("Domain Manager > Mailboxes" in p for p in passages)


# ---------------------------------------------------------------------------
# Test Mock Mode Labeling & Behavior
# ---------------------------------------------------------------------------

def test_mock_mode_flag_and_labeling(sample_retriever: KnowledgeRetriever):
    """Verify mock mode clearly labels answers, flags is_mock, and does not pretend to be Claude."""
    answerer = KnowledgeAnswerer(use_mock=True)
    results = sample_retriever.search("mailboxes", top_k=1)
    ans = answerer.answer("Where are mailboxes managed?", results)

    assert ans.is_mock is True
    assert ans.mode == "mock"
    assert ans.model == "mock-deterministic-grounded"
    assert ans.answer.startswith(MOCK_PREFIX)
    assert "Claude" not in ans.answer
    assert "I am Claude" not in ans.answer

    d = ans.to_dict()
    assert d["is_mock"] is True
    assert d["mode"] == "mock"


def test_is_mock_ai_enabled_default(monkeypatch):
    """Verify is_mock_ai_enabled defaults to True when env var is unset or true."""
    monkeypatch.delenv("USE_MOCK_AI", raising=False)
    assert is_mock_ai_enabled() is True

    monkeypatch.setenv("USE_MOCK_AI", "true")
    assert is_mock_ai_enabled() is True

    monkeypatch.setenv("USE_MOCK_AI", "1")
    assert is_mock_ai_enabled() is True

    monkeypatch.setenv("USE_MOCK_AI", "false")
    assert is_mock_ai_enabled() is False


# ---------------------------------------------------------------------------
# Test Grounded Evidence & Citations
# ---------------------------------------------------------------------------

def test_answer_relevant_evidence_mailbox(sample_retriever: KnowledgeRetriever):
    """Verify mailbox query produces grounded answer with correct citation URL and title."""
    answerer = KnowledgeAnswerer(use_mock=True)
    results = sample_retriever.search("Where are mailboxes managed?", top_k=3)
    ans = answerer.answer("Where are mailboxes managed?", results)

    assert ans.evidence_found is True
    assert len(ans.citations) >= 1
    # Must cite the Email Accounts article
    assert ans.citations[0].title == "Emails » Email Accounts"
    assert "5000558363-emails-email-accounts" in ans.citations[0].url

    # Answer text must contain the grounded location facts from documentation
    assert "Settings > Domain Manager > Mailboxes" in ans.answer


def test_answer_relevant_evidence_mass_email(sample_retriever: KnowledgeRetriever):
    """Verify mass email query produces grounded answer citing compliance and spam rules."""
    answerer = KnowledgeAnswerer(use_mock=True)
    results = sample_retriever.search("Can I send a mass email campaign?", top_k=2)
    ans = answerer.answer("Can I send a mass email campaign?", results)

    assert ans.evidence_found is True
    assert len(ans.citations) == 1
    assert "Mass Email Campaign" in ans.citations[0].title
    assert "5000558357" in ans.citations[0].url
    assert "spam" in ans.answer.lower()


def test_answer_relevant_evidence_forwarders(sample_retriever: KnowledgeRetriever):
    """Verify forwarders query cites removal from cPanel and alternative apps."""
    answerer = KnowledgeAnswerer(use_mock=True)
    results = sample_retriever.search("cPanel email forwarders removed", top_k=2)
    ans = answerer.answer("Are email forwarders supported in cPanel?", results)

    assert ans.evidence_found is True
    assert any("Email Forwarders" in c.title for c in ans.citations)
    assert "removed from our implementation of cpanel" in ans.answer.lower() or "not supported" in ans.answer.lower()


def test_citation_data_structure():
    """Verify Citation dataclass and dictionary conversion."""
    cit = Citation(title="Test Doc", url="https://example.com/test")
    assert cit.title == "Test Doc"
    assert cit.url == "https://example.com/test"
    assert cit.to_dict() == {"title": "Test Doc", "url": "https://example.com/test"}


# ---------------------------------------------------------------------------
# Test Insufficient & Missing Evidence Handling
# ---------------------------------------------------------------------------

def test_insufficient_evidence_unrelated_query(sample_retriever: KnowledgeRetriever):
    """Verify that unrelated queries explicitly state documentation does not establish answer."""
    answerer = KnowledgeAnswerer(use_mock=True)
    # Search for an unrelated concept
    results = sample_retriever.search("quantum computer warp drive", top_k=3)
    ans = answerer.answer("How do I activate the quantum warp drive?", results)

    assert ans.evidence_found is False
    assert INSUFFICIENT_EVIDENCE_PHRASE in ans.answer
    assert ans.citations == []


def test_insufficient_evidence_with_mismatched_results(sample_records: list[ArticleRecord]):
    """Verify that passing irrelevant results explicitly reports insufficient evidence."""
    answerer = KnowledgeAnswerer(use_mock=True)
    # Pass mailbox article for a question about refunds
    mailbox_result = SearchResult(
        title=sample_records[1].title,
        url=sample_records[1].url,
        snippet=sample_records[1].text[:100],
        score=0.1,
        article=sample_records[1],
    )
    ans = answerer.answer("What is the refund policy?", [mailbox_result])

    assert ans.evidence_found is False
    assert INSUFFICIENT_EVIDENCE_PHRASE in ans.answer
    assert ans.citations == []


def test_empty_search_results():
    """Verify empty search results list produces explicit insufficient evidence fallback."""
    answerer = KnowledgeAnswerer(use_mock=True)
    ans = answerer.answer("Any question", [])

    assert ans.evidence_found is False
    assert INSUFFICIENT_EVIDENCE_PHRASE in ans.answer
    assert ans.citations == []


def test_empty_and_whitespace_query(sample_retriever: KnowledgeRetriever):
    """Verify empty or whitespace query produces explicit insufficient evidence fallback."""
    answerer = KnowledgeAnswerer(use_mock=True)
    results = sample_retriever.search("mailboxes", top_k=1)

    ans_empty = answerer.answer("", results)
    assert ans_empty.evidence_found is False
    assert INSUFFICIENT_EVIDENCE_PHRASE in ans_empty.answer

    ans_whitespace = answerer.answer("   ", results)
    assert ans_whitespace.evidence_found is False
    assert INSUFFICIENT_EVIDENCE_PHRASE in ans_whitespace.answer


# ---------------------------------------------------------------------------
# Test Freedom From Hardcoding (Generic Extractor)
# ---------------------------------------------------------------------------

def test_no_hardcoding_with_synthetic_articles():
    """Verify the answerer operates purely on supplied evidence without hardcoding any topic."""
    synthetic_article = ArticleRecord(
        url="https://example.com/widget-calibration-guide",
        title="Widget Calibration Guide",
        text=(
            "Widget calibration is performed in Tools > Hardware Calibrate.\n"
            "Turn the blue dial clockwise to increase sensitivity.\n"
            "If the red indicator flashes, power down the unit immediately."
        ),
        retrieval_timestamp="2026-10-17T00:00:00+00:00",
        content_hash="synthetic_hash",
        metadata={},
    )
    search_result = SearchResult(
        title=synthetic_article.title,
        url=synthetic_article.url,
        snippet="Widget calibration is performed in Tools > Hardware Calibrate.",
        score=6.5,
        article=synthetic_article,
    )

    answerer = KnowledgeAnswerer(use_mock=True)
    ans = answerer.answer("How do I calibrate the widget?", [search_result])

    assert ans.evidence_found is True
    assert "Widget Calibration Guide" in ans.citations[0].title
    assert "https://example.com/widget-calibration-guide" == ans.citations[0].url
    assert "Tools > Hardware Calibrate" in ans.answer

    # Verify a second distinct query against the same synthetic article
    ans_dial = answerer.answer("How do I adjust the blue dial?", [search_result])
    assert ans_dial.evidence_found is True
    assert "blue dial clockwise" in ans_dial.answer


def test_convenience_function_answer_question(sample_retriever: KnowledgeRetriever):
    """Verify functional interface answer_question works identically to KnowledgeAnswerer.answer."""
    results = sample_retriever.search("mass email", top_k=1)
    ans = answer_question("Can I send a mass email campaign?", results, use_mock=True)

    assert isinstance(ans, AnswerResult)
    assert ans.evidence_found is True
    assert len(ans.citations) >= 1


def test_ask_end_to_end_with_sample_json():
    """Verify KnowledgeAnswerer.ask retrieves and answers using the actual sample_articles.json."""
    if not DEFAULT_SAMPLE_DATA_PATH.is_file():
        pytest.skip("sample_articles.json not found")

    answerer = KnowledgeAnswerer(use_mock=True)
    ans = answerer.ask("Where are email accounts and mailboxes managed?")

    assert ans.evidence_found is True
    assert len(ans.citations) >= 1
    assert "Emails » Email Accounts" in ans.citations[0].title
    assert "Settings > Domain Manager > Mailboxes" in ans.answer


# ---------------------------------------------------------------------------
# Test Claude AI Interface (Offline Mocked)
# ---------------------------------------------------------------------------

def test_claude_mode_invokes_anthropic_offline(sample_records: list[ArticleRecord]):
    """Verify Claude mode structures system prompt and user context properly when called."""
    mock_anthropic = MagicMock()
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_text_block = MagicMock()
    mock_text_block.type = "text"
    mock_text_block.text = (
        "Mailboxes are created in Settings > Domain Manager > Mailboxes.\n"
        "Source: Emails » Email Accounts (https://support.brilliantdirectories.com/support/solutions/articles/5000558363-emails-email-accounts)"
    )
    mock_response.content = [mock_text_block]
    mock_client.messages.create.return_value = mock_response
    mock_anthropic.Anthropic.return_value = mock_client

    result_item = SearchResult(
        title=sample_records[1].title,
        url=sample_records[1].url,
        snippet=sample_records[1].text[:120],
        score=5.0,
        article=sample_records[1],
    )

    with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
        answerer = KnowledgeAnswerer(
            use_mock=False,
            model="claude-3-5-sonnet-20241022",
            api_key="sk-ant-test-key-12345",
        )
        ans = answerer.answer("Where are mailboxes managed?", [result_item])

        assert ans.mode == "claude"
        assert ans.is_mock is False
        assert ans.model == "claude-3-5-sonnet-20241022"
        assert "Domain Manager" in ans.answer
        assert len(ans.citations) == 1
        assert ans.citations[0].title == sample_records[1].title

        # Verify Anthropic client was called with correct grounding rules
        mock_client.messages.create.assert_called_once()
        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["model"] == "claude-3-5-sonnet-20241022"
        assert INSUFFICIENT_EVIDENCE_PHRASE in call_kwargs["system"]


def test_claude_mode_missing_api_key_raises(sample_records: list[ArticleRecord]):
    """Verify Claude mode raises ValueError if API key is missing or dummy placeholder."""
    mock_anthropic = MagicMock()

    result_item = SearchResult(
        title=sample_records[0].title,
        url=sample_records[0].url,
        snippet="",
        score=1.0,
        article=sample_records[0],
    )

    with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": ""}):
            answerer = KnowledgeAnswerer(use_mock=False, api_key="")
            with pytest.raises(ValueError, match="Anthropic API key is not configured"):
                answerer.answer("test query", [result_item])


def test_claude_mode_missing_library_raises(sample_records: list[ArticleRecord]):
    """Verify Claude mode raises informative RuntimeError if anthropic package is not installed."""
    result_item = SearchResult(
        title=sample_records[0].title,
        url=sample_records[0].url,
        snippet="",
        score=1.0,
        article=sample_records[0],
    )

    with patch.dict("sys.modules", {"anthropic": None}):
        answerer = KnowledgeAnswerer(use_mock=False, api_key="sk-ant-test")
        with pytest.raises(RuntimeError, match="The 'anthropic' package is required"):
            answerer.answer("test query", [result_item])


# ---------------------------------------------------------------------------
# Test CLI
# ---------------------------------------------------------------------------

def test_cli_positional_query(capsys):
    """Verify CLI accepts positional question and outputs grounded response."""
    exit_code = main(["Where are email accounts and mailboxes managed?", "--mock"])
    assert exit_code == 0
    captured = capsys.readouterr().out
    assert "KNOWLEDGE BRAIN ANSWERER" in captured
    assert "MOCK MODE" in captured
    assert "Settings > Domain Manager > Mailboxes" in captured


def test_cli_query_flag(capsys):
    """Verify CLI accepts --query flag."""
    exit_code = main(["--query", "Can I send a mass email campaign?", "--mock"])
    assert exit_code == 0
    captured = capsys.readouterr().out
    assert "Mass Email Campaign" in captured
    assert "MOCK MODE" in captured


def test_cli_empty_query(capsys):
    """Verify CLI handles empty query without error."""
    exit_code = main(["--query", "", "--mock"])
    assert exit_code == 0
    captured = capsys.readouterr().out
    assert INSUFFICIENT_EVIDENCE_PHRASE in captured
