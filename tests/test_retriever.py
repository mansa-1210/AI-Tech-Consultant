"""Unit and integration tests for Knowledge Brain retriever module.

Tests local BM25 lexical search, ranking, edge cases (missing files, empty queries,
empty collections), snippet extraction, citation URL preservation, and CLI execution.
Fully offline — zero network calls or paid APIs required.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch
import pytest

from knowledge_brain.ingestion import ArticleRecord
from knowledge_brain.retriever import (
    DEFAULT_SAMPLE_DATA_PATH,
    KnowledgeRetriever,
    SearchResult,
    extract_relevant_snippet,
    load_articles_from_json,
    main,
    query_terms,
    tokenize,
)


# ---------------------------------------------------------------------------
# Sample Mock Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_records() -> list[ArticleRecord]:
    """Provide minimal in-memory article records for fast unit testing."""
    return [
        ArticleRecord(
            url="https://support.brilliantdirectories.com/support/solutions/articles/5000558357-can-i-send-a-mass-email-campaign-using-brilliant-directories-",
            title="Can I Send A Mass Email Campaign Using Brilliant Directories?",
            text=(
                "The built-in email system is designed for emailing active members who have requested to receive emails.\n"
                "A common example would be sending a monthly newsletter to members who have opted in.\n"
                "Sending mass unsolicited emails may be considered spam and damage your domain sending reputation."
            ),
            retrieval_timestamp="2026-10-17T15:57:53.430752+00:00",
            content_hash="hash1",
            metadata={"char_count": 280, "word_count": 45},
        ),
        ArticleRecord(
            url="https://support.brilliantdirectories.com/support/solutions/articles/5000558363-emails-email-accounts",
            title="Emails » Email Accounts",
            text=(
                "Email accounts and mailboxes at the website domain are created in Settings > Domain Manager > Mailboxes.\n"
                "Connect a custom domain to the website before creating mailboxes.\n"
                "Set up Outlook, Apple Mail, or a phone to connect to your custom mailbox."
            ),
            retrieval_timestamp="2026-10-17T15:57:54.305087+00:00",
            content_hash="hash2",
            metadata={"char_count": 255, "word_count": 40},
        ),
        ArticleRecord(
            url="https://support.brilliantdirectories.com/support/solutions/articles/5000558373-email-forwarders",
            title="Email Forwarders",
            text=(
                "The Email Forwarders functionality has been removed from cPanel as of June 2024.\n"
                "Forwarding caused issues including Sendgrid blocking emails and domain reputation harm.\n"
                "We recommend using desktop apps like Mozilla Thunderbird, Apple Mail, or Gmail instead."
            ),
            retrieval_timestamp="2026-10-17T15:57:55.371810+00:00",
            content_hash="hash3",
            metadata={"char_count": 270, "word_count": 42},
        ),
    ]


# ---------------------------------------------------------------------------
# Test Tokenization & Text Utilities
# ---------------------------------------------------------------------------

def test_tokenize_basic():
    """Verify standard text tokenization converts to lowercase and handles punctuation."""
    tokens = tokenize("Can I Send A Mass Email Campaign?")
    assert "mass" in tokens
    assert "email" in tokens
    assert "campaign" in tokens
    # Common stopwords like "can", "i", "a" are filtered out by default
    assert "can" not in tokens
    assert "a" not in tokens


def test_tokenize_all_stopwords_fallback():
    """Verify tokenization preserves tokens if query consists solely of stopwords."""
    tokens = tokenize("how to be")
    assert len(tokens) > 0
    assert "how" in tokens or "to" in tokens


def test_tokenize_empty_and_punctuation():
    """Verify empty strings or punctuation-only inputs produce empty token lists."""
    assert tokenize("") == []
    assert tokenize("   ") == []
    assert tokenize("!@#$%^&*()") == []


def test_query_terms_excludes_generic_knowledge_base_words():
    assert query_terms("What information is available in the knowledge base?") == []


# ---------------------------------------------------------------------------
# Test File Loading & Resilience
# ---------------------------------------------------------------------------

def test_load_articles_from_json_real_sample():
    """Verify loading real sample_articles.json returns valid ArticleRecord instances."""
    if not DEFAULT_SAMPLE_DATA_PATH.is_file():
        pytest.skip("sample_articles.json not present on disk")

    records = load_articles_from_json(DEFAULT_SAMPLE_DATA_PATH)
    assert len(records) >= 3
    for r in records:
        assert isinstance(r, ArticleRecord)
        assert r.url.startswith("https://support.brilliantdirectories.com/")
        assert len(r.title) > 0
        assert len(r.text) > 0


def test_load_articles_from_json_missing_file(tmp_path: Path):
    """Verify loading from non-existent path returns empty list gracefully without throwing."""
    missing_file = tmp_path / "does_not_exist.json"
    records = load_articles_from_json(missing_file)
    assert records == []


def test_load_articles_from_json_corrupted(tmp_path: Path):
    """Verify loading corrupted JSON returns empty list gracefully."""
    corrupted_file = tmp_path / "bad.json"
    corrupted_file.write_text("{ this is not valid json ]", encoding="utf-8")
    records = load_articles_from_json(corrupted_file)
    assert records == []


def test_load_articles_from_json_non_list(tmp_path: Path):
    """Verify loading a JSON object that is not a list returns empty list gracefully."""
    obj_file = tmp_path / "dict.json"
    obj_file.write_text(json.dumps({"key": "value"}), encoding="utf-8")
    records = load_articles_from_json(obj_file)
    assert records == []


# ---------------------------------------------------------------------------
# Test Snippet Extraction
# ---------------------------------------------------------------------------

def test_extract_relevant_snippet_finds_context():
    """Verify snippet extractor pinpoints the sentence containing query keywords."""
    text = (
        "Welcome to the portal. Here is general introduction.\n"
        "Sending mass unsolicited emails may be considered spam and damage your domain sending reputation.\n"
        "Thank you for reading."
    )
    snippet = extract_relevant_snippet(text, query_tokens={"spam", "reputation"}, max_chars=150)
    assert "spam" in snippet.lower()
    assert "reputation" in snippet.lower()
    assert "Welcome to the portal" not in snippet


def test_extract_relevant_snippet_empty_handling():
    """Verify snippet extractor handles empty text and empty queries without error."""
    assert extract_relevant_snippet("", {"query"}) == ""
    snippet = extract_relevant_snippet("Some documentation text.", set())
    assert "Some documentation text." in snippet


# ---------------------------------------------------------------------------
# Test BM25 Lexical Retrieval & Ranking
# ---------------------------------------------------------------------------

def test_retriever_relevant_match_mass_email(sample_records: list[ArticleRecord]):
    """Verify querying 'mass email newsletter' ranks the mass email article #1."""
    retriever = KnowledgeRetriever(articles=sample_records)
    results = retriever.search("mass email newsletter")

    assert len(results) > 0
    top = results[0]
    assert "Mass Email Campaign" in top.title
    assert top.score > 0.0
    assert "newsletter" in top.snippet.lower() or "mass" in top.snippet.lower()


def test_retriever_relevant_match_mailboxes(sample_records: list[ArticleRecord]):
    """Verify querying 'domain manager mailboxes' ranks Email Accounts #1."""
    retriever = KnowledgeRetriever(articles=sample_records)
    results = retriever.search("domain manager mailboxes")

    assert len(results) > 0
    top = results[0]
    assert "Email Accounts" in top.title
    assert top.score > 0.0


def test_retriever_relevant_match_cpanel_forwarders(sample_records: list[ArticleRecord]):
    """Verify querying 'cPanel forwarders Thunderbird' ranks Email Forwarders #1."""
    retriever = KnowledgeRetriever(articles=sample_records)
    results = retriever.search("cPanel forwarders Thunderbird")

    assert len(results) > 0
    top = results[0]
    assert "Email Forwarders" in top.title
    assert top.score > 0.0
    assert "cpanel" in top.snippet.lower() or "forwarders" in top.snippet.lower()


def test_retriever_returns_labeled_visual_evidence():
    article = ArticleRecord(
        url="https://example.test/visual",
        title="Visual setup guide",
        text="Use the settings page to configure the feature.",
        retrieval_timestamp="now",
        content_hash="visual",
        metadata={"images": [{"url": "https://example.test/toggle.png", "vision_description": "The blue publish toggle is enabled.", "description": "The blue publish toggle is enabled."}]},
    )
    result = KnowledgeRetriever(articles=[article]).search("blue publish toggle")[0]
    assert result.visual_evidence == [{"url": "https://example.test/toggle.png", "description": "The blue publish toggle is enabled.", "source": "vision_description"}]


def test_retriever_irrelevant_query(sample_records: list[ArticleRecord]):
    """Verify completely irrelevant queries return an empty result list."""
    retriever = KnowledgeRetriever(articles=sample_records)
    results = retriever.search("quantum gravity black hole astrophysics telescope")
    assert results == []


def test_retriever_rejects_incidental_single_term_match(sample_records: list[ArticleRecord]):
    """A document matching only one broad query term is not sufficient evidence."""
    retriever = KnowledgeRetriever(articles=sample_records)
    results = retriever.search("quantum gravity black hole astrophysics telescope")
    assert results == []


def test_retriever_results_include_query_evidence(sample_records: list[ArticleRecord]):
    retriever = KnowledgeRetriever(articles=sample_records)
    result = retriever.search("mass email newsletter", top_k=1)[0]
    assert {"mass", "email", "newsletter"}.issubset(result.matched_terms)
    assert result.query_coverage == 1.0


def test_retriever_empty_and_whitespace_query(sample_records: list[ArticleRecord]):
    """Verify empty or whitespace-only queries return an empty list gracefully."""
    retriever = KnowledgeRetriever(articles=sample_records)
    assert retriever.search("") == []
    assert retriever.search("   ") == []
    assert retriever.search("\n\t  ") == []
    assert retriever.search("???!!!") == []


def test_retriever_empty_article_collection():
    """Verify retriever handles empty article collection without ZeroDivisionError."""
    retriever = KnowledgeRetriever(articles=[])
    assert len(retriever.articles) == 0
    results = retriever.search("any query text")
    assert results == []


def test_retriever_missing_file_initialization(tmp_path: Path):
    """Verify retriever handles non-existent file path gracefully on init."""
    missing = tmp_path / "not_there.json"
    retriever = KnowledgeRetriever(articles_path=missing)
    assert len(retriever.articles) == 0
    assert retriever.search("test") == []


def test_retriever_citation_url_and_title_preservation(sample_records: list[ArticleRecord]):
    """Verify search results preserve the exact canonical URL and title for citations."""
    retriever = KnowledgeRetriever(articles=sample_records)
    results = retriever.search("cPanel")

    assert len(results) >= 1
    for r in results:
        assert isinstance(r, SearchResult)
        assert r.url.startswith("https://support.brilliantdirectories.com/support/solutions/articles/")
        assert len(r.title) > 0
        assert r.article is not None
        assert r.url == r.article.url
        assert r.title == r.article.title


def test_retriever_top_k_parameter(sample_records: list[ArticleRecord]):
    """Verify top_k parameter strictly constrains returned results count."""
    retriever = KnowledgeRetriever(articles=sample_records)
    # Query with terms appearing across multiple articles (e.g. 'email')
    results_k1 = retriever.search("email", top_k=1)
    results_k2 = retriever.search("email", top_k=2)

    assert len(results_k1) == 1
    assert len(results_k2) == 2


def test_retriever_score_descending_order(sample_records: list[ArticleRecord]):
    """Verify returned results are sorted in descending order of relevance score."""
    retriever = KnowledgeRetriever(articles=sample_records)
    results = retriever.search("email spam accounts", top_k=3)

    assert len(results) >= 2
    for i in range(len(results) - 1):
        assert results[i].score >= results[i + 1].score


def test_retriever_dict_input():
    """Verify retriever accepts raw dictionaries as well as ArticleRecord objects."""
    raw_dicts = [
        {
            "url": "https://example.com/test",
            "title": "Dict Article Title",
            "text": "This is raw dictionary text content mentioning widgets.",
            "retrieval_timestamp": "2026-10-17T00:00:00",
            "content_hash": "abc",
            "metadata": {},
        }
    ]
    retriever = KnowledgeRetriever(articles=raw_dicts)
    results = retriever.search("widgets")
    assert len(results) == 1
    assert results[0].title == "Dict Article Title"
    assert results[0].to_dict()["score"] > 0


# ---------------------------------------------------------------------------
# Test CLI Interface
# ---------------------------------------------------------------------------

def test_cli_positional_query(capsys: pytest.CaptureFixture[str]):
    """Verify CLI with positional query executes successfully."""
    exit_code = main(["mass", "email"])
    assert exit_code == 0
    captured = capsys.readouterr().out
    assert "KNOWLEDGE BRAIN RETRIEVER" in captured
    assert "Mass Email Campaign" in captured


def test_cli_query_flag(capsys: pytest.CaptureFixture[str]):
    """Verify CLI with --query flag executes successfully."""
    exit_code = main(["--query", "forwarders", "--top-k", "1"])
    assert exit_code == 0
    captured = capsys.readouterr().out
    assert "Email Forwarders" in captured


def test_cli_missing_data_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    """Verify CLI handles missing data file gracefully without crashing."""
    fake_data = tmp_path / "missing.json"
    exit_code = main(["--data", str(fake_data), "--query", "email"])
    assert exit_code == 0
    err_out = capsys.readouterr().err
    assert "Data file not found" in err_out


def test_cli_empty_query_flag(capsys: pytest.CaptureFixture[str]):
    """Verify CLI handles whitespace-only query flag without blocking or erroring."""
    exit_code = main(["--query", "   "])
    assert exit_code == 0
    captured = capsys.readouterr().out
    assert "Matches Found:       0" in captured


def test_cli_non_interactive_empty_args(capsys: pytest.CaptureFixture[str]):
    """Verify CLI with no arguments in non-interactive environment returns cleanly."""
    with patch("sys.stdin.isatty", return_value=False):
        exit_code = main([])
        assert exit_code == 0
        captured = capsys.readouterr().out
        assert "Matches Found:       0" in captured
