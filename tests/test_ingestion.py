"""Unit and integration tests for Knowledge Brain ingestion module.

All HTTP requests and network calls are mocked. Tests do NOT require
live internet access or paid APIs.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

from knowledge_brain.ingestion import (
    ArticleRecord,
    BrilliantDirectoriesCrawler,
    clean_extracted_text,
    compute_content_hash,
    extract_article_urls,
    is_article_url,
    parse_article_html,
    parse_sitemap_xml,
)

# ---------------------------------------------------------------------------
# Sample Mock Fixtures
# ---------------------------------------------------------------------------

SAMPLE_SITEMAP_XML = """<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url>
    <loc>https://support.brilliantdirectories.com/support/home</loc>
    <lastmod>2026-10-11T08:00:54+00:00</lastmod>
  </url>
  <url>
    <loc>https://support.brilliantdirectories.com/support/solutions</loc>
    <lastmod>2026-10-11T08:00:54+00:00</lastmod>
  </url>
  <url>
    <loc>https://support.brilliantdirectories.com/support/solutions/folders/5000264973</loc>
    <lastmod>2026-10-11T08:00:54+00:00</lastmod>
  </url>
  <url>
    <loc>https://support.brilliantdirectories.com/support/solutions/articles/5000558357-can-i-send-a-mass-email-campaign-using-brilliant-directories-</loc>
    <lastmod>2026-10-11T08:00:54+00:00</lastmod>
  </url>
  <url>
    <loc>https://support.brilliantdirectories.com/support/solutions/articles/5000558363-emails-email-accounts</loc>
    <lastmod>2026-10-11T08:00:54+00:00</lastmod>
  </url>
</urlset>
"""

SAMPLE_ARTICLE_HTML = """<!DOCTYPE html>
<html>
<head>
  <title>Can I Send A Mass Email Campaign Using Brilliant Directories? | Brilliant Directories Documentation</title>
  <script>console.log("tracking");</script>
  <style>body { font-family: sans-serif; }</style>
</head>
<body>
  <nav><a href="/home">Home</a></nav>
  <section class="main content">
    <article class="article-body">
      <h1 class="post-title heading">Can I Send A Mass Email Campaign Using Brilliant Directories?</h1>
      <p class="the-article-link"><small>Link: https://support.brilliantdirectories.com/support/solutions/articles/5000558357</small></p>
      <p>The built-in email system is designed for emailing active members.</p>
      <p>Sending unsolicited bulk emails can lead to domain blacklisting.</p>
    </article>
  </section>
  <footer><p>&copy; 2026 Brilliant Directories</p></footer>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Test Sitemap XML Parsing & URL Filtering
# ---------------------------------------------------------------------------

def test_parse_sitemap_xml_with_namespace():
    """Verify sitemap XML with namespaces parses all locations accurately."""
    urls = parse_sitemap_xml(SAMPLE_SITEMAP_XML)
    assert len(urls) == 5
    assert "https://support.brilliantdirectories.com/support/home" in urls
    assert (
        "https://support.brilliantdirectories.com/support/solutions/articles/5000558357-can-i-send-a-mass-email-campaign-using-brilliant-directories-"
        in urls
    )


def test_parse_sitemap_xml_without_namespace():
    """Verify sitemap XML without namespace parses correctly."""
    xml_no_ns = """<urlset>
        <url><loc>https://example.com/article1</loc></url>
        <url><loc>https://example.com/article2</loc></url>
    </urlset>"""
    urls = parse_sitemap_xml(xml_no_ns)
    assert urls == ["https://example.com/article1", "https://example.com/article2"]


def test_parse_sitemap_xml_empty_and_invalid():
    """Verify empty content returns empty list and invalid XML raises ValueError."""
    assert parse_sitemap_xml("") == []
    assert parse_sitemap_xml("   ") == []

    with pytest.raises(ValueError, match="Invalid sitemap XML"):
        parse_sitemap_xml("<urlset><unclosed></urlset>")


def test_is_article_url():
    """Verify distinguishing article pages from categories, folders, and homepage."""
    article_url = "https://support.brilliantdirectories.com/support/solutions/articles/5000558357-email"
    non_article_folder = "https://support.brilliantdirectories.com/support/solutions/folders/5000264973"
    non_article_category = "https://support.brilliantdirectories.com/support/solutions/12000001358"
    non_article_home = "https://support.brilliantdirectories.com/support/home"

    assert is_article_url(article_url) is True
    assert is_article_url(non_article_folder) is False
    assert is_article_url(non_article_category) is False
    assert is_article_url(non_article_home) is False


def test_extract_article_urls():
    """Verify filtering sitemap URLs down to only article documentation pages."""
    urls = parse_sitemap_xml(SAMPLE_SITEMAP_XML)
    article_urls = extract_article_urls(urls)
    assert len(article_urls) == 2
    for u in article_urls:
        assert "/support/solutions/articles/" in u


# ---------------------------------------------------------------------------
# Test Content Extraction, Cleaning, and Hashing
# ---------------------------------------------------------------------------

def test_clean_extracted_text():
    """Verify whitespace normalization and blank line collapsing."""
    raw = "  First line with   spaces.   \n\n\n\n   Second line.  \n   "
    cleaned = clean_extracted_text(raw)
    assert cleaned == "First line with spaces.\n\nSecond line."
    assert clean_extracted_text("") == ""


def test_compute_content_hash():
    """Verify deterministic SHA-256 calculation."""
    text1 = "This is documentation content."
    text2 = "This is documentation content."
    text3 = "Different content."

    hash1 = compute_content_hash(text1)
    hash2 = compute_content_hash(text2)
    hash3 = compute_content_hash(text3)

    assert hash1 == hash2
    assert hash1 != hash3
    assert len(hash1) == 64  # SHA-256 hex string length


def test_parse_article_html_standard():
    """Verify standard article parsing extracts title, clean text, hash, and metadata."""
    url = "https://support.brilliantdirectories.com/support/solutions/articles/5000558357-can-i-send-a-mass-email-campaign-using-brilliant-directories-"
    record = parse_article_html(SAMPLE_ARTICLE_HTML, url=url)

    assert isinstance(record, ArticleRecord)
    assert record.url == url
    assert record.title == "Can I Send A Mass Email Campaign Using Brilliant Directories?"
    assert "The built-in email system is designed for emailing active members." in record.text
    assert "Sending unsolicited bulk emails can lead to domain blacklisting." in record.text
    # Boilerplate should be removed
    assert "console.log" not in record.text
    assert "sans-serif" not in record.text
    assert "Home" not in record.text
    assert "2026 Brilliant Directories" not in record.text

    # Content hash and timestamp checks
    assert len(record.content_hash) == 64
    assert record.retrieval_timestamp.endswith("+00:00") or "T" in record.retrieval_timestamp
    assert record.metadata["word_count"] > 0
    assert record.metadata["char_count"] > 0


def test_parse_article_html_fallback_title():
    """Verify fallback title extraction when H1 is missing."""
    html_no_h1 = """<html>
        <head><title>Fallback Page Title | Brilliant Directories Documentation</title></head>
        <body><main><p>Important guide content here.</p></main></body>
    </html>"""
    record = parse_article_html(html_no_h1, url="https://example.com/test")
    assert record.title == "Fallback Page Title"
    assert "Important guide content here." in record.text


def test_parse_article_html_slug_title_fallback():
    """Verify fallback title generation from URL slug when both H1 and title are absent."""
    html_bare = "<html><body><p>Content without headings.</p></body></html>"
    record = parse_article_html(
        html_bare,
        url="https://support.brilliantdirectories.com/support/solutions/articles/12345-how-to-setup-domain",
    )
    assert "How To Setup Domain" in record.title
    assert "Content without headings." in record.text


# ---------------------------------------------------------------------------
# Test Crawler HTTP Interactions (Mocked Responses)
# ---------------------------------------------------------------------------

def test_fetch_page_success():
    """Verify fetch_page returns HTML text for 200 OK responses with text/html."""
    mock_session = MagicMock()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {"Content-Type": "text/html; charset=utf-8"}
    mock_response.text = SAMPLE_ARTICLE_HTML
    mock_session.get.return_value = mock_response

    crawler = BrilliantDirectoriesCrawler(session=mock_session)
    result = crawler.fetch_page("https://example.com/article")

    assert result == SAMPLE_ARTICLE_HTML
    mock_session.get.assert_called_once_with("https://example.com/article", timeout=10)


def test_fetch_page_skips_non_html():
    """Verify crawler gracefully skips non-HTML content types (e.g. PDF, images)."""
    mock_session = MagicMock()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {"Content-Type": "application/pdf"}
    mock_session.get.return_value = mock_response

    crawler = BrilliantDirectoriesCrawler(session=mock_session)
    result = crawler.fetch_page("https://example.com/document.pdf")

    assert result is None


def test_fetch_page_handles_http_errors():
    """Verify crawler handles 404 or 500 status codes gracefully."""
    mock_session = MagicMock()
    mock_response = MagicMock()
    mock_response.status_code = 404
    mock_response.reason = "Not Found"
    mock_session.get.return_value = mock_response

    crawler = BrilliantDirectoriesCrawler(session=mock_session)
    result = crawler.fetch_page("https://example.com/missing-article")

    assert result is None


def test_fetch_page_handles_timeout():
    """Verify crawler handles request timeouts gracefully."""
    mock_session = MagicMock()
    mock_session.get.side_effect = requests.exceptions.Timeout("Connection timed out")

    crawler = BrilliantDirectoriesCrawler(session=mock_session)
    result = crawler.fetch_page("https://example.com/slow-article")

    assert result is None


def test_fetch_page_handles_network_error():
    """Verify crawler handles network connection errors gracefully."""
    mock_session = MagicMock()
    mock_session.get.side_effect = requests.exceptions.ConnectionError("Failed to resolve")

    crawler = BrilliantDirectoriesCrawler(session=mock_session)
    result = crawler.fetch_page("https://example.com/unreachable-article")

    assert result is None


def test_inspect_sitemap():
    """Verify inspect_sitemap calculates accurate metrics without fetching pages."""
    mock_session = MagicMock()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.content = SAMPLE_SITEMAP_XML.encode("utf-8")
    mock_session.get.return_value = mock_response

    crawler = BrilliantDirectoriesCrawler(session=mock_session)
    stats = crawler.inspect_sitemap()

    assert stats["total_urls"] == 5
    assert stats["article_urls_count"] == 2
    assert stats["non_article_urls_count"] == 3
    assert len(stats["sample_article_urls"]) == 2


def test_ingest_articles_limit_and_error_recovery():
    """Verify crawler respects limit argument and recovers from failed individual pages."""
    mock_session = MagicMock()

    # First call: fetch sitemap
    sitemap_response = MagicMock()
    sitemap_response.status_code = 200
    sitemap_response.content = SAMPLE_SITEMAP_XML.encode("utf-8")

    # Second call (article 1): 200 OK
    art1_response = MagicMock()
    art1_response.status_code = 200
    art1_response.headers = {"Content-Type": "text/html"}
    art1_response.text = SAMPLE_ARTICLE_HTML

    # Third call (article 2): 500 Server Error
    art2_response = MagicMock()
    art2_response.status_code = 500
    art2_response.reason = "Internal Server Error"

    mock_session.get.side_effect = [
        sitemap_response,
        art1_response,
        art2_response,
    ]

    crawler = BrilliantDirectoriesCrawler(session=mock_session, request_delay=0.0)

    # There are 2 article URLs in SAMPLE_SITEMAP_XML
    records = crawler.ingest_articles(limit=2)

    # 1 succeeded, 1 failed gracefully
    assert len(records) == 1
    assert records[0].title == "Can I Send A Mass Email Campaign Using Brilliant Directories?"


def test_save_records(tmp_path: Path):
    """Verify records can be serialized and saved to JSON format."""
    record = ArticleRecord(
        url="https://example.com/article",
        title="Sample Article",
        text="Sample text content.",
        retrieval_timestamp="2026-10-17T00:00:00+00:00",
        content_hash="abc123hash",
        metadata={"char_count": 20},
    )

    dest_file = tmp_path / "subdir" / "articles.json"
    saved_path = BrilliantDirectoriesCrawler.save_records([record], dest_file)

    assert saved_path.exists()
    with open(saved_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert len(data) == 1
    assert data[0]["url"] == "https://example.com/article"
    assert data[0]["title"] == "Sample Article"
    assert data[0]["content_hash"] == "abc123hash"


def test_save_records_preserves_utf8_text(tmp_path: Path):
    """UTF-8 article text must round-trip on Windows and other platforms."""
    record = ArticleRecord(
        url="https://example.com/unicode",
        title="Unicode \ufeff title",
        text="A Unicode \ufeff marker must not prevent saving.",
        retrieval_timestamp="2026-10-18T00:00:00+00:00",
        content_hash="unicode-hash",
    )
    saved_path = BrilliantDirectoriesCrawler.save_records([record], tmp_path / "unicode.json")
    data = json.loads(saved_path.read_text(encoding="utf-8"))
    assert data[0]["text"] == record.text


# ---------------------------------------------------------------------------
# Test CLI Functions
# ---------------------------------------------------------------------------

@patch("knowledge_brain.ingestion.BrilliantDirectoriesCrawler.inspect_sitemap")
def test_cli_inspect_sitemap(mock_inspect, capsys):
    """Verify CLI --inspect-sitemap flag executes and displays stats."""
    mock_inspect.return_value = {
        "sitemap_url": "https://example.com/sitemap.xml",
        "total_urls": 10,
        "article_urls_count": 7,
        "non_article_urls_count": 3,
        "sample_article_urls": ["https://example.com/art1"],
        "sample_non_article_urls": ["https://example.com/home"],
    }

    from knowledge_brain.ingestion import main

    exit_code = main(["--inspect-sitemap"])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "BRILLIANT DIRECTORIES SITEMAP INSPECTION" in captured.out
    assert "Valid Article URLs:    7" in captured.out


@patch("knowledge_brain.ingestion.BrilliantDirectoriesCrawler.ingest_articles")
def test_cli_ingest_with_limit(mock_ingest, capsys, tmp_path: Path):
    """Verify CLI saves UTF-8 output before displaying an article summary."""
    mock_record = ArticleRecord(
        url="https://example.com/art1",
        title="Article \ufeff One",
        text="Content of article \ufeff one.",
        retrieval_timestamp="2026-10-17T00:00:00+00:00",
        content_hash="mockedhash123456789",
        metadata={"char_count": 23, "word_count": 4},
    )
    mock_ingest.return_value = [mock_record]

    from knowledge_brain.ingestion import main

    output_path = tmp_path / "crawl-output.json"
    exit_code = main(["--limit", "1", "--output", str(output_path)])
    assert exit_code == 0
    saved = json.loads(output_path.read_text(encoding="utf-8"))
    assert saved[0]["text"] == mock_record.text
    captured = capsys.readouterr()
    assert "INGESTION RESULTS: 1 Article(s) Processed" in captured.out
    assert "[OK] Saved 1 records" in captured.out
