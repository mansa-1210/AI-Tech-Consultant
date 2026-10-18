"""Knowledge Brain Ingestion Module.

Fetches and parses the official Brilliant Directories support sitemap,
crawls individual support articles respectfully, cleans HTML content, and extracts
structured article records (URL, title, clean text, timestamp, content hash).

NOTE: This is the first component of Task 1 (Knowledge Brain) focusing on data
ingestion and web scraping. It does NOT call Claude or any paid LLM API, and does
not claim to be a complete RAG system yet. Vector indexing and generation will be
added in subsequent steps.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import re
import sys
import time
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin
import xml.etree.ElementTree as ET

from bs4 import BeautifulSoup
import requests

# ---------------------------------------------------------------------------
# Default Configuration Constants
# ---------------------------------------------------------------------------
DEFAULT_SITEMAP_URL = "https://support.brilliantdirectories.com/support/sitemap.xml"
DEFAULT_USER_AGENT = "AI-Tech-Consultant-Bot/1.0 (+https://github.com/AI-Tech-Consultant; respectful documentation crawler)"
DEFAULT_TIMEOUT_SECONDS = 10
DEFAULT_REQUEST_DELAY_SECONDS = 0.5
ARTICLE_URL_INDICATOR = "/support/solutions/articles/"

# Setup module logger
logger = logging.getLogger(__name__)


@dataclass
class ArticleRecord:
    """Structured record representing an ingested documentation article.

    Attributes:
        url: Canonical URL of the documentation article.
        title: Extracted article heading or page title.
        text: Cleaned main body text without navigation, headers, or footers.
        retrieval_timestamp: ISO 8601 UTC timestamp of when the article was fetched.
        content_hash: SHA-256 hexadecimal hash of the cleaned text content.
        metadata: Additional metadata dictionary (character count, word count, etc.).
    """

    url: str
    title: str
    text: str
    retrieval_timestamp: str
    content_hash: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert record to a standard dictionary suitable for JSON serialization."""
        return asdict(self)


def compute_content_hash(text: str) -> str:
    """Compute a SHA-256 hash of the cleaned text to detect content changes.

    Args:
        text: The cleaned text string.

    Returns:
        Hexadecimal SHA-256 string.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def is_article_url(url: str) -> bool:
    """Determine if a given URL is a knowledge base article page.

    Brilliant Directories uses `/support/solutions/articles/` for documentation
    articles, distinguishing them from category listings and search pages.

    Args:
        url: The URL string to test.

    Returns:
        True if the URL matches the article pattern, False otherwise.
    """
    return ARTICLE_URL_INDICATOR in url


def parse_sitemap_xml(xml_content: str | bytes) -> List[str]:
    """Parse sitemap XML content and extract all location URLs.

    Handles XML namespaces transparently, whether standard Sitemaps.org
    schema or without namespace prefix.

    Args:
        xml_content: Raw XML content as string or bytes.

    Returns:
        A list of URL strings found in the sitemap.

    Raises:
        ValueError: If XML parsing fails or content is empty.
    """
    if not xml_content or not str(xml_content).strip():
        logger.warning("Empty sitemap XML content provided.")
        return []

    try:
        if isinstance(xml_content, str):
            xml_bytes = xml_content.encode("utf-8")
        else:
            xml_bytes = xml_content
        root = ET.fromstring(xml_bytes)
    except ET.ParseError as err:
        logger.error("Failed to parse sitemap XML: %s", err)
        raise ValueError(f"Invalid sitemap XML: {err}") from err

    urls: List[str] = []
    # Search for all tags ending with 'loc' to handle any XML namespace transparently
    for elem in root.iter():
        tag = elem.tag
        if tag.endswith("loc") and elem.text:
            cleaned_url = elem.text.strip()
            if cleaned_url:
                urls.append(cleaned_url)

    logger.info("Parsed %d total URLs from sitemap XML.", len(urls))
    return urls


def extract_article_urls(urls: List[str]) -> List[str]:
    """Filter a list of sitemap URLs down to individual article documentation URLs.

    Args:
        urls: List of URLs extracted from the sitemap.

    Returns:
        Filtered list of article URLs.
    """
    article_urls = [u for u in urls if is_article_url(u)]
    logger.info(
        "Extracted %d valid article URLs out of %d total sitemap entries.",
        len(article_urls),
        len(urls),
    )
    return article_urls


def clean_extracted_text(raw_text: str) -> str:
    """Normalize extracted text by cleaning excessive whitespace and blank lines.

    Args:
        raw_text: Text extracted from HTML elements.

    Returns:
        Cleaned and normalized text.
    """
    if not raw_text:
        return ""

    # Normalize horizontal whitespace on individual lines
    cleaned_lines: List[str] = []
    for line in raw_text.splitlines():
        line = re.sub(r"[ \t]+", " ", line).strip()
        cleaned_lines.append(line)

    text = "\n".join(cleaned_lines)
    # Collapse 3 or more consecutive newlines into 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def parse_article_html(html_content: str, url: str = "") -> ArticleRecord:
    """Parse HTML content into an ArticleRecord containing clean text and metadata.

    Removes boilerplate elements (scripts, styles, navbars, footers) and extracts
    the primary article title and main body text.

    Args:
        html_content: Raw HTML text of the page.
        url: The source URL of the page.

    Returns:
        An ArticleRecord containing the parsed and cleaned article data.
    """
    soup = BeautifulSoup(html_content, "html.parser")

    # Remove non-content tags
    unwanted_tags = [
        "script",
        "style",
        "nav",
        "footer",
        "header",
        "noscript",
        "svg",
        "form",
        "iframe",
    ]
    for tag in soup.find_all(unwanted_tags):
        tag.decompose()

    # 1. Extract Page Title
    # Try article heading first, then fall back to <title>, then to a URL slug
    title = ""
    h1_tag = soup.find("h1", class_=lambda c: c and "post-title" in c) or soup.find("h1")
    if h1_tag and h1_tag.get_text(strip=True):
        title = h1_tag.get_text(strip=True)
    elif soup.title and soup.title.string:
        title = soup.title.string.strip()
        # Clean common site suffix like " | Brilliant Directories Documentation"
        title = re.sub(
            r"\s*\|\s*Brilliant Directories.*$", "", title, flags=re.IGNORECASE
        ).strip()
    elif url:
        # Fallback: create title from the URL slug
        slug = url.rstrip("/").split("/")[-1]
        # Remove leading ID numbers if present (e.g. 5000558357-article-title)
        slug_clean = re.sub(r"^\d+-", "", slug).replace("-", " ").title()
        title = slug_clean or "Untitled Documentation Article"
    else:
        title = "Untitled Documentation Article"

    # 2. Extract Main Article Body
    # Look for Brilliant Directories article body container, fallback to main/body
    article_container = (
        soup.find("article", class_=lambda c: c and "article-body" in c)
        or soup.find("article")
        or soup.find("div", class_=lambda c: c and "article-body" in c)
        or soup.find("section", class_=lambda c: c and "main" in c)
        or soup.find("main")
        or soup.body
        or soup
    )

    # Remove article link banner if present (it duplicates the URL inside the body)
    for link_p in article_container.find_all(
        "p", class_=lambda c: c and "the-article-link" in c
    ):
        link_p.decompose()

    # Keep image references with their nearby accessible description.  Alt text is
    # intentionally labelled as source-provided, rather than treated as a vision fact.
    images = []
    for image in article_container.find_all("img"):
        source = image.get("src") or image.get("data-src")
        if source:
            images.append({
                "url": urljoin(url, source),
                "alt_text": image.get("alt", "").strip(),
                "description": image.get("alt", "").strip() or "Screenshot in this article; visual review required.",
            })

    raw_text = article_container.get_text(separator="\n", strip=True)
    clean_text = clean_extracted_text(raw_text)

    # 3. Content hash and timestamp
    timestamp = datetime.now(timezone.utc).isoformat()
    content_hash = compute_content_hash(clean_text)

    # 4. Additional metadata
    metadata = {
        "char_count": len(clean_text),
        "word_count": len(clean_text.split()) if clean_text else 0,
        "line_count": len(clean_text.splitlines()) if clean_text else 0,
        "images": images,
    }

    return ArticleRecord(
        url=url,
        title=title,
        text=clean_text,
        retrieval_timestamp=timestamp,
        content_hash=content_hash,
        metadata=metadata,
    )


class BrilliantDirectoriesCrawler:
    """Respectful crawler for Brilliant Directories Solution Center documentation.

    Fetches the XML sitemap, extracts article URLs, and downloads/parses
    individual pages with timeouts, rate-limiting delays, and explicit headers.
    """

    def __init__(
        self,
        sitemap_url: str = DEFAULT_SITEMAP_URL,
        user_agent: str = DEFAULT_USER_AGENT,
        timeout: int = DEFAULT_TIMEOUT_SECONDS,
        request_delay: float = DEFAULT_REQUEST_DELAY_SECONDS,
        session: Optional[requests.Session] = None,
    ) -> None:
        """Initialize the crawler with respectful settings.

        Args:
            sitemap_url: URL to the documentation XML sitemap.
            user_agent: User-Agent string identifying this bot.
            timeout: Network request timeout in seconds.
            request_delay: Pause duration in seconds between individual page requests.
            session: Optional custom requests.Session (useful for testing and mocking).
        """
        self.sitemap_url = sitemap_url
        self.user_agent = user_agent
        self.timeout = timeout
        self.request_delay = request_delay
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": self.user_agent})

    def fetch_sitemap(self) -> List[str]:
        """Fetch and parse the sitemap XML from the configured URL.

        Returns:
            List of URLs found in the sitemap.

        Raises:
            requests.RequestException: If the sitemap HTTP request fails.
            ValueError: If the XML content cannot be parsed.
        """
        logger.info("Fetching sitemap from: %s", self.sitemap_url)
        try:
            response = self.session.get(self.sitemap_url, timeout=self.timeout)
            response.raise_for_status()
        except requests.exceptions.Timeout as err:
            logger.error("Timeout (%ds) fetching sitemap from %s", self.timeout, self.sitemap_url)
            raise err
        except requests.exceptions.RequestException as err:
            logger.error("Request failed fetching sitemap from %s: %s", self.sitemap_url, err)
            raise err

        return parse_sitemap_xml(response.content)

    def inspect_sitemap(self) -> Dict[str, Any]:
        """Inspect the sitemap and compute descriptive statistics without fetching pages.

        Returns:
            A dictionary with counts and sample URLs.
        """
        all_urls = self.fetch_sitemap()
        article_urls = extract_article_urls(all_urls)
        non_article_urls = [u for u in all_urls if not is_article_url(u)]

        return {
            "sitemap_url": self.sitemap_url,
            "total_urls": len(all_urls),
            "article_urls_count": len(article_urls),
            "non_article_urls_count": len(non_article_urls),
            "sample_article_urls": article_urls[:5],
            "sample_non_article_urls": non_article_urls[:5],
        }

    def fetch_page(self, url: str) -> Optional[str]:
        """Fetch an individual page respectfully, handling errors and non-HTML types.

        Args:
            url: Page URL to fetch.

        Returns:
            Decoded HTML string if successful, or None if skipped/failed.
        """
        try:
            response = self.session.get(url, timeout=self.timeout)
            if response.status_code != 200:
                logger.warning(
                    "Skipping %s: HTTP status %d (%s)",
                    url,
                    response.status_code,
                    response.reason,
                )
                return None

            # Verify that response is HTML
            content_type = response.headers.get("Content-Type", "").lower()
            if "text/html" not in content_type and "application/xhtml+xml" not in content_type:
                logger.warning(
                    "Skipping %s: non-HTML Content-Type ('%s')",
                    url,
                    content_type,
                )
                return None

            return response.text

        except requests.exceptions.Timeout:
            logger.error("Timeout (%ds) fetching article URL: %s", self.timeout, url)
            return None
        except requests.exceptions.RequestException as err:
            logger.error("Network error fetching article URL %s: %s", url, err)
            return None
        except Exception as err:
            logger.error("Unexpected error fetching article URL %s: %s", url, err)
            return None

    def ingest_single_article(self, url: str) -> Optional[ArticleRecord]:
        """Fetch and parse a single article page.

        Args:
            url: Article URL.

        Returns:
            ArticleRecord if successful, or None if failed.
        """
        html_content = self.fetch_page(url)
        if html_content is None:
            return None

        try:
            record = parse_article_html(html_content, url=url)
            logger.info("Successfully ingested '%s' (%s)", record.title, record.url)
            return record
        except Exception as err:
            logger.error("Failed to parse HTML for article %s: %s", url, err)
            return None

    def ingest_articles(
        self,
        limit: Optional[int] = None,
        urls: Optional[List[str]] = None,
        articles_only: bool = True,
    ) -> List[ArticleRecord]:
        """Ingest a batch of documentation articles with rate limiting.

        Args:
            limit: Maximum number of articles to ingest. If None, ingests all.
            urls: Optional explicit list of URLs. If None, URLs are fetched from sitemap.
            articles_only: When True, filters out non-article URLs.

        Returns:
            List of successfully ingested ArticleRecords.
        """
        target_urls = urls
        if target_urls is None:
            all_sitemap_urls = self.fetch_sitemap()
            if articles_only:
                target_urls = extract_article_urls(all_sitemap_urls)
            else:
                target_urls = all_sitemap_urls

        if limit is not None and limit > 0:
            target_urls = target_urls[:limit]

        total = len(target_urls)
        logger.info("Starting ingestion of %d article(s)...", total)

        records: List[ArticleRecord] = []
        for index, url in enumerate(target_urls, start=1):
            logger.info("[%d/%d] Fetching article: %s", index, total, url)
            record = self.ingest_single_article(url)
            if record:
                records.append(record)

            # Apply polite delay between successive requests (skip delay after last request)
            if index < total and self.request_delay > 0:
                time.sleep(self.request_delay)

        logger.info(
            "Ingestion completed. Succeeded: %d, Failed/Skipped: %d, Total Requested: %d",
            len(records),
            total - len(records),
            total,
        )
        return records

    @staticmethod
    def save_records(records: List[ArticleRecord], output_path: str | Path) -> Path:
        """Save a list of ArticleRecords to a formatted JSON file.

        Args:
            records: List of ArticleRecord instances.
            output_path: File destination path.

        Returns:
            Path object of the saved file.
        """
        dest = Path(output_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        data = [r.to_dict() for r in records]
        with open(dest, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info("Saved %d article records to: %s", len(records), dest.resolve())
        return dest


def build_cli_parser() -> argparse.ArgumentParser:
    """Construct command-line argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Knowledge Brain Ingestion Tool (Task 1 Component 1)\n"
            "Fetches and parses Brilliant Directories support documentation sitemap and articles.\n"
            "NOTE: This tool performs data ingestion only; it does not call any AI/LLM APIs."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--inspect-sitemap",
        action="store_true",
        help="Inspect sitemap and display summary statistics without downloading pages.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of articles to crawl and ingest (e.g. --limit 3 for quick testing).",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Crawl every article URL in the sitemap. This can take many minutes; use only after checking the source policy.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save ingested records as a JSON file (e.g. knowledge_brain/data/sample_articles.json).",
    )
    parser.add_argument(
        "--sitemap-url",
        type=str,
        default=DEFAULT_SITEMAP_URL,
        help=f"Custom sitemap URL (defaults to {DEFAULT_SITEMAP_URL}).",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=DEFAULT_REQUEST_DELAY_SECONDS,
        help=f"Polite delay in seconds between requests (defaults to {DEFAULT_REQUEST_DELAY_SECONDS}s).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT_SECONDS,
        help=f"Request timeout in seconds (defaults to {DEFAULT_TIMEOUT_SECONDS}s).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose debug logging.",
    )

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point for the knowledge brain crawler.

    Args:
        argv: Command-line arguments (uses sys.argv[1:] if None).

    Returns:
        Exit code (0 for success, non-zero for error).
    """
    parser = build_cli_parser()
    args = parser.parse_args(argv)

    # A Windows console or redirected stream may otherwise default to a legacy
    # code page and fail while displaying valid article text (for example U+FEFF).
    # JSON persistence is UTF-8; keep console output compatible with it too.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    crawler = BrilliantDirectoriesCrawler(
        sitemap_url=args.sitemap_url,
        timeout=args.timeout,
        request_delay=args.delay,
    )

    try:
        # Mode 1: Inspect Sitemap Only
        if args.inspect_sitemap:
            print("\n" + "=" * 60)
            print(" BRILLIANT DIRECTORIES SITEMAP INSPECTION")
            print("=" * 60)
            stats = crawler.inspect_sitemap()
            print(f"Sitemap URL:           {stats['sitemap_url']}")
            print(f"Total URLs found:      {stats['total_urls']}")
            print(f"Valid Article URLs:    {stats['article_urls_count']}")
            print(f"Non-Article URLs:      {stats['non_article_urls_count']}")
            print("\nSample Article URLs (first 5):")
            for url in stats["sample_article_urls"]:
                print(f"  - {url}")
            print("\nSample Non-Article URLs (first 5):")
            for url in stats["sample_non_article_urls"]:
                print(f"  - {url}")
            print("=" * 60 + "\n")
            return 0

        # Mode 2: Ingest articles
        if args.full and args.limit is not None:
            parser.error("--full and --limit cannot be used together")
        limit = None if args.full else args.limit
        if limit is None and not args.full:
            # Beginner-safe default when running CLI without arguments:
            # Default to limit of 3 with an informative notice
            limit = 3
            print(
                "\n[INFO] No --limit specified. Defaulting to --limit 3 for safe testing.\n"
                "To ingest a different number, specify --limit N (e.g. --limit 5).\n"
            )

        target_label = "all sitemap article URLs" if args.full else f"up to {limit} documentation article(s)"
        print(f"\n[INFO] Ingesting {target_label}...")
        records = crawler.ingest_articles(limit=limit)

        # Persist before printing individual records. A display problem must
        # never discard a successful crawl that the caller explicitly asked us
        # to save.
        if args.output:
            crawler.save_records(records, args.output)
            print(f"\n[OK] Saved {len(records)} records to {args.output}")

        print("\n" + "=" * 60)
        print(f" INGESTION RESULTS: {len(records)} Article(s) Processed")
        print("=" * 60)
        for i, rec in enumerate(records, start=1):
            print(f"\n[{i}] Title:         {rec.title}")
            print(f"    URL:           {rec.url}")
            print(f"    Timestamp:     {rec.retrieval_timestamp}")
            print(f"    Content Hash:  {rec.content_hash[:16]}... (sha256)")
            print(f"    Text Length:   {rec.metadata.get('char_count', 0)} characters / {rec.metadata.get('word_count', 0)} words")
            snippet = rec.text.replace("\n", " ")[:140]
            print(f"    Snippet:       {snippet}...")

        print("=" * 60 + "\n")
        return 0

    except Exception as err:
        logger.error("Ingestion failed: %s", err, exc_info=args.verbose)
        print(f"\n[ERROR] Ingestion failed: {err}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
