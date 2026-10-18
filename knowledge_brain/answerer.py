"""Knowledge Brain Answer Generation Module.

Generates strictly grounded answers to technical questions using retrieved documentation
records from the Knowledge Brain retriever.

Key Features:
- Strict evidence grounding: Answers are synthesized exclusively from retrieved documentation
  passages; no product behaviors are invented or extrapolated.
- Explicit fallback: If documentation is missing, irrelevant, or insufficient, explicitly
  states: "The available documentation does not establish the answer."
- Citations: Returns structured citations containing the article title and original URL.
- Claude primary architecture: Designed for Anthropic Claude (e.g. Claude 3.5 Sonnet) as the
  primary answer generation engine.
- Mock mode support: Respects `USE_MOCK_AI=true` (default in local dev) to run offline,
  deterministic, and fully tested without requiring paid API keys or network calls.
- Clear mock labeling: Mock-generated answers are visibly flagged as mock mode.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import logging
import os
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from knowledge_brain.retriever import (
    DEFAULT_SAMPLE_DATA_PATH,
    KnowledgeRetriever,
    SearchResult,
    tokenize,
)

logger = logging.getLogger(__name__)

# Constants
INSUFFICIENT_EVIDENCE_PHRASE = "The available documentation does not establish the answer."
MOCK_PREFIX = "[MOCK MODE]"
DEFAULT_MOCK_MODEL = "mock-deterministic-grounded"
DEFAULT_CLAUDE_MODEL = "claude-3-5-sonnet-20241022"

# System prompt for Claude to enforce strict grounding and citations
CLAUDE_GROUNDED_SYSTEM_PROMPT = f"""You are an expert AI Technical Consultant specializing in the Brilliant Directories platform.
Your task is to answer technical and product questions accurately, grounded ONLY in the provided documentation context.

STRICT GROUNDING GUIDELINES:
1. Ground your answer exclusively in the supplied documentation excerpts. Do NOT invent, assume, or extrapolate platform behavior.
2. If the documentation is missing, irrelevant, or does not contain sufficient evidence to answer the question with certainty, you MUST explicitly state:
"{INSUFFICIENT_EVIDENCE_PHRASE}"
3. Do not speculate or offer unverified advice.
4. Include clear citations with the article title and original URL for all facts and claims cited in your response.
"""


def is_mock_ai_enabled() -> bool:
    """Check if mock AI mode is enabled via the USE_MOCK_AI environment variable.

    Defaults to True so local development and offline testing do not incur API costs.
    """
    raw_val = os.getenv("USE_MOCK_AI", "true").strip().lower()
    return raw_val in ("true", "1", "yes")


def normalize_token_root(token: str) -> str:
    """Normalize simple English inflectional suffixes (s, es, ed, ing) for robust matching.

    Args:
        token: A single lowercase alphanumeric token.

    Returns:
        Stemmed or normalized token string.
    """
    token = token.lower()
    if len(token) > 4:
        if token.endswith("ing"):
            return token[:-3]
        if token.endswith("ed"):
            return token[:-2]
        if token.endswith("es"):
            return token[:-2]
        if token.endswith("s") and not token.endswith("ss"):
            return token[:-1]
    return token


@dataclass
class Citation:
    """Structured citation referencing an official documentation source.

    Attributes:
        title: Title of the documentation article.
        url: Canonical URL of the article.
    """

    title: str
    url: str

    def to_dict(self) -> Dict[str, str]:
        """Convert citation to dictionary representation."""
        return {
            "title": self.title,
            "url": self.url,
        }


@dataclass
class AnswerResult:
    """Result of an answer generation request.

    Attributes:
        query: Original user question.
        answer: Formatted, grounded answer text.
        citations: List of source citations with article titles and canonical URLs.
        evidence_found: True if sufficient grounded evidence was found, False otherwise.
        mode: Operating mode ('mock' or 'claude').
        model: Identifier of the generation model used.
        raw_response: Raw response string from the underlying generator (optional).
    """

    query: str
    answer: str
    citations: List[Citation] = field(default_factory=list)
    evidence_found: bool = True
    mode: str = "mock"
    model: str = DEFAULT_MOCK_MODEL
    raw_response: Optional[str] = None

    @property
    def is_mock(self) -> bool:
        """Check whether this answer was generated via mock mode."""
        return self.mode == "mock"

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to a standard dictionary representation."""
        return {
            "query": self.query,
            "answer": self.answer,
            "citations": [c.to_dict() for c in self.citations],
            "evidence_found": self.evidence_found,
            "mode": self.mode,
            "model": self.model,
            "is_mock": self.is_mock,
        }


def clean_passage_spacing(text: str) -> str:
    """Clean up punctuation spacing artifacts like ' , ' or ' . ' in extracted text."""
    cleaned = re.sub(r"\s+([,.:;!?])", r"\1", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def extract_candidate_passages(text: str, title: Optional[str] = None) -> List[str]:
    """Split article body text into clean, coherent individual sentences.

    Normalizes irregular whitespace and line-break fragments before splitting
    on sentence boundaries to ensure multi-line sentences remain intact.

    Args:
        text: Raw article body text.
        title: Optional article title to strip if it prefixes the first sentence.

    Returns:
        List of clean candidate sentence strings.
    """
    if not text:
        return []

    # Strip isolated title heading if first line is redundant with title
    raw_lines = [line.strip() for line in text.split("\n") if line.strip()]
    if title and raw_lines:
        clean_title = re.sub(r"\s+", " ", title).strip().lower()
        if raw_lines[0].lower() == clean_title:
            raw_lines = raw_lines[1:]

    # Collapse all remaining whitespace and line fragments into continuous text
    normalized = " ".join(raw_lines).replace("\xa0", " ").strip()
    normalized = re.sub(r"\s+", " ", normalized).strip()
    if not normalized:
        return []

    # Split on sentence terminals followed by space or end-of-string
    raw_sentences = re.split(r"(?<=[.!?])\s+", normalized)
    candidates: List[str] = []

    for s in raw_sentences:
        s_clean = clean_passage_spacing(s)
        # Filter out very short non-informative fragments
        if len(s_clean) >= 20:
            candidates.append(s_clean)

    return candidates


def score_passage_relevance(
    passage: str,
    query_tokens: Sequence[str],
) -> Tuple[float, Set[str]]:
    """Score a candidate passage against normalized query tokens.

    Args:
        passage: Candidate sentence or passage from documentation.
        query_tokens: Extracted tokens from user query.

    Returns:
        Tuple of (relevance_score, matched_query_tokens).
    """
    passage_tokens = tokenize(passage, remove_stopwords=True)
    if not passage_tokens:
        return 0.0, set()

    passage_roots = {normalize_token_root(t) for t in passage_tokens}
    query_roots_map = {normalize_token_root(q): q for q in query_tokens}

    matched_query_tokens: Set[str] = set()
    for root, original_q in query_roots_map.items():
        if root in passage_roots:
            matched_query_tokens.add(original_q)

    if not matched_query_tokens:
        return 0.0, set()

    coverage = len(matched_query_tokens)
    density = coverage / (len(passage_tokens) + 1.0)
    score = (coverage * 4.0) + density
    return score, matched_query_tokens


def generate_mock_answer(
    query: str,
    search_results: Sequence[SearchResult],
) -> AnswerResult:
    """Generate a deterministic, grounded answer in mock mode without calling external APIs.

    Dynamically inspects retrieved documentation passages and extracts matching facts
    based on query terms. Does not hardcode answers to any specific question.

    Args:
        query: The user's question string.
        search_results: Retrieved documentation search results.

    Returns:
        Grounded AnswerResult instance labeled as mock mode.
    """
    clean_query = query.strip() if query else ""
    if not clean_query:
        return AnswerResult(
            query=clean_query,
            answer=f"{MOCK_PREFIX} {INSUFFICIENT_EVIDENCE_PHRASE} The query was empty.",
            citations=[],
            evidence_found=False,
            mode="mock",
            model=DEFAULT_MOCK_MODEL,
        )

    # If no search results provided
    if not search_results:
        return AnswerResult(
            query=clean_query,
            answer=f"{MOCK_PREFIX} {INSUFFICIENT_EVIDENCE_PHRASE} No matching documentation records were found.",
            citations=[],
            evidence_found=False,
            mode="mock",
            model=DEFAULT_MOCK_MODEL,
        )

    # Extract query tokens
    query_tokens = tokenize(clean_query, remove_stopwords=True)
    if not query_tokens:
        query_tokens = tokenize(clean_query, remove_stopwords=False)

    if not query_tokens:
        return AnswerResult(
            query=clean_query,
            answer=f"{MOCK_PREFIX} {INSUFFICIENT_EVIDENCE_PHRASE} Query contains no recognizable keywords.",
            citations=[],
            evidence_found=False,
            mode="mock",
            model=DEFAULT_MOCK_MODEL,
        )

    # Inspect each search result for relevant candidate passages
    all_matched_tokens: Set[str] = set()
    scored_articles: List[Dict[str, Any]] = []

    for res in search_results:
        source_text = res.article.text if (res.article and res.article.text) else res.snippet
        passages = extract_candidate_passages(source_text, title=res.title)
        if not passages and res.snippet:
            passages = [clean_passage_spacing(res.snippet)]

        article_scored_passages: List[Tuple[float, str, Set[str]]] = []
        article_matched_tokens: Set[str] = set()

        for passage in passages:
            score, matched = score_passage_relevance(passage, query_tokens)
            if score > 0.0:
                article_scored_passages.append((score, passage, matched))
                article_matched_tokens.update(matched)

        # Also check title match
        title_score, title_matched = score_passage_relevance(res.title, query_tokens)
        if title_matched:
            article_matched_tokens.update(title_matched)

        if article_scored_passages:
            # Sort passages in this article by score descending
            article_scored_passages.sort(key=lambda x: x[0], reverse=True)
            # Deduplicate similar passages
            selected_passages: List[str] = []
            seen_passage_prefixes: Set[str] = set()
            for p in article_scored_passages:
                prefix = p[1][:40].lower()
                if prefix not in seen_passage_prefixes:
                    seen_passage_prefixes.add(prefix)
                    selected_passages.append(p[1])
                if len(selected_passages) >= 2:
                    break

            all_matched_tokens.update(article_matched_tokens)
            scored_articles.append(
                {
                    "result": res,
                    "best_score": article_scored_passages[0][0],
                    "matched_tokens": article_matched_tokens,
                    "passages": selected_passages,
                }
            )

    # Relevance and sufficiency verification
    coverage_ratio = len(all_matched_tokens) / len(query_tokens) if query_tokens else 0.0

    is_insufficient = False
    if not scored_articles or not all_matched_tokens:
        is_insufficient = True
    elif len(query_tokens) >= 2 and len(all_matched_tokens) < 2 and coverage_ratio < 0.5:
        is_insufficient = True
    elif len(query_tokens) == 1 and len(all_matched_tokens) == 0:
        is_insufficient = True

    if is_insufficient:
        return AnswerResult(
            query=clean_query,
            answer=f"{MOCK_PREFIX} {INSUFFICIENT_EVIDENCE_PHRASE} The retrieved documentation does not provide sufficient information to address this query.",
            citations=[],
            evidence_found=False,
            mode="mock",
            model=DEFAULT_MOCK_MODEL,
        )

    # Rank scored articles by their best passage score
    scored_articles.sort(key=lambda a: a["best_score"], reverse=True)
    top_score = scored_articles[0]["best_score"]
    top_tokens = scored_articles[0]["matched_tokens"]

    # Only include articles that meet the relevance threshold relative to top article
    relevant_articles: List[Dict[str, Any]] = [scored_articles[0]]
    for art in scored_articles[1:]:
        matched_tokens = art["matched_tokens"]
        score = art["best_score"]
        # Skip if this article is only a strict subset of what top article already covers
        if len(matched_tokens) < len(top_tokens) and matched_tokens.issubset(top_tokens):
            continue
        # Keep if it brings distinct tokens with strong independent relevance
        if score >= (top_score * 0.6) and len(matched_tokens) >= 2:
            relevant_articles.append(art)

    # Collect citations and assemble answer
    citations: List[Citation] = []
    seen_urls: Set[str] = set()

    for item in relevant_articles:
        res = item["result"]
        if res.url not in seen_urls:
            seen_urls.add(res.url)
            citations.append(Citation(title=res.title, url=res.url))

    # Format synthesized grounded text
    if len(relevant_articles) == 1:
        single = relevant_articles[0]
        res = single["result"]
        passage_body = " ".join(single["passages"])
        lines = [
            f"{MOCK_PREFIX} Based on \"{res.title}\":",
            passage_body,
        ]
    else:
        lines = [
            f"{MOCK_PREFIX} Based on the provided documentation:",
            "",
        ]
        for item in relevant_articles:
            res = item["result"]
            lines.append(f"From \"{res.title}\":")
            for p in item["passages"]:
                lines.append(f"- {p}")
            lines.append("")

    lines.append("\nCitations:")
    for cit in citations:
        lines.append(f"- {cit.title} ({cit.url})")

    answer_text = "\n".join(lines).strip()

    return AnswerResult(
        query=clean_query,
        answer=answer_text,
        citations=citations,
        evidence_found=True,
        mode="mock",
        model=DEFAULT_MOCK_MODEL,
    )


def generate_claude_answer(
    query: str,
    search_results: Sequence[SearchResult],
    model: str = DEFAULT_CLAUDE_MODEL,
    api_key: Optional[str] = None,
) -> AnswerResult:
    """Generate an answer using Anthropic's Claude API.

    Args:
        query: User question.
        search_results: Retrieved search results to use as context.
        model: Claude model name to use.
        api_key: Anthropic API key (optional; defaults to ANTHROPIC_API_KEY env var).

    Returns:
        AnswerResult containing Claude's answer and citations.

    Raises:
        RuntimeError: If anthropic library is not installed.
        ValueError: If anthropic API key is missing or invalid.
    """
    clean_query = query.strip() if query else ""
    if not clean_query:
        return AnswerResult(
            query=clean_query,
            answer=f"{INSUFFICIENT_EVIDENCE_PHRASE} The query was empty.",
            citations=[],
            evidence_found=False,
            mode="claude",
            model=model,
        )

    if not search_results:
        return AnswerResult(
            query=clean_query,
            answer=f"{INSUFFICIENT_EVIDENCE_PHRASE} No matching documentation records were found.",
            citations=[],
            evidence_found=False,
            mode="claude",
            model=model,
        )

    # Dynamic import of anthropic package
    try:
        import anthropic
    except ImportError as err:
        raise RuntimeError(
            "The 'anthropic' package is required when USE_MOCK_AI=false. "
            "Install it with 'pip install anthropic' or set USE_MOCK_AI=true in your .env file."
        ) from err

    resolved_key = api_key or os.getenv("ANTHROPIC_API_KEY")
    if not resolved_key or resolved_key.strip() in ("", "your_anthropic_api_key_here"):
        raise ValueError(
            "Anthropic API key is not configured. Please set ANTHROPIC_API_KEY in your .env file "
            "or pass api_key to KnowledgeAnswerer, or set USE_MOCK_AI=true to use mock mode."
        )

    # Build context from search results
    context_blocks: List[str] = []
    citations: List[Citation] = []
    seen_urls: Set[str] = set()

    for idx, res in enumerate(search_results, start=1):
        content = res.article.text if (res.article and res.article.text) else res.snippet
        context_blocks.append(
            f"[Source {idx}]\n"
            f"Title: {res.title}\n"
            f"URL: {res.url}\n"
            f"Content:\n{content}\n"
        )
        if res.url not in seen_urls:
            seen_urls.add(res.url)
            citations.append(Citation(title=res.title, url=res.url))

    user_message = (
        f"DOCUMENTATION CONTEXT:\n\n{''.join(context_blocks)}\n\n"
        f"USER QUESTION:\n{clean_query}\n\n"
        "Provide a grounded answer following the strict grounding guidelines."
    )

    client = anthropic.Anthropic(api_key=resolved_key)
    response = client.messages.create(
        model=model,
        max_tokens=1024,
        system=CLAUDE_GROUNDED_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )

    answer_text = ""
    for block in response.content:
        if getattr(block, "type", "") == "text":
            answer_text += block.text

    # Check whether Claude concluded evidence is insufficient
    evidence_found = INSUFFICIENT_EVIDENCE_PHRASE not in answer_text
    active_citations = citations if evidence_found else []

    return AnswerResult(
        query=clean_query,
        answer=answer_text,
        citations=active_citations,
        evidence_found=evidence_found,
        mode="claude",
        model=model,
        raw_response=answer_text,
    )


def generate_gemini_answer(
    query: str,
    search_results: Sequence[SearchResult],
    model: str = "gemini-2.5-flash",
    api_key: Optional[str] = None,
) -> AnswerResult:
    """Generate a documentation-grounded answer using Google's Gemini API."""
    clean_query = query.strip() if query else ""

    if not clean_query or not search_results:
        reason = "The query was empty." if not clean_query else "No matching documentation records were found."
        return AnswerResult(
            query=clean_query,
            answer=f"{INSUFFICIENT_EVIDENCE_PHRASE} {reason}",
            citations=[],
            evidence_found=False,
            mode="gemini",
            model=model,
        )

    resolved_key = api_key or os.getenv("GEMINI_API_KEY")
    if not resolved_key or resolved_key.strip() == "PASTE_YOUR_KEY_HERE":
        raise ValueError("Set GEMINI_API_KEY in your .env file.")

    try:
        from google import genai
    except ImportError as err:
        raise RuntimeError(
            "Install the Gemini SDK with: pip install google-genai"
        ) from err

    context_blocks = []
    citations = []
    seen_urls = set()

    for index, result in enumerate(search_results, start=1):
        content = (
            result.article.text
            if result.article and result.article.text
            else result.snippet
        )
        context_blocks.append(
            f"[Source {index}]\n"
            f"Title: {result.title}\n"
            f"URL: {result.url}\n"
            f"Content:\n{content}\n"
        )
        if result.url not in seen_urls:
            seen_urls.add(result.url)
            citations.append(Citation(title=result.title, url=result.url))

    prompt = f"""
You are an AI technical consultant answering questions about Brilliant Directories.

Answer the question using ONLY the documentation sources below.
Do not invent facts or infer unsupported product behavior.
If the sources do not establish the answer, reply with this exact sentence:
"{INSUFFICIENT_EVIDENCE_PHRASE}"

Cite supporting source titles and URLs in your answer.
Do not claim that a source supports something it does not say.

DOCUMENTATION SOURCES:
{''.join(context_blocks)}

USER QUESTION:
{clean_query}
"""

    client = genai.Client(api_key=resolved_key)
    response = client.models.generate_content(
        model=model,
        contents=prompt,
    )
    answer_text = (response.text or "").strip()

    if not answer_text:
        answer_text = INSUFFICIENT_EVIDENCE_PHRASE

    evidence_found = INSUFFICIENT_EVIDENCE_PHRASE.lower() not in answer_text.lower()

    return AnswerResult(
        query=clean_query,
        answer=answer_text,
        citations=citations if evidence_found else [],
        evidence_found=evidence_found,
        mode="gemini",
        model=model,
        raw_response=answer_text,
    )


class KnowledgeAnswerer:
    """Knowledge Brain Answer Generation Engine.

    Synthesizes grounded answers strictly from retrieved documentation.
    Supports deterministic offline mock mode plus configured Anthropic Claude
    or Google Gemini providers.
    """

    def __init__(
        self,
        use_mock: Optional[bool] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        provider: Optional[str] = None,
    ) -> None:
        """Initialize the KnowledgeAnswerer.

        Args:
            use_mock: Explicitly enable or disable mock mode. If None, respects USE_MOCK_AI env var.
            model: Provider model name.
            api_key: Provider API key.
            provider: Optional explicit provider override ("claude" or "gemini").
        """
        self.use_mock = is_mock_ai_enabled() if use_mock is None else use_mock
        self.provider = (provider or os.getenv("AI_PROVIDER", "claude")).strip().lower()
        if self.provider not in ("claude", "gemini"):
            raise ValueError("AI_PROVIDER must be 'claude' or 'gemini'.")
        if self.provider == "gemini":
            self.model = model or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
            self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        else:
            self.model = model or os.getenv("CLAUDE_PRIMARY_MODEL", DEFAULT_CLAUDE_MODEL)
            self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")

    @property
    def mode(self) -> str:
        """Return the actual execution mode reflected in API responses."""
        return "mock" if self.use_mock else self.provider

    def answer(
        self,
        query: str,
        search_results: Sequence[SearchResult],
    ) -> AnswerResult:
        """Generate a grounded answer for a question using retrieved documentation search results.

        Args:
            query: The user's question string.
            search_results: Sequence of SearchResult objects from the retriever.

        Returns:
            AnswerResult with grounded answer text, citations, and metadata.
        """
        if self.use_mock:
            return generate_mock_answer(query, search_results)
        if self.provider == "gemini":
            return generate_gemini_answer(
                query=query,
                search_results=search_results,
                model=self.model,
                api_key=self.api_key,
            )
        return generate_claude_answer(
            query=query,
            search_results=search_results,
            model=self.model,
            api_key=self.api_key,
        )

    def ask(
        self,
        query: str,
        retriever: Optional[KnowledgeRetriever] = None,
        top_k: int = 3,
    ) -> AnswerResult:
        """Convenience method to retrieve documentation and generate an answer in one call.

        Args:
            query: The user's question string.
            retriever: Optional retriever instance. If None, initializes default KnowledgeRetriever.
            top_k: Number of search results to retrieve.

        Returns:
            AnswerResult instance.
        """
        if retriever is None:
            retriever = KnowledgeRetriever()
        results = retriever.search(query, top_k=top_k)
        return self.answer(query, results)


def answer_question(
    query: str,
    search_results: Sequence[SearchResult],
    use_mock: Optional[bool] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
) -> AnswerResult:
    """Convenience functional interface to answer a query using search results.

    Args:
        query: User question string.
        search_results: Retrieved documentation search results.
        use_mock: Optional flag to force mock mode.
        model: Optional model identifier.
        api_key: Optional Anthropic API key.

    Returns:
        AnswerResult object.
    """
    answerer = KnowledgeAnswerer(use_mock=use_mock, model=model, api_key=api_key)
    return answerer.answer(query, search_results)


def build_cli_parser() -> argparse.ArgumentParser:
    """Build command-line parser for the answerer tool."""
    parser = argparse.ArgumentParser(
        description=(
            "Knowledge Brain Answerer CLI (Task 1 Component 3)\n"
            "Answers user questions grounded strictly in retrieved Brilliant Directories documentation.\n"
            "Respects USE_MOCK_AI=true for zero-cost, deterministic local testing."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "query",
        nargs="*",
        help="Question to answer (e.g. 'Where are mailboxes managed?').",
    )
    parser.add_argument(
        "--query",
        "-q",
        dest="query_flag",
        type=str,
        default=None,
        help="Alternative flag to specify question.",
    )
    parser.add_argument(
        "--data",
        "-d",
        type=str,
        default=str(DEFAULT_SAMPLE_DATA_PATH),
        help=f"Path to JSON articles file (defaults to {DEFAULT_SAMPLE_DATA_PATH}).",
    )
    parser.add_argument(
        "--top-k",
        "-k",
        type=int,
        default=3,
        help="Number of documentation articles to retrieve (default: 3).",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        default=None,
        help="Force mock mode (bypasses Anthropic API).",
    )
    parser.add_argument(
        "--no-mock",
        action="store_false",
        dest="mock",
        help="Force real Claude API mode (requires ANTHROPIC_API_KEY).",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable debug logging output.",
    )

    return parser


def format_cli_output(result: AnswerResult) -> str:
    """Format AnswerResult nicely for command line display."""
    mode_label = (
        "MOCK MODE (USE_MOCK_AI=true)"
        if result.is_mock
        else f"{'GEMINI' if result.mode == 'gemini' else 'CLAUDE'} API ({result.model})"
    )
    evidence_label = (
        f"Sufficient evidence found ({len(result.citations)} source(s))"
        if result.evidence_found
        else "Insufficient or missing evidence"
    )

    lines: List[str] = [
        "\n" + "=" * 68,
        " KNOWLEDGE BRAIN ANSWERER -- GROUNDED DOCUMENTATION ASSISTANT",
        "=" * 68,
        f"Question:    \"{result.query}\"",
        f"Mode:        {mode_label}",
        f"Model:       {result.model}",
        f"Evidence:    {evidence_label}",
        "-" * 68,
        "",
        "[Answer]",
        result.answer,
        "",
        "-" * 68,
    ]

    if result.is_mock:
        lines.append("[NOTE] Mock mode active. Zero paid API calls made.")
    else:
        provider_name = "Google Gemini" if result.mode == "gemini" else "Anthropic Claude"
        lines.append(f"[NOTE] Generated via {provider_name} model {result.model}.")

    lines.append("=" * 68 + "\n")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point for the answerer.

    Args:
        argv: Command line arguments list (uses sys.argv[1:] if None).

    Returns:
        Exit code (0 for success).
    """
    parser = build_cli_parser()
    args = parser.parse_args(argv)

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Determine query string from positional arguments or flag
    query_parts = args.query if args.query else []
    query_str = args.query_flag if args.query_flag else " ".join(query_parts)

    data_path = Path(args.data)
    if not data_path.is_file():
        print(f"\n[WARNING] Data file not found: {data_path.resolve()}", file=sys.stderr)
        print("Please run the ingestion crawler first or check the path:", file=sys.stderr)
        print("  python -m knowledge_brain.ingestion --limit 3 --output knowledge_brain/data/sample_articles.json\n", file=sys.stderr)

    retriever = KnowledgeRetriever(articles_path=data_path)
    answerer = KnowledgeAnswerer(use_mock=args.mock)

    # If --query was provided (even if empty string) or stdin is not a TTY, don't block
    if not query_str.strip():
        if args.query_flag is not None or not sys.stdin.isatty():
            # Non-interactive or explicit empty query
            res = answerer.answer(query_str, [])
            print(format_cli_output(res))
            return 0

        # Interactive loop
        print("\n" + "=" * 68)
        print(" KNOWLEDGE BRAIN ANSWERER -- INTERACTIVE MODE")
        print("=" * 68)
        print(f"Loaded {len(retriever.articles)} article(s) from {data_path}")
        print(f"Mode: {'MOCK MODE (offline)' if answerer.use_mock else 'CLAUDE API'}")
        print("Enter a question about Brilliant Directories (or 'exit' to quit):")

        try:
            while True:
                user_input = input("\nQuestion > ").strip()
                if not user_input or user_input.lower() in ("exit", "quit", "q"):
                    print("Exiting answerer.")
                    break
                result = answerer.ask(user_input, retriever=retriever, top_k=args.top_k)
                print(format_cli_output(result))
        except (KeyboardInterrupt, EOFError):
            print("\nSession ended.")
        return 0

    result = answerer.ask(query_str, retriever=retriever, top_k=args.top_k)
    print(format_cli_output(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
