"""Knowledge Brain Retrieval Module.

Provides local, offline lexical search (Okapi BM25) over ingested documentation
records. Enables finding relevant articles for user queries without calling any
external embedding services or paid LLM APIs.

NOTE: This is the second component of Task 1 (Knowledge Brain). It provides
transparent keyword-based retrieval. Semantic vector embeddings (e.g. ChromaDB)
and Claude-powered response generation will be integrated in subsequent steps.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import json
import logging
import math
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Optional, Sequence, Set, Union

from knowledge_brain.ingestion import ArticleRecord

# Module logger
logger = logging.getLogger(__name__)

# Default path to sample articles relative to project root
DEFAULT_SAMPLE_DATA_PATH = Path("knowledge_brain") / "data" / "sample_articles.json"

# Tokenizer pattern matching alphanumeric words of length >= 2
WORD_REGEX = re.compile(r"\b[a-zA-Z0-9_\-\.]{2,}\b")

# Standard English stopwords to filter out low-information words
DEFAULT_STOPWORDS: Set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can", "can't", "cannot", "could",
    "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down",
    "during", "each", "few", "for", "from", "further", "had", "hadn't", "has",
    "hasn't", "have", "haven't", "having", "he", "her", "here", "hers", "herself",
    "him", "himself", "his", "how", "i", "if", "in", "into", "is", "isn't", "it",
    "it's", "its", "itself", "just", "me", "more", "most", "my", "myself", "no",
    "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought", "our",
    "ours", "ourselves", "out", "over", "own", "same", "she", "should", "shouldn't",
    "so", "some", "such", "than", "that", "the", "their", "theirs", "them",
    "themselves", "then", "there", "these", "they", "this", "those", "through",
    "to", "too", "under", "until", "up", "very", "was", "wasn't", "we", "were",
    "weren't", "what", "when", "where", "which", "while", "who", "whom", "why",
    "with", "won't", "would", "wouldn't", "you", "your", "yours", "yourself",
}


@dataclass
class SearchResult:
    """Structured representation of a search match.

    Attributes:
        title: Title of the matched documentation article.
        url: Canonical URL of the documentation article for citations.
        snippet: Excerpt from the article text highlighting relevant query context.
        score: Lexical relevance score (higher indicates greater relevance).
        article: Optional reference to the underlying ArticleRecord.
    """

    title: str
    url: str
    snippet: str
    score: float
    article: Optional[ArticleRecord] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert search result to a standard dictionary."""
        return {
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
            "score": round(self.score, 4),
        }


def tokenize(text: str, remove_stopwords: bool = True) -> List[str]:
    """Extract normalized lowercase tokens from text.

    Args:
        text: Source text string to tokenize.
        remove_stopwords: If True, filters out standard low-information words.

    Returns:
        List of lowercase token strings.
    """
    if not text:
        return []

    tokens = [t.lower() for t in WORD_REGEX.findall(text)]
    if remove_stopwords:
        filtered = [t for t in tokens if t not in DEFAULT_STOPWORDS]
        # If all tokens were filtered out (e.g. query was entirely stopwords like "how to"),
        # fall back to preserving all tokens so search doesn't fail silently.
        return filtered if filtered else tokens
    return tokens


def load_articles_from_json(file_path: Union[str, Path]) -> List[ArticleRecord]:
    """Safely load article records from a JSON file.

    Handles non-existent files, empty files, or corrupted data gracefully
    without raising fatal errors.

    Args:
        file_path: Path to the JSON articles file.

    Returns:
        List of ArticleRecord instances. Returns empty list if file is missing or invalid.
    """
    path = Path(file_path)
    if not path.is_file():
        logger.warning("Articles file not found: %s", path.resolve())
        return []

    try:
        with open(path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
    except (json.JSONDecodeError, OSError) as err:
        logger.warning("Failed to parse JSON articles from %s: %s", path, err)
        return []

    if not isinstance(raw_data, list):
        logger.warning("Expected JSON array of articles in %s, got %s", path, type(raw_data).__name__)
        return []

    records: List[ArticleRecord] = []
    for item in raw_data:
        if isinstance(item, dict):
            # Extract standard fields with sensible defaults
            rec = ArticleRecord(
                url=str(item.get("url", "")),
                title=str(item.get("title", "")),
                text=str(item.get("text", "")),
                retrieval_timestamp=str(item.get("retrieval_timestamp", "")),
                content_hash=str(item.get("content_hash", "")),
                metadata=item.get("metadata", {}) if isinstance(item.get("metadata"), dict) else {},
            )
            records.append(rec)

    logger.info("Successfully loaded %d article record(s) from %s", len(records), path)
    return records


def extract_relevant_snippet(
    text: str,
    query_tokens: Set[str],
    max_chars: int = 240,
) -> str:
    """Extract a concise, context-aware text snippet around matching query tokens.

    Normalizes whitespace across broken lines, splits into sentences, ranks
    each sentence by query term density, and returns a cohesive context window.

    Args:
        text: Full article body text.
        query_tokens: Set of normalized query tokens.
        max_chars: Maximum character length for the snippet.

    Returns:
        A cleaned, single-line text snippet with ellipses if truncated.
    """
    if not text:
        return ""

    # Normalize fragmented lines and irregular whitespace into continuous text
    normalized_text = re.sub(r"\s+", " ", text).strip()
    if not normalized_text:
        return ""

    # Split into candidate sentences
    sentences = [
        s.strip()
        for s in re.split(r"(?<=[.!?])\s+", normalized_text)
        if s.strip()
    ]
    if not sentences:
        sentences = [normalized_text]

    if not query_tokens:
        first = sentences[0]
        return first[:max_chars] + ("..." if len(first) > max_chars else "")

    best_idx = 0
    best_score = -1.0

    for idx, candidate in enumerate(sentences):
        cand_tokens = set(tokenize(candidate, remove_stopwords=False))
        overlap = cand_tokens.intersection(query_tokens)
        if not overlap:
            continue

        # Score based on distinct matches and density
        score = (len(overlap) * 3.0) + (len(overlap) / (len(cand_tokens) + 1.0))
        if score > best_score:
            best_score = score
            best_idx = idx

    chosen = sentences[best_idx]
    if len(chosen) <= max_chars:
        # If chosen sentence is relatively short, append next sentence(s) to provide context
        curr = chosen
        for next_idx in range(best_idx + 1, len(sentences)):
            nxt = sentences[next_idx]
            if len(curr) + len(nxt) + 1 <= max_chars:
                curr = f"{curr} {nxt}"
            else:
                remaining = max_chars - len(curr) - 4
                if remaining > 30:
                    curr = f"{curr} {nxt[:remaining]}..."
                break
        return curr

    # If the single matched sentence exceeds max_chars, window around first match
    lower_chosen = chosen.lower()
    first_match_pos = len(chosen)
    for token in query_tokens:
        pos = lower_chosen.find(token)
        if 0 <= pos < first_match_pos:
            first_match_pos = pos

    if first_match_pos < max_chars:
        return chosen[:max_chars].rstrip() + "..."

    start = max(0, first_match_pos - 40)
    end = min(len(chosen), start + max_chars)
    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(chosen) else ""
    return prefix + chosen[start:end].strip() + suffix


class KnowledgeRetriever:
    """Offline Lexical Search Engine for Documentation Articles.

    Implements Okapi BM25 scoring with title boosting and snippet extraction.
    Requires no paid APIs, no external embedding services, and runs purely offline
    using Python standard library constructs.

    Okapi BM25 Formula:
        Score(D, Q) = sum_{q in Q} IDF(q) * [ (f(q, D) * (k1 + 1)) / (f(q, D) + k1 * (1 - b + b * (|D| / avgdl))) ]

    Where:
        - f(q, D) is the term frequency of q in D (with title matches boosted).
        - IDF(q) = ln( (N - n(q) + 0.5) / (n(q) + 0.5) + 1 ) [Lucene BM25 formulation].
        - k1 calibrates term frequency saturation (default 1.5).
        - b calibrates document length normalization (default 0.75).
    """

    def __init__(
        self,
        articles: Optional[Sequence[Union[ArticleRecord, Dict[str, Any]]]] = None,
        articles_path: Optional[Union[str, Path]] = None,
        k1: float = 1.5,
        b: float = 0.75,
        title_boost: float = 2.5,
    ) -> None:
        """Initialize the retriever with either an in-memory collection or a file path.

        Args:
            articles: In-memory sequence of ArticleRecord instances or article dicts.
            articles_path: Path to a JSON file containing saved articles.
            k1: BM25 term frequency saturation parameter.
            b: BM25 document length normalization parameter.
            title_boost: Multiplier applied to terms found in article titles.
        """
        self.k1 = k1
        self.b = b
        self.title_boost = title_boost

        self.articles: List[ArticleRecord] = []
        self._doc_lengths: List[float] = []
        self._doc_term_freqs: List[Counter] = []
        self._doc_freqs: Counter = Counter()
        self._avg_doc_length: float = 0.0

        if articles is not None:
            self._load_records(articles)
        elif articles_path is not None:
            self.load_from_file(articles_path)
        elif DEFAULT_SAMPLE_DATA_PATH.is_file():
            self.load_from_file(DEFAULT_SAMPLE_DATA_PATH)
        else:
            logger.info("Retriever initialized with 0 articles (no articles or file provided).")

    def _load_records(self, items: Sequence[Union[ArticleRecord, Dict[str, Any]]]) -> None:
        """Normalize and index a sequence of records."""
        normalized: List[ArticleRecord] = []
        for item in items:
            if isinstance(item, ArticleRecord):
                normalized.append(item)
            elif isinstance(item, dict):
                normalized.append(
                    ArticleRecord(
                        url=str(item.get("url", "")),
                        title=str(item.get("title", "")),
                        text=str(item.get("text", "")),
                        retrieval_timestamp=str(item.get("retrieval_timestamp", "")),
                        content_hash=str(item.get("content_hash", "")),
                        metadata=item.get("metadata", {}) if isinstance(item.get("metadata"), dict) else {},
                    )
                )

        self.articles = normalized
        self._build_index()

    def load_from_file(self, file_path: Union[str, Path]) -> None:
        """Load and index article records from a JSON file.

        Args:
            file_path: Path to the JSON articles file.
        """
        records = load_articles_from_json(file_path)
        self.articles = records
        self._build_index()

    def _build_index(self) -> None:
        """Construct the BM25 inverted index from the current articles collection."""
        self._doc_lengths.clear()
        self._doc_term_freqs.clear()
        self._doc_freqs.clear()

        total_length = 0.0
        for article in self.articles:
            title_tokens = tokenize(article.title, remove_stopwords=True)
            body_tokens = tokenize(article.text, remove_stopwords=True)

            # Compute effective document length with title boosting
            effective_len = len(body_tokens) + (len(title_tokens) * self.title_boost)
            self._doc_lengths.append(effective_len)
            total_length += effective_len

            # Aggregate weighted term frequencies
            tf: Counter = Counter(body_tokens)
            for t_token in title_tokens:
                tf[t_token] += self.title_boost
            self._doc_term_freqs.append(tf)

            # Update document frequency for each unique term
            unique_terms = set(body_tokens).union(title_tokens)
            for term in unique_terms:
                self._doc_freqs[term] += 1

        doc_count = len(self.articles)
        self._avg_doc_length = (total_length / doc_count) if doc_count > 0 else 0.0
        logger.debug(
            "Built BM25 index over %d documents (avg_len=%.1f, vocab_size=%d)",
            doc_count,
            self._avg_doc_length,
            len(self._doc_freqs),
        )

    def _compute_idf(self, term: str) -> float:
        """Compute Lucene-style non-negative BM25 Inverse Document Frequency (IDF).

        Formula: ln( (N - n + 0.5) / (n + 0.5) + 1 )
        """
        n = self._doc_freqs.get(term, 0)
        n_docs = len(self.articles)
        if n == 0 or n_docs == 0:
            return 0.0
        return math.log(((n_docs - n + 0.5) / (n + 0.5)) + 1.0)

    def search(
        self,
        query: str,
        top_k: int = 3,
        min_score: float = 0.0,
    ) -> List[SearchResult]:
        """Search the indexed articles for relevant matches.

        Args:
            query: The user's search query string.
            top_k: Maximum number of results to return.
            min_score: Minimum relevance threshold for returning results.

        Returns:
            List of SearchResult objects sorted by descending relevance score.
            Returns an empty list if query is empty, articles collection is empty,
            or no articles score above min_score.
        """
        # Handle empty collection or empty query gracefully
        if not self.articles or not query or not query.strip():
            return []

        query_tokens = tokenize(query, remove_stopwords=True)
        if not query_tokens:
            return []

        doc_count = len(self.articles)
        if doc_count == 0 or self._avg_doc_length == 0.0:
            return []

        query_term_set = set(query_tokens)
        scores = [0.0] * doc_count

        for term in query_term_set:
            idf = self._compute_idf(term)
            if idf <= 0.0:
                continue

            for idx in range(doc_count):
                tf = self._doc_term_freqs[idx].get(term, 0.0)
                if tf <= 0.0:
                    continue

                doc_len = self._doc_lengths[idx]
                numerator = tf * (self.k1 + 1.0)
                denominator = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / self._avg_doc_length))

                scores[idx] += idf * (numerator / denominator)

        # Filter and rank matches
        ranked_results: List[SearchResult] = []
        for idx, score in enumerate(scores):
            if score > min_score:
                article = self.articles[idx]
                snippet = extract_relevant_snippet(
                    text=article.text,
                    query_tokens=query_term_set,
                    max_chars=220,
                )
                ranked_results.append(
                    SearchResult(
                        title=article.title,
                        url=article.url,
                        snippet=snippet,
                        score=score,
                        article=article,
                    )
                )

        # Sort descending by score
        ranked_results.sort(key=lambda r: r.score, reverse=True)
        return ranked_results[:top_k]


def build_cli_parser() -> argparse.ArgumentParser:
    """Build command-line parser for the retriever tool."""
    parser = argparse.ArgumentParser(
        description=(
            "Knowledge Brain Retriever CLI (Task 1 Component 2)\n"
            "Performs transparent, offline lexical search (BM25) over Brilliant Directories articles.\n"
            "NOTE: Does not call Claude or any paid LLM API. Embeddings and generation will be added in future steps."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "query",
        nargs="*",
        help="Search query text (e.g. 'mass email' or 'email forwarders').",
    )
    parser.add_argument(
        "--query",
        "-q",
        dest="query_flag",
        type=str,
        default=None,
        help="Alternative flag to specify search query.",
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
        help="Maximum number of relevant results to return (default: 3).",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable debug logging output.",
    )

    return parser


def format_search_results(query: str, results: List[SearchResult], total_articles: int) -> str:
    """Format search results nicely for command-line display."""
    lines: List[str] = [
        "\n" + "=" * 65,
        " KNOWLEDGE BRAIN RETRIEVER -- LOCAL LEXICAL SEARCH",
        "=" * 65,
        f"Query:               \"{query}\"",
        f"Articles Indexed:    {total_articles}",
        f"Matches Found:       {len(results)}",
        "-" * 65,
    ]

    if not results:
        lines.append("No matching documentation articles found for this query.")
        lines.append("Try rephrasing with different keywords (e.g. 'email', 'domain', 'cPanel').")
    else:
        for rank, res in enumerate(results, start=1):
            lines.append(f"\n[Rank {rank}] {res.title}")
            lines.append(f"  Relevance Score:  {res.score:.4f}")
            lines.append(f"  Citation URL:     {res.url}")
            lines.append(f"  Relevant Snippet: {res.snippet}")

    lines.append("\n" + "-" * 65)
    lines.append("[NOTE] This is an offline BM25 lexical retriever (Task 1 Component 2).")
    lines.append("       Dense vector embeddings and Claude AI generation will be added next.")
    lines.append("=" * 65 + "\n")

    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point for the retriever.

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
        print("Please verify the path or run the ingestion crawler first:", file=sys.stderr)
        print("  python -m knowledge_brain.ingestion --limit 3 --output knowledge_brain/data/sample_articles.json\n", file=sys.stderr)

    retriever = KnowledgeRetriever(articles_path=data_path)

    # If --query was provided (even if empty string) or stdin is not a TTY, don't block on interactive input
    if not query_str.strip():
        if args.query_flag is not None or not sys.stdin.isatty():
            # Non-interactive or explicit empty query
            print(format_search_results(query_str, [], len(retriever.articles)))
            return 0

        # Interactive mode for interactive terminal sessions
        print("\n" + "=" * 65)
        print(" KNOWLEDGE BRAIN RETRIEVER -- INTERACTIVE SEARCH")
        print("=" * 65)
        print(f"Loaded {len(retriever.articles)} article(s) from {data_path}")
        print("Enter a search question/keyword (or 'exit' to quit):")

        try:
            while True:
                user_input = input("\nSearch > ").strip()
                if not user_input or user_input.lower() in ("exit", "quit", "q"):
                    print("Exiting search.")
                    break
                results = retriever.search(user_input, top_k=args.top_k)
                print(format_search_results(user_input, results, len(retriever.articles)))
        except (KeyboardInterrupt, EOFError):
            print("\nSearch ended.")
        return 0

    results = retriever.search(query_str, top_k=args.top_k)
    print(format_search_results(query_str, results, len(retriever.articles)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
