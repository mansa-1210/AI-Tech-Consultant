"""Knowledge Brain package.

Component of the AI-Tech-Consultant project responsible for knowledge ingestion,
document indexing, and retrieval-augmented generation (RAG) over Brilliant Directories
support documentation.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from knowledge_brain.answerer import (
        AnswerResult,
        Citation,
        KnowledgeAnswerer,
        answer_question,
        generate_claude_answer,
        generate_mock_answer,
    )
    from knowledge_brain.api import (
        app,
        create_app,
    )
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
    from knowledge_brain.retriever import (
        KnowledgeRetriever,
        SearchResult,
        extract_relevant_snippet,
        load_articles_from_json,
        tokenize,
    )

__all__ = [
    "AnswerResult",
    "ArticleRecord",
    "BrilliantDirectoriesCrawler",
    "Citation",
    "KnowledgeAnswerer",
    "KnowledgeRetriever",
    "SearchResult",
    "answer_question",
    "app",
    "clean_extracted_text",
    "compute_content_hash",
    "create_app",
    "extract_article_urls",
    "extract_relevant_snippet",
    "generate_claude_answer",
    "generate_mock_answer",
    "is_article_url",
    "load_articles_from_json",
    "parse_article_html",
    "parse_sitemap_xml",
    "tokenize",
]


def __getattr__(name: str) -> Any:
    """Lazily import symbols from answerer, api, ingestion, or retriever module upon access."""
    if name in __all__:
        if name in ("app", "create_app"):
            import knowledge_brain.api as _api

            return getattr(_api, name)

        if name in (
            "AnswerResult",
            "Citation",
            "KnowledgeAnswerer",
            "answer_question",
            "generate_claude_answer",
            "generate_mock_answer",
        ):
            import knowledge_brain.answerer as _answerer

            return getattr(_answerer, name)

        if name in (
            "KnowledgeRetriever",
            "SearchResult",
            "extract_relevant_snippet",
            "load_articles_from_json",
            "tokenize",
        ):
            import knowledge_brain.retriever as _retriever

            return getattr(_retriever, name)

        import knowledge_brain.ingestion as _ingestion

        return getattr(_ingestion, name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

