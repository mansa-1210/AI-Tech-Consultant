"""Structure-preserving document chunking for Knowledge Brain ingestion."""
from __future__ import annotations

from dataclasses import replace
import re
from typing import Iterable, List

from knowledge_brain.ingestion import ArticleRecord, compute_content_hash


def chunk_article(article: ArticleRecord, max_chars: int = 1_600, overlap_chars: int = 180) -> List[ArticleRecord]:
    """Split an article at headings/paragraphs while retaining source metadata."""
    blocks = [part.strip() for part in re.split(r"\n\s*\n|(?=^#{1,3}\s)", article.text, flags=re.MULTILINE) if part.strip()]
    chunks: List[str] = []
    current = ""
    for block in blocks or [article.text]:
        if current and len(current) + len(block) + 2 > max_chars:
            chunks.append(current)
            current = current[-overlap_chars:] + "\n" + block if overlap_chars else block
        else:
            current = (current + "\n\n" + block).strip()
    if current:
        chunks.append(current)
    result: List[ArticleRecord] = []
    for index, text in enumerate(chunks):
        metadata = dict(article.metadata)
        metadata.update({"parent_url": article.url, "chunk_index": index, "chunk_count": len(chunks)})
        result.append(replace(article, text=text, content_hash=compute_content_hash(text), metadata=metadata))
    return result


def chunk_articles(articles: Iterable[ArticleRecord]) -> List[ArticleRecord]:
    return [chunk for article in articles for chunk in chunk_article(article)]
