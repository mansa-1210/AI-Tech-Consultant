# Knowledge Brain: Component 1 — Ingestion Pipeline

## Overview

> **Current provider note:** the answer service supports `AI_PROVIDER=claude` and `AI_PROVIDER=gemini`. It returns the actual execution mode and model in `/health` and `/ask`; a Gemini call is never labelled as mock or Claude.

> **Dataset status (2026-10-18):** the corrected full crawl saved 1,535 valid records to `knowledge_brain/data/full_articles.json` from 1,540 sitemap URLs; five URLs were skipped. The crawler saves requested output before previews and uses UTF-8-safe console output, preventing the prior Windows display failure from discarding results.

The **Knowledge Brain Ingestion Pipeline** is the first foundational component of Task 1 in the `AI-Tech-Consultant` system. Its purpose is to reliably and respectfully crawl the official Brilliant Directories Solution Center documentation, filter out non-article URLs, strip web boilerplate, and produce normalized, structured JSON article records ready for future chunking, vector indexing, and RAG retrieval.

> [!NOTE]
> This component handles ingestion and preprocessing only. It does **not** make any calls to Anthropic Claude or paid LLMs, and does not claim to be a complete RAG system yet.

---

## Architecture & Data Flow

```text
Sitemap XML URL
(https://support.brilliantdirectories.com/support/sitemap.xml)
       │
       ▼
[parse_sitemap_xml] ───► Extract all <loc> URLs (handles namespaces transparently)
       │
       ▼
[extract_article_urls] ──► Filter URLs containing '/support/solutions/articles/'
       │
       ▼
[BrilliantDirectoriesCrawler]
  ├── Respectful rate-limiting pause (default: 0.5s)
  ├── Explicit User-Agent header
  ├── Timeout enforcement (default: 10s)
  └── Error recovery (skips 404, 500, non-HTML gracefully)
       │
       ▼
[parse_article_html]
  ├── Decompose <script>, <style>, <nav>, <footer>, <header>, etc.
  ├── Extract primary title (from <h1>, <title>, or URL slug)
  ├── Extract main article text (<article class="article-body"> or fallbacks)
  ├── Normalize whitespace & blank lines
  ├── Generate UTC retrieval timestamp
  └── Calculate SHA-256 content hash
       │
       ▼
[ArticleRecord] ───► JSON export / Downstream ChromaDB ingestion
```

---

## Data Model: `ArticleRecord`

Each ingested article is captured as an `ArticleRecord` instance with the following schema:

| Field | Type | Description |
|---|---|---|
| `url` | `str` | Full canonical URL of the documentation article |
| `title` | `str` | Extracted title of the article |
| `text` | `str` | Cleaned, normalized main article body text |
| `retrieval_timestamp` | `str` | ISO 8601 UTC timestamp of retrieval (e.g. `2026-10-17T15:50:11.837619+00:00`) |
| `content_hash` | `str` | SHA-256 hexadecimal digest of the text content |
| `metadata` | `dict` | Supplementary metadata (e.g. `char_count`, `word_count`, `line_count`) |

---

## Usage Guide

### CLI Commands

All commands can be run via the Python module:

1. **Inspect Sitemap**:
   ```bash
   python -m knowledge_brain.ingestion --inspect-sitemap
   ```
   Outputs total URLs discovered, article counts, non-article counts, and samples of both.

2. **Ingest Limited Number of Articles for Safe Testing**:
   ```bash
   python -m knowledge_brain.ingestion --limit 3
   ```

3. **Save Ingestion Output to JSON**:
   ```bash
   python -m knowledge_brain.ingestion --limit 5 --output knowledge_brain/data/sample_articles.json
   ```

4. **Full Crawl** (time-consuming; run once only when ready):
   ```powershell
   .\.venv\Scripts\python.exe -m knowledge_brain.ingestion --full --output knowledge_brain/data/full_articles.json --delay 0.5
   ```
   The explicit `--output` path is required for persistence. Verify the JSON record count after it completes.

5. **Custom Tuning**:
   ```bash
   python -m knowledge_brain.ingestion --limit 10 --delay 1.0 --timeout 15 -v
   ```

---

## Testing Strategy

Tests are fully isolated and do not make live network calls:
- All HTTP responses are mocked using `unittest.mock.MagicMock` and `unittest.mock.patch`.
- Covers XML sitemap edge cases (with/without namespaces, invalid XML, empty XML).
- Covers HTML cleaning, fallback titles, and SHA-256 content hashing.
- Covers HTTP error handling (404, 500, timeouts, connection errors, non-HTML Content-Type).
- Covers batch ingestion, limit enforcement, and JSON serialization.

### Running Tests

```bash
pytest -v
```
or
```bash
pytest knowledge_brain/tests/ -v
```

---

# Knowledge Brain: Component 2 — Local Lexical Retriever

## Overview

The **Knowledge Brain Local Retriever** (`knowledge_brain/retriever.py`) is the second foundational component of Task 1 in the `AI-Tech-Consultant` system. It provides a local, offline lexical search engine built on the proven **Okapi BM25** ranking algorithm with title boosting and context-aware text snippet extraction.

> [!NOTE]
> This component provides transparent, offline keyword retrieval using only Python's standard library. It does **not** call Anthropic Claude or any paid LLM API, and does not require external embedding models. Semantic dense vector embeddings (e.g. ChromaDB) and Claude-powered generation will be integrated in subsequent milestones.

---

## Architecture & Scoring Mechanism

```text
User Question / Query
       │
       ▼
[tokenize & normalize] ──► Lowercase tokens, strip punctuation, filter stopwords
       │
       ▼
[KnowledgeRetriever]
  ├── Okapi BM25 Scoring (Lucene IDF formulation)
  │     Score(D, Q) = sum_{q in Q} IDF(q) * [ (f(q, D) * (k1 + 1)) / (f(q, D) + k1 * (1 - b + b * (|D| / avgdl))) ]
  │
  ├── Title Weight Multiplier (boost = 2.5x)
  ├── Graceful handling of missing files, empty queries, empty collections
  └── [extract_relevant_snippet] ──► Context window around highest keyword density
       │
       ▼
[SearchResult] (title, url, snippet, score, article)
```

### Data Model: `SearchResult`

| Field | Type | Description |
|---|---|---|
| `title` | `str` | Title of the matched documentation article |
| `url` | `str` | Full canonical URL of the documentation article for source citation |
| `snippet` | `str` | Context-aware excerpt highlighting query keyword matches |
| `score` | `float` | Computed BM25 relevance score (higher is more relevant) |
| `article` | `ArticleRecord \| None` | Reference to original ingested article record |

---

## Usage Guide

### PowerShell CLI Commands

1. **Search with Positional Query**:
   ```powershell
   python -m knowledge_brain.retriever "Can I send mass emails?"
   ```

2. **Search with Explicit Flag and Top-K**:
   ```powershell
   python -m knowledge_brain.retriever --query "cPanel forwarders" --top-k 1
   ```

3. **Specify Custom Data File**:
   ```powershell
   python -m knowledge_brain.retriever --data knowledge_brain/data/sample_articles.json "mailboxes"
   ```

4. **Interactive Search Mode**:
   ```powershell
   python -m knowledge_brain.retriever
   ```

### Python API Usage

```python
from knowledge_brain import KnowledgeRetriever, load_articles_from_json

# Initialize retriever using saved sample articles
retriever = KnowledgeRetriever(articles_path="knowledge_brain/data/sample_articles.json")

# Perform search
results = retriever.search("mass email campaign", top_k=3)

for res in results:
    print(f"Title:   {res.title}")
    print(f"URL:     {res.url}")
    print(f"Score:   {res.score:.4f}")
    print(f"Snippet: {res.snippet}\n")
```

---

## Testing Strategy

Tests are fully offline and verify:
- Tokenization, punctuation stripping, and stopword filtering (with all-stopword fallback).
- Safe JSON loading (missing files, invalid/corrupt JSON, non-list objects).
- Sentence-level snippet extraction with context preservation and clean truncation.
- Accurate BM25 ranking across all 3 sample Brilliant Directories articles.
- Edge cases: empty queries (`""`, `"   "`, `"???"`), empty article collections, and missing files.
- Strict preservation of canonical URLs and titles for source citation integrity.
- CLI argument parsing, flags, and non-interactive environment handling.


# Knowledge Brain: Component 3 — Grounded Answer Generator

## Overview

The **Knowledge Brain Answer Generator** (`knowledge_brain/answerer.py`) is the third component of Task 1 in the `AI-Tech-Consultant` system. It accepts a user question and retrieved search results from the BM25 retriever, producing answers grounded strictly and exclusively in the supplied documentation.

> [!NOTE]
> Anthropic Claude is the intended primary answer-generation model. However, to enable fast, deterministic, zero-cost offline development and testing, mock generation mode is enabled by default via `USE_MOCK_AI=true`. Mock responses are clearly labeled as `[MOCK MODE]` and do not pretend to be generated by Claude.

---

## Key Principles & Guardrails

1. **Strict Evidence Grounding**:
   Answers are synthesized exclusively from the provided documentation text and snippets. The system does not invent platform capabilities, speculate, or extrapolate beyond the provided text.

2. **Explicit Fallback on Missing or Insufficient Evidence**:
   If evidence is missing, irrelevant, or insufficient to answer the query with certainty, the answerer explicitly states:
   > *"The available documentation does not establish the answer."*

3. **Citations with Article Title & Canonical URL**:
   Every answer includes structured citations with the source article's title and its canonical documentation URL.

4. **Zero Hardcoded Answers**:
   The mock generator inspects candidate passages dynamically, normalizing morphology and scoring passage relevance across any article text. It operates without hardcoded strings for mailboxes or any other topic.

5. **Claude Primary Model Architecture**:
   When `USE_MOCK_AI=false` and an Anthropic API key is configured, generation is powered by Anthropic's Claude 3.5 Sonnet using a strict grounding system prompt.

---

## Data Models

### `Citation`

| Field | Type | Description |
|---|---|---|
| `title` | `str` | Title of the cited documentation article |
| `url` | `str` | Canonical URL of the documentation article |

### `AnswerResult`

| Field | Type | Description |
|---|---|---|
| `query` | `str` | The user's original question |
| `answer` | `str` | Grounded answer text (prefixed with `[MOCK MODE]` when in mock mode) |
| `citations` | `list[Citation]` | Source citations containing titles and URLs |
| `evidence_found` | `bool` | True if sufficient grounded evidence was found, False otherwise |
| `mode` | `str` | Operating mode: `'mock'` or `'claude'` |
| `model` | `str` | Model identifier (e.g. `'mock-deterministic-grounded'` or `'claude-3-5-sonnet-20241022'`) |
| `is_mock` | `bool` | Convenience property indicating if mock mode was used |

---

## Usage Guide

### CLI Commands

1. **Ask a Question (Positional Query)**:
   ```bash
   python -m knowledge_brain.answerer "Where are email accounts and mailboxes managed?"
   ```

2. **Ask with Query Flag**:
   ```bash
   python -m knowledge_brain.answerer --query "Can I send a mass email campaign?"
   ```

3. **Force Mock Mode**:
   ```bash
   python -m knowledge_brain.answerer --mock "Are email forwarders supported in cPanel?"
   ```

4. **Interactive Mode**:
   ```bash
   python -m knowledge_brain.answerer
   ```

### Python API Usage

```python
from knowledge_brain import KnowledgeAnswerer, KnowledgeRetriever

# Initialize retriever and answerer
retriever = KnowledgeRetriever()
answerer = KnowledgeAnswerer(use_mock=True)

# End-to-end question answering
result = answerer.ask("Where are email accounts and mailboxes managed?", retriever=retriever)

print(result.answer)
print(f"Evidence Found: {result.evidence_found}")
for citation in result.citations:
    print(f"Source: {citation.title} ({citation.url})")
```

---

# Knowledge Brain: Component 4 — HTTP API Service

## Overview

The **Knowledge Brain HTTP API** (`knowledge_brain/api.py`) is the integration interface of Task 1 in the `AI-Tech-Consultant` system. It exposes a lightweight, reliable HTTP REST API that downstream services—specifically **Task 2: Inbox Agent** and **Task 3: Meeting to Scope of Work (SOW)**—can call to check service health and ask technical questions grounded in Brilliant Directories platform documentation.

> [!NOTE]
> The HTTP API is built with Flask, providing high reliability, zero heavy external C dependencies, and instant in-memory test execution via `app.test_client()`. It operates completely offline by default under `USE_MOCK_AI=true`.

---

## Endpoints Specification

### 1. `GET /health` (and `GET /api/health`)

Checks service health, reports current operating mode, model name, and the count of loaded documentation articles.

- **Request**: No body required.
- **Success Response (200 OK)**:
  ```json
  {
    "status": "healthy",
    "service": "knowledge_brain",
    "mode": "mock",
    "model": "mock-deterministic-grounded",
    "articles_loaded": 3
  }
  ```
- **Error Response (405 Method Not Allowed)**:
  ```json
  {
    "error": "The method is not allowed for the requested URL.",
    "status_code": 405
  }
  ```

---

### 2. `POST /ask` (and `POST /query`, `POST /api/ask`)

Accepts a question, performs BM25 retrieval over documentation articles, generates a grounded answer, and returns structured citations.

- **Headers**: `Content-Type: application/json`
- **Request Body Parameters**:
  | Parameter | Type | Required | Default | Description |
  |---|---|---|---|---|
  | `query` (or `question`) | `str` | Yes | - | Technical question to answer |
  | `top_k` | `int` | No | `3` | Number of relevant documentation passages to retrieve |
  | `use_mock` | `bool` | No | `None` | Optional override for mock vs Claude mode |

- **Success Response (200 OK) — Supported Question**:
  ```json
  {
    "query": "Where are email accounts and mailboxes managed?",
    "answer": "[MOCK MODE] Based on \"Emails » Email Accounts\":\nEmail accounts (mailboxes) at the website's own domain, such as support@example.com, are created and managed in Settings > Domain Manager > Mailboxes. A free directoryup.com address has no mailboxes of its own.\n\nCitations:\n- Emails » Email Accounts (https://support.brilliantdirectories.com/support/solutions/articles/5000558363-emails-email-accounts)",
    "evidence_found": true,
    "evidence_status": "found",
    "citations": [
      {
        "title": "Emails » Email Accounts",
        "url": "https://support.brilliantdirectories.com/support/solutions/articles/5000558363-emails-email-accounts"
      }
    ],
    "mode": "mock",
    "model": "mock-deterministic-grounded",
    "is_mock": true
  }
  ```

- **Success Response (200 OK) — Unsupported Question**:
  ```json
  {
    "query": "What is the refund policy?",
    "answer": "[MOCK MODE] The available documentation does not establish the answer. No matching documentation records were found.",
    "evidence_found": false,
    "evidence_status": "insufficient_evidence",
    "citations": [],
    "mode": "mock",
    "model": "mock-deterministic-grounded",
    "is_mock": true
  }
  ```

- **Error Response (400 Bad Request)**:
  ```json
  {
    "error": "Missing required field: 'query' (or 'question')"
  }
  ```

---

## Inter-Service Integration (Tasks 2 & 3)

The API is specifically designed to be called by downstream Python services using standard `requests` or `urllib`:

```python
import requests

KNOWLEDGE_BRAIN_URL = "http://127.0.0.1:8501"

def query_knowledge_brain(question: str) -> dict:
    """Query Knowledge Brain for grounded platform facts."""
    response = requests.post(
        f"{KNOWLEDGE_BRAIN_URL}/ask",
        json={"query": question},
        timeout=10,
    )
    response.raise_for_status()
    return response.json()
```

---

## Usage Guide

### Starting the Server

```powershell
# Default (127.0.0.1:8501)
python -m knowledge_brain.api

# Custom host and port
python -m knowledge_brain.api --host 127.0.0.1 --port 8501

# Force mock mode
python -m knowledge_brain.api --mock
```

### Running Offline Tests

```powershell
pytest tests/test_api.py -v
```
