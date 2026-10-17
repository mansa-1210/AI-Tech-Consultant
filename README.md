# AI Tech Consultant

A solo capstone project implementing a suite of AI-driven tools for technology consulting workflows, powered by Anthropic's Claude as the central AI model.

> **Project Status**: Task 1 Components 1, 2, 3 & 4 (Ingestion Pipeline, BM25 Local Lexical Retriever, Grounded Answer Generator, and HTTP API) implemented and tested. Other components and tasks are in planning/development.

---

## Overview of the Three Tasks

1. **Knowledge Brain (`knowledge_brain/`)**
   - Python-based Retrieval-Augmented Generation (RAG) over Brilliant Directories Solution Center documentation.
   - **Component 1 (Active)**: Sitemap parsing, respectful web crawler, HTML text extraction, and article record hashing (`knowledge_brain/ingestion.py`).
   - **Component 2 (Active)**: Transparent, offline lexical search engine using Okapi BM25 with title boosting and snippet extraction (`knowledge_brain/retriever.py`).
   - **Component 3 (Active)**: Grounded answer generation engine with citation preservation, explicit fallback on insufficient evidence, Claude primary architecture, and offline deterministic mock mode (`knowledge_brain/answerer.py`).
   - **Component 4 (Active)**: Small, reliable HTTP API with `/health` and `/ask` endpoints for integration with Tasks 2 and 3 (`knowledge_brain/api.py`).
   - Scheduled daily content updates and human escalation capabilities (upcoming).

2. **Inbox Agent (`inbox_agent/`)**
   - Intelligent email classification and reply drafting workflow.
   - Features human-in-the-loop review and approval queues.
   - Incorporates feedback learning to refine future drafts.
   - Uses carefully evaluated, safe auto-send rules for low-risk, high-confidence scenarios.

3. **Meeting to Scope of Work (`meeting_to_sow/`)**
   - Multi-agent workflow transforming client meeting transcripts into technical deliverables.
   - Produces cited feasibility analyses, project scopes, granular task breakdowns, and realistic hour estimates.
   - Generates executive-ready proposals with human approval checkpoints.

---

## AI Architecture & Assistant Policy

- **Central AI Model**: Anthropic Claude (e.g., Claude 3.5 Sonnet for deep reasoning/synthesis, Claude 3.5 Haiku for classification).
- **Sole AI Coding Assistant**: Antigravity.
- **Development & Mocking**: During initial development and offline testing, mock responses are clearly labelled as `[MOCK MODE]` to allow complete testing before making paid API calls. The real Claude API can be enabled by setting `USE_MOCK_AI=false` and adding an `ANTHROPIC_API_KEY` in `.env`.
- **Knowledge Brain Status**: Ingestion, BM25 retrieval, grounded answer generation, and the HTTP API run locally and offline by default under `USE_MOCK_AI=true`. Vector embeddings will be integrated in subsequent milestones.

---

## Directory Structure

```text
AI-Tech-Consultant/
├── knowledge_brain/    # Task 1: RAG over Brilliant Directories Solution Center
│   ├── ingestion.py    # Component 1: Sitemap parser, crawler, and HTML extractor
│   ├── retriever.py    # Component 2: Local BM25 retriever & snippet extractor
│   ├── answerer.py     # Component 3: Grounded answer generator & citation engine
│   ├── api.py          # Component 4: Lightweight HTTP API for inter-service calls
│   ├── data/           # Crawled article JSON files (e.g. sample_articles.json)
│   └── tests/          # Module test re-exports
├── inbox_agent/        # Task 2: Email classification & reply drafting
├── meeting_to_sow/     # Task 3: Multi-agent meeting transcript to SOW
├── shared/             # Common utilities, Claude client wrapper, logging
├── tests/              # End-to-end integration and unit tests (mocked HTTP & offline)
│   ├── test_ingestion.py
│   ├── test_retriever.py
│   ├── test_answerer.py
│   └── test_api.py
├── docs/               # Architecture, evaluation, and user documentation
│   └── knowledge_brain.md
├── .env.example        # Environment variable template
├── .gitignore          # Git exclusion rules
├── pytest.ini          # Pytest configuration
├── requirements.txt    # Project Python dependencies
└── README.md           # Project overview and instructions
```

---

## Setup & Configuration

1. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```
2. Configure settings inside `.env` as needed (e.g., `USE_MOCK_AI=true` for local development without API charges).

---

## Running Knowledge Brain Ingestion

The ingestion crawler fetches and parses the official Brilliant Directories documentation sitemap (`https://support.brilliantdirectories.com/support/sitemap.xml`) and extracts structured records (`ArticleRecord` with URL, title, clean text, timestamp, and SHA-256 content hash).

### 1. Inspect the Sitemap (No Page Downloads)
View total URL count, article count, non-article count, and sample URLs:
```bash
python -m knowledge_brain.ingestion --inspect-sitemap
```

### 2. Ingest a Small Sample of Articles (Safe Testing)
Ingest a small number of pages respectfully with default rate-limiting delay:
```bash
python -m knowledge_brain.ingestion --limit 3
```

### 3. Ingest and Save to a JSON File
Save parsed records into a structured JSON file:
```bash
python -m knowledge_brain.ingestion --limit 5 --output knowledge_brain/data/sample_articles.json
```

### Optional Arguments
- `--sitemap-url <URL>`: Override sitemap location.
- `--delay <SECONDS>`: Seconds to wait between page requests (default: `0.5`).
- `--timeout <SECONDS>`: HTTP request timeout in seconds (default: `10`).
- `-v` / `--verbose`: Enable detailed debug logging.

---

## Running Knowledge Brain Retriever

The local retriever performs offline lexical search (Okapi BM25) over downloaded articles (`knowledge_brain/data/sample_articles.json`) with title boosting and relevant snippet extraction. It does not require any paid API or external embedding services.

### 1. Search with Positional Query
```powershell
python -m knowledge_brain.retriever "Can I send mass emails?"
```

### 2. Search with Flags and Top-K Limit
```powershell
python -m knowledge_brain.retriever --query "cPanel email forwarders" --top-k 1
```

### 3. Interactive Search Mode
```powershell
python -m knowledge_brain.retriever
```

---

## Running Knowledge Brain Answerer

The grounded answer generator (`knowledge_brain/answerer.py`) accepts user questions and retrieves matching documentation, synthesizing answers grounded strictly in the official articles and returning citations with article titles and canonical URLs. Under `USE_MOCK_AI=true` (default), it runs completely offline using a deterministic, generic extractor.

### 1. Ask with Positional Question
```powershell
python -m knowledge_brain.answerer "Where are email accounts and mailboxes managed?"
```

### 2. Ask with Query Flag
```powershell
python -m knowledge_brain.answerer --query "Can I send a mass email campaign?"
```

### 3. Interactive Question-Answering Mode
```powershell
python -m knowledge_brain.answerer
```

---

## Running Knowledge Brain HTTP API

The Knowledge Brain HTTP API (`knowledge_brain/api.py`) provides lightweight, robust HTTP endpoints designed for downstream consumption by the **Inbox Agent (Task 2)** and **Meeting to Scope of Work (Task 3)**.

### Starting the Server

Start the API server using the module CLI:
```powershell
python -m knowledge_brain.api
```

Optional CLI flags:
```powershell
# Custom host and port
python -m knowledge_brain.api --host 127.0.0.1 --port 8501

# Force mock mode and enable debug logging
python -m knowledge_brain.api --mock --debug

# Specify custom articles dataset
python -m knowledge_brain.api --data knowledge_brain/data/sample_articles.json
```

Or run via Flask CLI:
```powershell
flask --app knowledge_brain.api:app run --host 127.0.0.1 --port 8501
```

### Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/health` (or `/api/health`) | `GET` | Service health status, operating mode, and loaded article count |
| `/ask` (or `/query`, `/api/ask`) | `POST` | Grounded question-answering with citations and evidence status |

### Example Requests & Responses

#### 1. Health Check

**PowerShell (`Invoke-RestMethod`):**
```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8501/health" -Method Get
```

**cURL:**
```bash
curl -X GET http://127.0.0.1:8501/health
```

**Response (200 OK):**
```json
{
  "articles_loaded": 3,
  "mode": "mock",
  "model": "mock-deterministic-grounded",
  "service": "knowledge_brain",
  "status": "healthy"
}
```

#### 2. Grounded Question Answering (Supported Question)

**PowerShell (`Invoke-RestMethod`):**
```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8501/ask" `
  -Method Post `
  -ContentType "application/json" `
  -Body '{"query": "Where are email accounts and mailboxes managed?"}'
```

**cURL:**
```bash
curl -X POST http://127.0.0.1:8501/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "Where are email accounts and mailboxes managed?"}'
```

**Python (`requests`):**
```python
import requests

response = requests.post(
    "http://127.0.0.1:8501/ask",
    json={"query": "Where are email accounts and mailboxes managed?", "top_k": 3},
)
data = response.json()
print("Answer:", data["answer"])
print("Evidence Status:", data["evidence_status"])
for citation in data["citations"]:
    print(f"Source: {citation['title']} -> {citation['url']}")
```

**Response (200 OK):**
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

#### 3. Unsupported Question (Fallback on Insufficient Evidence)

**cURL:**
```bash
curl -X POST http://127.0.0.1:8501/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the refund policy?"}'
```

**Response (200 OK):**
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

#### 4. Invalid Input Handling (400 Bad Request)

**Request with missing query or empty payload:**
```bash
curl -X POST http://127.0.0.1:8501/ask \
  -H "Content-Type: application/json" \
  -d '{}'
```

**Response (400 Bad Request):**
```json
{
  "error": "Missing required field: 'query' (or 'question')"
}
```

---

## Running the Tests

All tests run completely offline without hitting external networks or requiring paid API credits.

Run the full test suite (92 tests across ingestion, retriever, answerer, and API):
```bash
pytest -v
```
Or execute specifically via Python module:
```bash
python -m pytest -v
```
Or target the Knowledge Brain module tests:
```bash
pytest knowledge_brain/tests/ -v
```


