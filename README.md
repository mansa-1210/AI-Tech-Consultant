# AI Tech Consultant

Three connected, local-first workflows for a software-services team:

- **Knowledge Brain** crawls Brilliant Directories support pages, retrieves evidence with BM25, and returns grounded, cited answers through a Flask API.
- **Inbox Agent** classifies 50 synthetic customer emails, calls that API, drafts a reply, and records human feedback. It never sends real email.
- **Meeting-to-SOW** reads three sample client meetings plus a follow-up, calls that same API for feasibility evidence, creates a preliminary scope and estimate, and flags all client-facing output for approval.

## Run

Windows PowerShell and Python 3.11+ are required. From the project root:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m app
```

This is the one command that starts the shared API and the browser console for all three workflows. Open `http://127.0.0.1:8501`. In a second terminal, run the complete test suite:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Copy `.env.example` to `.env`. It defaults to `USE_MOCK_AI=true` for safe, deterministic local runs. To make paid live-model calls, explicitly set `USE_MOCK_AI=false` and configure either `AI_PROVIDER=claude` with `ANTHROPIC_API_KEY` or `AI_PROVIDER=gemini` with `GEMINI_API_KEY`. The `/health` endpoint reports the actual selected provider and model. Keep `.env` out of Git.

For an incremental daily refresh, schedule this command with Windows Task Scheduler (daily) after confirming the source site's crawler policy:

```powershell
.\.venv\Scripts\python.exe -m knowledge_brain.sync
```

For the initial full crawl, use the explicit command below rather than the safe three-article CLI default:

```powershell
.\.venv\Scripts\python.exe -m knowledge_brain.ingestion --full --output knowledge_brain/data/full_articles.json --delay 0.5
```

Once that file is complete, the API automatically prefers it over the sample corpus. On 2026-10-18, the corrected crawl saved 1,535 valid records from 1,540 sitemap URLs (five were skipped); verify the count before a demo:

```powershell
.\.venv\Scripts\python.exe -c "import json; print(len(json.load(open('knowledge_brain/data/full_articles.json', encoding='utf-8'))))"
```

Set `KNOWLEDGE_BRAIN_DATA_PATH` only when using another JSON location.

Open `http://127.0.0.1:8501/` for the compact demo screen. The Inbox and Meeting workflows are runnable modules and deliberately use the Knowledge Brain only through its HTTP endpoint.

## Important limits

The local dataset currently contains 1,535 crawled articles and BM25 retrieval; it is a demonstrator, not a production vector database. Image understanding, scheduled daily sync, and real-model cost/latency measurements remain pending. All estimates are planning ranges and all customer-facing actions require human approval. See [DESIGN.md](DESIGN.md) and [EVALS.md](EVALS.md).
