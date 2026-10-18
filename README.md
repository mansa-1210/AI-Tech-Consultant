# AI Tech Consultant

Three connected, local-first workflows for a software-services team:

- **Knowledge Brain** crawls Brilliant Directories support pages, retrieves evidence with BM25, and returns grounded, cited answers through a Flask API.
- **Inbox Agent** classifies 50 clearly labelled synthetic test emails, calls that API, drafts a reply, and records human feedback. It never sends real email.
- **Meeting-to-SOW** reads three clearly labelled simulated role-play transcripts plus a simulated follow-up, calls that same API for feasibility evidence, creates a preliminary scope and estimate, and flags all client-facing output for approval. They are not recorded meetings.

## Run

Windows PowerShell and Python 3.11+ are required. From the project root, this one command creates the local virtual environment when needed, installs dependencies, and starts the shared API and browser console:

```powershell
powershell -ExecutionPolicy Bypass -File .\start.ps1
```

Open `http://127.0.0.1:8501`. In a second terminal, run tests as needed:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Copy `.env.example` to `.env`. It defaults to `USE_MOCK_AI=true` for safe, deterministic local runs. To make paid live-model calls, explicitly set `USE_MOCK_AI=false` and configure either `AI_PROVIDER=claude` with `ANTHROPIC_API_KEY` or `AI_PROVIDER=gemini` with `GEMINI_API_KEY`. The `/health` endpoint reports the actual selected provider and model. Keep `.env` out of Git.

For an incremental daily refresh, first review the source site's crawler policy, then register the included Windows Task Scheduler job. It writes the same `full_articles.json` corpus that the API serves:

```powershell
.\scripts\install_daily_sync_task.ps1
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

## Evaluation and limits

The local dataset currently contains 1,535 crawled articles and BM25 retrieval. Image descriptions are retrieved and returned with their provenance (`source_alt_text` or `vision_description`); actual vision descriptions have not been generated yet. `evaluations/cases.offline.json` provides 60 fixed offline evaluation cases, and `evaluations/runner.py` prepares measured two-model runs, but no live-model results are claimed. All estimates are planning ranges and all customer-facing actions require human approval. The full corpus is locally ignored, so it must be included in the final GitHub publication or attached as a release asset for a fresh clone to use it. See [docs/offline_completion_plan.md](docs/offline_completion_plan.md), [DESIGN.md](DESIGN.md), and [EVALS.md](EVALS.md).
