# Evaluation record

## What is measured locally

The repeatable baseline is the pytest suite. It tests parsing, crawling failures, retrieval ranking, grounded-answer fallback, API validation, 50-email loading, review safety, three initial meetings, and follow-up change detection. Run it with:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

The suite uses repository-local temporary files so it works in restricted Windows environments. The final local run on 2026-10-18 passed 141 cases: 96 Knowledge Brain cases (answerer, API, retriever, and ingestion), 20 Inbox cases, 21 Meeting-to-SOW cases, two console cases, and two incremental-sync cases. These are implementation regression cases, not independent real-world model-quality measurements.

## Quality and safety criteria

| Workflow | Pass condition | Safe failure |
|---|---|---|
| Knowledge Brain | Answer has retrieved evidence and citations | `insufficient_evidence`; create a human escalation |
| Inbox | Draft uses cited Knowledge Brain result | Escalation draft; review required; no email sent |
| Meeting-to-SOW | Features have feasibility evidence and stated assumptions | Pending approval; no proposal sent |

## Measurements not yet claimed

On 2026-10-18, a prior live Gemini `/ask` smoke test reportedly returned HTTP 200 with grounded evidence and one citation; this was not re-run here because the evaluation environment blocks outbound sockets (`WinError 10013`). After the full corpus was regenerated, a local forced-mock smoke test returned `/health` 200 with `articles_loaded: 1535`, and `/ask` 200 with grounded evidence and two citations. This verifies local corpus loading and deterministic grounded behavior only; it is not a live-model quality, cost, latency, or model-comparison result. The current Gemini integration does not expose token usage or billing data in its response, so cost is not measurable from this implementation. No paid-provider comparison, full-corpus quality measurement, or production measurement is claimed. To finish evaluation evidence, run the existing 20/20/21 regression cases plus 20 held-out cases per workflow against the completed corpus, record request latency, evidence/citation checks, human quality judgements, and failures. Run a second model only if the assignment rubric explicitly requires comparison. The API exposes the actual provider/model in `/health` and each `/ask` response so records cannot be mistaken for mock results.

## Known failures and mitigation

The 1,535-record BM25 corpus can still lack, miss, or retrieve stale evidence; the five skipped sitemap URLs have no persisted per-URL failure manifest. The answerer returns insufficient evidence instead of inventing an answer. A downstream API timeout creates an escalation-style draft/unknown feasibility result. Rules can misclassify unfamiliar wording; human review remains mandatory and resets autonomy on any edit or rejection.
