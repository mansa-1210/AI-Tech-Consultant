# Evaluation record

## What is measured locally

The repeatable baseline is the pytest suite. It tests parsing, crawling failures, retrieval ranking, grounded-answer fallback, API validation, 50-email loading, review safety, three initial meetings, and follow-up change detection. Run it with:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

The suite uses repository-local temporary files so it works in restricted Windows environments. The most recent local run on 2026-10-19 passed 158 cases. These are implementation regression cases, not independent real-world model-quality measurements.

## Quality and safety criteria

| Workflow | Pass condition | Safe failure |
|---|---|---|
| Knowledge Brain | Answer has retrieved evidence and citations | `insufficient_evidence`; create a human escalation |
| Inbox | Draft uses cited Knowledge Brain result | Escalation draft; review required; no email sent |
| Meeting-to-SOW | Features have feasibility evidence and stated assumptions | Pending approval; no proposal sent |

## Fixed offline evaluation plan

`evaluations/cases.offline.json` contains a fixed, authored 60-case held-out set: 20 Knowledge Brain questions, 20 Inbox Agent inputs, and 20 Meeting-to-SOW inputs. It is entirely synthetic and must be kept unchanged while comparing models. A full run entails 85 model calls: 20 Knowledge Brain calls, 20 Inbox calls, and 45 per-feature feasibility calls for the Meeting-to-SOW cases. The runner writes per-case model identifier, latency, observed token metadata, estimated cost, automatic checks, and failures.

Automatic Knowledge Brain checks cover the expected evidence status and declared answer terms; a reviewer must additionally assess groundedness, completeness, citation correctness, and abstention. Inbox checks cover category and mandatory review; a reviewer must assess routing, factuality, escalation usefulness, and tone. Meeting-to-SOW checks cover candidate-feature extraction, approval state, and blocked client sending; a reviewer must assess scope completeness, feasibility interpretation, estimate assumptions, and risks. See `docs/offline_completion_plan.md` for the pre-run token and cost estimates.

## Measurements not yet claimed

On 2026-10-18, a prior live Gemini `/ask` smoke test reportedly returned HTTP 200 with grounded evidence and one citation; this was not re-run here because the evaluation environment blocks outbound sockets (`WinError 10013`). After the full corpus was regenerated, a local forced-mock smoke test returned `/health` 200 with `articles_loaded: 1535`, and `/ask` 200 with grounded evidence and two citations. This verifies local corpus loading and deterministic grounded behavior only; it is not a live-model quality, cost, latency, or model-comparison result. The current Gemini integration now captures prompt and candidate token metadata when the SDK returns it, but no live token or billing data has been observed in this completion phase. No paid-provider comparison, full-corpus quality measurement, or production measurement is claimed. To finish evaluation evidence, run the fixed 60 held-out cases per selected model against the completed corpus, record request latency, evidence/citation checks, human quality judgements, token metadata, calculated cost, and failures. Run a second model only if the assignment rubric explicitly requires comparison. The API exposes the actual provider/model in `/health` and each `/ask` response so records cannot be mistaken for mock results.

On 2026-10-19, a forced-mock route check used the full 1,535-record corpus without making a paid-provider call. A mailbox question returned grounded evidence and two citations; an unrelated lunar-mining insurance question returned `insufficient_evidence`; the supported synthetic inbox email produced a cited `grounded_draft` with mandatory review; and Meeting-to-SOW produced a pending-approval proposal. The Northstar follow-up reported only email-notification and Salesforce removals/replacements plus an SMS-notifications addition. This is not a live Claude or Gemini evaluation: the available configuration selects Gemini, no Anthropic credential was configured, and Gemini quota availability was not retried.

## Known failures and mitigation

The 1,535-record BM25 corpus can still lack, miss, or retrieve stale evidence; the five skipped sitemap URLs have no persisted per-URL failure manifest. The answerer returns insufficient evidence instead of inventing an answer. A live provider timeout, quota error, or malformed provider response now returns HTTP 503 with `provider_unavailable`, and downstream workflows escalate without presenting it as missing documentation. Rules can misclassify unfamiliar wording; human review remains mandatory and resets autonomy on any edit or rejection.
