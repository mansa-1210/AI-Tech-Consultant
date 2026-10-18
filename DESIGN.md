# Design

## Why this exists

Small software teams lose time finding product facts, replying to repeated questions, and converting vague meetings into estimates. This project keeps those jobs connected: one evidence service supplies two assisted workflows. The design deliberately prefers an explicit “I do not know” over a confident unsupported answer.

```text
Brilliant Directories sitemap
          |
     polite crawler -> normalized article JSON -> BM25 retriever -> Knowledge Brain API
                                                              |                 |
                                                     Inbox Agent          Meeting-to-SOW
                                                     draft + review       scope + estimate
                                                              \                 /
                                                               human approval
```

## Implemented choices

The Knowledge Brain is a Python Flask service. It parses sitemap XML, fetches HTML with a delay and timeout, strips boilerplate, hashes text, stores article JSON, ranks records using BM25, and attaches canonical URLs to answers. It supports Claude or Gemini when configured, plus deterministic mock mode for reproducible offline tests. The `/health` response reports the actual execution mode (`mock`, `claude`, or `gemini`) and selected model. The endpoint contract is `POST /ask` (also `/query`), with `GET /health`, `POST /escalations`, and `POST /escalations/{id}/resolve` for auditable human review.

Inbox Agent and Meeting-to-SOW are separate Python modules and only ask product questions through HTTP; they do not import retrieval code. Inbox drafts require a human review, record approve/edit/reject outcomes, and only mark a category eligible after 20 unchanged consecutive approvals. It is still a simulation: no external mailbox or send capability exists.

Meeting-to-SOW separates extraction, feasibility check, estimate, and follow-up comparison. Every result is marked pending approval. Estimates are ranges with stated assumptions, so they are usable for discovery rather than misleading fixed quotes.

## Deliberately deferred

On 2026-10-18, the repaired ingestion command saved 1,535 valid records from 1,540 sitemap URLs; five URLs were skipped. It persists requested output before previews and configures UTF-8-safe console output, preventing the prior Windows `charmap` display failure from discarding a completed crawl. There is no vector database, screenshot vision model, production scheduler, authentication, email provider, or proposal delivery integration. The phrase/rule classifiers and feature extraction are transparent baselines, not general natural-language understanding. These boundaries are intentional and shown in output so they cannot be mistaken for production behaviour.

## Next steps

Persist a sitemap manifest with changed/removed URLs; add image descriptions linked to surrounding chunks; evaluate BM25 against embeddings and hybrid retrieval; protect the API with authentication; then run a bounded Gemini evaluation on a held-out set with recorded observable latency and failures. Autonomy should remain disabled until a monitored, category-specific quality policy is agreed.
