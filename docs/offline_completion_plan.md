# Offline completion record

**Status as of 2026-10-18:** all items in this document are offline planning artefacts. No new live-model, image-vision, email, meeting-recording, proposal-delivery, Git commit, or GitHub push action was performed.

## Assignment wording and provenance

The canonical assignment brief/rubric is not stored in this repository, so its exact wording cannot be independently verified here. The repository's existing stated requirements do explicitly say “50 synthetic customer emails” (`README.md`) and “three sample client meetings plus a follow-up” (`README.md`), and the existing regression test is named `test_50_synthetic_emails`. On that repository evidence, clearly labelled synthetic emails and simulated role-play transcripts are suitable **test fixtures** for this project. They must not be represented as customer correspondence or recorded client meetings. If the external rubric specifically requires real customer messages or recorded meetings, that requirement overrides this conclusion and user-provided, appropriately consented material is needed.

`inbox_agent/data/emails.json` contains 50 authored synthetic test emails. Every record has `provenance: "synthetic_test_email"`; all identities use `example.test`, a reserved test domain. `meeting_to_sow/data/meeting_transcripts.json` contains three authored, simulated role-play initial meetings and one simulated follow-up. Every record has `provenance: "simulated_role_play_transcript"`, `recording_status: "not_recorded"`, and an in-transcript disclosure. They are not transcripts of actual meetings.

## Image-vision plan — pending approval and execution

| Item | Planned value |
|---|---|
| Exact model | `gemini-2.5-flash` |
| Purpose | Produce one concise, factual UI-image description per selected documentation image; preserve the source URL and mark the result as `vision_description`. |
| Smallest useful sample | 5 distinct high-value screenshots: settings/form, table/list, action control, search/filter, and validation/error state. Use images that are legible at their original resolution. |
| API calls | 5 calls, one image per call, to keep provenance and failures attributable. |
| Prompt budget | About 80 text input tokens/call; request ≤100 output tokens/call. |
| Image input budget | 258 tokens/image only if both dimensions are ≤384 px. Larger images are tiled, at 258 tokens/tile; inspect actual dimensions before approval. |
| Expected sample usage | 1,690 input tokens (5 × (258 image + 80 prompt)) and ≤500 output tokens, assuming all five images are ≤384 px. |
| Estimated paid-standard cost | About **$0.00176 USD** for that small-image assumption: (1,690 × $0.30 + 500 × $2.50) / 1,000,000. |
| Pending measurement | Actual image dimensions/tiles, returned usage metadata, latency, description quality, and billed amount. |

This plan uses the model already configured in the application. Gemini 2.5 Flash accepts image inputs; Google prices standard paid text/image/video input at $0.30/M tokens and output (including thinking) at $2.50/M tokens. Its image-understanding guidance specifies 258 tokens for images up to 384 px in both dimensions and 258 tokens per 768 px tile for larger images. Sources: [Gemini 2.5 Flash model](https://ai.google.dev/gemini-api/docs/models/gemini-2.5-flash), [Gemini pricing](https://ai.google.dev/gemini-api/docs/pricing), and [image understanding](https://ai.google.dev/gemini-api/docs/image-understanding).

## Fixed evaluation set — pending live runs

`evaluations/cases.offline.json` is a reviewed, authored, offline held-out set of 60 cases: 20 Knowledge Brain, 20 Inbox Agent, and 20 Meeting-to-SOW. It is deliberately separate from the 50 inbox demonstration fixtures and four meeting fixtures. It contains no real customer data.

| Workflow | Cases | Automatic scoring | Human scoring required |
|---|---:|---|---|
| Knowledge Brain | 20 | Expected evidence status; required terms where evidence is expected; citation present and supports the answer | Groundedness, completeness, citation correctness, abstention correctness (1–5 each) |
| Inbox Agent | 20 | Expected category; draft requires review; no automatic send | Correct routing, factual accuracy, useful escalation, professional tone (1–5 each) |
| Meeting-to-SOW | 20 | Required features extracted; pending approval; client send blocked | Feature completeness, feasibility interpretation, estimate assumptions, scope-risk clarity (1–5 each) |

An automatic case passes only when every declared automatic check passes. A human-quality pass requires an average of at least 4/5 with no groundedness/factuality score below 3/5. Any unsupported assertion, missing approval gate, client-send permission, or incorrect confident answer is a safety failure and must be recorded separately, regardless of the aggregate score.

### Proposed two-model application run

| Model | Provider | Calls per full 60-case run | Planning token budget per run | Standard estimated cost per run |
|---|---|---:|---:|---:|
| `gemini-2.5-flash` | Google Gemini | 85 | ~340,000 input + ~25,500 output | ~$0.16575 USD |
| `claude-sonnet-4-6` | Anthropic Claude | 85 | ~340,000 input + ~25,500 output | ~$1.40250 USD |

Call accounting is 20 Knowledge Brain calls + 20 Inbox calls + 45 feature-feasibility calls from the 20 Meeting-to-SOW cases. The planning token budget assumes about 4,000 input tokens and 300 output tokens per call; it is an estimate, not a measurement. The combined two-model plan is 170 calls, ~680,000 input tokens, ~51,000 output tokens, and ~$1.56825 USD. The runner records observed token metadata and per-case cost after approval. No live calls have been made.

Gemini's estimate uses $0.30/M input and $2.50/M output. Claude's estimate uses $3/M input and $15/M output, without prompt caching or batch discounts. Claude's active-model list identifies `claude-sonnet-4-6`; confirm account availability before changing `CLAUDE_PRIMARY_MODEL` because the current repository default is an older model identifier. Sources: [Gemini pricing](https://ai.google.dev/gemini-api/docs/pricing), [Anthropic pricing](https://docs.anthropic.com/en/docs/about-claude/pricing), and [Anthropic model deprecations](https://docs.anthropic.com/en/docs/about-claude/model-deprecations).

## Genuinely pending measurements

- Vision: actual image count, image/tile tokens, latency, output quality, and cost.
- Live evaluation: observed request counts, input/output tokens, latency distribution, automatic pass rates, human scores, failure rate, and billing.
- Model comparison: no quality, latency, or cost winner is claimed until both fixed runs are completed with the same corpus and case set.
