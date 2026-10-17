# AI Tech Consultant

A solo capstone project implementing a suite of AI-driven tools for technology consulting workflows, powered by Anthropic's Claude as the central AI model.

> **Project Status**: Initial skeleton and repository structure created. Core task implementations and evaluations are currently in planning/development (not yet implemented or tested).

---

## Overview of the Three Tasks

1. **Knowledge Brain (`knowledge_brain/`)**
   - Python-based Retrieval-Augmented Generation (RAG) over Brilliant Directories Solution Center documentation.
   - Generates grounded answers with strict citations.
   - Provides honest fallback when retrieved evidence is insufficient.
   - Includes scheduled daily content updates and human escalation capabilities.

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
- **Development & Mocking**: During initial development and offline testing, mock responses are clearly labelled as mocks to allow complete testing before making paid API calls. The real Claude API can be enabled by configuring `.env`.

---

## Directory Structure

```text
AI-Tech-Consultant/
├── knowledge_brain/    # Task 1: RAG over Brilliant Directories Solution Center
├── inbox_agent/        # Task 2: Email classification & reply drafting
├── meeting_to_sow/     # Task 3: Multi-agent meeting transcript to SOW
├── shared/             # Common utilities, Claude client wrapper, logging
├── tests/              # End-to-end integration and unit tests
├── docs/               # Architecture, evaluation, and user documentation
├── .env.example        # Environment variable template
├── .gitignore          # Git exclusion rules
└── README.md           # Project overview and instructions
```

---

## Setup & Configuration

1. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```
2. Configure settings inside `.env` as needed (e.g., `USE_MOCK_AI=true` for local development without API charges).
