"""Knowledge Brain HTTP API.

Provides lightweight, robust HTTP endpoints for the Knowledge Brain service:
- GET  /health     - Service health check, loaded articles count, and operating mode
- GET  /api/health - Namespaced health check alias
- POST /ask        - Grounded question-answering with evidence status and citations
- POST /query      - Alias for /ask
- POST /api/ask    - Namespaced ask alias

Designed for reliable inter-service integration with Inbox Agent (Task 2)
and Meeting to Scope of Work (Task 3).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import sys
from typing import Any, Dict, Optional, Sequence, Union
from uuid import uuid4

from flask import Flask, Response, jsonify, request
from werkzeug.exceptions import HTTPException

try:
    from dotenv import load_dotenv

    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=False)
except ImportError:
    pass
from knowledge_brain.answerer import (
    DEFAULT_CLAUDE_MODEL,
    DEFAULT_MOCK_MODEL,
    INSUFFICIENT_EVIDENCE_PHRASE,
    AnswerResult,
    Citation,
    KnowledgeAnswerer,
    is_mock_ai_enabled,
)
from knowledge_brain.retriever import (
    DEFAULT_SAMPLE_DATA_PATH,
    KnowledgeRetriever,
)

logger = logging.getLogger(__name__)


def _load_escalations(path: Path) -> list[Dict[str, Any]]:
    """Load persisted human-review tasks, treating a missing file as empty."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, list) else []
    except (OSError, json.JSONDecodeError):
        return []


def _save_escalations(path: Path, items: list[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(items, indent=2, ensure_ascii=False), encoding="utf-8")


def create_app(
    retriever: Optional[KnowledgeRetriever] = None,
    answerer: Optional[KnowledgeAnswerer] = None,
    data_path: Optional[Union[str, Path]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Flask:
    """Application factory for the Knowledge Brain HTTP API.

    Args:
        retriever: Optional custom KnowledgeRetriever instance.
        answerer: Optional custom KnowledgeAnswerer instance.
        data_path: Optional path to JSON articles file if retriever is not provided.
        config: Optional configuration dictionary for Flask app.

    Returns:
        Configured Flask application instance.
    """
    app = Flask(__name__)

    if config:
        app.config.update(config)

    app.config.setdefault(
        "ESCALATIONS_PATH",
        str(Path(__file__).resolve().parent / "data" / "escalations.json"),
    )

    # Initialize retriever
    if retriever is not None:
        app.retriever = retriever  # type: ignore[attr-defined]
    elif data_path is not None:
        app.retriever = KnowledgeRetriever(articles_path=data_path)  # type: ignore[attr-defined]
    else:
        # Prefer a completed full-crawl file, while retaining the checked-in
        # sample corpus for a clean first run and isolated tests. An explicit
        # environment path remains available for custom deployments.
        configured_path = os.getenv("KNOWLEDGE_BRAIN_DATA_PATH")
        full_corpus_path = Path("knowledge_brain/data/full_articles.json")
        selected_path: Union[str, Path] = (
            configured_path
            or (full_corpus_path if full_corpus_path.is_file() else DEFAULT_SAMPLE_DATA_PATH)
        )
        app.retriever = KnowledgeRetriever(articles_path=selected_path)  # type: ignore[attr-defined]

    # Initialize answerer
    if answerer is not None:
        app.answerer = answerer  # type: ignore[attr-defined]
    else:
        app.answerer = KnowledgeAnswerer()  # type: ignore[attr-defined]

    # Register error handlers for consistent JSON error responses
    @app.errorhandler(HTTPException)
    def handle_http_exception(err: HTTPException) -> tuple[Response, int]:
        """Convert standard Werkzeug HTTP exceptions into structured JSON responses."""
        return (
            jsonify({
                "error": err.description or "HTTP error",
                "status_code": err.code or 500,
            }),
            err.code or 500,
        )

    @app.errorhandler(Exception)
    def handle_generic_exception(err: Exception) -> tuple[Response, int]:
        """Catch-all error handler for unexpected server exceptions."""
        logger.exception("Unhandled server exception: %s", err)
        return (
            jsonify({
                "error": "Internal server error",
                "detail": str(err),
            }),
            500,
        )

    # Health check endpoints
    @app.route("/health", methods=["GET"])
    @app.route("/api/health", methods=["GET"])
    def health_check() -> tuple[Response, int]:
        """Health check endpoint reporting service status, mode, and loaded article count."""
        active_retriever: KnowledgeRetriever = app.retriever  # type: ignore[attr-defined]
        active_answerer: KnowledgeAnswerer = app.answerer  # type: ignore[attr-defined]

        loaded_count = (
            len(active_retriever.articles)
            if hasattr(active_retriever, "articles") and active_retriever.articles is not None
            else 0
        )

        return (
            jsonify({
                "status": "healthy",
                "service": "knowledge_brain",
                "mode": active_answerer.mode,
                "model": (
                    DEFAULT_MOCK_MODEL if active_answerer.use_mock else active_answerer.model
                ),
                "articles_loaded": loaded_count,
            }),
            200,
        )

    # Question-answering endpoints
    @app.route("/ask", methods=["POST"])
    @app.route("/query", methods=["POST"])
    @app.route("/api/ask", methods=["POST"])
    def ask_endpoint() -> tuple[Response, int]:
        """Question-answering endpoint.

        Accepts JSON payload:
        {
            "query": "Where are mailboxes managed?",
            "top_k": 3,          # optional, default 3
            "use_mock": true      # optional override
        }

        Returns structured JSON with answer, evidence status, citations, and metadata.
        """
        # Validate Content-Type
        if not request.is_json:
            return (
                jsonify({
                    "error": "Invalid Content-Type; request must be application/json"
                }),
                400,
            )

        try:
            payload = request.get_json(force=False, silent=False)
        except Exception:
            return jsonify({"error": "Malformed JSON in request body"}), 400

        if payload is None or not isinstance(payload, dict):
            return jsonify({"error": "JSON body must be an object"}), 400

        # Validate question / query field
        query_val = payload.get("query")
        if query_val is None:
            query_val = payload.get("question")

        if query_val is None:
            return (
                jsonify({
                    "error": "Missing required field: 'query' (or 'question')"
                }),
                400,
            )

        if not isinstance(query_val, str):
            return (
                jsonify({
                    "error": "Field 'query' (or 'question') must be a string"
                }),
                400,
            )

        query_clean = query_val.strip()
        if not query_clean:
            return (
                jsonify({
                    "error": "Field 'query' (or 'question') cannot be empty"
                }),
                400,
            )

        # Validate optional top_k
        top_k = 3
        if "top_k" in payload:
            raw_top_k = payload["top_k"]
            if isinstance(raw_top_k, bool) or not isinstance(raw_top_k, int) or raw_top_k <= 0:
                return (
                    jsonify({
                        "error": "Field 'top_k' must be a positive integer"
                    }),
                    400,
                )
            top_k = raw_top_k

        # Validate optional use_mock
        use_mock = None
        if "use_mock" in payload:
            raw_use_mock = payload["use_mock"]
            if not isinstance(raw_use_mock, bool):
                return (
                    jsonify({
                        "error": "Field 'use_mock' must be a boolean"
                    }),
                    400,
                )
            use_mock = raw_use_mock

        # Execute retrieval and answer generation
        active_retriever: KnowledgeRetriever = app.retriever  # type: ignore[attr-defined]
        default_answerer: KnowledgeAnswerer = app.answerer  # type: ignore[attr-defined]

        if use_mock is not None and use_mock != default_answerer.use_mock:
            working_answerer = KnowledgeAnswerer(
                use_mock=use_mock,
                model=default_answerer.model,
                api_key=default_answerer.api_key,
                provider=default_answerer.provider,
            )
        else:
            working_answerer = default_answerer

        try:
            result: AnswerResult = working_answerer.ask(
                query=query_clean,
                retriever=active_retriever,
                top_k=top_k,
            )

            response_data = {
                "query": result.query,
                "answer": result.answer,
                "evidence_found": result.evidence_found,
                "evidence_status": "found" if result.evidence_found else "insufficient_evidence",
                "citations": [c.to_dict() for c in result.citations],
                "mode": result.mode,
                "model": result.model,
                "is_mock": result.is_mock,
            }
            return jsonify(response_data), 200

        except Exception as err:
            logger.exception("Unexpected error processing question '%s': %s", query_clean, err)
            return (
                jsonify({
                    "error": "An internal error occurred while generating the answer",
                    "detail": str(err),
                }),
                500,
            )

    @app.route("/escalations", methods=["POST"])
    def create_escalation() -> tuple[Response, int]:
        """Create a human-expert task for a question lacking safe evidence."""
        if not request.is_json:
            return jsonify({"error": "Invalid Content-Type; request must be application/json"}), 400
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict) or not isinstance(payload.get("question"), str):
            return jsonify({"error": "Field 'question' must be a non-empty string"}), 400
        question = payload["question"].strip()
        if not question:
            return jsonify({"error": "Field 'question' must be a non-empty string"}), 400

        path = Path(app.config["ESCALATIONS_PATH"])
        item = {
            "id": str(uuid4()),
            "question": question,
            "reason": str(payload.get("reason", "insufficient_evidence")),
            "evidence": payload.get("evidence", []),
            "status": "open",
            "approved_answer": None,
        }
        items = _load_escalations(path)
        items.append(item)
        _save_escalations(path, items)
        return jsonify(item), 201

    @app.route("/escalations/<escalation_id>/resolve", methods=["POST"])
    def resolve_escalation(escalation_id: str) -> tuple[Response, int]:
        """Record an approved expert answer; it remains auditable for later ingestion."""
        if not request.is_json:
            return jsonify({"error": "Invalid Content-Type; request must be application/json"}), 400
        payload = request.get_json(silent=True)
        answer = payload.get("answer") if isinstance(payload, dict) else None
        if not isinstance(answer, str) or not answer.strip():
            return jsonify({"error": "Field 'answer' must be a non-empty string"}), 400
        path = Path(app.config["ESCALATIONS_PATH"])
        items = _load_escalations(path)
        for item in items:
            if item.get("id") == escalation_id:
                item["status"] = "resolved"
                item["approved_answer"] = answer.strip()
                item["supporting_url"] = payload.get("supporting_url")
                _save_escalations(path, items)
                return jsonify(item), 200
        return jsonify({"error": "Escalation not found"}), 404

    return app


# Default module-level application instance for WSGI / CLI runners
app = create_app()


def parse_api_cli_args(args: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command line arguments for the Knowledge Brain API."""
    parser = argparse.ArgumentParser(
        description="Knowledge Brain HTTP API Server (Task 1 Component)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default=os.getenv("APP_HOST", "127.0.0.1"),
        help="Host interface to bind (default: 127.0.0.1 or APP_HOST env var)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("APP_PORT", "8501")),
        help="Port to bind (default: 8501 or APP_PORT env var)",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to articles JSON file (defaults to knowledge_brain/data/sample_articles.json)",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        default=False,
        help="Force mock AI mode regardless of environment variables",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Enable Flask debug mode",
    )
    return parser.parse_args(args)


def main(args: Optional[Sequence[str]] = None) -> None:
    """CLI entrypoint for running the Knowledge Brain HTTP API server."""
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parsed = parse_api_cli_args(args)
    logger.info("Initializing Knowledge Brain API on http://%s:%d ...", parsed.host, parsed.port)

    custom_retriever = (
        KnowledgeRetriever(articles_path=parsed.data) if parsed.data else None
    )
    custom_answerer = KnowledgeAnswerer(use_mock=True) if parsed.mock else None

    server_app = create_app(
        retriever=custom_retriever,
        answerer=custom_answerer,
    )
    server_app.run(
        host=parsed.host,
        port=parsed.port,
        debug=parsed.debug,
    )


if __name__ == "__main__":
    main()
