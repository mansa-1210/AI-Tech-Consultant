"""One-command local console for all three AI Tech Consultant workflows."""
from __future__ import annotations

import argparse
from typing import Any

from flask import Flask, jsonify, request, render_template_string

from inbox_agent import agent as inbox
from knowledge_brain.api import create_app as create_knowledge_app
from meeting_to_sow import agent as meetings


CONSOLE_HTML = """<!doctype html><title>AI Tech Consultant</title>
<style>body{max-width:1050px;margin:2rem auto;font:16px system-ui;line-height:1.5;background:#fafafa;color:#18212f}section{background:white;border:1px solid #dde3ea;border-radius:10px;padding:1rem;margin:1rem 0}textarea,select,button{font:inherit;padding:.55rem;margin:.2rem}textarea{width:96%;height:72px}pre{white-space:pre-wrap;background:#f3f6f8;padding:1rem;border-radius:6px;max-height:420px;overflow:auto}h1{margin-bottom:0}.note{color:#566}</style>
<h1>AI Tech Consultant</h1><p class=note>One shared Knowledge Brain; all customer-facing results require human approval.</p>
<section><h2>1. Knowledge Brain</h2><textarea id=q placeholder="Where are mailboxes managed?"></textarea><button onclick="ask()">Ask</button></section>
<section><h2>2. Inbox Agent</h2><select id=email></select><button onclick="draft()">Draft response</button></section>
<section><h2>3. Meeting to Scope</h2><select id=meeting></select><button onclick="scope()">Create scope + estimate</button></section>
<pre id=out>Loading examples…</pre>
<script>const out=document.querySelector('#out');async function call(url,body){let r=await fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});out.textContent=JSON.stringify(await r.json(),null,2)}async function ask(){call('/ask',{query:q.value})}async function draft(){call('/inbox/draft',{email_id:email.value})}async function scope(){call('/meetings/scope',{meeting_id:meeting.value})}async function load(){let e=await (await fetch('/inbox/emails')).json();email.innerHTML=e.map(x=>`<option value="${x.id}">${x.id}: ${x.subject}</option>`).join('');let m=await (await fetch('/meetings')).json();meeting.innerHTML=m.filter(x=>x.type==='initial').map(x=>`<option value="${x.meeting_id}">${x.client}</option>`).join('');out.textContent='Choose a workflow above.'}load()</script>"""


def create_app() -> Flask:
    app = create_knowledge_app()

    @app.get("/")
    def console() -> str:
        return render_template_string(CONSOLE_HTML)

    @app.get("/inbox/emails")
    def inbox_emails() -> Any:
        return jsonify(inbox.load_emails())

    @app.post("/inbox/draft")
    def inbox_draft() -> Any:
        payload = request.get_json(silent=True) or {}
        email_id = payload.get("email_id")
        email = next((item for item in inbox.load_emails() if item.get("id") == email_id), None)
        if email is None:
            return jsonify({"error": "Unknown email_id"}), 404
        return jsonify(inbox.prepare_draft(email))

    @app.post("/inbox/review")
    def inbox_review() -> Any:
        payload = request.get_json(silent=True) or {}
        email = payload.get("email")
        draft = payload.get("draft")
        decision = payload.get("decision")
        if not isinstance(email, dict) or not isinstance(draft, dict):
            return jsonify({"error": "email and draft objects are required"}), 400
        try:
            return jsonify(inbox.record_review(email, draft, decision, payload.get("edited_draft"), payload.get("final_sent_draft")))
        except ValueError as error:
            return jsonify({"error": str(error)}), 400

    @app.get("/meetings")
    def list_meetings() -> Any:
        return jsonify(meetings.load_meetings())

    @app.post("/meetings/scope")
    def meeting_scope() -> Any:
        payload = request.get_json(silent=True) or {}
        try:
            scope = meetings.create_scope(meetings.load_transcript_by_id(payload.get("meeting_id", "")))
        except KeyError:
            return jsonify({"error": "Unknown meeting_id"}), 404
        return jsonify({"scope": scope, "proposal": meetings.create_proposal(scope)})

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all AI Tech Consultant workflows.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8501)
    args = parser.parse_args()
    create_app().run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
