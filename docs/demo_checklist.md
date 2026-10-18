# Demo checklist

Keep the recording under 10 minutes. Start with your face on camera for the first 30 seconds.

1. Introduce the business problem and show the one-command start: `powershell -ExecutionPolicy Bypass -File .\start.ps1`.
2. Show `/health` reporting the selected provider, model, and loaded corpus size.
3. Ask one supported Knowledge Brain question and point out its citations and any labelled visual evidence.
4. Ask one unsupported question and show the explicit insufficient-evidence response and human escalation path.
5. Open a user-authored Inbox email, show its cited draft, manual review record, and why no email is actually sent.
6. Show a Meeting-to-SOW transcript, generated scope, estimate range, human-approval status, and the follow-up scope change.
7. Show the daily-sync command without running it: `python -m knowledge_brain.sync`.
8. Show the evaluation result files for both application models, including quality checks, latency, token usage, cost, failures, and safety outcomes.
9. State the remaining limitations honestly: visual descriptions without a `vision_description` field are source alt text, and autonomy remains simulation-only.
