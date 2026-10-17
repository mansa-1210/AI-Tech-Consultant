"""Grounded Inbox Agent with human review and simulated autonomy."""
import json, os
from pathlib import Path
import requests
ROOT=Path(__file__).resolve().parent
DATA_DIR=ROOT/"data"
EMAILS_PATH=DATA_DIR/"emails.json"
STATE_PATH=DATA_DIR/"review_state.json"
FEEDBACK_PATH=DATA_DIR/"feedback.jsonl"
KB_URL=os.getenv("KNOWLEDGE_BRAIN_URL","http://127.0.0.1:8501/ask")
AUTO_SEND_THRESHOLD=20
RULES={
"billing":("invoice","billing","payment","refund","charge","subscription","price"),
"custom_work":("custom","build","integration","develop","new feature","widget","crm"),
"account_access":("login","log in","password","access","locked"),
"support":("email","domain","website","member","import","campaign","forward")}
def classify_email(email):
    text=(email.get("subject","")+" "+email.get("body","")).lower()
    scores={c:sum(w in text for w in words) for c,words in RULES.items()}
    best=max(scores,key=scores.get)
    return best if scores[best] else "general"
def load_emails():
    return json.loads(EMAILS_PATH.read_text(encoding="utf-8"))
def load_state():
    try: return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except (OSError,ValueError): return {"consecutive_approvals":0,"autonomy_enabled":False,"last_category":None}
def save_state(state):
    DATA_DIR.mkdir(parents=True,exist_ok=True)
    STATE_PATH.write_text(json.dumps(state,indent=2),encoding="utf-8")
def ask_knowledge_brain(question):
    try:
        r=requests.post(KB_URL,json={"query":question},timeout=8); r.raise_for_status()
        data=r.json()
        if isinstance(data,dict): return data
    except (requests.RequestException,ValueError): pass
    return {"answer":"","evidence_found":False,"evidence_status":"unavailable","citations":[],"is_mock":True}
def prepare_draft(email,kb_result=None):
    category=classify_email(email)
    kb=kb_result if kb_result is not None else ask_knowledge_brain(email.get("subject","")+"\n"+email.get("body",""))
    grounded=bool(kb.get("evidence_found")) and bool(kb.get("answer")) and bool(kb.get("citations"))
    if grounded:
        answer=str(kb["answer"]).replace("[MOCK MODE]","").strip()
        draft="Hello,\n\nThank you for contacting us. Based on our support documentation:\n\n"+answer+"\n\nPlease let us know if you need further help.\n\nBest regards,\nSupport Team"
        status="grounded_draft"
    else:
        draft="Hello,\n\nThank you for reaching out. To avoid giving inaccurate information, I’m escalating your question to our support team for confirmation. We’ll follow up once we have a verified answer.\n\nBest regards,\nSupport Team"
        status="human_escalation"
    return {"email_id":email.get("id"),"category":category,"draft":draft,"status":status,"citations":kb.get("citations",[]) if grounded else [],"evidence_status":kb.get("evidence_status","unknown"),"mode":kb.get("mode","unknown"),"is_mock":bool(kb.get("is_mock",True)),"review_required":True,"auto_send_eligible":False}
def record_review(email,draft,decision,edited_draft=None):
    if decision not in {"approve","edit","reject"}: raise ValueError("decision must be approve, edit, or reject")
    state=load_state(); category=draft.get("category",classify_email(email))
    if decision=="approve":
        state["consecutive_approvals"]=state.get("consecutive_approvals",0)+1 if state.get("last_category")==category else 1
        state["last_category"]=category
    else:
        state["consecutive_approvals"]=0; state["autonomy_enabled"]=False; state["last_category"]=category
    if state["consecutive_approvals"]>=AUTO_SEND_THRESHOLD: state["autonomy_enabled"]=True
    save_state(state)
    FEEDBACK_PATH.parent.mkdir(parents=True,exist_ok=True)
    record={"email_id":email.get("id"),"category":category,"decision":decision,"edited":decision=="edit","edited_draft":edited_draft if decision=="edit" else None,"draft_status":draft.get("status"),"simulated_auto_send":bool(state["autonomy_enabled"] and decision=="approve")}
    with FEEDBACK_PATH.open("a",encoding="utf-8") as f: f.write(json.dumps(record,ensure_ascii=False)+"\n")
    return {"recorded":True,"state":state,"feedback":record,"notice":"SIMULATION ONLY: no email was sent."}
if __name__=="__main__":
    data=load_emails(); print(f"Loaded {len(data)} synthetic emails.")
    for email in data[:5]:
        result=prepare_draft(email); print(email["id"],result["category"],result["status"]); print(result["draft"])
