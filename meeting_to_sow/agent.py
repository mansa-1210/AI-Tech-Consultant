"""Meeting-to-SOW workflow. Estimates are planning assumptions and require human approval."""
import json, os, re
from pathlib import Path
import requests
ROOT=Path(__file__).resolve().parent
DATA_DIR=ROOT/"data"
TRANSCRIPTS_PATH=DATA_DIR/"meeting_transcripts.json"
KB_URL=os.getenv("KNOWLEDGE_BRAIN_URL","http://127.0.0.1:8501/ask")
# Transparent initial planning ranges, not vendor-validated quotes.
TASK_ESTIMATES={
"requirements and acceptance criteria":(3,5),
"configuration and implementation":(8,16),
"data import and validation":(4,8),
"custom integration discovery":(4,8),
"custom integration implementation":(12,24),
"notifications discovery":(2,4),
"notification implementation":(4,10),
"QA and user acceptance testing":(5,8),
"deployment and handover":(3,5),
}
CUSTOM_TERMS=("custom integration","salesforce","crm connection","sms notification","sms notifications","custom dashboard")
PRODUCT_TERMS=("email notifications","member import","csv import","contact form","instructor profiles","agent profiles","property listings","search page")
def load_meetings():
    return json.loads(TRANSCRIPTS_PATH.read_text(encoding="utf-8"))
def ask_knowledge_brain(question):
    try:
        r=requests.post(KB_URL,json={"query":question},timeout=8); r.raise_for_status()
        data=r.json()
        if isinstance(data,dict): return data
    except (requests.RequestException,ValueError): pass
    return {"answer":"","evidence_found":False,"citations":[],"evidence_status":"unavailable","is_mock":True}
def extract_features(transcript):
    t=transcript.lower().replace("custom crm integration", "custom integration")
    found=[]
    for term in PRODUCT_TERMS:
        if term in t: found.append({"feature":term,"initial_classification":"candidate_product_configuration"})
    for term in CUSTOM_TERMS:
        if term in t and not any(x["feature"]==term for x in found):
            found.append({"feature":term,"initial_classification":"candidate_custom_or_needs_review"})
    # de-duplicate and preserve order
    seen=set(); return [x for x in found if not (x["feature"] in seen or seen.add(x["feature"]))]
def estimate_tasks(features):
    tasks=[{"task":"requirements and acceptance criteria","hours_min":3,"hours_max":5,"assumption":"Planning estimate; validate in discovery."}]
    for item in features:
        feature=item["feature"]
        if any(k in feature for k in ("import","csv")):
            tasks.append({"task":"data import and validation","hours_min":4,"hours_max":8,"assumption":"Assumes clean CSV and agreed field mapping."})
        elif any(k in feature for k in ("integration","salesforce","crm")):
            tasks.append({"task":"custom integration discovery","hours_min":4,"hours_max":8,"assumption":"API access and documentation must be confirmed."})
            tasks.append({"task":"custom integration implementation","hours_min":12,"hours_max":24,"assumption":"High uncertainty; depends on API/authentication and data mapping."})
        elif any(k in feature for k in ("notification","email")):
            tasks.append({"task":"notifications discovery","hours_min":2,"hours_max":4,"assumption":"Confirm channels, triggers, consent and templates."})
            tasks.append({"task":"notification implementation","hours_min":4,"hours_max":10,"assumption":"Channel support and provider configuration need validation."})
        else:
            tasks.append({"task":"configuration and implementation","hours_min":8,"hours_max":16,"assumption":"Depends on exact requirements and existing platform capabilities."})
    tasks.extend([
      {"task":"QA and user acceptance testing","hours_min":5,"hours_max":8,"assumption":"Includes agreed acceptance criteria and one review cycle."},
      {"task":"deployment and handover","hours_min":3,"hours_max":5,"assumption":"Assumes access and client review are timely."}])
    return tasks
def create_scope(meeting):
    transcript=meeting.get("transcript","")
    features=extract_features(transcript)
    checks=[]
    for item in features:
        result=ask_knowledge_brain("Is this supported by standard Brilliant Directories product settings or does it require custom work? Feature: "+item["feature"])
        evidence=bool(result.get("evidence_found")) and bool(result.get("answer")) and bool(result.get("citations"))
        checks.append({"feature":item["feature"],"initial_classification":item["initial_classification"],
          "feasibility":"evidence_found_needs_human_interpretation" if evidence else "unknown_needs_human_review",
          "knowledge_answer":result.get("answer","") if evidence else "",
          "citations":result.get("citations",[]) if evidence else [],
          "is_mock":bool(result.get("is_mock",True))})
    tasks=estimate_tasks(features)
    return {"meeting_id":meeting.get("meeting_id"),"client":meeting.get("client"),"scope_version":1,
      "source_transcript":transcript,"features":checks,"tasks":tasks,
      "hours_min":sum(t["hours_min"] for t in tasks),"hours_max":sum(t["hours_max"] for t in tasks),
      "assumptions":["Estimates are preliminary planning ranges, not a fixed quote.","Unknown product feasibility must be verified by a human using cited documentation.","Custom integrations require API, access, security and data-mapping discovery."],
      "risks":["Unverified feature feasibility","Scope and third-party API uncertainty","Client access or feedback delays"],
      "approval_status":"pending_human_approval","client_send_allowed":False,"is_mock_or_unverified":any(x["is_mock"] or x["feasibility"].startswith("unknown") for x in checks)}
def compare_follow_up(original_scope, followup_meeting):
    new_features=extract_features(followup_meeting.get("transcript",""))
    original={x["feature"] for x in original_scope.get("features",[])}
    text=followup_meeting.get("transcript","").lower()
    # Detect explicit removal/replacement phrases and added requests.
    changes=[]
    for feat in original:
        if (("remove "+feat) in text or ("remove "+feat.replace(" "," ")) in text or ("instead of "+feat) in text):
            changes.append({"feature":feat,"change":"removed_or_replaced","scope_impact":"Remove from planned deliverables; confirm replacement."})
    for item in new_features:
        if item["feature"] not in original:
            changes.append({"feature":item["feature"],"change":"added","scope_impact":"New requirement; estimate and feasibility review needed."})
    if "instead of email notifications" in text:
        changes.append({"feature":"email notifications","change":"replaced","scope_impact":"Confirm whether email notifications are removed and SMS is the replacement."})
        if not any(c["feature"]=="sms notifications" for c in changes):
            changes.append({"feature":"sms notifications","change":"added","scope_impact":"New channel needs feasibility, provider and consent review."})
    return {"original_meeting_id":original_scope.get("meeting_id"),"follow_up_meeting_id":followup_meeting.get("meeting_id"),
      "changes":changes,"requires_reestimate":bool(changes),"approval_status":"pending_human_approval","client_send_allowed":False}
def load_transcript_by_id(meeting_id):
    for m in load_meetings():
        if m["meeting_id"]==meeting_id: return m
    raise KeyError(f"Unknown meeting id: {meeting_id}")
if __name__=="__main__":
    for meeting in load_meetings():
        if meeting.get("type")=="initial":
            scope=create_scope(meeting)
            print(json.dumps(scope,indent=2,ensure_ascii=False))
