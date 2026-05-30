# Databricks Notebook
# ClinicalPath · Microsoft Teams Bot
# Adaptive Card Q&A → Decision Engine → Teams Response
# ─────────────────────────────────────────────────────
# Depends on: 02_decision_engine (run_decision, preauth_request)
#             medschm.silver.protocol_rules  (protocol lookup)
#             medschm.gold.preauth_decisions (decision history)
# Deploy:     Databricks Apps  OR  Azure Container Apps
# ─────────────────────────────────────────────────────

# COMMAND ----------
# %pip install botframework-connector botbuilder-core botbuilder-schema \
#              botbuilder-dialogs aiohttp fastapi uvicorn openai
# dbutils.library.restartPython()

# COMMAND ----------
# ── 1. IMPORTS & CONFIG ──────────────────────────────

import json, hashlib
from datetime import datetime

from openai import AzureOpenAI
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

import aiohttp
from fastapi import FastAPI, Request, Response
import uvicorn

from botbuilder.core import (
    BotFrameworkAdapter, BotFrameworkAdapterSettings,
    TurnContext, MessageFactory
)
from botbuilder.schema import Activity, ActivityTypes

spark = SparkSession.builder.getOrCreate()

# ── Secrets ───────────────────────────────────────────
BOT_APP_ID     = dbutils.secrets.get("kv-medschm", "teams-bot-app-id")
BOT_APP_SECRET = dbutils.secrets.get("kv-medschm", "teams-bot-app-secret")
AZURE_OAI_ENDPOINT = dbutils.secrets.get("kv-medschm", "azure-oai-endpoint")
AZURE_OAI_KEY      = dbutils.secrets.get("kv-medschm", "azure-oai-key")
CHAT_DEPLOYMENT    = "gpt-4o"

client = AzureOpenAI(
    azure_endpoint=AZURE_OAI_ENDPOINT,
    api_key=AZURE_OAI_KEY,
    api_version="2024-02-01",
)

adapter_settings = BotFrameworkAdapterSettings(BOT_APP_ID, BOT_APP_SECRET)
adapter = BotFrameworkAdapter(adapter_settings)

# COMMAND ----------
# ── 2. INTENT CLASSIFIER ─────────────────────────────
# Classifies free-text Teams messages into one of four intents.

INTENT_PROMPT = """
Classify the following Teams message from a managed care analyst into one intent.

Intents:
  preauth_request  — analyst wants to submit or check a pre-authorisation
  protocol_query   — analyst asks about a protocol, ICD-10 codes, basket of care
  decision_lookup  — analyst asks about a previous decision or case reference
  general          — anything else

Reply with a single JSON object: {"intent": "<intent>", "entities": {<key fields extracted>}}
Extract entities where present: member_id, protocol_id, icd10_code, case_ref, question.

Message: "{message}"
"""

def classify_intent(message: str) -> dict:
    resp = client.chat.completions.create(
        model=CHAT_DEPLOYMENT,
        response_format={"type": "json_object"},
        temperature=0,
        messages=[{
            "role": "user",
            "content": INTENT_PROMPT.format(message=message)
        }]
    )
    return json.loads(resp.choices[0].message.content)

# COMMAND ----------
# ── 3. ADAPTIVE CARDS ────────────────────────────────

def card_preauth_form(protocols: list[dict]) -> dict:
    """Input form — analyst fills member + clinical fields."""
    protocol_choices = [
        {"title": f"{p['protocol_id']} — {p['protocol_name']}", "value": p["protocol_id"]}
        for p in protocols
    ]
    return {
        "type": "AdaptiveCard",
        "version": "1.5",
        "body": [
            {
                "type": "TextBlock",
                "text": "🏥 Pre-Authorisation Request",
                "size": "Large",
                "weight": "Bolder",
                "color": "Accent"
            },
            {
                "type": "ColumnSet",
                "columns": [
                    {
                        "type": "Column", "width": "stretch",
                        "items": [
                            {"type": "TextBlock", "text": "Member ID", "weight": "Bolder"},
                            {
                                "type": "Input.Text",
                                "id": "member_id",
                                "placeholder": "MBR-00000"
                            },
                            {"type": "TextBlock", "text": "ICD-10 Code", "weight": "Bolder", "spacing": "Medium"},
                            {
                                "type": "Input.Text",
                                "id": "icd10_code",
                                "placeholder": "e.g. D59.1"
                            },
                        ]
                    },
                    {
                        "type": "Column", "width": "stretch",
                        "items": [
                            {"type": "TextBlock", "text": "Scheme", "weight": "Bolder"},
                            {
                                "type": "Input.ChoiceSet",
                                "id": "scheme",
                                "style": "compact",
                                "choices": [
                                    {"title": "MedSchm Classic",   "value": "MedSchm Classic"},
                                    {"title": "MedSchm Saver",     "value": "MedSchm Saver"},
                                    {"title": "MedSchm Essential", "value": "MedSchm Essential"},
                                ]
                            },
                            {"type": "TextBlock", "text": "Treating Discipline", "weight": "Bolder", "spacing": "Medium"},
                            {
                                "type": "Input.ChoiceSet",
                                "id": "treating_practitioner_discipline",
                                "style": "compact",
                                "choices": [
                                    {"title": "General Practitioner",  "value": "general_practitioner"},
                                    {"title": "Physician",             "value": "physician"},
                                    {"title": "Haematologist",         "value": "haematologist"},
                                    {"title": "Paediatrician",         "value": "paediatrician"},
                                    {"title": "Specialist Physician",  "value": "specialist_physician"},
                                ]
                            },
                        ]
                    }
                ]
            },
            {
                "type": "TextBlock",
                "text": "Protocol",
                "weight": "Bolder",
                "spacing": "Medium"
            },
            {
                "type": "Input.ChoiceSet",
                "id": "protocol_id",
                "style": "compact",
                "placeholder": "Select protocol",
                "choices": protocol_choices
            },
            {
                "type": "ColumnSet",
                "spacing": "Medium",
                "columns": [
                    {
                        "type": "Column", "width": "stretch",
                        "items": [
                            {"type": "TextBlock", "text": "Member registered?", "weight": "Bolder"},
                            {
                                "type": "Input.Toggle",
                                "id": "member_registered",
                                "title": "Yes",
                                "valueOn": "true",
                                "valueOff": "false",
                                "value": "true"
                            }
                        ]
                    },
                    {
                        "type": "Column", "width": "stretch",
                        "items": [
                            {"type": "TextBlock", "text": "Active membership?", "weight": "Bolder"},
                            {
                                "type": "Input.Toggle",
                                "id": "active_membership",
                                "title": "Yes",
                                "valueOn": "true",
                                "valueOff": "false",
                                "value": "true"
                            }
                        ]
                    },
                    {
                        "type": "Column", "width": "stretch",
                        "items": [
                            {"type": "TextBlock", "text": "Chronic auth confirmed?", "weight": "Bolder"},
                            {
                                "type": "Input.Toggle",
                                "id": "chronic_auth_confirmed",
                                "title": "Yes",
                                "valueOn": "true",
                                "valueOff": "false",
                                "value": "true"
                            }
                        ]
                    },
                    {
                        "type": "Column", "width": "stretch",
                        "items": [
                            {"type": "TextBlock", "text": "Care in basket?", "weight": "Bolder"},
                            {
                                "type": "Input.Toggle",
                                "id": "care_items_in_basket",
                                "title": "Yes",
                                "valueOn": "true",
                                "valueOff": "false",
                                "value": "true"
                            }
                        ]
                    },
                ]
            },
            {
                "type": "ColumnSet",
                "spacing": "Small",
                "columns": [
                    {
                        "type": "Column", "width": "stretch",
                        "items": [
                            {"type": "TextBlock", "text": "Comorbid autoimmune condition?", "weight": "Bolder"},
                            {
                                "type": "Input.Toggle",
                                "id": "comorbid_autoimmune",
                                "title": "Yes",
                                "valueOn": "true",
                                "valueOff": "false",
                                "value": "false"
                            }
                        ]
                    },
                    {
                        "type": "Column", "width": "stretch",
                        "items": [
                            {"type": "TextBlock", "text": "Clinical motivation submitted?", "weight": "Bolder"},
                            {
                                "type": "Input.Toggle",
                                "id": "clinical_motivation_submitted",
                                "title": "Yes",
                                "valueOn": "true",
                                "valueOff": "false",
                                "value": "false"
                            }
                        ]
                    },
                ]
            }
        ],
        "actions": [
            {
                "type": "Action.Submit",
                "title": "⚡ Run Decision Engine",
                "data": {"action": "run_preauth"}
            }
        ]
    }


def card_decision_result(result: dict) -> dict:
    """Decision output card — colour-coded by outcome."""
    outcome   = result["outcome"]
    confidence = int(result["confidence"] * 100)

    colour_map = {
        "APPROVE":                "Good",
        "REJECT":                 "Attention",
        "REFER_MEDICAL_ADVISOR":  "Warning",
        "ERROR":                  "Attention",
    }
    icon_map = {
        "APPROVE":               "✅",
        "REJECT":                "❌",
        "REFER_MEDICAL_ADVISOR": "⚠️",
        "ERROR":                 "🔴",
    }
    label_map = {
        "APPROVE":               "PRE-AUTHORISATION APPROVED",
        "REJECT":                "PRE-AUTHORISATION REJECTED",
        "REFER_MEDICAL_ADVISOR": "REFERRED TO MEDICAL ADVISOR",
        "ERROR":                 "ENGINE ERROR",
    }

    colour = colour_map.get(outcome, "Default")
    icon   = icon_map.get(outcome, "❓")
    label  = label_map.get(outcome, outcome)

    flags = result.get("flags", {})
    flag_lines = []
    for fkey, flist in flags.items():
        if flist:
            flag_lines.append(f"**{fkey.replace('_',' ').title()}:** {', '.join(flist)}")

    facts = [
        {"title": "Case Ref",    "value": result["case_ref"]},
        {"title": "Protocol",    "value": result.get("protocol_id", "—")},
        {"title": "Member",      "value": result["member_id"]},
        {"title": "Confidence",  "value": f"{confidence}%"},
        {"title": "Decided At",  "value": result["decided_at"][:19].replace("T", " ")},
    ]

    body = [
        {
            "type": "TextBlock",
            "text": f"{icon} {label}",
            "size": "Large",
            "weight": "Bolder",
            "color": colour,
            "wrap": True
        },
        {
            "type": "FactSet",
            "facts": facts
        },
        {
            "type": "TextBlock",
            "text": "**Clinical Rationale**",
            "weight": "Bolder",
            "spacing": "Medium"
        },
        {
            "type": "TextBlock",
            "text": result["rationale"],
            "wrap": True,
            "color": "Default"
        },
    ]

    if flag_lines:
        body.append({
            "type": "TextBlock",
            "text": "**Decision Flags**",
            "weight": "Bolder",
            "spacing": "Medium"
        })
        for fl in flag_lines:
            body.append({"type": "TextBlock", "text": fl, "wrap": True, "spacing": "Small"})

    actions = []
    if outcome == "APPROVE":
        actions.append({
            "type": "Action.Submit",
            "title": "✅ Confirm & Issue Auth",
            "style": "positive",
            "data": {"action": "confirm_auth", "case_ref": result["case_ref"]}
        })
    if outcome in ("APPROVE", "REJECT", "REFER_MEDICAL_ADVISOR"):
        actions.append({
            "type": "Action.Submit",
            "title": "🔄 Override Decision",
            "data": {"action": "override", "case_ref": result["case_ref"]}
        })
    actions.append({
        "type": "Action.Submit",
        "title": "📋 New Request",
        "data": {"action": "new_request"}
    })

    return {
        "type": "AdaptiveCard",
        "version": "1.5",
        "body": body,
        "actions": actions
    }


def card_protocol_info(protocol: dict) -> dict:
    """Protocol Q&A answer card."""
    icd_lines = " · ".join(
        f"{i['code']} {i['description']}"
        for i in (protocol.get("icd10_codes") or [])
    )
    basket_lines = "\n".join(
        f"- {item['description']} ({item['discipline']}) — codes: {', '.join(item['billing_codes'])}"
        for item in (protocol.get("basket_of_care") or [])
    )
    return {
        "type": "AdaptiveCard",
        "version": "1.5",
        "body": [
            {
                "type": "TextBlock",
                "text": f"📘 {protocol['protocol_name']} ({protocol['protocol_id']})",
                "size": "Large", "weight": "Bolder", "color": "Accent", "wrap": True
            },
            {"type": "TextBlock", "text": protocol.get("therapeutic_summary", ""), "wrap": True},
            {"type": "TextBlock", "text": "**ICD-10 Codes**", "weight": "Bolder", "spacing": "Medium"},
            {"type": "TextBlock", "text": icd_lines or "—", "wrap": True},
            {"type": "TextBlock", "text": "**Basket of Care**", "weight": "Bolder", "spacing": "Medium"},
            {"type": "TextBlock", "text": basket_lines or "—", "wrap": True, "fontType": "Monospace"},
        ],
        "actions": [
            {
                "type": "Action.Submit",
                "title": "🏥 Submit Pre-Auth Request",
                "data": {"action": "new_request", "protocol_id": protocol["protocol_id"]}
            }
        ]
    }


def card_error(message: str) -> dict:
    return {
        "type": "AdaptiveCard",
        "version": "1.5",
        "body": [
            {"type": "TextBlock", "text": f"🔴 {message}", "color": "Attention", "wrap": True}
        ]
    }

# COMMAND ----------
# ── 4. BOT HANDLER ───────────────────────────────────

def _load_protocols() -> list[dict]:
    return (
        spark.table("medschm.silver.protocol_rules")
        .select("protocol_id", "protocol_name", "therapeutic_summary",
                "icd10_codes", "basket_of_care")
        .collect()
    )


def _coerce_booleans(data: dict) -> dict:
    """Convert 'true'/'false' strings from Adaptive Card toggles to Python bool."""
    bool_fields = [
        "member_registered", "active_membership", "chronic_auth_confirmed",
        "care_items_in_basket", "comorbid_autoimmune", "clinical_motivation_submitted"
    ]
    for f in bool_fields:
        if f in data:
            data[f] = data[f] in (True, "true", "True", 1, "1")
    return data


async def handle_message(turn_context: TurnContext):
    activity = turn_context.activity

    # ── Adaptive Card submit ──────────────────────────
    if activity.type == ActivityTypes.message and activity.value:
        data   = activity.value
        action = data.get("action", "")

        if action == "run_preauth":
            data = _coerce_booleans(data)
            case_ref = "PA-" + hashlib.md5(
                (data.get("member_id","") + datetime.utcnow().isoformat()).encode()
            ).hexdigest()[:8].upper()

            result = preauth_request(
                case_ref=case_ref,
                protocol_id=data.get("protocol_id"),
                member_id=data.get("member_id"),
                case_fields={k: v for k, v in data.items()
                             if k not in ("action", "protocol_id", "member_id")}
            )
            card = card_decision_result(result)
            await turn_context.send_activity(
                MessageFactory.attachment({
                    "contentType": "application/vnd.microsoft.card.adaptive",
                    "content": card
                })
            )
            return

        if action in ("new_request",):
            protocols = [dict(r.asDict()) for r in _load_protocols()]
            card = card_preauth_form(protocols)
            await turn_context.send_activity(
                MessageFactory.attachment({
                    "contentType": "application/vnd.microsoft.card.adaptive",
                    "content": card
                })
            )
            return

        if action == "confirm_auth":
            await turn_context.send_activity(
                MessageFactory.text(
                    f"✅ Auth issued for case **{data.get('case_ref')}**. "
                    f"Member and treating practitioner have been notified."
                )
            )
            return

    # ── Free-text message ─────────────────────────────
    if activity.type == ActivityTypes.message and activity.text:
        text   = activity.text.strip()
        intent = classify_intent(text)
        action = intent.get("intent")
        entities = intent.get("entities", {})

        # ── protocol_query ────────────────────────────
        if action == "protocol_query":
            pid = entities.get("protocol_id")
            if pid:
                row = (
                    spark.table("medschm.silver.protocol_rules")
                    .filter(F.col("protocol_id") == pid)
                    .first()
                )
                if row:
                    card = card_protocol_info(dict(row.asDict()))
                    await turn_context.send_activity(
                        MessageFactory.attachment({
                            "contentType": "application/vnd.microsoft.card.adaptive",
                            "content": card
                        })
                    )
                    return
            # Fallback: RAG answer via GPT-4o
            rag_answer = _rag_answer(text)
            await turn_context.send_activity(MessageFactory.text(rag_answer))
            return

        # ── decision_lookup ───────────────────────────
        if action == "decision_lookup":
            case_ref = entities.get("case_ref")
            if case_ref:
                row = (
                    spark.table("medschm.gold.preauth_decisions")
                    .filter(F.col("case_ref") == case_ref)
                    .orderBy(F.col("decided_at").desc())
                    .first()
                )
                if row:
                    r = row.asDict()
                    summary = (
                        f"**Case {r['case_ref']}** — {r['outcome']} "
                        f"({int(float(r['confidence'])*100)}% confidence)\n\n"
                        f"{r['rationale']}"
                    )
                    await turn_context.send_activity(MessageFactory.text(summary))
                    return
            await turn_context.send_activity(
                MessageFactory.text("I couldn't find that case reference. Please check the case ref and try again.")
            )
            return

        # ── preauth_request → open the form ──────────
        if action == "preauth_request":
            protocols = [dict(r.asDict()) for r in _load_protocols()]
            card = card_preauth_form(protocols)
            await turn_context.send_activity(
                MessageFactory.attachment({
                    "contentType": "application/vnd.microsoft.card.adaptive",
                    "content": card
                })
            )
            return

        # ── general / fallback ────────────────────────
        rag_answer = _rag_answer(text)
        await turn_context.send_activity(MessageFactory.text(rag_answer))


def _rag_answer(question: str) -> str:
    """Answer a free-text protocol question using GPT-4o + Delta table context."""
    protocols = spark.table("medschm.silver.protocol_rules").collect()
    context = "\n\n".join(
        f"Protocol {r['protocol_id']} — {r['protocol_name']}:\n{r['therapeutic_summary']}"
        for r in protocols
    )
    resp = client.chat.completions.create(
        model=CHAT_DEPLOYMENT,
        temperature=0.2,
        max_tokens=300,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a clinical protocol assistant for a South African managed care PA engine. "
                    "Answer concisely using only the provided protocol context. "
                    "If unsure, say so — do not invent clinical information."
                )
            },
            {
                "role": "user",
                "content": f"Protocol context:\n{context[:6000]}\n\nQuestion: {question}"
            }
        ]
    )
    return resp.choices[0].message.content.strip()

# COMMAND ----------
# ── 5. FASTAPI APP ───────────────────────────────────

app = FastAPI(title="ClinicalPath Teams Bot")

@app.post("/api/messages")
async def messages(request: Request):
    body    = await request.json()
    headers = dict(request.headers)
    activity = Activity().deserialize(body)

    auth_header = headers.get("authorization", "")

    async def call_bot(turn_context: TurnContext):
        await handle_message(turn_context)

    try:
        await adapter.process_activity(activity, auth_header, call_bot)
        return Response(status_code=200)
    except Exception as e:
        return Response(content=str(e), status_code=500)

@app.get("/health")
def health():
    return {"status": "ok", "service": "ClinicalPath Teams Bot"}

# COMMAND ----------
# ── 6. START SERVER (Databricks Apps) ───────────────
# In Databricks Apps, the entry point is the `app` FastAPI object above.
# For local testing, uncomment the line below:

# uvicorn.run(app, host="0.0.0.0", port=3978)

print("""
╔══════════════════════════════════════════════════╗
║  ClinicalPath Teams Bot — ready                  ║
║                                                  ║
║  POST /api/messages  ←  Teams webhook            ║
║  GET  /health        ←  liveness probe           ║
║                                                  ║
║  Register this URL in Azure Bot Service          ║
║  under Messaging Endpoint                        ║
╚══════════════════════════════════════════════════╝
""")
