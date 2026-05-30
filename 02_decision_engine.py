# Databricks Notebook
# ClinicalPath · Pre-Auth Decision Engine
# Protocol Criteria Evaluator → APPROVE / REJECT / REFER
# ─────────────────────────────────────────────────────
# Depends on: medschm.silver.protocol_criteria_flat
#             medschm.silver.protocol_rules
# Output:     medschm.gold.preauth_decisions
# ─────────────────────────────────────────────────────

# COMMAND ----------
# %pip install openai
# dbutils.library.restartPython()

# COMMAND ----------
# ── 1. IMPORTS & CONFIG ──────────────────────────────

import json, re
from datetime import datetime, date
from dataclasses import dataclass, field, asdict
from typing import Any

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField,
    StringType, ArrayType, MapType,
    BooleanType, TimestampType
)

from openai import AzureOpenAI

spark = SparkSession.builder.getOrCreate()

AZURE_OAI_ENDPOINT = dbutils.secrets.get("kv-medschm", "azure-oai-endpoint")
AZURE_OAI_KEY      = dbutils.secrets.get("kv-medschm", "azure-oai-key")
CHAT_DEPLOYMENT    = "gpt-4o"

client = AzureOpenAI(
    azure_endpoint=AZURE_OAI_ENDPOINT,
    api_key=AZURE_OAI_KEY,
    api_version="2024-02-01",
)

# ── Decision outcomes ─────────────────────────────────
APPROVE = "APPROVE"
REJECT  = "REJECT"
REFER   = "REFER_MEDICAL_ADVISOR"

# COMMAND ----------
# ── 2. DATA CLASSES ──────────────────────────────────

@dataclass
class CriterionResult:
    criterion_id:  str
    criteria_tier: str          # mandatory | conditional | rejection
    description:   str
    field:         str
    passed:        bool
    actual_value:  Any
    expected_value: Any
    failure_message: str | None = None

@dataclass
class DecisionResult:
    case_ref:        str
    protocol_id:     str
    protocol_name:   str
    member_id:       str
    outcome:         str        # APPROVE | REJECT | REFER_MEDICAL_ADVISOR
    confidence:      float      # 0.0 – 1.0
    rationale:       str
    mandatory_results:   list[CriterionResult] = field(default_factory=list)
    conditional_results: list[CriterionResult] = field(default_factory=list)
    rejection_results:   list[CriterionResult] = field(default_factory=list)
    triggered_rejections:   list[str] = field(default_factory=list)
    triggered_conditionals: list[str] = field(default_factory=list)
    failed_mandatory:       list[str] = field(default_factory=list)
    decided_at:      str = field(default_factory=lambda: datetime.utcnow().isoformat())

# COMMAND ----------
# ── 3. CRITERION EVALUATOR ───────────────────────────

def _parse_value(raw: str) -> Any:
    """Deserialise stored JSON value string back to Python type."""
    try:
        return json.loads(raw)
    except Exception:
        return raw


def evaluate_criterion(criterion: dict, case: dict) -> CriterionResult:
    """
    Evaluate one criterion row from protocol_criteria_flat against the case dict.

    Supported operators: eq, neq, gt, gte, lt, lte, in, not_in, present, regex
    Supported types:     boolean, numeric_threshold, enum_match,
                         document_present, date_window, regex_match
    """
    cid        = criterion["criterion_id"]
    tier       = criterion["criteria_tier"]
    desc       = criterion["description"]
    field_name = criterion["field"]
    operator   = criterion["operator"]
    expected   = _parse_value(criterion["value"])
    ctype      = criterion.get("criterion_type", "boolean")
    fail_msg   = criterion.get("failure_message")

    actual = case.get(field_name)

    # ── date_window: document must be within N days of today ──────────
    if ctype == "date_window":
        within_days = int(expected) if expected else 30
        doc_date = actual
        if doc_date is None:
            passed = False
        else:
            if isinstance(doc_date, str):
                doc_date = date.fromisoformat(doc_date)
            delta = (date.today() - doc_date).days
            passed = delta <= within_days
        return CriterionResult(cid, tier, desc, field_name, passed, str(actual), f"within {within_days} days", fail_msg)

    # ── document_present: truthy check ───────────────────────────────
    if ctype == "document_present" or operator == "present":
        passed = actual not in (None, "", [], False)
        return CriterionResult(cid, tier, desc, field_name, passed, actual, "present", fail_msg)

    # ── regex ─────────────────────────────────────────────────────────
    if operator == "regex" or ctype == "regex_match":
        passed = bool(re.match(str(expected), str(actual or "")))
        return CriterionResult(cid, tier, desc, field_name, passed, actual, expected, fail_msg)

    # ── equality / comparison ─────────────────────────────────────────
    try:
        if operator == "eq":
            passed = actual == expected
        elif operator == "neq":
            passed = actual != expected
        elif operator == "gt":
            passed = float(actual) > float(expected)
        elif operator == "gte":
            passed = float(actual) >= float(expected)
        elif operator == "lt":
            passed = float(actual) < float(expected)
        elif operator == "lte":
            passed = float(actual) <= float(expected)
        elif operator == "in":
            passed = actual in expected
        elif operator == "not_in":
            passed = actual not in expected
        else:
            passed = False
    except (TypeError, ValueError):
        passed = False

    return CriterionResult(cid, tier, desc, field_name, passed, actual, expected, fail_msg)

# COMMAND ----------
# ── 4. CORE DECISION FUNCTION ────────────────────────

def run_decision(case: dict) -> DecisionResult:
    """
    Evaluate a pre-auth case against its protocol and return a DecisionResult.

    case dict must include:
      protocol_id   - identifies which protocol to load
      case_ref      - unique reference for this request
      member_id     - member identifier
      + all clinical fields referenced in criteria
    """
    protocol_id = case["protocol_id"]
    case_ref    = case["case_ref"]
    member_id   = case["member_id"]

    # ── Load criteria from Delta ──────────────────────
    criteria_rows = (
        spark.table("medschm.silver.protocol_criteria_flat")
        .filter(F.col("protocol_id") == protocol_id)
        .collect()
    )

    if not criteria_rows:
        raise ValueError(f"No criteria found for protocol_id='{protocol_id}'")

    protocol_name = criteria_rows[0]["protocol_name"]

    mandatory_rows   = [r for r in criteria_rows if r["criteria_tier"] == "mandatory"]
    conditional_rows = [r for r in criteria_rows if r["criteria_tier"] == "conditional"]
    rejection_rows   = [r for r in criteria_rows if r["criteria_tier"] == "rejection"]

    # ── Evaluate rejection first (short-circuit) ──────
    rejection_results = [evaluate_criterion(dict(r), case) for r in rejection_rows]
    triggered_rejections = [r.criterion_id for r in rejection_results if r.passed]

    if triggered_rejections:
        fail_msgs = [
            r.failure_message or r.description
            for r in rejection_results if r.passed
        ]
        rationale = _build_rationale(
            client, case, protocol_id, REJECT,
            triggered_ids=triggered_rejections,
            messages=fail_msgs
        )
        return DecisionResult(
            case_ref=case_ref, protocol_id=protocol_id,
            protocol_name=protocol_name, member_id=member_id,
            outcome=REJECT, confidence=1.0,
            rationale=rationale,
            rejection_results=rejection_results,
            triggered_rejections=triggered_rejections,
        )

    # ── Evaluate mandatory ────────────────────────────
    mandatory_results = [evaluate_criterion(dict(r), case) for r in mandatory_rows]
    failed_mandatory  = [r.criterion_id for r in mandatory_results if not r.passed]

    if failed_mandatory:
        fail_msgs = [
            r.failure_message or r.description
            for r in mandatory_results if not r.passed
        ]
        rationale = _build_rationale(
            client, case, protocol_id, REJECT,
            triggered_ids=failed_mandatory,
            messages=fail_msgs
        )
        return DecisionResult(
            case_ref=case_ref, protocol_id=protocol_id,
            protocol_name=protocol_name, member_id=member_id,
            outcome=REJECT, confidence=0.95,
            rationale=rationale,
            mandatory_results=mandatory_results,
            failed_mandatory=failed_mandatory,
        )

    # ── Evaluate conditional ──────────────────────────
    conditional_results    = [evaluate_criterion(dict(r), case) for r in conditional_rows]
    triggered_conditionals = [r.criterion_id for r in conditional_results if r.passed]

    if triggered_conditionals:
        rationale = _build_rationale(
            client, case, protocol_id, REFER,
            triggered_ids=triggered_conditionals,
            messages=[r.description for r in conditional_results if r.passed]
        )
        return DecisionResult(
            case_ref=case_ref, protocol_id=protocol_id,
            protocol_name=protocol_name, member_id=member_id,
            outcome=REFER, confidence=0.90,
            rationale=rationale,
            mandatory_results=mandatory_results,
            conditional_results=conditional_results,
            triggered_conditionals=triggered_conditionals,
        )

    # ── All clear → Approve ───────────────────────────
    rationale = _build_rationale(
        client, case, protocol_id, APPROVE,
        triggered_ids=[], messages=[]
    )
    return DecisionResult(
        case_ref=case_ref, protocol_id=protocol_id,
        protocol_name=protocol_name, member_id=member_id,
        outcome=APPROVE, confidence=0.98,
        rationale=rationale,
        mandatory_results=mandatory_results,
        conditional_results=conditional_results,
    )

# COMMAND ----------
# ── 5. GPT-4o RATIONALE GENERATOR ───────────────────

RATIONALE_PROMPT = """
You are a South African managed care clinical reviewer writing a concise, professional
pre-authorisation decision rationale. Use plain clinical English. Maximum 4 sentences.

Protocol: {protocol_id}
Decision: {outcome}
Criteria involved:
{criteria_block}

Case summary:
{case_summary}

Write the rationale paragraph only. No headings, no bullet points.
"""

def _build_rationale(
    oai_client, case: dict, protocol_id: str,
    outcome: str, triggered_ids: list, messages: list
) -> str:
    criteria_block = "\n".join(
        f"- {cid}: {msg}" for cid, msg in zip(triggered_ids, messages)
    ) or "All criteria satisfied."

    case_summary = (
        f"Member {case.get('member_id')} | ICD-10: {case.get('icd10_code')} | "
        f"Discipline: {case.get('treating_practitioner_discipline')} | "
        f"Scheme: {case.get('scheme')}"
    )

    try:
        resp = oai_client.chat.completions.create(
            model=CHAT_DEPLOYMENT,
            temperature=0.2,
            max_tokens=220,
            messages=[{
                "role": "user",
                "content": RATIONALE_PROMPT.format(
                    protocol_id=protocol_id,
                    outcome=outcome,
                    criteria_block=criteria_block,
                    case_summary=case_summary,
                )
            }]
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"Automated rationale unavailable ({e}). Decision: {outcome}."

# COMMAND ----------
# ── 6. BATCH PROCESSING FUNCTION ─────────────────────

def process_batch(cases: list[dict]) -> list[DecisionResult]:
    """Run the decision engine over a list of case dicts."""
    results = []
    for case in cases:
        try:
            result = run_decision(case)
        except Exception as e:
            result = DecisionResult(
                case_ref=case.get("case_ref", "UNKNOWN"),
                protocol_id=case.get("protocol_id", "UNKNOWN"),
                protocol_name="UNKNOWN",
                member_id=case.get("member_id", "UNKNOWN"),
                outcome="ERROR",
                confidence=0.0,
                rationale=str(e),
            )
        results.append(result)
        print(f"  {result.case_ref} → {result.outcome} ({result.confidence:.0%})")
    return results

# COMMAND ----------
# ── 7. SAMPLE CASES ──────────────────────────────────

sample_cases = [
    {
        # ── Should APPROVE ────────────────────────────
        "case_ref":                         "PA-2025-001",
        "protocol_id":                      "DTP-113S",
        "member_id":                        "MBR-00312",
        "scheme":                           "MedSchm Classic",
        "icd10_code":                       "D59.1",
        "member_registered":                True,
        "active_membership":                True,
        "chronic_auth_confirmed":           True,
        "treating_practitioner_discipline": "physician",
        "care_items_in_basket":             True,
        "clinical_motivation_submitted":    False,
        "comorbid_autoimmune":              False,
    },
    {
        # ── Should REJECT — wrong ICD-10 ─────────────
        "case_ref":                         "PA-2025-002",
        "protocol_id":                      "DTP-113S",
        "member_id":                        "MBR-00489",
        "scheme":                           "MedSchm Saver",
        "icd10_code":                       "J45.0",            # Asthma — wrong protocol
        "member_registered":                True,
        "active_membership":                True,
        "chronic_auth_confirmed":           True,
        "treating_practitioner_discipline": "general_practitioner",
        "care_items_in_basket":             True,
        "clinical_motivation_submitted":    False,
        "comorbid_autoimmune":              False,
    },
    {
        # ── Should REFER — drug-induced ICD + comorbid autoimmune ──
        "case_ref":                         "PA-2025-003",
        "protocol_id":                      "DTP-113S",
        "member_id":                        "MBR-00201",
        "scheme":                           "MedSchm Classic",
        "icd10_code":                       "D59.0",            # Drug-induced → conditional
        "member_registered":                True,
        "active_membership":                True,
        "chronic_auth_confirmed":           True,
        "treating_practitioner_discipline": "haematologist",
        "care_items_in_basket":             True,
        "clinical_motivation_submitted":    False,
        "comorbid_autoimmune":              True,               # Triggers CC-003
    },
]

print("Running decision engine...\n")
results = process_batch(sample_cases)

# COMMAND ----------
# ── 8. BUILD OUTPUT DATAFRAME ────────────────────────

DECISION_SCHEMA = StructType([
    StructField("case_ref",              StringType()),
    StructField("protocol_id",           StringType()),
    StructField("protocol_name",         StringType()),
    StructField("member_id",             StringType()),
    StructField("outcome",               StringType()),
    StructField("confidence",            StringType()),
    StructField("rationale",             StringType()),
    StructField("triggered_rejections",  ArrayType(StringType())),
    StructField("triggered_conditionals",ArrayType(StringType())),
    StructField("failed_mandatory",      ArrayType(StringType())),
    StructField("criteria_trace",        StringType()),   # full JSON trace
    StructField("decided_at",            StringType()),
])

def result_to_row(r: DecisionResult) -> dict:
    # Serialise full criteria trace for audit
    trace = {
        "mandatory":   [asdict(c) for c in r.mandatory_results],
        "conditional": [asdict(c) for c in r.conditional_results],
        "rejection":   [asdict(c) for c in r.rejection_results],
    }
    return {
        "case_ref":               r.case_ref,
        "protocol_id":            r.protocol_id,
        "protocol_name":          r.protocol_name,
        "member_id":              r.member_id,
        "outcome":                r.outcome,
        "confidence":             str(round(r.confidence, 4)),
        "rationale":              r.rationale,
        "triggered_rejections":   r.triggered_rejections,
        "triggered_conditionals": r.triggered_conditionals,
        "failed_mandatory":       r.failed_mandatory,
        "criteria_trace":         json.dumps(trace),
        "decided_at":             r.decided_at,
    }

decisions_df = spark.createDataFrame(
    [result_to_row(r) for r in results],
    schema=DECISION_SCHEMA
)

display(decisions_df.select(
    "case_ref", "protocol_id", "member_id",
    "outcome", "confidence", "rationale"
))

# COMMAND ----------
# ── 9. WRITE TO GOLD TABLE ───────────────────────────

(
    decisions_df
    .write
    .format("delta")
    .mode("append")
    .option("mergeSchema", "true")
    .saveAsTable("medschm.gold.preauth_decisions")
)

print("✓ Written to medschm.gold.preauth_decisions")

# COMMAND ----------
# ── 10. DECISION SUMMARY VIEW ────────────────────────

spark.sql("""
    SELECT
        outcome,
        COUNT(*)                                        AS total,
        ROUND(AVG(CAST(confidence AS DOUBLE)) * 100, 1) AS avg_confidence_pct
    FROM medschm.gold.preauth_decisions
    GROUP BY outcome
    ORDER BY total DESC
""").show()

# COMMAND ----------
# ── 11. EXPOSE AS DATABRICKS MODEL SERVING FUNCTION ─
# Wrap run_decision for calling from Teams bot or API gateway.

def preauth_request(
    case_ref:    str,
    protocol_id: str,
    member_id:   str,
    case_fields: dict,
) -> dict:
    """
    Entry point for external callers (Teams bot, REST API).
    Merges identity fields with clinical case_fields and runs the engine.
    Returns a serialisable dict.
    """
    case = {"case_ref": case_ref, "protocol_id": protocol_id, "member_id": member_id}
    case.update(case_fields)
    result = run_decision(case)

    return {
        "case_ref":    result.case_ref,
        "outcome":     result.outcome,
        "confidence":  result.confidence,
        "rationale":   result.rationale,
        "flags": {
            "triggered_rejections":   result.triggered_rejections,
            "triggered_conditionals": result.triggered_conditionals,
            "failed_mandatory":       result.failed_mandatory,
        },
        "decided_at": result.decided_at,
    }
