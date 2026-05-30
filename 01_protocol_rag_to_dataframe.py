# Databricks Notebook
# ClinicalPath · Protocol RAG Pipeline
# OCR Images → Structured DataFrame → Decision Engine
# ─────────────────────────────────────────────────────

# COMMAND ----------
# %pip install openai langchain langchain-openai langchain-community
# dbutils.library.restartPython()

# COMMAND ----------
# ── 1. IMPORTS & CONFIG ──────────────────────────────

import re, json
from datetime import datetime

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField,
    StringType, ArrayType, MapType,
    BooleanType, IntegerType
)

from openai import AzureOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain.vectorstores import DatabricksVectorSearch
from langchain.docstore.document import Document

spark = SparkSession.builder.getOrCreate()

# ── Azure OpenAI ─────────────────────────────────────
AZURE_OAI_ENDPOINT   = dbutils.secrets.get("kv-medschm", "azure-oai-endpoint")
AZURE_OAI_KEY        = dbutils.secrets.get("kv-medschm", "azure-oai-key")
AZURE_OAI_VERSION    = "2024-02-01"
CHAT_DEPLOYMENT      = "gpt-4o"
EMBED_DEPLOYMENT     = "text-embedding-3-large"

# ── Storage paths ─────────────────────────────────────
BRONZE_PATH  = "dbfs:/mnt/bronze/protocol/"
SILVER_PATH  = "dbfs:/mnt/silver/protocol/"
VS_ENDPOINT  = "ep-clinicalpath-dev"        # Databricks Vector Search endpoint
VS_INDEX     = "medschm.preauth.protocol_index"

client = AzureOpenAI(
    azure_endpoint=AZURE_OAI_ENDPOINT,
    api_key=AZURE_OAI_KEY,
    api_version=AZURE_OAI_VERSION,
)

# COMMAND ----------
# ── 2. READ OCR TEXT TABLE FROM BRONZE ───────────────
# Expected bronze table columns: dbfs_path (string), ocr_text (string)
# Populated upstream by Azure Document Intelligence / Vision OCR job

ocr_df = spark.read.table("medschm.bronze.protocol_ocr_raw")

# Preview
display(ocr_df.limit(5))

# COMMAND ----------
# ── 3. GROUP MULTI-PAGE IMAGES INTO ONE PROTOCOL DOC ─
# Images sharing the same date prefix belong to the same protocol document.

ocr_df = ocr_df.withColumn(
    "doc_key",
    F.regexp_extract("dbfs_path", r"(\d{8})_\d{6}", 1)   # e.g. 20250805
)

protocol_docs = (
    ocr_df
    .groupBy("doc_key")
    .agg(
        F.collect_list(F.struct("dbfs_path", "ocr_text")).alias("pages"),
        F.concat_ws("\n\n", F.collect_list("ocr_text")).alias("full_text")
    )
)

display(protocol_docs.limit(3))

# COMMAND ----------
# ── 4. EXTRACTION PROMPT ─────────────────────────────

SYSTEM_PROMPT = """
You are a clinical protocol parser for a South African managed care PA engine.
Extract ONLY the fields listed. Be precise — do not infer what is not stated.
Return valid JSON matching the schema. Use null for missing fields.
"""

EXTRACTION_PROMPT = """
Extract the following from the medical protocol text below.

Return this exact JSON structure:

{
  "protocol_id":          "<DTP code e.g. DTP-113S or UNKNOWN>",
  "protocol_name":        "<full condition name>",
  "condition_category":   "<one of: haematology|oncology|endocrinology|cardiology|pulmonology|nephrology|neurology|rheumatology|infectious_disease|general>",
  "dtp_code":             "<DTP code>",
  "therapeutic_summary":  "<2-3 sentence clinical summary from Section 1>",

  "registration_routes":  ["<telephonic|email|digital_platform|hcp_initiated|broker>"],

  "mandatory_criteria": [
    {
      "criterion_id":   "<MC-001>",
      "section_ref":    "<e.g. 1.1>",
      "description":    "<exact clinical requirement>",
      "criterion_type": "<boolean|numeric_threshold|enum_match|document_present|date_window>",
      "weight":         "<critical|high|medium|low>",
      "field":          "<case field name this maps to>",
      "operator":       "<eq|neq|gt|gte|lt|lte|in|not_in|present>",
      "value":          "<expected value or list>",
      "failure_message":"<rejection text if this criterion fails>"
    }
  ],

  "conditional_criteria": [
    {
      "criterion_id":  "<CC-001>",
      "section_ref":   "<e.g. 2.2a>",
      "description":   "<condition that triggers Medical Advisor referral>",
      "field":         "<case field>",
      "operator":      "<operator>",
      "value":         "<value>"
    }
  ],

  "rejection_criteria": [
    {
      "criterion_id":   "<RC-001>",
      "section_ref":    "<e.g. 2.3a>",
      "description":    "<condition that triggers immediate rejection>",
      "field":          "<case field>",
      "operator":       "<operator>",
      "value":          "<value>",
      "failure_message":"<rejection letter text>"
    }
  ],

  "icd10_codes": [
    { "code": "<D59.0>", "description": "<description>", "is_primary": true }
  ],

  "basket_of_care": [
    {
      "description":        "<service description>",
      "discipline":         "<gp|physician|pathology|radiology|specialist>",
      "billing_codes":      ["<0190>"],
      "quantity_per_year":  <int or null>
    }
  ],

  "care_triggers": [
    "<chronic_auth_success|claim_with_icd10|hospital_auth|clinical_motivation>"
  ],

  "doc_key": "<date string from filename>"
}

PROTOCOL TEXT:
\"\"\"
{ocr_text}
\"\"\"
"""

# COMMAND ----------
# ── 5. EXTRACTION FUNCTION ───────────────────────────

def extract_protocol(doc_key: str, full_text: str) -> dict:
    """Call Azure OpenAI to extract structured protocol fields."""
    try:
        response = client.chat.completions.create(
            model=CHAT_DEPLOYMENT,
            response_format={"type": "json_object"},
            temperature=0,
            messages=[
                {"role": "system",  "content": SYSTEM_PROMPT},
                {"role": "user",    "content": EXTRACTION_PROMPT.format(ocr_text=full_text[:12000])},
            ],
        )
        result = json.loads(response.choices[0].message.content)
        result["doc_key"]          = doc_key
        result["extraction_ts"]    = datetime.utcnow().isoformat()
        result["extraction_status"] = "success"
        return result

    except Exception as e:
        return {
            "doc_key":          doc_key,
            "extraction_status": "error",
            "error_message":    str(e),
            "extraction_ts":    datetime.utcnow().isoformat(),
        }


# COMMAND ----------
# ── 6. RUN EXTRACTION OVER ALL PROTOCOL DOCS ─────────

rows = protocol_docs.select("doc_key", "full_text").collect()

extracted = [extract_protocol(r["doc_key"], r["full_text"]) for r in rows]

# COMMAND ----------
# ── 7. DEFINE OUTPUT SCHEMA ──────────────────────────

CRITERION_SCHEMA = StructType([
    StructField("criterion_id",   StringType()),
    StructField("section_ref",    StringType()),
    StructField("description",    StringType()),
    StructField("criterion_type", StringType()),
    StructField("weight",         StringType()),
    StructField("field",          StringType()),
    StructField("operator",       StringType()),
    StructField("value",          StringType()),   # serialised as string for flexibility
    StructField("failure_message",StringType()),
])

CARE_ITEM_SCHEMA = StructType([
    StructField("description",       StringType()),
    StructField("discipline",        StringType()),
    StructField("billing_codes",     ArrayType(StringType())),
    StructField("quantity_per_year", IntegerType()),
])

ICD10_SCHEMA = StructType([
    StructField("code",        StringType()),
    StructField("description", StringType()),
    StructField("is_primary",  BooleanType()),
])

PROTOCOL_SCHEMA = StructType([
    StructField("doc_key",              StringType()),
    StructField("protocol_id",          StringType()),
    StructField("protocol_name",        StringType()),
    StructField("condition_category",   StringType()),
    StructField("dtp_code",             StringType()),
    StructField("therapeutic_summary",  StringType()),
    StructField("registration_routes",  ArrayType(StringType())),
    StructField("mandatory_criteria",   ArrayType(CRITERION_SCHEMA)),
    StructField("conditional_criteria", ArrayType(CRITERION_SCHEMA)),
    StructField("rejection_criteria",   ArrayType(CRITERION_SCHEMA)),
    StructField("icd10_codes",          ArrayType(ICD10_SCHEMA)),
    StructField("basket_of_care",       ArrayType(CARE_ITEM_SCHEMA)),
    StructField("care_triggers",        ArrayType(StringType())),
    StructField("extraction_status",    StringType()),
    StructField("extraction_ts",        StringType()),
])

# COMMAND ----------
# ── 8. NORMALISE & BUILD DATAFRAME ───────────────────

def normalise_row(r: dict) -> dict:
    """Coerce extracted dict to match the Spark schema."""

    def to_criteria(lst):
        out = []
        for c in (lst or []):
            out.append({
                "criterion_id":    c.get("criterion_id"),
                "section_ref":     c.get("section_ref"),
                "description":     c.get("description"),
                "criterion_type":  c.get("criterion_type"),
                "weight":          c.get("weight"),
                "field":           c.get("field"),
                "operator":        c.get("operator"),
                "value":           json.dumps(c.get("value")),   # stringify for schema
                "failure_message": c.get("failure_message"),
            })
        return out

    def to_icd10(lst):
        return [
            {"code": i.get("code"), "description": i.get("description"),
             "is_primary": bool(i.get("is_primary", False))}
            for i in (lst or [])
        ]

    def to_care(lst):
        out = []
        for item in (lst or []):
            qty = item.get("quantity_per_year")
            out.append({
                "description":       item.get("description"),
                "discipline":        item.get("discipline"),
                "billing_codes":     item.get("billing_codes") or [],
                "quantity_per_year": int(qty) if qty else None,
            })
        return out

    return {
        "doc_key":              r.get("doc_key"),
        "protocol_id":          r.get("protocol_id"),
        "protocol_name":        r.get("protocol_name"),
        "condition_category":   r.get("condition_category"),
        "dtp_code":             r.get("dtp_code"),
        "therapeutic_summary":  r.get("therapeutic_summary"),
        "registration_routes":  r.get("registration_routes") or [],
        "mandatory_criteria":   to_criteria(r.get("mandatory_criteria")),
        "conditional_criteria": to_criteria(r.get("conditional_criteria")),
        "rejection_criteria":   to_criteria(r.get("rejection_criteria")),
        "icd10_codes":          to_icd10(r.get("icd10_codes")),
        "basket_of_care":       to_care(r.get("basket_of_care")),
        "care_triggers":        r.get("care_triggers") or [],
        "extraction_status":    r.get("extraction_status", "unknown"),
        "extraction_ts":        r.get("extraction_ts"),
    }


normalised = [normalise_row(r) for r in extracted]

protocol_df = spark.createDataFrame(normalised, schema=PROTOCOL_SCHEMA)

display(protocol_df)

# COMMAND ----------
# ── 9. WRITE TO SILVER DELTA TABLE ───────────────────

(
    protocol_df
    .write
    .format("delta")
    .mode("merge")
    .option("mergeSchema", "true")
    .saveAsTable("medschm.silver.protocol_rules")
)

print("✓ Written to medschm.silver.protocol_rules")

# COMMAND ----------
# ── 10. FLATTEN CRITERIA FOR ENGINE LOOKUP ───────────
# The engine joins on (protocol_id, criterion_id) — explode nested arrays.

criteria_df = (
    protocol_df
    .select(
        "protocol_id",
        "protocol_name",
        "condition_category",
        F.explode(
            F.array(
                F.struct(F.lit("mandatory")   .alias("tier"), F.col("mandatory_criteria")  .alias("criteria")),
                F.struct(F.lit("conditional") .alias("tier"), F.col("conditional_criteria").alias("criteria")),
                F.struct(F.lit("rejection")   .alias("tier"), F.col("rejection_criteria")  .alias("criteria")),
            )
        ).alias("tier_block")
    )
    .select(
        "protocol_id",
        "protocol_name",
        "condition_category",
        F.col("tier_block.tier").alias("criteria_tier"),
        F.explode("tier_block.criteria").alias("criterion"),
    )
    .select(
        "protocol_id",
        "protocol_name",
        "condition_category",
        "criteria_tier",
        F.col("criterion.criterion_id")   .alias("criterion_id"),
        F.col("criterion.section_ref")    .alias("section_ref"),
        F.col("criterion.description")    .alias("description"),
        F.col("criterion.criterion_type") .alias("criterion_type"),
        F.col("criterion.weight")         .alias("weight"),
        F.col("criterion.field")          .alias("field"),
        F.col("criterion.operator")       .alias("operator"),
        F.col("criterion.value")          .alias("value"),
        F.col("criterion.failure_message").alias("failure_message"),
    )
)

display(criteria_df)

(
    criteria_df
    .write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("medschm.silver.protocol_criteria_flat")
)

print("✓ Written to medschm.silver.protocol_criteria_flat")

# COMMAND ----------
# ── 11. POPULATE VECTOR STORE FOR RAG ────────────────
# Each criterion + therapeutic summary becomes a searchable chunk.

embedder = AzureOpenAIEmbeddings(
    azure_deployment=EMBED_DEPLOYMENT,
    azure_endpoint=AZURE_OAI_ENDPOINT,
    api_key=AZURE_OAI_KEY,
    api_version=AZURE_OAI_VERSION,
)

splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=80)

docs_for_vs = []

for row in protocol_df.collect():
    # Therapeutic summary chunk
    for chunk in splitter.split_text(row["therapeutic_summary"] or ""):
        docs_for_vs.append(Document(
            page_content=chunk,
            metadata={
                "protocol_id":  row["protocol_id"],
                "chunk_type":   "therapeutic_summary",
                "protocol_name": row["protocol_name"],
            }
        ))

    # ICD-10 code chunks
    icd_block = " | ".join(
        f"{i['code']}: {i['description']}" for i in (row["icd10_codes"] or [])
    )
    if icd_block:
        docs_for_vs.append(Document(
            page_content=f"Applicable ICD-10 codes for {row['protocol_name']}: {icd_block}",
            metadata={"protocol_id": row["protocol_id"], "chunk_type": "icd10_codes"}
        ))

    # Criteria chunks (one per criterion for precision retrieval)
    for tier in ["mandatory_criteria", "conditional_criteria", "rejection_criteria"]:
        for c in (row[tier] or []):
            text = (
                f"Protocol {row['protocol_id']} | {tier} | {c['criterion_id']} "
                f"(Section {c['section_ref']}): {c['description']}"
            )
            docs_for_vs.append(Document(
                page_content=text,
                metadata={
                    "protocol_id":  row["protocol_id"],
                    "chunk_type":   tier,
                    "criterion_id": c["criterion_id"],
                }
            ))

    # Basket of care chunks
    for item in (row["basket_of_care"] or []):
        text = (
            f"Basket of care ({row['protocol_id']}): {item['description']} | "
            f"Discipline: {item['discipline']} | Codes: {', '.join(item['billing_codes'] or [])}"
        )
        docs_for_vs.append(Document(page_content=text,
            metadata={"protocol_id": row["protocol_id"], "chunk_type": "basket_of_care"}))

print(f"Total chunks prepared for vector store: {len(docs_for_vs)}")

vs = DatabricksVectorSearch.from_documents(
    documents=docs_for_vs,
    embedding=embedder,
    endpoint_name=VS_ENDPOINT,
    index_name=VS_INDEX,
)

print(f"✓ Vector store populated: {VS_INDEX}")

# COMMAND ----------
# ── 12. QUICK SANITY CHECK ───────────────────────────

print("\n── Protocol Rules Table ─────────────────────────")
spark.sql("SELECT protocol_id, protocol_name, extraction_status FROM medschm.silver.protocol_rules").show(truncate=False)

print("\n── Flat Criteria Table ──────────────────────────")
spark.sql("""
    SELECT criteria_tier, COUNT(*) as count
    FROM medschm.silver.protocol_criteria_flat
    GROUP BY criteria_tier
""").show()

print("\n── Sample Mandatory Criteria ────────────────────")
spark.sql("""
    SELECT protocol_id, criterion_id, section_ref, field, operator, value
    FROM medschm.silver.protocol_criteria_flat
    WHERE criteria_tier = 'mandatory'
    ORDER BY protocol_id, criterion_id
""").show(truncate=False)
