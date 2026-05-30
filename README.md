Here's the pipeline end-to-end in 12 clean steps:
What each step does:
Step  What happens

1 Config — Azure OAI keys from Key Vault, paths


2 Read the bronze OCR table (path + raw text per image)


3 Group multi-page images into one protocol document by date prefix


4–5 Extraction prompt + function — GPT-4o with json_object mode, zero temperature


6 Run extraction over every protocol doc


7–8 Define Spark schema, normalise the extracted dicts to match it

9 Write nested protocol DataFrame → medschm.silver.protocol_rules

10 Explode criteria arrays → flat medschm.silver.protocol_criteria_flat (what the engine joins on)

11 Chunk and embed every criterion + ICD-10 + basket item → Databricks Vector Search12Sanity check queries
