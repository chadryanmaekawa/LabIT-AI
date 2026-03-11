import json
import os
import re
from pathlib import Path
from typing import Dict, Any

from openai import OpenAI


def slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def strip_code_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```json\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def get_openai_client() -> OpenAI | None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    return OpenAI(api_key=api_key)


def extract_assay_schema_from_text(document_text: str) -> Dict[str, Any]:
    client = get_openai_client()
    if client is None:
        raise ValueError("OPENAI_API_KEY is not set.")

    prompt = f"""
You are a laboratory diagnostics informatics expert.

Extract a normalized assay library entry from the package insert text.

Return ONLY valid JSON with this structure:
{{
  "assay_id": "",
  "vendor": "",
  "assay_name": "",
  "methodology": "",
  "panel_size": 0,
  "specimen": {{
    "specimen_type": "",
    "container": "",
    "minimum_volume": "",
    "transport": "",
    "stability": ""
  }},
  "components": [
    {{
      "name": "",
      "interface_code": "",
      "loinc": "",
      "result_values": {{
        "positive": "Detected",
        "negative": "Not Detected",
        "invalid": "Invalid"
      }}
    }}
  ],
  "notes": "",
  "metadata": {{
    "clinical_category": ""
  }}
}}

Rules:
- Infer panel_size from number of components if not explicit
- Use blank strings when fields are unknown
- Include all discrete analytes / targets you can identify
- Use simple result values where not clearly specified
- Do not include markdown
- Do not include commentary outside JSON

Package insert text:
{document_text[:30000]}
""".strip()

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.1,
        messages=[
            {
                "role": "system", 
                "content": "You extract structured assay data from diagnostics documentation."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    raw = response.choices[0].message.content or ""
    raw = strip_code_fences(raw)
    parsed = json.loads(raw)

    if not parsed.get("assay_id"):
        assay_name = parsed.get("assay_name", "assay")
        parsed["assay_id"] = slugify(assay_name)

    if not parsed.get("panel_size"):
        parsed["panel_size"] = len(parsed.get("components", []))

    return parsed


def save_assay_schema(schema: Dict[str, Any], output_dir: str = "assay_library/schemas") -> str:
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    assay_id = schema.get("assay_id", "assay")
    filename = f"{slugify(assay_id)}.json"
    path = Path(output_dir) / filename

    with open(path, "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2)

    return str(path)
