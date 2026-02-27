"""
Application-wide constants and configuration.
"""

# Maps each provider to the recommended model based on document type.
MODEL_MAP: dict[str, dict[str, str]] = {
    "OpenAI": {"Digital": "gpt-4o-mini", "Scanned": "gpt-4o"},
    "Gemini": {"Digital": "gemini-1.5-flash", "Scanned": "gemini-1.5-pro"},
    "Anthropic": {"Digital": "claude-3-5-haiku-latest", "Scanned": "claude-3-5-sonnet-latest"},
}

DEFAULT_JSON_SCHEMA_PROMPT: str = """Extract the following fields from the document and return ONLY valid JSON:
{
  "customer_name": "",
  "contract_id": "",
  "state": "",
  "term": ""
}
Do NOT include any explanation – only the JSON object."""
