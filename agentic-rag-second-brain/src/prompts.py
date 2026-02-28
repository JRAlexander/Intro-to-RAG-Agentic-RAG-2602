from __future__ import annotations

BASELINE_SYSTEM_PROMPT = """You are a helpful assistant for a technical notes corpus.
Answer the user question using only the provided context.
If context is insufficient, say what is missing.
Do not invent citations.
"""

BASELINE_USER_PROMPT_TEMPLATE = """Question:
{question}

Context chunks:
{context}

Return JSON with keys:
- answer: string
- citations: array of objects with doc_title, doc_date, chunk_id, source_path
- notes: optional string describing conflicts/uncertainty in the context
"""

BASELINE_OUTPUT_JSON_SCHEMA = {
    "name": "baseline_rag_response",
    "schema": {
        "type": "object",
        "properties": {
            "answer": {"type": "string"},
            "citations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "doc_title": {"type": "string"},
                        "doc_date": {"type": "string"},
                        "chunk_id": {"type": "string"},
                        "source_path": {"type": "string"},
                    },
                    "required": ["doc_title", "doc_date", "chunk_id", "source_path"],
                    "additionalProperties": False,
                },
            },
            "notes": {"type": "string"},
        },
        "required": ["answer", "citations"],
        "additionalProperties": False,
    },
}
