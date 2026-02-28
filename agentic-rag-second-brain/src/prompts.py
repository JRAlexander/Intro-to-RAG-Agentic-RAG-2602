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

RECENCY_REWRITE_SYSTEM_PROMPT = """You improve a retrieval query for an internal notes corpus.
If the user asks for current/best/should we use recommendations, rewrite the query to emphasize the latest recommendation by date.
Keep the query concise and include wording equivalent to: prefer latest notes by date.
Return only the rewritten query text.
"""

RECENCY_REWRITE_USER_PROMPT_TEMPLATE = """Original user query:
{query}
"""

EVIDENCE_GRADER_SYSTEM_PROMPT = """You grade whether retrieved evidence is sufficient to answer a user query with recency awareness.
Return strict JSON with keys:
- evidence_ok: boolean
- confidence: one of high, medium, low
- rewrite_hint: string
- rationale: string
"""

EVIDENCE_GRADER_USER_PROMPT_TEMPLATE = """Query:
{query}

Rewritten query:
{rewritten_query}

Recency window days: {recency_days}
Required recent chunks: {min_recent_chunks}

Retrieved chunks:
{chunks}
"""

EVIDENCE_GRADER_JSON_SCHEMA = {
    "name": "evidence_grade",
    "schema": {
        "type": "object",
        "properties": {
            "evidence_ok": {"type": "boolean"},
            "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
            "rewrite_hint": {"type": "string"},
            "rationale": {"type": "string"},
        },
        "required": ["evidence_ok", "confidence", "rewrite_hint", "rationale"],
        "additionalProperties": False,
    },
}

AGENTIC_GENERATION_SYSTEM_PROMPT = """You are a grounded assistant for technical notes.
Use only the provided context. Do not invent facts or citations.
Prefer recommendations from the most recent notes when conflicts exist.
Return strict JSON with keys: answer, citations, confidence, next_step.
- citations must include doc_title, doc_date, chunk_id, source_path.
- confidence must be high|medium|low.
- next_step should be an empty string unless confidence is low.
"""

AGENTIC_GENERATION_USER_PROMPT_TEMPLATE = """User query:
{query}

Rewritten retrieval query:
{rewritten_query}

Retrieved context:
{context}

Computed confidence: {confidence}
"""

AGENTIC_GENERATION_JSON_SCHEMA = {
    "name": "agentic_rag_response",
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
            "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
            "next_step": {"type": "string"},
        },
        "required": ["answer", "citations", "confidence", "next_step"],
        "additionalProperties": False,
    },
}
