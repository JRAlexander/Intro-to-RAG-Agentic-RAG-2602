from __future__ import annotations

import json
from typing import Any

from openai import OpenAI

from src.prompts import (
    BASELINE_OUTPUT_JSON_SCHEMA,
    BASELINE_SYSTEM_PROMPT,
    BASELINE_USER_PROMPT_TEMPLATE,
)
from src.retrieval import retrieve_chunks


def build_context(chunks: list[dict[str, Any]], max_context_chars: int) -> str:
    parts: list[str] = []
    used_chars = 0
    for idx, chunk in enumerate(chunks, start=1):
        block = (
            f"[{idx}] doc_title={chunk['doc_title']} | doc_date={chunk['doc_date']} "
            f"| chunk_id={chunk['chunk_id']} | source_path={chunk['source_path']}\n"
            f"{chunk['text']}\n"
        )
        if used_chars + len(block) > max_context_chars:
            break
        parts.append(block)
        used_chars += len(block)
    return "\n".join(parts)


def baseline_rag_answer(
    index,
    query: str,
    *,
    top_k: int,
    model: str,
    temperature: float,
    max_context_chars: int,
) -> dict[str, Any]:
    chunks = retrieve_chunks(index=index, query=query, top_k=top_k)
    context = build_context(chunks=chunks, max_context_chars=max_context_chars)

    client = OpenAI()
    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        response_format={"type": "json_schema", "json_schema": BASELINE_OUTPUT_JSON_SCHEMA},
        messages=[
            {"role": "system", "content": BASELINE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": BASELINE_USER_PROMPT_TEMPLATE.format(question=query, context=context),
            },
        ],
    )

    content = response.choices[0].message.content or "{}"
    parsed = json.loads(content)

    return {
        "query": query,
        "answer": parsed.get("answer", ""),
        "citations": parsed.get("citations", []),
        "notes": parsed.get("notes", ""),
        "retrieved_chunks": chunks,
    }
