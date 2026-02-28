from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from openai import OpenAI

from src.prompts import (
    AGENTIC_GENERATION_JSON_SCHEMA,
    AGENTIC_GENERATION_SYSTEM_PROMPT,
    AGENTIC_GENERATION_USER_PROMPT_TEMPLATE,
    EVIDENCE_GRADER_JSON_SCHEMA,
    EVIDENCE_GRADER_SYSTEM_PROMPT,
    EVIDENCE_GRADER_USER_PROMPT_TEMPLATE,
    RECENCY_REWRITE_SYSTEM_PROMPT,
    RECENCY_REWRITE_USER_PROMPT_TEMPLATE,
)
from src.rag_baseline import build_context
from src.retrieval import retrieve_chunks


class AgenticRagState(TypedDict):
    user_query: str
    rewritten_query: str
    retrieved_chunks: list[dict[str, Any]]
    evidence_ok: bool
    confidence: Literal["high", "medium", "low"]
    retry_count: int
    decision_trace: list[str]
    final_answer: dict[str, Any]


RECENCY_HINT_TOKENS = (
    "current",
    "latest",
    "most recent",
    "best",
    "should we use",
    "recommended",
    "recommendation",
)
STOPWORDS = {
    "what",
    "which",
    "when",
    "where",
    "how",
    "should",
    "would",
    "could",
    "use",
    "we",
    "the",
    "a",
    "an",
    "is",
    "are",
    "to",
    "for",
    "of",
    "on",
    "in",
    "and",
}


def _parse_date(value: str) -> datetime | None:
    try:
        return datetime.strptime(value, "%Y-%m-%d")
    except (TypeError, ValueError):
        return None


def _latest_doc_date_from_chunks(chunks: list[dict[str, Any]]) -> datetime | None:
    dates = [_parse_date(chunk.get("doc_date", "")) for chunk in chunks]
    valid = [date for date in dates if date is not None]
    return max(valid) if valid else None


def _latest_doc_date_from_corpus(raw_notes_dir: Path | str) -> datetime | None:
    notes_path = Path(raw_notes_dir)
    if not notes_path.exists():
        return None

    latest: datetime | None = None
    for note_file in notes_path.glob("*.md"):
        match = re.match(r"(\d{4}-\d{2}-\d{2})-", note_file.name)
        if not match:
            continue
        maybe_date = _parse_date(match.group(1))
        if maybe_date is None:
            continue
        if latest is None or maybe_date > latest:
            latest = maybe_date
    return latest


def _extract_topic_keywords(query: str) -> set[str]:
    tokens = re.findall(r"[a-zA-Z0-9_-]+", query.lower())
    return {token for token in tokens if len(token) > 2 and token not in STOPWORDS}


def _has_topic_match(query: str, chunks: list[dict[str, Any]]) -> bool:
    keywords = _extract_topic_keywords(query)
    if not keywords:
        return True

    joined = " ".join(chunk.get("text", "").lower() for chunk in chunks)
    return any(word in joined for word in keywords)


def _contains_conflict_signals(chunks: list[dict[str, Any]]) -> bool:
    model_mentions = set()
    for chunk in chunks:
        text = chunk.get("text", "").lower()
        for model_name in [
            "text-embedding-3-small",
            "text-embedding-3-large",
            "text-embedding-ada-002",
        ]:
            if model_name in text:
                model_mentions.add(model_name)
    return len(model_mentions) > 1


def build_agentic_rag_graph(
    index,
    *,
    openai_model: str,
    temperature: float,
    top_k: int,
    max_context_chars: int,
    max_retries: int,
    recency_days: int,
    evidence_min_recent_chunks: int,
    use_llm_grader: bool,
    raw_notes_dir: Path | str,
):
    client = OpenAI()
    latest_corpus_doc_date = _latest_doc_date_from_corpus(raw_notes_dir=raw_notes_dir)

    def rewrite_with_recency_intent(state: AgenticRagState) -> AgenticRagState:
        user_query = state["user_query"]
        should_force_recency = any(token in user_query.lower() for token in RECENCY_HINT_TOKENS)

        rewritten_query = user_query
        if should_force_recency:
            response = client.chat.completions.create(
                model=openai_model,
                temperature=0,
                messages=[
                    {"role": "system", "content": RECENCY_REWRITE_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": RECENCY_REWRITE_USER_PROMPT_TEMPLATE.format(query=user_query),
                    },
                ],
            )
            rewritten_query = (response.choices[0].message.content or "").strip() or user_query
            if "latest notes by date" not in rewritten_query.lower():
                rewritten_query = f"{rewritten_query}. Prefer latest notes by date."

        state["rewritten_query"] = rewritten_query
        state["decision_trace"].append(f"rewrite: {rewritten_query}")
        return state

    def retrieve(state: AgenticRagState) -> AgenticRagState:
        chunks = retrieve_chunks(index=index, query=state["rewritten_query"], top_k=top_k)
        state["retrieved_chunks"] = chunks
        chunk_summary = [
            f"{chunk.get('chunk_id', '?')}|{chunk.get('doc_date', '?')}|{chunk.get('doc_title', '')}"
            for chunk in chunks
        ]
        state["decision_trace"].append("retrieve: " + ", ".join(chunk_summary))
        return state

    def _heuristic_grade(state: AgenticRagState) -> tuple[bool, str, str]:
        chunks = state["retrieved_chunks"]
        effective_latest = latest_corpus_doc_date or _latest_doc_date_from_chunks(chunks)
        if effective_latest is None:
            return False, "low", "No parseable doc_date found in corpus or retrieved chunks."

        recent_chunks = 0
        for chunk in chunks:
            parsed = _parse_date(chunk.get("doc_date", ""))
            if parsed is None:
                continue
            age_days = (effective_latest - parsed).days
            if age_days <= recency_days:
                recent_chunks += 1

        topic_match = _has_topic_match(state["user_query"], chunks)
        enough_recent = recent_chunks >= evidence_min_recent_chunks
        evidence_ok = enough_recent and topic_match

        confidence = "low"
        if evidence_ok and not _contains_conflict_signals(chunks):
            confidence = "high" if recent_chunks >= max(evidence_min_recent_chunks, 2) else "medium"
        elif evidence_ok:
            confidence = "medium"

        rationale = (
            f"recent_chunks={recent_chunks}, min_required={evidence_min_recent_chunks}, "
            f"topic_match={topic_match}, conflict_signals={_contains_conflict_signals(chunks)}"
        )
        return evidence_ok, confidence, rationale

    def grade_evidence(state: AgenticRagState) -> AgenticRagState:
        if use_llm_grader:
            chunks_text = "\n\n".join(
                f"- score={chunk.get('score')} | doc_date={chunk.get('doc_date')} | "
                f"doc_title={chunk.get('doc_title')} | chunk_id={chunk.get('chunk_id')}\n"
                f"{chunk.get('text', '')[:350]}"
                for chunk in state["retrieved_chunks"]
            )
            response = client.chat.completions.create(
                model=openai_model,
                temperature=0,
                response_format={"type": "json_schema", "json_schema": EVIDENCE_GRADER_JSON_SCHEMA},
                messages=[
                    {"role": "system", "content": EVIDENCE_GRADER_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": EVIDENCE_GRADER_USER_PROMPT_TEMPLATE.format(
                            query=state["user_query"],
                            rewritten_query=state["rewritten_query"],
                            recency_days=recency_days,
                            min_recent_chunks=evidence_min_recent_chunks,
                            chunks=chunks_text,
                        ),
                    },
                ],
            )
            parsed = json.loads(response.choices[0].message.content or "{}")
            state["evidence_ok"] = bool(parsed.get("evidence_ok", False))
            state["confidence"] = parsed.get("confidence", "low")
            rationale = parsed.get("rationale", "")
        else:
            evidence_ok, confidence, rationale = _heuristic_grade(state)
            state["evidence_ok"] = evidence_ok
            state["confidence"] = confidence

        state["decision_trace"].append(
            f"grade: evidence_ok={state['evidence_ok']}, confidence={state['confidence']} ({rationale})"
        )
        return state

    def retry_or_continue(state: AgenticRagState) -> AgenticRagState:
        if not state["evidence_ok"] and state["retry_count"] < max_retries:
            state["retry_count"] += 1
            latest_year = str(latest_corpus_doc_date.year) if latest_corpus_doc_date else "latest"
            state["rewritten_query"] = (
                f"{state['rewritten_query']} As of the latest notes, prefer superseded decisions and focus on {latest_year} updates."
            )
            state["decision_trace"].append(
                f"retry: attempt={state['retry_count']} strengthened query={state['rewritten_query']}"
            )
        else:
            state["decision_trace"].append(
                f"continue: evidence_ok={state['evidence_ok']} retry_count={state['retry_count']}"
            )
        return state

    def route_after_retry(state: AgenticRagState) -> str:
        if not state["evidence_ok"] and state["retry_count"] <= max_retries and "retry:" in state["decision_trace"][-1]:
            return "retrieve"
        return "generate_with_citations"

    def generate_with_citations(state: AgenticRagState) -> AgenticRagState:
        context = build_context(chunks=state["retrieved_chunks"], max_context_chars=max_context_chars)
        response = client.chat.completions.create(
            model=openai_model,
            temperature=temperature,
            response_format={"type": "json_schema", "json_schema": AGENTIC_GENERATION_JSON_SCHEMA},
            messages=[
                {"role": "system", "content": AGENTIC_GENERATION_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": AGENTIC_GENERATION_USER_PROMPT_TEMPLATE.format(
                        query=state["user_query"],
                        rewritten_query=state["rewritten_query"],
                        context=context,
                        confidence=state["confidence"],
                    ),
                },
            ],
        )
        parsed = json.loads(response.choices[0].message.content or "{}")
        if state["confidence"] == "low" and not parsed.get("next_step"):
            parsed["next_step"] = (
                "Do you want the latest recommendation or the historical recommendation from earlier 2025 notes?"
            )

        parsed["confidence"] = state["confidence"]
        state["final_answer"] = parsed
        state["decision_trace"].append("generate: completed answer with citations")
        return state

    workflow = StateGraph(AgenticRagState)
    workflow.add_node("rewrite_with_recency_intent", rewrite_with_recency_intent)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_evidence", grade_evidence)
    workflow.add_node("retry_or_continue", retry_or_continue)
    workflow.add_node("generate_with_citations", generate_with_citations)

    workflow.add_edge(START, "rewrite_with_recency_intent")
    workflow.add_edge("rewrite_with_recency_intent", "retrieve")
    workflow.add_edge("retrieve", "grade_evidence")
    workflow.add_edge("grade_evidence", "retry_or_continue")
    workflow.add_conditional_edges(
        "retry_or_continue",
        route_after_retry,
        {
            "retrieve": "retrieve",
            "generate_with_citations": "generate_with_citations",
        },
    )
    workflow.add_edge("generate_with_citations", END)

    return workflow.compile()


def run_agentic_rag(graph, query: str) -> dict[str, Any]:
    initial_state: AgenticRagState = {
        "user_query": query,
        "rewritten_query": query,
        "retrieved_chunks": [],
        "evidence_ok": False,
        "confidence": "low",
        "retry_count": 0,
        "decision_trace": [],
        "final_answer": {},
    }
    return graph.invoke(initial_state)
