from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import settings
from src.graph import build_agentic_rag_graph, run_agentic_rag
from src.ingestion import chunk_documents, load_markdown_documents
from src.rag_baseline import baseline_rag_answer
from src.retrieval import load_persisted_index


@dataclass(frozen=True)
class EvalQuestion:
    qid: str
    question: str
    category: str
    drift_topic: str | None = None
    should_clarify: bool = False


def load_golden_questions(path: str | Path) -> list[EvalQuestion]:
    rows: list[EvalQuestion] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            rows.append(
                EvalQuestion(
                    qid=item["id"],
                    question=item["question"],
                    category=item["category"],
                    drift_topic=item.get("drift_topic"),
                    should_clarify=bool(item.get("should_clarify", False)),
                )
            )
    return rows


def _parse_date(date_text: str) -> datetime | None:
    try:
        return datetime.strptime(date_text, "%Y-%m-%d")
    except Exception:
        return None


def build_chunk_catalog(raw_notes_dir: str | Path) -> tuple[dict[str, dict[str, Any]], dict[str, set[str]]]:
    docs = load_markdown_documents(raw_notes_dir)
    nodes = chunk_documents(docs)

    chunk_by_id: dict[str, dict[str, Any]] = {}
    topic_chunks: dict[str, set[str]] = {}

    for node in nodes:
        metadata = dict(getattr(node, "metadata", {}))
        chunk_id = metadata.get("chunk_id", "")
        if not chunk_id:
            continue
        chunk_by_id[chunk_id] = metadata

        tags = metadata.get("tags", []) or []
        if isinstance(tags, str):
            tags = [t.strip() for t in tags.split(",") if t.strip()]
        for tag in tags:
            topic_chunks.setdefault(tag, set()).add(chunk_id)

    return chunk_by_id, topic_chunks


def _extract_citation_chunk_ids(citations: list[dict[str, Any]]) -> set[str]:
    return {
        str(c.get("chunk_id", "")).strip()
        for c in (citations or [])
        if isinstance(c, dict) and str(c.get("chunk_id", "")).strip()
    }


def _newest_window_chunk_ids(
    *,
    drift_topic: str,
    chunk_by_id: dict[str, dict[str, Any]],
    topic_chunks: dict[str, set[str]],
    window_days: int,
) -> set[str]:
    candidate_ids = topic_chunks.get(drift_topic, set())
    dated: list[tuple[str, datetime]] = []

    for chunk_id in candidate_ids:
        dt = _parse_date(str(chunk_by_id.get(chunk_id, {}).get("doc_date", "")))
        if dt is not None:
            dated.append((chunk_id, dt))

    if not dated:
        return set()

    max_date = max(dt for _, dt in dated)
    cutoff = max_date - timedelta(days=window_days)
    return {chunk_id for chunk_id, dt in dated if dt >= cutoff}


def _score_run(
    *,
    question: EvalQuestion,
    answer: str,
    citations: list[dict[str, Any]],
    retrieved_chunks: list[dict[str, Any]],
    latency_s: float,
    retries: int,
    chunk_by_id: dict[str, dict[str, Any]],
    topic_chunks: dict[str, set[str]],
    newest_window_days: int,
) -> dict[str, Any]:
    cited_ids = _extract_citation_chunk_ids(citations)

    citation_present = len(cited_ids) > 0
    citation_valid = citation_present and all(cid in chunk_by_id for cid in cited_ids)

    recency_correct: bool | None = None
    if question.category == "drift" and question.drift_topic:
        newest_ids = _newest_window_chunk_ids(
            drift_topic=question.drift_topic,
            chunk_by_id=chunk_by_id,
            topic_chunks=topic_chunks,
            window_days=newest_window_days,
        )
        recency_correct = len(cited_ids.intersection(newest_ids)) > 0 if newest_ids else False

    checks_failed = []
    if not citation_present:
        checks_failed.append("citation_present")
    if not citation_valid:
        checks_failed.append("citation_valid")
    if recency_correct is False:
        checks_failed.append("recency_correct")

    return {
        "id": question.qid,
        "question": question.question,
        "category": question.category,
        "drift_topic": question.drift_topic,
        "should_clarify": question.should_clarify,
        "answer": answer,
        "citations": citations,
        "retrieved_chunks": retrieved_chunks,
        "latency_s": latency_s,
        "retries": retries,
        "citation_present": citation_present,
        "citation_valid": citation_valid,
        "recency_correct": recency_correct,
        "checks_failed": checks_failed,
    }


def run_eval(
    *,
    golden_path: str | Path,
    chroma_dir: str | Path | None = None,
    embed_model: str | None = None,
    top_k: int = 6,
    openai_model: str | None = None,
    temperature: float = 0,
    max_context_chars: int = 10_000,
    max_retries: int = 2,
    recency_days: int = 365,
    evidence_min_recent_chunks: int = 1,
    use_llm_grader: bool = False,
    newest_window_days: int = 60,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    questions = load_golden_questions(golden_path)

    index = load_persisted_index(
        chroma_dir=Path(chroma_dir or settings.chroma_dir),
        embed_model=embed_model or settings.embed_model,
    )

    chunk_by_id, topic_chunks = build_chunk_catalog(settings.raw_notes_dir)

    graph = build_agentic_rag_graph(
        index=index,
        top_k=top_k,
        openai_model=openai_model or settings.openai_model,
        temperature=temperature,
        max_context_chars=max_context_chars,
        max_retries=max_retries,
        recency_days=recency_days,
        evidence_min_recent_chunks=evidence_min_recent_chunks,
        use_llm_grader=use_llm_grader,
    )

    baseline_rows: list[dict[str, Any]] = []
    agentic_rows: list[dict[str, Any]] = []

    for q in questions:
        t0 = time.perf_counter()
        base = baseline_rag_answer(
            index=index,
            query=q.question,
            top_k=top_k,
            model=openai_model or settings.openai_model,
            temperature=temperature,
            max_context_chars=max_context_chars,
        )
        base_latency = time.perf_counter() - t0
        baseline_rows.append(
            _score_run(
                question=q,
                answer=base.get("answer", ""),
                citations=base.get("citations", []),
                retrieved_chunks=base.get("retrieved_chunks", []),
                latency_s=base_latency,
                retries=0,
                chunk_by_id=chunk_by_id,
                topic_chunks=topic_chunks,
                newest_window_days=newest_window_days,
            )
        )

        t1 = time.perf_counter()
        agentic_state = run_agentic_rag(graph, q.question)
        agentic_latency = time.perf_counter() - t1
        final_answer = agentic_state.get("final_answer", {})
        agentic_rows.append(
            _score_run(
                question=q,
                answer=final_answer.get("answer", ""),
                citations=final_answer.get("citations", []),
                retrieved_chunks=agentic_state.get("retrieved_chunks", []),
                latency_s=agentic_latency,
                retries=int(agentic_state.get("retry_count", 0) or 0),
                chunk_by_id=chunk_by_id,
                topic_chunks=topic_chunks,
                newest_window_days=newest_window_days,
            )
        )

    baseline_df = pd.DataFrame(baseline_rows)
    baseline_df["pipeline"] = "baseline"

    agentic_df = pd.DataFrame(agentic_rows)
    agentic_df["pipeline"] = "agentic"

    if os.getenv("USE_LLM_EVAL", "0") == "1":
        try:
            from src.eval_judge import llm_judge_dataframe

            baseline_df = llm_judge_dataframe(baseline_df)
            agentic_df = llm_judge_dataframe(agentic_df)
        except Exception as exc:
            print(f"Skipping LLM judge due to error: {exc}")

    return baseline_df, agentic_df


def build_comparison_report(baseline_df: pd.DataFrame, agentic_df: pd.DataFrame) -> pd.DataFrame:
    def summarize(df: pd.DataFrame, scope_name: str, mask: pd.Series | None = None) -> dict[str, Any]:
        scoped = df if mask is None else df[mask]
        return {
            "scope": scope_name,
            "citation_present_rate": float(scoped["citation_present"].mean()) if len(scoped) else 0.0,
            "citation_valid_rate": float(scoped["citation_valid"].mean()) if len(scoped) else 0.0,
            "recency_correct_rate": float(scoped["recency_correct"].dropna().mean())
            if scoped["recency_correct"].notna().any()
            else None,
            "avg_retries": float(scoped["retries"].mean()) if len(scoped) else 0.0,
            "avg_latency_s": float(scoped["latency_s"].mean()) if len(scoped) else 0.0,
            "num_questions": int(len(scoped)),
        }

    frames = []
    for df in [baseline_df, agentic_df]:
        pipe = str(df["pipeline"].iloc[0])
        all_row = summarize(df, "overall")
        all_row["pipeline"] = pipe
        drift_row = summarize(df, "drift", df["category"] == "drift")
        drift_row["pipeline"] = pipe
        frames.extend([all_row, drift_row])

    return pd.DataFrame(frames)[
        [
            "pipeline",
            "scope",
            "num_questions",
            "citation_present_rate",
            "citation_valid_rate",
            "recency_correct_rate",
            "avg_retries",
            "avg_latency_s",
        ]
    ]


def top_failures(df: pd.DataFrame, n: int = 3) -> pd.DataFrame:
    scored = df.copy()
    scored["num_failed_checks"] = scored["checks_failed"].map(len)
    ranked = scored.sort_values(["num_failed_checks", "latency_s"], ascending=[False, False])
    return ranked.head(n)
