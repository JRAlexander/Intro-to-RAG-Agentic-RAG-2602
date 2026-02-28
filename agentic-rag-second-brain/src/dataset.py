from pathlib import Path

from llama_index.core import Document

from .config import settings


def load_note_documents(raw_notes_dir: str | None = None) -> list[Document]:
    """Load markdown/text notes into LlamaIndex Document objects."""

    notes_path = Path(raw_notes_dir or settings.raw_notes_dir)
    if not notes_path.exists():
        return []

    documents: list[Document] = []
    for file_path in sorted(notes_path.glob("**/*")):
        if not file_path.is_file() or file_path.suffix.lower() not in {".md", ".txt"}:
            continue
        text = file_path.read_text(encoding="utf-8")
        documents.append(
            Document(
                text=text,
                metadata={
                    "source": settings.default_source_tag,
                    "path": str(file_path),
                    "filename": file_path.name,
                },
            )
        )

    return documents
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from src.config import config


NOTE_SPECS: List[Dict[str, object]] = [
    {
        "filename": "2025-01-10-embedding-model-cost-first.md",
        "title": "Embedding Model Decision: Cost-First Default",
        "date": "2025-01-10",
        "tags": ["embeddings", "architecture", "cost"],
        "body": """We should standardize on EmbedLite-v1 for now because the projected monthly query volume is high and token costs dominate. Retrieval quality is acceptable in internal tests for broad topical queries.\n\nDecision: Use EmbedLite-v1 as default for all note ingestion pipelines until quality complaints increase.""",
    },
    {
        "filename": "2025-03-18-embedding-evaluation-q1.md",
        "title": "Q1 Embedding Evaluation Notes",
        "date": "2025-03-18",
        "tags": ["embeddings", "evaluation"],
        "body": """Compared EmbedLite-v1 and EmbedPro-v2 on historical question sets. Precision at top-5 improved with EmbedPro-v2 on nuanced technical prompts, but the cost increase is significant.\n\nNo immediate migration; keep tracking support tickets.""",
    },
    {
        "filename": "2025-07-05-embedding-model-quality-shift.md",
        "title": "Embedding Model Decision Update: Quality Priority",
        "date": "2025-07-05",
        "tags": ["embeddings", "architecture", "quality"],
        "body": """Query logs now show many semantically subtle questions where EmbedLite-v1 misses context. A pilot with EmbedPro-v2 reduced misses and improved user trust.\n\nDecision change: move default embedding model to EmbedPro-v2 for quality, with caching to offset cost.""",
    },
    {
        "filename": "2025-10-21-embedding-rollout-postmortem.md",
        "title": "Embedding Rollout Postmortem",
        "date": "2025-10-21",
        "tags": ["embeddings", "postmortem"],
        "body": """After switching to EmbedPro-v2, retrieval quality improved in support and product docs. Latency impact was small after batch scheduling updates.\n\nFollow-up: keep quality-first embedding model while monitoring budget alerts weekly.""",
    },
    {
        "filename": "2025-02-02-chunking-large-windows.md",
        "title": "Chunking Strategy v1: Large Windows",
        "date": "2025-02-02",
        "tags": ["chunking", "retrieval"],
        "body": """Initial ingestion uses large chunks (1200 chars, no overlap). The goal is fast indexing and fewer chunks per document.\n\nDecision: keep chunk size large while corpus is still small.""",
    },
    {
        "filename": "2025-04-14-chunking-feedback.md",
        "title": "Chunking Feedback from Pilot",
        "date": "2025-04-14",
        "tags": ["chunking", "evaluation"],
        "body": """Large chunk windows often blend unrelated sections. Retrieval returns context that is partly relevant but noisy.\n\nPotential adjustment: lower chunk size and add slight overlap to preserve sentence continuity.""",
    },
    {
        "filename": "2025-09-03-chunking-small-overlap.md",
        "title": "Chunking Strategy v2: Smaller Chunks + Overlap",
        "date": "2025-09-03",
        "tags": ["chunking", "retrieval", "quality"],
        "body": """New experiments show better grounding with smaller chunks (420 chars) and 60-char overlap. This keeps topical boundaries cleaner while retaining context bridges.\n\nDecision change: adopt smaller chunks with overlap as default ingestion policy.""",
    },
    {
        "filename": "2025-11-12-chunking-maintenance.md",
        "title": "Chunking Maintenance Checklist",
        "date": "2025-11-12",
        "tags": ["chunking", "operations"],
        "body": """Operationally, smaller chunks increased node count but improved answer citation precision. Re-index cadence remains weekly.\n\nAction: keep v2 chunk settings and document rationale for future contributors.""",
    },
    {
        "filename": "2025-03-07-meeting-search-quality.md",
        "title": "Weekly Meeting: Search Quality Review",
        "date": "2025-03-07",
        "tags": ["meeting", "search"],
        "body": """Team reviewed common user journeys and identified ambiguous query phrasing as a top issue. Decided to expand test prompts before next sprint.\n\nOwner assignments were recorded for evaluation and telemetry cleanup.""",
    },
    {
        "filename": "2025-05-22-research-hybrid-retrieval.md",
        "title": "Research Snippet: Hybrid Retrieval",
        "date": "2025-05-22",
        "tags": ["research", "retrieval"],
        "body": """A short literature scan suggests dense+sparse hybrid retrieval helps on acronym-heavy corpora. We are not implementing this yet, but we should store queries for later replay experiments.""",
    },
    {
        "filename": "2025-06-30-meeting-onboarding-notes.md",
        "title": "Onboarding Meeting Notes",
        "date": "2025-06-30",
        "tags": ["meeting", "onboarding"],
        "body": """New contributors requested clearer setup docs and deterministic sample data. We agreed notebooks should run without external API keys for early onboarding.""",
    },
    {
        "filename": "2025-08-15-research-metadata-schema.md",
        "title": "Metadata Schema Research",
        "date": "2025-08-15",
        "tags": ["research", "metadata"],
        "body": """Proposed adding normalized metadata fields (doc_date, doc_title, tags, source_path) to improve filtering and interpretability in demos. The schema should remain provider-agnostic.""",
    },
    {
        "filename": "2025-10-02-meeting-demo-retro.md",
        "title": "Demo Retro: Internal Stakeholder Session",
        "date": "2025-10-02",
        "tags": ["meeting", "retro"],
        "body": """Stakeholders responded positively to timeline-based comparisons that highlight changing recommendations. Requested clearer examples of drift in embedding and chunking decisions.""",
    },
    {
        "filename": "2025-12-01-roadmap-notes-q1.md",
        "title": "Roadmap Notes for Q1 Planning",
        "date": "2025-12-01",
        "tags": ["planning", "roadmap"],
        "body": """Priorities for next quarter include refining retrieval observability and adding baseline evaluation scripts. Maintain deterministic sample corpus as a teaching asset.""",
    },
]


def _render_note(spec: Dict[str, object]) -> str:
    tags_yaml = "\n".join([f"  - {tag}" for tag in spec["tags"]])
    return (
        "---\n"
        f"title: {spec['title']}\n"
        f"date: {spec['date']}\n"
        "tags:\n"
        f"{tags_yaml}\n"
        "---\n\n"
        f"{spec['body']}\n"
    )


def ensure_dataset_exists(force_rebuild: bool = False) -> Dict[str, object]:
    notes_dir: Path = config.DATA_RAW_NOTES_DIR
    notes_dir.mkdir(parents=True, exist_ok=True)

    existing_markdown = sorted(notes_dir.glob("*.md"))
    if force_rebuild:
        for existing in existing_markdown:
            existing.unlink()

    created_or_updated: List[str] = []
    for spec in NOTE_SPECS:
        path = notes_dir / str(spec["filename"])
        content = _render_note(spec)
        if force_rebuild or (not path.exists()) or (path.read_text(encoding="utf-8") != content):
            path.write_text(content, encoding="utf-8")
            created_or_updated.append(path.name)

    all_files = sorted(p.name for p in notes_dir.glob("*.md"))
    return {
        "notes_dir": str(notes_dir),
        "num_notes": len(all_files),
        "filenames": all_files,
        "created_or_updated": created_or_updated,
        "force_rebuild": force_rebuild,
    }
