from pathlib import Path
import os
from dataclasses import dataclass


@dataclass
class Settings:
    """Runtime settings loaded from environment variables."""

    raw_notes_dir: str = os.getenv("RAW_NOTES_DIR", "data/raw/notes")
    default_source_tag: str = os.getenv("DEFAULT_SOURCE_TAG", "second-brain")
    embed_model: str = os.getenv("EMBED_MODEL", "text-embedding-3-small")
    chroma_dir: str = os.getenv("CHROMA_DIR", "./data/processed/chroma")
    reset_index: str = os.getenv("RESET_INDEX", "0")
    top_k: str = os.getenv("TOP_K", "6")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    temperature: str = os.getenv("TEMPERATURE", "0")
    max_context_chars: str = os.getenv("MAX_CONTEXT_CHARS", "10000")
    max_retries: str = os.getenv("MAX_RETRIES", "2")
    recency_days: str = os.getenv("RECENCY_DAYS", "365")
    evidence_min_recent_chunks: str = os.getenv("EVIDENCE_MIN_RECENT_CHUNKS", "1")
    evidence_threshold: str = os.getenv("EVIDENCE_THRESHOLD", "0.65")
    use_llm_grader: str = os.getenv("USE_LLM_GRADER", "0")


settings = Settings()


@dataclass(frozen=True)
class Config:
    PROJECT_ROOT: Path
    DATA_RAW_NOTES_DIR: Path
    DATA_PROCESSED_DIR: Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
config = Config(
    PROJECT_ROOT=PROJECT_ROOT,
    DATA_RAW_NOTES_DIR=PROJECT_ROOT / "data" / "raw" / "notes",
    DATA_PROCESSED_DIR=PROJECT_ROOT / "data" / "processed",
)
